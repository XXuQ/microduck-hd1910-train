"""Rev-a MicroDuck: Feetech **HD-1910** servos (2S) and the **measured** masses (834 g).

Installed into microduck_rl as `src/mjlab_microduck/robot/hd1910_constants.py`.
Same joints, home pose, collisions and observation/action layout as the stock robot — only the
models' masses and the actuator differ, so a policy trained here drops into robotd unchanged.

* Models: `robot/microduck_hd1910/`, built by `training/hd1910/make_models.py` from
  microduck_rl's own `microduck/` XMLs (they have drifted from `hardware/original/mjcf`:
  STAND2 keyframe, backlash variants, collision edits), with

      15 x HD-1910  22.8 g          = 342.0 g
      battery       measured 100 g
      head          measured 255 g  (neck_pitch + yaw_roll_motion + jaw_soft)
      total         measured 834 g

  measured 2026-11-15 with a kitchen scale; the same numbers produce
  `robot/microduck_hd1910/` via `make_models.py` (run by `setup.sh`), which is what
  `training/hd1910/replay_onnx_bam.py` replays. The head is 255 g against the stock model's
  256.2 g (-0.6%), so the measured head barely moves anything; the battery is the real
  difference (+25 g, +33%).

* Actuator: the published **1910 M6** fit (`1910_m6.json`, LuwuDynamics/xgoduck_rl at
  `326d77a`, Apache-2.0, SHA-256 ef2d51ad...). Its `"actuator": "sts3215"` selects BAM's own
  Feetech controller -- the law it was fitted with -- so nothing extra needs registering.
  kt 0.624 N.m/A, R 4.91 ohm, per-radian duty 0.219 x P (vs the HL-2915 fit's 1.086 / 9.49 /
  1.06 at P=16: a different servo entirely). Upstream trained at `kp_fw = 5`; this robot runs
  register 50 = 16, so `HD1910_KP` is 16 -- the law is linear in P and the fit spanned P 4-64,
  but 16 has not been validated on a bench. (The other two public parameter sets -- Feetech's
  own gitee fork and what kissqy uses -- are documented in the git history.)

* **Measured on the robot (2026-11-15, `robot-collect-data.sh`)**: `[bus] position_kp = 16`,
  `position_kp_head = 16`, `position_kd = 8`, `current_limit_ma = 1300` at 0.65 mA/count,
  `[battery] cells = [2] / max_volts = 8.4`, pack reading 8.004 V at 87.8 %, control loop
  49.99 Hz, hottest servo 33 C, CPU 61.6 C. The walk policy is feed-forward
  (`obs[1,61] -> actions[1,14]`, verified with onnxruntime), so `model_api: 1`.
  `robot.state` while standing reports `safety.gain = 160` (200 x standing_gain_ratio 0.8), i.e.
  the servos hold at Kp 13 when standing and 16 when walking -- so `HD1910_KP = 16` is the
  walking figure BAM wants, not a standing one.

  **2026-11-18 更新**：`HD1910_KP` 改为 5，按 kissqy 的实机验证配置（见下方常量注释）。
  代价是**部署必须同步改**：`[bus] position_kp = 5`（robotd 侧的 `gain = 5`）。
  训练与部署的 kp 必须一致，否则等于把 5 训练的策略喂给 16 的舵机，transfer 会退化。

* Per-joint actuator spread: the stock BAM actuator scales gain and friction once per env;
  these scale them per joint, so a policy meets one hip weaker than the other.

* Head: the head assembly is ~30 % of the robot on this build, and the HAT, Orange Pi, fan,
  camera, ToF, microphones and speaker all sit at its front. `HEAD_COM_FORWARD_M` is kept at
  5 mm as the v2 training condition (training randomizes +/-10 mm around it anyway).
"""

import os
from pathlib import Path

from dataclasses import dataclass

import mujoco
import torch
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# `1910_m6.json` carries `"actuator": "sts3215"`, which selects BAM's own Feetech controller
# (the class LuwuDynamics fitted with). 该类在 mjlab 路径下缺 q_target_smooth 初始化，
# 本文件末尾的补丁负责修好它（详见下方 patched compute_control）。
# If you switch to Feetech's gitee params (`ftservo/ftservo-bam`, whose JSON says
# `"actuator": "hd1910"`), install `training/hd1910/hd1910.py` beside this file so that
# name is registered.
from mjlab_microduck.actuator import (
    BacklashEncoderBamActuator,
    BacklashEncoderBamActuatorCfg,
    FrictionDRBamActuator,
    FrictionDRBamActuatorCfg,
)
from mjlab_microduck.robot.microduck_constants import (
    BACKLASH_HOME_FRAME,
    FULL_COLLISION,
    HOME_FRAME,
    name_servo_collision_geoms,
)

# ---------------------------------------------------------------------------
# BAM `STS3215Actuator` 的 mjlab 兼容性补丁
#
# `1910_m6.json` 的 `"actuator": "sts3215"` 会选中 bam.feetech.actuator.STS3215Actuator
# （LuwuDynamics 拟合时用的那套律）。但该类的 `q_target_smooth` 只在台架用的
# `load_log()` 里创建，而 mjlab 训练路径从不调用 `load_log()` —— 于是第一次
# `compute_control()` 直接抛 AttributeError，训练和回放都起不来。
#
# 这里按它 `load_log()` 里那一行补一个惰性初始化，保持拟合律本身不变：
#   load_log():  self.q_target_smooth = np.zeros_like(self.kp)
# warp/torch 后端下 kp 在 GPU 上，所以用 q_target.clone() 保持同 device/dtype。
# ---------------------------------------------------------------------------
from bam.feetech.actuator import STS3215Actuator as _STS3215Actuator  # noqa: E402

_STS3215_ORIG_COMPUTE_CONTROL = _STS3215Actuator.compute_control


def _sts3215_compute_control_lazy_smooth(self, q_target, q, dq, dt):
    if not hasattr(self, "q_target_smooth"):
        if hasattr(q_target, "clone"):  # torch (warp/mjlab)
            self.q_target_smooth = q_target.clone()
        else:                            # numpy (mujoco.CPU rehearsal)
            import numpy as _np
            self.q_target_smooth = _np.array(q_target, copy=True)
    return _STS3215_ORIG_COMPUTE_CONTROL(self, q_target, q, dq, dt)


_STS3215Actuator.compute_control = _sts3215_compute_control_lazy_smooth

_DIR: Path = Path(os.path.dirname(__file__)) / "microduck_hd1910"
HD1910_PARAMS: Path = _DIR / "1910_m6.json"

for _p in (_DIR / "robot_walk.xml", _DIR / "robot_allcollisions.xml",
           _DIR / "robot_walk_backlash.xml", _DIR / "robot_allcollisions_backlash.xml",
           HD1910_PARAMS):
    assert _p.exists(), f"missing: {_p} (run training/hd1910/make_models.py)"

# Register-50 Kp the policies are trained at.
#
# 5 = 1910 台架/实机档。依据：kissqy 的 HD-1910 行走模型（实机走通）在其部署契约里写明
#   training_physics.actuators[0].kp_fw = 5.0，且用的是**同一份** hd1910_m6.json
#   （kt 到小数点后 16 位一致）。对应刚度约 0.65 N·m/rad，是 XL330 策略（0.55）的 1.2 倍。
#
# 为什么不是 16：16（≈2.07 N·m/rad，3.8× XL330）是从 HL-2915 那套配方借来的，
#   原注释写的是「keep it for HD-1910 until a bench fit says otherwise」——现在有了。
#   而且那个 0.4-1.1 的随机化是乘在 base 上的（16×0.4=6.4），永远覆盖不到 5。
#
# 可用环境变量覆盖，便于做 kp 对比实验：HD1910_KP=8 bash train_auto.sh ...
HD1910_KP = float(os.environ.get("HD1910_KP", "5"))

_KWARGS = dict(
    json_path=str(HD1910_PARAMS),
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=float(HD1910_KP),
    # 2S LiPo. Read off the robot on 2026-11-15: 8.004 V at 87.8 % on the robot's own display.
    # A full 2S is 2 x 4.20 = **8.4 V** off the charger -- that is the top of this range, and
    # also the servo's rating (LiHV's 8.7 V is not allowed). Empty is 2 x 3.30 = 6.6 V.
    # Careful with the 87.8 %: duck-control maps 3.30-4.10 V per cell (`CELL_FULL_V` is "full
    # under load", not the charge voltage), so the display reads ~10 points high -- by 4.2 V/cell
    # that pack is 78 %. It does not change this range, only how full it looks.
    # Deliberately unlike the HL-2915 file's 3S (10.0, 12.8): 3S would burn these servos.
    vin_range=(6.6, 8.4),
    vin_drop_gain_range=(0.0, 0.15),
    vin_min=6.6,
    # Bench: current rises 2-3 ms after the write; robotd adds a 16-device read (8.8 ms)
    # and inference before it. Steps of the physics timestep.
    delay_min_lag=2,
    delay_max_lag=6,
)


def _per_joint(actuator) -> None:
    """Widen the per-env (N, 1) gain and friction scales to per joint (N, J).

    Both only ever multiply (N, J) quantities -- `kp * kp_scale` into the firmware control
    law, `friction_scale` into the friction budget -- so a per-joint tensor broadcasts the
    same way a per-env one does. The kd scale stays per env.
    """
    n, j = actuator.kp_scale.shape[0], len(actuator.ctrl_ids)

    def ones():
        return torch.ones(n, j, dtype=actuator.kp_scale.dtype, device=actuator.kp_scale.device)

    actuator.kp_scale, actuator.default_kp_scale = ones(), ones()
    actuator.friction_scale, actuator.default_friction_scale = ones(), ones()


class PerJointBamActuator(FrictionDRBamActuator):
    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        _per_joint(self)


class PerJointBacklashBamActuator(BacklashEncoderBamActuator):
    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        _per_joint(self)


@dataclass(kw_only=True)
class PerJointBamActuatorCfg(FrictionDRBamActuatorCfg):
    def build(self, entity, target_ids, target_names) -> PerJointBamActuator:
        return PerJointBamActuator(self, entity, target_ids, target_names)


@dataclass(kw_only=True)
class PerJointBacklashBamActuatorCfg(BacklashEncoderBamActuatorCfg):
    def build(self, entity, target_ids, target_names) -> PerJointBacklashBamActuator:
        return PerJointBacklashBamActuator(self, entity, target_ids, target_names)


hd1910_actuators = PerJointBamActuatorCfg(**_KWARGS)
hd1910_backlash_actuators = PerJointBacklashBamActuatorCfg(**_KWARGS)

# Head CoM moved forward, metres. In the head body's frame the robot's forward axis is -z
# (at the home pose), so this comes off `jaw_soft`'s ipos z. Training randomizes the head
# CoM +-10 mm around it as well.
HEAD_COM_FORWARD_M = 0.005


def _spec(name):
    def load() -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(_DIR / name))
        # 舵机外壳碰撞几何的命名。
        #
        # 官方的 allcollisions 变体走 `get_allcollisions_spec()`，它在加载后调
        # `name_servo_collision_geoms()` 把每个 xl330 外壳网格命名成
        # `<body>_<k>_servo_collision`。VelStand 的保护性摔倒用一个 contact sensor
        # 盯这些几何体（`primary="^.*_servo_collision$"`），正则一条都匹配不上就直接
        # ValueError: Not all regular expressions are matched。
        #
        # 我们这里直接 `from_file` 绕过了那一步，于是所有走 HD1910_BACKLASH_ROBOT_CFG
        # 的任务（VelStand / SitStand / StandUp / LieDown）在建环境时就崩，只有用
        # robot_walk.xml 的 Velocity 系列不受影响。下面补上同样的命名，只对
        # allcollisions 变体做——和官方 get_allcollisions_spec 的边界一致。
        if "allcollisions" in name:
            spec = name_servo_collision_geoms(spec)
        if HEAD_COM_FORWARD_M:
            head = spec.body("jaw_soft")
            ipos = list(head.ipos)
            ipos[2] -= HEAD_COM_FORWARD_M
            head.ipos = ipos
        return spec

    return load


def _robot(xml, actuators, init_state=HOME_FRAME, collisions=(FULL_COLLISION,)):
    return EntityCfg(
        spec_fn=_spec(xml),
        init_state=init_state,
        collisions=collisions,
        articulation=EntityArticulationInfoCfg(
            actuators=(actuators,),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


HD1910_WALK_ROBOT_CFG = _robot("robot_walk.xml", hd1910_actuators)
HD1910_STANDUP_ROBOT_CFG = _robot("robot_allcollisions.xml", hd1910_actuators)
HD1910_WALK_BACKLASH_ROBOT_CFG = _robot("robot_walk_backlash.xml", hd1910_backlash_actuators,
                                        BACKLASH_HOME_FRAME)
HD1910_BACKLASH_ROBOT_CFG = _robot("robot_allcollisions_backlash.xml", hd1910_backlash_actuators,
                                   BACKLASH_HOME_FRAME)
