#!/usr/bin/env python3
"""摩擦对照探针：同一个 checkpoint，只改脚底摩擦，看走距怎么变。

为什么要单独做这个：
  我们默认训练的 Slippery 任务，脚底摩擦是从 (0.15, 1.3) 里均匀抽的；
  而唯一在实机上走通的公开配置（kissqy）用的是 (0.7, 1.3)，下限高 4.7 倍。
  如果同一个策略在 0.7 下能走、在 0.15 下只能踏步，那"摩擦"就是主因之一。

做法：固定指令（直走 vx），把 foot_friction 事件的范围钉成单个值 (μ, μ)，
其余随机化（kp、关节摩擦）全部关掉，保证只有摩擦一个变量在动。

用法（引擎 venv）：
    uv run python tools/probe_friction.py \
        --task Mjlab-Velocity-Flat-Backlash-Slippery-V6LP-HD1910-MicroDuck \
        --ckpt <run>/model_999.pt --vx 0.25 --seconds 8
"""

import argparse
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def resolve_engine(arg):
    """--engine 优先；否则 .engine_path（仓库或当前目录）；再否则找带 logs/rsl_rl 的目录。"""
    if arg:
        return Path(arg)
    for base in (REPO, Path.cwd()):
        f = Path(base) / ".engine_path"
        if f.is_file():
            return Path(f.read_text().strip())
    for c in (REPO.parent / "microduck_rl", Path.home() / "microduck_rl",
              Path.home() / "ducktest" / "microduck_rl", Path.cwd()):
        if (Path(c) / "logs" / "rsl_rl").is_dir():
            return Path(c)
    return Path.cwd()


def resolve_ckpt(arg, engine):
    p = Path(arg)
    if p.is_file():
        return p
    # 允许只给 run 片段，取里面最新的 checkpoint
    cands = sorted((engine).glob(f"logs/rsl_rl/*/*{arg}*/model_*.pt"),
                   key=lambda x: int("".join(c for c in x.stem if c.isdigit()) or 0))
    if cands:
        return cands[-1]
    found = sorted(str(x.parent.name) for x in (engine).glob("logs/rsl_rl/*/*/model_*.pt"))
    sys.exit(f"❌ 找不到 checkpoint：{arg}（引擎 {engine}）"
             + (f"\n   现有 run：{', '.join(sorted(set(found))[-6:])}" if found else ""))


def yaw_of(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def run_one(task, ckpt, mu, a):
    cfg = load_env_cfg(task, play=True)
    agent_cfg = load_rl_cfg(task)
    cfg.scene.num_envs = a.envs
    cfg.viewer.height = 64
    cfg.viewer.width = 64

    # 只留摩擦一个变量：钉死指令，关掉 kp / 关节摩擦随机化
    cmd = cfg.commands["twist"]
    cmd.ranges.lin_vel_x = (a.vx, a.vx)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.ranges.heading = None
    cmd.resampling_time_range = (1e6, 1e6)
    cmd.rel_standing_envs = 0.0
    cmd.heading_command = False

    ff = cfg.events.get("foot_friction")
    if ff is None:
        sys.exit("❌ 这个任务没有 foot_friction 事件，无法做摩擦对照")
    ff.params["ranges"] = (mu, mu)
    for name in ("randomize_joint_actuators", "randomize_joint_friction"):
        cfg.events.pop(name, None)

    env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=a.device)
    runner.load(str(ckpt), map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)
    robot = env.unwrapped.scene["robot"]

    step_dt = float(getattr(env.unwrapped, "step_dt", 0.0)) or (
        float(cfg.decimation) * float(getattr(getattr(cfg, "sim", None), "dt", 0.005)))
    steps = max(1, int(round(a.seconds / step_dt)))
    expected = a.vx * steps * step_dt

    def obs_of(x):
        return x[0] if isinstance(x, tuple) else x

    fwd, drift, lat, fall = [], [], [], []
    for _ in range(a.episodes):
        obs = obs_of(env.reset())
        p0 = robot.data.root_link_pos_w.clone()
        y0 = [yaw_of(robot.data.root_link_quat_w[i].cpu()) for i in range(a.envs)]
        fell = torch.zeros(a.envs, dtype=torch.bool, device=a.device)
        for _s in range(steps):
            with torch.no_grad():
                act = policy(obs)
            obs = obs_of(env.step(act)[0])
            fell |= (robot.data.root_link_pos_w[:, 2] < a.fallen_z)
        p1 = robot.data.root_link_pos_w
        for i in range(a.envs):
            fwd.append(float(p1[i, 0] - p0[i, 0]))
            lat.append(float(p1[i, 1] - p0[i, 1]))
            d = math.degrees(yaw_of(robot.data.root_link_quat_w[i].cpu()) - y0[i])
            drift.append(abs((d + 180) % 360 - 180))
            fall.append(1.0 if bool(fell[i]) else 0.0)

    n = len(fwd)
    out = (sum(fwd) / n, sum(drift) / n, sum(lat) / n, 100.0 * sum(fall) / n)
    del env
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out, expected


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint 路径，或 run 名片段（取最新）")
    ap.add_argument("--mus", default="0.15,0.4,0.7,1.0,1.3", help="要测的摩擦值，逗号分隔")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--vx", type=float, default=0.25)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--fallen-z", type=float, default=0.075)
    a = ap.parse_args()

    a.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    configure_torch_backends()

    engine = resolve_engine(a.engine)
    ckpt = resolve_ckpt(a.ckpt, engine)

    mus = [float(x) for x in a.mus.split(",") if x.strip()]
    print(f"checkpoint : {ckpt}")
    print(f"指令       : vx={a.vx} m/s 直走 {a.seconds}s · {a.envs} 环境 × {a.episodes} 回合")
    print("随机化     : 只改脚底摩擦，kp / 关节摩擦随机化已关闭")
    print()
    print(f"{'μ 摩擦':>8} {'走距 m':>8} {'走距比':>7} {'偏航 °':>8} {'侧漂 m':>8} {'摔倒 %':>7}")
    print("-" * 56)

    results = []
    for mu in mus:
        (fwd, drift, lat, fall), expected = run_one(a.task, ckpt, mu, a)
        ratio = fwd / expected if expected else 0.0
        results.append((mu, fwd, ratio, drift, lat, fall))
        print(f"{mu:8.2f} {fwd:8.3f} {ratio:7.2f} {drift:8.1f} {lat:8.3f} {fall:6.0f}%")

    print("-" * 56)
    best = max(results, key=lambda r: r[1])
    print(f"最好的 μ = {best[0]:.2f}（走距 {best[1]:.2f} m / 比 {best[2]:.2f}）")
    spread = max(r[1] for r in results) - min(r[1] for r in results)
    print(f"摩擦带来的走距跨度 = {spread:.2f} m"
          + ("  → 敏感，摩擦是主因之一" if spread > 0.5 else "  → 不敏感，摩擦不是主因"))


if __name__ == "__main__":
    main()
