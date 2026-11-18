#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 microduck_rl 训练环境里、用 **BAM（HD-1910 参数）** 执行器回放 ONNX 策略。

两种模式：
  * 默认（无头）：跑满 --duration 秒，打印行程/姿态指标后退出。
  * `--viewer`：打开 MuJoCo 窗口，**可用键盘实时控制**，关窗结束并打印指标。

为什么不用 DuckBuddy 的 `training/sim_walk.py`：它用**简单 position 执行器**
（直接写目标角），而策略是在 BAM 力矩/电流模型下训出来的 —— 同一个
`hd1910_walk_kp16_6500.onnx` 在 sim_walk 里几乎原地不动（5 s 前进 ~2 mm），
在 BAM 下能正常走。这个脚本复刻训练侧执行器，所以回放结果可信。

依赖：在 microduck_rl 仓库根目录、用它的 venv 跑（`uv run python <本脚本>`）。

用法（在 microduck_rl 根目录）：
    # 无头跑 8 秒出指标
    uv run python training/hd1910/replay_onnx_bam.py \
        --policy /path/hd1910_walk_kp16_6500.onnx --duration 8

    # 可视化 + 键盘控制（默认原地站立，按 W 开始走）
    uv run python training/hd1910/replay_onnx_bam.py \
        --policy /path/hd1910_walk_kp16_6500.onnx \
        --scene src/mjlab_microduck/robot/microduck_hd1910/scene_walk.xml --viewer

键盘（先点一下 3D 窗口让它获得焦点）：
    W / ↑      前进速度 +0.05 m/s        S / ↓   后退 / 减速
    A / ←      左移 +0.05 m/s            D / →   右移
    Q          左转 +0.2 rad/s           E       右转
    空格       立即停止（vx=vy=vyaw=0）
    R          重置到 STAND 姿态
    H          打印帮助

性能：`--render-hz`（默认 30）控制渲染频率，低于控制频率时跳帧同步。
WSLg 下把渲染降到 30 Hz 通常就能解决「卡」；再不行试 --render-hz 20。
"""
import argparse
import math
import os
import sys
import time

# WSL2 关键：默认 Mesa 会退到 llvmpipe（CPU 软渲染，640x480 一帧 320 ms）。
# 强制走 WSLg 的 d3d12 后端才是 GPU：实测 3.0 ms/帧，快 100 倍。
# （前置条件：用户要在 render/video 组里，否则 /dev/dri 打不开）
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "d3d12")
os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, "scripts")
import infer_policy as ip  # noqa: E402
import bam.model as bam_model_mod  # noqa: E402

# GLFW key codes
K_W, K_A, K_S, K_D, K_Q, K_E, K_R, K_H, K_SPACE = 87, 65, 83, 68, 81, 69, 82, 72, 32
K_UP, K_DOWN, K_LEFT, K_RIGHT = 265, 264, 263, 262
K_P = 80

HELP = """键盘（窗口内按键，或直接在终端按；两者都可用）：
  W / ↑   前进 +0.05 m/s        S / ↓   后退/减速
  A / ←   左移 +0.05 m/s        D / →   右移
  Q       左转 +0.2 rad/s       E       右转
  空格    立即停止              R       重置姿态
  P       随机推它一把（模拟外力扰动）
  H       打印本帮助

鼠标（MuJoCo 窗口内）：
  按住 Ctrl + 左键拖动 = 在鸭子身上施加力（点哪儿推哪儿）
  左键拖拽 = 转视角   右键拖拽/滚轮 = 平移/缩放
"""

# 终端字符 -> GLFW keycode（TerminalInput 返回 "w" / "up" 这类符号名）
_TERM_ALIAS = {
    "w": K_W, "up": K_UP, "s": K_S, "down": K_DOWN,
    "a": K_A, "left": K_LEFT, "d": K_D, "right": K_RIGHT,
    "q": K_Q, "e": K_E, " ": K_SPACE, "r": K_R, "h": K_H, "p": K_P,
}


def main():
    if not os.path.isdir("src/mjlab_microduck"):
        raise SystemExit(
            "请先 cd 到引擎根目录（microduck_rl/）再运行：\n"
            "  cd ~/microduck_rl\n"
            "  uv run python training/hd1910/replay_onnx_bam.py --policy <onnx> --viewer\n"
            f"（当前目录: {os.getcwd()}）")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", required=True, help="ONNX 策略路径")
    ap.add_argument("--scene", default="src/mjlab_microduck/robot/microduck_hd1910/scene_walk.xml")
    ap.add_argument("--params", default="src/mjlab_microduck/robot/microduck_hd1910/1910_m6.json")
    ap.add_argument("--vx", type=float, default=None,
                    help="初始前进速度；不给则 0（可视化默认原地站立）")
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--vyaw", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--action-scale", type=float, default=1.0)
    ap.add_argument("--kp-fw", type=float, default=16.0)
    ap.add_argument("--vin", type=float, default=7.4)
    ap.add_argument("--current-limit", type=float, default=1.3)
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument("--viewer", action="store_true", help="打开可视化窗口（关窗结束）")
    ap.add_argument("--render-hz", type=float, default=30.0, help="可视化渲染频率（默认 30）")
    ap.add_argument("--speed", type=float, default=1.0, help="仿真速度倍率（1.0 = 实时）")
    ap.add_argument("--exit-after", type=float, default=0.0,
                    help="可视模式下跑满 N 秒后自动退出（0 = 一直跑到关窗；用于性能自检）")
    ap.add_argument("--profile", action="store_true", help="统计各阶段耗时（物理/策略/渲染）")
    ap.add_argument("--auto-push", type=float, default=None, metavar="SEC",
                    help="在第 SEC 秒自动推一把（等同于按 P），用于无人值守的抗扰测试")
    ap.add_argument("--record", default=None, metavar="MP4",
                    help="把回放渲染成 mp4（不依赖 WSLg 传输，播放最顺）")
    ap.add_argument("--record-fps", type=float, default=50.0)
    ap.add_argument("--record-width", type=int, default=960)
    ap.add_argument("--record-height", type=int, default=720)
    ap.add_argument("--record-cam", default="follow",
                    choices=["follow", "side", "front", "top"],
                    help="录制机位：follow 追尾斜视（默认）/ side 侧视 / front 正视 / top 俯视")
    # 兼容旧命令：实时节流现在是内置行为，这个参数保留为空操作
    ap.add_argument("--realtime", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    # 可视化模式下默认站定，避免"一开窗鸭子就跑没影"
    vx = a.vx if a.vx is not None else 0.0
    vy, vyaw = a.vy, a.vyaw

    bam = bam_model_mod.load_model(a.params)
    bam.actuator.kp = a.kp_fw
    bam.actuator.vin = a.vin
    bam.actuator.max_current = a.current_limit
    bam.actuator.q_target_smooth = np.zeros_like(bam.actuator.kp)  # 见 README 11.1
    print(f"BAM: kt={bam.kt.value:.4f} R={bam.R.value:.4f} kp={bam.actuator.kp:.0f} "
          f"vin={bam.actuator.vin:.2f} max_current={bam.actuator.max_current}")

    model, data, bam_ctrl, act_names = ip.load_mujoco_with_bam(
        a.scene, bam, timestep=0.005, vin_drop_gain=0.075, vin_min=6.6)
    print(f"模型: nbody={model.nbody} nu={model.nu} 总质量={sum(model.body_mass)*1000:.1f} g")

    pol = ip.PolicyInference(model, data, walking_onnx_path=a.policy,
                             action_scale=a.action_scale, bam_ctrl=bam_ctrl,
                             use_projected_gravity=True, new_cmd_obs=True)

    def reset_pose():
        k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        if k >= 0:
            mujoco.mj_resetDataKeyframe(model, data, k)
        else:
            print("警告: 场景无 STAND keyframe，用默认 qpos")
        mujoco.mj_forward(model, data)

    reset_pose()

    # 外力扰动状态（P 键 / --auto-push 推一把）
    push_left = [0]          # 剩余控制步
    push_vec = np.zeros(6)

    def do_push():
        ang = np.random.uniform(0, 2 * np.pi)
        mag = np.random.uniform(0.8, 1.5)          # N
        push_vec[:] = [mag * np.cos(ang), mag * np.sin(ang), 0.0,
                       np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02), 0.0]
        push_left[0] = int(round(0.15 * a.control_hz))   # 持续 0.15 s
        print(f"[push] 施加外力 {mag:.2f} N，方向 {np.degrees(ang):.0f}°")

    def on_key(code):
        """按键入口：GUI 给 GLFW keycode(int)，终端给符号名(str)。"""
        nonlocal vx, vy, vyaw
        if isinstance(code, str):
            code = _TERM_ALIAS.get(code if len(code) > 1 else code.lower())
            if code is None:
                return
        step_v, step_yaw = 0.05, 0.2
        if code == K_P:
            do_push()
            return
        if code in (K_W, K_UP):
            vx = min(vx + step_v, 0.4)
        elif code in (K_S, K_DOWN):
            vx = max(vx - step_v, -0.3)
        elif code in (K_A, K_LEFT):
            vy = min(vy + step_v, 0.25)
        elif code in (K_D, K_RIGHT):
            vy = max(vy - step_v, -0.25)
        elif code == K_Q:
            vyaw = min(vyaw + step_yaw, 1.2)
        elif code == K_E:
            vyaw = max(vyaw - step_yaw, -1.2)
        elif code == K_SPACE:
            vx = vy = vyaw = 0.0
        elif code == K_R:
            reset_pose()
            vx = vy = vyaw = 0.0
        elif code == K_H:
            print(HELP)
            return
        else:
            return
        print(f"[cmd] vx={vx:+.2f} m/s  vy={vy:+.2f}  vyaw={vyaw:+.2f} rad/s")

    trunk_bid = model.site_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")]
    nsub = int(round((1.0 / a.control_hz) / model.opt.timestep))

    xs, ys, zs, rolls, pitches, vxs = [], [], [], [], [], []
    fell_at = None
    t_wall0 = time.time()
    t_pol = t_phys = t_sync = 0.0
    n_sync = 0

    viewer = None
    term = None
    if a.viewer:
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key,
                                              show_left_ui=False, show_right_ui=False)
        # 终端按键（不用点窗口也能控）；非 TTY 环境下会失败，降级为仅窗口按键
        try:
            term = ip.TerminalInput()
            term.__enter__()
        except Exception as e:  # noqa: BLE001
            term = None
            print(f"（终端按键不可用: {e} —— 请先点一下 3D 窗口再按键）")
        print("\n" + HELP)
        print(f"\n当前指令 vx={vx:+.2f}（默认站立；按 W 开始走）\n")

        _FS = mujoco.mjtFontScale.mjFONTSCALE_150.value
        _GP = mujoco.mjtGridPos.mjGRID_TOPLEFT.value

        def _hud():
            """窗口左上角显示当前指令与操作提示。"""
            try:
                viewer.set_texts([(
                    _FS, _GP,
                    "WASD/方向键 控制 · 空格 停 · P 推 · Ctrl+左键拖 = 点哪推哪",
                    f"cmd  vx {vx:+.2f} m/s   vy {vy:+.2f}   vyaw {vyaw:+.2f}",
                )])
            except Exception as e:  # noqa: BLE001
                if not getattr(_hud, "warned", False):
                    print(f"（HUD 不可用，已忽略: {e}）")
                    _hud.warned = True

        _hud()
        N = 10 ** 9
        sync_every = max(1, int(round(a.control_hz / max(a.render_hz, 1.0))))
        t_start = time.time()
    else:
        N = int(a.duration * a.control_hz)

    # 录制：离屏渲染，每 rec_every 个控制步抓一帧
    renderer = None
    cam = None
    writer = None
    n_frames = 0
    if a.record:
        # MuJoCo 默认离屏帧缓冲是 640x480，要更高分辨率必须先放大 offsize
        if model.vis.global_.offwidth < a.record_width:
            model.vis.global_.offwidth = a.record_width
        if model.vis.global_.offheight < a.record_height:
            model.vis.global_.offheight = a.record_height
        renderer = mujoco.Renderer(model, height=a.record_height, width=a.record_width)
        cam = mujoco.MjvCamera()
        _CAMS = {"follow": (0.85, -14.0, 135.0), "side": (0.85, -8.0, 90.0),
                 "front": (0.80, -10.0, 180.0), "top": (0.70, -70.0, 135.0)}
        cam.distance, cam.elevation, cam.azimuth = _CAMS[a.record_cam]
        rec_every = max(1, int(round(a.control_hz / max(a.record_fps, 1.0))))
        print(f"录制中：{a.record}（{a.record_width}x{a.record_height} @ {a.record_fps:g} fps，"
              f"机位 {a.record_cam}）")
        import imageio.v2 as iio
        # 边渲染边写盘：不要攒帧（960x720x3 ≈ 2MB/帧，几百帧就上 GB，WSL 会换页卡死）
        writer = iio.get_writer(a.record, fps=int(round(a.record_fps)), macro_block_size=1)

    try:
        for i in range(N):
            if viewer is not None and not viewer.is_running():
                break
            if term is not None:
                for _k in term.get_keys():
                    on_key(_k)
            if a.auto_push is not None and i == int(a.auto_push * a.control_hz):
                do_push()

            pol.set_vel_cmd(vx, vy, vyaw)
            _t0 = time.perf_counter() if a.profile else 0.0
            act = pol.infer()
            t_pol += time.perf_counter() - _t0 if a.profile else 0.0
            pol.apply_action(act)
            _t1 = time.perf_counter() if a.profile else 0.0
            # 外力扰动（P 键）：写到 trunk 的 xfrc_applied，作用 0.15 s
            if push_left[0] > 0:
                data.xfrc_applied[trunk_bid] = push_vec
                push_left[0] -= 1
            else:
                data.xfrc_applied[trunk_bid] = 0.0
            for _ in range(nsub):
                bam_ctrl.update()
                mujoco.mj_step(model, data)
            t_phys += time.perf_counter() - _t1 if a.profile else 0.0

            if viewer is not None:
                if i % sync_every == 0:
                    _t2 = time.perf_counter() if a.profile else 0.0
                    if i % (sync_every * 10) == 0:
                        _hud()
                    viewer.sync()
                    t_sync += time.perf_counter() - _t2 if a.profile else 0.0
                    n_sync += 1
                if a.exit_after > 0 and (i + 1) >= a.exit_after * a.control_hz:
                    break
                # 实时节流：按目标墙钟对齐，落后时不补（避免越追越卡）
                target = t_start + (i + 1) / (a.control_hz * a.speed)
                dt_sleep = target - time.time()
                if dt_sleep > 0:
                    time.sleep(dt_sleep)
                elif dt_sleep < -0.5:
                    t_start = time.time() - (i + 1) / (a.control_hz * a.speed)

            if renderer is not None and i % rec_every == 0:
                cam.lookat[:] = data.qpos[0:3]
                renderer.update_scene(data, camera=cam)
                writer.append_data(renderer.render())
                n_frames += 1

            quat = data.xquat[trunk_bid].copy()
            R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
            g = R.T @ np.array([0.0, 0.0, -1.0])
            vb = R.T @ data.qvel[0:3].copy()
            roll = math.atan2(-g[1], -g[2])
            pitch = math.atan2(g[0], math.hypot(g[1], g[2]))

            xs.append(float(data.qpos[0])); ys.append(float(data.qpos[1]))
            zs.append(float(data.qpos[2]))
            rolls.append(math.degrees(roll)); pitches.append(math.degrees(pitch))
            vxs.append(float(vb[0]))
            if fell_at is None and (data.qpos[2] < 0.05 or g[2] > -0.5):
                fell_at = i / a.control_hz
    finally:
        if term is not None:
            try:
                term.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        if viewer is not None:
            viewer.close()
        if renderer is not None:
            renderer.close()
        if writer is not None:
            writer.close()

    if a.record and n_frames:
        print(f"已保存视频: {a.record}（{n_frames} 帧，{n_frames / a.record_fps:.1f} s）")

    if not xs:
        print("没有跑到任何步（窗口立刻被关闭）")
        return

    n = len(xs); tail = slice(n // 3, n)
    wall = time.time() - t_wall0
    sim = n / a.control_hz
    print()
    print("=" * 78)
    print(f"末次指令 vx={vx:+.2f}, action_scale={a.action_scale}, 实跑 {sim:.1f}s, "
          f"BAM HD-1910 (kp{a.kp_fw:.0f}/{a.current_limit}A/{a.vin}V)")
    if viewer is not None:
        print(f"  性能: 墙钟 {wall:.1f}s / 仿真 {sim:.1f}s = 实时率 {sim / wall * 100:.0f}%"
              f"（<100% 说明渲染跟不上，降低 --render-hz）")
    if a.profile:
        print(f"  耗时分解（{n} 步）: 策略 {t_pol:.2f}s  物理 {t_phys:.2f}s  "
              f"渲染 {t_sync:.2f}s（{n_sync} 帧，均 {t_sync / max(n_sync,1)*1000:.1f} ms/帧）")
    print(f"  前进 Δx = {xs[-1]-xs[0]:+.3f} m  侧漂 Δy = {ys[-1]-ys[0]:+.3f} m")
    print(f"  vx 稳态均值 = {np.mean(vxs[tail]):+.3f} ± {np.std(vxs[tail]):.3f} m/s")
    print(f"  躯干高度 z = {np.mean(zs[tail]):.3f} m")
    print(f"  roll RMS = {np.sqrt(np.mean(np.square(rolls[tail]))):.1f}°   "
          f"pitch RMS = {np.sqrt(np.mean(np.square(pitches[tail]))):.1f}°")
    print(f"  摔倒 = {'否 ✓' if fell_at is None else f'是 @ {fell_at:.1f}s ❌'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
