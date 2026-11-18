#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HD-1910 建模查看器：逐个查看/调整 14 个舵机关节的角度。

用途（对应「3D 验证 / 看舵机旋转角度」）：
  * 交互摆位：选中某个关节，按 [-]/[+] 转它，窗口左上角实时显示角度（度/弧度）与限位；
    可以一眼看出每个舵机的旋转方向、零位、软限位是否和真机一致。
  * 静态查看：**关掉重力、不跑物理**，只做 mj_forward —— 纯运动学摆姿，
    不会因为松手就瘫倒，适合核对建模。
  * `--list`  ：打印 14 个关节的零位/限位/执行器编号（纯文本，秒查）。
  * `--sweep` ：自动把每个关节从下限扫到上限，统计过程中出现的接触数，
                用来发现「转到某个角度会自穿透/撞壳」的建模问题。

用法（在 microduck_rl 根目录）：
    uv run python training/hd1910/joint_viewer.py            # 交互查看（默认 834 g）
    uv run python training/hd1910/joint_viewer.py --list     # 只打印关节表
    uv run python training/hd1910/joint_viewer.py --sweep    # 扫限位查接触

交互按键（先点一下 3D 窗口让它获得焦点）：
    Tab / N      下一个关节            P        上一个关节
    [ / -        角度 −2°              ] / =    角度 +2°
    R            全部回零位（home）    0        全部归 0
    C            相机复位（对准整机）  H        打印帮助

鼠标：左键拖=转视角，右键拖=平移，**滚轮=缩放**（推太近会进到模型内部，按 C 复位）。
"""
import argparse
import math
import os
import sys

# WSL2：强制 GPU（否则 Mesa 退回 llvmpipe，慢 100 倍）
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "d3d12")
os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import mujoco.viewer

DEFAULT_SCENE = "src/mjlab_microduck/robot/microduck_hd1910/scene_walk.xml"

# 策略/真机使用的 14 个舵机关节（顺序与 obs/action 一致）
JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]

# 零位（STAND / HOME_FRAME），弧度
HOME = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0872665, "left_hip_pitch": -0.457924,
    "left_knee": -0.004940, "left_ankle": 0.452984,
    "neck_pitch": 0.349066, "head_pitch": 0.349066, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0872665, "right_hip_pitch": 0.457924,
    "right_knee": 0.004940, "right_ankle": -0.452984,
}

K_TAB, K_N, K_P, K_R, K_0, K_H, K_C = 258, 78, 80, 82, 48, 72, 67
K_LBRACKET, K_MINUS, K_RBRACKET, K_EQUAL = 91, 45, 93, 61


def joint_info(model):
    """返回 [(name, qpos_adr, lo, hi), ...]，只保留存在且有执行器的关节。"""
    out = []
    for name in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        lo, hi = model.jnt_range[jid]
        out.append((name, int(model.jnt_qposadr[jid]), float(lo), float(hi)))
    return out


def fmt_row(name, adr, lo, hi, cur=None):
    home = HOME.get(name, 0.0)
    s = (f"{name:<16} qpos[{adr:>2}]  零位 {math.degrees(home):+7.1f}°  "
         f"限位 [{math.degrees(lo):+6.1f}°, {math.degrees(hi):+6.1f}°]")
    if cur is not None:
        s += f"  当前 {math.degrees(cur):+7.1f}°"
    return s


def do_list(info):
    print(f"{'关节':<18}{'qpos':>6}{'零位':>9}{'下限':>9}{'上限':>9}{'行程':>9}")
    print("-" * 62)
    for name, adr, lo, hi in info:
        home = HOME.get(name, 0.0)
        print(f"{name:<18}{adr:>6}{math.degrees(home):>8.1f}°"
              f"{math.degrees(lo):>8.1f}°{math.degrees(hi):>8.1f}°"
              f"{math.degrees(hi - lo):>8.1f}°")
    print("-" * 62)
    print(f"共 {len(info)} 个关节")


def do_sweep(model, data, info, steps=12):
    """把每个关节从下限扫到上限，统计**机体之间**的接触（找自穿透/撞壳）。

    只看机体-机体接触：地板接触（body 0）不算，否则任何姿态都会被判成"有接触"。
    """
    model.opt.gravity[:] = 0.0

    def rest_pose():
        data.qpos[:] = 0.0
        data.qpos[2] = 0.12          # 抬到空中，避开地板
        data.qpos[3] = 1.0           # free joint 四元数 w=1（正立）
        for n2, a2, _, _ in info:
            data.qpos[a2] = HOME.get(n2, 0.0)
        mujoco.mj_forward(model, data)

    def self_contacts():
        n = 0
        for c in range(data.ncon):
            b1 = model.geom_bodyid[data.contact.geom1[c]]
            b2 = model.geom_bodyid[data.contact.geom2[c]]
            if b1 != 0 and b2 != 0:   # 0 = world/地板，排除
                n += 1
        return n

    print(f"{'关节':<18}{'扫描范围':>18}{'机体自接触':>12}{'首次接触角':>12}  备注")
    print("-" * 76)
    bad = 0
    for name, adr, lo, hi in info:
        rest_pose()
        worst, worst_deg = 0, None
        for v in np.linspace(lo, hi, steps):
            data.qpos[adr] = v
            mujoco.mj_forward(model, data)
            c = self_contacts()
            if c > worst:
                worst, worst_deg = c, math.degrees(v)
        note = "ok" if worst == 0 else "⚠ 该角度下机体互相接触，检查建模"
        bad += 1 if worst else 0
        rng = f"[{math.degrees(lo):+.0f}°, {math.degrees(hi):+.0f}°]"
        cd = "—" if worst_deg is None else f"{worst_deg:+.0f}°"
        print(f"{name:<18}{rng:>18}{worst:>12}{cd:>12}  {note}")
    print("-" * 76)
    print(f"{len(info)} 个关节中 {bad} 个在行程内出现机体自接触（0 最好）")
    return bad


def interactive(scene, info, step_deg):
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    info = joint_info(model)
    if not info:
        raise SystemExit("场景里找不到 HD-1910 的 14 个关节")

    model.opt.gravity[:] = 0.0        # 纯运动学查看，不塌

    def reset_home():
        data.qpos[:] = 0.0
        data.qpos[2] = 0.12
        data.qpos[3] = 1.0            # 机体朝上（free joint 的四元数 w）
        for n, adr, _, _ in info:
            data.qpos[adr] = HOME.get(n, 0.0)
        mujoco.mj_forward(model, data)

    reset_home()
    sel = [0]
    step = [math.radians(step_deg)]
    viewer_ref = [None]      # on_key 里复位相机要用

    def hud(viewer):
        # 注意：MuJoCo 自带的位图字体只有 ASCII，写中文会变成方块。
        # 中文说明一律走终端 print，HUD 只用英文。
        name, adr, lo, hi = info[sel[0]]
        cur = data.qpos[adr]
        left = "Tab/P:joint  [-]/[=]:-2deg/+2deg  R:home  0:zero  C:reset-cam"
        right = (f"[{sel[0] + 1}/{len(info)}] {name}  {math.degrees(cur):+.1f}deg "
                 f"({cur:+.3f}rad)  lim[{math.degrees(lo):+.0f},{math.degrees(hi):+.0f}]  "
                 f"con {data.ncon}")
        try:
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150.value,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT.value, left, right)])
        except Exception as e:  # noqa: BLE001
            if not getattr(hud, "warned", False):
                print(f"（HUD 不可用，已忽略: {e}）")
                hud.warned = True

    def frame_camera(viewer):
        """把相机对准整机（滚轮 zoom 进模型内部后，按 C 就能出来）。"""
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.16]
        viewer.cam.distance = 0.55
        viewer.cam.elevation = -18.0
        viewer.cam.azimuth = 140.0

    def on_key(code):
        if code in (K_TAB, K_N):
            sel[0] = (sel[0] + 1) % len(info)
        elif code == K_P:
            sel[0] = (sel[0] - 1) % len(info)
        elif code in (K_LBRACKET, K_MINUS):
            name, adr, lo, hi = info[sel[0]]
            data.qpos[adr] = float(np.clip(data.qpos[adr] - step[0], lo, hi))
            mujoco.mj_forward(model, data)
            print(f"  {name}: {math.degrees(data.qpos[adr]):+.1f}°  接触 {data.ncon}")
        elif code in (K_RBRACKET, K_EQUAL):
            name, adr, lo, hi = info[sel[0]]
            data.qpos[adr] = float(np.clip(data.qpos[adr] + step[0], lo, hi))
            mujoco.mj_forward(model, data)
            print(f"  {name}: {math.degrees(data.qpos[adr]):+.1f}°  接触 {data.ncon}")
        elif code == K_R:
            reset_home()
            print("  已回到零位")
        elif code == K_C:
            frame_camera(viewer_ref[0])
            print("  相机已复位（对准整机）")
        elif code == K_0:
            for _, adr, _, _ in info:
                data.qpos[adr] = 0.0
            mujoco.mj_forward(model, data)
            print("  所有关节归 0")
        elif code == K_H:
            print(__doc__)

    print("=" * 74)
    print("HD-1910 建模查看器（静态运动学，不塌）")
    print("  Tab/N 下一个关节 · P 上一个 · [-] 减小 · [=] 增大 · R 回零位 · 0 全零 · C 相机复位")
    print("  鼠标滚轮是缩放，推太近会钻进模型里（看到一片灰就是这原因），按 C 立刻回到正常取景")
    print("=" * 74)
    for i, (n, adr, lo, hi) in enumerate(info):
        print(f"  [{i + 1:>2}] {fmt_row(n, adr, lo, hi)}")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key,
                                      show_left_ui=False, show_right_ui=False) as v:
        viewer_ref[0] = v
        frame_camera(v)
        hud(v)
        v.sync()
        n = 0
        while v.is_running():
            v.sync()
            n += 1
            if n % 30 == 0:
                hud(v)
    print("已关闭查看器")


def require_engine_root():
    """这些脚本必须从引擎根目录（microduck_rl/）运行，否则相对路径全失效。"""
    if not os.path.isdir("src/mjlab_microduck"):
        raise SystemExit(
            "请先 cd 到引擎根目录（microduck_rl/）再运行：\n"
            "  cd ~/microduck_rl\n"
            "  uv run python training/hd1910/joint_viewer.py --list\n"
            f"（当前目录: {os.getcwd()}）")


def main():
    require_engine_root()
    ap = argparse.ArgumentParser(description="HD-1910 建模/舵机角度查看器")
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--base", action="store_true",
                    help="用官方 806g 基准模型查看（对比用）")
    ap.add_argument("--list", action="store_true", help="只打印关节/限位表")
    ap.add_argument("--sweep", action="store_true", help="扫每个关节限位并统计接触")
    ap.add_argument("--step-deg", type=float, default=2.0, help="每次按键转多少度（默认 2°）")
    a = ap.parse_args()

    scene = a.scene
    if a.base:
        scene = scene.replace("microduck_hd1910/scene_walk.xml",
                              "microduck_hd1910_base/scene_walk.xml")

    model = mujoco.MjModel.from_xml_path(scene)
    info = joint_info(model)
    if not info:
        raise SystemExit(f"{scene} 里找不到 HD-1910 关节")

    if a.list:
        do_list(info)
    elif a.sweep:
        do_sweep(model, mujoco.MjData(model), info)
    else:
        interactive(scene, info, a.step_deg)


if __name__ == "__main__":
    main()
