#!/usr/bin/env python3
"""扫描一个 run 里的所有 checkpoint，用真实物理实测「走了多远 / 偏了多少」。

为什么要扫而不是取最后一个：rsl_rl 的曲线（error_vel_xy、reward）会骗人 ——
history 里出现过 error_vel_xy 卡在 0.42 但摔倒率极低、实际在原地踏步的 checkpoint。
硬指标只有两个：**固定时长走了多远、偏航多少度**。

脚本来自远端机器上 /root/sweep_checkpoints.py（作者前一轮实验用），这里改成：
  · argparse 代替位置参数，支持 --engine / --run 自动定位
  · 控制步长从 env cfg 里算（原来是写死 0.02s）
  · 加达标判定（走距 ≥ 指令期望的 70%、|偏航| ≤ 20°、摔倒 ≤ 10%）
  · 输出按走距排序，给推荐 checkpoint，可选 CSV

用法（必须用引擎的 venv）：

    cd <引擎目录>
    uv run python <仓库>/tools/sweep_checkpoints.py \
        --task Mjlab-Velocity-Flat-Backlash-Slippery-V6LP-HD1910-MicroDuck \
        --vx 0.25 --seconds 10

    # 指定某个 run / 只看最近 4 个 checkpoint / 少开点环境跑快点
    uv run python .../sweep_checkpoints.py --run 2026-11-18_09-46-00_kp5-1500 \
        --last 4 --envs 4 --episodes 1 --seconds 4

退出码：0 = 至少有一个 checkpoint 达标；1 = 全部未达标。
"""

import argparse
import csv
import glob
import math
import os
import sys
import time
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


# ----------------------------------------------------------------- 定位
def resolve_engine(arg):
    if arg:
        return Path(arg)
    f = REPO / ".engine_path"
    if f.is_file():
        return Path(f.read_text().strip())
    for c in (REPO.parent / "microduck_rl", Path.home() / "microduck_rl",
              Path.home() / "ducktest" / "microduck_rl"):
        if (c / ".git").is_dir():
            return c
    return None


def candidate_engines(primary):
    """resolved 引擎优先，其余常见位置作为兜底（这台机器上常有两三份引擎）。"""
    out = []
    for c in ([primary] if primary else []) + [
            REPO.parent / "microduck_rl", Path.home() / "microduck_rl",
            Path.home() / "ducktest" / "microduck_rl"]:
        if c and (c / ".git").is_dir() and c.resolve() not in [p.resolve() for p in out]:
            out.append(c)
    return out


def collect_runs(engine):
    runs = {}
    for p in engine.glob("logs/rsl_rl/*/*/model_*.pt"):
        runs.setdefault(p.parent.name, str(p.parent))
    return runs


def find_run(engine, name):
    """返回 (引擎, run 名, run 目录)。指定了 name 但当前引擎没有时，去其它引擎找。"""
    engines = candidate_engines(engine)
    if not engines:
        sys.exit("❌ 找不到任何引擎目录：先跑 setup.sh，或用 --engine 指定")

    if name:
        for eng in engines:
            runs = collect_runs(eng)
            for k, v in runs.items():
                if name in k:
                    if engine and eng.resolve() != Path(engine).resolve():
                        print(f"提示   : run 在 {eng} 里，已自动切换引擎")
                    return eng, k, Path(v)
        allruns = sorted({k for e in engines for k in collect_runs(e)})
        sys.exit(f"❌ 找不到 run: {name}\n   现有（{len(allruns)} 个，最近 8）："
                 + ", ".join(allruns[-8:]))

    # 没指定：取所有引擎里最新的那个 run
    best = None
    for eng in engines:
        runs = collect_runs(eng)
        if not runs:
            continue
        k, v = max(runs.items(), key=lambda kv: os.path.getmtime(kv[1]))
        if best is None or os.path.getmtime(v) > os.path.getmtime(best[2]):
            best = (eng, k, Path(v))
    if best is None:
        sys.exit("❌ 引擎里找不到任何 checkpoint（logs/rsl_rl/*/*/model_*.pt）")
    return best


def ckpt_index(p):
    digits = "".join(c for c in os.path.basename(p) if c.isdigit())
    return int(digits) if digits else 0


# ----------------------------------------------------------------- 几何
def yaw_of(q):
    """mujoco wxyz → yaw(rad)"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_deg(a):
    return (a + 180) % 360 - 180


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, help="任务名（uv run list-envs | grep HD1910）")
    ap.add_argument("--run", default=None, help="run 名（含片段即可），默认最新")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--vx", type=float, default=0.25, help="指令前进速度 m/s")
    ap.add_argument("--seconds", type=float, default=10.0, help="每回合时长（仿真秒）")
    ap.add_argument("--envs", type=int, default=8, help="并行环境数（越多越稳，越占显存）")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--last", type=int, default=99, help="只测最后 N 个 checkpoint")
    ap.add_argument("--every", type=int, default=1, help="每 N 个取一个（加速）")
    ap.add_argument("--fallen-z", type=float, default=0.075, help="躯干低于此高度算摔倒")
    ap.add_argument("--min-ratio", type=float, default=0.70, help="走距达标线（占指令期望）")
    ap.add_argument("--max-drift", type=float, default=20.0, help="偏航达标线（度）")
    ap.add_argument("--max-fall", type=float, default=10.0, help="摔倒率达标线（%）")
    ap.add_argument("--csv", default=None, help="把结果写成 CSV")
    a = ap.parse_args()

    engine = resolve_engine(a.engine)
    if not engine:
        sys.exit("❌ 找不到引擎目录：先跑 setup.sh，或用 --engine 指定")
    engine, run_name, run_dir = find_run(engine, a.run)
    os.chdir(engine)
    ckpts = sorted(glob.glob(str(run_dir / "model_*.pt")), key=ckpt_index)
    ckpts = ckpts[-a.last:][::max(1, a.every)]
    if not ckpts:
        sys.exit(f"❌ {run_dir} 里没有 model_*.pt")

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(a.task, play=True)
    agent_cfg = load_rl_cfg(a.task)
    env_cfg.scene.num_envs = a.envs
    env_cfg.viewer.height = 64
    env_cfg.viewer.width = 64

    # 钉死指令：只测「直走」，不要随机转向/侧移
    cmd = env_cfg.commands["twist"]
    cmd.ranges.lin_vel_x = (a.vx, a.vx)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.ranges.heading = None
    cmd.resampling_time_range = (1e6, 1e6)
    cmd.rel_standing_envs = 0.0
    cmd.heading_command = False

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    step_dt = float(getattr(env.unwrapped, "step_dt", 0.0)) or (
        float(env_cfg.decimation) * float(getattr(getattr(env_cfg, "sim", None), "dt", 0.005)))
    steps = max(1, int(round(a.seconds / step_dt)))
    expected = a.vx * steps * step_dt

    runner_cls = load_runner_cls(a.task) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    robot = env.unwrapped.scene["robot"]

    def _obs(x):
        return x[0] if isinstance(x, tuple) else x

    print(f"引擎   : {engine}")
    print(f"任务   : {a.task}")
    print(f"run    : {run_name}")
    print(f"控制   : {1/step_dt:.0f} Hz · 每回合 {steps} 步 = {steps*step_dt:.1f} s · 期望走 {expected:.2f} m")
    print(f"环境   : {a.envs} 并行 × {a.episodes} 回合 · checkpoint {len(ckpts)} 个")
    print(f"达标线 : 走距 ≥ {a.min_ratio*100:.0f}% 期望（{expected*a.min_ratio:.2f} m）"
          f" · |偏航| ≤ {a.max_drift:.0f}° · 摔倒 ≤ {a.max_fall:.0f}%")
    print()
    print(f"{'checkpoint':>16} {'走距 m':>8} {'走距比':>7} {'偏航 °':>8} {'侧漂 m':>8} {'摔倒 %':>7}  判定")
    print("-" * 78)

    rows = []
    t0 = time.time()
    for ck in ckpts:
        try:
            runner.load(ck, map_location=device)
        except Exception as e:
            print(f"{os.path.basename(ck):>16}  加载失败: {e}")
            continue
        policy = runner.get_inference_policy(device=device)

        fwd_all, drift_all, lat_all, fall_all = [], [], [], []
        for _ in range(a.episodes):
            obs = _obs(env.reset())
            p0 = robot.data.root_link_pos_w.clone()
            q0 = robot.data.root_link_quat_w.clone()
            y0 = [yaw_of(q0[i].cpu()) for i in range(a.envs)]
            fell = torch.zeros(a.envs, dtype=torch.bool, device=device)
            for _s in range(steps):
                with torch.no_grad():
                    act = policy(obs)
                out = env.step(act)
                obs = _obs(out[0])
                fell |= (robot.data.root_link_pos_w[:, 2] < a.fallen_z)
            p1 = robot.data.root_link_pos_w
            q1 = robot.data.root_link_quat_w
            for i in range(a.envs):
                fwd_all.append(float(p1[i, 0] - p0[i, 0]))
                lat_all.append(float(p1[i, 1] - p0[i, 1]))
                drift_all.append(abs(wrap_deg(math.degrees(yaw_of(q1[i].cpu()) - y0[i]))))
                fall_all.append(1.0 if bool(fell[i]) else 0.0)

        n = len(fwd_all)
        fwd = sum(fwd_all) / n
        drift = sum(drift_all) / n
        lat = sum(lat_all) / n
        fallp = 100.0 * sum(fall_all) / n
        ratio = fwd / expected if expected else 0.0
        ok = (ratio >= a.min_ratio) and (drift <= a.max_drift) and (fallp <= a.max_fall)
        rows.append({"ckpt": os.path.basename(ck), "fwd": fwd, "ratio": ratio, "drift": drift,
                     "lat": lat, "fall": fallp, "pass": ok})
        print(f"{os.path.basename(ck):>16} {fwd:8.3f} {ratio:7.2f} {drift:8.1f} {lat:8.3f} {fallp:6.0f}%  "
              f"{'✅ 达标' if ok else '❌'}")

    if not rows:
        sys.exit("❌ 一个 checkpoint 都没测成功")

    passed = [r for r in rows if r["pass"]]
    best = max(passed, key=lambda r: r["fwd"]) if passed else max(rows, key=lambda r: r["fwd"])
    print("-" * 78)
    print(f"推荐   : {best['ckpt']}（走距 {best['fwd']:.2f} m / 比 {best['ratio']:.2f}"
          f" / 偏航 {best['drift']:.1f}° / 摔倒 {best['fall']:.0f}%）")
    if passed:
        print(f"结论   : ✅ {len(passed)}/{len(rows)} 个达标")
    else:
        print("结论   : ❌ 全部未达标 —— 这批 checkpoint 走不起来，别取最后一个直接上机")
    print(f"耗时   : {time.time()-t0:.0f} s")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"CSV    : {a.csv}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
