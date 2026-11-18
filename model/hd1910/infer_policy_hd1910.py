#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用**官方教程同一个入口**（scripts/infer_policy.py）跑 HD-1910 策略，
但把执行器参数换成 HD-1910 的 BAM 拟合（1910_m6.json）。

为什么要这个包装：
    `scripts/infer_policy.py` 里的 `load_bam_model()` 写死了内置的 **xl330** M6 模型
    （kt≈0.366、R≈2.81）。用它跑 HD-1910 策略时，BAM 算出的力矩偏小约 1.7 倍，
    实测表现是「起步一秒就趴下」（trunk_z 掉到 -110 mm）。HD-1910 的正确参数是
    kt≈0.624、R≈4.91（LuwuDynamics 拟合，见 1910_m6.json）。

用法（在 microduck_rl 根目录）：
    uv run python training/hd1910/infer_policy_hd1910.py \
        --walking <hd1910_walk_kp16_6500.onnx> \
        --scene src/mjlab_microduck/robot/microduck_hd1910/scene_walk.xml \
        --new-cmd-obs --kp-fw 16 --vin 7.4 --current-limit 1.3

参数与官方脚本完全一致（本脚本只替换 BAM 模型来源，其余全部转发）。
注意 `--new-cmd-obs` 必须加：不加会用旧版 58 维观测，和 61 维策略对不上。
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, "scripts")
import infer_policy as ip  # noqa: E402
import bam.model as bam_model  # noqa: E402

DEFAULT_PARAMS = "src/mjlab_microduck/robot/microduck_hd1910/1910_m6.json"


def _load_hd1910_bam(kp_fw, vin, max_current, params_path=DEFAULT_PARAMS):
    """替代 infer_policy.load_bam_model：换成 HD-1910 的拟合参数。"""
    m = bam_model.load_model(params_path)
    m.actuator.kp = kp_fw
    m.actuator.vin = vin
    m.actuator.max_current = max_current if (max_current and max_current > 0) else None
    # STS3215Actuator 的 q_target_smooth 只在台架 load_log() 里创建（bam.mujoco 走 numpy 后端）
    m.actuator.q_target_smooth = np.zeros_like(m.actuator.kp)
    print(f"[hd1910] BAM 用 {params_path}: kt={m.kt.value:.4f} R={m.R.value:.4f} "
          f"kp={m.actuator.kp:.0f} vin={m.actuator.vin:.2f} "
          f"max_current={m.actuator.max_current}")
    return m


def main():
    if not os.path.isdir("src/mjlab_microduck"):
        raise SystemExit(
            "请先 cd 到引擎根目录（microduck_rl/）再运行：\n"
            "  cd ~/microduck_rl\n"
            "  uv run python training/hd1910/infer_policy_hd1910.py --walking <onnx> --new-cmd-obs\n"
            f"（当前目录: {os.getcwd()}）")
    # 允许用 --params 覆盖默认 json；其余参数原样转给官方脚本
    params = DEFAULT_PARAMS
    argv = sys.argv[1:]
    if "--params" in argv:
        i = argv.index("--params")
        params = argv[i + 1]
        del argv[i:i + 2]

    ip.load_bam_model = lambda kp_fw, vin, max_current: _load_hd1910_bam(
        kp_fw, vin, max_current, params)

    sys.argv = ["infer_policy.py"] + argv
    ip.main()


if __name__ == "__main__":
    main()
