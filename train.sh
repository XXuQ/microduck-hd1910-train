#!/usr/bin/env bash
# =============================================================================
# microduck-hd1910-train 一键训练脚本
#
# 用法：
#   bash train.sh                      # 用默认推荐任务（走路 Slippery-V6LP）+ 自动探测显存
#   bash train.sh <任务名> [num_envs]  # 指定任务和并行环境数
#
# 前置：已跑过 bash setup.sh（环境搭好，microduck_rl 目录存在）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="${RL_DIR:-$SCRIPT_DIR/../microduck_rl}"

# 默认推荐任务：走路（齿隙+低通+低摩擦，834g 档）
DEFAULT_TASK="Mjlab-Velocity-Flat-Backlash-Slippery-V6LP-HD1910-MicroDuck"
TASK="${1:-$DEFAULT_TASK}"

# --- 自动探测 GPU 显存，决定 num_envs ---
detect_num_envs() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local mem_mb
        mem_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [ -n "$mem_mb" ]; then
            if [ "$mem_mb" -ge 22000 ]; then echo 4096;   # 24G
            elif [ "$mem_mb" -ge 14000 ]; then echo 2048; # 16G
            elif [ "$mem_mb" -ge 9000 ]; then echo 1024;  # 10-12G
            else echo 512; fi                             # 8G
            return
        fi
    fi
    echo 1024  # 默认保守
}
NUM_ENVS="${2:-$(detect_num_envs)}"

echo "======================================================"
echo "  microduck-hd1910-train 训练"
echo "======================================================"
echo "  任务:     $TASK"
echo "  并行环境: $NUM_ENVS"
echo "  显存:     $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '未知') MB"
echo "======================================================"
echo

if [ ! -d "$RL_DIR/.git" ]; then
    echo "错误：未找到 microduck_rl 环境（$RL_DIR）"
    echo "请先运行：bash $SCRIPT_DIR/setup.sh"
    exit 1
fi

cd "$RL_DIR"

echo "[1/3] 确认任务已注册 ..."
uv run list-envs 2>/dev/null | grep -q "$TASK" && echo "  任务存在 ✓" || {
    echo "  警告：任务 '$TASK' 不在注册表里"
    echo "  可用任务（HD1910）："
    uv run list-envs 2>/dev/null | grep HD1910 || true
    echo "  请用上面的任务名重新执行：bash train.sh <任务名>"
    exit 1
}

echo
echo "[2/3] 开始训练（首次会编译 MuJoCo Warp，可能等几分钟）..."
echo "  训练命令：uv run train $TASK --env.scene.num-envs $NUM_ENVS"
echo
uv run train "$TASK" --env.scene.num-envs "$NUM_ENVS"

echo
echo "[3/3] 训练结束。检查点保存位置："
find . -maxdepth 3 -name "model_*.pt" -newermt "-1 hour" 2>/dev/null | tail -5 || echo "  （请用 wandb 或 run 目录确认 checkpoint 路径）"
echo
echo "======================================================"
echo "  下一步：导出 ONNX + 部署到鸭子"
echo "  uv run scripts/export.py $TASK --wandb-run-path <你的run路径>"
echo "======================================================"
