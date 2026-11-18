#!/usr/bin/env bash
# =============================================================================
# 250 轮冒烟训练：验证「环境能训起来」而不是「训出会走的策略」。
#
# 检查点：
#   - 环境能创建、warp 能编译、显存不炸
#   - 250 轮跑完后 reward 有变化（不是恒 NaN / 恒定值）
#   - 落盘 model_*.pt，能接着 play / export
#
# 用法：
#   bash smoke_train.sh                 # 默认 512 envs × 250 轮
#   bash smoke_train.sh 256 250         # 6GB 显存建议这样
#   TASK=Mjlab-Velocity-Flat-Backlash-HD1910-MicroDuck bash smoke_train.sh
# =============================================================================
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:-$HERE/../microduck_rl}"
ENVS="${1:-512}"
ITERS="${2:-250}"
TASK="${TASK:-Mjlab-Velocity-Flat-HD1910-MicroDuck}"

cd "$ENGINE" || { echo "找不到引擎目录 $ENGINE（先跑 setup.sh）"; exit 1; }

# 默认 logger 是 wandb，没有 API key 会直接报错退出。
# 冒烟测试用 offline 模式（本地记录，不需要登录）；想上传就 `uv run wandb login` 后
# export WANDB_MODE=online 再跑。
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "============================================================"
echo "  冒烟训练"
echo "    任务      : $TASK"
echo "    并行环境  : $ENVS"
echo "    轮数      : $ITERS"
echo "    引擎目录  : $ENGINE"
echo "============================================================"

before=$(ls -1d logs/rsl_rl/*/*/ 2>/dev/null | wc -l)
start=$(date +%s)

uv run train "$TASK" \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$ITERS" 2>&1 | tee /tmp/smoke_train.log
rc=${PIPESTATUS[0]}

dur=$(( $(date +%s) - start ))
echo
echo "============================================================"
echo "  退出码 $rc，耗时 ${dur}s"
echo "============================================================"

if [ "$rc" -ne 0 ]; then
    echo "❌ 训练失败，最后 30 行日志："
    tail -30 /tmp/smoke_train.log
    exit "$rc"
fi

RUN=$(ls -1dt logs/rsl_rl/*/*/ 2>/dev/null | head -1)
echo "  最新 run 目录: $RUN"
CKPT=$(ls -1t "$RUN"model_*.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    echo "  ✅ 已落盘 checkpoint: $CKPT"
else
    echo "  ⚠ 没有找到 model_*.pt（可能是 max_iterations 太小还没到保存点）"
fi

echo
echo "  看 reward 是否在变（有变化≠NaN 即环境正常）："
grep -E "Mean reward|mean_reward|reward" /tmp/smoke_train.log | tail -5 | sed 's/^/    /'

echo
echo "  下一步："
echo "    uv run play $TASK --checkpoint-file $CKPT"
echo "    uv run python training/hd1910/joint_viewer.py     # 看舵机角度"
