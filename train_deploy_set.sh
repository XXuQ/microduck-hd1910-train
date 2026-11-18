#!/usr/bin/env bash
# =============================================================================
# 按「官方部署套件」重训 HD-1910 策略：6 个槽位，每个 1 万轮
#
# 清单来源（官方 pollen-robotics/microduck 源码，不是猜的）：
#   deploy/robotd.toml  —— 槽位默认值，逐行抄下来的
#   duck-control/src/policy.rs —— 速度幅值阈值 0.05 选站立、技能由调度器显式请求
#
# 官方部署的 7 个槽位 / 6 个网络：
#   walk        = velstand.onnx          走 + 零指令站住（同一个网络）
#   stand       = none                   velstand 自己站，官方已废弃独立站立网络
#   sitstand    = alpha_sitstand.onnx    坐 ↔ 站
#   ground_pick = alpha_ground_pick.onnx A 键捡东西
#   kick_left   = ball_kick_left.onnx    踢球（左）
#   kick_right  = ball_kick_right.onnx   踢球（右）
#   roulade     = roulade.onnx           X 键前滚翻
#
# 所以一共要 6 个网络，不是 7 个 —— stand 槽是 none。
# 「摔倒自己爬起来」不在这个清单里：官方用 limp_fall 控制逻辑接管（增益降到 50 装死，
# 落地后缓回站立姿势再交给零指令的 velstand），所以 StandUp 任务训了也没有部署槽。
#
# 轮数：官方文档写 4000，但 kissqy 实机随包模型的 checkpoint 是 9599～11000，
# 说明 4000 只够看出趋势，收敛在 1 万。本脚本默认 10000。
#
# 用法：
#   bash train_deploy_set.sh                        # 6 个槽全跑，1 万轮/个
#   bash train_deploy_set.sh --only walk,sitstand   # 只训指定槽位
#   bash train_deploy_set.sh --dry-run              # 只打印计划，不训练
#   bash train_deploy_set.sh --num-envs 4096        # 大显存卡
#   bash train_deploy_set.sh --walk widefriction    # walk 换配方（slippery|widefriction|lowfriction|velstand）
#   bash train_deploy_set.sh --kick-official        # 明知道硬件不对，仍训官方踢球任务
#
# 两段式训练（先看 1000 轮再决定要不要继续）：
#   # 第一段：1000 轮
#   bash train_deploy_set.sh --only walk --iters 1000
#   # 看完曲线正常，第二段接着训到 1 万轮（--iters 写最终总轮数）
#   bash train_deploy_set.sh --only walk --iters 10000 \
#        --resume logs/rsl_rl/velocity/<run>/model_999.pt
#
# 前置：已跑过 bash setup.sh（引擎目录就位）
# =============================================================================
set -uo pipefail

# 需要 bash ≥ 4.4（空数组展开 "${arr[@]}" 在老版本 + set -u 下会报 unbound variable）
if [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 4 ]; }; then
  echo "❌ 需要 bash ≥ 4.4，当前是 ${BASH_VERSION}"
  echo "   Ubuntu/Debian: sudo apt install --only-upgrade bash"
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:-}"

ITERS=10000
NUM_ENVS=2048
SAVE_INTERVAL=500
ONLY=""
RESUME=""
DRY=0
WALK_VARIANT="slippery"
KICK_OFFICIAL=0

# ---------- 槽位 → HD-1910 任务映射 ----------
# 空字符串 = 没有 HD-1910 变体，默认跳过（见 --kick-official）
SLOTS=(walk sitstand ground_pick kick_left kick_right roulade)
declare -A TASK=(
  [sitstand]="Mjlab-SitStand-Flat-Backlash-HD1910-MicroDuck"
  [ground_pick]="Mjlab-GroundPick-Flat-Backlash-Mass920-Sym-Cloth-HD1910-MicroDuck"
  [roulade]="Mjlab-Roulade-Flat-Backlash-HD1910-MicroDuck"
  # 官方只有 XL330/737g 的 BallKick，没有 HD-1910 变体（已核对任务注册表）
  [kick_left]=""
  [kick_right]=""
)

# 官方 walk 槽是 VelStand（走路 + 零指令站立合并成一个网络）。
#
# 但当前引擎版本（0daab16）上，HD-1910 的 VelStand 任务建环境就崩：
#   ValueError: Not all regular expressions are matched: ^.*_servo_collision$
# 根因是 834g 资产的 allcollisions XML 里 15 个舵机只有 class="visual" 几何、
# 没有 class="collision" 几何（官方是每个舵机两条），VelStand 的保护性摔倒要一个
# contact sensor 盯舵机壳，正则匹配为空就直接抛错。
# 在资产补丁落地之前，walk 槽默认用 Velocity：同样的 61→14 契约、同样带 V6LP 低通，
# 只是零指令时不会像 velstand 那样稳住不动（靠 2% 零命令环境顺带学）。
# 资产修好后用 --walk velstand 切回官方方案。
set_walk_task() {
  case "$1" in
    slippery)     TASK[walk]="Mjlab-Velocity-Flat-Backlash-Slippery-V6LP-HD1910-MicroDuck"
                  WALK_LOWPASS="0.5 / 0.7（V6LP）" ;;
    widefriction) TASK[walk]="Mjlab-Velocity-Flat-Backlash-WideFriction-V6LP-HD1910-MicroDuck"
                  WALK_LOWPASS="0.5 / 0.7（V6LP）" ;;
    lowfriction)  TASK[walk]="Mjlab-Velocity-Flat-Backlash-LowFriction-V6LP-HD1910-MicroDuck"
                  WALK_LOWPASS="0.5 / 0.7（V6LP）" ;;
    velstand)     TASK[walk]="Mjlab-VelStand-Flat-Backlash-WideFriction-V6LP-Sym-HD1910-MicroDuck"
                  WALK_LOWPASS="0.5 / 0.7（V6LP）"
                  echo "  ⚠ --walk velstand：官方方案（走+站一个网络）。舵机碰撞几何已补，能建环境，"
                  echo "    但会去下载官方 XL330 的 BC 教师（需要 wandb key / HF，且硬件错配），"
                  echo "    先按 velstand 那条路关掉 BC（ENABLE_EXPERT_BC=False）再训。" ;;
    *) echo "❌ --walk 只接受 slippery | widefriction | lowfriction | velstand"; exit 1 ;;
  esac
}
set_walk_task "$WALK_VARIANT"

# ---------- 参数 ----------
while [ $# -gt 0 ]; do
  case "$1" in
    --iters)         ITERS="${2:?}"; shift 2 ;;
    --num-envs)      NUM_ENVS="${2:?}"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="${2:?}"; shift 2 ;;
    --only)          ONLY="${2:?}"; shift 2 ;;
    --resume)        RESUME="${2:?}"; shift 2 ;;
    --engine)        ENGINE="${2:?}"; shift 2 ;;
    --walk)          WALK_VARIANT="${2:?}"; set_walk_task "$WALK_VARIANT"; shift 2 ;;
    --dry-run)       DRY=1; shift ;;
    --kick-official) KICK_OFFICIAL=1; shift ;;
    -h|--help)       sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "❌ 未知参数 $1（--help 看用法）"; exit 1 ;;
  esac
done

if [ "$KICK_OFFICIAL" = "1" ]; then
  TASK[kick_left]="Mjlab-BallKick-Flat-Backlash-MicroDuck"
  TASK[kick_right]="Mjlab-BallKick-Flat-Backlash-MicroDuck"
fi

# `--only walk,sitstand` → 只保留这些槽
SELECTED=("${SLOTS[@]}")
if [ -n "$ONLY" ]; then
  SELECTED=()
  IFS=',' read -ra WANT <<< "$ONLY"
  for w in "${WANT[@]}"; do
    w="$(echo "$w" | tr -d ' ')"
    found=0
    for s in "${SLOTS[@]}"; do [ "$s" = "$w" ] && found=1; done
    [ "$found" = "1" ] || { echo "❌ 未知槽位 '$w'（可选：${SLOTS[*]}）"; exit 1; }
    SELECTED+=("$w")
  done
fi

# ---------- 前置检查 ----------
# 引擎目录：--engine 优先；否则读 setup.sh 写下的 .engine_path（支持自定义路径）；
# 再按常见位置找（仓库同级 → 仓库下的 engine/ → ~/microduck_rl → ~/ducktest/microduck_rl）
if [ -z "$ENGINE" ] && [ -f "$HERE/.engine_path" ]; then
  ENGINE="$(tr -d '\r\n' < "$HERE/.engine_path")"
fi
if [ -z "$ENGINE" ]; then
  for c in "$HERE/../microduck_rl" "$HERE/engine" "$HOME/microduck_rl" "$HOME/ducktest/microduck_rl"; do
    if [ -d "$c/.git" ]; then ENGINE="$(cd "$c" && pwd)"; break; fi
  done
fi

[ -d "$ENGINE/.git" ] || {
  echo "❌ 找不到引擎目录 ${ENGINE:-（未找到）}"
  echo "   先跑：bash $HERE/setup.sh，或用 --engine /path/to/microduck_rl 指定"; exit 1; }

# --resume：两段式训练用。⚠ --agent.max-iterations 在带 --init-checkpoint 时是
# 「还要再跑多少轮」而不是总轮数（实测 max=10 从 model_4 续 → 日志显示 4/14）。
# 所以这里按 checkpoint 文件名里的迭代号换算，用户侧 --iters 仍然写「最终总轮数」。
ITERS_ARG="$ITERS"
if [ -n "$RESUME" ]; then
  [ "${#SELECTED[@]}" -eq 1 ] || {
    echo "❌ --resume 只能配单个槽位：--only walk --resume <ckpt>"; exit 1; }
  if [ ! -f "$RESUME" ] && [ ! -f "$ENGINE/$RESUME" ]; then
    echo "❌ 找不到 checkpoint：$RESUME"; exit 1
  fi
  CUR=$(basename "$RESUME" | sed -nE 's/^model_([0-9]+)\.pt$/\1/p')
  [ -n "${CUR:-}" ] || CUR=0
  ITERS_ARG=$(( ITERS - CUR ))
  [ "$ITERS_ARG" -gt 0 ] || {
    echo "❌ checkpoint 已在第 ${CUR} 轮，已达到 --iters $ITERS。要续就直接调大 --iters"; exit 1; }
fi

# ---------- 计划表 ----------
line() { echo "------------------------------------------------------------"; }
echo "============================================================"
echo "  官方 6 槽部署套件 · HD-1910 重训"
echo "  引擎    : $ENGINE"
echo "  轮数    : $ITERS 轮/个（kissqy 实机收敛点在 9599～11000）"
echo "  并行    : $NUM_ENVS 环境"
echo "  存档    : 每 $SAVE_INTERVAL 轮"
[ -n "$RESUME" ] && echo "  续训    : 从第 ${CUR} 轮接着跑 ${ITERS_ARG} 轮（目标 ${ITERS}）"
line
printf "%-12s %-58s %s\n" "槽位" "任务" "状态"

TRAIN_LIST=()
SKIPPED=()
for s in "${SELECTED[@]}"; do
  t="${TASK[$s]:-}"
  if [ -z "$t" ]; then
    printf "%-12s %-58s %s\n" "$s" "—" "跳过（无 HD-1910 变体）"
    SKIPPED+=("$s")
  else
    printf "%-12s %-58s %s\n" "$s" "$t" "待训"
    TRAIN_LIST+=("$s")
  fi
done

# 时长估算：按 RTX 2060 实测 256/1024/2048/4096 → 1.80/2.23/2.99/4.63 s/轮拟合
SPI=$(awk -v n="$NUM_ENVS" 'BEGIN{printf "%.2f", 1.5 + 0.76*(n/1024)}')
if [ "${#TRAIN_LIST[@]}" -gt 0 ]; then
  EST_H=$(awk -v i="$ITERS" -v s="$SPI" -v k="${#TRAIN_LIST[@]}" 'BEGIN{printf "%.1f", i*s*k/3600}')
  line
  echo "  预估每轮 ${SPI}s · 共 ${#TRAIN_LIST[@]} 个任务 · 约 ${EST_H} 小时（按 RTX 2060 实测拟合）"
fi

if [ "${#SKIPPED[@]}" -gt 0 ]; then
  echo
  echo "  ⚠ 以下槽位没有 HD-1910 变体，本次不训：${SKIPPED[*]}"
  echo "    官方只有 Mjlab-BallKick-Flat-Backlash-MicroDuck（XL330 / 737g），"
  echo "    拿它训出来的策略上不了你的飞特 HD-1910 + 834g。"
  echo "    要真的会踢球，得先照 hd1910_tasks.py 的写法补一个 BallKick 的 HD-1910 变体。"
fi

if [ "$DRY" = "1" ]; then
  line
  echo "  --dry-run：只打印命令，不训练"
  for s in "${TRAIN_LIST[@]}"; do
    echo "  uv run train ${TASK[$s]} --env.scene.num-envs $NUM_ENVS \\"
    echo "      --agent.max-iterations $ITERS_ARG --agent.save-interval $SAVE_INTERVAL --agent.run-name deploy-$s${RESUME:+ --init-checkpoint $RESUME}"
  done
  exit 0
fi

[ "${#TRAIN_LIST[@]}" -gt 0 ] || { echo; echo "没有可训的槽位。"; exit 0; }

# ---------- 预检：任务名是否真的注册了 ----------
# 任务名以 list-envs 实际输出为准（官方文档也会漂移）。这里先查一遍，
# 免得几十小时的训练跑到第 3 个任务才发现名字写错。
echo
echo "  校验任务注册表 ..."
VALID="$(mktemp)"
( cd "$ENGINE" && uv run list-envs 2>/dev/null ) > "$VALID" || true
MISSING=()
for s in "${TRAIN_LIST[@]}"; do
  grep -q "${TASK[$s]}" "$VALID" || MISSING+=("$s")
done
rm -f "$VALID"
if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "  ❌ 以下任务没在注册表里（引擎或 tasks 文件版本不对）："
  for s in "${MISSING[@]}"; do echo "     $s → ${TASK[$s]}"; done
  echo "  跑 uv run list-envs | grep HD1910 看实际任务名"
  exit 1
fi
echo "  ✅ 任务名全部有效"

# ---------- 逐个训练 ----------
cd "$ENGINE" || exit 1
mkdir -p "$HERE/logs"
export WANDB_MODE="${WANDB_MODE:-offline}"

DONE=0
FAILED=()
RUNS=()
RESUME_ARG=()
[ -n "$RESUME" ] && RESUME_ARG=(--init-checkpoint "$RESUME")
for s in "${TRAIN_LIST[@]}"; do
  i=$((DONE + 1))
  echo
  echo "============================================================"
  echo "  [$i/${#TRAIN_LIST[@]}] 训练槽位：$s"
  echo "  任务：${TASK[$s]}"
  line
  uv run train "${TASK[$s]}" \
      --env.scene.num-envs "$NUM_ENVS" \
      --agent.max-iterations "$ITERS_ARG" \
      --agent.save-interval "$SAVE_INTERVAL" \
      --agent.run-name "deploy-$s" "${RESUME_ARG[@]}" 2>&1 | tee "$HERE/logs/deploy-$s.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "  ❌ $s 训练失败（退出码 $rc），日志：logs/deploy-$s.log"
    FAILED+=("$s")
    continue
  fi
  # 注意尾部斜杠：不带 `/` 时 $RUN 是目录名而非 "目录/"，拼 model_*.pt 会粘在一起
  RUN=$(ls -1dt logs/rsl_rl/*/*deploy-$s*/ 2>/dev/null | head -1)
  CKPT=$(ls -1t "$RUN"model_*.pt 2>/dev/null | tail -1)
  echo "  ✅ $s 完成 · checkpoint：$CKPT"
  RUNS+=("$s|${TASK[$s]}|$CKPT")
  DONE=$((DONE + 1))
done

# ---------- 汇总 ----------
echo
echo "============================================================"
echo "  训练结束：成功 $DONE 个，失败 ${#FAILED[@]} 个"
[ "${#FAILED[@]}" -gt 0 ] && echo "  失败槽位：${FAILED[*]}"
line
echo "  导出 ONNX（normalizer 烘进图里，必须用官方 export）："
for r in "${RUNS[@]:-}"; do
  slot="${r%%|*}"; rest="${r#*|}"; task="${rest%%|*}"; ckpt="${rest##*|}"
  if [ -z "$ckpt" ]; then
    echo "    # $slot：没找到 checkpoint（轮数太少还没到存档点，或 --save-interval 太大）"
    continue
  fi
  echo "    uv run scripts/export.py $task \\"
  echo "        --checkpoint-file $ckpt   # → deploy_${slot}.onnx"
done
line
echo "  部署（robotd.toml 的 [policy]，对照官方 deploy/robotd.toml）："
echo "    walk        = deploy_walk.onnx"
echo "    stand       = none                    # velstand 自己站，别装独立站立网络"
echo "    sitstand    = deploy_sitstand.onnx"
echo "    ground_pick = deploy_ground_pick.onnx"
echo "    roulade     = deploy_roulade.onnx"
echo "    head_lowpass = 0.5                    # 必须和训练一致：walk 槽 = $WALK_LOWPASS"
echo "    legs_lowpass = 0.7"
echo "    action_scale = 0.7                    # kissqy 实机配置；0.9/1.0 实测会锁死不动"
echo "    gain = 5                              # 飞特舵机；别抄官方的 200（那是 XL330 标度）"
echo "    limp_fall = false                     # kissqy 关掉了；要装死保护就设 true"
echo "============================================================"
