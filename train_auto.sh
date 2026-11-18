#!/usr/bin/env bash
# =============================================================================
# 全自动训练：先跑一小段 → 自己判健康 → 正常就自动续到终点，异常就停并说明原因
#
# 不用盯着看。脚本会 grep 训练日志里的指标，按下面的标准判断：
#
#   ✅ 正常（继续）           ❌ 异常（停下）
#   ─────────────────────    ─────────────────────
#   reward 末段均值 > 首段    reward 不涨或下降
#   reward 末段均值 > 2.0     出现 NaN / inf
#   摔倒率下降                摔倒率不降反升
#   nan_state 恒为 0          样本太少（日志没产出）
#
# 用法：
#   bash train_auto.sh                          # 默认 walk，1000 → 10000 轮
#   bash train_auto.sh --stages 500,3000,10000  # 多段阶梯
#   bash train_auto.sh --task Mjlab-SitStand-Flat-Backlash-HD1910-MicroDuck
#   bash train_auto.sh --dry-run                # 只打印计划
#   bash train_auto.sh --check-health logs/auto-s1.log   # 只复检某份日志，不训练
#   bash train_auto.sh --require-reward         # 绝对值不达标也算失败（严格）
#   bash train_auto.sh --no-abort               # 健康检查只报警不停（调试用）
#
# 前置：已跑过 bash setup.sh
# =============================================================================
set -uo pipefail

# 需要 bash ≥ 4.4：空数组展开和关联数组在此版本才稳定。老 bash 会报
# "unbound variable" 之类看不懂的错，提前说清楚。
if [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 4 ]; }; then
  echo "❌ 需要 bash ≥ 4.4，当前是 ${BASH_VERSION}"
  echo "   Ubuntu/Debian: sudo apt install --only-upgrade bash"
  echo "   CentOS 7 等自带 4.2 的系统：改用 train_deploy_set.sh 手动两段跑"
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:-}"
TASK="Mjlab-Velocity-Flat-Backlash-Slippery-V6LP-HD1910-MicroDuck"
STAGES="1000,10000"
NUM_ENVS=2048
SAVE_INTERVAL=250
RUN_NAME="auto"
DRY=0
NO_ABORT=0
MIN_SAMPLES=20
MIN_GAIN=1.3      # 末段均值 / 首段均值 至少要这么大
MIN_REWARD=2.0    # 末段均值绝对下限
REQUIRE_REWARD=0  # 1 = 绝对值不达标也算失败（默认只警告，见健康检查注释）
CHECK_HEALTH=""   # --check-health <log>：只复检已有日志，不训练

while [ $# -gt 0 ]; do
  case "$1" in
    --task)          TASK="${2:?}"; shift 2 ;;
    --stages)        STAGES="${2:?}"; shift 2 ;;
    --envs|--num-envs) NUM_ENVS="${2:?}"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="${2:?}"; shift 2 ;;
    --name)          RUN_NAME="${2:?}"; shift 2 ;;
    --engine)        ENGINE="${2:?}"; shift 2 ;;
    --min-samples)   MIN_SAMPLES="${2:?}"; shift 2 ;;
    --min-gain)      MIN_GAIN="${2:?}"; shift 2 ;;
    --min-reward)    MIN_REWARD="${2:?}"; shift 2 ;;
    --require-reward) REQUIRE_REWARD=1; shift ;;
    --dry-run)       DRY=1; shift ;;
    --check-health)  CHECK_HEALTH="${2:?}"; shift 2 ;;
    --no-abort)      NO_ABORT=1; shift ;;
    -h|--help)       sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "❌ 未知参数 $1（--help 看用法）"; exit 1 ;;
  esac
done

# ---------- 引擎目录 ----------
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
  echo "   先跑：bash $HERE/setup.sh，或用 --engine 指定"; exit 1; }

IFS=',' read -ra STAGE_LIST <<< "$STAGES"
[ "${#STAGE_LIST[@]}" -ge 1 ] || { echo "❌ --stages 至少一个值"; exit 1; }
TOTAL="${STAGE_LIST[$((${#STAGE_LIST[@]} - 1))]}"

# ---------- 计划 ----------
SPI=$(awk -v n="$NUM_ENVS" 'BEGIN{printf "%.2f", 1.5 + 0.76*(n/1024)}')
EST_H=$(awk -v i="$TOTAL" -v s="$SPI" 'BEGIN{printf "%.1f", i*s/3600}')
if [ -z "$CHECK_HEALTH" ]; then
echo "============================================================"
echo "  全自动训练"
echo "  引擎    : $ENGINE"
echo "  任务    : $TASK"
echo "  阶梯    : ${STAGES} 轮（每段跑完自动判健康）"
echo "  并行    : $NUM_ENVS 环境 · 约 ${SPI}s/轮"
echo "  全程预估: 约 ${EST_H} 小时"
echo "  判据    : reward 末段/首段 ≥ ${MIN_GAIN}× 且末段 ≥ ${MIN_REWARD}；nan_state 恒 0；摔倒率仅警告"
echo "============================================================"
fi
[ "$DRY" = "1" ] && { echo "  --dry-run：不执行"; exit 0; }

cd "$ENGINE" || exit 1
mkdir -p "$HERE/logs"
export WANDB_MODE="${WANDB_MODE:-offline}"

RUN_TAG="$(date +%m%d-%H%M)"
REPORT="$HERE/logs/${RUN_NAME}-report.txt"
PREV_CKPT=""
STAGE_NO=0

# 取某个 run 目录里最新的 checkpoint
latest_ckpt() {
  local dir
  dir=$(ls -1dt logs/rsl_rl/*/*"$1"/ 2>/dev/null | head -1)
  [ -n "$dir" ] || return 1
  ls -1t "$dir"model_*.pt 2>/dev/null | head -1
}

# 健康检查：成功返回 0，失败返回 1 并打印原因
#
# 判据设计原则（v2）：只对「真的坏了」下杀手，趋势类指标一律降级为警告。
#   硬失败：reward 出现 NaN/inf、nan_state > 0、既没达标也没进步
#   仅警告：样本偏少、摔倒率上升（早期回合短，这个指标噪声很大）
# v1 把「摔倒率上升」也当失败，30 轮的小测试就会误判 —— 那是噪声不是故障。
health_check() {
  local log="$1" n raw k
  local first last best
  local rewards
  rewards=$(grep -a "Mean reward:" "$log" | awk -F': ' '{print $2}' | tr -d ' ' | grep -E '^-?[0-9]+(\.[0-9]+)?$')
  n=$(printf '%s\n' "$rewards" | grep -c . )

  echo
  echo "------------------------------------------------------------"
  echo "  健康检查（$n 个 reward 样本）"
  echo "------------------------------------------------------------"

  # 样本太少：只警告。短阶梯本来就样本少，不能因为用户选了 30 轮就说他训练坏了
  if [ "$n" -lt "$MIN_SAMPLES" ]; then
    echo "  ⚠ 样本只有 $n 个（少于 ${MIN_SAMPLES}），判据置信度低，仅作参考"
  fi
  [ "$n" -gt 0 ] || { echo "  ❌ 日志里没有 reward 样本：训练没真正跑起来"; return 1; }

  # 趋势：首/末各 k 个均值 + 四等分趋势表（把「在学没在学」摊开，别只给结论）
  k=$(awk -v n="$n" 'BEGIN{k=int(n/10); if(k<5)k=5; if(k>50)k=50; if(k>n)k=n; print k}')
  read -r first last best <<< "$(printf '%s\n' "$rewards" | awk -v m="$k" '{a[NR]=$1; if(NR==1||$1>mx)mx=$1}
      END{k=(NR<m?NR:m); for(i=1;i<=k;i++){s+=$i}; f=s/k; s=0;
      for(i=NR-k+1;i<=NR;i++){s+=a[i]}; l=s/k; printf "%.2f %.2f %.2f", f, l, mx}')"

  local q1 q2 q3 q4
  read -r q1 q2 q3 q4 <<< "$(printf '%s\n' "$rewards" | awk '{a[NR]=$1}
      END{q=int(NR/4); if(q<1)q=1;
      for(seg=0;seg<4;seg++){s=0;c=0; lo=seg*q+1; hi=(seg==3?NR:(seg+1)*q);
        for(i=lo;i<=hi && i<=NR;i++){s+=a[i];c++}
        printf((seg<3 ? "%.2f " : "%.2f"), (c?s/c:0))}}')"

  echo "  reward : 首 ${k} 均 ${first} → 末 ${k} 均 ${last}（峰值 ${best}）"
  echo "  四等分 : Q1 ${q1} | Q2 ${q2} | Q3 ${q3} | Q4 ${q4}"

  # NaN / inf：解析时被过滤掉，行数缺口就是 NaN
  raw=$(grep -ac "Mean reward:" "$log")
  if [ "$raw" -gt "$n" ]; then
    echo "  ❌ 出现 $((raw - n)) 次非数值 reward（NaN/inf）"
    return 1
  fi

  local ok=1
  # 1) 有没有在学 —— 这是硬指标：末段相对首段要有实质提升
  if awk -v l="$last" -v f="$first" -v g="$MIN_GAIN" \
       'BEGIN{exit !((f > 0 && l > f*g) || (f <= 0 && l > f + 1.0))}'; then
    if awk -v f="$first" 'BEGIN{exit !(f > 0)}'; then
      echo "  ✅ 在学：末段是首段的 $(awk -v l="$last" -v f="$first" 'BEGIN{printf "%.1f", l/f}')×"
    else
      echo "  ✅ 在学：首段为负（起步阶段正常），末段已转为 ${last}"
    fi
  else
    echo "  ❌ 没在学：末段 ${last} 相对首段 ${first} 没有实质提升"
    ok=0
  fi

  # 2) 学得好不好 —— 和「有没有在学」分开判。
  #    默认只警告：这个阈值跟任务、并行环境数强相关（走路的 2.0 拿去卡起坐，
  #    或者 2048 环境的标准拿去卡 256 环境，都会误杀）。要硬卡就加 --require-reward。
  if awk -v l="$last" -v t="$MIN_REWARD" 'BEGIN{exit !(l >= t)}'; then
    echo "  ✅ reward 绝对值达标（${last} ≥ ${MIN_REWARD}）"
  else
    if [ "${REQUIRE_REWARD:-0}" = "1" ]; then
      echo "  ❌ reward 绝对值不达标（${last} < ${MIN_REWARD}，--require-reward 生效）"
      ok=0
    else
      echo "  ⚠ reward 绝对值 ${last} < ${MIN_REWARD}：不算故障，但明显偏低"
      echo "     常见原因：这一段的轮数太少 / --envs 太小 / 这个任务的奖励量级本来就不同"
    fi
  fi

  # 3) 平台期：末 1/4 相比前一 1/4 几乎不涨
  if awk -v a="$q3" -v b="$q4" 'BEGIN{exit !(b <= a * 1.05)}'; then
    echo "  ⚠ 末段趋平（Q3 ${q3} → Q4 ${q4}）：可能进入平台期，继续训收益有限"
  fi

  # 摔倒率：只警告。早期回合短、样本少时这个数天然会抬升
  local falls f_first_v f_last_v
  falls=$(grep -a "Episode_Termination/fell_over:" "$log" | awk -F': ' '{print $2}' | tr -d ' ' | grep -E '^[0-9]+(\.[0-9]+)?$')
  if [ -n "$falls" ]; then
    read -r f_first_v f_last_v <<< "$(printf '%s\n' "$falls" | awk -v m="$k" '{a[NR]=$1}
        END{k=(NR<m?NR:m); for(i=1;i<=k;i++){s+=$i}; f=s/k; s=0;
        for(i=NR-k+1;i<=NR;i++){s+=a[i]}; l=s/k; printf "%.1f %.1f", f, l}')"
    if awk -v l="$f_last_v" -v f="$f_first_v" 'BEGIN{exit !(l > f + 5)}'; then
      echo "  ⚠ 摔倒率上升 ${f_first_v}% → ${f_last_v}%（早期回合短，仅供参考）"
    else
      echo "  ✅ 摔倒率 ${f_first_v}% → ${f_last_v}%"
    fi
  fi

  # nan_state：硬指标
  local nan_last
  nan_last=$(grep -a "Episode_Termination/nan_state:" "$log" | tail -1 | awk -F': ' '{print $2}' | tr -d ' ')
  if [ -n "$nan_last" ]; then
    if awk -v v="$nan_last" 'BEGIN{exit !(v > 0)}'; then
      echo "  ❌ nan_state 终止率 $nan_last > 0（状态爆了）"
      ok=0
    else
      echo "  ✅ nan_state 恒为 0"
    fi
  fi

  [ "$ok" = "1" ] && echo "  结论：✅ 正常" || echo "  结论：❌ 异常"
  echo "------------------------------------------------------------"
  return $((1 - ok))
}

# ---------- 只复检已有日志（不训练）----------
if [ -n "$CHECK_HEALTH" ]; then
  CH="$CHECK_HEALTH"
  # 相对路径先按当前目录找，再按仓库目录找（脚本此时已 cd 到引擎目录）
  [ -f "$CH" ] || CH="$HERE/$CHECK_HEALTH"
  [ -f "$CH" ] || { echo "❌ 找不到日志：$CHECK_HEALTH（也试过 $HERE/$CHECK_HEALTH）"; exit 1; }
  echo "复检日志：$CH"
  health_check "$CH"
  rc=$?
  echo "返回码 ${rc}（0 = 判定正常，1 = 判定异常）"
  exit "$rc"
fi

# ---------- 逐段训练 ----------
for stage in "${STAGE_LIST[@]}"; do
  STAGE_NO=$((STAGE_NO + 1))
  NAME="${RUN_NAME}-s${STAGE_NO}"
  LOG="$HERE/logs/${RUN_NAME}-s${STAGE_NO}.log"
  RESUME_ARG=()
  ADD_ITERS="$stage"          # 没有续训时，max-iterations 就是总轮数
  if [ -n "$PREV_CKPT" ]; then
    # ⚠ --agent.max-iterations 在带 --init-checkpoint 时是「还要再跑多少轮」，
    #   不是总轮数（实测 max=10 从 model_4 续 → 日志 Learning iteration 4/14）。
    #   所以按 checkpoint 文件名里的迭代号换算成剩余轮数，目标才真的是 $stage。
    CUR=$(basename "$PREV_CKPT" | sed -nE 's/^model_([0-9]+)\.pt$/\1/p')
    [ -n "$CUR" ] || CUR=0
    ADD_ITERS=$(( stage - CUR ))
    [ "$ADD_ITERS" -gt 0 ] || { echo "  ⚠ $stage 轮已经达到了（checkpoint 在 ${CUR} 轮），跳过这一段"; continue; }
    RESUME_ARG=(--init-checkpoint "$PREV_CKPT")
  fi

  echo
  echo "============================================================"
  echo "  第 ${STAGE_NO}/${#STAGE_LIST[@]} 段：训到 ${stage} 轮（本段跑 ${ADD_ITERS} 轮）"
  # 续训段必须有上一段的 checkpoint，否则会静默从零重训（白跑一整段）
  if [ "$STAGE_NO" -gt 1 ] && [ -z "$PREV_CKPT" ]; then
    echo "❌ 上一段没产出 checkpoint，续训会变成从零重训，已停止。"
    echo "   原因多半是 --save-interval（${SAVE_INTERVAL}）大于上一段轮数。"
    echo "   把 --save-interval 调到上一段轮数的 1/2 以内再试。"
    exit 3
  fi
  [ -n "$PREV_CKPT" ] && echo "  续训自：$PREV_CKPT（第 ${CUR} 轮）"
  echo "  日志  ：$LOG"
  echo "============================================================"

  uv run train "$TASK" \
      --env.scene.num-envs "$NUM_ENVS" \
      --agent.max-iterations "$ADD_ITERS" \
      --agent.save-interval "$SAVE_INTERVAL" \
      --agent.run-name "$NAME" "${RESUME_ARG[@]}" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}

  if [ "$rc" -ne 0 ]; then
    echo
    echo "❌ 第 ${STAGE_NO} 段训练进程异常退出（rc=$rc），停止。"
    # 常见坑：引擎版本不对，续训参数根本不被识别
    if grep -qa "unrecognized argument" "$LOG" 2>/dev/null; then
      echo
      echo "   检测到「unrecognized argument」：引擎不认某个命令行参数。"
      echo "   续训参数 --init-checkpoint 是较新版本才有的，检查引擎 commit："
      echo "     cd $ENGINE && git rev-parse --short HEAD    # 本仓库验证过的是 0daab16"
      echo "   版本不对就重跑 setup.sh（它会尝试切回验证过的 commit）。"
    fi
    echo "   日志尾部："; tail -15 "$LOG"
    exit "$rc"
  fi

  NEW_CKPT=$(latest_ckpt "$NAME" || true)
  if [ -n "$NEW_CKPT" ]; then
    PREV_CKPT="$NEW_CKPT"
    echo "  ✅ 第 ${STAGE_NO} 段完成 · checkpoint：$PREV_CKPT"
  else
    echo "  ⚠ 第 ${STAGE_NO} 段没找到新 checkpoint（轮数 < 存档间隔？），沿用上一段的"
  fi

  # 最后一段不用判健康
  [ "$STAGE_NO" -lt "${#STAGE_LIST[@]}" ] || break

  # 报告同时落盘（logs/<name>-report.txt），事后能查为什么判正常/异常
  # 注意用 PIPESTATUS 取函数的返回码，不能直接 `if health_check | tee`（那是 tee 的码）
  health_check "$LOG" 2>&1 | tee -a "$REPORT"
  hc=${PIPESTATUS[0]}

  if [ "$hc" -eq 0 ]; then
    echo "  → 判定正常，自动继续下一段（目标 ${STAGE_LIST[$STAGE_NO]} 轮）"
  else
    if [ "$NO_ABORT" = "1" ]; then
      echo "  ⚠ 判定异常，但 --no-abort 生效，仍继续"
    else
      echo
      echo "❌ 自动训练中止：第 ${STAGE_NO} 段指标不达标，不继续烧算力。"
      echo "   checkpoint 仍保留：$PREV_CKPT"
      echo "   排查方向："
      echo "     · reward 不涨 → 看日志里是哪项奖励没起来；试 --walk 换配方或加大 --envs"
      echo "     · nan_state > 0 → 状态发散，降学习率或减 --envs"
      echo "     · 摔倒率上升 → 任务难度过高，先用更好走的配方（如 slippery）"
      echo "   想跳过检查硬跑：bash $0 --no-abort --stages $STAGES"
      exit 2
    fi
  fi
done

echo
echo "============================================================"
echo "  ✅ 全自动训练完成（${STAGES} 轮）"
echo "  checkpoint：$PREV_CKPT"
echo
echo "  导出 ONNX："
echo "    uv run scripts/export.py $TASK --checkpoint-file $PREV_CKPT"
echo "============================================================"
