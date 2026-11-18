#!/usr/bin/env bash
# =============================================================================
# microduck-hd1910-train 一键环境搭建
#
# 在任意 Linux + NVIDIA GPU 机器上，从零搭出「834g / HD-1910 / 2S」的训练环境：
#   1. 克隆官方 microduck_rl 训练引擎（mjlab），自动挑可用镜像
#   2. uv sync 装依赖
#   3. 白名单覆盖：只补网格资产 + add_backlash.py，官方 XML 保持仓库版本
#   4. 拷入 HD-1910 定制文件（constants / tasks / BAM / 回放 / 3D 查看器）
#   5. 由 806g 基准生成 **834g** 资产（头部缩放 + 独立电池 + 铝件加重 + 舵机碰撞体）
#   6. 注册任务并自检（任务数、834.0 g、执行器数量）
#
# 用法：
#   bash setup.sh                 # 在仓库同级目录建 microduck_rl/
#   bash setup.sh /path/to/engine # 指定引擎目录
#
# 前置：Linux + NVIDIA GPU（≥8G）+ uv（curl -LsSf https://astral.sh/uv/install.sh | sh）
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${1:-$HERE/../microduck_rl}"

# 国内网络默认走镜像；海外或镜像不可用时可 export MICRODUCK_MIRROR= 走直连
MIRROR="${MICRODUCK_MIRROR-https://gh-proxy.com/}"
UPSTREAM="https://github.com/pollen-robotics/microduck_rl.git"
# 本仓库验证过的官方引擎版本。官方 HEAD 一直在动（任务注册、env cfg、资产都会变），
# 换台机器一键训练时如果拉到更新的 HEAD，可能出现任务名对不上、参数被改名。
# 留着这个值：拉完发现不是它，就尝试切回去；切不回去会明确警告而不是静默跑错。
# 想跟随官方最新：export ENGINE_COMMIT=
ENGINE_COMMIT="${ENGINE_COMMIT-0daab16}"

line() { echo "------------------------------------------------------------"; }
fail() { echo; echo "❌ $*"; exit 1; }

echo "============================================================"
echo "  microduck-hd1910-train 环境搭建"
echo "  目标引擎目录 : $WORK_DIR"
echo "  模型          : 834 g / HD-1910 / 2S（Kp${HD1910_KP:-5}, 1.3 A, 7.4 V）"
line

# ---------- 0. 前置检查（早失败，报错可读）----------
echo "[0/6] 前置检查 ..."

# 只支持 Linux / WSL2：训练依赖 MuJoCo Warp + CUDA
case "$(uname -s)" in
  Linux) : ;;
  *) fail "本仓库只能在 Linux 或 WSL2 上跑（当前系统：$(uname -s)）。
   训练依赖 MuJoCo Warp + NVIDIA CUDA，macOS / Windows 原生都跑不了。
   Windows 请用 WSL2：wsl --install -d Ubuntu" ;;
esac

# 脚本用到 bash 4.4+ 的语法（空数组展开等），老 bash 报错极难懂
if [ "${BASH_VERSINFO[0]}" -lt 4 ] \
   || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 4 ]; }; then
  fail "需要 bash >= 4.4（当前 ${BASH_VERSION}）。Ubuntu: sudo apt install --only-upgrade bash"
fi

command -v git >/dev/null 2>&1 || fail "没装 git。Ubuntu: sudo apt install -y git"
command -v python3 >/dev/null 2>&1 || fail "没装 python3。Ubuntu: sudo apt install -y python3"
command -v uv >/dev/null 2>&1 || fail "没装 uv。执行： curl -LsSf https://astral.sh/uv/install.sh | sh && source \$HOME/.local/bin/env"
echo "  git $(git --version | awk '{print $3}') / python3 $(python3 -V | awk '{print $2}') / $(uv --version)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "  ⚠ 没有 nvidia-smi：训练需要 NVIDIA GPU（MuJoCo Warp 不支持 CPU/AMD/Mac）"
elif ! nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
  echo "  ⚠ 检测不到 GPU（驱动未装好？WSL 需在 Windows 侧装 NVIDIA 驱动）"
else
  echo "  GPU: $(nvidia-smi -L | head -1 | sed 's/GPU 0: //; s/ (UUID.*//')"
fi

# 磁盘：官方引擎 + CUDA wheels + venv 约 10–12 GB
AVAIL_GB=$(df -BG --output=avail "$(dirname "$WORK_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${AVAIL_GB:-}" ]; then
  echo "  可用磁盘: ${AVAIL_GB} GB"
  [ "$AVAIL_GB" -lt 12 ] && echo "  ⚠ 建议留 ≥12 GB（官方仓库 + ~2 GB CUDA wheels + venv）"
fi

# 网络：GitHub 拉代码/依赖，PyPI 拉 wheels
if command -v curl >/dev/null 2>&1; then
  if curl -sm 6 -o /dev/null https://github.com 2>/dev/null; then
    echo "  GitHub 可达"
  else
    echo "  ⚠ GitHub 直连不通：将走镜像 $MIRROR；若也失败请设 MICRODUCK_MIRROR=http://..."
    echo "    （bam 是 git 依赖，需 GitHub；可配 gh-proxy：
      git config --global url.\"https://gh-proxy.com/https://github.com/Rhoban/\".insteadOf \"https://github.com/Rhoban/\"）"
  fi
  # uv sync 从 PyPI 拉 wheels，这一步不通训练环境装不起来
  if curl -sm 6 -o /dev/null https://pypi.org/simple/ 2>/dev/null; then
    echo "  PyPI 可达"
  else
    echo "  ⚠ PyPI 直连不通：uv sync 可能很慢或失败"
    echo "    可设镜像： export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/"
  fi
fi
echo

# ---------- 1. 克隆官方引擎 ----------
if [ ! -d "$WORK_DIR/.git" ]; then
  echo "[1/6] 克隆 microduck_rl ..."
  mkdir -p "$(dirname "$WORK_DIR")"
  if [ -n "$MIRROR" ] && git clone --depth 1 "${MIRROR}${UPSTREAM}" "$WORK_DIR" 2>/dev/null; then
    echo "  经镜像克隆完成"
  else
    echo "  镜像不可用，改用直连（可能较慢）..."
    rm -rf "$WORK_DIR"
    git clone --depth 1 "$UPSTREAM" "$WORK_DIR"
  fi
else
  echo "[1/6] 引擎目录已存在，跳过克隆"
fi
cd "$WORK_DIR"
GOT="$(git rev-parse --short HEAD)"
echo "  当前 commit: $GOT"

# 官方引擎漂移检查：拉到的不是本仓库验证过的版本就尝试切回去
if [ -n "${ENGINE_COMMIT:-}" ] && [ "${GOT#"$ENGINE_COMMIT"}" = "$GOT" ]; then
  echo "  官方已更新到 $GOT（本仓库在 $ENGINE_COMMIT 上验证过），尝试切回 ..."
  if git fetch --depth 1 origin "$ENGINE_COMMIT" >/dev/null 2>&1 \
     && git checkout -q FETCH_HEAD 2>/dev/null; then
    echo "  已锁定到 $(git rev-parse --short HEAD)"
  else
    echo "  ⚠ 切不回 $ENGINE_COMMIT，继续用 $GOT。"
    echo "    若出现「任务名不存在 / 参数未知 / XML not found」，先怀疑是引擎漂移："
    echo "    export ENGINE_COMMIT=<一个可用的 commit> 后重跑，或 export ENGINE_COMMIT= 跟随最新。"
  fi
fi

# ---------- 2. 覆盖模型目录（只盖真正需要的，白名单）----------
echo "[2/6] 叠加固化模型（白名单覆盖，官方新增文件一律保留）..."
ROBOT=src/mjlab_microduck/robot
# 以前这里是 `cp -rf model/robot/microduck/. $ROBOT/microduck/` 整目录覆盖。
# 那样会把官方后来改过的 XML 一起顶回旧版：实测发现官方 robot_allcollisions.xml
# 新加的 <contact> 排除块被覆盖掉之后，连官方原版的 VelStand 任务都建不起来
# （报 ^.*_servo_collision$ 匹配失败）。所以改成白名单，只盖两样东西：
#   assets/         —— HD-1910 的 XML 用 meshdir="../microduck/assets" 读这里的网格
#   add_backlash.py —— 第 5 步给 backlash 变体注入 2° 齿隙要用它
# make_models.py 并不读 microduck/ 下的任何 XML（它只读 microduck_hd1910_base），
# 所以冻结 XML 没有收益，只有副作用。
mkdir -p "$ROBOT/microduck/assets"
cp -rf "$HERE/model/robot/microduck/assets/." "$ROBOT/microduck/assets/"
cp -f  "$HERE/model/robot/microduck/add_backlash.py" "$ROBOT/microduck/"
rm -rf "$ROBOT/microduck_hd1910_base"
cp -r "$HERE/model/robot/microduck_hd1910_base" "$ROBOT/"
echo "  网格资产 + add_backlash.py 就位；官方 XML 保持仓库版本不被覆盖"
echo "  HD-1910 基准 806 g 就位"

# ---------- 3. 安装依赖 ----------
echo "[3/6] uv sync（首次约 2 GB CUDA wheels）..."
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
uv sync || fail "uv sync 失败。常见原因：PyPI 不通（设 UV_INDEX_URL 镜像）或磁盘不足（≥12 GB）"
# 装残了（uv sync 被中断 / 磁盘满）会留下一个「metadata 在、文件缺」的 venv：
# uv 认为装好了，直到 import 时报 No module named 'torch._strobelight'。
# 这里主动验一次导入，把错误挡在第 3 步，而不是等第 6 步注册任务时才炸。
uv run python -c "import mjlab, rsl_rl, torch, mujoco, bam" \
  || fail "依赖导入失败（venv 可能装残了）：rm -rf $WORK_DIR/.venv 后重跑 bash setup.sh"
uv run python -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# ---------- 4. 拷入定制文件 ----------
echo "[4/6] 拷入 HD-1910 定制文件 ..."
mkdir -p "$ROBOT/microduck_hd1910" training/hd1910
cp "$HERE/model/hd1910/hd1910_constants.py" "$ROBOT/"
# 必须落成 tasks/hd1910.py：tasks/__init__.py 里 import 的名字是 hd1910
cp "$HERE/model/hd1910/hd1910_tasks.py"     src/mjlab_microduck/tasks/hd1910.py
cp "$HERE/model/hd1910/hd1910.py"           src/mjlab_microduck/actuator/ 2>/dev/null || true
# 回放 / 官方入口封装 / 3D 关节查看器（在引擎根目录下用 uv run 调用）
cp "$HERE/model/hd1910/replay_onnx_bam.py"     training/hd1910/
cp "$HERE/model/hd1910/infer_policy_hd1910.py" training/hd1910/
cp "$HERE/model/hd1910/joint_viewer.py"        training/hd1910/
cp "$HERE/model/hd1910/record_compare.sh"      training/hd1910/ 2>/dev/null || true

if ! grep -q "from mjlab_microduck.tasks import hd1910" src/mjlab_microduck/tasks/__init__.py 2>/dev/null; then
  echo 'from mjlab_microduck.tasks import hd1910  # noqa: F401' >> src/mjlab_microduck/tasks/__init__.py
fi
echo "  定制文件就位，任务已注册"

# ---------- 5. 生成 834 g 资产 ----------
echo "[5/6] 生成 834 g 资产（806 g 基准 → 834 g）..."
python3 "$HERE/model/hd1910/make_models.py" \
    "$ROBOT/microduck_hd1910_base" \
    "$ROBOT/microduck_hd1910"
# make_models 只处理已存在的 XML；backlash 变体用官方 add_backlash.py 注入 2°
OUT="$ROBOT/microduck_hd1910"
for f in robot_walk robot_allcollisions; do
  cp "$OUT/$f.xml" "$OUT/${f}_backlash.xml"
  uv run python "$ROBOT/microduck/add_backlash.py" "$OUT/${f}_backlash.xml" --backlash-deg 2.0 >/dev/null
done
ls -1 "$OUT"/*.xml | sed 's/^/    /'

# ---------- 6. 自检 ----------
echo "[6/6] 自检 ..."
uv run python - "$OUT" <<'PY'
import sys, mujoco
from mjlab.tasks.registry import list_tasks

out = sys.argv[1]
m = mujoco.MjModel.from_xml_path(f"{out}/robot_walk.xml")
mass = sum(m.body_mass) * 1000
assert abs(mass - 834.0) < 0.05, f"总质量 {mass:.1f} g ≠ 834.0 g"
hd = [t for t in list_tasks() if "HD1910" in t]
assert hd, "没有注册到 HD1910 任务（检查 hd1910_constants/tasks 是否就位）"
print(f"  834 g 资产 OK（{mass:.1f} g，{m.nu} 个执行器）")
print(f"  HD-1910 任务 {len(hd)} 个，例如：{sorted(hd)[0]}")
PY

# 记下引擎路径：train_deploy_set.sh 优先读它，这样用自定义路径装的也能直接跑
echo "$WORK_DIR" > "$HERE/.engine_path"

echo
echo "============================================================"
echo "  完成！接下来："
echo
echo "    cd $WORK_DIR"
echo
echo "    # ① 看舵机旋转角度（3D 建模查看器，可交互）"
echo "    uv run python training/hd1910/joint_viewer.py"
echo
echo "    # ② 250 轮冒烟训练（验证环境真的能训起来）"
echo "    bash $HERE/smoke_train.sh"
echo
echo "    # ③ 正式训练（任务名以 list-envs 为准）"
echo "    uv run list-envs | grep HD1910"
echo "    uv run train <任务名> --env.scene.num-envs 2048"
echo "============================================================"
