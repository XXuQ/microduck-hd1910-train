#!/usr/bin/env bash
# =============================================================================
# 录制 834 / 806 行走对比视频（不经过 WSLg 实时窗口，播放最顺）
#
# 用法（在引擎根目录 microduck_rl/ 下执行）：
#   POLICY=/abs/path/walk.onnx bash <repo>/training/hd1910/record_compare.sh
#   POLICY=walk.onnx bash .../record_compare.sh 0.2 8 side 0.2
#        参数：1=速度(m/s)  2=时长(秒)  3=机位(follow|side|front|top)  4=偏航补偿
#
#   # 也可以把 onnx 作为第一个参数直接给
#   bash .../record_compare.sh walk.onnx 0.2 8 side 0.2
#
# 默认机位 follow（追尾斜视），默认偏航补偿 +0.2（抵消策略固有右漂，走直）
# 输出目录：$PWD/render_out（可用 OUTDIR= 覆盖）
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:-$PWD}"

# 参数 @1 可能是 onnx 路径，也可能是速度
POLICY="${POLICY:-}"
if [ -z "$POLICY" ] && [ -n "${1:-}" ] && [[ "${1}" == *.onnx ]]; then
    POLICY="$1"; shift
fi
if [ -z "$POLICY" ]; then
    cat <<'USAGE'
缺少策略文件。请任选一种方式给出：
  POLICY=/abs/path/walk.onnx bash record_compare.sh 0.2 8
  bash record_compare.sh /abs/path/walk.onnx 0.2 8
USAGE
    exit 1
fi
[ -f "$POLICY" ] || { echo "找不到策略文件: $POLICY"; exit 1; }
POLICY="$(cd "$(dirname "$POLICY")" && pwd)/$(basename "$POLICY")"

VX="${1:-0.20}"
DUR="${2:-8}"
CAM="${3:-follow}"
YAW="${4:-0.20}"

[ -f "$ENGINE/src/mjlab_microduck/robot/microduck_hd1910/scene_walk.xml" ] || {
    echo "当前目录不是引擎根目录：$ENGINE"
    echo "请 cd 到 microduck_rl/ 再运行（或设 ENGINE=/path/to/microduck_rl）"
    exit 1
}
cd "$ENGINE"

S834=src/mjlab_microduck/robot/microduck_hd1910/scene_walk.xml
S806=src/mjlab_microduck/robot/microduck_hd1910_base/scene_walk.xml
OUTDIR="${OUTDIR:-$PWD/render_out}"
TMP="${TMPDIR:-/tmp}/hd1910_rec"
mkdir -p "$OUTDIR" "$TMP"

rec() {
    local tag="$1" scene="$2"
    echo "===== 录制 $tag ====="
    uv run python -u training/hd1910/replay_onnx_bam.py \
        --policy "$POLICY" --scene "$scene" \
        --vx "$VX" --vyaw "$YAW" --duration "$DUR" \
        --record "$TMP/$tag.mp4" --record-cam "$CAM" \
        2>&1 | grep -E "录制中|已保存|前进|vx 稳态|roll RMS|摔倒"
    cp "$TMP/$tag.mp4" "$OUTDIR/$tag.mp4"
    echo
}

rec walk834 "$S834"
rec walk806 "$S806"

echo "输出目录: $OUTDIR"
ls -lh "$OUTDIR"/walk83*.mp4 2>/dev/null

# 仅在 WSL 里尝试用 Windows 资源管理器打开
if command -v explorer.exe >/dev/null 2>&1 && command -v wslpath >/dev/null 2>&1; then
    WINPATH=$(wslpath -w "$OUTDIR" 2>/dev/null || true)
    if [ -n "$WINPATH" ]; then
        echo "用 Windows 资源管理器打开: $WINPATH"
        timeout 8 explorer.exe "$WINPATH" >/dev/null 2>&1 || true
    fi
fi
echo "完成。"
