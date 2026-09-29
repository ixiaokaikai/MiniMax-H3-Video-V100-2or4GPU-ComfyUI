#!/usr/bin/env bash
# ============================================================
#  节点自检 — 核对两份工作流用到的 40 种节点类型是否都能加载
#
#  用法:
#    bash scripts/check_nodes.sh                    # ComfyUI 没在跑 → 自己起一个临时实例(端口自动挑空闲的)测完就关
#    bash scripts/check_nodes.sh --url http://127.0.0.1:8188   # 测已运行的实例(如果你本来就开着 ComfyUI)
#
#  为什么需要: 节点包装不上时 ComfyUI 不会报错退出, 只在网页上显示
#  "missing node type", 工作流打开一片红。这个脚本把问题在命令行就暴露出来。
# ============================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$ROOT/.venv/bin/python"
MAIN="$ROOT/ComfyUI/main.py"
TMP_LOG="/tmp/h3_check_nodes.log"
PORT=""
URL=""
STARTED=0

while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

# 工作流里用到的全部节点类型 (缺任何一个, 打开工作流都会有红框)
# 用数组而不是空格分隔字符串: 有的类型名自带空格 (如 "Video Slice")
REQUIRED=(
  CLIPLoaderDisTorch2MultiGPU ComfyMathExpression ComfySwitchNode CreateVideo GetVideoComponents
  ImageScale ImageScaleToTotalPixels ImageUpscaleWithModel KSamplerSelect LTXVConcatAVLatent LTXVSeparateAVLatent
  LoadAudio LoadImage LoadVideo MiniMaxH3DualClockSamplerT8 MiniMaxH3ReferenceToVideo
  MiniMaxH3ScheduledSolAttentionPatch MinimaxH3LatentUpscalerNode3D PrimitiveBoolean PrimitiveFloat PrimitiveInt
  PrimitiveStringMultiline RandomNoise RayBasicGuider RayBasicScheduler RayInitializer RayLoraLoader
  RayMiniMaxH3SigmaShift RayUNETLoader ResolutionSelector SaveVideo SolAttnV100 SplitSigmas TESpeedMiniMaxH3
  UpscaleModelLoader VAEDecode VAEDecodeAudio VAELoader "Video Slice" XFuserSamplerCustomAdvanced
)

[ -x "$PY" ] || { echo "❌ 没找到虚拟环境 ($PY)，请先运行 bash install/install.sh"; exit 1; }
PYBIN="$PY"

cleanup() {
  if [ "$STARTED" -eq 1 ] && [ -n "$PORT" ]; then
    pkill -f "main.py --listen 127.0.0.1 --port $PORT" 2>/dev/null || true
    sleep 2
    pkill -9 -f "main.py --listen 127.0.0.1 --port $PORT" 2>/dev/null || true
    pkill -f "ray::RayWorker" 2>/dev/null || true
  fi
}
trap cleanup EXIT

fetch_object_info() {   # $1=base url → 打印节点表 JSON 到 stdout (本机直连)
  "$PYBIN" - "$1" <<'PY'
import sys, urllib.request
base = sys.argv[1].rstrip("/")
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 本机服务直连, 不受环境网络设置影响
for path in ("/object_info", "/api/object_info"):
    try:
        with opener.open(base + path, timeout=30) as r:
            if r.status == 200:
                sys.stdout.write(r.read().decode("utf-8", "replace"))
                sys.exit(0)
    except Exception:
        continue
sys.exit(1)
PY
}

if [ -z "$URL" ]; then
  [ -f "$MAIN" ] || { echo "❌ 没找到 $MAIN，请先运行 bash install/install.sh"; exit 1; }
  # 动态挑一个空闲端口 (硬编码端口可能被别的服务占着, 拿到别的服务的响应会误判)
  if [ -z "$PORT" ]; then
    PORT=$("$PYBIN" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
  fi
  URL="http://127.0.0.1:$PORT"
  echo "临时启动 ComfyUI (端口 $PORT) 做节点自检..."
  ( cd "$ROOT/ComfyUI" && nohup "$PY" main.py --listen 127.0.0.1 --port "$PORT" > "$TMP_LOG" 2>&1 & )
  STARTED=1
  for i in $(seq 1 60); do
    sleep 2
    if fetch_object_info "$URL" >/dev/null 2>&1; then break; fi
    if grep -q "already in use" "$TMP_LOG" 2>/dev/null; then
      echo "❌ 端口 $PORT 被占用"; tail -5 "$TMP_LOG"; exit 1
    fi
    printf "."
  done
  echo
fi

INFO_FILE="$(mktemp)"
if ! fetch_object_info "$URL" > "$INFO_FILE" 2>/dev/null || [ ! -s "$INFO_FILE" ]; then
  echo "❌ 拿不到 $URL/object_info — ComfyUI 没起来或端口不对。日志: $TMP_LOG"
  [ "$STARTED" -eq 1 ] && tail -20 "$TMP_LOG"
  rm -f "$INFO_FILE"; exit 1
fi

# 先确认拿到的是 ComfyUI 的节点表 (端口被别的服务占用时会返回一个合法但无关的 JSON)
if ! "$PYBIN" -c '
import json,sys
d=json.load(open(sys.argv[1]))
ok = isinstance(d, dict) and len(d) > 200 and "LoadImage" in d and "VAELoader" in d
sys.exit(0 if ok else 1)
' "$INFO_FILE" 2>/dev/null; then
  echo "❌ $URL 的响应不是 ComfyUI 的节点表 (节点数/关键节点对不上)。可能是端口被别的服务占用, 请用 --port 换一个。"
  rm -f "$INFO_FILE"; exit 1
fi

TOTAL=${#REQUIRED[@]}
TOTAL_NODES=$("$PYBIN" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$INFO_FILE")
MISSING=$("$PYBIN" -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(" ".join(n for n in sys.argv[2:] if n not in d))
' "$INFO_FILE" "${REQUIRED[@]}" 2>/dev/null)
rm -f "$INFO_FILE"

if [ -z "$MISSING" ]; then
  echo "✅ 节点自检通过: $TOTAL/$TOTAL 种节点全部可用 (实例共加载 $TOTAL_NODES 种节点)"
  echo "   打开网页 → Workflow → Open, 选对应卡数的工作流出片即可。"
  exit 0
fi

echo "❌ 缺少节点类型: $MISSING"
echo "   看 $ROOT/comfyui.log (或 $TMP_LOG) 里对应节点包的 import 报错。"
echo "   常见原因: Python 版本不是 3.12 / raylight 没装成 (pip install -e custom_nodes/raylight)。"
exit 1
