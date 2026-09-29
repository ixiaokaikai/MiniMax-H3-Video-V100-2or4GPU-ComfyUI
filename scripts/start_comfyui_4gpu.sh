#!/usr/bin/env bash
# ============================================================
#  ComfyUI 启动 — 四卡模式 (全部 4 张卡, 最快)
#  工作流: workflows/H3-全能版本_四卡Ray版_v1.json
# ============================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$ROOT/.venv/bin/python"
MAIN="$ROOT/ComfyUI/main.py"
LOG="$ROOT/comfyui.log"
URL="http://127.0.0.1:8188"

# 四卡: 全部可见。外部可覆盖, 如 export CUDA_VISIBLE_DEVICES=2,3,4,5
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# ⚠️ 注意: 若你的机器上还插着别的小卡 (如亮机卡), 上面的编号是"按速度排序"的前 4 张,
#    不是 nvidia-smi 的物理编号。拿不准就先 nvidia-smi 看一眼再调整。

[ -x "$PY" ] || { echo "❌ 没找到虚拟环境, 请先运行 bash install/install.sh"; exit 1; }

if ss -ltn 2>/dev/null | grep -q ":8188 "; then
  echo "ComfyUI 已在运行 (8188 端口被占用)。直接打开 $URL"
  exit 0
fi

# --disable-smart-memory: 跑完不缓存模型, 防止空闲时霸占内存 (低内存机器每轮多花 20~70 秒重新装载)
# --fp16-unet: V100 没有 bf16 硬件, 主模型走 fp16; 捆绑的 minimax_h3_fp16_fix.py 负责数值精度
# --fp16-vae:  fp16 解码比 fp32 快约 4 倍 (≤1K 安全); 2K 以上大图要最稳就 export VAE_MODE=fp32
VAE_MODE="${VAE_MODE:-fp16}"

echo "启动 ComfyUI (四卡, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)..."
nohup "$PY" "$MAIN" --listen 0.0.0.0 --port 8188 --disable-smart-memory --fp16-unet --${VAE_MODE}-vae > "$LOG" 2>&1 &
PID=$!
echo "  PID: $PID   日志: $LOG"
echo "等待就绪 (首次加载节点约 30~60 秒)..."

for i in $(seq 1 45); do
  sleep 2
  if grep -q "Starting server" "$LOG" 2>/dev/null; then
    echo
    echo "✅ ComfyUI 已就绪"
    echo "   本机:   $URL"
    echo "   局域网: http://<本机IP>:8188"
    echo
    echo "   加载 workflows/H3-全能版本_四卡Ray版_v1.json 即可出片。"
    echo "   模型只在点「生成」时才载入显存, 启动阶段不占显存。"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "❌ 启动失败, 最近日志:"
    tail -25 "$LOG"
    exit 1
  fi
  printf "."
done
echo
echo "⏰ 90 秒未就绪, 最近日志:"
tail -25 "$LOG"
exit 1
