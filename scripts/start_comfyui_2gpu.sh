#!/usr/bin/env bash
# ============================================================
#  ComfyUI 启动 — 两卡模式 (只用 2 张卡, 与别的常驻 GPU 服务共存)
#  工作流: workflows/H3-全能版本_2卡Ray版_v1.json
#
#  特点:
#   - 只暴露 2 张 GPU, 其余卡对 ComfyUI 及其 Ray worker 物理不可见
#   - H3_VRAM_AUTORELEASE=1: 每次出片后自动杀 Ray worker 释放显存,
#     代价是下一单全冷启动 (多约 2 分钟), 但显存不常驻
#   - FSDP_CPU_OFFLOAD 已在工作流里打开: 2卡分片 10G/卡 + 编码器, 不开必 OOM
#     → 系统内存需 ~30G, 建议 ≥64G
#
#  选卡: 默认自动挑 2 张可用卡 (显存 ≥14G 且算力 ≥7.0, 自动跳过亮机卡/老架构卡),
#        想指定就自己 export CUDA_VISIBLE_DEVICES=2,3
# ============================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$ROOT/.venv/bin/python"
MAIN="$ROOT/ComfyUI/main.py"
LOG="$ROOT/comfyui.log"
URL="http://127.0.0.1:8188"
WANT=2

[ -x "$PY" ] || { echo "❌ 没找到虚拟环境, 请先运行 bash install/install.sh"; exit 1; }

# ---------- 选卡 ----------
pick_gpus() {   # $1=需要几张; 输出逗号分隔的物理卡号(PCI 顺序)到 stdout
  local want="$1" picked=() small="" idx tot cc
  # 同时按显存(>=14G)和算力(>=7.0)筛: 亮机卡/老架构卡 torch cu128 跑不了(sm_61 直接 no kernel image)
  while IFS=',' read -r idx tot cc; do
    tot="${tot// /}"; cc="${cc// /}"
    if [ "${tot:-0}" -ge 14000 ] && [ "${cc:-0}" != "0" ] && awk -v c="${cc:-0}" 'BEGIN{exit !(c >= 7.0)}'; then
      picked+=("$idx")
    else
      small+="GPU${idx}(${tot}M,sm${cc}) "
    fi
  done < <(nvidia-smi --query-gpu=index,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null)
  if [ "${#picked[@]}" -lt "$want" ]; then
    echo "❌ 只检测到 ${#picked[@]} 张可用 GPU (显存 ≥14G 且算力 ≥7.0; 两卡模式需要 $want 张)。" >&2
    echo "   本项目面向 2卡/4卡 V100-16G。" >&2
    [ -n "$small" ] && echo "   (已跳过小显存卡: $small)" >&2
    exit 1
  fi
  local out; out=$(IFS=,; echo "${picked[*]:0:$want}")
  echo "  选卡: $out  (跳过的小卡: ${small:-无})" >&2
  echo "$out"
}

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  SELECTED="$CUDA_VISIBLE_DEVICES"
  echo "使用你指定的 CUDA_VISIBLE_DEVICES=$SELECTED"
else
  SELECTED="$(pick_gpus "$WANT")" || exit 1
fi
# CUDA 默认按 FASTEST_FIRST 排号, 与 nvidia-smi 的 PCI 顺序不一致 → 钉成 PCI 顺序,
# 这样"卡号"在 nvidia-smi / CUDA / Ray worker 三处含义一致 (插了亮机卡的机器尤其关键)
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="$SELECTED"
export H3_VRAM_AUTORELEASE=1

if ss -ltn 2>/dev/null | grep -q ":8188 "; then
  echo "ComfyUI 已在运行 (不会重复启动, 也不会切换卡数)。"
  echo "⚠️  如果当前跑的是四卡模式, 请先 bash scripts/stop_comfyui.sh 再运行本脚本。"
  exit 0
fi

# 预检: 这 2 张卡必须基本空闲 (防别的常驻服务没停干净; 两卡模式的卖点就是与别的东西共存)
for D in ${SELECTED//,/ }; do
  read -r TOT FREE <<<"$(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits | awk -F', *' -v i="$D" '$1==i {print $2, $3}')"
  if [ -n "${TOT:-}" ] && [ "${TOT:-0}" -gt 0 ]; then
    PCT=$(( FREE * 100 / TOT ))
  else
    PCT=0
  fi
  if [ "$PCT" -lt 85 ]; then
    echo "❌ GPU $D 只剩 ${PCT}% 空闲, 应该是一张基本空着的卡。先 nvidia-smi 看一眼, 本脚本不启动。"
    exit 1
  fi
  echo "✅ GPU $D 空闲 ${PCT}%"
done

VAE_MODE="${VAE_MODE:-fp16}"

echo
echo "启动 ComfyUI (两卡, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)..."
nohup "$PY" "$MAIN" --listen 0.0.0.0 --port 8188 --disable-smart-memory --fp16-unet --${VAE_MODE}-vae > "$LOG" 2>&1 &
PID=$!
echo "  PID: $PID   日志: $LOG"
echo "等待就绪 (首次加载节点约 30~60 秒)..."

for i in $(seq 1 45); do
  sleep 2
  if grep -q "Starting server" "$LOG" 2>/dev/null; then
    echo
    echo "✅ ComfyUI 已就绪 (两卡模式)"
    echo "   本机:   $URL"
    echo "   局域网: http://<本机IP>:8188"
    echo
    echo "   加载 workflows/H3-全能版本_2卡Ray版_v1.json 即可出片。"
    echo "   注意: 两卡版每单约 7 分钟 (5s/720p), 跑完自动释放显存。"
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
