#!/usr/bin/env bash
# ============================================================
#  ComfyUI 启动 — 四卡模式 (4 张卡一起算, 最快)
#  工作流: workflows/H3-全能版本_四卡Ray版_v1.json
#
#  选卡: 默认自动挑 4 张可用卡 (显存 ≥14G 且算力 ≥7.0, 自动跳过亮机卡/老架构卡),
#        想指定就自己 export CUDA_VISIBLE_DEVICES=2,3,4,5
# ============================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$ROOT/.venv/bin/python"
MAIN="$ROOT/ComfyUI/main.py"
LOG="$ROOT/comfyui.log"
URL="http://127.0.0.1:8188"
WANT=4

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
    echo "❌ 只检测到 ${#picked[@]} 张可用 GPU (显存 ≥14G 且算力 ≥7.0; 四卡模式需要 $want 张)。" >&2
    echo "   双卡 / 卡数不够: 用 bash scripts/start_comfyui_2gpu.sh" >&2
    echo "   想手动指定: export CUDA_VISIBLE_DEVICES=<卡号,逗号分隔>" >&2
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
NGPU=$(awk -F, '{print NF}' <<<"$SELECTED")
[ "$NGPU" -eq "$WANT" ] || echo "⚠️  你指定了 $NGPU 张卡, 四卡版工作流按 4 张卡分片 (CUDA_VISIBLE_DEVICES 尽量给 4 张)"

if ss -ltn 2>/dev/null | grep -q ":8188 "; then
  echo "ComfyUI 已在运行 (8188 端口被占用)。直接打开 $URL"
  exit 0
fi

# 显存占用提示 (四卡模式期望独占这几张卡; 占用高不拦, 只提醒)
for D in ${SELECTED//,/ }; do
  read -r TOT FREE <<<"$(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits | awk -F', *' -v i="$D" '$1==i {print $2, $3}')"
  if [ -n "${TOT:-}" ] && [ "${TOT:-0}" -gt 0 ]; then PCT=$(( FREE * 100 / TOT )); else PCT=0; fi
  [ "$PCT" -lt 80 ] && echo "⚠️  GPU $D 只剩 ${PCT}% 空闲, 有别的东西在占显存 (四卡模式建议独占)"
done

# --disable-smart-memory: 跑完不缓存模型, 防止空闲时霸占内存 (低内存机器每轮多花 20~70 秒重新装载)
# --fp16-unet: V100 没有 bf16 硬件, 主模型走 fp16; 捆绑的 minimax_h3_fp16_fix.py 负责数值精度
# --fp16-vae:  fp16 解码比 fp32 快约 4 倍 (≤1K 安全); 2K 以上大图要最稳就 export VAE_MODE=fp32
VAE_MODE="${VAE_MODE:-fp16}"

# 监听地址: 默认 0.0.0.0 (方便从别的机器用浏览器); 只本机用就 export H3_LISTEN=127.0.0.1
LISTEN_ADDR="${H3_LISTEN:-0.0.0.0}"
echo "启动 ComfyUI (四卡, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, 监听 $LISTEN_ADDR)..."
if [ "$LISTEN_ADDR" = "0.0.0.0" ]; then
  echo "⚠️  ComfyUI 没有任何登录鉴权: 监听 0.0.0.0 时同一内网里任何人都能打开界面、提交任务(并借此在你机器上执行节点)。"
  echo "    只在可信内网使用, 不要做端口转发暴露到公网; 只想本机访问: H3_LISTEN=127.0.0.1 bash scripts/start_comfyui_4gpu.sh"
fi
nohup "$PY" "$MAIN" --listen "$LISTEN_ADDR" --port 8188 --disable-smart-memory --fp16-unet --${VAE_MODE}-vae > "$LOG" 2>&1 &
PID=$!
echo "  PID: $PID   日志: $LOG"
echo "等待就绪 (首次加载节点约 30~60 秒)..."

for i in $(seq 1 45); do
  sleep 2
  if grep -q "Starting server" "$LOG" 2>/dev/null; then
    echo
    echo "✅ ComfyUI 已就绪"
    echo "   本机:   $URL"
    [ "$LISTEN_ADDR" = "0.0.0.0" ] && echo "   局域网: http://<本机IP>:8188"
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
