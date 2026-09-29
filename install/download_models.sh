#!/usr/bin/env bash
# ============================================================
#  模型一键下载 (断点续传)
#  模型不进仓库，全部从官方 HF/GitHub 链接下载
#
#  用法:
#    bash install/download_models.sh             # 必需 5 个 (约 44G)
#    bash install/download_models.sh --optional  # 再加 3 个可选 (约 26G)
#
#  网络环境连不上 huggingface.co 时 (如国内网络):
#    HF_ENDPOINT=https://hf-mirror.com bash install/download_models.sh
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMFY="$ROOT/ComfyUI"
HF="${HF_ENDPOINT:-https://huggingface.co}"

WITH_OPTIONAL=0
[ "${1:-}" = "--optional" ] && WITH_OPTIONAL=1

[ -d "$COMFY" ] || { echo "❌ 先运行 bash install/install.sh"; exit 1; }

# 优先 aria2c 16 连接，否则 wget 单连接
if command -v aria2c >/dev/null; then
  dl() { aria2c -x16 -s16 -c --allow-overwrite=true --auto-file-renaming=false -o "$2" "$1"; }
  echo "下载工具: aria2c (16 连接)"
else
  command -v wget >/dev/null || { echo "❌ 需要 aria2 或 wget (sudo apt install aria2 或 wget)"; exit 1; }
  dl() { wget -c -O "$2" "$1"; }
  echo "下载工具: wget (单连接；装 aria2 可大幅提速: sudo apt install aria2)"
fi

fetch() { # $1=URL  $2=目标文件  $3=说明
  local dir; dir="$(dirname "$2")"
  mkdir -p "$dir"
  if [ -f "$2" ]; then echo "  ⏭  已存在，跳过: $(basename "$2")"; return 0; fi
  echo "  ↓  $(basename "$2")  ($3)"
  dl "$1" "$2"
}

echo "============================================"
echo "  必需模型 (5 个, 约 44G)"
echo "============================================"
fetch "$HF/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
      "$COMFY/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
      "21G 主模型 int8"
fetch "$HF/Merserk/MiniMax-H3-INT4-ConvRot/resolve/main/qwen3vl_32b_minimax_h3_int4_convrot.safetensors" \
      "$COMFY/models/text_encoders/qwen3vl_32b_minimax_h3_int4_convrot.safetensors" \
      "15G 文本编码器 int4 (默认)"
fetch "$HF/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" \
      "$COMFY/models/vae/minimax_h3_video_vae_fp16.safetensors" \
      "5.2G 视频 VAE"
fetch "$HF/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors" \
      "$COMFY/models/vae/minimax_h3_audio_vae_fp32.safetensors" \
      "0.6G 音频 VAE"
fetch "$HF/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" \
      "$COMFY/models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" \
      "2G 8步 turbo LoRA (加速开关)"

if [ "$WITH_OPTIONAL" -eq 1 ]; then
  echo
  echo "============================================"
  echo "  可选模型 (3 个, 约 26G)"
  echo "============================================"
  fetch "$HF/linjian257/qwen3vl_32b_minimax_h3_int8_convrot_uncensored-by-linjian257/resolve/main/qwen3vl_32b_minimax_h3_int8_convrot_uncensored-by-linjian257.safetensors" \
        "$COMFY/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot_uncensored-by-linjian257.safetensors" \
        "25G int8 编码器 (「文本编码器」开关切 true 才用到)"
  fetch "$HF/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors" \
        "$COMFY/models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors" \
        "0.7G 潜空间上采样 (latent 放大支路)"
  fetch "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
        "$COMFY/models/upscale_models/RealESRGAN_x4plus.pth" \
        "64M ESRGAN 超分 (图像空间放大支路)"
fi

echo
echo "✅ 下载完成。模型现状:"
find "$COMFY/models" -type f \( -name '*.safetensors' -o -name '*.pth' \) -exec du -h {} \; 2>/dev/null
echo
echo "下一步: bash scripts/start_comfyui_4gpu.sh (四卡) 或 scripts/start_comfyui_2gpu.sh (两卡)"
