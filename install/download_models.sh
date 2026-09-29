#!/usr/bin/env bash
# ============================================================
#  模型一键下载 (断点续传)
#  模型不进仓库，全部从官方 HF/GitHub 链接下载
#
#  用法:
#    bash install/download_models.sh             # 必需 5 个 (约 44G)
#    bash install/download_models.sh --optional  # 再加 2 个可选 (约 0.8G)
#    bash install/download_models.sh --verify    # 只核对已下载文件的 sha256 (不下载)
#
#  下载不动 / 太慢时可换国内镜像源(可选):
#    HF_ENDPOINT=https://hf-mirror.com bash install/download_models.sh
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMFY="$ROOT/ComfyUI"
HF="${HF_ENDPOINT:-https://huggingface.co}"

WITH_OPTIONAL=0
MODE="download"
for a in "$@"; do
  case "$a" in
    --optional) WITH_OPTIONAL=1 ;;
    --verify)   MODE="verify" ;;
    *) echo "❌ 未知参数: $a"; exit 1 ;;
  esac
done

MANIFEST="$SCRIPT_DIR/model_checksums.txt"

size_of() { # 兼容 GNU/BSD stat
  stat -Lc %s "$1" 2>/dev/null || stat -f %z "$1"
}
expect_of() { # $1=相对 ComfyUI/models 的路径 -> "sha256 字节数"
  awk -v p="$1" '$1 ~ /^[0-9a-f]{64}$/ && $3 == p {print $1" "$2; exit}' "$MANIFEST"
}
verify_sha256() { # $1=文件; 返回 0 通过 / 1 失败(缺清单则跳过)
  local f="$1" rel="${1#"$COMFY/models/"}" spec h sz real
  spec="$(expect_of "$rel")"
  [ -n "$spec" ] || { echo "  ⚠️  清单里没有 $(basename "$f"), 跳过校验"; return 0; }
  h="${spec% *}"; sz="${spec#* }"
  real="$(size_of "$f")"
  [ "$real" = "$sz" ] || { echo "  ❌ 大小不符: 实际 $real, 期望 $sz (下载不完整?)"; return 1; }
  echo "  🔒 核对 $(basename "$f") 的 sha256 ($(du -h "$f" | cut -f1))..."
  if [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$h" ]; then
    echo "  ✅ sha256 一致"
    return 0
  fi
  echo "  ❌ sha256 不一致: 文件损坏或被替换"
  return 1
}

[ -d "$COMFY" ] || { echo "❌ 先运行 bash install/install.sh"; exit 1; }

if [ "$MODE" = "verify" ]; then
  echo "=============================================="
  echo "  核对已下载模型的 sha256 (清单: install/model_checksums.txt)"
  echo "=============================================="
  [ -f "$MANIFEST" ] || { echo "❌ 找不到清单 $MANIFEST"; exit 1; }
  bad=0; miss=0
  while read -r h sz rel; do
    [ -n "${h:-}" ] || continue
    case "$h" in \#*) continue ;; esac
    f="$COMFY/models/$rel"
    if [ ! -f "$f" ]; then
      echo "  ⚪ 未下载: $rel"
      miss=$((miss + 1))
      continue
    fi
    if [ "$(size_of "$f")" != "$sz" ]; then
      echo "  ❌ 大小不符: $rel (实际 $(size_of "$f"), 期望 $sz)"
      bad=$((bad + 1))
      continue
    fi
    if [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$h" ]; then
      echo "  ✅ $rel"
    else
      echo "  ❌ sha256 不一致: $rel"
      bad=$((bad + 1))
    fi
  done < "$MANIFEST"
  echo
  if [ "$bad" -eq 0 ]; then
    echo "✅ 已下载的模型全部通过校验 (未下载 $miss 个)"
    exit 0
  fi
  echo "❌ 有 $bad 个文件校验失败: 删掉它们再运行下载脚本即可重下"
  exit 1
fi

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
  if [ -f "$2" ]; then
    echo "  ⏭  已存在，跳过: $(basename "$2") ($(du -h "$2" | cut -f1))"
    local rel="${2#"$COMFY/models/"}" spec
    spec="$(expect_of "$rel")"
    if [ -n "$spec" ] && [ "$(size_of "$2")" != "${spec#* }" ]; then
      echo "     ❌ 大小与清单不符 → 判定为下载不完整, 已删除, 请重新运行本脚本"
      rm -f "$2"; exit 1
    fi
    return 0
  fi
  echo "  ↓  $(basename "$2")  ($3)"
  dl "$1" "$2"
  verify_sha256 "$2" || { echo "     → 已删除该文件, 请重新运行本脚本 (会重新下载)"; rm -f "$2"; exit 1; }
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
  echo "  可选模型 (2 个, 约 0.8G)"
  echo "============================================"
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
