#!/usr/bin/env bash
# ============================================================
#  MiniMax H3 2/4卡 V100 ComfyUI 环境 — 一键安装
#  用法: bash install/install.sh [--skip-verify]
#
#  网络慢/被墙时可设环境变量(都可选):
#    GITHUB_MIRROR=https://ghproxy.net/   bash install/install.sh   # ComfyUI 走 GitHub 加速前缀
#    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ ...      # pip 走国内源
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMFY="$ROOT/ComfyUI"
VENV="$ROOT/.venv"
COMFY_REV="73c9bad4d21e7addbe1d13bc92eee0f1431b017d"   # ComfyUI v0.37.0 (本项目实测版本)
GITHUB="${GITHUB_MIRROR:-https://github.com}"

SKIP_VERIFY=0
[ "${1:-}" = "--skip-verify" ] && SKIP_VERIFY=1

log()  { echo -e "\033[1;32m[install]\033[0m $*"; }
die()  { echo -e "\033[1;31m[install] ❌ $*\033[0m" >&2; exit 1; }

# ---------- 0. 基础工具 ----------
for t in git patch; do
  command -v "$t" >/dev/null || die "缺少工具 $t (Ubuntu: sudo apt install $t)"
done
command -v nvidia-smi >/dev/null || die "找不到 nvidia-smi，请先安装 NVIDIA 驱动 (≥525.60.13, 建议装最新版)"

# ---------- 1. GPU 检查 ----------
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
[ "$GPU_COUNT" -ge 2 ] || die "需要 ≥2 张 NVIDIA GPU (当前检测到 $GPU_COUNT 张)。本项目面向 2卡/4卡 V100-16G"
echo "检测到 $GPU_COUNT 张 NVIDIA GPU:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
if nvidia-smi nvlink -s 2>/dev/null | grep -q "Link"; then
  log "检测到 NVLink — 多卡并行的推荐配置"
else
  echo "⚠️  未检测到 NVLink：PCIe 也能跑，但多卡并行会明显变慢，推荐 SXM+NVLink 卡"
fi

# ---------- 2. 内存 / 磁盘检查 ----------
RAM_GB=$(free -g 2>/dev/null | awk '/^Mem/ {print $2}')
if [ -n "${RAM_GB:-}" ]; then
  if [ "$RAM_GB" -lt 32 ]; then
    die "系统内存 ${RAM_GB}G 不足 (两卡模式 FSDP 分片约吃 30G, 建议 ≥64G)"
  elif [ "$RAM_GB" -lt 48 ]; then
    echo "⚠️  系统内存 ${RAM_GB}G 偏少: 建议跑四卡模式 (内存压力小), 两卡模式可能吃紧"
  fi
fi
DISK_AVAIL_KB=$(df -k "$ROOT" | awk 'NR==2 {print $4}')
[ "${DISK_AVAIL_KB:-0}" -gt 10485760 ] || die "磁盘可用空间不足 10G (环境约需 8G: torch+venv)"

# ---------- 3. Python 3.12 ----------
command -v python3.12 >/dev/null || die "需要 Python 3.12（部分节点包预编译扩展仅支持 cp312）。Ubuntu: sudo apt install python3.12 python3.12-venv"

# ---------- 4. ComfyUI 核心 (钉死本项目实测版本) ----------
if [ -d "$COMFY/.git" ]; then
  log "ComfyUI 已存在 ($COMFY)，跳过克隆"
else
  log "克隆 ComfyUI @ v0.37.0 (源: $GITHUB)..."
  mkdir -p "$COMFY"
  git init -q "$COMFY"
  git -C "$COMFY" remote add origin "$GITHUB/comfyanonymous/ComfyUI.git"
  git -C "$COMFY" fetch -q --depth 1 origin "$COMFY_REV"
  git -C "$COMFY" checkout -q FETCH_HEAD
fi

# ---------- 5. venv + 依赖 ----------
if [ ! -x "$VENV/bin/python" ]; then
  log "创建虚拟环境 $VENV ..."
  python3.12 -m venv "$VENV"
fi
PIP="$VENV/bin/pip"
$PIP install -q -U pip
log "安装 torch 2.10.0+cu128 (≈2.5G，需要几分钟)..."
$PIP install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
log "安装 ComfyUI 核心依赖 ..."
$PIP install -r "$COMFY/requirements.txt"

# ---------- 6. 核心补丁: int8 convrot 反量化修复 ----------
cd "$COMFY"
if patch -p1 --dry-run < "$SCRIPT_DIR/patches/comfy-ops-int8embedfix-20260926.patch" >/dev/null 2>&1; then
  patch -p1 < "$SCRIPT_DIR/patches/comfy-ops-int8embedfix-20260926.patch"
  log "已应用核心补丁: ops.py int8-embed 反量化修复"
elif grep -q "qdata_is_quant" comfy/ops.py; then
  log "核心补丁已存在，跳过"
else
  die "核心补丁应用失败，请手动检查 install/patches/comfy-ops-int8embedfix-20260926.patch"
fi

# ---------- 7. 自定义节点 (本仓库已捆绑全部节点包) ----------
log "拷贝节点包 → ComfyUI/custom_nodes/ ..."
cp -R "$ROOT/custom_nodes/." "$COMFY/custom_nodes/"

# ---------- 8. 多卡依赖 (raylight: ray/xfuser 序列并行 + FSDP) ----------
log "安装多卡依赖 (ray/xfuser 钉死本项目实测版本) + raylight ..."
$PIP install "ray==2.58.0" "xfuser==0.7.0"
$PIP install -e "$COMFY/custom_nodes/raylight"

# ---------- 9. TE-Speed 核心钩子 (官方补丁脚本) ----------
if grep -q "_run_blocks" "$COMFY/comfy/ldm/minimax/model.py"; then
  log "TE-Speed 钩子已存在，跳过"
else
  "$VENV/bin/python" "$COMFY/custom_nodes/TE-Speed-MiniMaxH3-OSS/patch_model.py" --comfy-ui "$COMFY" || true
  if grep -q "_run_blocks" "$COMFY/comfy/ldm/minimax/model.py"; then
    log "已应用 TE-Speed 核心钩子 (model.py)"
  else
    echo "⚠️  TE-Speed 钩子未生效（不影响启动，仅影响 TE-Speed 节点可用性），请手动检查"
  fi
fi

# ---------- 10. 示例参考图 (图1 开关默认开着, 需要一个占位文件) ----------
if [ ! -f "$COMFY/input/ref_image_1.png" ]; then
  cp "$ROOT/assets/ref_image_1.png" "$COMFY/input/ref_image_1.png"
  log "已放置示例参考图 input/ref_image_1.png (换成你自己的参考图即可)"
fi

# ---------- 11. 工作流注册进 Web UI (Workflow→Open 可直接选) ----------
mkdir -p "$COMFY/user/default/workflows"
cp "$ROOT"/workflows/*.json "$COMFY/user/default/workflows/"
log "工作流已复制到 ComfyUI/user/default/workflows/ (网页端 Workflow→Open 可见)"

# ---------- 12. 冒烟验证: 起 90 秒确认能正常加载 ----------
if [ "$SKIP_VERIFY" -eq 0 ]; then
  log "冒烟验证: 临时启动 ComfyUI (端口 8199)..."
  cd "$COMFY"
  nohup "$VENV/bin/python" main.py --listen 127.0.0.1 --port 8199 > "$ROOT/comfyui_install_check.log" 2>&1 &
  PID=$!
  OK=0
  for i in $(seq 1 45); do
    sleep 2
    if grep -q "Starting server" "$ROOT/comfyui_install_check.log" 2>/dev/null; then OK=1; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then break; fi
  done
  kill "$PID" 2>/dev/null || true
  sleep 2
  kill -9 "$PID" 2>/dev/null || true
  if [ "$OK" -eq 1 ]; then
    log "✅ 冒烟验证通过: ComfyUI 正常启动"
  else
    echo "❌ 冒烟验证失败，最近日志:"
    tail -30 "$ROOT/comfyui_install_check.log"
    exit 1
  fi
fi

echo
log "🎉 环境安装完成!"
echo
echo "下一步:"
echo "  1. bash install/download_models.sh     # 下载模型 (必需约 44G; --optional 再加约 1G)"
echo "  2. bash scripts/start_comfyui_4gpu.sh   # 四卡启动 (双卡机用 start_comfyui_2gpu.sh)"
echo "  3. 浏览器打开 http://127.0.0.1:8188 → Workflow → Open 选对应卡数的工作流"
