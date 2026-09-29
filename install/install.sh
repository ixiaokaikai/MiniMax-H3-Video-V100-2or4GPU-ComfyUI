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
WITH_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --skip-verify) SKIP_VERIFY=1 ;;
    --with-models) WITH_MODELS=1 ;;   # 装完接着跑 download_models.sh (模型已存在会自动跳过)
    *) echo "⚠️  未知参数: $arg (可用: --skip-verify  --with-models)" ;;
  esac
done

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
# 小显存卡提示: 工作站上常另插一张亮机卡, 启动脚本会自动跳过它
SMALL_CARDS=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits \
              | awk -F', *' '$3 < 14000 {printf "GPU%s(%s %sM) ", $1, $2, $3}')
if [ -n "${SMALL_CARDS:-}" ]; then
  echo "⚠️  显存 <14G 的卡: $SMALL_CARDS"
  echo "    这类卡跑不了本项目; 启动脚本会自动跳过, 只挑 ≥14G 的卡"
  echo "    想手动指定: export CUDA_VISIBLE_DEVICES=<用逗号分隔的卡号>"
fi
if nvidia-smi nvlink -s 2>/dev/null | grep -q "Link"; then
  log "检测到 NVLink — 多卡并行的推荐配置"
else
  echo "⚠️  未检测到 NVLink：PCIe 也能跑，但多卡并行会明显变慢，推荐 SXM+NVLink 卡"
fi

# ---------- 2. 内存 / 磁盘检查 ----------
# 用 /proc/meminfo (与语言环境无关; free -g 在中文 locale 下输出"内存："会读不到)
RAM_GB=$(awk '/^MemTotal:/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)
if [ -z "${RAM_GB:-}" ]; then
  RAM_GB=$(LANG=C free -g 2>/dev/null | awk '/^Mem/ {print $2}')
fi
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
command -v python3.12 >/dev/null || die "需要 Python 3.12（部分节点包预编译扩展仅支持 cp312）。Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
# 只有 python3.12 不够: Debian/Ubuntu 上 venv/ensurepip 由 python3.12-venv 包提供, 缺了会在建虚拟环境时失败
python3.12 -c 'import ensurepip' >/dev/null 2>&1 || die "python3.12 缺少 venv 模块（无法创建虚拟环境）。Ubuntu/Debian: sudo apt install python3.12-venv，然后重跑本脚本"

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
# 若用户指定了国内 pip 镜像( PIP_INDEX_URL ), 再加官方 PyPI 作兜底:
# 实测 aliyun/清华等镜像缺 comfyui-frontend-package (ComfyUI 必需包), 单索引会直接安装失败
PIP_FALLBACK=()
if [ -n "${PIP_INDEX_URL:-}" ]; then
  PIP_FALLBACK=(--extra-index-url "https://pypi.org/simple")
  log "检测到 PIP_INDEX_URL=$PIP_INDEX_URL (镜像缺包时自动回落到官方 PyPI)"
fi
$PIP install -q -U pip
log "安装 torch 2.10.0+cu128 (≈2.5G-4G，需要几分钟；期间没有输出是 pip 进度条不写日志，属正常)..."
$PIP install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
log "安装 ComfyUI 核心依赖 ..."
$PIP install "${PIP_FALLBACK[@]}" -r "$COMFY/requirements.txt"

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
$PIP install "${PIP_FALLBACK[@]}" "ray==2.58.0" "xfuser==0.7.0"
$PIP install "${PIP_FALLBACK[@]}" -e "$COMFY/custom_nodes/raylight"


# ---------- 9. TE-Speed 核心钩子 (comfy/ldm/minimax/model.py) ----------
# 说明: 上游 TE-Speed 的 patch_model.py 目前不认 ComfyUI v0.37.0 的 model.py
# (它要求 make_prefetch_queue 的调用跨行, 而 v0.37.0 里是单行), 会直接报错退出。
# 所以这里用本仓库自带、按本项目实测环境生成的确定性补丁 (等价于上游钩子效果)。
if grep -q "_run_blocks" "$COMFY/comfy/ldm/minimax/model.py"; then
  log "TE-Speed 钩子已存在，跳过"
elif patch -p1 -d "$COMFY" --dry-run < "$SCRIPT_DIR/patches/comfy-minimax-te-speed-hook-20260929.patch" >/dev/null 2>&1; then
  patch -p1 -d "$COMFY" < "$SCRIPT_DIR/patches/comfy-minimax-te-speed-hook-20260929.patch" >/dev/null
  log "已应用 TE-Speed 核心钩子 (model.py, via install/patches/)"
else
  # 兜底: 试试上游自带脚本 (适用于将来它对得上的版本)
  "$VENV/bin/python" "$COMFY/custom_nodes/TE-Speed-MiniMaxH3-OSS/patch_model.py" --comfy-ui "$COMFY" || true
  if grep -q "_run_blocks" "$COMFY/comfy/ldm/minimax/model.py"; then
    log "已应用 TE-Speed 核心钩子 (via 上游 patch_model.py)"
  else
    echo "⚠️  TE-Speed 钩子未生效（仅影响可选的 TE-Speed 加速节点，两份工作流里该节点默认关闭），请手动检查"
  fi
fi

# ---------- 10. 占位素材 (工作流引用的输入文件; 缺任何一个, 整单校验都会失败) ----------
mkdir -p "$COMFY/input"
PLACED=0
for f in "$ROOT/assets/ref_image_1.png" "$ROOT"/assets/placeholders/*; do
  [ -e "$f" ] || continue
  base="$(basename "$f")"
  if [ ! -f "$COMFY/input/$base" ]; then
    cp "$f" "$COMFY/input/$base"
    PLACED=$((PLACED + 1))
  fi
done
[ "$PLACED" -gt 0 ] && log "已放置 $PLACED 个占位素材到 ComfyUI/input/ (换成你自己的图/视频/音频即可)"

# ---------- 11. 工作流注册进 Web UI (Workflow→Open 可直接选) ----------
mkdir -p "$COMFY/user/default/workflows"
cp "$ROOT"/workflows/*.json "$COMFY/user/default/workflows/"
log "工作流已复制到 ComfyUI/user/default/workflows/ (网页端 Workflow→Open 可见)"

# ---------- 12. 冒烟验证: 临时起实例, 用 HTTP 探活 + 节点自检 ----------
if [ "$SKIP_VERIFY" -eq 0 ]; then
  # 端口动态挑空闲的: 固定端口被别的服务占用时, ComfyUI 会打印 "Starting server" 之后才因端口冲突退出,
  # 只看日志关键字会得到"假通过"; 因此这里既不用固定端口, 也不看日志关键字, 直接探 HTTP。
  PORT=$("$VENV/bin/python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
  log "冒烟验证: 临时启动 ComfyUI (端口 $PORT)..."
  cd "$COMFY"
  nohup "$VENV/bin/python" main.py --listen 127.0.0.1 --port "$PORT" > "$ROOT/comfyui_install_check.log" 2>&1 &
  PID=$!
  OK=0
  for i in $(seq 1 45); do
    sleep 2
    if "$VENV/bin/python" - "$PORT" <<'PYEOF' 2>/dev/null
import sys, urllib.request
o = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 绕开代理设置, 本机服务直连
try:
    with o.open("http://127.0.0.1:%s/object_info" % sys.argv[1], timeout=5) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PYEOF
    then OK=1; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then break; fi
  done
  CHECK_RC=0
  if [ "$OK" -eq 1 ]; then
    log "✅ 冒烟验证通过: ComfyUI 正常启动"
    # 逐个核对工作流用到的节点类型 (缺节点的坑只在网页上表现为红框, 这里提前暴露)
    bash "$ROOT/scripts/check_nodes.sh" --url "http://127.0.0.1:$PORT" </dev/null || CHECK_RC=1
  else
    echo "❌ 冒烟验证失败，最近日志:"
    tail -30 "$ROOT/comfyui_install_check.log"
    kill "$PID" 2>/dev/null || true
    exit 1
  fi
  kill "$PID" 2>/dev/null || true
  sleep 2
  kill -9 "$PID" 2>/dev/null || true
  if [ "$CHECK_RC" -ne 0 ]; then
    echo "⚠️  节点自检未全部通过 (见上)。环境基本可用, 但打开工作流可能有些节点显示红框。"
    echo "    排查: 看 $ROOT/comfyui.log 里对应节点包的 import 报错; 重新自检: bash scripts/check_nodes.sh"
  fi
fi

echo
log "🎉 环境安装完成!"

if [ "$WITH_MODELS" -eq 1 ]; then
  echo
  log "接着下载模型 (已存在的会自动跳过)..."
  bash "$ROOT/install/download_models.sh"
fi

echo
echo "下一步:"
if [ "$WITH_MODELS" -eq 0 ]; then
  echo "  1. bash install/download_models.sh     # 下载模型 (必需 5 个约 44G; --optional 再加约 0.8G)"
  echo "     国内网络: HF_ENDPOINT=https://hf-mirror.com bash install/download_models.sh"
fi
echo "  启动四卡: bash scripts/start_comfyui_4gpu.sh"
echo "  启动双卡: bash scripts/start_comfyui_2gpu.sh"
echo "  浏览器打开 http://127.0.0.1:8188 → Workflow → Open 选对应卡数的工作流 → 改提示词 → Queue Prompt"
