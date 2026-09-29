#!/usr/bin/env bash
# ============================================================
#  ComfyUI 停止 — 释放全部显存
# ============================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! pgrep -f "$ROOT/ComfyUI/main.py" >/dev/null 2>&1; then
  echo "ComfyUI 没在运行, 无需操作。"
  exit 0
fi

echo "正在停止 ComfyUI..."
pkill -f "$ROOT/ComfyUI/main.py" 2>/dev/null || true
sleep 3

if pgrep -f "$ROOT/ComfyUI/main.py" >/dev/null 2>&1; then
  echo "优雅停止没成功, 强制结束..."
  pkill -9 -f "$ROOT/ComfyUI/main.py" 2>/dev/null || true
  sleep 2
fi

# 清掉 RayLight 的 Ray worker 残留 (本地 Ray 集群常驻, 主进程退了 worker 不会自动退)
pkill -f "ray::RayWorker" 2>/dev/null || true
sleep 1

if pgrep -f "$ROOT/ComfyUI/main.py" >/dev/null 2>&1; then
  echo "❌ 仍有进程残留:"
  pgrep -af "$ROOT/ComfyUI/main.py"
  exit 1
fi

echo
echo "✅ ComfyUI 已停止, 显存已释放。"
echo "当前 GPU 显存占用:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
  awk -F', ' '{printf "  GPU%s: %s MiB\n", $1, $2}'
