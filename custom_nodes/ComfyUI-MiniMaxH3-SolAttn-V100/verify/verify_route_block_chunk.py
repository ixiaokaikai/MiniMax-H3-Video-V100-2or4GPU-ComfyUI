# -*- coding: utf-8 -*-
"""verify_route_block_chunk.py — SolAttn 流式分块路由验证（2026-08-23）

routing.build_sparse_csr(route_chunk>0) 输出应与 route_chunk=0（全量）**逐位一致**
（每行独立计算，无跨行依赖；scores 是同一 einsum 的行子集），但中间张量
scores/sel/rank/scatter 峰值 /route_chunk。验证点：
  1. CPU 随机：chunk=8/32/64 vs full 输出 (cnt, off) 逐位一致（含 topk_k / nnz_s / sink 组合）
  2. GPU 随机：S=3000/6154/29650 chunk=32 vs full 逐位一致
  3. GPU 真实激活质量：S=6154 快照，chunk 路由+kernel vs dense，rel 与 full 完全一致
  4. GPU 速度 + 峰值显存：S=98512 / 174112，full vs chunk=32 路由耗时 + max_memory_allocated
  5. nvidia-smi 200ms 监控（util/power/temp）

用法：python verify/verify_route_block_chunk.py [--cpu-only]
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

import torch

_PLUGIN_DIR = Path(__file__).resolve().parent.parent    # 插件根目录
sys.path.insert(0, str(_PLUGIN_DIR))

import routing

_ext = ".so" if sys.platform.startswith("linux") else ".pyd"
_cands = sorted(_PLUGIN_DIR.glob(f"comfy_v100_solattn_cuda*{_ext}"))
if not _cands:
    _cands = [f for f in sorted(_PLUGIN_DIR.glob("comfy_v100_solattn_cuda*"))
              if f.suffix in (".pyd", ".so")]
if not _cands:
    raise RuntimeError("SolAttn: 缺少 kernel 文件（.pyd/.so）")
torch.ops.load_library(str(_cands[0]))

H, D = 56, 128
SMI_Q = "utilization.gpu,power.draw,temperature.gpu,memory.used"


def _smi():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + SMI_Q,
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        p = out.splitlines()[0].replace(" MiB", "").split(", ")
        return float(p[0]), float(p[1]), float(p[2]), float(p[3]) / 1024.0
    except Exception:
        return None


def gpu_monitor(stop_evt, results):
    while not stop_evt.is_set():
        s = _smi()
        if s:
            results.append(s)
        time.sleep(0.2)


def check(cond, msg):
    if not cond:
        print(f"[FAIL] {msg}")
        sys.exit(1)
    print(f"[ OK ] {msg}")


def make_qkv(S, Hd=H, Dd=D, device="cpu"):
    q = torch.randn(1, Hd, S, Dd, dtype=torch.float16, device=device)
    k = torch.randn(1, Hd, S, Dd, dtype=torch.float16, device=device)
    return (q.squeeze(0).permute(1, 0, 2),
            k.squeeze(0).permute(1, 0, 2))


def build(q3, k3, scale, route_chunk, topk_k=32, nnz_s=None, sink_tokens=0):
    return routing.build_sparse_csr(q3, k3, tau=0.75, scale=scale, sink_tokens=sink_tokens,
                                    nnz_s=nnz_s, topk_k=topk_k, route_chunk=route_chunk)


def eq_csr(a, b):
    return (torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            and torch.equal(a[2], b[2]) and torch.equal(a[3], b[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--snapshot", default=None, help="H3 激活快照 .pt 路径（默认当前目录 _h3_sparse_snapshots.pt）")
    args = ap.parse_args()

    # ---------- 1. CPU 随机：chunk vs full 逐位一致 ----------
    print("=== 1. CPU 随机（含参数组合）===")
    torch.manual_seed(0)
    for S, Hd in [(3000, 8), (6154, 8)]:
        q3, k3 = make_qkv(S, Hd)
        scale = D ** -0.5
        NB = (S + 63) // 64
        nnz_s = max(256, NB // 2)
        for chunk in (64, 128, 256):
            full = build(q3, k3, scale, 0, topk_k=32, nnz_s=nnz_s, sink_tokens=1000)
            part = build(q3, k3, scale, chunk, topk_k=32, nnz_s=nnz_s, sink_tokens=1000)
            check(eq_csr(full, part), f"S={S} H={Hd} chunk={chunk} topk+nnz+sink == full 逐位一致")
        # topk_k=0（无保底）+ 无 sink + 全尺寸 nnz_s=None
        full = build(q3, k3, scale, 0, topk_k=0, nnz_s=None, sink_tokens=0)
        part = build(q3, k3, scale, 128, topk_k=0, nnz_s=None, sink_tokens=0)
        check(eq_csr(full, part), f"S={S} H={Hd} chunk=128 无topk无sink全尺寸 == full")

    if args.cpu_only:
        print("\nCPU 验证 ALL PASS（GPU 部分跳过）")
        return

    # ---------- GPU ----------
    torch.cuda.init()
    stop = threading.Event()
    mon = []
    t_mon = threading.Thread(target=gpu_monitor, args=(stop, mon), daemon=True)
    t_mon.start()

    print("\n=== 2. GPU 随机：chunk=128 vs full 逐位一致 ===")
    for S in (3000, 6154, 29650):
        q3, k3 = make_qkv(S, device="cuda")
        scale = D ** -0.5
        NB = (S + 63) // 64
        nnz_s = max(256, NB // 2)
        full = build(q3, k3, scale, 0, topk_k=32, nnz_s=nnz_s)
        part = build(q3, k3, scale, 128, topk_k=32, nnz_s=nnz_s)
        check(eq_csr(full, part), f"S={S} chunk=128 == full（cnt/off/ccnt/cidx 逐位）")
        dens = float(part[0].float().mean().item()) / NB
        print(f"       密度 ≈ {dens * 100:.1f}%（NB={NB}）")

    print("\n=== 3. GPU 真实激活质量（S=6154 快照）===")
    snap_path = args.snapshot if getattr(args, "snapshot", None) else "_h3_sparse_snapshots.pt"
    snap = torch.load(snap_path, map_location="cpu", weights_only=True)
    q, k, v = [t.to("cuda") for t in snap[6154]]
    S = q.shape[-2]
    q3 = q.squeeze(0).permute(1, 0, 2)
    k3 = k.squeeze(0).permute(1, 0, 2)
    v3 = v.squeeze(0).permute(1, 0, 2)
    scale = D ** -0.5
    q4 = q3.unsqueeze(0).permute(0, 2, 1, 3)
    k4 = k3.unsqueeze(0).permute(0, 2, 1, 3)
    v4 = v3.unsqueeze(0).permute(0, 2, 1, 3)
    out_dense = torch.nn.functional.scaled_dot_product_attention(
        q4, k4, v4, scale=scale).permute(0, 2, 1, 3).reshape(1, S, H * D).squeeze(0).float()
    cu = torch.tensor([0, S], dtype=torch.int32, device="cuda")
    rels = {}
    for chunk in (0, 128):
        cnt, off, ccnt, cidx = build(q3, k3, scale, chunk, topk_k=32, nnz_s=max(256, S // 64 // 2))
        out, _ = torch.ops.comfy_v100_solattn_cuda.varlen_fwd_sparse(
            q3, k3, v3, None, cu, cu, cnt, off, ccnt, cidx, S, S, scale)
        out_f = out.reshape(S, H * D).float()
        rels[chunk] = (out_f - out_dense).abs().mean().item() / out_dense.abs().mean().item()
    check(rels[0] == rels[128], f"chunk=128 稀疏输出 rel({rels[128]:.4f}) == full rel({rels[0]:.4f})（sel 逐位一致）")
    print(f"       tau=0.75+topk32 vs dense: rel = {rels[128]:.4f}")

    print("\n=== 4. GPU 速度 + 峰值显存（路由，不含 kernel）===")
    for S in (98512, 174112):
        q3, k3 = make_qkv(S, device="cuda")
        scale = D ** -0.5
        NB = (S + 63) // 64
        nnz_s = max(256, NB // 2)
        # 默认参数（route_chunk=256）必须与 full 逐位一致
        full = build(q3, k3, scale, 0, topk_k=32, nnz_s=nnz_s)
        dflt = routing.build_sparse_csr(q3, k3, tau=0.75, scale=scale, sink_tokens=0,
                                        nnz_s=nnz_s, topk_k=32)
        check(eq_csr(full, dflt), f"S={S} 默认参数(route_chunk=256) == full 逐位一致")
        for chunk in (0, 128, 256):
            try:
                torch.cuda.reset_peak_memory_stats()
                ts = []
                for _ in range(3):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    build(q3, k3, scale, chunk, topk_k=32, nnz_s=nnz_s)
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
                peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                print(f"  S={S:,} chunk={chunk:>3}: 路由 {min(ts):7.1f} ms | active峰值 {peak:5.2f} GB")
            except torch.cuda.OutOfMemoryError:
                print(f"  S={S:,} chunk={chunk:>3}: OOM（16GB 上限）")
                torch.cuda.empty_cache()
        del q3, k3
        torch.cuda.empty_cache()

    stop.set()
    t_mon.join(timeout=1)
    if mon:
        u = [m[0] for m in mon]; p = [m[1] for m in mon]; t = [m[2] for m in mon]
        print(f"\n[GPU 监控 {len(mon)} 采样] util mean={sum(u)/len(u):.0f}% max={max(u):.0f}% | "
              f"power mean={sum(p)/len(p):.0f}W max={max(p):.0f}W | temp mean={sum(t)/len(t):.0f}C max={max(t):.0f}C")
    print("\nALL PASS")


if __name__ == "__main__":
    main()
