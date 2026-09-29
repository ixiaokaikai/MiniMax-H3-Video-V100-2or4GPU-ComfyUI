# -*- coding: utf-8 -*-
"""verify_sla_topk_gpu.py — SLA topk 路由 GPU 端到端验证（V100 真机）

用真实激活（_h3_sparse_snapshots.pt，S=6154/H=56/fp16）验证：
1. SLA topk 路由产出的 CSR 能被 sm70 keep-or-drop kernel 正确消费
   （varlen_fwd_sparse 跑通，稀疏输出 vs dense SDPA 的 rel 误差合理）
2. SolAttn 阈值路由对照（同一 kernel，同一激活）
3. 耗时：路由 ms / 稀疏 kernel ms / dense SDPA ms
4. nvidia-smi util/power 监控（200ms 采样，GPU 铁律）

运行环境：ComfyUI 自带 python（3.12 + torch cu128 + V100）。脚本放 verify/ 目录内运行，
快照路径作参数传入（默认插件根目录 _h3_sparse_snapshots.pt）：
  python verify/verify_sla_topk_gpu.py <激活快照.pt>
"""
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

_PLUGIN_DIR = Path(__file__).resolve().parent.parent    # 插件根目录
sys.path.insert(0, str(_PLUGIN_DIR))

from sla_routing import build_sparse_csr_topk          # noqa: E402
from routing import build_sparse_csr                    # noqa: E402

_EXT = ".so" if sys.platform.startswith("linux") else ".pyd"
_PYD_CANDIDATES = sorted(_PLUGIN_DIR.glob(f"comfy_v100_solattn_cuda*{_EXT}"))
if not _PYD_CANDIDATES:
    _PYD_CANDIDATES = [
        f for f in sorted(_PLUGIN_DIR.glob("comfy_v100_solattn_cuda*"))
        if f.suffix in (".pyd", ".so")
    ]
PYD = str(_PYD_CANDIDATES[0]) if _PYD_CANDIDATES else str(
    _PLUGIN_DIR / "comfy_v100_solattn_cuda.cp312-win_amd64.pyd"
)
SNAP = sys.argv[1] if len(sys.argv) > 1 else str(
    _PLUGIN_DIR / "_h3_sparse_snapshots.pt"
)


class GpuMonitor:
    """nvidia-smi util/power 采样（200ms），finally terminate（GPU 铁律）"""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.rows = []
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                )
                self.rows.append(out.stdout.strip())
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)

    def report(self):
        if not self.rows:
            return "no samples"
        utils, powers = [], []
        for r in self.rows:
            parts = r.replace(" ", "").split(",")
            if len(parts) >= 2:
                try:
                    utils.append(int(parts[0]))
                    powers.append(float(parts[1]))
                except ValueError:
                    pass
        return (f"samples={len(self.rows)} util[max/avg]={max(utils) if utils else 0}/"
                f"{sum(utils)//len(utils) if utils else 0}% "
                f"power[max/avg]={max(powers) if powers else 0:.0f}/"
                f"{sum(powers)/len(powers) if powers else 0:.0f}W")


def bench(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000    # ms


def main():
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 7
    torch.ops.load_library(PYD)
    assert hasattr(torch.ops.comfy_v100_solattn_cuda, "varlen_fwd_sparse")

    snap = torch.load(SNAP, map_location="cuda", weights_only=True)
    q, k, v = snap[6154]                                   # (1,56,6154,128) fp16
    S, H, D = q.shape[-2], q.shape[1], q.shape[-1]
    scale = D ** -0.5
    q3 = q.squeeze(0).permute(1, 0, 2).contiguous()        # [S,H,D]
    k3 = k.squeeze(0).permute(1, 0, 2).contiguous()
    v3 = v.squeeze(0).permute(1, 0, 2).contiguous()
    cu = torch.tensor([0, S], dtype=torch.int32, device="cuda")
    print(f"激活: S={S} H={H} D={D} fp16 | GPU={torch.cuda.get_device_name(0)}")

    # dense 对照（fp16 SDPA，同精度）
    q4, k4, v4 = q3.permute(1, 0, 2).unsqueeze(0), k3.permute(1, 0, 2).unsqueeze(0), v3.permute(1, 0, 2).unsqueeze(0)
    o_dn = torch.nn.functional.scaled_dot_product_attention(q4, k4, v4, scale=scale)

    def run(csr):
        cnt, off, ccnt, cidx = csr
        return torch.ops.comfy_v100_solattn_cuda.varlen_fwd_sparse(
            q3, k3, v3, None, cu, cu, cnt, off, ccnt, cidx, S, S, scale)[0]

    with GpuMonitor() as mon:
        print("\n== SLA topk（sparsity_ratio=0.85, sink=1024）==")
        csr_sla = build_sparse_csr_topk(q3, k3, topk_ratio=0.15, sink_tokens=1024)
        dens = float(csr_sla[0].float().mean()) / ((S + 63) // 64)
        out_sla = run(csr_sla).reshape(S, H, D).permute(1, 0, 2).unsqueeze(0)
        rel_sla = (out_sla.float() - o_dn.float()).norm() / o_dn.float().norm()
        t_route_sla = bench(lambda: build_sparse_csr_topk(q3, k3, topk_ratio=0.15, sink_tokens=1024))
        t_kernel_sla = bench(lambda: run(csr_sla))
        print(f"  密度={dens*100:.1f}% | 稀疏vs dense rel-L2={rel_sla:.4f} | "
              f"路由 {t_route_sla:.1f}ms | kernel {t_kernel_sla:.2f}ms")

        print("\n== SLA topk + audio 段保底（模拟 audio_range，8s 音频 640 行）==")
        ar = (1024 + 64, 1024 + 64 + 640)
        csr_aud = build_sparse_csr_topk(q3, k3, topk_ratio=0.15, sink_tokens=1024,
                                        audio_range=ar)
        dens_a = float(csr_aud[0].float().mean()) / ((S + 63) // 64)
        out_aud = run(csr_aud).reshape(S, H, D).permute(1, 0, 2).unsqueeze(0)
        rel_aud = (out_aud.float() - o_dn.float()).norm() / o_dn.float().norm()
        t_kernel_aud = bench(lambda: run(csr_aud))
        print(f"  密度={dens_a*100:.1f}% | 稀疏vs dense rel-L2={rel_aud:.4f} | "
              f"kernel {t_kernel_aud:.2f}ms（audio 块保底）")

        print("\n== SolAttn 阈值路由对照（tau=1.0, sink=1024）==")
        csr_sol = build_sparse_csr(q3, k3, tau=1.0, sink_tokens=1024)
        dens_sol = float(csr_sol[0].float().mean()) / ((S + 63) // 64)
        out_sol = run(csr_sol).reshape(S, H, D).permute(1, 0, 2).unsqueeze(0)
        rel_sol = (out_sol.float() - o_dn.float()).norm() / o_dn.float().norm()
        t_route_sol = bench(lambda: build_sparse_csr(q3, k3, tau=1.0, sink_tokens=1024))
        t_kernel_sol = bench(lambda: run(csr_sol))
        print(f"  密度={dens_sol*100:.1f}% | 稀疏vs dense rel-L2={rel_sol:.4f} | "
              f"路由 {t_route_sol:.1f}ms | kernel {t_kernel_sol:.2f}ms")

        t_dense = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q4, k4, v4, scale=scale))
        print(f"\n== dense SDPA 对照 ==\n  kernel SDPA {t_dense:.2f}ms")

    print(f"\n== nvidia-smi 监控 ==\n  {mon.report()}")
    print("\nDONE")


if __name__ == "__main__":
    main()
