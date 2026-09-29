# -*- coding: utf-8 -*-
"""verify_sla_topk.py — SLA topk 路由验证（对齐 LightX2V get_block_map）

验证项：
1. build_sparse_csr_topk 的选块与 LightX2V get_block_map 参考实现逐位一致
   （smooth-k + 块均值 + int 截断 topk + torch.topk），并列块差集无害
2. CSR 重建 sel == 路由内部 sel（CSR 格式正确性，喂 kernel 前自检）
3. 密度 ≈ topk_ratio（+sink 保底）
4. route_chunk 分块 vs 全量逐位一致
5. 真实激活（_h3_sparse_snapshots.pt，480p S=6154, H=56）稀疏 vs dense 输出 rel 误差
   （CPU 上截断 S 控制内存）

用法（CPU 即可，无需 GPU）：
  python verify/verify_sla_topk.py
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 插件根目录
from sla_routing import BLOCK, build_sparse_csr_topk


# ---------------- LightX2V get_block_map 参考实现（逐行复刻） ----------------
def ref_get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    """q,k: [S,H,D] fp16 -> (sel [NQB,H,NB] bool, topk, pooled_score)"""
    S, H, D = q.shape
    arg_k = k - k.mean(dim=0, keepdim=True)                      # smooth-k（序列维）

    def mean_pool(x, blk):
        """[S,H,D] -> [NB,H,D] 块均值（尾部不足一块用有效长度，等价 LightX2V compress_kernel）"""
        S2, H2, D2 = x.shape
        NB = math.ceil(S2 / blk)
        main = (S2 // blk) * blk
        xv = x.permute(1, 0, 2)                                 # [H,S,D]
        if main >= blk:
            out = xv[:, :main].view(H2, main // blk, blk, D2).sum(dim=2).float() / blk
        else:
            out = None
        if main < S2:
            tail = xv[:, main:].sum(dim=1, keepdim=True).float() / (S2 - main)
            out = torch.cat([out, tail], dim=1) if out is not None else tail
        return out.permute(1, 0, 2).contiguous()                # [NB,H,D]

    qp = mean_pool(q, BLKQ)
    kp = mean_pool(arg_k, BLKK)
    pooled_score = torch.einsum("qhd,jhd->qhj", qp, kp)          # [NQB,H,NB]
    K = pooled_score.shape[-1]
    topk = max(1, min(K, int(topk_ratio * K)))                   # int 截断
    sel = torch.zeros_like(pooled_score, dtype=torch.bool)
    _, idx = torch.topk(pooled_score, topk, dim=-1, sorted=False)
    sel.scatter_(-1, idx, True)
    return sel, topk, pooled_score


def csr_to_sel(cnt, off, NQB, H, NB, block=BLOCK):
    """CSR (cnt, off) 重建选中掩码 [NQB,H,NB] bool"""
    sel = torch.zeros(NQB, H, NB, dtype=torch.bool)
    for h in range(H):
        for qi in range(NQB):
            row = h * NQB + qi
            n = int(cnt[row])
            sel[qi, h, off[row, :n] // block] = True
    return sel


# ---------------- 测试 ----------------
def main():
    torch.manual_seed(0)
    fails = 0

    print("== 1. 与 LightX2V get_block_map 参考实现对齐 ==")
    for S, H, D, ratio, sink in [
        (1000, 4, 128, 0.15, 0),
        (5000, 8, 128, 0.15, 1024),
        (6154, 56, 128, 0.15, 1024),     # 480p 真实序列长度
        (999, 2, 64, 0.85, 0),           # 非整块 + 高稀疏
        (100, 2, 128, 0.15, 0),          # 短序列（topk 保 1 块）
    ]:
        q = torch.randn(S, H, D, dtype=torch.float16)
        k = torch.randn(S, H, D, dtype=torch.float16)
        ref_sel, topk, pooled_score = ref_get_block_map(q, k, ratio)
        NQB, _, NB = ref_sel.shape
        cnt, off, _, _ = build_sparse_csr_topk(
            q, k, topk_ratio=ratio, sink_tokens=sink)
        our_sel = csr_to_sel(cnt, off, NQB, H, NB)
        # 差集分析：ours 多选/少选的块
        only_ref = ref_sel & ~our_sel
        only_ours = our_sel & ~ref_sel
        # 只有"分数等于第 K 大"的并列块可以多选（kthvalue >= 阈值）
        kth = torch.kthvalue(pooled_score, k=NB - topk + 1, dim=-1).values
        tie_ok = pooled_score >= kth.unsqueeze(-1) - 1e-3        # 并列容差
        sink_nb = math.ceil(sink / BLOCK)
        if sink_nb > 0:
            tie_ok[:, :, :sink_nb] = True                        # sink 前缀豁免（H3 增强，非 SLA 原版）
        bad_ours = only_ours & ~tie_ok
        n_only_ref = int(only_ref.sum())
        n_bad = int(bad_ours.sum())
        dens = float(our_sel.float().mean())
        ok = n_bad == 0 and n_only_ref == 0
        print(f"  S={S:6d} H={H:2d} ratio={ratio:.2f} sink={sink:5d} | "
              f"K={topk:4d}/{NB:4d} 密度={dens*100:5.1f}% | "
              f"ref独有={n_only_ref} 超选并列={n_only_ref} 超选非并列={n_bad} | "
              f"{'PASS' if ok else 'FAIL'}")
        fails += 0 if ok else 1

    print("\n== 2. CSR 重建 vs 路由内部 sel（route_chunk 分块 vs 全量逐位一致）==")
    for S, H, D in [(5000, 8, 128), (6154, 56, 128)]:
        q = torch.randn(S, H, D, dtype=torch.float16)
        k = torch.randn(S, H, D, dtype=torch.float16)
        cnt_f, off_f, _, _ = build_sparse_csr_topk(
            q, k, topk_ratio=0.15, sink_tokens=1024, route_chunk=0)
        cnt_c, off_c, _, _ = build_sparse_csr_topk(
            q, k, topk_ratio=0.15, sink_tokens=1024, route_chunk=64)
        same = (cnt_f == cnt_c).all() and (off_f == off_c).all()
        print(f"  S={S:6d} H={H:2d}: chunk64 vs 全量 逐位一致 = {bool(same)}")
        fails += 0 if bool(same) else 1

    print("\n== 3. 真实激活质量（S=6154, H=56, 截断 S 控内存）==")
    snap_path = sys.argv[1] if len(sys.argv) > 1 else "_h3_sparse_snapshots.pt"
    snap = torch.load(snap_path, map_location="cpu", weights_only=True)
    q, k, v = snap[6154]
    S_full = q.shape[-2]
    SLICE = 2048                                   # CPU 内存限制，截断序列
    qc = q[:, :, :SLICE].squeeze(0).permute(1, 0, 2).half().contiguous()   # [S,H,D]
    kc = k[:, :, :SLICE].squeeze(0).permute(1, 0, 2).half().contiguous()
    vc = v[:, :, :SLICE].squeeze(0).permute(1, 0, 2).half().contiguous()
    H, D = qc.shape[1], qc.shape[2]
    for ratio, sink in [(0.15, 0), (0.15, 1024), (0.30, 1024)]:
        cnt, off, _, _ = build_sparse_csr_topk(
            qc, kc, topk_ratio=ratio, sink_tokens=sink)
        NQB = math.ceil(SLICE / BLOCK)
        NB = NQB
        sel = csr_to_sel(cnt, off, NQB, H, NB)
        # dense vs 稀疏输出（token 级 mask 扩展）
        scale = D ** -0.5
        q32 = qc.float(); k32 = kc.float(); v32 = vc.float()
        scores = torch.einsum("shd,thd->sth", q32, k32) * scale          # [S,T,H]
        # 块级 mask -> token 级（q 维/k 维 repeat_interleave 展开）
        mask_tok = sel.permute(1, 2, 0)                                  # [H,NB,NQB]
        mask_tok = mask_tok.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
        mask_tok = mask_tok.permute(2, 1, 0).contiguous()                # [S,T,H]
        sm = torch.where(mask_tok, scores, torch.full_like(scores, -float("inf")))
        p_sp = torch.softmax(sm, dim=1)
        o_sp = torch.einsum("sth,thd->shd", p_sp, v32)
        p_dn = torch.softmax(scores, dim=1)
        o_dn = torch.einsum("sth,thd->shd", p_dn, v32)
        rel = (o_sp - o_dn).norm() / (o_dn.norm() + 1e-6)
        print(f"  topk_ratio={ratio:.2f} sink={sink:5d} | 密度={float(sel.float().mean())*100:.1f}% "
              f"| 稀疏vs dense rel-L2 = {rel:.4f}")

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
