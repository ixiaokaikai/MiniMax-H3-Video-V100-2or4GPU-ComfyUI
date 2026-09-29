# -*- coding: utf-8 -*-
"""sla_routing.py — SLA topk 块稀疏路由（对齐 LightX2V dynamic_sparse_attn）

SLA（Sparse-Linear Attention，thu-ml/SLA, arXiv 2509.24006）推理路径的
**块选择**部分：与 Sol-Attn 的阈值路由（keep-or-drop）不同，SLA 是
**每行保留分数最高 top-K 块**（K = max(1, round(topk_ratio × NB))）。

配套权重：lightx2v/Minimax-h3-Turbo-SLA LoRA（4 步蒸馏 + 85% 稀疏蒸馏，
官方 LightX2V config `dynamic_sparse_attn_setting.sparsity_ratio = 0.85`）。
LoRA 本身只是权重（让模型在稀疏下仍出好图），加速完全靠这里的稀疏路由 +
块稀疏 kernel（本插件预编译的 sm70 keep-or-drop kernel，V100 特供）。

块选择逻辑对齐 LightX2V `get_block_map`：
  pooled_score = pooled_qblocks @ pooled_kblocks^T（块均值打分）
  topk = max(1, min(K, round(topk_ratio * K)))  # 每行保留块数
  sel = 每行分数最高的 topk 个块

与 Sol-Attn 路由的差异（有意为之）：
- Sol-Attn：分数 > mean+τ·std 的块全保留（密度随内容浮动 ~26-40%）
- SLA：每行固定保留 top-K（密度锁定 topk_ratio，85% 稀疏 = 15% 密度）

H3 特有：text/cond/ref/audio 前缀不可稀疏 → sink_tokens 保底（SLA 原版无，
本实现按 H3 场景补充，与 SolAttn 的 h3_prefix_tokens 语义一致）。

输出 CSR (cnt, off, ccnt, cidx) 与 SolAttn 同格式（off 存块起始 token 偏移，
块大小 64），可直接喂 flash-attn 的 keep-or-drop sparse kernel
（`torch.ops.comfy_v100_solattn_cuda.varlen_fwd_sparse`，sm70 预编译）。
"""
import math

import torch

BLOCK = 64


def _pad_to_block(t, block=BLOCK):
    S = t.shape[0]
    NB = math.ceil(S / block)
    pad = NB * block - S
    if pad:
        return torch.nn.functional.pad(t, (0, 0, 0, 0, 0, pad)), NB
    return t.contiguous(), NB


def _block_means(q, k, block=BLOCK):
    """块均值（视图化，零 pad/float 拷贝；fp16 块和 64 项，误差 ~1e-3）。
    q,k: [S,H,D] fp16 -> (qc [NQB,H,D] fp32, kc [NB,H,D] fp32, NB)"""
    S, H, D = q.shape
    NB = math.ceil(S / block)
    main = (S // block) * block
    qv = q.permute(1, 0, 2)
    kv = k.permute(1, 0, 2)
    if main >= block:
        qc = qv[:, :main].view(H, main // block, block, D).sum(dim=2).float() / block
        kc = kv[:, :main].view(H, main // block, block, D).sum(dim=2).float() / block
    else:
        qc = kc = None
    if main < S:
        qc_t = qv[:, main:].sum(dim=1, keepdim=True).float() / (S - main)
        kc_t = kv[:, main:].sum(dim=1, keepdim=True).float() / (S - main)
        qc = torch.cat([qc, qc_t], dim=1) if qc is not None else qc_t
        kc = torch.cat([kc, kc_t], dim=1) if kc is not None else kc_t
    qc = qc.permute(1, 0, 2).contiguous()
    kc = kc.permute(1, 0, 2).contiguous()
    return qc, kc, NB


def csr_from_sel(sel, block=BLOCK, nnz_s=None):
    """选中掩码 [NQB,H,NB] -> CSR (cnt, off, ccnt, cidx)，off 存 token 起始偏移
    off: [NQB*H, NNZ_S]（每行最多 NNZ_S 槽；nnz_s=None 时 NNZ_S=NB 全尺寸）
    ⚠️ 行序必须 = (head, query_block)（kernel 索引 (bidh*NUM_ROWS + m_block)）。
    全向量化 + scatter（无 nonzero、无 Python 循环、零 GPU->CPU 同步）。
    """
    NQB, H, NB = sel.shape
    b_h = NQB * H
    sel_perm = sel.permute(1, 0, 2).reshape(b_h, NB)              # 行 = h*NQB + qi
    rank = sel_perm.cumsum(dim=1, dtype=torch.int32) - 1
    slot = NB if nnz_s is None else min(int(nnz_s), NB)
    if slot < NB:
        sel_perm = sel_perm & (rank < slot)                       # cap 行内选中数
    cnt = sel_perm.sum(dim=1, dtype=torch.int32)
    off = torch.zeros(b_h, slot, dtype=torch.int32, device=sel.device)
    row_idx = torch.arange(b_h, device=sel.device, dtype=torch.int32)[:, None].expand(b_h, NB)
    col = (torch.arange(NB, device=sel.device, dtype=torch.int32) * block)[None, :].expand(b_h, NB)
    off[row_idx[sel_perm], rank[sel_perm]] = col[sel_perm]
    ccnt = torch.zeros(b_h, dtype=torch.int32, device=sel.device)
    cidx = torch.zeros(b_h, 1, dtype=torch.int32, device=sel.device)
    return cnt.contiguous(), off.contiguous(), ccnt.contiguous(), cidx.contiguous()


def build_sparse_csr_topk(q, k, topk_ratio=0.15, scale=None, neighbor=0,
                          sink_tokens=0, audio_range=None, block=BLOCK,
                          route_chunk=256):
    """SLA topk 块稀疏路由（逐行对齐 LightX2V get_block_map）。

    q,k: [S,H,D] fp16 -> (cnt, off, ccnt, cidx) 同 SolAttn CSR 格式。

    对齐点（对照 LightX2V `lightx2v/common/ops/attn/utils/sla_util.py`）：
      1. **smooth-k**：k 先减序列均值（SageAttention 技巧）再池化——官方训练路由
         用去中心化 key，块均值反映"变化"而非绝对大小，选块显著不同，必须对齐；
      2. 打分 = pooled_qblocks @ pooled_kblocks^T（不乘 scale——排序不受正 scale 影响）；
      3. topk = max(1, min(K, int(topk_ratio × K)))（int 截断，短序列保 1 块）；
      4. 块大小统一 64（LightX2V BLKQ/BLKK 可分离，本实现受 sm70 kernel 约束）。

    topk 选块用 kthvalue（无排序；并列多选无害——分数相等的块多保留几个无损，
    与 torch.topk 的精确 K 个语义在排序上等价）。

    保底（H3 特有，SLA 原版无）：
    - sink_tokens: 序列前缀（text/cond/ref）token 数，强制保留前 N 块；
    - audio_range: (start_token, end_token) **音频段**（PackedLayout.segments 的
      audio 段行范围）。H3 序列是 [text | ref | audio | video]，音频段不在前缀
      sink 内——SLA topk 稀疏会砍掉音频行的大部分 key 块，**音频质量崩**
      （实测：<500Hz 能量 34% vs dense 7%，开头/结尾最严重）。audio_range
      段强制保留（不稀疏）。

    route_chunk: 流式分块（S=174k, NB=2720, H=40 时 scores 全量 ~1.2GB fp32，
    chunk=256 后 ~110MB；每行独立计算，输出与全量逐位一致）。
    """
    S, H, D = q.shape
    k_center = k - k.mean(dim=0, keepdim=True)          # smooth-k（对齐 LightX2V，序列维去均值）
    qc, kc, NB = _block_means(q, k_center, block)       # 打分用去中心化 key 的块均值
    NQB = NB
    K = max(1, min(NB, int(topk_ratio * NB)))
    # 每行选中数上界 = top-K + 邻域 + sink + 音频段 + 并列余量；slot 用上界省显存
    sink_nb = math.ceil(sink_tokens / block)
    audio_nb = 0
    if audio_range is not None and audio_range[1] > audio_range[0]:
        audio_nb = ((audio_range[1] + block - 1) // block
                    - audio_range[0] // block)
    slot = min(NB, K + 2 * neighbor + sink_nb + audio_nb + 16)
    off = torch.zeros(H * NQB, slot, dtype=torch.int32, device=q.device)
    cnt = torch.zeros(H * NQB, dtype=torch.int32, device=q.device)
    off_v = off.view(H, NQB, slot)
    cnt_v = cnt.view(H, NQB)
    step = route_chunk if route_chunk and route_chunk < NQB else NQB   # 0/超大 = 全量单次
    for c in range(0, NQB, step):
        C = min(step, NQB - c)
        sc = torch.einsum("qhd,jhd->qhj", qc[c:c + C], kc)        # [C,H,NB] 对齐 get_block_map
        if K >= NB:
            sel = torch.ones((C, H, NB), dtype=torch.bool, device=sc.device)
        else:
            kth = torch.kthvalue(sc, k=NB - K + 1, dim=-1).values   # 第 K 大
            sel = sc >= kth.unsqueeze(-1)                           # 每行 ≥K 块
        if neighbor > 0:
            qi = torch.arange(c, c + C, device=sc.device)
            kj = torch.arange(NB, device=sc.device)
            sel |= ((kj[None, None, :] - qi[:, None, None]).abs() <= neighbor)
        if sink_nb > 0:
            sel[:, :, :sink_nb] = True
        if audio_nb > 0:
            a_blk = audio_range[0] // block
            sel[:, :, a_blk:a_blk + audio_nb] = True
        cnt_c, off_c, _, _ = csr_from_sel(sel, block=block, nnz_s=slot)
        off_v[:, c:c + C, :] = off_c.reshape(H, C, slot)
        cnt_v[:, c:c + C] = cnt_c.reshape(H, C)
    ccnt = torch.zeros(H * NQB, dtype=torch.int32, device=q.device)
    cidx = torch.zeros(H * NQB, 1, dtype=torch.int32, device=q.device)
    return cnt, off, ccnt, cidx
