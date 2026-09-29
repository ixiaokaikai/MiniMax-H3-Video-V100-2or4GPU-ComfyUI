# -*- coding: utf-8 -*-
"""sla_nodes.py — SLA Top-K (V100)：MiniMax H3 + lightx2v SLA LoRA 的稀疏加速节点。

独立节点（与 SolAttn 节点互不干扰，同一 MODEL 上二者不能同时启用——
optimized_attention_override 只有一个槽位）。

用途：配合 lightx2v/Minimax-h3-Turbo-SLA LoRA（ComfyUI LoraLoader 直接加载，
标准 rank-128 LoRA，只改 qkv/out/mlp 线性权重）使用。LoRA 本身不含稀疏逻辑，
加速完全靠本节点的 **SLA topk 块稀疏路由**（sla_routing.py，对齐 LightX2V
dynamic_sparse_attn / get_block_map）+ 预编译 sm70 keep-or-drop kernel
（与 SolAttn 同一 pyd，CSR 格式兼容）。

机制：
  1. pooled 块均值打分 qc·kc^T（对齐 SLA 训练路由）
  2. 每行保留分数最高 top-K 块（sparsity_ratio=0.85 → 15% 密度，对齐官方 config）
  3. H3 前缀 sink 保底（text/cond/ref/audio，h3_prefix_tokens）
  4. 稀疏块用 keep-or-drop kernel 计算（V100 sm70 特供）

参数语义与 SolAttn 节点对齐（start_percent/end_percent/min_tokens/dense_blocks/
h3_prefix_tokens 完全相同），便于已有 SolAttn 工作流迁移。
"""
import logging
import math
import sys
from pathlib import Path

import torch

LOGGER = logging.getLogger("SLAV100")
SPARSE_KEY = "sla_sparse"
START_PERCENT_KEY = "sla_start_percent"
END_PERCENT_KEY = "sla_end_percent"
MIN_TOKENS_KEY = "sla_min_tokens"
DENSE_BLOCKS_KEY = "sla_dense_blocks"
PREFIX_TOKENS_KEY = "sla_prefix_tokens"
SPARSITY_KEY = "sla_sparsity_ratio"
_BLOCK_INDEX_HOOKED = set()
_REPORTED_STATS = set()


def _load_extension():
    """加载预编译 keep-or-drop sparse kernel（与 SolAttn 同一库，幂等）。

    Windows 匹配 .pyd，Linux 匹配 .so（见 nodes.py 的 _kernel_suffix 约定）。
    """
    try:
        _ = torch.ops.comfy_v100_solattn_cuda.varlen_fwd_sparse
        return                                          # 已被本插件/其他节点加载
    except (AttributeError, RuntimeError):
        pass
    pyd_dir = Path(__file__).resolve().parent
    ext = ".so" if sys.platform.startswith("linux") else ".pyd"
    candidates = sorted(pyd_dir.glob(f"comfy_v100_solattn_cuda*{ext}"))
    if not candidates:
        raise RuntimeError(
            "SLAV100: 缺少预编译 kernel comfy_v100_solattn_cuda*.pyd/.so（应随插件分发）"
        )
    torch.ops.load_library(str(candidates[0]))


def _load_fp16safe_nodes():
    """内嵌 FP16Safe 实现模块（fp16safe.py，自包含）。"""
    from . import fp16safe
    return fp16safe


class _quiet_print:
    """临时抑制 print 输出（内嵌 FP16Safe 的日志噪声）。"""

    def __enter__(self):
        import builtins
        self._orig = builtins.print
        builtins.print = lambda *a, **k: None

    def __exit__(self, *exc):
        import builtins
        builtins.print = self._orig
        return False


def parse_blocks(spec, count):
    """解析块规格为绝对索引集合（kijai 同款语法）。"""
    if not spec:
        return frozenset()
    out = set()
    for part in spec.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        if part.startswith("-") and part.count("-") == 1:
            out.add(int(part) + count)
        elif "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo < 0:
                lo += count
            if hi < 0:
                hi += count
            out.update(range(lo, hi + 1))
        else:
            idx = int(part)
            if idx < 0:
                idx += count
            out.add(idx)
    return frozenset(out)


def _sparse_attn(q, k, v, heads, scale, opts):
    """SLA topk 块稀疏路径。q,k,v: (B,H,S,D) fp16"""
    from . import sla_routing
    batch = q.shape[0]
    if batch != 1:
        return None
    q3 = q.squeeze(0).permute(1, 0, 2)                    # [S,H,D] 非连续（kernel 支持）
    k3 = k.squeeze(0).permute(1, 0, 2)
    v3 = v.squeeze(0).permute(1, 0, 2)
    S, H, D = q3.shape
    if S < 64 or H != heads:
        return None
    try:
        _ = torch.ops.comfy_v100_solattn_cuda.varlen_fwd_sparse
    except (AttributeError, RuntimeError):
        return None                                       # pyd 未加载（异常分发）
    prefix_tokens = int(opts.get(PREFIX_TOKENS_KEY, 0))
    sparsity_ratio = float(opts.get(SPARSITY_KEY, 0.85))
    topk_ratio = max(0.01, min(0.99, 1.0 - sparsity_ratio))
    # 音频段保底：forward 桥接 hook 把 PackedLayout 的 audio 段注入
    # transformer_options["sla_layout"]；音频段强制不稀疏（H3 音频质量修复）
    layout_info = opts.get("sla_layout") or {}
    audio_range = None
    audio = layout_info.get("audio")
    if audio is not None and audio[1] > audio[0]:
        audio_range = (int(audio[0]), int(audio[1]))
    cnt, off, ccnt, cidx = sla_routing.build_sparse_csr_topk(
        q3, k3, topk_ratio=topk_ratio, scale=scale, sink_tokens=prefix_tokens,
        audio_range=audio_range)
    if S not in _REPORTED_STATS:
        _REPORTED_STATS.add(S)
        dens = float(cnt.float().mean().item()) / max((S + 63) // 64, 1)
        a_info = (f"audio_rows={audio_range[1] - audio_range[0]}"
                  if audio_range else "audio=UNKNOWN")
        print(f"[SLAV100][v0.2] S={S} blocks={math.ceil(S / 64)} 密度≈{dens * 100:.1f}% "
              f"（topk_ratio={topk_ratio:.2f}）| prefix_tokens={prefix_tokens} | "
              f"{a_info}（保底）", flush=True)
    softmax_scale = float(scale if scale is not None else 1.0 / math.sqrt(D))
    cu = torch.tensor([0, S], dtype=torch.int32, device=q.device)
    out, _lse = torch.ops.comfy_v100_solattn_cuda.varlen_fwd_sparse(
        q3, k3, v3, None, cu, cu, cnt, off, ccnt, cidx, S, S, softmax_scale)
    return out.reshape(1, S, H * D)                       # [B,S,H*D]（与 comfy_attention 同布局）


def _unsupported_reasons(q, k, v, heads, mask, skip_reshape, skip_output_reshape, opts):
    """返回不走稀疏的原因列表（非空 = 回退原版 SDPA）。"""
    reasons = []
    if mask is not None:
        reasons.append("mask")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        reasons.append("dtype")
    S = q.shape[-2] if q.dim() >= 3 else 0
    min_tokens = int(opts.get(MIN_TOKENS_KEY, 1024))
    if S < min_tokens:
        reasons.append("min-tokens")
    start_percent = float(opts.get(START_PERCENT_KEY, 0.2))
    end_percent = float(opts.get(END_PERCENT_KEY, 1.0))
    step = opts.get("step")
    total = opts.get("total_steps")
    if step is not None and total:
        frac = int(step) / max(int(total) - 1, 1)
        if frac < start_percent or frac > end_percent:
            reasons.append("sampling-window")
    else:
        sigmas = opts.get("sigmas")
        if sigmas is not None and len(sigmas) > 0 and float(sigmas[0]) > 14.0:
            reasons.append("sigma-warmup")
    dense_blocks = opts.get(DENSE_BLOCKS_KEY) or frozenset()
    block = opts.get("sla_block")
    if block is not None and block in dense_blocks:
        reasons.append("dense-block")
    return reasons


def _install_block_index(model):
    """发布当前 block index 到 transformer_options（dense_blocks 判断用）。"""
    blocks = getattr(model, "blocks", None)
    if blocks is None or id(model) in _BLOCK_INDEX_HOOKED:
        return
    for index, block in enumerate(blocks):
        def make_hook(index):
            def hook(_module, _args, kwargs):
                options = kwargs.get("transformer_options")
                if isinstance(options, dict):
                    options["sla_block"] = index
                return None
            return hook
        block.register_forward_pre_hook(make_hook(index), with_kwargs=True)
    _BLOCK_INDEX_HOOKED.add(id(model))


def sla_attention_override(original, q, k, v, heads, mask=None, attn_precision=None,
                           skip_reshape=False, skip_output_reshape=False, **kwargs):
    """attention override: 满足条件走 SLA topk 稀疏，否则原版 SDPA 兜底。"""
    opts = kwargs.get("transformer_options") or {}
    if opts.get(SPARSE_KEY, False):
        reasons = _unsupported_reasons(q, k, v, heads, mask, skip_reshape,
                                       skip_output_reshape, opts)
        if not reasons:
            try:
                out_s = _sparse_attn(q, k, v, heads, kwargs.get("scale"), opts)
                if out_s is not None:
                    return out_s
            except Exception as exc:
                LOGGER.warning("[SLAV100] sparse fallback: %s", exc)
    return original(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                    skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
                    **kwargs)


sla_attention_override._sla_override = True


def _install_layout_hook(diffusion_model):
    """桥接 minimax_payload -> transformer_options（幂等）。

    H3 的 PackedLayout（含音频段行范围）经 minimax_payload 传给 model forward
    （extra_conds），**不经过 transformer_options**——attention override 拿不到。
    这里包一层 diffusion_model.forward：把 layout.segments 里的 audio 段
    (start, end) 行范围注入 transformer_options["sla_layout"]，供 override
    做音频段保底（SLA topk 稀疏会砍掉音频行的 key 块 → 音频质量崩，实测
    <500Hz 能量 34% vs dense 7%，开头/结尾最严重）。
    """
    if getattr(diffusion_model, "_sla_layout_hooked", False):
        return
    import types
    orig_forward = diffusion_model.forward

    def wrapped(self, x, timestep, context, transformer_options=None,
                minimax_payload=None, **kwargs):
        if transformer_options is None:
            transformer_options = {}
        payload = minimax_payload or {}
        layout = payload.get("layout")
        if layout is not None and isinstance(transformer_options, dict):
            segs = getattr(layout, "segments", None)
            if segs:
                audio = next(((a, b) for a, b, k in segs if k == "audio"), None)
                video = next(((a, b) for a, b, k in segs if k == "video"), None)
                transformer_options["sla_layout"] = {
                    "seq_len": int(getattr(layout, "seq_len", 0)),
                    "audio": (int(audio[0]), int(audio[1])) if audio else None,
                    "video": (int(video[0]), int(video[1])) if video else None,
                }
        return orig_forward(x, timestep, context,
                            transformer_options=transformer_options,
                            minimax_payload=minimax_payload, **kwargs)

    diffusion_model.forward = types.MethodType(wrapped, diffusion_model)
    diffusion_model._sla_layout_hooked = True


class SLAV100:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "fp16_safe": ("BOOLEAN", {"default": True,
                                          "tooltip": "内嵌 FP16Safe（prescale /16 + 熔断 + fp32 重跑兜底）"}),
                "sparsity_ratio": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 0.99, "step": 0.01,
                                             "tooltip": "SLA 稀疏率：跳过的 key 块比例。0.85 = lightx2v 官方蒸馏值（每行保留 15% 块）；更高更快但可能掉质量"}),
                "start_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "采样进度低于此比例走 dense（turbo 4-step 下第一步自然 dense）"}),
                "end_percent": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "采样进度高于此比例走 dense（尾部保真）"}),
                "min_tokens": ("INT", {"default": 1024, "min": 64, "max": 65536, "step": 64,
                                       "tooltip": "序列短于此 token 数不走稀疏"}),
                "dense_blocks": ("STRING", {"default": "0-1,-1",
                                            "tooltip": "保留 dense 的 transformer 块，如 '0-1'=前两层，'0-2,-1'=前三层+最后一层；空=全部稀疏"}),
                "h3_prefix_tokens": ("INT", {"default": 1024, "min": 0, "max": 65536, "step": 64,
                                             "tooltip": "KV sink 保底：保护序列开头 token 不被稀疏。保持默认 1024 即可，与序列长度 S 无关；勿设为 ≈S（会破坏路由，实测导致末尾帧崩坏）"}),
            },
            "optional": {
                "debug_nan": ("BOOLEAN", {"default": False}),
                "profile": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "sol_attn"

    DESCRIPTION = (
        "SLA Top-K (V100): lightx2v Minimax-h3-Turbo-SLA LoRA 的稀疏加速节点。"
        "LoRA 用 LoraLoader 加载（标准 rank-128 LoRA），本节点提供 SLA topk 块稀疏"
        "（每行保留 top-K 块，85% 稀疏对齐官方）+ sm70 keep-or-drop kernel 加速。"
        "与 SolAttn 节点互斥（同一 MODEL 只用一个）。"
    )

    def patch(self, model, fp16_safe=True, sparsity_ratio=0.85, start_percent=0.2,
              end_percent=0.9, min_tokens=1024, dense_blocks="0-1,-1",
              h3_prefix_tokens=1024, debug_nan=False, profile=False):
        if not torch.cuda.is_available() or not any(
            torch.cuda.get_device_capability(index) == (7, 0)
            for index in range(torch.cuda.device_count())
        ):
            raise RuntimeError("SLAV100 requires an SM70/V100 CUDA device.")
        if fp16_safe:
            fp16safe_mod = _load_fp16safe_nodes()
            with _quiet_print():
                model, = fp16safe_mod.MiniMaxH3FP16Safe().patch(
                    model, debug_nan=bool(debug_nan), profile=bool(profile)
                )
            LOGGER.info("[SLAV100] FP16Safe embedded")
        _load_extension()
        patched = model.clone()
        transformer_options = patched.model_options.setdefault("transformer_options", {})
        existing = transformer_options.get("optimized_attention_override")
        if existing is not None and not getattr(existing, "_sla_override", False):
            raise RuntimeError("SLAV100: another optimized_attention_override is already active on this MODEL.")
        transformer_options["optimized_attention_override"] = sla_attention_override
        transformer_options[SPARSE_KEY] = True
        transformer_options[SPARSITY_KEY] = float(sparsity_ratio)
        transformer_options[START_PERCENT_KEY] = float(start_percent)
        transformer_options[END_PERCENT_KEY] = float(end_percent)
        transformer_options[MIN_TOKENS_KEY] = int(min_tokens)
        diffusion = model.model.diffusion_model
        n_blocks = len(getattr(diffusion, "blocks", None) or ())
        transformer_options[DENSE_BLOCKS_KEY] = parse_blocks(dense_blocks, n_blocks)
        transformer_options[PREFIX_TOKENS_KEY] = int(h3_prefix_tokens)
        _install_block_index(diffusion)
        _install_layout_hook(diffusion)                 # 音频段保底：桥接 layout -> options
        LOGGER.info("[SLAV100][v0.2] active: fp16_safe=%s sparsity=%.2f window=[%.2f,%.2f] "
                    "min_tokens=%d dense_blocks=%s prefix=%d audio_guard=auto",
                    bool(fp16_safe), float(sparsity_ratio), float(start_percent),
                    float(end_percent), int(min_tokens), dense_blocks or "(none)",
                    int(h3_prefix_tokens))
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "SLAV100": SLAV100,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SLAV100": "SLA Top-K (V100)",
}
