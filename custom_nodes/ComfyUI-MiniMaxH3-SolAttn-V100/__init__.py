# -*- coding: utf-8 -*-
"""Sol-Attn (V100) — MiniMax H3 在 V100 上的注意力稀疏加速（ComfyUI 自定义节点）。

含两个独立节点（互斥使用，同一 MODEL 只挂一个）：
- SolAttnV100：Sol-Attn 阈值路由（keep-or-drop，默认）
- SLAV100：SLA topk 块稀疏（配合 lightx2v Minimax-h3-Turbo-SLA LoRA）
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:                                    # SLA 节点独立文件；加载失败不影响 SolAttn 本体
    from .sla_nodes import NODE_CLASS_MAPPINGS as _SLA_MAPPINGS
    from .sla_nodes import NODE_DISPLAY_NAME_MAPPINGS as _SLA_DISPLAY
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_SLA_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_SLA_DISPLAY}
except Exception:                       # noqa: BLE001 — SLA 节点为实验特性，缺失不阻断
    pass

__version__ = "1.2.1"
VERSION_TAG = "[SolAttn-V100][V1.2.1]"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__", "VERSION_TAG"]
