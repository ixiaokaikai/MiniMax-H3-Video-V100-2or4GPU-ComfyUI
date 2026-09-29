"""Explicit capability matrix for the isolated H3 outpaint execution route."""
from __future__ import annotations

from .video_outpaint_identity import native_stock_model_identity


OUTPAINT_COMBINATION_POLICY = {
    "native_stock_20_steps": "supported",
    "pinned_kj_low_vram_attention": "supported_memory_optimization_not_speed_claim",
    "pinned_kj_chunk_ffn": "supported_memory_optimization_not_speed_claim",
    "regional_routing_plus_pinned_kj": "supported_contract_human_quality_review_pending",
    "dlss_nr_after_completed_compose": "supported_independent_postprocess",
    "turbo_lora": "unsupported_no_verified_identity_or_outpaint_acceptance",
    "speed": "unsupported_no_verified_composition_adapter",
    "sla_attention": "unsupported_competing_attention_owner",
    "vdn": "unsupported_different_model_architecture_and_execution_contract",
    "fast_h3": "unsupported_no_verified_composition_adapter",
    "prompt_relay": "unsupported_competing_diffusion_wrapper_and_attention_owner",
    "unknown_lora_or_wrapper": "unsupported_fail_closed",
}


def outpaint_model_compatibility_report(model):
    try:
        identity = native_stock_model_identity(model)
    except (ValueError, RuntimeError) as exc:
        return {
            "schema": "t8.h3.video_outpaint.compatibility/v1",
            "ready": False,
            "status": "unsupported_fail_closed",
            "reason": str(exc),
            "model_filename_used_as_evidence": False,
            "policy": OUTPAINT_COMBINATION_POLICY,
        }
    return {
        "schema": "t8.h3.video_outpaint.compatibility/v1",
        "ready": True,
        "status": "verified_execution_identity",
        "identity": identity,
        "model_filename_used_as_evidence": False,
        "quality_or_vram_acceptance_implied": False,
        "policy": OUTPAINT_COMBINATION_POLICY,
    }
