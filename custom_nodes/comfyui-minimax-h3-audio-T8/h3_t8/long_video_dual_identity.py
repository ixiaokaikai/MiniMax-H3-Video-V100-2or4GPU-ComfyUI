"""Content identity for native H3 dual-stage inputs, never a filename claim."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import MethodType

import torch

from .h3_core_compat import plain_attention_backend
from .long_video_in_node_loop_advanced import _sha256_json, _check_interrupted
from .relay_kj_memory import inspect_memory_composition
from .relay_sol_backend import capture_composed_backend
from .runtime_precision_identity import matmul_precision_identity
from .video_outpaint_identity import value_identity


def content_identity(value):
    if isinstance(value, torch.Tensor) and str(value.dtype).startswith("torch.float8_"):
        if value.device.type == "meta" or value.layout != torch.strided or not value.is_contiguous():
            raise ValueError("Float8 identity requires a materialized contiguous tensor")
        digest = hashlib.sha256()
        flat = value.detach().reshape(-1)
        # CPU isfinite does not implement all float8 formats. Widen only a
        # bounded1MiB chunk for the check, hash original bytes (no dequantize).
        for start in range(0, flat.numel(), 1024 * 1024):
            _check_interrupted()
            chunk = flat[start:start + 1024 * 1024].cpu()
            if not torch.isfinite(chunk.float()).all():
                raise ValueError("execution tensor has nonfinite values")
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
        return {"tensor_sha256": digest.hexdigest(), "dtype": str(value.dtype), "shape": list(value.shape)}
    # New Core stores LoRA descriptors as adapter instances, older Core as
    # tuples. Inspect only this exact inert weight container, never arbitrary
    # object repr, callback code, subclass or adapter forward hooks.
    try:
        from comfy.weight_adapter.lora import LoRAAdapter
    except ImportError:
        LoRAAdapter = None
    if LoRAAdapter is not None and type(value) is LoRAAdapter:
        if set(vars(value)) != {"loaded_keys", "weights"} or not all(
                type(key) is str for key in value.loaded_keys):
            raise ValueError("LoRAAdapter contains runtime mutations outside its weight descriptor")
        return {"adapter": "comfy.LoRAAdapter", "implementation": _implementation(LoRAAdapter),
                "loaded_keys": sorted(value.loaded_keys), "weights": content_identity(value.weights)}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "items": [content_identity(item) for item in value]}
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: content_identity(item) for key, item in sorted(value.items())}
    return value_identity(value, interrupt_check=_check_interrupted)


def _implementation(function):
    path = inspect.getsourcefile(function)
    if path is None:
        raise ValueError("Stage implementation source is unavailable")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_model_identity(model):
    from comfy.model_base import MiniMaxH3
    if not isinstance(model.model, MiniMaxH3):
        raise ValueError("Dual-model native4+4 loop requires native H3 MODELs; VDN uses a separate8+4 contract")
    # New algorithms need a deliberate identity and owner adapter. A plausible
    # string representation or UUID does not prove byte-identical execution.
    for name in ("weight_wrapper_patches", "additional_models", "wrappers", "callbacks", "injections",
                 "hook_patches", "forced_hooks", "current_hooks"):
        if getattr(model, name, None):
            raise ValueError(f"Dual-stage content identity does not yet cover MODEL {name}")
    attachments = {key: value for key, value in getattr(model, "attachments", {}).items() if value is not None}
    lora_metadata = attachments.pop("t8_h3_lora_metadata", None)
    if lora_metadata is not None and (type(lora_metadata) is not dict or
            not all(type(key) is str and type(value) is str for key, value in lora_metadata.items())):
        raise ValueError("H3 LoRA metadata must be the loader's plain safetensors string map")
    if attachments:
        raise ValueError("Dual-stage MODEL has unknown attachments; identity adapter required")
    memory = inspect_memory_composition(model)
    allowed = {} if memory is None else {key[:-8]: method for key, method in memory["methods"].items()}
    if set(model.object_patches) != (set() if memory is None else set(memory["methods"])):
        raise ValueError("Dual-stage MODEL has object patches outside the audited memory route")
    options = dict(model.model_options.get("transformer_options", {}))
    override = options.pop("optimized_attention_override", None)
    backend = capture_composed_backend(override)
    if backend is not None:
        backend_contract = backend.report()
        backend_contract.pop("completed_calls", None)
    elif override is not None:
        plain = plain_attention_backend(override)
        if plain is None:
            raise ValueError("Dual-stage MODEL contains an unrecognized attention owner")
        backend_contract = {"kind": "core_plain", "name": plain, "implementation": _implementation(override)}
    else:
        from comfy.ldm.modules import attention
        backend_contract = {"kind": "core_global", "name": attention.optimized_attention.__name__,
                            "implementation": _implementation(attention.optimized_attention)}
    if "sol_take_forward" in options:
        # inspect_memory_composition already authenticates this exact callable.
        if memory is None:
            raise ValueError("Sol forward delegate without its verified KJ memory owner")
        options.pop("sol_take_forward")
    configuration = {key: value for key, value in model.model_options.items() if key != "transformer_options"}
    configuration["transformer_options"] = options
    implementations = {}
    classes = []
    for name, module in model.model.named_modules():
        if module._forward_pre_hooks or module._forward_hooks:
            raise ValueError("Dual-stage MODEL contains shared live hooks; no cross-branch identity proof")
        current = vars(module).get("forward")
        expected = allowed.get(name)
        if current is not None and not (
            isinstance(current, MethodType) and current.__self__ is module
            and (current.__func__ is type(module).forward or
                 (expected is not None and current.__func__ is expected.__func__))
        ):
            raise ValueError("Dual-stage model forward was replaced outside the selected MODEL")
        cls = type(module)
        if cls not in implementations:
            implementations[cls] = _implementation(cls)
        classes.append((name, cls.__module__ + "." + cls.__qualname__))
    state = model.model_state_dict()
    if not state:
        raise ValueError("Dual-stage MODEL has no loaded tensor state")
    # Core's backup is the original pre-LoRA tensor for an already-loaded
    # branch. Use it instead of hashing the currently applied branch's weights.
    for key, backup in getattr(model, "backup", {}).items():
        if key not in state or not isinstance(backup.weight, torch.Tensor):
            raise ValueError("Cannot reconstruct original MODEL identity from this backup format")
        state[key] = backup.weight
    data = {"schema": "t8.h3.dual_stage_model/v1", "classes": classes,
            "implementations": sorted(implementations.values()), "state": content_identity(state),
            "patches": content_identity(model.patches), "model_options": content_identity(configuration),
            "lora_metadata": content_identity(lora_metadata),
            "backend": backend_contract,
            "memory": None if memory is None else {key: memory[key] for key in
                ("kind", "head_chunks", "ffn_settings", "source_sha256s")},
            "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "matmul_precision": matmul_precision_identity(),
                        "deterministic": torch.are_deterministic_algorithms_enabled()}}
    config = getattr(model.model, "model_config", None)
    data["unet_config"] = content_identity(getattr(config, "unet_config", None))
    data["manual_cast_dtype"] = str(getattr(model.model, "manual_cast_dtype", None))
    return {"sha256": _sha256_json(data), "schema": data["schema"], "backend": backend_contract,
            "memory": data["memory"], "model_filename_trusted": False,
            "tensor_count": len(state), "lora_target_count": len(model.patches)}
