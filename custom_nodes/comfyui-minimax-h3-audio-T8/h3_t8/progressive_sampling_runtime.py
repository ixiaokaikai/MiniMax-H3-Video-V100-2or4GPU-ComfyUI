"""T8 learned progressive first sampling on native H3 FLOW_AV/Euler.

Uses Comfy's sampler and the existing T8 learned resizer. Not SelfLift-zero/rich,
not a completed-first-pass restart, and no external SelfLift code is imported.
All execution wrappers belong to a disposable MODEL clone, never global state.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .progressive_sampling_contract import EvaluationLedger, euler_sampler_space_step, plan_progressive_first_sample


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lifter_identity(model_name, precision):
    import folder_paths
    from safetensors import safe_open
    from .learned_latent_upscale_advanced import PRECISIONS, EXPECTED_STATE_CONTRACT
    if not isinstance(model_name, str) or not model_name.strip() or model_name == "none":
        raise ValueError("Select an installed H3 learned upscaler; no nearest fallback")
    if precision not in PRECISIONS:
        raise ValueError("Unsupported learned-upscaler precision")
    path = Path(folder_paths.get_full_path_or_raise("latent_upscale_models", model_name)).resolve()
    # Check architecture before any low-stage sampling, without allocating weights.
    # The folder also contains non-H3 models, which must fail before spending GPU time.
    try:
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            if set(checkpoint.keys()) != set(EXPECTED_STATE_CONTRACT):
                raise ValueError("expected the 322-tensor H3 3D learned-resizer architecture")
            for key, shape in EXPECTED_STATE_CONTRACT.items():
                tensor = checkpoint.get_slice(key)
                if tuple(tensor.get_shape()) != shape or tensor.get_dtype() not in {"F16", "BF16", "F32"}:
                    raise ValueError(f"incompatible learned-resizer tensor: {key}")
    except Exception as error:
        raise ValueError(f"Not a compatible H3 learned upscaler: {path.name}: {error}") from error
    return {"name": model_name, "path": str(path), "sha256": _sha256_file(path), "precision": precision,
            "header_tensor_count": len(EXPECTED_STATE_CONTRACT)}


def _empty_option(value):
    """Do not coerce unknown tensors/callables to bool while checking patches."""
    return value is None or value is False or (isinstance(value, (dict, list, tuple)) and not value)


def validate_native_model(model, sampler):
    from .tst_model import detach_tst_model
    model, _ = detach_tst_model(model)
    import comfy.k_diffusion.sampling
    import comfy.latent_formats
    import comfy.model_base
    import comfy.model_sampling
    import comfy.samplers
    from .h3_core_compat import plain_attention_backend
    from .progressive_attention import inspect_progressive_attention
    av_type = getattr(comfy.model_sampling, "ModelSamplingAV", None)
    if av_type is None or type(getattr(model, "model", None)) is not comfy.model_base.MiniMaxH3:
        raise ValueError("Progressive first sampling requires current Core native H3 FLOW_AV; old routes remain unchanged")
    sampling = model.get_model_object("model_sampling")
    if not isinstance(sampling, (av_type,)) or not isinstance(sampling, comfy.model_sampling.CONST):
        raise ValueError("Configure native H3 Euler/AV sampling, not dual_clock_euler")
    if getattr(sampling, "multiplier", None) != 1000:
        raise ValueError("Native H3 timestep multiplier must be 1000")
    if getattr(sampling.noise_scaling, "__func__", None) is not comfy.model_sampling.CONST.noise_scaling:
        raise ValueError("Custom noise scaling is not qualified")
    if getattr(sampling.inverse_noise_scaling, "__func__", None) is not comfy.model_sampling.CONST.inverse_noise_scaling:
        raise ValueError("Custom restart scaling is not qualified")
    if type(model.get_model_object("latent_format")) is not comfy.latent_formats.MiniMaxH3AV:
        raise ValueError("Native H3 AV latent format is required")
    if not math.isfinite(float(sampling.audio_scale)) or float(sampling.audio_scale) <= 0:
        raise ValueError("Invalid native audio scale")
    noise_scale = getattr(sampling, "noise_scale", 1.0)
    if isinstance(noise_scale, bool) or not isinstance(noise_scale, (int, float)) or not math.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("Native noise scale must be finite and positive")
    memory, selected_backend, _ = inspect_progressive_attention(model)
    allowed_objects = {'model_sampling'} | (set(memory['methods']) if memory else set())
    if set(getattr(model, "object_patches", {})) - allowed_objects:
        raise ValueError("Unqualified MODEL object patches; only native model_sampling is supported")
    if not isinstance(sampler, comfy.samplers.KSAMPLER) or sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler:
        raise ValueError("Only native Euler is qualified for this first-sampling route")
    if sampler.inpaint_options or any(k != "s_churn" or v != 0 for k, v in sampler.extra_options.items()):
        raise ValueError("Custom Euler options are not qualified")
    for name in ("wrappers", "callbacks", "injections", "hook_patches", "additional_models"):
        if any(bool(value) for value in getattr(model, name, {}).values()):
            raise ValueError(f"Progressive first sampling does not yet compose MODEL {name}")
    for name in ("t8_minimax_h3_openvdn_contract_v2", "t8_fast_h3_vsa_gate_contract_v1"):
        if getattr(model, "get_attachment", lambda key: None)(name) is not None:
            raise ValueError("Keep VDN/FastH3 on their existing independent workflows")
    options = model.model_options
    if any(not _empty_option(value) for key, value in options.items() if key != "transformer_options"):
        raise ValueError("A sampler/model wrapper already owns this MODEL branch")
    transformer = options.get("transformer_options", {})
    shifts = {"minimax_h3_sigma_shift_video": ("sigma_shift_video", sampling.shift),
              "minimax_h3_sigma_shift_audio": ("sigma_shift_audio", sampling.audio_shift)}
    for key, (attribute, expected) in shifts.items():
        value = transformer.get(key, getattr(model.model.diffusion_model, attribute, None))
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value <= 0 or value != expected):
            raise ValueError(f"Native AV shift mismatch: {key}")
    for key, value in transformer.items():
        if key in shifts or _empty_option(value):
            continue
        if key == "optimized_attention_override" and plain_attention_backend(value) is not None:
            continue
        if key == 'optimized_attention_override' and selected_backend is not None:
            continue
        if memory is not None and key in {'minimax_head_chunks', 'sol_take_forward'}:
            continue
        raise ValueError(f"Unqualified transformer option: {key}; use a separate native MODEL branch")
    return sampling


def prepare_stage_conditioning(conditioning, plan, *, positive, guide_resize='legacy_bilinear'):
    if guide_resize not in {'legacy_bilinear', 'preserve_mean'}:
        raise ValueError('Unknown guide_resize policy')
    if not isinstance(conditioning, list) or len(conditioning) != 1:
        raise ValueError("Initial progressive scope requires one conditioning entry")
    item = conditioning[0]
    if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[1], dict):
        raise ValueError("Invalid CONDITIONING")
    embedding, metadata = item
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 3 or embedding.shape[0] != 1:
        raise ValueError("Expected batch-1 text conditioning")
    allowed = {"pooled_output", "guidance", "minimax_keyframes", "minimax_frame_count", "minimax_token_tags"}
    if set(metadata) - allowed:
        raise ValueError(f"Reference/area/hook conditioning not yet qualified: {sorted(set(metadata) - allowed)}")
    # Native Qwen H3 uses one tag per embedding token: text=1, vision=0.
    # These select AdaLN modality rows, not a spatial mask. Both stages use the
    # same text/vision embeddings, so preserve the tags exactly (including int64).
    tags = metadata.get("minimax_token_tags")
    if tags is not None:
        if (not isinstance(tags, torch.Tensor) or tags.dtype != torch.long or
                tuple(tags.shape) not in {(embedding.shape[1],), (1, embedding.shape[1])} or
                not bool(((tags == 0) | (tags == 1)).all())):
            raise ValueError("minimax_token_tags must be batch-1 int64 text/vision tags matching the embedding length")
    high_meta = metadata.copy()
    low_meta = metadata.copy()
    keyframes = metadata.get("minimax_keyframes", [])
    if not isinstance(keyframes, list):
        raise ValueError("minimax_keyframes must be a list")
    if plan.task == "t2va" and keyframes:
        raise ValueError("T2VA cannot contain keyframe conditioning")
    if plan.task == "i2va" and positive and not keyframes:
        raise ValueError("I2VA requires a first-frame conditioning reference")
    if keyframes:
        if len(keyframes) != 1 or not isinstance(keyframes[0], dict) or set(keyframes[0]) != {"resolved_frame_index", "latent"}:
            raise ValueError("Initial I2VA supports exactly one native first-frame reference")
        frame = keyframes[0]
        if type(frame["resolved_frame_index"]) is not int or frame["resolved_frame_index"] != 0:
            raise ValueError("Only first-frame I2VA is qualified")
        reference = frame["latent"]
        expected = (1, 24, 1, plan.target_height // 16, plan.target_width // 16)
        if not isinstance(reference, torch.Tensor) or tuple(reference.shape) != expected:
            raise ValueError(f"Reference latent must match target canvas: {expected}")
        if not reference.is_floating_point() or not bool(torch.isfinite(reference).all()):
            raise ValueError("Reference latent must be finite floating point")
        original_frame = reference[:, :, 0].float()
        resized = F.interpolate(original_frame, size=(plan.low_height // 16, plan.low_width // 16),
                                mode="bilinear", align_corners=False)
        if guide_resize == 'preserve_mean':
            resized = resized - resized.mean((-2, -1), keepdim=True) + original_frame.mean((-2, -1), keepdim=True)
        resized = resized.to(reference.dtype).unsqueeze(2)
        high_meta["minimax_keyframes"] = [frame.copy()]
        low_meta["minimax_keyframes"] = [{**frame, "latent": resized}]
        expected_frames = 5 + (plan.video_shape[2] - 2) // 5 * 17
        if metadata.get("minimax_frame_count") != expected_frames:
            raise ValueError("Keyframe frame-count metadata does not match the latent")
    return [[embedding, low_meta]], [[embedding, high_meta]]


def _lift_video(video_vae_space, audio_placeholder, plan, identity):
    import comfy.nested_tensor
    from .learned_latent_upscale_advanced import learned_upscale_h3_av_latent
    latent = {"samples": comfy.nested_tensor.NestedTensor((video_vae_space, audio_placeholder))}
    output, width, height, receipt = learned_upscale_h3_av_latent(
        latent, identity["name"], "target_dimensions", 2., 0.5,
        plan.target_width, plan.target_height, "honor_dimensions_exp", 1.1,
        identity["precision"], "offload_after")
    report = json.loads(receipt)
    if (width, height) != (plan.target_width, plan.target_height) or report.get("status") != "ok":
        raise RuntimeError("Learned lift did not execute the planned target canvas")
    if report.get("model", {}).get("sha256") != identity["sha256"]:
        raise RuntimeError("Learned weights changed or cached identity is stale")
    if _sha256_file(identity["path"]) != identity["sha256"]:
        raise RuntimeError("Learned checkpoint changed during sampling")
    return output["samples"].unbind()[0], report


def _resource_snapshot(device, reserve_bytes):
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()
    result = {"kind": "boundary_snapshot_not_peak", "device": str(device)}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        result.update(free_bytes=int(free), total_bytes=int(total),
                      torch_allocated_bytes=torch.cuda.memory_allocated(device))
        if free < reserve_bytes:
            raise RuntimeError("Insufficient free GPU memory at progressive stage boundary; no automatic retry")
    return result


def _native_stage(model, sampler, sigmas, latent, noise, positive, negative, cfg, seed, callback,
                  denoise_mask=None):
    import comfy.samplers
    error = None
    try:
        return comfy.samplers.sample(model, noise, positive, negative, cfg, model.load_device,
                                      sampler, sigmas, model.model_options, latent_image=latent,
                                      callback=callback, disable_pbar=True, seed=seed, denoise_mask=denoise_mask)
    except BaseException as caught:
        error = caught
        raise
    finally:
        # Core retains the loaded MODEL after sampling. Our disposable stage
        # clone shares its module objects with the caller; retaining composed
        # KJ/motion methods would leak stage ownership into identity/reuse.
        # Restore only this stage's installed objects, keeping weight residency
        # and LoRA backups. Core partially_load reapplies object patches on the
        # next managed load, including when weights are already resident.
        try:
            _restore_stage_objects(model)
        except BaseException as cleanup_error:
            if error is None:
                raise
            message = f'Progressive stage object cleanup failed: {cleanup_error}'
            if hasattr(error, 'add_note'):
                error.add_note(message)
            else:
                import logging
                logging.warning(message)


def _restore_stage_objects(model):
    import comfy.utils
    backups = model.object_patches_backup
    if not backups:
        return
    if any(path not in model.object_patches
           or comfy.utils.get_attr(model.model, path) is not model.object_patches[path]
           for path in backups):
        raise RuntimeError('Progressive stage cannot restore objects owned by another MODEL')
    model.unpatch_model(unpatch_weights=False)


@torch.inference_mode()
def sample_progressive_h3(model, positive, negative, av_latent, sampler, sigmas, *,
                          upscaler_model, seed, cfg=1., low_evaluations=6,
                          low_scale=0.5, task="t2va", precision="fp16",
                          reserve_vram_mib=1024, callback=None, model_hires=None,
                          guide_resize='legacy_bilinear', eav_mode='disabled', eav_tau=4.,
                          eav_start_video_progress=0., eav_end_video_progress=1.,
                          eav_max_workspace_mib=32, eav_g_hard_limit=1.5,
                          input_mode='empty', continuation=None, checkpoint=None, producers=None,
                          tst_mode='disabled', tst_tau=.2, tst_max_workspace_mib=256):
    """Execute learned-only native AV stages, returning LATENT and a report.

    The report covers this sampler node, not text/VAE/output end-to-end time.
    There is no automatic fallback, pixel-anchor path, tiling or attention patch.
    """
    import comfy.model_management as mm
    import comfy.nested_tensor
    import comfy.patcher_extension
    import comfy.sample
    from .core import nested_av_parts
    from .learned_latent_upscale_advanced import learned_upscale_geometry
    from .progressive_stage_models import validate_stage_pair
    from .progressive_attention import backend_phase_report, backend_snapshot, prepare_progressive_attention
    from .progressive_relay import detach_relay_input, prepare_relay_stage, strip_paired_conditioning
    from .progressive_eav import audit_progressive_eav_stage, prepare_progressive_eav, summarize_progressive_eav
    from .progressive_sampling_contract import plan_progressive_initialized_sample
    from .progressive_masking import normalize_av_masks, resize_video_source, sampler_with_clean_anchor
    started = time.perf_counter()
    producer_identity = None
    if producers is not None:
        from .progressive_producers import verify_producers
        producer_identity = verify_producers(producers)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be unsigned 64-bit")
    if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(cfg) or not 0 <= cfg <= 100:
        raise ValueError("cfg must be finite and between 0 and 100")
    if type(reserve_vram_mib) is not int or reserve_vram_mib < 512:
        raise ValueError("GPU reserve must be an integer of at least 512 MiB")
    if eav_mode not in {'disabled', 'report_only', 'apply_exp'}:
        raise ValueError('Unknown progressive EAV mode')
    eav_enabled = eav_mode != 'disabled'
    if tst_mode not in {'disabled', 'report_only', 'apply_exp'}:
        raise ValueError('Unknown progressive TST mode')
    tst_enabled = tst_mode != 'disabled'
    if tst_enabled and cfg != 1.:
        raise ValueError('Progressive TST requires native CFG1 for the exact forward/layer clock')
    if input_mode not in {'empty', 'initialized_av_exp'}:
        raise ValueError('Unknown progressive input_mode')
    initialized = input_mode == 'initialized_av_exp'
    if continuation is not None:
        from .progressive_continuation_runtime import PreparedProgressiveContinuation
        if type(continuation) is not PreparedProgressiveContinuation:
            raise ValueError('Unknown progressive continuation contract')
        if not initialized or cfg != 1. or guide_resize != 'legacy_bilinear':
            raise ValueError('Continuation requires native initialized CFG1 without guide override')
        continuation.verify(positive=positive, av_latent=av_latent)
    if eav_enabled and cfg != 1.:
        raise ValueError('Progressive EAV requires CFG1 for native batch-1 routing')
    if not isinstance(av_latent, dict) or "samples" not in av_latent:
        raise ValueError("Expected an initial AV LATENT dictionary with samples")
    if set(av_latent) - {"samples", "noise_mask"}:
        raise ValueError("Unqualified initial latent metadata")
    video, audio = nested_av_parts(av_latent)
    if initialized:
        plan = plan_progressive_initialized_sample(video, audio, sigmas, low_evaluations=low_evaluations,
                                                    low_scale=low_scale, task=task)
        high_mask = normalize_av_masks(av_latent.get('noise_mask'), video, audio)
        # The crop is a shape/finite-value view only, never a LOW source image.
        # Bind this exact normalized mask for both the EAV guard and sampling.
        if continuation is None:
            low_mask = normalize_av_masks(av_latent.get('noise_mask'),
                video[..., :plan.low_height // 16, :plan.low_width // 16], audio)
        else:
            continuation.verify(plan=plan)
            # LOW has its own reference-guided template. HIGH known-prefix
            # masks must never be resized into LOW's generation region.
            low_mask = normalize_av_masks(continuation.low[1].get('noise_mask'),
                                          *continuation.low[1]['samples'].unbind())
    else:
        plan = plan_progressive_first_sample(video, audio, sigmas, low_evaluations=low_evaluations,
                                             low_scale=low_scale, task=task, noise_mask=av_latent.get("noise_mask"))
        high_mask = None
        low_mask = None
    lift_geometry = learned_upscale_geometry(plan.low_width // 16, plan.low_height // 16,
        "target_dimensions", 2., .5, plan.target_width, plan.target_height, "honor_dimensions_exp", 1.1)
    schedule = sigmas.detach().clone()
    source_model = model
    from .tst_model import detach_tst_model
    tst_inputs = [model] + ([model_hires] if model_hires is not None and model_hires is not model else [])
    model, low_tst_spec = detach_tst_model(model)
    if model_hires is None or model_hires is source_model:
        model_hires_clean, high_tst_spec = model, low_tst_spec
    else:
        model_hires_clean, high_tst_spec = detach_tst_model(model_hires)
    if low_tst_spec is not None or high_tst_spec is not None:
        if tst_enabled:
            raise ValueError('Choose TST MODEL nodes or sampler TST options, not both')
        if cfg != 1.:
            raise ValueError('Progressive TST MODEL nodes require CFG1')
        for spec in (low_tst_spec, high_tst_spec):
            if spec is not None and tuple(spec['full_sigmas']) != plan.sigmas:
                raise ValueError('TST MODEL full schedule differs from progressive SIGMAS')
        tst_enabled = True
        tst_specs = {'low': low_tst_spec, 'high': high_tst_spec}
    else:
        spec = dict(mode=tst_mode, tau=tst_tau, workspace=tst_max_workspace_mib) if tst_enabled else None
        tst_specs = {'low': spec, 'high': spec}
    # Relay checkpoint identity uses its actual unwrapped inputs; TST config is
    # separately bound below and the original MODEL owners are verified again.
    source_model = model
    source_positive, source_negative = positive, negative
    model, relay_contract = detach_relay_input(model)
    if model_hires_clean is source_model:
        high_model, high_relay_contract = model, relay_contract
    else:
        high_model, high_relay_contract = detach_relay_input(model_hires_clean)
    if relay_contract is None and high_relay_contract is not None:
        raise ValueError('Progressive Relay requires paired low MODEL/CONDITIONING inputs')
    if relay_contract is not None:
        if continuation is not None:
            raise ValueError('Continuation requires a dedicated projected Relay composer')
        if cfg != 1.:
            raise ValueError('Progressive Relay currently requires CFG1 for native batch-1 routing')
        if high_relay_contract is not None and high_relay_contract['binding'] != relay_contract['binding']:
            raise ValueError('Progressive low/high Relay bindings differ')
        positive = strip_paired_conditioning(positive, relay_contract, required=True)
        negative = strip_paired_conditioning(negative, relay_contract, required=False)
    sampling = validate_native_model(model, sampler)
    high_sampling = sampling if high_model is model else validate_native_model(high_model, sampler)
    stage_models = validate_stage_pair(model, high_model, sampling, high_sampling)
    stage_models['separate_input'] = model_hires is not None
    stage_models['shared_base_object'] = model.model is high_model.model
    tst_runtimes, tst_reports = {}, {}
    if tst_enabled:
        from .tst_runtime import TSTQueryRuntime, TST_RUNTIME_KEY
        for phase, stage_model, start, end, width, height in (
                ('low', model, 0, plan.low_evaluations, plan.low_width, plan.low_height),
                ('high', high_model, plan.low_evaluations, plan.total_evaluations, plan.target_width, plan.target_height)):
            spec = tst_specs[phase]
            if spec is None:
                tst_reports[phase] = {'mode': 'disabled', 'identity': True}
                continue
            tst_runtimes[phase] = TSTQueryRuntime(plan.sigmas, stage_start=start, stage_end=end,
                layer_count=len(stage_model.model.diffusion_model.blocks), frames=plan.video_shape[2],
                spatial_tokens=(height // 32) * (width // 32), mode=spec['mode'], tau=spec['tau'],
                max_workspace_mib=spec['workspace'])
    checkpoint_lifter = None
    if checkpoint is not None:
        from .progressive_checkpoint import ProgressiveCheckpointSession
        if type(checkpoint) is not ProgressiveCheckpointSession or cfg != 1.:
            raise ValueError('Checkpoint requires its exclusive native CFG1 session')
        relay_binding = None
        if relay_contract is not None:
            from .progressive_relay_checkpoint import ProgressiveRelayCheckpointBinding
            relay_binding = ProgressiveRelayCheckpointBinding(source_model,
                model_hires_clean,
                source_positive, source_negative, sampler)
        checkpoint_lifter = _lifter_identity(upscaler_model, precision)
        checkpoint.bind(model, high_model, sampler, plan,
            inputs={'positive': positive, 'negative': negative, 'av_latent': av_latent,
                    'prepared': continuation.identity if continuation is not None else None},
            settings=dict(seed=seed, cfg=cfg, guide_resize=guide_resize, input_mode=input_mode,
                eav_mode=eav_mode, eav_tau=eav_tau, eav_start=eav_start_video_progress,
                eav_end=eav_end_video_progress, eav_workspace=eav_max_workspace_mib,
                eav_hard_limit=eav_g_hard_limit, reserve_vram_mib=reserve_vram_mib,
                **({'tst': {phase: owner.config for phase, owner in tst_runtimes.items()}} if tst_enabled else {})),
            lifter=checkpoint_lifter, continuation=continuation, producers=producers, relay_binding=relay_binding)
    if continuation is None:
        low_positive, high_positive = prepare_stage_conditioning(positive, plan, positive=True, guide_resize=guide_resize)
        low_negative, high_negative = prepare_stage_conditioning(negative, plan, positive=False, guide_resize=guide_resize)
    else:
        low_positive, high_positive = continuation.low[0], continuation.high[0]
        low_negative, high_negative = low_positive, high_positive  # explicit CFG1
        model, high_model = continuation.phase_models(model, high_model)
    ledger = EvaluationLedger(plan)
    relay_reports = None
    if continuation is not None and continuation.relay is not None:
        from .progressive_continuation_relay import prepare_continuation_relay_stage
        branch, low_positive, low_negative, low_backend, low_relay_report = prepare_continuation_relay_stage(
            model, continuation, low=True)
        high_branch, high_positive, high_negative, high_backend, high_relay_report = prepare_continuation_relay_stage(
            high_model, continuation, low=False)
        relay_reports = {'low': low_relay_report, 'high': high_relay_report}
    elif relay_contract is not None:
        # Check the original target before spending GPU time on the low stage.
        high_branch, high_positive, high_negative, high_backend, high_relay_report = prepare_relay_stage(
            high_model, relay_contract, high_positive, high_negative, plan, low=False,
            backend_contract=high_relay_contract)
        branch, low_positive, low_negative, low_backend, low_relay_report = prepare_relay_stage(
            model, relay_contract, low_positive, low_negative, plan, low=True,
            backend_contract=relay_contract)
        relay_reports = {'low': low_relay_report, 'high': high_relay_report}
    elif eav_enabled:
        # EAV builds its own authenticated backend owner; do not stack the
        # standalone progressive attention wrapper underneath it.
        branch, high_branch = model.clone(), high_model.clone()
        low_backend = high_backend = None
    else:
        branch, low_backend = prepare_progressive_attention(model, tst_enabled='low' in tst_runtimes)
        if high_model is model and bool(tst_specs['low']) == bool(tst_specs['high']):
            high_branch, high_backend = branch, low_backend
        else:
            high_branch, high_backend = prepare_progressive_attention(high_model, tst_enabled='high' in tst_runtimes)
    eav_runtimes = {}
    eav_reports = {}
    if eav_enabled:
        low_mask_contract = high_mask_contract = None
        if initialized:
            from .progressive_eav_masks import NativeProgressiveMaskContract
            low_mask_contract = NativeProgressiveMaskContract(branch, low_mask,
                (*video.shape[:-2], plan.low_height // 16, plan.low_width // 16), audio.shape, continuation=continuation)
            high_mask_contract = NativeProgressiveMaskContract(high_branch, high_mask, video.shape, audio.shape,
                                                                continuation=continuation)
        eav_options = dict(mode=eav_mode, tau=eav_tau, start=eav_start_video_progress,
                           end=eav_end_video_progress, workspace=eav_max_workspace_mib,
                           hard_limit=eav_g_hard_limit)
        branch, eav_runtimes['low'] = prepare_progressive_eav(
            branch, schedule, plan, relay_report=relay_reports['low'] if relay_reports else None,
            mask_contract=low_mask_contract, **eav_options)
        high_branch, eav_runtimes['high'] = prepare_progressive_eav(
            high_branch, schedule, plan, relay_report=relay_reports['high'] if relay_reports else None,
            mask_contract=high_mask_contract, **eav_options)
    identity = checkpoint_lifter if checkpoint_lifter is not None else _lifter_identity(upscaler_model, precision)
    owned_branches = [branch] if high_branch is branch else [branch, high_branch]
    device = torch.device(branch.load_device)
    intermediate = mm.intermediate_device()
    snapshots = []
    active_stage = "low"
    branch_evaluations = {"low": 0, "high": 0}
    apply_calls = {"low": 0, "high": 0}
    network_shapes = {}
    boundary = {}
    timings = {}
    attention_reports = {}
    checkpoint_report = {'enabled': checkpoint is not None, 'reused_low': False}

    def checked_resources():
        for tst_input in tst_inputs:
            detach_tst_model(tst_input)
        active_device = device if active_stage == 'low' else torch.device(high_branch.load_device)
        snapshots.append(_resource_snapshot(active_device, reserve_vram_mib * 1024**2))

    def measured_forward(apply_model, arguments):
        apply_calls[active_stage] += 1
        branch_evaluations[active_stage] += len(arguments.get("cond_or_uncond", ()))
        return apply_model(arguments["input"], arguments["timestep"], **arguments["c"])

    def measured_network(executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        # The native BaseModel boundary unpacks these shapes and invokes H3
        # once. Keeping observation outside the DiT leaves its sole owner slot
        # available for Relay/EAV; do not weaken their inner-wrapper checks.
        shapes = kwargs.get('latent_shapes')
        if not isinstance(shapes, (list, tuple)) or len(shapes) != 2:
            raise RuntimeError('Progressive observer requires native packed AV shapes')
        if active_stage in tst_runtimes:
            if not isinstance(t, torch.Tensor) or t.numel() != 1 or not bool(torch.isfinite(t).all()):
                raise RuntimeError('TST requires a single finite actual native video sigma')
            options = transformer_options or {}
            if TST_RUNTIME_KEY in options:
                raise RuntimeError('TST refuses an existing or foreign query owner')
            with tst_runtimes[active_stage].forward(float(t.detach().cpu().item())):
                result = executor(x, t, c_concat, c_crossattn, control,
                    {**options, TST_RUNTIME_KEY: tst_runtimes[active_stage]}, **kwargs)
        else:
            result = executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        ledger.record(active_stage, forward=True)
        network_shapes.setdefault(active_stage, [list(shape) for shape in shapes])
        return result

    def progress(step, prediction, state, total):
        expected = ledger.expected[active_stage]
        if step != ledger.callbacks[active_stage] or total != expected:
            raise RuntimeError("Sampler callbacks no longer match the qualified stage contract")
        ledger.record(active_stage)
        checked_resources()
        if active_stage == "low" and step + 1 == plan.low_evaluations:
            predicted_video, predicted_audio = prediction.unbind()
            _, audio_state = state.unbind()
            boundary["clean_video"] = predicted_video.detach().to(device=intermediate, dtype=torch.float32, copy=True)
            boundary["audio_next"] = euler_sampler_space_step(audio_state, predicted_audio,
                                                               plan.prediction_sigma, plan.resume_sigma).to(intermediate)
        if callback is not None:
            offset = 0 if active_stage == "low" else plan.low_evaluations
            callback(step + offset, prediction, state, plan.total_evaluations)

    wrapper_kind = comfy.patcher_extension.WrappersMP.APPLY_MODEL
    wrapper_key = "t8_progressive_forward_counter_v1"
    for owned in owned_branches:
        owned.set_model_unet_function_wrapper(measured_forward)
        owned.add_wrapper_with_key(wrapper_kind, wrapper_key, measured_network)
    try:
        checked_resources()
        if continuation is not None:
            low_video = continuation.low[1]['samples'].unbind()[0].to(intermediate)
        else:
            low_video = (resize_video_source(video, plan.low_height // 16, plan.low_width // 16).to(intermediate)
                         if initialized else torch.zeros(
                             (*video.shape[:-2], plan.low_height // 16, plan.low_width // 16),
                             dtype=video.dtype, device=intermediate))
        low_template = comfy.nested_tensor.NestedTensor((low_video, audio))
        # Native CPU noise layout/seed; distinct low/high shapes are recorded.
        low_noise = comfy.sample.prepare_noise(low_template, seed)
        anchor_audio_noise = low_noise.unbind()[1] if high_mask is not None else None
        attention_before = backend_snapshot(low_backend)
        start = time.perf_counter()
        restored = checkpoint.load_low() if checkpoint is not None else None
        if restored is None:
            _native_stage(branch, sampler, schedule[:plan.low_evaluations + 1], low_template,
                          low_noise, low_positive, low_negative, cfg, seed, progress, denoise_mask=low_mask)
        else:
            tensors, historical, receipt = restored
            boundary.update({key: value.to(intermediate) for key, value in tensors.items()})
            # Counts describe THIS execution. Do not forge callbacks/model
            # forwards for the already completed LOW stage loaded from disk.
            ledger.expected['low'] = 0
            checkpoint_report.update(reused_low=True, receipt=receipt, historical_low_report=historical)
        timings["low_sampling_including_model_prepare"] = time.perf_counter() - start
        attention_reports['low'] = backend_phase_report(low_backend, attention_before)
        if 'low' in tst_runtimes:
            if restored is None:
                tst_reports['low'] = tst_runtimes['low'].snapshot(require_complete=True)
            else:
                previous = historical.get('tst')
                if not isinstance(previous, dict) or not previous.get('completed'):
                    raise RuntimeError('Restored LOW lacks completed TST execution evidence')
                tst_reports['low'] = {**previous, 'execution_scope': 'restored_low_not_current_forwards'}
        if eav_enabled:
            if restored is None:
                eav_reports['low'] = audit_progressive_eav_stage(eav_runtimes['low'], plan, 'low', branch)
            else:
                eav_reports['low'] = {**historical['eav'], 'execution_scope': 'restored_low_not_current_forwards'}
            if restored is None and low_backend is None and 'composed_attention_backend' in eav_runtimes['low'].config:
                attention_reports['low'] = {**eav_runtimes['low'].config['composed_attention_backend'],
                                            'counter_scope': 'this_phase_completed_calls_only'}
        if checkpoint is not None and restored is None:
            checkpoint_report['receipt'] = checkpoint.save_low(boundary,
                dict(callbacks=ledger.callbacks['low'], actual_forwards=ledger.forwards['low'],
                     eav=eav_reports.get('low'), attention=attention_reports['low'],
                     relay=relay_reports['low'] if relay_reports else None,
                     **({'tst': tst_reports['low']} if tst_enabled else {})))
        del low_template, low_noise, low_video, low_positive, low_negative, low_mask
        if ledger.callbacks["low"] != ledger.expected['low'] or set(boundary) != {"clean_video", "audio_next"}:
            raise RuntimeError("Low stage ended without a complete boundary prediction")
        checked_resources()
        start = time.perf_counter()
        latent_format = model.get_model_object("latent_format")
        clean_vae = latent_format.process_out(boundary.pop("clean_video"))
        lifted_vae, lift_report = _lift_video(clean_vae, torch.zeros_like(audio), plan, identity)
        del clean_vae
        if tuple(lifted_vae.shape) != tuple(video.shape) or not bool(torch.isfinite(lifted_vae).all()):
            raise RuntimeError("Learned lift produced an invalid video tensor")
        clean_high = latent_format.process_in(lifted_vae.float())
        del lifted_vae
        high_seed = (seed + 1) % 2**64
        high_noise = comfy.sample.prepare_noise(clean_high, high_seed).to(clean_high)
        sigma = schedule[plan.low_evaluations].to(clean_high)
        high_state = sampling.noise_scaling(sigma, high_noise, clean_high)
        high_sampler = sampler
        if high_mask is not None:
            # Clean target-space source is NOT the lifted LOW prediction or the
            # noisy HIGH restart. Native H3 handles all mask/time/audio math.
            high_sampler = sampler_with_clean_anchor(sampler,
                comfy.nested_tensor.NestedTensor((video, audio)),
                comfy.nested_tensor.NestedTensor((high_noise.to(intermediate), anchor_audio_noise)))
        del anchor_audio_noise
        del clean_high, high_noise
        # Encode states so Core's zero-noise restart reconstructs both marginals.
        restart_video = latent_format.process_out(sampling.inverse_noise_scaling(sigma, high_state))
        del high_state
        audio_next = boundary.pop("audio_next")
        restart_audio = sampling.inverse_noise_scaling(sigma.to(audio_next), audio_next) / float(sampling.audio_scale)
        del audio_next
        restart = comfy.nested_tensor.NestedTensor((restart_video.to(intermediate), restart_audio.to(intermediate)))
        del restart_video, restart_audio
        restart_noise = comfy.nested_tensor.NestedTensor([torch.zeros_like(part) for part in restart.unbind()])
        timings["transition_including_lift_and_reload_policy"] = time.perf_counter() - start
        active_stage = "high"
        checked_resources()
        attention_before = backend_snapshot(high_backend)
        start = time.perf_counter()
        result = _native_stage(high_branch, high_sampler, schedule[plan.low_evaluations:], restart,
                               restart_noise, high_positive, high_negative, cfg, seed, progress,
                               denoise_mask=high_mask)
        timings["high_sampling_including_model_prepare"] = time.perf_counter() - start
        attention_reports['high'] = backend_phase_report(high_backend, attention_before)
        if 'high' in tst_runtimes:
            tst_reports['high'] = tst_runtimes['high'].snapshot(require_complete=True)
        if eav_enabled:
            eav_reports['high'] = audit_progressive_eav_stage(eav_runtimes['high'], plan, 'high', high_branch)
            if high_backend is None and 'composed_attention_backend' in eav_runtimes['high'].config:
                attention_reports['high'] = {**eav_runtimes['high'].config['composed_attention_backend'],
                                             'counter_scope': 'this_phase_completed_calls_only'}
        output_video, output_audio = nested_av_parts({"samples": result})
        if tuple(output_video.shape) != tuple(video.shape) or tuple(output_audio.shape) != tuple(audio.shape):
            raise RuntimeError("Final AV shapes differ from the plan")
        if not all(bool(torch.isfinite(value).all()) for value in (output_video, output_audio)):
            raise RuntimeError("Final samples contain NaN or Inf")
        counts = ledger.finish()
        if counts["actual_forwards"] != apply_calls:
            raise RuntimeError("Native H3 network execution count differs from apply_model calls")
        if relay_reports is not None:
            for phase, phase_model in (('low', branch), ('high', high_branch)):
                expected = {'forward': ledger.expected[phase],
                            'routed_attention': ledger.expected[phase] * len(phase_model.model.diffusion_model.blocks)}
                if relay_reports[phase]['completed_calls'] != expected:
                    raise RuntimeError(f'Progressive {phase} Relay did not execute every native H3 block')
                relay_reports[phase]['status'] = ('restored_low_no_current_stage_routing'
                    if phase == 'low' and checkpoint_report['reused_low']
                    else 'verified_native_stage_routing_quality_unverified')
        counts.update(apply_model_calls=apply_calls, forward_boundary="completed_native_h3_apply_model_executor")
        continuation_report = continuation.verify() if continuation is not None else None
        if checkpoint is not None:
            checkpoint.verify()
        report = {**plan.report(), "status": "sampled_quality_unverified", "runtime_identity": "native_h3_euler_av",
                  "sampler_seconds": time.perf_counter() - started, "timings": timings,
                  "timing_scope": "sampler_node_not_end_to_end_no_forced_sync", "counts": counts,
                  "cfg_branch_evaluations": branch_evaluations, "upscaler": identity,
                  "stage_models": stage_models,
                  "initialization": {'mode': input_mode, 'mask_present': high_mask is not None,
                      'start_sigma': plan.sigmas[0], 'unmasked_source_erased_at_sigma1': plan.sigmas[0] == 1,
                      'known_region_policy': 'native_clean_source_anchor_not_noisy_restart' if high_mask is not None
                                             else 'no_known_region_lock',
                      'continuation_and_resume_qualified': False},
                  "prompt_relay": relay_reports,
                  "eav": {**eav_reports, 'summary': summarize_progressive_eav(eav_reports, plan)} if eav_enabled
                         else {'mode': 'disabled', 'identity': True},
                  "tst": tst_reports if tst_enabled else {'mode': 'disabled', 'identity': True},
                  "attention": {**attention_reports, 'shared_counter': False,
                                'shared_backend_instance': high_backend is low_backend and low_backend is not None},
                  "lift_geometry": lift_geometry,
                  "network_input_shapes": network_shapes,
                  "lift_report": lift_report, "resource_observations": snapshots,
                  "noise": {"low_seed": seed, "high_seed": high_seed,
                            "low_video_shape": [*video.shape[:-2], plan.low_height // 16, plan.low_width // 16],
                            "audio_shape": list(audio.shape), "high_video_shape": list(video.shape),
                            "policy": "core_prepare_noise_cpu_low_joint_high_video_seed_plus_one"},
                  "reference_policy": ('first_frame_latent_mean_preserving_low_original_high' if guide_resize == 'preserve_mean'
                                       else 'first_frame_latent_bilinear_low_original_high'),
                  "pixel_anchor": False, "highres_tiling": False,
                  "existing_vdn_two_pass_modified": False}
        if checkpoint is not None:
            checkpoint_report['reused_evaluations'] = plan.low_evaluations if restored is not None else 0
            checkpoint_report['scope'] = 'native_low_boundary_only_not_complete_chain_resume'
            report['checkpoint'] = checkpoint_report
        if producers is not None:
            if verify_producers(producers) != producer_identity:
                raise ValueError('Progressive producer binding changed during sampling')
            report['producers'] = producer_identity
        if continuation_report is not None:
            report['continuation'] = continuation_report
            report['reference_policy'] = 'accepted_rgb_low_completed_av_high_native_prefix'
        return {"samples": result.to(intermediate)}, json.dumps(report, ensure_ascii=False, allow_nan=False)
    finally:
        for owned in owned_branches:
            owned.model_options.pop("model_function_wrapper", None)
            owned.remove_wrappers_with_key(wrapper_kind, wrapper_key)
        boundary.clear()
