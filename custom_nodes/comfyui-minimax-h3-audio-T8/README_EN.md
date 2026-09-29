# MiniMax H3 Audio T8

**v1.81.0** (2026-09-15) adds two independent H3 low-memory EXP nodes without a KJNodes dependency: head-grouped attention with early intermediate release, and conditional packed-token SwiGLU chunking. In one fixed three-second run, the default `head_chunks=4 + chunks=2` reduced observed peak use by 306.73 MiB (2.06%) while adding about 2.92% wall time; this is not a universal memory or speed claim. New Speech Studio reference-voice graphs conservatively trim low-energy alignment padding while preserving explicit settings in old workflows, and the Sol verification probe now follows the current `sink_blocks` interface. See the [release notes](docs/RELEASE_1.81.0.md).

**v1.80.0** (2026-09-14) adds SelfLift LOW4 → learned latent upscale → HIGH4, independent LOW/HIGH MODEL branches, TST, and the in-node long-video composition, together with nine bound-sample-reviewed SelfLift/TaoMate EXP workflows. EAV now defaults to the reviewed `tau=8` over 15%–90%; users may raise it gradually for stronger motion but must recheck identity, deformation, flicker, and audio. Qualification is limited to the bound short samples and is not a universal speed, VRAM, or quality claim. See the [release notes](docs/RELEASE_1.80.0.md).

**v1.79.6** (2026-09-14) closes the official non-Starlight Topaz scope: all 14 regular and 5 interpolation dropdown models completed real official production-worker runs. It adds an explicit 10-bit SDR to HEVC Main10 path, GPU-index selection, and a wired upscale-then-interpolate workflow, while fixing lossless-master output auditing. HDR, VFR, interlaced and rotated inputs are still refused rather than silently normalized. Commercial weights remain local and are not shipped. See the [release notes](docs/RELEASE_1.79.6.md).

**v1.79.5** (2026-09-14) hardens Topaz delivery: manual controls are filtered by each model definition; official network templates prevent false missing-weight errors for Rhea, Nyx XL and Theia; audio is preflighted into MP4 or MKV before GPU work; unqualified 10-bit/HDR inputs cannot be silently reduced to 8-bit; and ComfyUI receives frame/audit progress. Apollo completed a real short mechanical run, with visual quality still pending. See the [release notes](docs/RELEASE_1.79.5.md).

**v1.79.4** (2026-09-14) fixes the regular Topaz route that produced a 12.9GB lossless MOV from a 15-second clip and then exhausted the disk through a redundant SaveVideo. The node now directly saves a high-quality H.264 NVENC MP4 by default; lossless is an explicit audit-only profile. It adds model-use guidance, auto/manual parameters with visible controls, and a separate official `tvai_fi` 2x/4x interpolation node/workflow. Topaz need not remain open, but it must be licensed and the selected model must already be downloaded. See the [release notes](docs/RELEASE_1.79.4.md).

GitHub main update for 2026-09-13: the specific LTX clip, recovered five-second Tao dialogue clip, and latest [eight-second Dance accepted-picture LOW-context contrast](docs/DANCE_ACCEPTED_PICTURE_20260913.md) passed full human review. The accepted Dance workflow preserves 4+4, HIGH and original music; do not regenerate accepted clips. Older Dance and both [depth experiments](docs/DEPTH_REFERENCE_EXP.md) remain failed and are not shipped as recommended workflows. Tao's original finalization guard failure is retained; accepted media does not qualify fresh packaged-worker end-to-end GPU reliability. See the [original seam safeguard](docs/DUAL_MODEL_SEAM_FIX_20260913.md) and [resource, AdaLN, HJL and audio boundaries](docs/LOCAL_RELIABILITY_SCOPE_20260913.md). No universal quality or duration claim.

New in **v1.79.0**: [dual-model4+learned-upscale+4 loops](docs/DUAL_MODEL_LONG_VIDEO_EXP.md), [official regular Topaz upscaling](docs/TOPAZ_EXP.md), and [R1 reliability fixes](docs/R1_RELIABILITY_20260911.md). New templates default to two segments/eight seconds; the reviewed sample was accepted for visuals, audio, lip-sync and seams. Existing workflows remain unchanged. OpenVDN no longer rejects an upstream Sol attention override by presence alone; VDN still uses its own attention math, with no combined acceleration claim. Starlight remains paused and is not a qualified feature. See [release notes](docs/RELEASE_1.79.0.md).

New in 1.77.0: independent EXP workflows for [progressive T2VA/I2VA sampling](examples/workflows/28-progressive-sampling) and [DLSS 2x frame interpolation](examples/workflows/29-dlss-fi). Progressive sampling reduced end-to-end time by about 37–38% in the tested three-second warm runs. Reviewed visuals were accepted as comparable, and scored portrait dialogue/audio/lip-sync cases passed; game audio differences remain, and the 32-second route has not passed. Frame interpolation doubles FPS while preserving duration, resolution and the original audio; it is not upscaling or H3 generation acceleration. Separately supplied DLSSG runtime files and dependencies are required. Nothing is downloaded or installed automatically, and existing workflows are unchanged. See the [release notes](docs/RELEASE_1.77.0.md).

[简体中文](README.md) | English

Implementation files now live under `h3_t8/` to keep the repository homepage short. Workflows remain in `examples/workflows/`; model locations, node IDs, parameters and links are unchanged. The three existing TRT command-line entrypoints still work from the root. See [repository layout](docs/REPOSITORY_LAYOUT.md) for developer details.

MiniMax H3 Audio T8 is a ComfyUI node pack for joint video and audio generation. It includes practical workflows for text and image animation, first/last-frame control, image/video/audio references, long video, lip sync, acceleration, and final-video restoration.

Current version: **1.81.0** · 336 nodes · GPL-3.0-or-later

Version 1.78.1 organizes implementation code under `h3_t8/` for a shorter repository homepage. Existing workflows, model locations, node parameters and the three root TRT commands are unchanged. No model downloads are needed. See the [layout and update guide](docs/REPOSITORY_LAYOUT.md).

New in 1.78.0: an optional [TRT VAE backend](docs/TRT_VAE_EXP.md) and [five workflows](examples/workflows/30-trt-vae), including local compilation, decoder-only, Full encode/decode and one-sampler same-latent comparison. The combined short-video and static-text review was accepted overall; this remains EXP, without long-video qualification. A separate TensorRT environment and locally compiled engines are required. Existing workflows, generation weights and audio VAE are unchanged.

**No total output-stage speed benefit was measured:** native 30.54s versus TRT 31.33s including loading, decoding and saving, excluding sampling. Faster bare decoding is not a promise of faster complete generation. See [1.78.0 release notes](docs/RELEASE_1.78.0.md).

The previous release's video outpainting and 12 EXP workflows remain available, with joint decoding by default. Results vary by source: some sections can still show strips, repeated textures or visible seams. Try a short candidate first. This is not a seamless-quality guarantee, and the remaining defects have not been proven to be solely model limitations.

## Where to start

If this is your first time using the pack:

1. Run a basic H3 workflow from [`examples/workflows/01-basic-generation`](examples/workflows/01-basic-generation).
2. For image or audio control, use the workflows under [`02-audio-control`](examples/workflows/02-audio-control) and the matching reference folders.
3. For OpenVDN eight-step generation, download [`t8star/Vdn-Minimax-H3-Comfy`](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy), then use an `OpenVDN_DMD8_*_Advanced.json` workflow from [`10-speed`](examples/workflows/10-speed).
4. For WASD character and camera control, download [`t8star/Minimax-H3-World-Comfy`](https://huggingface.co/t8star/Minimax-H3-World-Comfy), then use [`26-h3-world`](examples/workflows/26-h3-world); the first release is fixed to first-frame I2VA.
5. For long video or music video work, start with [`04-long-video`](examples/workflows/04-long-video) or [`24-mv-lipsync`](examples/workflows/24-mv-lipsync).
6. For image or finished-video upscaling, use [`25-dlss-nr`](examples/workflows/25-dlss-nr), an optional Windows RTX post-process.
7. To repair a short broken-face interval without repainting the whole clip, use a 2026-09-05 Window workflow from [`06-face-refine`](examples/workflows/06-face-refine).
8. To extend an existing video's canvas, use [`27-video-outpaint`](examples/workflows/27-video-outpaint). Preview the geometry first, then generate, review, confirm, and continue a candidate.

Advanced workflows include notes on the canvas. Replace the model and input media before running them. Avoid stacking several LoRAs, attention backends, or sampler owners unless the workflow explicitly asks for it.

Three dual-model 4+4 long-video examples now expose an optional external LoRA per stage, using
either Core PyTorch, KJ H3 Sage, or the audited `ComfyUI-sol-attn / SolAttentionPatch` backend.
Each stage must remain independent: `base → this stage's Turbo LoRA → this stage's optional
external LoRA → exactly one attention backend → the matching MODEL input`. External LoRA is
`disabled` by default. Put compatible adapters under `models/loras` and use this project's H3
compatibility loader, not the generic `LoraLoaderModelOnly`. Do not stack Sage, the PyTorch
selector, and Sol. See the [`04-long-video`](examples/workflows/04-long-video) README and the
[accelerator wiring guide](examples/workflows/04-long-video/DUAL_MODEL_ACCELERATOR_CONNECTIONS.md).

## What it can do

### H3 generation and references

- Text-to-video with audio (T2VA)
- First-frame image-to-video (I2VA)
- Last-frame generation (L2VA)
- First-and-last-frame generation (FL2VA)
- Single-image and multi-image Ref2VA
- Video, audio, and mixed references
- Native joint video/audio decoding and saving
- H3-World first-frame I2VA with WASD and IJKL/F character/camera timelines

### Audio and lip sync

- Source-audio locking, voice references, dialogue, and final-track mixing
- New Speech Studio reference-voice graphs default to `auto_reference_voice`, which applies conservative low-energy boundary trimming only to reference voices; described voices and explicit settings in saved workflows are unchanged
- Vocal Lock: drive H3 with isolated vocals and mix the full song only once at the end
- Audio Refine routes for Turbo, PDD, EAV, Prompt Relay, and long video
- Local ASR, speaker, and SyncNet verification tools

The project's 32-second Vocal Lock V3 sample completed five serial H3 shots, per-shot lip-sync checks, and a full human review. The reviewer accepted that exact video. This is evidence for the reviewed sample, not a guarantee for every face, song, or prompt.

### Long video

- Multi-keyframe planning and segmented continuation
- One-queue serial rendering
- Resume support and accepted manifests
- Native Masked Context Plan B
- Optional Color Match, enabled by default, to reduce seam color changes
- Dual-model seam work remains unqualified: the failed post-sampling latent bridge has been removed from delivery and retained only for diagnostic reproduction. The original single jump remains unresolved. An unlocked coarse pass has NOT been proven causal; that claim is retracted. Both22/39-frame contexts are supported. CPU tests and endpoint scores cannot replace full-video review.

### Video outpainting

- Extend any side manually, or derive the canvas from a target aspect ratio and anchor
- New nodes default to `joint_decode`: decode the whole canvas together to avoid contour breaks caused by hard source pasteback. The source area also passes through VAE reconstruction, so its details may change; source pixels are not guaranteed to remain identical
- Optional `preserve_source` pastes the exact source pixels back before lossy encoding, but the boundary may still show a seam. Both modes retain existing source audio
- Generate only the first window for review and require explicit confirmation before continuing. Continuation and saving retain the candidate's mode; older candidates without a mode record keep source-preserving behavior
- Define shot cuts and per-shot prompts, and optionally route regional prompts. Default-on Color Match affects the extended area only in `preserve_source` and resets at cuts; `joint_decode` bypasses source-edge color and geometry correction
- Resume completed windows after cancellation or restart only when source, model, and runtime identity checks pass. Core, KJ, or node-code updates must not bypass these checks
- Use a conservative all-intra H.264 output for decoder reliability
- Optionally run the completed video through the separate DLSS-NR 2x workflow, with a new visual review afterward

No outpainting-specific checkpoint or converted weight is required. The workflows use the existing H3 FL2VA model, Qwen3-VL encoder, video VAE, and audio VAE. Generation also requires a separate [`ComfyUI-KJNodes`](https://github.com/kijai/ComfyUI-KJNodes) install; the tested route uses Stock20 with its H3 low-memory Attention and FFN patches. Turbo, SPEED, SLA, OpenVDN, FastH3, and similar sampler/attention owners cannot currently be stacked with this route. A complete 32-second output was generated and original-audio preservation verified, but visible defects remain in some expanded regions. This release is authorized with those known limitations, not an all-material quality pass. See the [outpainting workflow guide](examples/workflows/27-video-outpaint/README.md).

### Acceleration and finishing

- Two independent H3 low-memory EXP nodes: `Low VRAM Attention` defaults to four head groups with early intermediate release; `Chunk FeedForward` defaults to two SwiGLU token chunks only above 4096 packed tokens. They do not require KJNodes and can be used alone or in either order; see [H3 memory-node guide](docs/H3_MEMORY_NODES_EXP.md) for boundaries

- OpenVDN DMD eight-step and Stage B 50-step execution
- PDD, SLA, SPEED, FastH3 VSA, and Enhance-A-Video integrations
- Optional built-in `beta57` schedule in the stable dual-clock node, with no RES4LYF install required; `native_flow` remains the default
- DLSS-NR image/short-frame/file-video upscaling, plus FlashVSR, RealBasicVSR, RAFT, and Skin Finish
- Face Refine Window: render only selected continuous bad-face intervals, then accept or reject them manually
- Experimental two-stage MiniMax H3 plus LTX-2.5 upscaling

Features labeled `Advanced` or `EXP` should be used through their bundled workflows. Acceleration methods are usually alternatives, not add-ons to be stacked together.

## Installation

### ComfyUI Manager

Search for `MiniMax H3 Audio T8`, install it, then fully restart ComfyUI.

### Manual installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git minimax-h3-audio-T8
```

This node pack follows recent native MiniMax H3 APIs in ComfyUI. If every node turns red or appears missing, update the ComfyUI core, frontend, and Manager together, then exit and restart the application.

## Model folders

| Model | ComfyUI folder |
| --- | --- |
| H3 diffusion model | `models/diffusion_models` |
| Qwen3-VL text encoder | `models/text_encoders` |
| Video and audio VAEs | `models/vae` |
| Turbo, PDD, SLA, and other LoRAs | `models/loras` |
| Complete OpenVDN model bundle | [`t8star/Vdn-Minimax-H3-Comfy`](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy); already arranged under the correct `models` folders |
| H3-World action LoRA | [`t8star/Minimax-H3-World-Comfy`](https://huggingface.co/t8star/Minimax-H3-World-Comfy), arranged for `models/loras/minimax/H3-World`; original source: [`DANNY621/H3-World`](https://huggingface.co/DANNY621/H3-World) |
| FlashVSR / RealBasicVSR | `models/upscale_models` or the workflow's named folder |
| DLSS-NR v1.3 external runtime (not a model) | `models/DLSS-NR/1.3` (user supplied; never downloaded or redistributed by the node) |
| Face, optical-flow, and segmentation models | The folder named by the workflow or its documentation |

Similar filenames do not guarantee compatible H3 structures. Use the model and LoRA combination named by the workflow.

## Face Refine Window: repair only selected intervals

[`examples/workflows/06-face-refine`](examples/workflows/06-face-refine) contains three workflows dated 2026-09-05:

- `Window_Manual_Review` processes one window, shows a preview, and waits for a manual accept or reject.
- `Window_Studio_Serial` records decisions for several windows in a recoverable manifest while still running only one H3 prompt at a time.
- `Window_Studio_Compose` rebuilds the final clip from the manifest without loading H3, including recovery after a crash between the last commit and save.

Enter bad-face intervals as zero-based inclusive frame ranges, for example `0-23`. The planner adds the continuous context
required by H3, but context and edge padding can never enter the accepted region. Preview and reject return the exact source
pixels. Accept requires an explicit confirmation. The final video always receives the complete original audio; generated
window audio is discarded.

Studio decisions are ordered and durable. An accepted window cannot be rolled back or rendered again, and a restarted job
continues from the first undecided window. A process-level project lock prevents two jobs from competing for the GPU. The
nodes never decide that a face is “better” and never auto-accept a candidate.

Generation workflows include the optional
`MiniMaxH3FaceRefineSamplerMaskPatchV11T8Advanced`, with **`enabled=false` by default**.
Disabled, it returns the exact input MODEL and LATENT unchanged, retaining the original route.
Enable it manually to apply the correction published in
[`ComfyUI-H3-FaceRefine v1.1.1`](https://github.com/Carasibana/ComfyUI-H3-FaceRefine/tree/d7ae3ee1ec445ea29fff7fc7366fa6fe85bdc2f5):
the video `noise_mask` stays on the sampler path only, held video latents are re-noised to the current
sigma, and the locked-audio mask condition still reaches H3. When enabled, the node checks the live
video/audio mask hashes against the denoise report and rejects mismatched masks, incompatible models, or conflicting
MODEL patches. The dated Parity, Window Manual, and Studio Serial workflows are already wired through
it. Compose-only does not load H3 and does not need it. No new model, pip package, or external executable
is required.

On the tested RTX 4060 Ti 16 GB, a 2 GiB reserve completed three cold and three warm runs at both 90 and 124 frames, followed
by three consecutive windows: 15/15 prompts passed. The lowest free VRAM was 678 MiB, and the final process-private deltas
were 40.74, 34.20, and 35.12 MiB, below the preregistered 256 MiB staircase limit. This is evidence for that exact machine
and workflow, not a universal 16 GB guarantee. The feature remains Advanced EXP: the first real bad-face sample and audio
mechanics passed, while broader perceptual face quality still requires full human blind review. A new
same-material 90-frame run with the v1.1.1 correction also passed strictly serial execution, exact source
PCM preservation, and a 717.8 MiB minimum-free-VRAM observation. In the September 6 blind review,
the user slightly preferred **B (the original route)** for overall quality, first-24-frame facial features,
and grid/blur; temporal consistency and seams tied. The original remains the default; the correction
is available for manual comparison. This single fixture does not establish universal quality or complete broader face-refinement acceptance.

## DLSS-NR: optional Windows RTX post-processing

[`examples/workflows/25-dlss-nr`](examples/workflows/25-dlss-nr) contains four independent workflows:
runtime audit, image, short-video frames, and file-backed video. They start from v1.3 `Standard + 2x`.

**DLSS-NR needs no new PyTorch or safetensors model, but it does need an external executable runtime.**
Download the full `video2dlssnr_release.zip` from the official
[`video2dlssnr` v1.3 release](https://github.com/DaniilSokolyuk/video2dlssnr/releases/tag/v1.3).
Do not use the light archive, and do not install the upstream ComfyUI node package. This project never
downloads, installs, or redistributes the EXE or proprietary NVIDIA DLLs.

Requirements:

- Windows 10/11, an NVIDIA RTX GPU, and NVIDIA driver **616.56 or newer**
- No extra pip package for the image or frame workflows; they use ComfyUI's existing Torch, NumPy, and PyAV environment
- The `Video File` workflow also needs `ffprobe` available on `PATH`, normally provided by an FFmpeg installation
- Applicable external-runtime and NVIDIA terms still apply; no acknowledgement checkbox is required in the node

Keep the full ZIP, copy the four files from its `out` folder into the `bin` folder below, and copy
[`examples/runtime-manifests/dlss-nr-v1.3.json`](examples/runtime-manifests/dlss-nr-v1.3.json)
as `t8-runtime-manifest.json`:

```text
ComfyUI/models/DLSS-NR/1.3/
├── t8-runtime-manifest.json
├── video2dlssnr_release.zip
└── bin/
    ├── video2dlssnr.exe
    ├── nvngx_dlss.dll
    ├── nvngx_dlssnr.dll
    └── nvngx.dll_dlssnr.dll
```

Run `Runtime_Audit` first. It verifies the full archive and extracted-file hashes, driver, GPU mapping,
and a real feature probe. Only continue after it reports `READY`. Runtime v1.2 is limited to 1x NR-only;
the default 2x workflows require v1.3.

All four methods passed the non-regression gate on the three fixed review clips. Human review found
different skin and texture character but no universal winner. Standard is therefore a practical
starting point, not a claim that DLSS-NR is always best. Upscaling cannot repair identity, lip-sync,
or real texture information already missing from the source.

## H3-World: WASD character and camera control

Upstream project: [`Danzer1xxxxChan/H3-World`](https://github.com/Danzer1xxxxChan/H3-World) ·
ComfyUI model bundle: [`t8star/Minimax-H3-World-Comfy`](https://huggingface.co/t8star/Minimax-H3-World-Comfy) ·
original model: [`DANNY621/H3-World`](https://huggingface.co/DANNY621/H3-World)

The first release has one deliberate contract: a first-frame I2VA at 832x480, 124 frames, and 24 fps.
Its 37 latent-time controls can move the character forward, backward, or sideways and tilt or pan the
camera. The `custom` preset accepts a tiled JSON timeline for mixed actions. Other input aspect ratios
are scaled to cover and center-cropped, never stretched.

Download the LoRA:

```bash
hf download t8star/Minimax-H3-World-Comfy --include "loras/**" --local-dir ComfyUI/models
```

The resulting path must be
`ComfyUI/models/loras/minimax/H3-World/step-10000.safetensors`. The bundled file is byte-identical to
the pinned upstream revision, with SHA-256
`DDD9187B920B1E52C2D090F4E264FD83D8D433EFC2A5B159E58883AEAF96E526`; T8star did not convert, merge,
or quantize it. It is already a directly loadable ComfyUI LoRA with 104 A/B pairs. The workflow also
uses the existing full `minimax_h3_fl2va_int8_convrot.safetensors`, Qwen3-VL
encoder, video VAE, and audio VAE. It adds no pip dependency, but its safe final MP4 writer requires
`ffmpeg` on `PATH`. Start from [`examples/workflows/26-h3-world`](examples/workflows/26-h3-world) and do
not stack OpenVDN, SLA, VSA, Sol-Attn, BlockCache, or another model/attention owner on this route.

The fixed parking-garage blind review passed: the reviewer correctly identified the continuously moving
H3-World candidate, accepted its stability and both audio tracks, and rated visual quality as a tie. This
promotes the fixed contract to a formal Advanced feature; it is not a universal quality or VRAM claim.

## OpenVDN: the recommended eight-step route

### New-Core compatibility and two-pass refinement (released in v1.75.0)

This update keeps existing nodes working after Core changes and adds low-resolution VDN generation followed by learned latent upscaling and a second sampling pass. Existing single-pass workflows remain available; no newly converted base model is required.

- **Keep the global Sage option enabled.** `--use-sage-attention` selects a default attention backend; it is not another node taking ownership of VDN. VDN retains its required attention algorithm, rather than becoming a Sage implementation of VDN.
- **Core's built-in sparse attention and the external Sol plugin are different integrations.** The new compatibility code bypasses authenticated Core sparse replacements only on the VDN branch. Other MODEL branches remain unchanged. Unknown model/attention replacements still produce a conflict diagnostic.
- **Two refinement routes are available.** VDN eight steps → learned 2x latent upscale → VDN four steps, or an independent native-H3 second pass with the new EMA B LoRA. Do not add a generic EMA LoRA to the VDN branch.
- **First-pass audio is retained by default.** The second pass refines the picture without regenerating the soundtrack. Both routes use the existing `models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors`. Only native-H3 refinement additionally uses `models/loras/minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors`. Saving requires FFmpeg; there are no new automatic downloads or installation steps.

The nine-input full/pruned two-pass matrix is complete. The reviewed 0.52MP pairs had similar visuals; T2VA music, speech and both candidates' lip-sync were normal, and I2VA ambience had no noise. Four workflows are installed in source/user folders; see the [two-pass guide](examples/workflows/10-speed/VDN_TWO_PASS.md). The isolated Core/VDN version passed its full 2435-test suite, trained-model DynamicVRAM refinement and archive import, and was released in commit `3769d70`. That release does not include the unfinished outpainting revision. This does not guarantee sharper results or safe memory use for every 16GB GPU or plugin combination; [compatibility limits](docs/CORE_VDN_COMPATIBILITY.md) distinguish tested and unverified scope.

Complete model bundle: [`t8star/Vdn-Minimax-H3-Comfy`](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy)

The model repository already follows ComfyUI's `diffusion_models`, `text_encoders`, and `vae` directory layout. After your model-access request is approved, download it directly into `ComfyUI/models`:

```bash
hf auth login
hf download t8star/Vdn-Minimax-H3-Comfy --local-dir ComfyUI/models
```

Then load a formal OpenVDN workflow from [`examples/workflows/10-speed`](examples/workflows/10-speed). The set includes:

- T2VA
- first-frame I2VA
- last-frame L2VA
- first-and-last-frame FL2VA
- single-image Ref2VA
- multi-image Ref2VA
- reference video with its audio
- standalone reference audio
- first-frame plus reference-audio Hybrid

OpenVDN upstream documents T2VA. The other modes are T8 extensions implemented and real-tested through ComfyUI's native MiniMax H3 conditioning layout.

### Full and supported pruned bases both work

The formal workflows default to:

```text
models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors
```

This is the full, non-pruned base with a 2688-column AdaLN input. After installing the updated model bundle, the same workflow can also use:

```text
models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
```

The Composer hashes the base checkpoint's `adaln_t_table` and automatically selects the matching curve-projected Turbo adapter; no extra LoRA node is required. The adapter keeps 208 directly compatible targets and converts the 51 AdaLN targets into eight-column LoRA factors plus 51 bias residuals, for 310 applied patches. Full-width bases continue to use the native 2688-column adapter.

Compatibility is bound to the curve-table content, not merely to a filename. The model bundle's FL2VA pruned INT8/ConvRot base is supported, and the FL2VA pruned FP8 base shares the same statically verified curve signature. An unknown pruned/curve-basis checkpoint still fails before sampling with a specific diagnostic instead of silently using the wrong adapter.

### Validation performed

Using the full base above, the project ran I2VA, L2VA, FL2VA, single-image, two-image, video-plus-audio, audio-only, and first-frame-plus-audio routes one at a time on the same RTX 4060 Ti 16 GB. It then repeated T2VA and the same eight multimodal routes with the pruned INT8 base at 320x192x39. Every pruned route completed:

- 800 OpenVDN branch tensors
- 104 default-adapter targets
- 259 logical turbo-adapter targets, with bias residuals for all 51 AdaLN targets and 310 applied patches in total
- eight Euler/native-flow steps with video/audio shifts 12/3
- strict H.264 video, AAC audio, and combined decoding
- zero `ERROR lora` events in the runtime log

This proves that the tested workflows compose correctly with both base types. It does not promise identical quality or memory use for every prompt, reference, GPU, or driver. The full-base matrix retained 535–890 MiB; the nine short pruned runs fell as low as 290 MiB, and only T2VA and I2VA cleared this project's 512 MiB margin. On 16 GB GPUs, run one H3 generation at a time and reduce the canvas or frame count when needed.

## Common problems

### All nodes are red or missing

Update the ComfyUI core, frontend, and Manager, then fully restart. Updating only this custom node may not be enough.

### Out of memory at startup or sampling

Lower the resolution, frame count, and number of references. Close other generation tasks. Do not run two H3 jobs concurrently on a 16 GB GPU.

### Noisy or unusually quiet audio

Check the sampler, scheduler, step count, video/audio shifts, and exact LoRA named by the workflow. Generic EMA, Ref2VA, OpenVDN turbo, and other acceleration LoRAs are not interchangeable.

### OpenVDN reports an AdaLN mismatch

Confirm that the updated model bundle contains `stage-dmd-step-250/adapters/turbo_pruned_curve_fl2va/adapter_model.safetensors`. If it does and the error remains, the selected pruned base has an unsupported curve signature. Use the bundle's FL2VA pruned INT8 base or fall back to the full `minimax_h3_fl2va_int8_convrot.safetensors`; do not rename files to bypass the signature check.

### Can OpenVDN be combined with SLA, VSA, Sol-Attn, or BlockCache?

OpenVDN already owns the model branch and adapters. Do not add another MODEL or attention owner. More acceleration nodes do not mean more speed when they compete for the same forward pass.

## Documentation

- [Workflow index](examples/workflows)
- [ComfyUI and model guide](docs/README_ComfyUI.md)
- [Verification record](docs/VERIFICATION_REPORT.md)
- [Feature inventory](features.json)

## Licenses

The custom-node source is GPL-3.0-or-later.

Model files have separate terms. MiniMax H3 and its derivatives use the MiniMax H3 Community License Agreement. Its defined Applicable Territory excludes the European Union, the United Kingdom, the Republic of Korea, and the United States of America. Read the complete agreement and Acceptable Use Policy before downloading, running, or redistributing the models.

This GitHub repository does not contain model weights. The separate Hugging Face bundle preserves the relevant license, NOTICE, source links, and modification notices.
