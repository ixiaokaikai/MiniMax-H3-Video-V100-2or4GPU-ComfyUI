# ComfyUI-MiniMaxH3-SolAttn-V100

**Attention sparsity accelerator for MiniMax H3 on V100 (ComfyUI custom node) · v1.2.0 (two nodes: Sol-Attn + SLA Top-K)**

> 中文版: [README.md](README.md) · Development/design docs: [DEVELOPMENT.md](DEVELOPMENT.md) / [DEVELOPMENT_EN.md](DEVELOPMENT_EN.md)
>
> **AI-assisted development notice**: This project was developed by the author with the help of an AI assistant (DeepSeek-V4-Flash). The author has no computer-science background; code and docs were co-written with AI. The project is shared as-is under the MIT license; if you run into issues, please open an Issue and we will help within our capabilities.

> ⏸️ **Maintenance pause (from 2026-09-01)**
>
> This project is developed and maintained by a single person (me). I'm away for work and the GPU machine (V100) is not at hand, so **no testing, reproduction or verification is possible** — **maintenance and updates are paused from 2026-09-01**.
>
> Every change here must be validated on real hardware (kernel behavior, routing quality, video and audio output all need end-to-end measurement). Without a GPU I cannot confirm a change actually works, nor reproduce reported problems, so I'm pausing rather than publishing untested updates.
>
> - Released versions (v1.2.x and earlier) keep working as-is;
> - During the pause I **cannot handle Issues / PRs** or respond to usage feedback (there is no other maintainer);
> - If you want to keep using it: troubleshooting notes are in [DEVELOPMENT.md](DEVELOPMENT.md); for stability switch to the FP16Safe node (dense); to change the code, **vibe-coding tools (AI coding assistants)** can help you patch it yourself;
> - Thanks for your understanding.

Two independent nodes (mutually exclusive on the same MODEL — attach only one):

- **Sol-Attn (V100)** (default, v1.1.1): Sol-Attn threshold routing (keep-or-drop), training-free, no paired LoRA needed. On V100 (sm_70), MiniMax H3 drops from **71-74 s/step to 43 s/step at 480p/10s (~1.7×); 24 s/step vs 33 s/step dense at 960×544/5s (+27%) with quality visually ≈ dense**.
- **SLA Top-K (V100)** (new in v1.2.0): SLA (Sparse-Linear Attention) top-K block sparsity, paired with the lightx2v **Minimax-h3-Turbo-SLA LoRA** (4-step distilled + 85% sparsity-distilled). Density 15-19%; 15-17% faster than Sol-Attn on long sequences; audio segment auto-guarded.

---

## Credits (what this project references and uses)

Everything here is a **proven, existing implementation** combined together. The acceleration comes entirely from:

| Component | Source | Role |
|---|---|---|
| **Sol-Attn sparse algorithm** | [arXiv 2607.24027](https://arxiv.org/abs/2607.24027) (NVlabs/Sana sol-engine, `techniques/sparse_backends/sol_attn/preprocess.py`) | kc/vc block stats + diag threshold + column-mean routing + keep-or-drop (`routing.py` re-implements the official algorithm; per-head independent routing) |
| **keep-or-drop kernel** | direct source [rwashy/H3-V100](https://github.com/rwashy/H3-V100) (repo is GPL-3.0-only; its FlashAttention CUDA component is BSD 3-Clause — only that BSD component is copied); upstream [flash-attention](https://github.com/Dao-AILab/flash-attention) (Tri Dao) + [Icbears/flash-attention-v100](https://github.com/Icbears/flash-attention-v100) | selected blocks computed exactly, others skipped (`comfy_v100_solattn_cuda`: source in `native/`, or prebuilt pyd from Release) |
| **fp16 NaN safety** | [ComfyUI-MiniMaxH3-FP16Safe](https://github.com/aaalll12322/ComfyUI-MiniMaxH3-FP16Safe) v6.8.0 (logic embedded as `fp16safe.py`) | prescale /16 + deferred fuse + fp32 re-run fallback (**self-contained, no separate FP16Safe install needed**) |
| **Parameter style** | [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) | tau / start_percent / end_percent / dense_blocks / sink parameter family |
| **SLA sparse algorithm** (SLA Top-K node) | [arXiv 2509.24006](https://arxiv.org/abs/2509.24006) (thu-ml/SLA) | top-K block sparse routing (`sla_routing.py`: smooth-k + block-mean scoring + per-row top-K, line-by-line aligned with [LightX2V `get_block_map`](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/common/ops/attn/utils/sla_util.py)) |
| **SLA LoRA weights** (SLA Top-K node) | [lightx2v/Minimax-h3-Turbo-SLA](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA) (Apache-2.0) | 4-step distilled + 85% sparsity-distilled LoRA (standard rank-128; loaded natively by ComfyUI LoraLoader, no adaptation needed) |

---

## Problem

Two core bottlenecks of MiniMax H3 on V100:

1. **Attention dominates** (~79% of per-step time at 480p), while PyTorch SDPA on V100 only reaches ~37T;
2. **fp16 compute NaN** (H3 activations genuinely reach ~5e5, far beyond the fp16 limit ±65504); official dtype support is bf16/fp32 only, and V100 has no bf16 hardware so it falls back to fp32 (4× slower).

Sol-Attn sparse idea: most attention scores are noise — **compute only the few high-value KV blocks exactly** (keep-or-drop) and skip the rest, removing the O(n²) bulk.

---

## Measured performance (user's real ComfyUI, 480p/10s)

| Config | s/step | Quality |
|---|---|---|
| Pure FP16Safe (baseline) | 71-74s (480p/10s); 33s (960×544/5s) | normal |
| **Sol-Attn v1.1.1 (recommended `tau=0.75, topk=32`)** | **43s (480p/10s); 24s (960×544/5s); 44s (1280×736/4s)** | **visually ≈ dense** |
| **SLA Top-K v1.2.0 (`sparsity_ratio=0.85` + SLA LoRA)** | **41s (480p/10s); 20s (960×544/5s); 38s (1280×736/4s); 115s (1280×736/8s)** | **visually ≈ dense, audio normal (audio guard)** |

- vs pure FP16Safe: **~1.7×** at 480p/10s; **+27%** at 960×544/5s (top-K guarantee keeps quality ≈ dense on high-motion/text/complex-action scenes)
- SLA Top-K vs Sol-Attn (same seed/prompt/LoRA): **38s vs 44s at 1280×736/4s (-14%); 115s vs 135s at 1280×736/8s (-15%); 20s vs 24s at 960×544/5s (-17%)** — SLA runs at lower density (15-19% vs 40%), and the lightx2v SLA LoRA's sparsity distillation keeps quality
- Single attention (S=29650): sparse kernel **4.55×** vs SDPA (route+kernel 2.77×)
- Routing v1.1: **6.2ms @ S=29650 / 35.6ms @ S=98512** (view-based block stats, zero pad copies); kernel consumes non-contiguous inputs (saves 3 copies)

## Quality comparison (v1.2.0, 1280×736/4s, same prompt/reference/seed/SLA LoRA)

<p align="center">
  <a href="https://raw.githubusercontent.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/main/videos/sla_topk_1280x736.mp4"><img src="https://raw.githubusercontent.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/main/videos/sla_topk_1280x736.gif" width="30%" alt="SLA Top-K (click for full video)"></a>
  <a href="https://raw.githubusercontent.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/main/videos/solattn_1280x736.mp4"><img src="https://raw.githubusercontent.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/main/videos/solattn_1280x736.gif" width="30%" alt="Sol-Attn (click for full video)"></a>
  <a href="https://raw.githubusercontent.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/main/videos/dense_1280x736.mp4"><img src="https://raw.githubusercontent.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/main/videos/dense_1280x736.gif" width="30%" alt="dense (click for full video)"></a>
</p>

**Left: SLA Top-K** (38 s/step, audio normal) ｜ **Center: Sol-Attn** (44 s/step) ｜ **Right: dense** (80 s/step) ｜ *GIF is a 360-wide preview (~1.8 MB); click for the 1280×736 original*

<p align="center">
  <img src="https://raw.githubusercontent.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/main/videos/sla_vs_sol_vs_dense_compare.png" width="70%" alt="SLA Top-K vs Sol-Attn vs dense frame comparison (1s/3s)">
</p>

Frame differences are the same order as Sol-Attn vs dense (SSIM ~0.72-0.75, the normal sparse-vs-dense gap); SLA audio <500Hz energy 11% (dense 8%, 34% when broken) — the audio-segment guard fixes the audio degradation seen before v1.2.0.

---

## Install

```bash
cd ComfyUI/custom_nodes
# Option 1: git clone (when published to GitHub)
git clone https://github.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100.git
# Option 2: copy the whole folder into custom_nodes/ (kernel needed, see below)
```

**Getting the kernel** (`comfy_v100_solattn_cuda*.pyd` / `*.so`, pick one; the plugin auto-matches any file whose name starts with `comfy_v100_solattn_cuda` — pyd on Windows, so on Linux; no specific Python-version suffix required):
- **Release prebuilt (Windows)**: download from [GitHub Release](https://github.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/releases) (Windows + Python 3.12, zero compilation). **v1.1.1 is a pure-Python change — the kernel was NOT recompiled (sha256 `1518648115fa4c527a541ba996c59e0c5ff4c33bbca0ece82d4d56ea367c9f87`, same as v1.0.0) — grab the same pyd from the v1.0.0 Release**
- **Build from source**: full source ships in `native/` (incl. CUTLASS), one command:
  - **Windows** (MSVC; needs Visual Studio Build Tools + CUDA Toolkit + Windows SDK):
    ```bash
    cd native && python setup.py build_ext --inplace
    # copy the artifact (comfy_v100_solattn_cuda.cpXXX-win_amd64.pyd, XXX follows the
    # Python version used to build) to the plugin root — the plugin detects it automatically
    ```
  - **Linux** (GCC; needs CUDA Toolkit 12.x + python3-dev + ninja):
    ```bash
    cd native && CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=8 python setup_linux.py build_ext --inplace
    # artifact (comfy_v100_solattn_cuda.cpython-XXX-x86_64-linux-gnu.so, auto-stripped)
    # copy to the plugin root — the plugin detects it automatically
    ```

Restart ComfyUI. Find **"Sol-Attn (V100)"** under the `sol_attn` category.

**Dependencies**:
- **ComfyUI** (with `comfy/ldm/minimax`, PR #15224)
- **NVIDIA V100 (sm_70)** only (raises at startup on other architectures)
- **No extra Python packages, no extra plugins**: FP16Safe logic embedded (`fp16safe.py`); the CUTLASS kernel source ships in `native/`, prebuilt pyd from Release

---

## Usage

```
H3 model ──> Sol-Attn (V100) ──> sampler (KSampler etc.)
```

The single node performs fp16 safety + sparse. **No separate FP16Safe node needed.**

### Parameters

| Param | Default | Description |
|---|---|---|
| `fp16_safe` | true | Embed FP16Safe (prescale /16 + deferred fuse + fp32 re-run). Disable only if another node provides fp16 safety |
| `tau` | 0.75 | Sparse routing threshold β: higher = sparser/faster. 0.75 ≈ 40% density (V100 real-machine, quality ≈ dense, recommended default); 1.0 ≈ 26% density (faster, quality drops on high-motion/text scenes) |
| `start_percent` | 0.2 | Run dense before this sampling progress (paper uses 0.2; with turbo 4-step the first step is naturally dense) |
| `end_percent` | 0.9 | Run dense after this sampling progress (0.9 = trailing 10% of steps stay dense, recommended default; 1.0 = no trailing dense, fastest) |
| `min_tokens` | 1024 | Sequences shorter than this stay dense (SDPA) |
| `dense_blocks` | "0-1,-1" | Transformer blocks kept dense, e.g. `"0-1"` = first two, `"0-2,-1"` = first three + last (-1 counts from the end); empty = sparsify all |
| `h3_prefix_tokens` | 1024 | KV sink guard: protects the first tokens of the sequence from being sparsified. **Keep the default 1024; it is unrelated to the sequence length S** — the `S=...` in the console log is the sequence length, not a reference value for this parameter. **Do NOT set it ≈ S** (the sink then covers nearly the whole sequence, beyond its design semantics; measured to corrupt the last frames) |
| `topk_blocks` | 32 | **Per-row guaranteed blocks (quality fix)**: threshold routing is a "mean-alignment detector" that filters out low-alignment high-motion/new-content blocks (lost hands/edges/limbs). Forces top-K highest-score blocks per row. 0 = off (v1.0 behavior). 32 ≈ 2% density cost on 1540 blocks |
| `debug_nan` / `profile` | false | Pass-through FP16Safe NaN detection / timing stats |

### Recommended configs

- **Default config = recommended config (v1.1.1, out-of-the-box)**: `tau=0.75, start_percent=0.2, end_percent=0.9, dense_blocks="0-1,-1", topk_blocks=32, h3_prefix_tokens=1024` → 43 s/step at 480p/10s; 24 s/step at 960×544/5s (quality ≈ dense). **Keep `h3_prefix_tokens` at the default 1024; do not adjust it from the console `[SolAttn] S=...`** (S is the sequence length — unrelated)
- **Speed-first**: `tau=1.0, topk_blocks=16` (lower density, faster; verify quality yourself)
- **Max quality**: `topk_blocks=64` (more guaranteed blocks per row; most stable on motion/text; slower)
- **Conservative**: `dense_blocks="0-2,-1"` or `end_percent=0.8` (more layers / trailing dense, sturdier quality, slightly slower)
- **Small resolutions** (608 and below): attention share is low, sparse gains are small — prefer `end_percent=0` (fully dense) or skip the plugin

---

## SLA Top-K (V100) node (new in v1.2.0)

**Purpose**: pair with the [lightx2v/Minimax-h3-Turbo-SLA](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA) LoRA (4-step distilled + 85% sparsity-distilled). Load the LoRA with the native ComfyUI **LoraLoader** (standard rank-128; keys match the ComfyUI H3 implementation verbatim — no adaptation needed); this node supplies the matching **SLA top-K block sparsity** (per-row top-K blocks, 85% sparsity aligned with lightx2v's official config).

```
H3 model ──> LoraLoader(SLA LoRA) ──> SLA Top-K (V100) ──> sampler (turbo, 4 steps)
```

### Parameters

| Param | Default | Description |
|---|---|---|
| `sparsity_ratio` | 0.85 | Fraction of key blocks skipped (0.85 = lightx2v's official distilled value; keep 15% + guards ≈ 18-19% density). Higher = faster, may drop quality |
| `start_percent` / `end_percent` | 0.2 / 0.9 | Sampling window (same semantics as Sol-Attn; with turbo 4-step steps 1 & 4 are naturally dense) |
| `min_tokens` | 1024 | Sequences shorter than this stay dense |
| `dense_blocks` | "0-1,-1" | Transformer blocks kept dense (same semantics as Sol-Attn) |
| `h3_prefix_tokens` | 1024 | text/cond/ref prefix sink guard |
| **audio guard** | auto | Reads the H3 `PackedLayout` audio segment automatically (audio sits right before video, second-to-last segment) and **forces it dense** — SLA top-K sparse drops audio key blocks and breaks audio (measured <500Hz energy 34% vs normal 8-17%); guarded it recovers |

### Measured (V100, same-seed comparisons)

- 1280×736/4s: **38 s/step** vs Sol-Attn 44s / dense 80s; audio normal (<500Hz energy 11%, dense 8%)
- 1280×736/8s: **115 s/step** vs Sol-Attn 135s (-15%)
- 960×544/5s: **20 s/step** vs Sol-Attn 24s (-17%); 480p/10s: 41s (≈ Sol-Attn; small sequences gain little)

### Notes

- **Mutually exclusive** with the Sol-Attn node (one MODEL, one `optimized_attention_override` slot)
- SLA LoRA and a plain 4-step turbo LoRA run at the same speed (a LoRA only changes weights, not the inference path); **the SLA LoRA's value is quality** (distilled for 85% sparsity — a plain turbo LoRA may break at low density)
- The full-paper SLA linear-attention compensation needs paired fine-tuned weights (proj_l), which the lightx2v LoRA does not contain and was not distilled for (measured: adding it degrades rel from 0.22 to 0.92). So this implementation is pure top-K sparsity — that IS the lightx2v SLA distillation target and its capability ceiling

---

## Principles (summary)

1. **Routing (routing.py, official Sol-Attn algorithm)**: kc block-mean / vc block-sum → diag threshold (analytic projection in key space) → column-mean routing (|neighbor ±1) → CSR mask. Per-head independent; at 26% density rel-L2 is 0.22 vs dense, yet **real videos show no visible loss**.
2. **Kernel**: keep-or-drop sparse kernel (selected blocks computed exactly, others skipped), fp16 + head_dim 128, sm70 CUTLASS.
3. **FP16Safe**: x/16 prescale → qkv → attention → out_proj, unscaled ×16 in fp32; deferred isfinite fuse re-runs the whole forward in fp32 if triggered.

---

## Known limitations

- Verified only on **V100 (sm_70)** + Windows + Python 3.12 (cp312 pyd); other platforms require rebuilding native.
- Sparse routing runs in Python (~10ms @ S=29650) — still optimizable; the kernel itself is fast (4.55×).
- Sparse quality is judged by real video (PSNR/eyes); for sensitive scenes (fine text), raise `dense_blocks` / `end_percent` or lower `tau`.
- Dense fallback = plain SDPA.
- The SLA Top-K node only guarantees quality with the **lightx2v SLA LoRA** (weights distilled for its sparse path); paired with a plain turbo LoRA, quality may drop at low density.
- The SLA audio guard depends on the ComfyUI H3 `PackedLayout` (bridged through the model forward); if ComfyUI changes that interface, this needs re-adapting.

---

## Changelog

- **v1.2.2 (2026-08-28)**: **Fixed misleading `h3_prefix_tokens` docs (issue #3)** — the old console log / tooltip / README wording ("suggest ≥ actual prefix") led users to set the parameter from the sequence length S; one user set ≈S (19008) and the last frames corrupted. Now uniformly: "**keep default 1024 — unrelated to S, do NOT set ≈ S**" (console log, node tooltip, README param table & recommended config in both languages updated); DEVELOPMENT gained a pitfall entry.
- **v1.2.1 (2026-08-27)**: **Linux support** (from issue #2 community contribution, thanks lesca) — new `native/setup_linux.py` (GCC build + auto-strip of debug symbols, artifact ~1MB ≈ Windows pyd); nodes & verify scripts now load the kernel by platform (`.pyd` on Windows / `.so` on Linux), `sla_nodes.py` included; verified on WSL2 + V100 (SLA topk density 28.9% rel-L2 0.2176, SolAttn 40.0%/0.1970).
- **v1.2.0 (2026-08-23)**: new **SLA Top-K (V100)** node (`sla_nodes.py` + `sla_routing.py`, separate files, separate node) — SLA top-K block sparsity for the lightx2v Minimax-h3-Turbo-SLA LoRA (smooth-k + block-mean scoring + per-row top-K, line-by-line aligned with LightX2V `get_block_map`); **audio-segment auto guard** (PackedLayout bridge, fixes audio corruption under SLA sparsity); includes the v1.2 streaming chunked routing (`route_chunk`, -74% routing VRAM on long sequences). Real machine: 1280×736/4s 38 s/step, 8s 115 s/step (vs Sol-Attn 44s/135s), 960×544/5s 20 s/step; audio & video normal. LoRA loads with native LoraLoader (no adaptation needed).
- **v1.1.1 (2026-08-22, WIP)**: ① routing perf — removed per-layer GPU→CPU sync, rank int32, off half memory, view-based block stats (S=98512 route 182→35.6ms; S=174112 no longer OOM); ② **quality fix top-K guarantee** (`topk_blocks`, default 32): `combined = min(threshold, kthvalue(K-th largest))` keeps ≥K blocks per row; real-activation rel 0.1157→0.0619 (-47%), fixes hand/edge loss on high-motion/cut/text/complex-action scenes; ③ prefix debug: console prints `[SolAttn][v1.1.1] S=... density=... prefix_tokens=...`; ④ real-machine 960×544/5s (S=20822): **24 s/step vs 33 s/step dense (+27%), quality visually ≈ dense**; 480p/10s (S≈98512) 42 s/step (tau=0.75). Recommended `tau=0.75, end_percent=0.9, dense_blocks="0-1,-1", topk_blocks=32`.
- **v1.0.0 (2026-08-20) initial release**: single node = embedded FP16Safe (`fp16safe.py`, v6.8.0 logic, self-contained) + Sol-Attn sparse (keep-or-drop, sparse-only kernel). Measured 480p/10s at **43 s/step, no visible loss** (~1.7× vs pure FP16Safe 71-74 s/step). Parameters aligned to kijai style (tau / start_percent / end_percent / min_tokens / dense_blocks / h3_prefix_tokens); dense fallback = plain SDPA; kernel source in `native/` (self-buildable), prebuilt pyd from Release. Tag `[SolAttn-V100][V1.0]`.

---

## Citation

The sparse algorithms used by this project come from the Sol-Attn paper (Sol-Attn node) and the SLA paper (SLA Top-K node). If this project helps you, please also cite:

```bibtex
@article{solattn,
  title={Sol-Attn: Training-free Sparse Attention for Accelerating Image and Video Generation},
  author={NVlabs / Sana sol-engine team},
  journal={arXiv preprint arXiv:2607.24027},
  year={2026}
}

@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention},
  author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jianfei Chen and Jun Zhu},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}
```

- Sol-Attn paper: https://arxiv.org/abs/2607.24027 ｜ Official code: <https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends/sol_attn> ｜ Project page: https://nvlabs.github.io/Sana/Sol-Attn/
- SLA paper: https://arxiv.org/abs/2509.24006 ｜ Official code: https://github.com/thu-ml/SLA ｜ SLA LoRA: https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA ｜ LightX2V: https://github.com/ModelTC/LightX2V
