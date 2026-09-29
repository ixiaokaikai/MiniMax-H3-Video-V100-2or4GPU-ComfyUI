# DEVELOPMENT — ComfyUI-MiniMaxH3-SolAttn-V100 design & dev notes

> 中文版: [DEVELOPMENT.md](DEVELOPMENT.md) · Usage: [README.md](README.md) / [README_EN.md](README_EN.md)
>
> **AI-assisted development notice**: This project was developed by the author with the help of an AI assistant (DeepSeek-V4-Flash). The author has no computer-science background. This document records design decisions, measured data and pitfalls for future maintenance.

## 1. Goal

Accelerate MiniMax H3 video-generation attention on **V100 (sm_70, single 16GB, no bf16/fp8 hardware)** (attention is ~79% of per-step time at 480p). Baseline: pure FP16Safe at 71-74 s/step (480p/10s).

Design principles (explicit user requirements):
- **Reuse proven, existing implementations** — no reinventing wheels (Sol-Attn official algorithm + flash-attn official sparse kernel);
- **Single node**: one `Sol-Attn (V100)` node = FP16Safe (fp16 safety) + Sol-Attn sparse; no chain of nodes needed;
- Every performance/quality conclusion is **measured on real hardware** (GPU util/power/VRAM monitoring + user's real-video judgment).

## 2. Architecture

```
Sol-Attn (V100) node (single-node patch)
├─ fp16_safe=True → embed FP16Safe (_load_fp16safe_nodes scans custom_nodes for its nodes.py)
│    prescale x/16 → qkv_proj → attention → out_proj, unscaled ×16 in fp32
│    deferred isfinite fuse: re-run whole forward in fp32 if triggered
├─ optimized_attention_override (dispatch)
│    ├─ sparse eligible → routing.py (kc/vc + diag threshold + neighbor ±1 + sink)
│    │    → CSR mask → varlen_fwd_sparse (keep-or-drop kernel, prebuilt pyd)
│    └─ not eligible (mask / non-fp16 / short seq / outside window / dense_blocks) → plain SDPA
└─ sparse defenses: sampling window (start_percent/end_percent) + dense_blocks
                    + h3_prefix_tokens (KV sink)
```

### Component sources (all proven, existing implementations)

| Component | Source | Role |
|---|---|---|
| `routing.py` | Sol-Attn official algorithm ([arXiv 2607.24027](https://arxiv.org/abs/2607.24027), NVlabs sol-engine preprocess.py re-implementation) | BLOCK=64, kc block-mean, vc block-sum, diag threshold, column-mean routing, CSR generation |
| `comfy_v100_solattn_cuda.pyd` | built from native/ source (flash-attn official sparse kernel, sm70 port, sparse-only) | only `varlen_fwd_sparse` op; source in `native/`, prebuilt pyd from Release |
| FP16Safe | ComfyUI-MiniMaxH3-FP16Safe v6.8.0 (logic embedded as `fp16safe.py`) | self-contained: `_load_fp16safe_nodes` imports the built-in module, no external plugin |

## 3. Key design decisions

1. **Sparse = keep-or-drop (per-head independent)**: official Sol-Attn routing + sparse kernel computes selected blocks exactly, skips the rest. Block-mean zeroth-order approximation (kijai's fused two-level) measured *worse* on H3 (4.3e-2 vs keep-or-drop 5.8e-3 in the old comparison; final judgment is real-video quality).
2. **Kernel from the PAI branch, not the official one**: the upstream fork's sparse kernel P-layout conversion (hand-written shfl) outputs all-zeros on sm70; the PAI/Alibaba branch's `convert_layout_C_to_A_v2` (reusing the dense kernel template) is correct. Lesson: **reuse = full implementation + real-hardware verification, not just reading code**.
3. **Dense fallback = plain SDPA**: on long sequences with limited VRAM, extra kernels' memory peaks and swapping can eat all compute gains → keep the memory footprint minimal. User-measured 43 s/step (faster than the combo with the extra kernel). Lesson: **a fast kernel ≠ fast end-to-end; benchmark end-to-end on real hardware**.
4. **Vectorized routing**: `csr_from_sel` originally had a `for r in uniq` Python loop (1562ms @ S=6154); advanced indexing + cumsum slots brought it to **10ms @ S=29650**.
5. **`_sparse_attn` does not require S%64**: the kernel (is_even_MN=false path) handles arbitrary sequence lengths (verified at S=6154).
6. **Sampling window in percent semantics** (kijai style): `start_percent/end_percent` derived from `step/total_steps` in transformer_options; falls back to sigma>14 check (turbo 4-step step-1 sigma≈14.64 → naturally dense).
7. **Top-K guarantee (v1.1.1 quality fix)**: Sol-Attn threshold routing is a "mean-alignment detector" (keeps only `scores > mean+τ·std`), so **high-motion/new-content blocks with low key-query alignment get filtered out** (real-machine: hands/edges/limbs lost on dynamic frames). Fix = per-row top-K guarantee: `combined_threshold = min(threshold, kthvalue(K-th largest))` — kthvalue is O(N) selection, no sort; `sel = scores >= combined` is mathematically threshold-routing ∪ top-K. Node param `topk_blocks` (default 32). Real-activation rel 0.1157→0.0619 (-47%).
8. **The `var` in the threshold formula is the cross-block variance of kc** (a global constant, identical for every block) — "var raises the threshold and prunes dynamic blocks" is a misconception; dynamic blocks are pruned only because of low alignment. Do **not** switch to `mean - α·var` (would pull in irrelevant blocks); top-K is the right fix.
9. **Routing perf (v1.1)**: removed `sel_perm.any()` per-layer GPU→CPU sync (stalls the weight-prefetch pipeline under LOW_VRAM); rank int32; off supports nnz_s shrink (kernel reads num_blks at runtime; extra slots unread); **view-based block stats** (fp16 view+sum, zero F.pad/float copies — F.pad on non-contiguous input is ~17× slower). S=98512 route 182→35.6ms; S=174112 no longer OOM.
10. **SLA top-K routing (v1.2, SLAV100 node), line-by-line aligned with LightX2V `get_block_map`**: ① **smooth-k** — subtract the sequence mean from k before pooling (SageAttention trick; the official training router uses a centered key, so block means reflect "change" rather than absolute magnitude — selection differs significantly, must align); ② scoring = pooled_qblocks @ pooled_kblocks^T (no scale — sorting is unaffected by a positive scale); ③ topk = `int(topk_ratio × NB)` truncation (short sequences keep 1 block); ④ selection via kthvalue (no sort; extra ties harmless — keeping equal-score blocks is lossless). Do **not** reuse the Sol-Attn threshold routing (density floats with content); SLA is fixed top-K.
11. **Audio-segment guard (v1.2 quality fix)**: the H3 sequence is `[text | ref | audio | video]` — audio is the second-to-last segment (before video), so `h3_prefix_tokens` only protects text/ref → SLA top-K sparse drops the audio rows' key blocks → **audio quality collapses** (real-machine: <500Hz energy 34% vs dense 8%, worst at the start/end). Fix: a forward bridge hook injects the audio segment (a,b) from `minimax_payload`'s `PackedLayout.segments` into transformer_options (the payload does not pass through transformer_options — a bridge is required), and the router forces the audio segment dense. Real machine: <500Hz 34%→11% (≈dense 8%). Sol-Attn has no such issue (threshold routing, 40% density, self-adapting).
12. **Linear-compensation experiment (v1.2 conclusion: not viable)**: full-paper SLA = sparse + linear attention compensation (φ(Q)(φ(K)ᵀV)/φ(Q)Σφ(K)) + a Proj projection layer, but the linear part **requires paired fine-tuned weights** (proj_l eases the softmax-vs-linear distribution gap). The lightx2v LoRA has no proj_l and was not distilled for that path — measured on V100 (undistilled activations): pure sparse rel 0.218 → sparse+linear **0.915 (4× worse)**. **Conclusion: pure top-K IS the lightx2v SLA distillation target and its capability ceiling**; paper-grade 95% sparsity would need self-trained weights (no training resources — not realistic).
13. **PyTorch Linux vs Windows wheels can have different arch lists (measured, v1.2.1)**: our Linux torch 2.7.1+cu128 had no sm_70 (`['sm_75', ...]`) → matmul failed on V100 with `no kernel image`; Linux 2.10.0+cu128 has sm_70 (everything passes; Windows 2.8.0 also has it). If you hit this, check `torch.cuda.get_arch_list()` and switch torch versions.
14. **Linux .so bloat = debug symbols**: Ubuntu default CFLAGS add `-g`; DWARF is ~5MB (5.88MB vs 972KB Windows pyd); `strip --strip-debug` → 984KB, on par. Strip must run in `run()` (the inplace copy happens in the run phase) — putting it in `build_extensions()` does nothing.
15. **Linux venv/build missing pieces (standard Ubuntu traps)**: `python3 -m venv` needs `python3.12-venv` installed first (ensurepip); building C++ extensions needs `python3.12-dev` (Python.h) + `ninja-build`.
16. **WSL sudo password hangs non-interactive shells**: use `wsl -d <distro> -u root` to run as root directly, bypassing the sudo password prompt (non-interactive shells wait forever on input).
17. **Mirror false impressions**: curl without `-L` shows 0B from the NVIDIA repo (301 redirect); Tsinghua/Aliyun mirrors have no `wsl-ubuntu` CUDA dir (404) — the CUDA toolkit only comes from NVIDIA's official repo; torch wheels from Aliyun `pytorch-wheels/cu128/` (flat dir, not PEP 503 — use `--find-links`, and wheel filenames must contain `+cu128`).

## 4. Measured data (V100-SXM2-16GB, torch 2.8.0+cu128, ComfyUI Python 3.12)

### 4.1 Kernel math correctness (vs PyTorch keep-or-drop simulation)

| Test | rel-L2 |
|---|---|
| All-selected mask (= dense semantics) | 3.0e-4 |
| Single block S=64 | 5.9e-6 |
| Per-row different masks S=256/1024/2048/4096 (random) | 2.1-2.7e-4 |
| S=6154 real activations (incl. non-64-multiple tail) | 2.1e-4 |

### 4.2 Speed (real activations / 480p scale)

| Scenario | Time | Ratio |
|---|---|---|
| SDPA (S=6154) | 30.3ms | 1× |
| sparse kernel (S=6154) | 12.3ms | 2.46× |
| SDPA (S=29650) | 814ms | 1× |
| sparse kernel (S=29650) | 179ms | 4.55× |
| route+kernel (S=29650) | 294ms | 2.77× |

### 4.3 End-to-end (user's real ComfyUI, 480p/10s)

| Config | s/step | vs baseline |
|---|---|---|
| Pure FP16Safe | 71-74 | 1× |
| **Sol-Attn v1.0 (tau=1.0)** | **43** | **~1.7×, no visible loss** |

### 4.4 Routing optimization

| Version | Routing time @S=29650 | Notes |
|---|---|---|
| v1.0 (nonzero loop) | 1562ms | per-row Python loop |
| v1.0 (scatter+masked_fill v1) | 115ms | **buggy**: `masked_fill_(~sel_perm)` clears column positions while scatter writes slot positions → unsaved columns pollute slot 0 → kernel vs sim rel 2.67 |
| **v1.0 (advanced indexing)** | **10ms** | writes only selected positions; correctness restored rel 2.85e-4 |

Also: the kernel consumes non-contiguous q/k/v directly (stride-based access, measured faster) → saves 3 contiguous copies (~7ms).

### 4.5 Sparse quality (honest record)

Per-head keep-or-drop at τ=1.0: density 26.4% (S=6154 real activations), rel-L2 vs dense = **0.223**. The earlier "5.8e-3" was an artifact of head-union computation (per-head routing but unioned compute → real density far above the reported value); the inconsistent metric was discarded. **Despite rel 0.22, real videos (turbo 4step) show no visible loss** — final quality is judged on real hardware (user requirement, replacing pure L∞/rel metrics).


**⚠️ v1.0 quality judgement was insufficient** (user feedback 2026-08-21): lossless on static/medium scenes, but **high-motion / fast cuts / dense text / complex actions** showed deformed hands, melted edges, abnormal limb extensions. Root cause = threshold routing's mean-alignment detector prunes low-alignment new-content blocks (see §3.7). Fix = top-K guarantee (v1.1.1).

### 4.6 v1.1.1 validation (routing perf + top-K quality)

**Routing perf** (V100, non-contiguous input, same path as the node):

| S | v1.0 route | v1.1 route | peak mem (v1.1) |
|---|---|---|---|
| 29650 | 107ms | **6.2ms** | 1.04GB |
| 98512 | 182ms | **35.6ms** | 5.01GB |
| 174112 | est. 450ms+ (OOM risk) | **110.7ms** | 11.94GB |

**Top-K quality** (real-activation snapshot S=6154, rel vs dense):

| config | rel | density | note |
|---|---|---|---|
| tau=1.0 topk=0 (v1.0) | 0.1157 | 25.9% | baseline |
| tau=1.0 topk=32 | 0.0640 (-45%) | 36.3% | guarantee only |
| **tau=0.75 topk=32 (recommended)** | **0.0619 (-47%)** | 38.6% | quality/speed balance |
| tau=0.75 topk=64 | 0.0237 (-80%) | 66.6% | max quality, slower |

Top-K covers ~10 extra threshold-missed blocks per row (exactly the dynamic/new-content blocks).

**End-to-end (user real machine, 960×544/5s, S=20822, FLOW_AV, LOW_VRAM)**:

| setup | s/step | vs dense |
|---|---|---|
| Pure FP16Safe (dense) | 33 | 1× |
| **Sol-Attn v1.1.1 (tau=0.75 + topk=32)** | **24** | **+27% faster, quality visually ≈ dense** |

Note: 480p/10s (S≈98512) measured 42 s/step (v1.1 route + tau=0.75) — not comparable to the S=20822 workflow (different seq).

**GPU temperature insight**: sparse kernel at util 99% runs 51-59°C (dense full-load 70°C) = "high-occupancy low-power" (TC not saturated). **Temperature ≠ idle**; don't judge headroom by temperature.
## 5. Pitfalls

1. **diff newline pollution**: CRLF/LF mixing makes diff flag every line → use `diff --strip-trailing-cr`.
2. **nvcc template error line offsets**: reported line numbers can be off by a couple of lines for template instantiation errors; do not suspect stale-cache compilation.
3. **Scope errors**: `rows_this_block`/`warp_row_base` defined in a nested block, referenced outside → inline as expressions.
4. **Sandbox recycle-bin**: `setup.py build_ext --inplace`'s final pyd copy fails on safe-delete (recycle-bin unavailable) → after a successful build, `cp` the artifact manually (build/lib.win-amd64-cpython-312/*.pyd).
5. **ninja abnormal exit (0x40000004)**: `_bt`/`build` state corrupted → wipe both directories and rebuild.
6. **Parallel builds**: default is serial (single core) → use `MAX_JOBS=4` (4m48s).
7. **Debug printf**: SPARSE_* prints inside the kernel are debug residue — remove before release (perf + spam).
8. **scatter+masked_fill trap**: `off.scatter_(1, rank, col)` writes "slots" while `masked_fill_(~sel)` clears "column positions" — the two coordinate systems pollute each other. Use advanced indexing `off[rows, pos] = col[cols]` to write only selected positions.
9. **VRAM-peak swapping eats gains (long seq + limited VRAM)**: a fast kernel ≠ fast end-to-end; an extra kernel's memory peak forces model swapping, eating all compute gains. Always benchmark end-to-end on real hardware, not just the kernel in isolation.
10. **Incomplete component removal on rename**: besides deleting the module files, also clean nodes.py imports, transformer_options keys, parameters and log references — grep each one for leftovers.
11. **`h3_prefix_tokens` ≈ S corrupts the last frames (issue #3, confirmed 2026-08-28)**: the console log `S=...` is the *sequence length* and was misread as a suggested value for this parameter → a user set 19008 (> S=18921's 296 blocks × 64) → the sink guard covered nearly the whole sequence, beyond its design semantics → last frames broke + audio artifacts. **Default 1024 is correct; it is unrelated to S.** Lesson: never phrase docs/logs as "suggest ≥ actual prefix" (which steers users toward S) — all such wording has been replaced with "keep default, do not set ≈ S".

## 6. Known limits & next steps

- **Routing overhead**: Python layer ~35.6ms @ S=98512 (v1.1), theoretically ~1-2ms at GEMM level. Ideas: fused routing CUDA kernel (recompile pyd, remove the scores intermediate); intra-step inter-layer route reuse (P1, needs PSNR verification). Ideas: torch.compile / in-kernel routing (kijai fused two-level).
- **Sparse quality ceiling**: top-K guarantee (v1.1.1) fixed high-motion/new-content pruning, but density/quality balance still relies on tau/topk tuning. Very sensitive scenes (fine text / complex limbs): try topk=64 or expand dense_blocks/end_percent.
- **tau_profile (per-block tau)**: kijai supports per-layer tau (low τ for sensitive layers, high τ for insensitive); currently a global tau — can be added later.
- **Platform**: verified only V100/sm_70; Windows (pyd) + Linux (so, since v1.2.1, verified on WSL2) both supported; pyd/so must be rebuilt per target.
- **Candidate routes**: ① own keep-or-drop CUTLASS kernel; ② study NVlabs/Sana sol-engine `models/minimax_h3/A100/adapter.py` (official H3 adaptation, 3.95×-4.52×); ③ wait for official reference implementations.

## 7. Version history

- **v1.2.1 (2026-08-27)**: **Linux support** (from issue #2 community contribution, thanks lesca) — new `native/setup_linux.py` (GCC branch + auto-strip, artifact ~1MB); `nodes.py`/`sla_nodes.py`/verify scripts load the kernel per-platform (`.pyd`/`.so` with fallback); verify scripts moved into `verify/` and internal version tags removed (`verify_routing_v11.py`→`verify_routing.py` etc.), paths made relative (cross-platform); `__init__.py` bumped to 1.2.1; `.gitignore` adds `*.so`. Verified on WSL2 + V100: `verify_sla_topk_gpu.py` SLA topk density 28.9% rel-L2 0.2176, SolAttn 40.0%/0.1970, dense 27.21ms; `verify_routing.py` ALL PASS; stripped .so loads and both nodes register.
- **v1.2.0 (2026-08-23)**: new **SLA Top-K (V100)** node (`sla_nodes.py` + `sla_routing.py`, separate files, separate node; mutually exclusive with Sol-Attn): SLA top-K block sparsity (smooth-k + block-mean scoring + per-row top-K, line-by-line aligned with LightX2V `get_block_map`); **audio-segment auto guard** (PackedLayout forward bridge; fixes audio corruption under SLA sparsity, <500Hz 34%→11%); linear-compensation experiment concluded not viable (pure top-K = the lightx2v distillation target); includes v1.2 streaming chunked routing (`route_chunk`, S=174k routing VRAM 5.7→1.5GB). Real machine (same seed): 1280×736/4s 38 s/step (vs Sol 44s / dense 80s), 8s 115s (vs Sol 135s), 960×544/5s 20s (vs Sol 24s), 480p/10s 41s. Verification: 5/5 bit-exact vs LightX2V reference, chunk consistency, GPU kernel compatibility.
- **v1.1.1 (2026-08-22)**: node defaults set to the recommended values `tau=0.75, end_percent=0.9, dense_blocks="0-1,-1", h3_prefix_tokens=1024, topk_blocks=32` (out-of-the-box).
- **v1.0.0 (2026-08-20) initial release**: single node = embedded FP16Safe (fp16safe.py, v6.8.0 logic) + Sol-Attn sparse (keep-or-drop, sparse-only kernel). Measured 480p/10s at **43 s/step, no visible loss** (~1.7× vs pure FP16Safe 71-74 s/step). kijai-style parameters; dense fallback = plain SDPA; kernel source in `native/`, prebuilt pyd from Release.
