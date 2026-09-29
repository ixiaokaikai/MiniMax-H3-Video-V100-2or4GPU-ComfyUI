# MiniMax H3 Video Generation · 2/4× V100 ComfyUI Bundle

[**中文**](README.md) | **English**

A **ready-to-run** ComfyUI multi-GPU environment that runs [MiniMax H3](https://huggingface.co/Comfy-Org/MiniMax-H3) video generation on **2 or 4 × V100-16G** GPUs (reference image/video/audio → video with audio, via RayLight sequence parallelism + FSDP weight sharding).

- All custom node packs are bundled — **nothing to hunt for, nothing to debug**
- Model weights are **not** in the repo (saves space): official links + a one-shot download script
- The environment is pinned to a **verified-working** combination: ComfyUI v0.37.0 + torch 2.10.0+cu128 + ray 2.58.0 + xfuser 0.7.0

---

## 1. What you need

| Item | Requirement | Notes |
|---|---|---|
| GPU | **2 or 4 × V100-16G** | SXM + **NVLink** strongly recommended (multi-GPU bandwidth); PCIe cards work but the scaling gain drops sharply |
| OS | **Linux x86_64** | Ubuntu 22.04 / 24.04 best (verified on 24.04); Debian 12+ and similar distros work. **No Windows / macOS** |
| Driver | **NVIDIA ≥525.60.13** (latest recommended) | **No CUDA toolkit needed** — torch cu128 ships its own runtime |
| RAM | 2-GPU mode **≥64G**; 4-GPU mode ≥48G | 2-GPU mode needs ~30G (FSDP shards + text-encoder virtual pool on CPU) |
| Python | **3.12** (required) | Some bundled node packs ship cp312-only prebuilt extensions |
| Disk | ~8G for the environment + 44G for required models | **≥60G** free recommended |
| Downloader | `aria2` optional | With it, model downloads use 16 connections (much faster); otherwise `wget` |

Other 16G cards (A100/A6000 etc.) will most likely work, but have not been verified on the author's machine.

### Models (not in the repo — required before generating)

Required 5 (downloaded by `download_models.sh`, ~44G total):

| Model | Size | Destination | Source |
|---|---|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (main model) | 21G | `models/diffusion_models/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `qwen3vl_32b_minimax_h3_int4_convrot.safetensors` (text encoder) | 15G | `models/text_encoders/` | [Merserk/MiniMax-H3-INT4-ConvRot](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot) |
| `minimax_h3_video_vae_fp16.safetensors` (video VAE) | 5.2G | `models/vae/` | Comfy-Org/MiniMax-H3 |
| `minimax_h3_audio_vae_fp32.safetensors` (audio VAE) | 0.6G | `models/vae/` | Comfy-Org/MiniMax-H3 |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` (8-step turbo LoRA) | 2G | `models/loras/` | Comfy-Org/MiniMax-H3 |

Optional 2 (add `--optional`, ~0.8G total): `minimax_h3_latent_upscaler_3d_fp16.safetensors` (0.7G, latent upscale branch), `RealESRGAN_x4plus.pth` (64M, image-space upscale branch).

> The text encoder is hard-wired to the int4 build above. Optional style embeddings live under `embeddings/` in Comfy-Org/MiniMax-H3.

## 2. How to install

```bash
# 0) System dependencies (one line on Ubuntu/Debian; install the same packages elsewhere)
sudo apt install -y git patch python3.12 python3.12-venv     # do not skip python3.12-venv

# 1) Get the project
git clone https://github.com/ixiaokaikai/MiniMax-H3-Video-V100-2or4GPU-ComfyUI.git && cd MiniMax-H3-Video-V100-2or4GPU-ComfyUI

# 2) Install the environment + download models (single command: clones ComfyUI v0.37.0,
#    builds a venv, installs torch/deps, copies node packs, applies core patches,
#    smoke-tests the server + checks 40 node types, then downloads the required models)
bash install/install.sh --with-models
# Cannot reach huggingface.co from your network:
#   HF_ENDPOINT=https://hf-mirror.com bash install/install.sh --with-models
# Or run the two steps separately: bash install/install.sh  then  bash install/download_models.sh

# 3) Start
bash scripts/start_comfyui_4gpu.sh     # 4-GPU machine
bash scripts/start_comfyui_2gpu.sh     # 2-GPU machine
```

The start script **auto-selects usable GPUs** (≥14G VRAM and compute capability ≥7.0, skipping display/older cards) and prints which ones it picked. To pick manually: `export CUDA_VISIBLE_DEVICES=1,2,3,4` before starting. Stop with `bash scripts/stop_comfyui.sh`.

Open `http://127.0.0.1:8188` (LAN: `http://<host-ip>:8188`) → **Workflow → Open** and pick the workflow matching your GPU count (registered in that menu during install) → edit the prompt → **Queue Prompt**. Models are loaded into VRAM only when you hit generate; startup uses none.

### Which workflow

| File | Start script | GPU config |
|---|---|---|
| `workflows/H3-全能版本_四卡Ray版_v1.json` | `scripts/start_comfyui_4gpu.sh` | FSDP sharded over 4 GPUs, text encoder 4×25% |
| `workflows/H3-全能版本_2卡Ray版_v1.json` | `scripts/start_comfyui_2gpu.sh` | FSDP over 2 GPUs + CPU offload, text encoder 2×50% |

Key switches (in-panel "① 面板"; defaults are tuned and generate out of the box):

| Switch | Default | Meaning |
|---|---|---|
| turbo / accelerate | `true` | true = 8-step turbo LoRA (fast, recommended); false = official 20 steps (more than 2× slower, steadier) |
| reference image 1 | `true` | identity reference, file `ComfyUI/input/ref_image_1.png`; turn off if unused |
| image 2/3/4, audio 1/2, video 1/2 | `false` | click the filename in the node to swap files; off = that slot does not contribute |
| upscale | `false` | false = direct output; true = ESRGAN branch (needs the optional model) |
| resolution | `#115` | 0.4MP = 480×864 portrait / 0.9MP = 720×1280 portrait / 0.98MP = 1344×768 (official 768p) |
| duration | `#132` (seconds) | converted to frames at 24fps (aligned to 17n+5) |
| output size | `#963` width / `#964` height | default 1280×720 |

Write the prompt in the multiline box in "① 面板" (`#138`); reference material is addressed inside the prompt as `<Picture 1>`, `<Video 1>`, `<Audio 1>` — the numbers must match the slots you actually enabled. Reference videos must be 24fps, 2–15 s.

### Reference material (`ComfyUI/input/`)

`install.sh` drops in **neutral placeholder assets** shipped with the repo, so a fresh install can be queued immediately; overwrite them with your own files (same names) whenever you like.
Prefilled filenames in the workflow:

| Slot | Filename | Default |
|---|---|---|
| image 1 (identity) | `ref_image_1.png` | on (placeholder) |
| image 2 / 3 / 4 | `h3_frame_ref_1.png` / `h3_frame_ref_2.png` / `h3_frame_ref_3.png` | off |
| audio 1 / 2 | `h3_ref_voice_1.wav` / `ref_audio_2.wav` | off (1 s silence) |
| video 1 / 2 | `ref_video_1.mp4` / `ref_video_2.mp4` | off (2 s 24fps gray) |

Placeholders live under `assets/` (`ref_image_1.png` + `placeholders/`) — they only exist so the graph can be queued, not for actual output.

## 3. Performance

Hardware: 4×V100-SXM2-16G (full NVLink) + 64G RAM + Ubuntu 24.04. Basis: **generate at 0.4MP, then spatially upscale to ≈0.92MP output** (the workflow default; 1280×720 in 16:9, 736×1320 in the default 9:16 portrait).

| Mode | 5 s / ≈0.92MP | Notes |
|---|---|---|
| **2 × V100-16G** | **≈7 min** | Minimum viable setup. Bundled **auto VRAM-release** plugin: kills the Ray workers after each render so VRAM returns to the system — coexists with other GPU services |
| **4 × V100-16G** | **≈3 min 20 s** | All GPUs in parallel, roughly 2× faster than 2 GPUs |

Measured on a **clean install, first render** (this repo's installer, Ubuntu 24.04 + 4×V100, real output — not a dry run):

| | 4 GPU | 2 GPU |
|---|---|---|
| Whole job | 261 s | 408 s |
| of which 8-step denoise | 135 s (~17 s/step) | ~355 s (~38 s/step) |
| Output | 736×1320 / 24fps / 124 frames / 5.17 s / with audio | same |

- The denoise stage is ~2× faster on 4 GPUs; the whole job only 1.6×, because encode → denoise → VAE decode/mux are physically serial and cannot be shortened by adding GPUs.
- The first render costs 2–3 min extra (Ray cluster startup + FSDP loading of the 20G model) — included in the numbers above. 2-GPU mode releases VRAM after every job, so every 2-GPU render starts cold.
- Different cards / RAM / PCIe will differ. Numbers are the author's measurements, for reference only.

## 4. Things to know

**Before installing / running**

- **Only `int8_convrot` weights are supported** (`*_int8_convrot.safetensors`). The acceleration chain and core patches are tuned for **int8_convrot + ComfyUI v0.37.0**; fp8/bf16/GGUF weights are not guaranteed. Try those in a copy of the environment.
- **Versions are pinned.** Upgrading = rebuilding: `rm -rf ComfyUI .venv` then re-run `install.sh` (models are kept; node packs and workflows are re-copied). To move to another ComfyUI version, change `COMFY_REV` in `install.sh`, but **first confirm the core patches in `install/patches/` still apply** (they were generated against v0.37.0).
- **Networks:** if huggingface.co is unreachable use `HF_ENDPOINT=https://hf-mirror.com`; slow ComfyUI clone: `GITHUB_MIRROR=https://ghproxy.net/`; slow pip: `PIP_INDEX_URL=<mirror>/simple/` (the script falls back to official PyPI for packages your mirror lacks). Installing `aria2` speeds model downloads up a lot.
- **Minutes of silence during install is normal**: torch pulls ~4G of dependencies and pip's progress bar is not written into a redirected log. To confirm it is alive: `ps aux | grep pip`. Interrupted model downloads resume — just re-run.

**While running**

- **First render takes 2–3 min extra** (Ray cluster + 20G FSDP load). 2-GPU mode releases VRAM after each job, so *every* 2-GPU render is a cold start.
- **RAM:** 2-GPU mode needs ~30G; with <32G available the kernel OOM-killer may kill it with no visible error — use more RAM or run 4-GPU mode (lighter). VRAM ceiling: ≈12G/16G per card in 2-GPU mode, ≈15G/16G in 4-GPU mode; both are by design.
- **Placeholder assets must exist even with their switches off** — deleting them fails prompt validation ("Invalid image/audio/video file"). Replace with your own files using the same names.
- **Resolution and duration dominate cost.** Default = generate at 0.4MP then upscale; to generate 720p directly set `#115` to 0.9MP (or 0.98MP = official 768p). VRAM and time grow with resolution.
- **"Missing node type" / red nodes in the UI:** run `bash scripts/check_nodes.sh` — it boots a temporary instance and checks all 40 node types used by the workflows, listing whatever is missing; details in `comfyui.log` at the project root.
- **GPU count other than exactly 2 or 4** (3-card, 6-card, mixed display card): fine. The start script auto-selects usable GPUs (≥14G VRAM, compute capability ≥7.0); to pin them: `export CUDA_VISIBLE_DEVICES=1,2,3,4`.
- **No NVLink still works** (PCIe + NCCL), but multi-GPU scaling drops noticeably and speed approaches 2-GPU figures.

## Layout

```
MiniMax-H3-Video-V100-2or4GPU-ComfyUI/
├── install/install.sh          # one-shot environment install (smoke test + 40-node check)
├── install/download_models.sh  # one-shot model download (resumable)
├── install/patches/            # ComfyUI core patches (int8 dequant fix, TE-Speed hook)
├── workflows/                  # 2-GPU / 4-GPU workflows
├── custom_nodes/               # all node packs (copied into ComfyUI at install time)
├── assets/                     # placeholder reference assets (copied into input/)
├── scripts/                    # start (2/4 GPU), stop, node self-check
└── THIRD_PARTY.md              # third-party components and licenses
```

After installation, ComfyUI itself lives in `ComfyUI/` (git clone, pinned to v0.37.0) and the virtualenv in `.venv/`.

## Disclaimer

- Model weights are not in this repository. Download them from the official links above and follow each publisher's license and terms of use; responsibility for generated content rests with the user.
- Third-party node packs are bundled as-is, each with its own LICENSE — see [THIRD_PARTY.md](THIRD_PARTY.md).
