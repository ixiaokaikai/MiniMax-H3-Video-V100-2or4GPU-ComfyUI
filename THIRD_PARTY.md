# 第三方组件与许可证

本仓库捆绑的第三方代码（`custom_nodes/` 下各目录按原样保留，含各自 LICENSE 文件），以及运行时依赖的外部仓库。

## 运行时克隆/安装

| 组件 | 版本钉死 | 许可证 | 说明 |
|---|---|---|---|
| [ComfyUI 核心](https://github.com/comfyanonymous/ComfyUI) | v0.37.0 (`73c9bad`) | GPL-3.0 | `install.sh` 克隆；另含本仓库 `install/patches/` 里的 int8 反量化修复补丁 + TE-Speed 官方钩子补丁 |

## 捆绑节点包（`custom_nodes/`）

| 组件 | 来源 | 版本 | 许可证 | 作用 |
|---|---|---|---|---|
| raylight | [komikndr/raylight](https://github.com/komikndr/raylight) | 1.10.0 | Apache-2.0 | 多卡核心：Ray worker + xFuser 序列并行 + FSDP 权重分片 |
| ComfyUI-MultiGPU | [pollockjj/ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) | 作者机实测副本 | GPL-3.0 | 多卡 CLIP 拆分加载（CLIPLoaderDisTorch2MultiGPU）等 |
| comfyui-minimax-h3-audio-T8 | [T8mars/comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) | v1.81.0 (`1464a9f`) | GPL-3.0-or-later | H3 双时钟采样器、稀疏注意力补丁节点、音频节点 |
| TE-Speed-MiniMaxH3-OSS | [HELPMEEADICE/TE-Speed-MiniMaxH3-OSS](https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS) | `c1dacf4` | LGPL-3.0 | TE 缓存加速节点 + 核心钩子补丁脚本（本工作流中已断开，保留备用） |
| ComfyUI-MiniMaxH3-SolAttn-V100 | 上游作者 `aaalll12322`（MIT，2026-09-01 起暂停维护） | v1.2.0 | MIT | V100 注意力稀疏加速（本工作流中已断开，保留备用） |
| ComfyUI-sol-attn | [Saganaki22/ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn) | `930a4d6` (v0.6.2) | Apache-2.0 | H3 稀疏注意力调度补丁节点（`MiniMaxH3ScheduledSolAttentionPatch`，本工作流中已断开，保留备用） |
| Comfyui_Minimax_h3_latent_Upscaler | [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | master | MIT | latent 空间上采样节点 |

每个捆绑目录都带自己的 LICENSE 文件（`ComfyUI-MiniMaxH3-SolAttn-V100/licenses/` 里还有其二进制所依赖的 CUTLASS / BSD-3 条款）。

### 预编译二进制（需要留意）

`custom_nodes/ComfyUI-MiniMaxH3-SolAttn-V100/comfy_v100_solattn_cuda.cpython-312-x86_64-linux-gnu.so`
是本仓库里唯一的预编译二进制（V100 稀疏注意力 CUDA 内核，上游未公开内核源码，因此只能按原样捆绑）。它**只支持 Linux x86_64 + Python 3.12**（仓库已删掉随包附带的 Windows 版 `.pyd`）。核对与自检方式：

```bash
sha256sum custom_nodes/ComfyUI-MiniMaxH3-SolAttn-V100/comfy_v100_solattn_cuda.cpython-312-x86_64-linux-gnu.so
# 期望 cbcafb46b1dfb1540eb1ef365cd44e4184423f76c8a82cd8000069539127581c
# 该文件不参与本项目核心出片链路（对应节点在工作流里是断开状态）；介意二进制可整个删掉该目录，出片不受影响
```


## 本仓库自研部分（MIT，见 LICENSE）

| 文件 | 说明 |
|---|---|
| `custom_nodes/comfyui-h3-vram-autorelease/` | 两卡模式每次出片后自动杀 Ray worker 释放显存 + 失效 RayLight 缓存 |
| `custom_nodes/minimax_h3_fp16_fix.py` | V100（无 bf16 硬件）fp16 推理的精确数学修复（不降精度） |
| `install/`、`scripts/`、`workflows/`、`assets/` | 安装/启动脚本、调好的工作流、示例素材 |

## 模型（不进仓库，自行下载）

| 模型 | 发布方 | 许可 |
|---|---|---|
| MiniMax H3 各权重（DiT/VAE/LoRA/embeddings） | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | 见该仓库各文件说明 |
| int4 文本编码器 | [Merserk/MiniMax-H3-INT4-ConvRot](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot) | 见该仓库说明 |
| latent 上采样模型（可选） | [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) | MIT |
| RealESRGAN_x4plus（可选） | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | 见该仓库说明 |

> 各第三方组件以其仓库当前 LICENSE 文件为准；模型以各发布方页面条款为准。

## 许可与再分发要点

- **模型**：`Comfy-Org/MiniMax-H3`、`Merserk/MiniMax-H3-INT4-ConvRot` 在 Hugging Face 上的标签是 `license: other`（发布方自定义许可），不是 MIT/Apache；权重不进本仓库，使用时以模型页原始条款为准。
- **copyleft 组件**：ComfyUI 本体 GPL-3.0，`ComfyUI-MultiGPU` GPL-3.0，`comfyui-minimax-h3-audio-T8` GPL-3.0-or-later —— 本仓库按原样保留其源码与 LICENSE。本仓库不预打包安装产物（`install.sh` 现场克隆 ComfyUI），因此以本仓库形态分发不产生额外义务；若二次分发"已装好"的整套环境，需自行履行 GPL 源码提供义务。
- **预编译二进制**：仅 `ComfyUI-MiniMaxH3-SolAttn-V100` 的 `.so`（上游未公开源码，已给 sha256，且不在核心出片链路，可删）。
