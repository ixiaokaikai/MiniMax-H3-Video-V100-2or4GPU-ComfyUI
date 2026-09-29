# MiniMax H3 视频生成 · V100 双卡/四卡 ComfyUI 工程

一套**下载即用**的 ComfyUI 多卡环境，用 **2 张或 4 张 V100-16G** 跑 [MiniMax H3](https://huggingface.co/Comfy-Org/MiniMax-H3) 视频生成（参考图/视频/音频 → 带音频的视频，RayLight 序列并行 + FSDP 权重分片）。

- 全部节点包已捆绑进仓库，**不用自己找、不用调试**
- 模型文件**不进仓库**（省空间），给好官方下载链接 + 一键下载脚本
- 环境钉死在作者机器上**实测通过**的版本组合（ComfyUI v0.37.0 + torch 2.10.0+cu128 + ray 2.58 + xfuser 0.7.0）

---

## 速度参考（作者机器实测）

硬件：4×V100-SXM2-16G（NVLink 全通）+ 64G 内存 + Ubuntu 24.04。

| 模式 | 5 秒 / 720p（输出 1280×720） | 备注 |
|---|---|---|
| **四卡** | **约 3 分 20 秒** | 全卡并行，最快 |
| **两卡** | **约 7 分钟** | 只用 2 张卡，可与别的常驻 GPU 服务共存 |

其他口径：4 卡 480p(864×480)/2 秒/8 步 → 缩放 720p 约 66 秒。
首次出片包含 Ray 集群启动 + FSDP 装载 20G 模型，比上表多 2~3 分钟（冷启动）；两卡模式每单跑完自动卸显存，下一单必然冷启动。
你的卡型/内存不同会有差异，属正常。

## 硬件要求

| 项 | 要求 | 说明 |
|---|---|---|
| GPU | **2 张或 4 张 V100-16G** | SXM + **NVLink** 强烈推荐（多卡并行的带宽基础）；PCIe 卡能跑但明显慢 |
| 系统 | **Linux x86_64**（Ubuntu 22.04/24.04 最佳，作者实测 Ubuntu 24.04；Debian 12+ 等主流发行版均可） | 预编译扩展按 Linux x86_64 打包；**Windows/macOS 不支持** |
| NVIDIA 驱动 | **≥525.60.13**（建议直接装最新版） | **不用装 CUDA 工具包**：torch cu128 自带运行时 |
| 系统内存 | 两卡模式 **≥64G**（32G 勉强）；四卡模式 ≥48G | 两卡模式 FSDP 分片 + 编码器虚拟池走 CPU，约吃 30G |
| Python | **3.12**（必需） | 个别节点包预编译扩展仅 cp312；Ubuntu: `sudo apt install python3.12 python3.12-venv` |
| 磁盘 | 环境约 8G + 必需模型 44G | 建议可用空间 **≥60G** |

> 其他 16G 显存的卡（A100/A6000 等）大概率也能跑，但未在作者机器上验证，出问题请先查 ComfyUI/raylight 的 Issue。

## 快速开始（三步）

```bash
# 0) 拿到工程
git clone <本仓库地址> && cd minimax-h3-v100-multigpu

# 1) 装环境（自动克隆 ComfyUI v0.37.0、建 venv、装 torch/依赖、拷节点包、打补丁、冒烟验证）
bash install/install.sh

# 2) 下载模型（必需 5 个约 44G；可选 2 个再加约 0.8G，断点续传）
bash install/download_models.sh            # 国内网络可加前缀:
# HF_ENDPOINT=https://hf-mirror.com bash install/download_models.sh

# 3) 启动 + 出片
bash scripts/start_comfyui_4gpu.sh         # 四卡机
# bash scripts/start_comfyui_2gpu.sh        # 双卡机 / 只想用 2 张卡
```

浏览器打开 `http://127.0.0.1:8188`（局域网 `http://<机器IP>:8188`），
**Workflow → Open** 选对应卡数的工作流（安装时已自动注册进这个菜单），改提示词，点 **Queue Prompt**。
模型只在点「生成」时才载入显存，启动阶段不占显存。

> **English quick start** — Linux x86_64, 2 or 4 × V100-16G (NVLink recommended), NVIDIA driver ≥525.60.13, Python 3.12, ≥64G RAM for 2-GPU mode.
> `git clone … → bash install/install.sh → bash install/download_models.sh (use HF_ENDPOINT=https://hf-mirror.com in CN) → bash scripts/start_comfyui_4gpu.sh (or 2gpu) → open http://127.0.0.1:8188`.

## 模型清单（仓库不带模型，首跑前必读）

必需 5 个（`download_models.sh` 默认下载，约 44G）：

| 模型 | 大小 | 放到 | 来源 |
|---|---|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（主模型） | 21G | `models/diffusion_models/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `qwen3vl_32b_minimax_h3_int4_convrot.safetensors`（文本编码器·默认） | 15G | `models/text_encoders/` | [Merserk/MiniMax-H3-INT4-ConvRot](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot) |
| `minimax_h3_video_vae_fp16.safetensors`（视频 VAE） | 5.2G | `models/vae/` | Comfy-Org/MiniMax-H3 |
| `minimax_h3_audio_vae_fp32.safetensors`（音频 VAE） | 0.6G | `models/vae/` | Comfy-Org/MiniMax-H3 |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`（8 步 turbo LoRA） | 2G | `models/loras/` | Comfy-Org/MiniMax-H3 |

可选 2 个（`download_models.sh --optional`，约 0.8G）：

| 模型 | 大小 | 用途 |
|---|---|---|
| `minimax_h3_latent_upscaler_3d_fp16.safetensors` | 0.7G | latent 放大支路（[LBH-123-AI](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)） |
| `RealESRGAN_x4plus.pth` | 64M | ESRGAN 图像空间超分支路 |

> 文本编码器固定用 int4 省内存版（上面必需清单第 2 个），工作流里已接死，无需再选。

> 各模型均从**官方/发布方链接**自行下载，遵循其各自的许可与使用条款。
> 风格 embedding（`minimaxh3_*.safetensors`）可选，在 Comfy-Org/MiniMax-H3 的 `embeddings/` 目录。

## 工作流怎么用

两份工作流节点相同，区别只在卡数配置（2 卡版 FSDP 2 卡分片 + CPU offload，编码器 2×50%；四卡版 4×25%，见工作流内笔记节点）：

| 文件 | 配合 |
|---|---|
| `workflows/H3-全能版本_四卡Ray版_v1.json` | `scripts/start_comfyui_4gpu.sh` |
| `workflows/H3-全能版本_2卡Ray版_v1.json` | `scripts/start_comfyui_2gpu.sh` |

关键开关（工作流内「① 面板」，默认值已调好，能直接出片）：

| 开关 | 默认 | 说明 |
|---|---|---|
| 加速开关 | `true` | true=8 步 turbo LoRA（快，推荐）/ false=官方 20 步（慢一倍以上，更稳） |
| 图1 开关 | `true` | 人物/身份参考图。**默认开着**，参考图放 `ComfyUI/input/ref_image_1.png`（仓库已带一张占位示例图）；不用参考图就关掉 |
| 图2/3/4、音频1/2、视频1/2 开关 | `false` | 素材区点文件名换文件；关掉 = 完全不读文件、不花时间 |
| 放大开关 | `false` | false=直出 / true=走 ESRGAN（需可选模型 8） |
| 分辨率 | `#115 ResolutionSelector` | 0.4MP=864×480 / 0.9=1280×720 / 0.98=1344×768（官方 768p） |
| 时长 | `#132`（秒数） | 24fps 自动换算帧数（17n+5 对齐） |
| 输出尺寸 | `#963` 宽 / `#964` 高 | 默认 1280×720 |

提示词写「① 面板」里那个多行文本框（`#138`）；参考素材在提示词里用 `<Picture 1>`、`<Video 1>`、`<Audio 1>` 引用，**编号要和你实际打开的素材对得上**。

参考视频要求 24fps、2~15 秒。参考图/视频/音频放 `ComfyUI/input/`，节点上点文件名改引用。

## 目录结构

```
minimax-h3-v100/
├── README.md                  # 本文档
├── THIRD_PARTY.md             # 第三方组件清单与许可证
├── install/
│   ├── install.sh             # 一键装环境
│   ├── download_models.sh     # 一键下载模型
│   └── patches/               # ComfyUI 核心补丁 (int8 反量化修复)
├── workflows/                 # 两份工作流 (2卡/4卡)
├── custom_nodes/              # 全部节点包 (捆绑, install 时拷入 ComfyUI)
│   ├── raylight/                          # 多卡核心 (Ray/xFuser/FSDP)
│   ├── ComfyUI-MultiGPU/                  # 多卡 CLIP/检查点加载
│   ├── comfyui-minimax-h3-audio-T8/       # H3 双时钟采样器等
│   ├── TE-Speed-MiniMaxH3-OSS/            # TE 加速 (+核心钩子补丁)
│   ├── ComfyUI-MiniMaxH3-SolAttn-V100/    # V100 稀疏注意力 (本工作流中已断开)
│   ├── Comfyui_Minimax_h3_latent_Upscaler/ # latent 上采样
│   ├── comfyui-h3-vram-autorelease/       # 2卡模式跑完自动卸显存
│   └── minimax_h3_fp16_fix.py             # V100 无 bf16 的 fp16 精度修复
├── assets/ref_image_1.png     # 占位示例参考图 (install 时放入 input/)
└── scripts/
    ├── start_comfyui_4gpu.sh  # 四卡启动
    ├── start_comfyui_2gpu.sh  # 两卡启动 (自动卸显存)
    └── stop_comfyui.sh        # 停止 + 释放显存
```

安装后 ComfyUI 本体在 `ComfyUI/`（git 克隆，钉死 v0.37.0），虚拟环境在 `.venv/`。

## FAQ

**OOM / 被内核杀掉**
两卡模式系统内存不够（<32G 可用）时会被 OOM killer 直接杀掉进程（无报错）。换大内存或改跑四卡模式（内存压力小）。显存侧两卡版每卡最满 ≈12G/16G，四卡版 ≈15G/16G，属设计内水位。

**首单特别慢**
第一次出片要起 Ray 集群 + FSDP 装载 20G 模型，比稳态多 2~3 分钟，正常。

**提示"缺少节点类型 / missing node type"**
`custom_nodes` 里有某个包 import 报错。看 `comfyui.log`（工程根目录）里该包的报错行，一般是 Python 版本不是 3.12 或 `pip install -e custom_nodes/raylight` 没跑成功。重装：删掉 `ComfyUI` 和 `.venv` 重跑 `install.sh`。

**国内下载模型慢 / 卡住**
`HF_ENDPOINT=https://hf-mirror.com bash install/download_models.sh`；或装 aria2 走 16 连接。下载完核对文件大小（清单里标了）。

**pip / GitHub 慢或连不上**
- pip：`export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/` 后再跑 `install.sh`（对所有 pip 生效）
- GitHub（克隆 ComfyUI）：`GITHUB_MIRROR=https://ghproxy.net/ bash install/install.sh`（换任意可用的 GitHub 加速前缀）

**怎么升级 / 重装**
环境全部钉死版本，升级 = 重建：`rm -rf ComfyUI .venv` 后重跑 `install.sh`（模型不动，节点包和工作流会自动重新拷贝）。想换 ComfyUI 新版本：改 `install.sh` 里的 `COMFY_REV` 再重建，但**换版本前请确认核心补丁和新版兼容**（`install/patches/` 按 v0.37.0 生成）。

**卡数不是正好 2/4（比如 6 卡机 / 3 卡机）**
能装。两卡模式默认用前 2 张卡，四卡模式默认用前 4 张；想指定哪几张卡：`export CUDA_VISIBLE_DEVICES=1,2`（或 `2,3,4,5`）再启动。

**NVLink 没有会怎样**
能跑（走 PCIe + NCCL），但四卡并行收益明显缩水，速度向两卡靠拢。SXM 版 V100 自带 NVLink，最省事。

**想改 ComfyUI 版本 / 换权重（fp8、bf16、GGUF）**
本工程的加速链和核心补丁都是按 **int8_convrot 权重 + v0.37.0** 调的，换版本/换权重类型请先在副本环境里试。

## 免责声明

- 模型权重**不在本仓库**，请自行从上述官方链接下载，并遵守各模型发布方的许可与使用条款；生成内容的合规性由使用者负责。
- 第三方节点包按原样捆绑（含各自 LICENSE），清单见 [THIRD_PARTY.md](THIRD_PARTY.md)。
- 速度数字为作者机器实测值，仅供参考，不构成任何承诺。
