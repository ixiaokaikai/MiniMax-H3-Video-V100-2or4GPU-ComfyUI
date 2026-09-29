# MiniMax H3 视频生成 · V100 双卡/四卡 ComfyUI 工程

一套**下载即用**的 ComfyUI 多卡环境：用 **2 张或 4 张 V100-16G** 跑 [MiniMax H3](https://huggingface.co/Comfy-Org/MiniMax-H3) 视频生成（参考图/视频/音频 → 带音频的视频，RayLight 序列并行 + FSDP 权重分片）。

- 节点包全部捆绑在仓库里，**不用自己找、不用调**
- 模型文件不进仓库（省空间），给官方链接 + 一键下载脚本
- 环境钉死在**实测通过**的版本组合：ComfyUI v0.37.0 + torch 2.10.0+cu128 + ray 2.58.0 + xfuser 0.7.0

---

## 一、需要什么

| 项 | 要求 | 说明 |
|---|---|---|
| GPU | **2 张或 4 张 V100-16G** | SXM + **NVLink** 强烈推荐（多卡带宽基础）；PCIe 卡能跑但多卡收益明显缩水 |
| 系统 | **Linux x86_64** | Ubuntu 22.04 / 24.04 最佳（实测 Ubuntu 24.04），Debian 12+ 等主流发行版均可；**不支持 Windows / macOS** |
| 驱动 | **NVIDIA ≥525.60.13**（建议最新版） | **不用装 CUDA 工具包**，torch cu128 自带运行时 |
| 内存 | 双卡模式 **≥64G**；四卡模式 ≥48G | 双卡模式 FSDP 分片 + 编码器虚拟池走 CPU，约吃 30G |
| Python | **3.12**（必需） | 部分节点包的预编译扩展只支持 cp312 |
| 磁盘 | 环境约 8G + 必需模型 44G | 建议可用空间 **≥60G** |
| 下载工具 | `aria2` 可选 | 装了会用 16 连接下载（快很多）；没有则用 `wget` 单连接 |

其他 16G 显存卡型（A100/A6000 等）大概率也能跑，但未在本机验证过。

### 模型清单（仓库不带模型，出片前必须下齐）

必需 5 个（`download_models.sh` 默认下载，合计约 44G）：

| 模型 | 大小 | 放到 | 来源 |
|---|---|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（主模型） | 21G | `models/diffusion_models/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `qwen3vl_32b_minimax_h3_int4_convrot.safetensors`（文本编码器） | 15G | `models/text_encoders/` | [Merserk/MiniMax-H3-INT4-ConvRot](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot) |
| `minimax_h3_video_vae_fp16.safetensors`（视频 VAE） | 5.2G | `models/vae/` | Comfy-Org/MiniMax-H3 |
| `minimax_h3_audio_vae_fp32.safetensors`（音频 VAE） | 0.6G | `models/vae/` | Comfy-Org/MiniMax-H3 |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`（8 步 turbo LoRA） | 2G | `models/loras/` | Comfy-Org/MiniMax-H3 |

可选 2 个（`--optional` 追加，约 0.8G）：`minimax_h3_latent_upscaler_3d_fp16.safetensors`（0.7G，latent 放大支路）、`RealESRGAN_x4plus.pth`（64M，图像空间放大支路）。

> 文本编码器固定用上表 int4 版，工作流里已接死，无需再选。风格 embedding 可选，见 Comfy-Org/MiniMax-H3 的 `embeddings/`。

## 二、怎么安装

```bash
# 0) 系统依赖（Ubuntu/Debian 一行；其他发行版装同名包）
sudo apt install -y git patch python3.12 python3.12-venv     # python3.12-venv 别漏

# 1) 拿工程
git clone <本仓库地址> && cd minimax-h3-v100-multigpu

# 2) 装环境 + 下模型（一条命令；自动克隆 ComfyUI v0.37.0、建 venv、装 torch/依赖、
#    拷节点包、打核心补丁、冒烟启动 + 40 种节点自检，然后下载必需模型）
bash install/install.sh --with-models
# 国内网络连不上 huggingface.co 时:
#   HF_ENDPOINT=https://hf-mirror.com bash install/install.sh --with-models
# 也可以分开跑: bash install/install.sh   然后   bash install/download_models.sh

# 3) 启动
bash scripts/start_comfyui_4gpu.sh     # 四卡机
bash scripts/start_comfyui_2gpu.sh     # 双卡机
```

启动脚本会**自动挑选可用卡**（显存 ≥14G 且算力 ≥7.0，机器上另插亮机卡/老卡时自动跳过），并打印选了哪几张。
想自己指定：`export CUDA_VISIBLE_DEVICES=1,2,3,4` 再跑启动脚本。停止用 `bash scripts/stop_comfyui.sh`。

浏览器打开 `http://127.0.0.1:8188`（局域网 `http://<机器IP>:8188`）→ **Workflow → Open** 选对应卡数的工作流（安装时已注册进这个菜单）→ 改提示词 → **Queue Prompt**。
模型只在点「生成」时才载入显存，启动阶段不占显存。

### 用哪个工作流

| 文件 | 配合脚本 | 卡数配置 |
|---|---|---|
| `workflows/H3-全能版本_四卡Ray版_v1.json` | `scripts/start_comfyui_4gpu.sh` | FSDP 4 卡分片，编码器 4×25% |
| `workflows/H3-全能版本_2卡Ray版_v1.json` | `scripts/start_comfyui_2gpu.sh` | FSDP 2 卡分片 + CPU offload，编码器 2×50% |

关键开关（工作流内「① 面板」，默认值已调好，可直接出片）：

| 开关 | 默认 | 说明 |
|---|---|---|
| 加速开关 | `true` | true = 8 步 turbo LoRA（快，推荐）；false = 官方 20 步（慢一倍以上，更稳） |
| 图1 开关 | `true` | 人物/身份参考图，文件 `ComfyUI/input/ref_image_1.png`；不用参考图就关掉 |
| 图2/3/4、音频1/2、视频1/2 开关 | `false` | 素材区点文件名换文件；关掉 = 该槽位不参与生成 |
| 放大开关 | `false` | false = 直出；true = 走 ESRGAN（需可选模型） |
| 分辨率 | `#115` | 0.4MP = 竖版 480×864 / 0.9MP = 竖版 720×1280 / 0.98MP = 1344×768（官方 768p） |
| 时长 | `#132`（秒） | 24fps 自动换算帧数（17n+5 对齐） |
| 输出尺寸 | `#963` 宽 / `#964` 高 | 默认 1280×720（竖版素材会自动按比例出竖版尺寸） |

提示词写「① 面板」的多行文本框（`#138`）；素材在提示词里用 `<Picture 1>`、`<Video 1>`、`<Audio 1>` 引用，编号要和实际打开的素材对上。参考视频要求 24fps、2~15 秒。

### 参考素材（`ComfyUI/input/`）

`install.sh` 会把仓库自带的**中性占位素材**放进去，所以装完即可直接提交跑通；换成你自己的同名文件即可。
工作流预填的文件名：

| 槽位 | 文件名 | 默认开关 |
|---|---|---|
| 图1（身份参考） | `ref_image_1.png` | 开（占位图） |
| 图2 / 图3 / 图4 | `h3_frame_ref_1.png` / `h3_frame_ref_2.png` / `h3_frame_ref_3.png` | 关 |
| 音频1 / 音频2 | `h3_ref_voice_1.wav` / `ref_audio_2.wav` | 关（1 秒静音） |
| 视频1 / 视频2 | `ref_video_1.mp4` / `ref_video_2.mp4` | 关（2 秒 24fps 灰底） |

占位素材在 `assets/` 下（`ref_image_1.png` + `placeholders/`），只是让工作流能提交跑通的空壳，别拿它出片。

## 三、成绩表现

硬件：4×V100-SXM2-16G（NVLink 全通）+ 64G 内存 + Ubuntu 24.04。基准口径：**0.4MP 生成 → 空间放大到约 0.92MP 输出**（工作流默认档；16:9 时即 1280×720，9:16 竖版时 736×1320）。

| 模式 | 5 秒 / 约 0.92MP | 说明 |
|---|---|---|
| **双卡（2×16G）** | **约 7 分钟** | 最低可行配置。自带**自动卸显存**插件：每次出片后杀掉 Ray worker、显存归还系统，双卡机可与其他 GPU 服务共存 |
| **四卡（4×16G）** | **约 3 分 20 秒** | 全卡并行，比双卡快约一倍 |

干净安装后的**首单实测**（本仓库安装脚本在 Ubuntu 24.04 + 4×V100 上装完直接出片，非空跑）：

| 实测项 | 四卡 | 双卡 |
|---|---|---|
| 整单耗时 | 261 秒 | 408 秒 |
| 其中 8 步去噪 | 135 秒（约 17 s/步） | 约 355 秒（约 38 s/步） |
| 产出 | 736×1320 / 24fps / 124 帧 / 5.17 秒 / 含音轨 | 同左 |

- 去噪段四卡约快 **2 倍**；整单只快 1.6 倍，因为编码 → 去噪 → VAE 解码/封装这些阶段物理上是串行的，不能靠加卡缩短。
- 首次出片要多花 2~3 分钟（起 Ray 集群 + FSDP 装载 20G 模型），上表首单实测已含这段；双卡模式每单跑完自动卸显存，下一单必然是冷启动。
- 换卡型/内存/PCIe 会有差异，属正常；速度数字为作者机器实测，仅供参考。

## 四、注意事项

**装/跑之前**

- **权重类型只能用 int8_convrot**（`*_int8_convrot.safetensors`）。本工程的加速链 + 核心补丁都是按 **int8_convrot + ComfyUI v0.37.0** 调的，换成 fp8/bf16/GGUF 权重不保证能跑；想换，请先在副本环境里试。
- **环境钉死版本**，升级 = 重建：`rm -rf ComfyUI .venv` 后重跑 `install.sh`（模型不用重下，节点包和工作流会自动重新拷）。想换 ComfyUI 版本要改 `install.sh` 里的 `COMFY_REV`，但**先确认 `install/patches/` 里的核心补丁与新版本兼容**（补丁按 v0.37.0 生成）。
- **网络**：连不上 huggingface.co 时加 `HF_ENDPOINT=https://hf-mirror.com`；ComfyUI 克隆慢可加 `GITHUB_MIRROR=https://ghproxy.net/`；pip 慢可设 `PIP_INDEX_URL=<你的源>/simple/`（设了国内源时脚本会自动用官方 PyPI 兜底取缺的包）。装 `aria2` 下载模型会快很多。
- **装的时候日志几分钟没动静是正常的**：装 torch 要下约 4G 依赖，pip 进度条写不进重定向的日志。想确认还在干活：`ps aux | grep pip`。模型下载中断直接重跑，会断点续传。

**跑的时候**

- **首次出片多 2~3 分钟**：要起 Ray 集群 + FSDP 装载 20G 模型；双卡模式每单跑完自动卸显存，所以**双卡每单都是冷启动**。
- **系统内存**：双卡模式约吃 30G，可用内存不足 32G 会被内核 OOM 直接杀掉（无明显报错）——换大内存或改跑四卡模式（内存压力小）。显存水位：双卡每卡最满约 12G/16G，四卡约 15G/16G，属设计内。
- **参考素材那个槽位开关关着，文件也必须在**：`install.sh` 已放好占位素材，删掉会整单校验失败（报 "Invalid image/audio/video file"）。换自己的素材就同名覆盖。
- **分辨率和时长是成本主因**：默认 0.4MP 生成再放大输出；想直接生成 720p，把 `#115` 切 0.9MP（或 0.98MP = 官方 768p），显存和时间随分辨率上升。
- **网页上出现"缺失节点类型 / 红框"**：先跑 `bash scripts/check_nodes.sh`，它会起一个临时实例逐个核对工作流用到的 40 种节点，缺哪个直接列出来；报错细节看工程根目录的 `comfyui.log`。
- **卡数不是正好 2 或 4**（3 卡机 / 6 卡机 / 混插亮机卡）：能装。启动脚本自动挑可用卡（显存 ≥14G 且算力 ≥7.0）；想指定就 `export CUDA_VISIBLE_DEVICES=1,2,3,4` 再启动。
- **没有 NVLink 也能跑**（走 PCIe + NCCL），但多卡并行收益明显缩水，速度会向双卡靠拢。

## 目录结构

```
minimax-h3-v100-multigpu/
├── install/install.sh          # 一键装环境（含冒烟启动 + 40 种节点自检）
├── install/download_models.sh  # 一键下载模型（断点续传）
├── install/patches/            # ComfyUI 核心补丁（int8 反量化修复、TE-Speed 钩子）
├── workflows/                  # 双卡 / 四卡 工作流
├── custom_nodes/               # 全部节点包（install 时拷入 ComfyUI）
├── assets/                     # 参考素材占位文件（install 时放入 input/）
├── scripts/                    # 四卡/双卡启动、停止、节点自检
└── THIRD_PARTY.md              # 第三方组件与许可证
```

安装后 ComfyUI 本体在 `ComfyUI/`（git 克隆，钉死 v0.37.0），虚拟环境在 `.venv/`。

## 免责声明

- 模型权重不在本仓库，请从上述官方链接自行下载，并遵守各发布方的许可与使用条款；生成内容的合规性由使用者负责。
- 第三方节点包按原样捆绑（含各自 LICENSE），清单见 [THIRD_PARTY.md](THIRD_PARTY.md)。
