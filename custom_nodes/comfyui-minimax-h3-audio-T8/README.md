# MiniMax H3 Audio T8

简体中文 | [English](README_EN.md)

2026-09-15 **v1.81.0**：新增两个互不依赖 KJNodes 的 H3 低显存 EXP 节点：按头分组并提前释放中间量的 `Low VRAM Attention`，以及仅在长 packed token 序列上启用的 `Chunk FeedForward`。固定 3 秒短片中，默认 `head_chunks=4 + chunks=2` 相对基线少用约 306.73 MiB 峰值显存（2.06%），耗时增加约 2.92%；不是通用显存或提速承诺。Speech Studio 新建参考音色工作流会自动保守裁掉低能量对齐留白，旧工作流保存的显式设置不变；Sol 验证探针同步适配当前 `sink_blocks` 接口。详见[更新说明](docs/RELEASE_1.81.0.md)。

2026-09-14 **v1.80.0**：新增 SelfLift LOW4→学习型 latent 放大→HIGH4、独立 LOW/HIGH MODEL、TST 和节点内长视频组合，并交付 9 份已完成绑定样片验收的 SelfLift/TaoMate EXP 工作流。EAV 默认采用已审的 `tau=8、15%–90%`；想要更高动态可逐步提高，但必须复查身份、形变、闪烁和声音。所有结论仅限绑定短片，不作通用提速、显存或画质承诺。详见[更新说明](docs/RELEASE_1.80.0.md)。

2026-09-14 **v1.79.6**：完成星光以外的正式Topaz闭环。14个高清模型和5个插帧模型均已走通真实官方生产worker；新增明确的10-bit SDR→HEVC Main10保持路径、多卡GPU序号和“先高清再插帧”串联工作流，并修复无损母版被8-bit审计误拒绝。HDR、VFR、隔行和旋转元数据仍明确拒绝，不静默转换；商业权重只下载到本机，不随仓库发布。详见[更新说明](docs/RELEASE_1.79.6.md)。

2026-09-14 **v1.79.5**：继续修复Topaz交付边界。手动参数现在按模型定义过滤，固定预设模型不再因不支持的滑块报错；权重检查遵循正式网络模板，修复Rhea、Nyx XL和Theia误报未下载；运行前按音频编码选择MP4/MKV；10bit/HDR不再静默降成8bit；ComfyUI进度条显示推理帧与审计阶段。Apollo已完成真实短片机械验证，画质仍待人审。详见[更新说明](docs/RELEASE_1.79.5.md)。

2026-09-14 **v1.79.4**：修复正式Topaz把15秒视频保存成12.9GB无损MOV、再接SaveVideo写满磁盘的问题。高清节点现在默认一次性GPU编码高质量H.264 MP4，禁止再接SaveVideo；超大无损母版保留为高级审计选项。新增模型用途说明、自动/手动参数及独立中文滑块，并新增正式`tvai_fi` 2x/4x插帧节点和工作流。使用时无需保持Topaz软件打开，但须已登录并先下载模型。详见[更新说明](docs/RELEASE_1.79.4.md)。

2026-09-13 GitHub 主线更新：指定LTX成片、Tao新对白5秒恢复片及最新[Dance 4+4／前段成片LOW上下文8秒样片](docs/DANCE_ACCEPTED_PICTURE_20260913.md)均已完整人审接受，不重复生成或审片。Dance正确示例与接缝防回归说明已保存；旧Dance与两条深度失败实验没有进入正式工作流。Tao原生成任务收尾触发内存保护仍保留失败，媒体接受不等于新封装完整GPU可靠性认证。[可选深度参考](docs/DEPTH_REFERENCE_EXP.md)仅作负实验说明；另见[资源清理、AdaLN、HJL与音频边界](docs/LOCAL_RELIABILITY_SCOPE_20260913.md)。

本次更新 **v1.79.0**：新增[双模型4＋潜空间放大＋4内循环](docs/DUAL_MODEL_LONG_VIDEO_EXP.md)、[正式Topaz高清后处理](docs/TOPAZ_EXP.md)和[R1可靠性修复](docs/R1_RELIABILITY_20260911.md)。新双模型模板默认两段8秒，已评样片的画面、声音、口型和接缝可接受，旧工作流不变。OpenVDN不再因上游Sol的注意力override单独报错；VDN仍使用自身注意力计算，不宣称两种算法叠加提速。星光暂停，未作为已完成能力发布。详见[更新说明](docs/RELEASE_1.79.0.md)。

这是一个面向 MiniMax H3 的 ComfyUI 节点包。它不只做文生视频，还把图生视频、首尾帧、参考图、参考音频、长视频、口型、加速和成片修复整理成可以直接使用的工作流。

当前版本：**1.81.0** · 336 个节点 · GPL-3.0-or-later

1.78.1 整理了仓库目录：实现代码集中到 `h3_t8/`，首页不再堆满 Python 文件。旧工作流、模型位置、节点参数和三个 TRT 命令入口保持不变；不需要重新下载模型。详见 [目录结构与更新说明](docs/REPOSITORY_LAYOUT.md)。

1.78.0 新增的 [TRT VAE 可选后端](docs/TRT_VAE_EXP.md) 和 [5 份工作流](examples/workflows/30-trt-vae)继续保留：安装检查、本机编译、仅视频解码、完整编解码，以及同潜空间双路对照。已审短片和文字对照获得整体认可，保留 EXP；长片未验证。需要独立 TensorRT 环境和本机编译引擎，旧工作流、生成模型和音频 VAE 不变。

**不保证整流程更快。** 本机短片包含加载、解码和保存时，原生约 30.54 秒、TRT 约 31.33 秒，没有总耗时收益；不能把裸解码提速当成整条生成提速。安装与限制见 [1.78.0 发布说明](docs/RELEASE_1.78.0.md)。

1.77.0 新增的渐进首采和 DLSS 2x 插帧仍作为独立 EXP 节点提供，不替换旧工作流。视频扩画默认联合解码，部分素材可能出现条带、重复纹理或接缝，不承诺无缝扩画。

[渐进首采 T2VA / I2VA](examples/workflows/28-progressive-sampling)先小画幅采 6 步，学习型潜空间放大后再采 2 步。本机约 3 秒短片热测少用约 37%～38% 端到端时间；已审画面评价相近，对白组声音和口型正常，但游戏组仍有音量和音效内容差异，32 秒路线未通过，不保证所有素材同样提速或保音。

[DLSS 视频插帧 2x](examples/workflows/29-dlss-fi)把 24fps 变成 48fps、30fps 变成 60fps，保持时长、分辨率和原音轨；不是超分，也不加速 H3 生成。已审短片可接受，不保证所有快动作和 HUD 都无伪影。需要用户自行准备 DLSSG 运行文件和依赖，见目录 README；节点不自动下载或安装。更多变更见 [1.77.0 发布说明](docs/RELEASE_1.77.0.md)。

## 先从哪里开始

代码已集中到 `h3_t8/`，方便在首页直接阅读说明。工作流仍在 `examples/workflows/`，模型存放位置、节点名称、参数和连线不变；已有工作流不需要重新制作。旧的三个 TRT 命令入口也保留在根目录。开发者路径说明见 [目录结构](docs/REPOSITORY_LAYOUT.md)。

如果你第一次使用，建议按这个顺序：

1. 从 [`examples/workflows/01-basic-generation`](examples/workflows/01-basic-generation) 跑一条普通 H3 视频。
2. 需要参考图、首尾帧或音频控制时，看 [`02-audio-control`](examples/workflows/02-audio-control) 和对应的参考工作流。
3. 需要 OpenVDN 8 步时，先下载 [`t8star/Vdn-Minimax-H3-Comfy`](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy)，再用 [`10-speed`](examples/workflows/10-speed) 里的 `OpenVDN_DMD8_*_Advanced.json`。
4. 需要 WASD 人物/镜头控制时，下载 [`t8star/Minimax-H3-World-Comfy`](https://huggingface.co/t8star/Minimax-H3-World-Comfy)，再看 [`26-h3-world`](examples/workflows/26-h3-world)；首版固定为首帧 I2VA。
5. 需要长视频或 MV 时，看 [`04-long-video`](examples/workflows/04-long-video) 和 [`24-mv-lipsync`](examples/workflows/24-mv-lipsync)。
6. 需要图片或成片超分时，看 [`25-dlss-nr`](examples/workflows/25-dlss-nr)；这是 Windows RTX 专用的可选后处理。
7. 只想修一小段崩脸、不想重绘整条视频时，看 [`06-face-refine`](examples/workflows/06-face-refine) 中 2026-09-05 的 Window 工作流。
8. 需要给现有视频扩上下左右画面时，看 [`27-video-outpaint`](examples/workflows/27-video-outpaint)；第一次先跑范围预览，再走候选、审图、确认和接续。

每个高级工作流都带画布说明。先替换模型和输入素材，再运行；不要一开始就把多个 LoRA、Attention 加速器和采样器叠在一起。

双模型 4+4 长视频现在另附三份可选外部 LoRA 示例，分别使用 Core PyTorch、KJ H3 Sage
或已审计的 `ComfyUI-sol-attn / SolAttentionPatch`。一采、二采必须各走独立链路：
`底模 → 本路 Turbo LoRA → 本路可选外部 LoRA → 本路唯一 Attention 后端 → 对应 MODEL`。
外部 LoRA 默认 `disabled`；启用前放入 `models/loras`，并使用本项目的 H3 兼容加载器。
不要改用通用 `LoraLoaderModelOnly`，也不要串接 Sage、PyTorch selector 和 Sol。
具体文件及限制见 [`04-long-video`](examples/workflows/04-long-video) 的 README 和
[加速器接线说明](examples/workflows/04-long-video/DUAL_MODEL_ACCELERATOR_CONNECTIONS.md)。

## 能做什么

### H3 生成和参考控制

- 文生视频 T2VA
- 首帧图生视频 I2VA
- 尾帧生成 L2VA
- 首尾帧生成 FL2VA
- 单张或多张参考图 Ref2VA
- 参考视频、参考音频和混合参考
- 原生视频与音频联合生成、解码和保存
- H3-World 首帧 I2VA：用 WASD 和 IJKL/F 时间线控制人物与镜头

### 音频和口型

- 原声锁定、参考音色、对白和音轨混合
- Speech Studio 新建参考音色任务默认使用 `auto_reference_voice`：仅对参考音色模式保守裁掉低能量对齐留白；描述音色和旧工作流的显式 `none`/`conservative_energy` 不变
- Vocal Lock：用独立人声驱动画面，完整歌曲只在最终成片混入一次
- Audio Refine：可接 Turbo、PDD、EAV、Prompt Relay 和长视频路线
- 本地 ASR、说话人和 SyncNet 检查工具

项目里的 32 秒 Vocal Lock V3 样片已经完成五镜头串行生成、逐镜口型检查和真人完整观看。用户最终反馈为“32秒这个已经没问题了，完美”。这只代表该样片通过，不代表所有素材都会自动得到同样效果。

### 长视频

- 多关键帧和分段续写
- 一次排队、节点内串行生成
- 断点恢复和 accepted manifest
- Native Masked Context Plan B
- 可选 Color Match，默认开启，用于减轻分段接缝颜色跳变
- 显式 `accepted_picture_low_context_v1`：上一段实际成片末39帧经原resize与视频VAE，作为下一段LOW视频参考；HIGH、音频、4+4及3D放大器不改。旧JSON默认仍为独立low-x0续接。见[原接缝示例与防回归说明](docs/DUAL_MODEL_SEAM_FIX_20260913.md)及[新增Dance单项GPU对照／人审接受记录](docs/DANCE_ACCEPTED_PICTURE_20260913.md)。两份指定样例接受，不是任意素材、Depth或长时长保证。

### 视频扩画

- 支持上下左右自定义扩边，也可以按目标宽高比和锚点自动计算画布
- 新节点默认 `joint_decode`（联合解码）：整幅画面一起经过 VAE 解码，避免硬贴原片造成的轮廓断口。原片区域也会重建，细节可能变化，不能保证像素不变
- 可选 `preserve_source`（保留原片）：在有损编码前精确回贴原片像素，但新增画面与原片之间可能仍有接缝；两种模式都保留已有原音轨
- 先生成第一个窗口作为候选，人工审图并确认后才继续整条视频；接续和保存沿用候选模式，不会偷偷换模式。旧候选没有模式记录时仍按保留原片处理
- 支持逐镜切点、逐镜提示词、区域提示词和原片人物框审计。Color Match 默认开启，但只在保留原片模式处理扩区，并在切镜时重置；联合解码跳过边缘修色和几何校正
- 取消或重启后可以复用已完成窗口，但需要核对素材、模型和运行配置；更新 Core、KJ 或节点代码后，不能直接跳过缓存身份检查
- 最终 MP4 使用更保守的全帧内 H.264，文件会比常规编码大，但可避开本机已经复现的多线程解码坏帧问题
- 可在成片后单独接 DLSS-NR 2x；超分会改变整幅画面质感，所以仍需要另行看片

扩画不需要新的专用模型或转换权重，直接使用现有 H3 FL2VA 主模型、Qwen3-VL、视频 VAE 和音频 VAE。生成工作流还需要单独安装 [`ComfyUI-KJNodes`](https://github.com/kijai/ComfyUI-KJNodes)，当前测试路线固定使用 Stock20 与 KJ 的 H3 低显存 Attention/FFN。Turbo、SPEED、SLA、OpenVDN、FastH3 等组合暂不允许直接叠加；兼容审计工作流会提前说明冲突。已实际生成完整 32 秒并核对原音轨，但部分扩画区仍有瑕疵；本次按已知限制发布，不等于全部素材画质通过。详见 [扩画工作流说明](examples/workflows/27-video-outpaint/README.md)。

### 加速和成片修复

- 两个独立的 H3 低显存 EXP 节点：`Low VRAM Attention` 默认按 4 组拆分注意力头并提前释放中间量；`Chunk FeedForward` 默认仅在 packed token 超过 4096 时分 2 块执行 SwiGLU。两者都不依赖 KJNodes，可单用或按任意顺序串联；详细边界见 [H3 低显存节点说明](docs/H3_MEMORY_NODES_EXP.md)
- OpenVDN DMD 8 步 / Stage B 50 步
- PDD、SLA、SPEED、FastH3 VSA、Enhance-A-Video
- 稳定双时钟采样节点内置可选 `beta57` 调度器，无需安装 RES4LYF；默认仍是 `native_flow`
- DLSS-NR 图片/短视频帧/长视频文件超分，以及 FlashVSR、RealBasicVSR、RAFT、Skin Finish
- Face Refine Window：只生成选中的连续坏脸时间窗，预览后由人决定接受或拒绝
- NVIDIA H3 + LTX-2.5 两阶段超分实验路线

带 `Advanced` 或 `EXP` 的功能需要使用对应工作流。不同加速方案通常不能叠加；节点会尽量提前拒绝明显冲突。

## 安装

### ComfyUI Manager

在 Manager 中搜索 `MiniMax H3 Audio T8`，安装后完全重启 ComfyUI。

### 手动安装

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git minimax-h3-audio-T8
```

本节点依赖较新的 ComfyUI 原生 MiniMax H3 支持。如果节点全部变红、工作流提示缺少节点，先更新 ComfyUI 本体、前端和 Manager，再彻底退出并重新启动。

## 模型放哪里

常用目录如下：

| 模型 | ComfyUI 目录 |
| --- | --- |
| H3 主模型 | `models/diffusion_models` |
| Qwen3-VL 文本编码器 | `models/text_encoders` |
| 视频与音频 VAE | `models/vae` |
| Turbo、PDD、SLA 等 LoRA | `models/loras` |
| OpenVDN 完整模型包 | [`t8star/Vdn-Minimax-H3-Comfy`](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy)；仓库已按 `models` 下的正确目录整理 |
| H3-World 动作 LoRA | [`t8star/Minimax-H3-World-Comfy`](https://huggingface.co/t8star/Minimax-H3-World-Comfy) 已按 `models/loras/minimax/H3-World` 的相对目录整理；原始来源为 [`DANNY621/H3-World`](https://huggingface.co/DANNY621/H3-World) |
| FlashVSR / RealBasicVSR | `models/upscale_models` 或工作流注明的专用目录 |
| DLSS-NR v1.3 外部运行时（不是模型） | `models/DLSS-NR/1.3`（用户自行取得，节点不下载或分发） |
| 人脸、光流和分割模型 | 工作流或对应文档注明的目录 |

不同 H3 基模和 LoRA 不是随便混用的。文件名相近也不代表结构兼容。

## Face Refine Window：只修选中的时间窗

[`examples/workflows/06-face-refine`](examples/workflows/06-face-refine) 里有三份 2026-09-05 工作流：

- `Window_Manual_Review`：一次处理一个窗口，先预览，再手工接受或拒绝。
- `Window_Studio_Serial`：把多个窗口的决定写入可恢复清单；仍然一次只跑一个 H3 任务。
- `Window_Studio_Compose`：所有决定完成后，或提交后在保存前崩溃时，不加载 H3，直接从清单重建成片。

这条路线不会把整段正常人脸一起重画。你先用 0 基、闭区间帧号填写坏脸范围，例如 `0-23`；节点会补足
H3 所需的连续上下文，但上下文和边缘补帧永远不会自动进入最终接受区。预览和拒绝会逐像素返回原片；接受
必须显式勾选确认。最终视频始终使用完整原始音轨，窗口生成出来的音频会被丢弃。

Studio 版会把每个窗口的接受/拒绝结果按顺序保存。接受后的窗口不能回退或重复执行；崩溃后会从第一个未决定
窗口继续。同一项目有进程级锁，避免两个任务同时抢显存。它不会替你判断哪张脸更好，也不会自动接受候选。

生成工作流提供可选的 `MiniMaxH3FaceRefineSamplerMaskPatchV11T8Advanced`，**`enabled` 默认关闭**，
关闭时 MODEL 和 LATENT 原样直通，保持原有路线。手动开启才应用上游
[`ComfyUI-H3-FaceRefine v1.1.1`](https://github.com/Carasibana/ComfyUI-H3-FaceRefine/tree/d7ae3ee1ec445ea29fff7fc7366fa6fe85bdc2f5)
的采样遮罩修正：视频 `noise_mask` 只交给采样器，保留帧按当前 sigma 重加噪；锁定音频的
mask 条件仍原样送入 H3。开启时会核对 LATENT 中的音视频遮罩哈希和去噪报告，拒绝遮罩不匹配、
不兼容模型或冲突 MODEL patch。Parity、Window Manual 和 Studio Serial
工作流已接好；Compose-only 不加载 H3，所以不需要该节点。这个修正不需要新模型、pip 包或外部程序。

实测边界：RTX 4060 Ti 16GB 上，2 GiB 预留配置完成 90 帧、124 帧各 3 次冷启动和 3 次暖启动，再连续
执行 3 个窗口，共 15/15 次成功；最低空闲显存 678 MiB，最终进程 private 增量分别为 40.74、34.20 和
35.12 MiB，没有超过预注册的 256 MiB 阶梯门。这个数字只适用于该机器和固定工作流，不代表所有 16GB
显卡都安全。当前功能仍标为 Advanced EXP；首个真实坏脸样本和音频机械检查已通过，但多素材人脸质量仍需
用户看完整盲测后决定。v1.1.1 修正后的同素材 90 帧实测也已严格串行跑通，原始 PCM 完全一致、最低空闲显存
717.8 MiB。2026-09-06 盲评中，用户认为 **B（旧路线）好一点**：总体、前 24 帧五官、网格/发虚均选 B，
时序与接缝持平。因此保留旧路线为默认，新修正仅供手动对比；本次结果不代表所有素材，也不能证明整个人脸修复功能的画质已经全面验收。

## DLSS-NR：Windows RTX 可选后处理

[`examples/workflows/25-dlss-nr`](examples/workflows/25-dlss-nr) 提供运行时检查、图片、短视频帧序列和
文件视频四份独立工作流。默认使用 v1.3 `Standard + 2x`。

**DLSS-NR 不需要新的 PyTorch / safetensors 模型，但需要外部程序。** 请从
[`video2dlssnr` v1.3 官方 Release](https://github.com/DaniilSokolyuk/video2dlssnr/releases/tag/v1.3)
取得 `video2dlssnr_release.zip` 完整包；不要用 light 包，也不需要安装上游的 ComfyUI 节点包。本项目
不会自动下载、安装或分发其中的 EXE 和 NVIDIA DLL。

运行条件：

- Windows 10/11、NVIDIA RTX 显卡、NVIDIA 驱动 **616.56 或更新版本**
- 图片和帧序列路线不增加新的 pip 依赖；使用 ComfyUI 已有的 Torch、NumPy 和 PyAV 环境
- `Video File` 路线还要求 `ffprobe` 可在 `PATH` 中找到；通常安装 FFmpeg 后即可获得
- 外部运行时及 NVIDIA 的适用许可仍需遵守；节点不再要求勾选接受，环境检查通过即可运行

正确目录结构如下。保留完整 ZIP，把其中 `out` 目录的四个文件复制到这里的 `bin`，并把本仓库的
[`examples/runtime-manifests/dlss-nr-v1.3.json`](examples/runtime-manifests/dlss-nr-v1.3.json)
复制为 `t8-runtime-manifest.json`：

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

先运行 `Runtime_Audit`。节点会校验完整包及已解压文件的哈希、驱动、GPU 映射和真实 feature probe；
只有显示 `READY` 后才运行超分。v1.2 只允许 1× NR-only，默认 2× 工作流必须使用 v1.3。

三类固定素材的四路盲测均通过非退化门。人审结论是不同高清方法会带来不同皮肤和质感，没有一种在所有
素材上一定更好。因此这里的Standard只是推荐起点；超分不修复源片已有的身份、口型或真实纹理问题。

## H3-World：WASD 人物与镜头控制

上游项目：[`Danzer1xxxxChan/H3-World`](https://github.com/Danzer1xxxxChan/H3-World) ·
ComfyUI 模型包：[`t8star/Minimax-H3-World-Comfy`](https://huggingface.co/t8star/Minimax-H3-World-Comfy) ·
原始模型：[`DANNY621/H3-World`](https://huggingface.co/DANNY621/H3-World)

首版只做一件明确的事：输入一张首帧图，以 832×480、124 帧、24fps 生成一条约 5.17 秒的 I2VA，
并让 37 个潜空间时间点分别接收人物与镜头动作。预设包括前进、后退、左右移动、上下倾斜、左右摇镜和
快速摇镜；`custom` 可以用 JSON 分段组合动作。非 832×480 的首帧会等比覆盖后居中裁剪，不会直接拉伸。

下载 LoRA：

```powershell
hf download t8star/Minimax-H3-World-Comfy --include "loras/**" --local-dir ComfyUI/models
```

下载后的完整路径应为
`ComfyUI/models/loras/minimax/H3-World/step-10000.safetensors`。模型包中的文件与上游固定 revision
逐字节一致，SHA-256 为
`DDD9187B920B1E52C2D090F4E264FD83D8D433EFC2A5B159E58883AEAF96E526`；我们没有转换、合并或量化它。
这个 LoRA 已经是 ComfyUI 可直接加载的 104 对 A/B 权重。工作流还需要现有的完整
`minimax_h3_fl2va_int8_convrot.safetensors`、Qwen3-VL 编码器、视频 VAE 和音频 VAE。它不增加新的
pip 依赖，但最终安全保存要求 `ffmpeg` 可在 `PATH` 中找到。请直接使用
[`examples/workflows/26-h3-world`](examples/workflows/26-h3-world) 的工作流，不要叠加 OpenVDN、SLA、
VSA、Sol-Attn、BlockCache 或另一个模型/Attention 接管节点。

固定停车场样本的匿名盲测已通过：用户正确识别出持续前进的 H3-World 版本，确认动作稳定、两边声音
正常，画面质量持平。该结论支持这条固定合同转为正式 Advanced 功能，不代表任意人物、动作或显存配置
都能得到相同结果。

## OpenVDN：推荐的 8 步路线

### 新版 Core 兼容与二次采样（已在 v1.75.0 发布）

这一轮解决的是两件事：更新 ComfyUI 后仍能使用原有节点，以及让 VDN 先生成小图视频、再做潜空间放大和第二次采样。现有单采工作流继续保留，不需要重新转换底模。

- **全局 Sage 开关可以保留。** `--use-sage-attention` 是默认注意力后端，不等于另一个节点接管了 VDN。VDN 自己需要的注意力仍按它的算法执行，不会因此变成 Sage 版 VDN。
- **Core 自带的稀疏注意力节点与外部 Sol 插件不是一回事。** 对已识别的 Core 补丁，新兼容代码只在 VDN 分支上避让它；其他 MODEL 分支不变。未知的模型或注意力替换仍会提示冲突，不能随意叠加。
- **二采有两条路线。** 一条是 VDN 8 步 → 学习型 2x 潜空间放大 → VDN 4 步；另一条在第二次采样改用独立原生 H3 分支和新版 EMA B。不要给 VDN 分支再加通用 EMA LoRA。
- **默认保留第一遍的声音。** 第二遍主要细化画面，不重新生成音轨。两条路线都需要已有的 `models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors`；只有原生 H3 二采另需 `models/loras/minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors`。保存需要 FFmpeg，没有新增自动下载或安装步骤。

九类输入 × 完整/pruned 底模的二采测试已跑完；0.52MP 对照中，古典音乐、人声和两边口型正常，I2VA 环境声也无杂音，两组画面都差不多。四张二采工作流已交付到项目和用户目录，使用方法见 [二采工作流说明](examples/workflows/10-speed/VDN_TWO_PASS.md)。独立 Core/VDN 版本通过 2435 项完整回归、真实模型 DynamicVRAM 二采和解包导入，已在提交 `3769d70` 发布；本轮未完成的扩画修订不在该次发布中。二采不保证总比单采更清楚，也不保证所有 16GB 显卡和插件组合安全；[兼容范围](docs/CORE_VDN_COMPATIBILITY.md)中列出实际测试边界。

完整模型包：[`t8star/Vdn-Minimax-H3-Comfy`](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy)

模型仓库已经按 ComfyUI 的 `diffusion_models`、`text_encoders` 和 `vae` 目录整理好。获得模型访问权限后，可以直接下载到 `ComfyUI/models`：

```powershell
hf auth login
hf download t8star/Vdn-Minimax-H3-Comfy --local-dir ComfyUI/models
```

然后打开 [`examples/workflows/10-speed`](examples/workflows/10-speed) 中的正式 OpenVDN 工作流。目前提供：

- T2VA
- I2VA
- L2VA
- FL2VA
- 单图 Ref2VA
- 多图 Ref2VA
- 参考视频加原音轨
- 独立参考音频
- 首帧加参考音频 Hybrid

OpenVDN 上游公开说明的是 T2VA；其他模式是本项目利用 ComfyUI 原生 H3 条件布局实现并真实跑通的扩展。

### 完整底模和 pruned 底模都可以用

正式工作流默认使用：

```text
models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors
```

这是完整、非 pruned、AdaLN 输入宽度为 2688 的底模。安装新版模型包后，也可以在同一工作流里改用：

```text
models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
```

Composer 会读取底模里的 `adaln_t_table`，按内容 SHA 自动选择配套的 curve-projected Turbo adapter；用户不需要再接一个 LoRA 节点。该适配器保留 208 个原样可用的目标，并把 51 个 AdaLN 目标转换成 8 维 LoRA 加 51 个偏置残差，共实际应用 310 个补丁。完整底模仍走原生 2688 维 adapter。

兼容是按曲线表内容识别，不是只看文件名。目前支持模型包中的 FL2VA pruned INT8/ConvRot（同曲线表的 FL2VA pruned FP8 也通过静态签名门）；未知 pruned/curve-basis 模型仍会在采样前给出明确错误，避免套错适配器后静默生成。

### 已完成的验证

使用完整底模，项目在同一台 RTX 4060 Ti 16GB 上严格串行测试了 I2VA、L2VA、FL2VA、单图、双图、视频加音频、独立音频和首帧加音频共八条路线。随后又用 pruned INT8 底模在 320×192×39 下串行复跑了 T2VA 和同样八条多模态路线。每条 pruned 测试都完成：

- 800 个 OpenVDN 分支张量
- 104 个 default adapter 目标
- 259 个逻辑 turbo adapter 目标，其中 51 个 AdaLN 目标带独立偏置残差，共 310 个实际补丁
- 8 个 Euler/native-flow 步骤，video/audio shift 为 12/3
- 原生 H.264 视频、AAC 音频和联合严格解码
- 运行日志中 `ERROR lora = 0`

这些结果证明工作流和两类底模都能正确组合，不等于所有提示词、参考图、声音或显卡都已经通过画质验收。完整底模八条短测试最低剩余显存为 535–890MiB；pruned 九条短测试中最低只有 290MiB，仅 T2VA 和 I2VA 超过本项目的 512MiB 余量门。16GB 显卡仍应一次只跑一个任务，并根据实际占用降低分辨率或帧数。

## 常见问题

### 节点全红或找不到

先更新 ComfyUI、前端和 Manager，再完全退出重启。只更新本插件通常不够。

### 一运行就显存不足

先降低分辨率、帧数和参考素材数量，关闭同时运行的其他生成任务。16GB 显卡不要并发跑两条 H3。

### 声音变成噪音或音量异常

先检查工作流指定的 sampler、scheduler、步数、video/audio shift 和 LoRA。不要把通用 EMA、Ref2VA LoRA、OpenVDN turbo 或其他加速 LoRA随意互换。

### OpenVDN 提示 AdaLN 不兼容

先确认新版模型包中存在 `stage-dmd-step-250/adapters/turbo_pruned_curve_fl2va/adapter_model.safetensors`。如果文件存在仍报错，说明所选 pruned 底模的曲线表与已支持版本不同；换用模型包中的 FL2VA pruned INT8，或暂时换回完整 `minimax_h3_fl2va_int8_convrot.safetensors`。不要靠改文件名绕过签名检查。

### 能不能叠加 SLA、VSA、Sol-Attn 或 BlockCache

OpenVDN 自己接管模型分支和 adapter。不要再叠加其他 MODEL 或 Attention 接管节点；这不是“多加一个会更快”，通常只会造成冲突或错误结果。

## 文档

- [工作流总览](examples/workflows)
- [ComfyUI 使用与模型说明](docs/README_ComfyUI.md)
- [验证记录](docs/VERIFICATION_REPORT.md)
- [功能清单](features.json)

## 许可证

节点源码使用 GPL-3.0-or-later。

模型有各自的许可证。MiniMax H3 及其衍生模型遵循 MiniMax H3 Community License Agreement；协议定义的适用地区不包括欧盟、英国、韩国和美国。下载、运行或再分发模型前，请阅读模型仓库中的完整协议和 Acceptable Use Policy。

本 GitHub 仓库不包含模型权重。OpenVDN 模型包在 Hugging Face 单独提供，并保留原始许可、NOTICE、来源和修改说明。
