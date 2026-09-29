# ComfyUI-MiniMaxH3-SolAttn-V100

**MiniMax H3 在 V100 上的注意力稀疏加速插件（ComfyUI 自定义节点）· v1.2.0（双节点：Sol-Attn + SLA Top-K）**

> English version: [README_EN.md](README_EN.md) · 开发/设计文档: [DEVELOPMENT.md](DEVELOPMENT.md) / [DEVELOPMENT_EN.md](DEVELOPMENT_EN.md)
>
> **AI 辅助开发声明**：本项目由作者借助 AI 助手（DeepSeek-V4-Flash）完成开发，作者本人无计算机专业背景，代码与文档均由 AI 辅助编写。项目按 MIT 协议以现状分享；使用中如有问题，欢迎提 Issue，我们会在能力范围内协助解决。

> ⏸️ **维护暂停公告（自 2026-09-01 起）**
>
> 本项目由我一人开发与维护。我外出工作，GPU 环境（V100）不在身边，**无法进行任何测试、复现与验证**，因此本项目**自 2026-09-01 起暂停维护与更新**。
>
> 本插件的任何改动都必须真机验证（kernel 行为、路由质量、画面与音频都需要端到端实测）；没有 GPU 就无法确认改动是否可用，也无法确认用户反馈的问题能否复现，因此我选择暂停，而不是发布未经测试的内容。
>
> - 已发布版本（v1.2.x 及更早）不受影响，可正常使用；
> - 暂停期间我**无法处理 Issue / PR**，也无法回应使用反馈（项目没有其他维护者接手）；
> - 需要继续用的话：排障参考 [DEVELOPMENT.md](DEVELOPMENT.md)，求稳可换 FP16Safe 节点（dense），想改代码可借助 **vibe coding 工具（AI 编程助手）自行处理**；
> - 感谢理解。

插件包含两个独立节点（同一 MODEL 上互斥使用，只挂一个）：

- **Sol-Attn (V100)**（v1.1.1 默认）：Sol-Attn 阈值路由（keep-or-drop），无需训练、无需配套 LoRA。480p/10s 从 71-74s/步 降到 43s/步（~1.7×），画质肉眼无损。
- **SLA Top-K (V100)**（v1.2.0 新增）：SLA（Sparse-Linear Attention）topk 块稀疏，配合 lightx2v **Minimax-h3-Turbo-SLA LoRA**（4 步蒸馏 + 85% 稀疏蒸馏）。密度 15-19%，长序列比 Sol-Attn 再快 15-17%，音频段自动保底。

---

## 引用与使用的内容（Credit）

本项目是**现成、验证过的实现**的组合，核心加速全部来自：

| 组件 | 来源 | 用途 |
|---|---|---|
| **Sol-Attn 稀疏算法** | [arXiv 2607.24027](https://arxiv.org/abs/2607.24027)（NVlabs/Sana sol-engine，`techniques/sparse_backends/sol_attn/preprocess.py`） | kc/vc 块统计 + diag 阈值 + 列均值路由 + keep-or-drop（`routing.py` 按官方算法复刻，per-head 独立路由） |
| **keep-or-drop kernel** | 直接来源 [rwashy/H3-V100](https://github.com/rwashy/H3-V100)（整体 GPL-3.0-only，其中 FlashAttention CUDA 组件 BSD 3-Clause；仅复制 BSD 组件）；上游 [flash-attention](https://github.com/Dao-AILab/flash-attention)（Tri Dao）+ [Icbears/flash-attention-v100](https://github.com/Icbears/flash-attention-v100) | 选中块精确计算、未选中块跳过（`comfy_v100_solattn_cuda`：源码在 `native/`，或 Release 预编译 pyd） |
| **fp16 NaN 安全** | [ComfyUI-MiniMaxH3-FP16Safe](https://github.com/aaalll12322/ComfyUI-MiniMaxH3-FP16Safe) v6.8.0（逻辑内嵌为 `fp16safe.py`） | prescale /16 + 熔断 + fp32 重跑兜底（**自包含，无需单独安装 FP16Safe 插件**） |
| **参数风格** | [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) | tau / start_percent / end_percent / dense_blocks / sink 参数体系对齐 |
| **SLA 稀疏算法**（SLAV100 节点） | [arXiv 2509.24006](https://arxiv.org/abs/2509.24006)（thu-ml/SLA） | topk 块稀疏路由（`sla_routing.py`：smooth-k + 块均值打分 + 每行 top-K，逐行对齐 [LightX2V `get_block_map`](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/common/ops/attn/utils/sla_util.py)） |
| **SLA LoRA 权重**（SLAV100 节点） | [lightx2v/Minimax-h3-Turbo-SLA](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA)（Apache-2.0） | 4 步蒸馏 + 85% 稀疏蒸馏 LoRA（标准 rank-128，ComfyUI 原生 LoraLoader 直接加载，无需适配） |

---

## 解决的问题

MiniMax H3 在 V100 上的两个核心瓶颈：

1. **attention 是绝对大头**（480p 时占每步 ~79% 时间），而 PyTorch SDPA 在 V100 上仅 ~37T 吞吐；
2. **fp16 计算会 NaN**（H3 激活值真实可达 50 万，远超 fp16 上限），官方只支持 bf16/fp32，V100 无 bf16 硬件只能回落 fp32（慢 4×）。

Sol-Attn 稀疏的核心思路：attention 大多数 score 是噪声，**只对少量高价值 KV 块做精确计算**（keep-or-drop），其余直接跳过——省掉 O(n²) 的大头。

---

## 实测性能（用户真实 ComfyUI）

| 方案 | 每步耗时 | 画质 |
|---|---|---|
| 纯 FP16Safe（基线） | 71-74s（480p/10s）；33s（960×544/5s） | 正常 |
| **Sol-Attn v1.1.1（推荐 `tau=0.75, topk=32`）** | **43s（480p/10s）；24s（960×544/5s）；44s（1280×736/4s）** | **肉眼≈dense** |
| **SLA Top-K v1.2.0（`sparsity_ratio=0.85` + SLA LoRA）** | **41s（480p/10s）；20s（960×544/5s）；38s（1280×736/4s）；115s（1280×736/8s）** | **肉眼≈dense，音频正常（音频段保底）** |

- vs 纯 FP16Safe：Sol-Attn 480p/10s **~1.7×**；960×544/5s **+27% 加速**，质量与 dense 肉眼接近（top-k 保底修复了 v1.0 在高动态/文字/复杂动作下的手指/边缘问题）
- SLA Top-K vs Sol-Attn（同 seed 同 LoRA 配置对比，1280×736/4s）：**38s vs 44s（-14%）**；长序列（1280×736/8s）**115s vs 135s（-15%）**，960×544/5s **20s vs 24s（-17%）**——SLA 密度更低（15-19% vs 40%），靠 lightx2v SLA LoRA 的稀疏蒸馏保证质量
- 单次 attention（S=29650）：sparse kernel 比 SDPA 快 **4.55×**（路由+kernel 合计 2.77×）
- 路由 v1.1 已优化：**6.2ms @ S=29650 / 35.6ms @ S=98512**（视图化块统计，零 pad 拷贝）；kernel 直接吃非连续输入省 3 次拷贝

## 质量对比（v1.2.0，1280×736/4s，同提示词/参考图/种子/SLA LoRA）

<p align="center">
  <a href="https://raw.githubusercontent.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/main/videos/sla_topk_1280x736.mp4"><img src="https://raw.githubusercontent.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/main/videos/sla_topk_1280x736.gif" width="30%" alt="SLA Top-K（点击查看原视频）"></a>
  <a href="https://raw.githubusercontent.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/main/videos/solattn_1280x736.mp4"><img src="https://raw.githubusercontent.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/main/videos/solattn_1280x736.gif" width="30%" alt="Sol-Attn（点击查看原视频）"></a>
  <a href="https://raw.githubusercontent.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/main/videos/dense_1280x736.mp4"><img src="https://raw.githubusercontent.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/main/videos/dense_1280x736.gif" width="30%" alt="dense（点击查看原视频）"></a>
</p>

**左：SLA Top-K**（38s/步，音频正常）｜**中：Sol-Attn**（44s/步）｜**右：dense**（80s/步）｜*GIF 为 360 宽预览（约 1.8MB），点击查看 1280×736 原视频*

<p align="center">
  <img src="https://raw.githubusercontent.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/main/videos/sla_vs_sol_vs_dense_compare.png" width="70%" alt="SLA Top-K vs Sol-Attn vs dense 帧对比（1s/3s）">
</p>

画面差异与 Sol-Attn vs dense 同量级（SSIM ~0.72-0.75，稀疏 vs dense 的正常差距）；SLA 音频 <500Hz 能量 11%（dense 8%、音频崩坏时为 34%），音频段保底修复了 v1.2.0 之前 SLA 稀疏下的音频退化。

---

## 安装

```bash
cd ComfyUI/custom_nodes
# 方式一：git clone（若已发布到 GitHub）
git clone https://github.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100.git
# 方式二：手动复制整个文件夹到 custom_nodes/（需要 kernel：见下方"kernel 获取"）
```

**kernel 获取**（`comfy_v100_solattn_cuda*.pyd` / `*.so`，二选一；插件启动时自动匹配 `comfy_v100_solattn_cuda` 前缀的 pyd（Windows）或 so（Linux），**文件名不要求特定 Python 版本后缀**）：
- **Release 预编译（Windows）**：从 [GitHub Release](https://github.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/releases) 下载（Windows + Python 3.12，零编译）。**v1.1.1 为纯 Python 改动，kernel 未重编译（sha256 `1518648115fa4c527a541ba996c59e0c5ff4c33bbca0ece82d4d56ea367c9f87` 与 v1.0.0 相同）——从 v1.0.0 Release 下载同一份 pyd 即可**
- **源码编译**：仓库自带完整源码（`native/`，含 CUTLASS），一条命令：
  - **Windows**（MSVC，需 Visual Studio Build Tools + CUDA Toolkit + Windows SDK）：
    ```bash
    cd native && python setup.py build_ext --inplace
    # 产物（comfy_v100_solattn_cuda.cpXXX-win_amd64.pyd，XXX 随编译用的
    # Python 版本变化）复制到插件根目录即可，插件自动识别
    ```
  - **Linux**（GCC，需 CUDA Toolkit 12.x + python3-dev + ninja）：
    ```bash
    cd native && CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=8 python setup_linux.py build_ext --inplace
    # 产物（comfy_v100_solattn_cuda.cpython-XXX-x86_64-linux-gnu.so，已自动 strip）
    # 复制到插件根目录即可，插件自动识别
    ```

重启 ComfyUI。工作流中在 `sol_attn` 分类下找到 **"Sol-Attn (V100)"** 节点。

**依赖**：
- **ComfyUI**（含 `comfy/ldm/minimax`，PR #15224）
- **NVIDIA V100（sm_70）**——其他架构启动时报错
- **无额外 Python 包、无额外插件**：FP16Safe 逻辑内嵌（`fp16safe.py`），CUTLASS kernel 源码随仓库分发（`native/`），预编译 pyd 从 Release 获取

---

## 使用

```
H3 模型 ──> Sol-Attn (V100) ──> 采样器（KSampler 等）
```

单节点同时完成 fp16 安全 + 稀疏，**不需要**再串联 FP16Safe 节点。

### 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `fp16_safe` | true | 内嵌 FP16Safe（prescale /16 + 熔断 + fp32 重跑兜底）。关闭需工作流中另有 fp16 安全措施 |
| `tau` | 0.75 | 稀疏路由阈值 β：越大越稀疏越快。0.75 ≈ 40% 密度（V100 真机质量≈dense，推荐默认）；1.0 ≈ 26% 密度（更快，高动态/文字场景质量下降） |
| `start_percent` | 0.2 | 采样进度低于此比例走 dense（论文用 0.2；turbo 4-step 下第一步自然 dense） |
| `end_percent` | 0.9 | 采样进度高于此比例走 dense（0.9=尾部 10% 步保真，推荐默认；1.0=不启用尾部 dense，最快） |
| `min_tokens` | 1024 | 序列短于此 token 数不走稀疏（直接 SDPA） |
| `dense_blocks` | "0-1,-1" | 保留 dense 的 transformer 块，如 `"0-1"`=前两层，`"0-2,-1"`=前三层+最后一层（-1 从末尾数）；空=全部稀疏 |
| `h3_prefix_tokens` | 1024 | KV sink 保底：保护序列开头 token 不被稀疏。**保持默认 1024 即可，与序列长度 S 无关**——控制台日志里的 `S=...` 是序列长度，不是该参数的参考值；**勿设为 ≈S**（sink 保护覆盖几乎全序列，超出设计语义，实测导致末尾几帧画面崩坏） |
| `topk_blocks` | 32 | **每行保底块数（质量修复）**：阈值路由是"均值对齐检测"，高动态/新内容块对齐度低会被过滤（手/边缘/肢体在动态帧丢失）。topk 强制每行保留分数最高 K 块。0=关闭（v1.0 行为）。32 对 1540 块 ≈ 2% 密度开销 |
| `debug_nan` / `profile` | false | 透传 FP16Safe 的 NaN 检测 / 耗时统计 |

### 推荐配置

- **默认配置 = 推荐配置（v1.1.1，开箱即用）**：`tau=0.75, start_percent=0.2, end_percent=0.9, dense_blocks="0-1,-1", topk_blocks=32, h3_prefix_tokens=1024` → 480p/10s 43s/步；960×544/5s 24s/步（质量≈dense）。`h3_prefix_tokens` **保持默认 1024 即可，不要按控制台 `[SolAttn] S=...` 调整**（S 是序列长度，两者无关）
- **速度优先**：`tau=1.0, topk_blocks=16`（密度更低更快，画质需自行确认）
- **极致质量**：`topk_blocks=64`（每行保底更多，动态/文字细节最稳，速度损失明显）
- **保守**：`dense_blocks="0-2,-1"` 或 `end_percent=0.8`（更多层/尾部走 dense，质量更稳，稍慢）
- **小分辨率**（608 及以下）：attention 占比低，稀疏收益小，建议 `end_percent=0` 全 dense 或不用本插件

---

## SLA Top-K (V100) 节点（v1.2.0 新增）

**用途**：配合 [lightx2v/Minimax-h3-Turbo-SLA](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA) LoRA（4 步蒸馏 + 85% 稀疏蒸馏）使用。LoRA 用 ComfyUI 原生 **LoraLoader** 直接加载（标准 rank-128，键名与 ComfyUI H3 实现逐字匹配，无需适配）；本节点提供配套的 **SLA topk 块稀疏**（每行保留分数最高 top-K 块，85% 稀疏对齐 lightx2v 官方 config）。

```
H3 模型 ──> LoraLoader(SLA LoRA) ──> SLA Top-K (V100) ──> 采样器（turbo，4 步）
```

### 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `sparsity_ratio` | 0.85 | SLA 稀疏率：跳过的 key 块比例（0.85 = lightx2v 官方蒸馏值，每行保留 15% 块 + 保底 ≈ 18-19% 密度）。更高更快但可能掉质量 |
| `start_percent` / `end_percent` | 0.2 / 0.9 | 采样窗口（同 Sol-Attn 语义；turbo 4 步下第 1、4 步自然 dense） |
| `min_tokens` | 1024 | 序列短于此 token 数不走稀疏 |
| `dense_blocks` | "0-1,-1" | 保留 dense 的 transformer 块（同 Sol-Attn 语义） |
| `h3_prefix_tokens` | 1024 | text/cond/ref 前缀 sink 保底 |
| **音频段保底** | 自动 | 从 H3 `PackedLayout` 自动获取音频段（audio 在序列倒数第二段、video 之前），**强制不稀疏**——SLA topk 稀疏会砍掉音频行 key 块导致音频崩（实测 <500Hz 能量 34% vs 正常 8-17%），保底后恢复 |

### 实测（V100 真机，同 seed 对比）

- 1280×736/4s：**38s/步** vs Sol-Attn 44s / dense 80s；音频正常（<500Hz 能量 11%，dense 8%）
- 1280×736/8s：**115s/步** vs Sol-Attn 135s（-15%）
- 960×544/5s：**20s/步** vs Sol-Attn 24s（-17%）；480p/10s：41s（与 Sol-Attn 持平，短序列收益小）

### 注意

- 与 Sol-Attn 节点**互斥**（同一 MODEL 只挂一个；`optimized_attention_override` 单槽位）
- SLA LoRA 与普通 4 步 turbo LoRA 速度相同（LoRA 只改权重，推理路径不变）；**SLA LoRA 的价值在质量**（85% 稀疏下蒸馏过，普通 turbo LoRA 在低密度下可能崩）
- 论文完整 SLA 的线性注意力补偿需要配套微调权重（proj_l），lightx2v LoRA 不含且未蒸馏该路径（实测线性补偿使误差 rel 0.22→0.92 恶化），因此本实现为纯 topk 稀疏——即 lightx2v SLA 的蒸馏目标与能力上限

---

## 原理（摘要）

1. **路由（routing.py，官方 Sol-Attn 算法 + v1.1 优化）**：kc 块均值/vc 块和 → diag 阈值（key 空间解析投影）→ 列均值路由（| 邻域 ±1）→ **top-K 保底**（每行保留分数最高 K 块，动态内容兜底）→ CSR 掩码。per-head 独立。
2. **kernel**：keep-or-drop sparse kernel（选中块精确计算，未选中块跳过），fp16 + head_dim 128，sm70 CUTLASS。
3. **FP16Safe**：x/16 prescale → qkv → attention → out_proj 后 /16 还原（fp32），deferred isfinite 熔断，触发则整 forward fp32 重跑。

---

## 已知限制

- 仅验证 **V100（sm_70）** + Windows + Python 3.12（cp312 pyd）；其他平台需自行编译 native。
- sparse kernel 仅支持**全序列单次调用**（qlen==klen）；若 ComfyUI 路径将 attention 分块调用会输出错误（当前依赖全序列路径，改动上游需重测）。
- 稀疏质量以真机肉眼/PSNR 为准；极敏感场景（精细文字/复杂肢体）建议 `topk_blocks=64` 或调高 `dense_blocks` / `end_percent`。
- dense 兜底 = 原版 SDPA。
- SLA Top-K 节点仅对 **lightx2v SLA LoRA（或在其稀疏路径下蒸馏的权重）** 有质量保障；配合普通 turbo LoRA 时低密度下质量可能下降。
- SLA 音频段保底依赖 ComfyUI H3 的 `PackedLayout`（通过 forward 桥接获取）；ComfyUI 上游若改动该接口需同步适配。

---

## 版本历史

- **v1.2.2（2026-08-28）**：**修复 `h3_prefix_tokens` 文档误导（issue #3）**——原日志/tooltip/README 中"建议 = text+cond+ref+audio 实际 token 数"的措辞被误读为应按序列长度 S 调整，用户设成 ≈S（19008）后末尾几帧画面崩坏；现统一改为"**保持默认 1024 即可，与 S 无关，勿设为 ≈S**"（日志、节点 tooltip、README 中英参数表与推荐配置同步更新），DEVELOPMENT 增补踩坑记录。
- **v1.2.1（2026-08-27）**：**Linux 支持**（源自 issue #2 社区贡献，感谢 lesca）——新增 `native/setup_linux.py`（GCC 编译 + 自动 strip 调试符号，产物 ~1MB 与 Windows pyd 相当）；节点与验证脚本的 kernel 加载改为按平台匹配 `.pyd`（Windows）/ `.so`（Linux），`sla_nodes.py` 同步支持；WSL2 + V100 真机验证通过（SLA topk 密度 28.9% rel-L2 0.2176、SolAttn 40.0%/0.1970）。
- **v1.2.0（2026-08-23）**：新增 **SLA Top-K (V100)** 节点（`sla_nodes.py` + `sla_routing.py`，独立文件独立节点）——配合 lightx2v Minimax-h3-Turbo-SLA LoRA 的 SLA topk 块稀疏（smooth-k + 块均值打分 + 每行 top-K，逐行对齐 LightX2V `get_block_map`）；**音频段自动保底**（PackedLayout 桥接，修复 SLA 稀疏下音频崩坏）；含 v1.2 路由流式分块（`route_chunk`，长序列路由显存 -74%）。真机：1280×736/4s 38s/步、8s 115s/步（vs Sol-Attn 44s/135s）、960×544/5s 20s/步；音频/画面正常。LoRA 用原生 LoraLoader 加载（无需适配）。
- **v1.1.1（2026-08-22）**：①路由性能——去每层 GPU→CPU 同步、rank int32、off 显存减半、块统计视图化（S=98512 路由 182→35.6ms，S=174112 不再 OOM）；②**质量修复 top-K 保底**（`topk_blocks`，default 32）：`combined = min(threshold, kthvalue(第K大))` 每行保底 K 块，真实激活 rel 0.1157→0.0619（-47%），解决高动态/切镜/文字/复杂动作下手部与边缘丢失；③prefix debug：控制台打印 `[SolAttn][v1.1.1] S=... 密度=... prefix_tokens=...`；④真机 960×544/5s（S=20822）：**24s/步 vs dense 33s/步（+27%），质量肉眼≈dense**；480p/10s（S≈98512）42s/步（tau=0.75）。推荐配置 `tau=0.75, end_percent=0.9, dense_blocks="0-1,-1", topk_blocks=32`。
- **v1.0.0（2026-08-20）正式版**：单节点 = 内嵌 FP16Safe（`fp16safe.py`，v6.8.0 逻辑，自包含）+ Sol-Attn 稀疏（keep-or-drop，sparse-only kernel）。480p/10s 实测 **43s/步，画质肉眼无损**（~1.7× vs 纯 FP16Safe 71-74s）。参数对齐 kijai 风格（tau / start_percent / end_percent / min_tokens / dense_blocks / h3_prefix_tokens）；dense 兜底 = 原版 SDPA；kernel 源码在 `native/` 可自行编译，Release 提供预编译 pyd。版本串 `[SolAttn-V100][V1.0]`。

---

## Citation

本项目使用的稀疏算法来自 Sol-Attn 论文（Sol-Attn 节点）与 SLA 论文（SLA Top-K 节点）。如果本项目对你有帮助，请同时引用：

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

- Sol-Attn 论文：https://arxiv.org/abs/2607.24027 ｜ 官方代码：<https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends/sol_attn> ｜ 项目页：https://nvlabs.github.io/Sana/Sol-Attn/
- SLA 论文：https://arxiv.org/abs/2509.24006 ｜ 官方代码：https://github.com/thu-ml/SLA ｜ SLA LoRA：https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA ｜ LightX2V：https://github.com/ModelTC/LightX2V
