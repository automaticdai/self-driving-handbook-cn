# VLM 基础模型

视觉语言模型（Vision-Language Model, VLM）是连接视觉感知与语言理解的桥梁。在自动驾驶领域，VLM 能够将摄像头捕获的复杂交通场景转化为可推理的语义表示，为场景理解、决策规划提供强大的基础能力。本章将系统介绍 VLM 的核心基础模型及其关键技术。

## 1. Vision Transformer（ViT）

### 1.1 基本原理

Vision Transformer（ViT）由 Dosovitskiy 等人于 2020 年提出，首次将 Transformer 架构直接应用于图像分类任务，打破了卷积神经网络（CNN）在视觉领域的长期主导地位。其核心思想是将图像切分为固定大小的 **图像块（Patch）**，将每个 Patch 线性投影为一个 Token，然后输入标准 Transformer 编码器进行处理。

### 1.2 Patch Embedding

给定输入图像 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$，将其划分为 $N$ 个大小为 $P \times P$ 的图像块，其中：

$$N = \frac{H \times W}{P^2}$$

每个图像块 $\mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C}$ 经过线性投影得到 Patch Embedding：

$$\mathbf{z}_0^i = \mathbf{x}_p^i \mathbf{E} + \mathbf{e}_{pos}^i, \quad \mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$$

其中 $D$ 为隐藏层维度，$\mathbf{e}_{pos}^i$ 为位置编码。

### 1.3 位置编码

由于 Transformer 本身不具备位置感知能力，需要显式添加位置编码。ViT 使用可学习的一维位置编码：

$$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{z}_0^1; \mathbf{z}_0^2; \dots; \mathbf{z}_0^N] + \mathbf{E}_{pos}$$

其中 $\mathbf{x}_{class}$ 是可学习的分类 Token，$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$ 为位置编码矩阵。

### 1.4 多头自注意力机制

Transformer 编码器的核心是多头自注意力（Multi-Head Self-Attention, MSA）。对于第 $l$ 层：

$$\mathbf{Q} = \mathbf{z}_{l-1}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{z}_{l-1}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{z}_{l-1}\mathbf{W}_V$$

单头注意力计算如下：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

多头注意力将 $h$ 个注意力头的输出拼接后投影：

$$\text{MSA}(\mathbf{z}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}_O$$

每层 Transformer 块的完整计算为：

$$\mathbf{z}_l' = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}$$

$$\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}_l')) + \mathbf{z}_l'$$

### 1.5 ViT 主要变体

| 模型 | 核心创新 | 主要优势 |
|------|---------|---------|
| **DeiT** | 知识蒸馏训练策略，引入蒸馏 Token | 无需大规模预训练数据，仅用 ImageNet 即可训练 |
| **Swin Transformer** | 移位窗口（Shifted Window）注意力，层级特征图 | 线性计算复杂度，适合密集预测任务 |
| **BEiT** | 掩码图像建模（MIM），类似 BERT 的预训练 | 自监督预训练，学习更强的视觉表示 |
| **EVA** | 结合 MIM 与 CLIP 的大规模预训练 | 在多种视觉任务上取得优异性能 |

其中 **Swin Transformer** 的窗口注意力机制将计算复杂度从 $O(N^2)$ 降低到 $O(N \cdot M^2)$（$M$ 为窗口大小），特别适合处理高分辨率图像，在自动驾驶场景中应用广泛。

---

## 2. CLIP

### 2.1 对比学习框架

CLIP（Contrastive Language-Image Pre-training）由 OpenAI 于 2021 年发布，通过对比学习在 4 亿（400M）图像-文本对上进行预训练，建立了图像与自然语言之间的强对齐关系。

CLIP 包含两个编码器：

```
┌─────────────────────────────────────────────────┐
│                  CLIP 架构                       │
│                                                  │
│  ┌──────────┐                 ┌──────────────┐  │
│  │  图像编码器 │                │  文本编码器    │  │
│  │ (ViT/RN)  │                │ (Transformer) │  │
│  └─────┬────┘                 └──────┬───────┘  │
│        │                             │           │
│        ▼                             ▼           │
│  ┌──────────┐                 ┌──────────────┐  │
│  │ 图像嵌入  │  ◄── 对比学习 ──►│  文本嵌入     │  │
│  │   I_i     │                │    T_j        │  │
│  └──────────┘                 └──────────────┘  │
│                                                  │
│  目标: 最大化配对样本相似度，最小化非配对样本相似度  │
└─────────────────────────────────────────────────┘
```

### 2.2 InfoNCE 损失函数

CLIP 使用对称的 InfoNCE 损失函数。给定一个批次中 $N$ 个图像-文本对，图像到文本的损失为：

$$\mathcal{L}_{i2t} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(\mathbf{I}_i, \mathbf{T}_i) / \tau)}{\sum_{j=1}^{N}\exp(\text{sim}(\mathbf{I}_i, \mathbf{T}_j) / \tau)}$$

文本到图像的损失为：

$$\mathcal{L}_{t2i} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(\mathbf{T}_i, \mathbf{I}_i) / \tau)}{\sum_{j=1}^{N}\exp(\text{sim}(\mathbf{T}_i, \mathbf{I}_j) / \tau)}$$

总损失为两者的平均：

$$\mathcal{L}_{CLIP} = \frac{1}{2}(\mathcal{L}_{i2t} + \mathcal{L}_{t2i})$$

其中 $\text{sim}(\cdot, \cdot)$ 为余弦相似度，$\tau$ 为可学习的温度参数。

### 2.3 零样本分类能力

CLIP 的一个重要特性是零样本（Zero-shot）分类能力。通过将类别名称转换为文本提示（如"一张{类别名}的照片"），计算图像与所有类别文本的相似度，即可实现无需微调的分类：

$$\hat{y} = \arg\max_{c} \text{sim}(f_{img}(\mathbf{x}), f_{txt}(\text{prompt}_c))$$

### 2.4 在自动驾驶中的应用

| 应用方向 | 说明 | 代表工作 |
|---------|------|---------|
| **开放词汇检测** | 检测训练时未见过的物体类别 | OWL-ViT, GLIP |
| **场景分类** | 对交通场景进行语义理解 | 基于 CLIP 的场景标注 |
| **异常检测** | 识别罕见或意外的道路事件 | 基于 CLIP 相似度的异常评分 |
| **跨模态检索** | 用自然语言检索驾驶场景 | 基于 CLIP 的场景数据库检索 |

---

## 3. LLaVA / InternVL 架构

### 3.1 三阶段架构

现代多模态大模型普遍采用"视觉编码器 + 投影器 + 大语言模型"的三阶段架构。LLaVA（Large Language and Vision Assistant）是这一架构的代表性工作。

```
┌──────────────────────────────────────────────────────────┐
│              LLaVA / InternVL 三阶段架构                   │
│                                                           │
│  阶段一            阶段二              阶段三               │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐   │
│  │ 视觉编码器 │───►│  投影器        │───►│  大语言模型     │   │
│  │ (ViT-L/G) │    │ (MLP/Q-Former)│    │ (LLaMA/Vicuna)│   │
│  └──────────┘    └──────────────┘    └───────────────┘   │
│       │                │                     │            │
│  视觉 Token        对齐映射              文本生成          │
│  提取              到语言空间            推理与回答         │
└──────────────────────────────────────────────────────────┘
```

具体来说：

- **视觉编码器**：通常使用 CLIP 预训练的 ViT（如 ViT-L/14），将输入图像编码为一组视觉特征向量
- **投影器（Projector）**：将视觉特征映射到语言模型的嵌入空间。LLaVA 使用简单的线性层或两层 MLP
- **大语言模型**：接收视觉 Token 和文本 Token 的拼接序列，生成回答

### 3.2 视觉指令微调

LLaVA 的训练分为两个阶段：

1. **预训练阶段**：冻结视觉编码器和 LLM，仅训练投影器。使用约 60 万条图像-文本描述数据进行特征对齐
2. **指令微调阶段**：冻结视觉编码器，同时微调投影器和 LLM。使用多模态指令数据（约 15 万条对话、推理、描述数据）

LLaVA 1.5 在此基础上进一步改进：将线性投影器替换为两层 MLP，使用更高分辨率的图像输入（336×336），在多个基准测试上取得显著提升。

### 3.3 InternVL 架构

InternVL 是由上海人工智能实验室开发的高性能多模态模型，具有以下特点：

- **大规模视觉编码器**：InternViT-6B 是目前最大的开源视觉编码器之一，参数量达到 60 亿
- **动态分辨率**：支持根据输入图像的宽高比动态调整分辨率，将图像分割为多个子图分别编码，有效处理不同比例的图像
- **渐进式训练**：从对比学习预训练到生成式微调，逐步提升模型的多模态能力

InternVL 2.0 的动态分辨率策略：

$$N_{tiles} = \arg\min_{n \leq N_{max}} \left| \frac{H}{W} - \frac{n_h}{n_w} \right|, \quad n = n_h \times n_w$$

其中 $n_h, n_w$ 为高和宽方向的分块数，$N_{max}$ 为最大分块数。

---

## 4. 多模态 Token 化与跨模态注意力

### 4.1 统一视觉与语言 Token

多模态大模型的核心理念是将视觉信息和语言信息统一到同一个 Token 序列中进行处理。给定图像 $\mathbf{x}_{img}$ 和文本 $\mathbf{x}_{txt}$，统一后的序列为：

$$\mathbf{z} = [\underbrace{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_M}_{\text{视觉 Token}}, \underbrace{\mathbf{t}_1, \mathbf{t}_2, \dots, \mathbf{t}_L}_{\text{文本 Token}}]$$

其中视觉 Token $\mathbf{v}_i = \text{Proj}(f_{vis}(\mathbf{x}_{img})_i)$，$M$ 为视觉 Token 数量，$L$ 为文本 Token 数量。

### 4.2 跨模态注意力机制

跨模态注意力（Cross-Attention）使文本生成过程能够动态关注视觉信息的不同部分。在跨模态注意力层中，Query 来自语言 Token，Key 和 Value 来自视觉 Token：

$$\mathbf{Q} = \mathbf{z}_{txt}\mathbf{W}_Q^{cross}$$

$$\mathbf{K} = \mathbf{z}_{vis}\mathbf{W}_K^{cross}, \quad \mathbf{V} = \mathbf{z}_{vis}\mathbf{W}_V^{cross}$$

$$\text{CrossAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

跨模态注意力通常以门控方式（Gated Cross-Attention）插入到 LLM 的自注意力层之间：

$$\mathbf{z}_{txt}' = \mathbf{z}_{txt} + \alpha \cdot \text{CrossAttn}(\mathbf{z}_{txt}, \mathbf{z}_{vis})$$

其中 $\alpha$ 为可学习的门控参数，初始化为 0，使得模型在训练初期保持原有语言模型的能力。

### 4.3 视觉 Token 对语言生成的引导

在自回归生成过程中，视觉 Token 提供了关键的上下文信息。语言模型在生成第 $t$ 个文本 Token 时，注意力权重分布反映了模型对图像不同区域的关注：

$$p(t_k | t_{<k}, \mathbf{v}_{1:M}) = \text{softmax}(\mathbf{W}_{out} \cdot \text{Transformer}([\mathbf{v}_{1:M}; t_{1:k-1}]))$$

这种机制使得模型能够"看到"图像中的具体内容，并据此生成准确的描述或做出合理的推理。

---

## 5. 其他重要基础模型

### 5.1 BLIP-2

BLIP-2 由 Salesforce 提出，其核心创新是 **Q-Former**（Querying Transformer）模块：

- 使用一组可学习的查询向量（32 个），通过交叉注意力从冻结的视觉编码器中提取最相关的视觉特征
- 两阶段预训练：先进行视觉-语言表示学习，再进行视觉-语言生成学习
- 参数高效：仅需训练 Q-Former（约 1.88 亿参数），视觉编码器和 LLM 均保持冻结

```
┌─────────────────────────────────────────┐
│             BLIP-2 架构                  │
│                                          │
│  冻结的视觉编码器 ──► Q-Former ──► 冻结的 LLM │
│  (ViT-G, 1B)      (188M)     (OPT/FlanT5) │
│                      │                    │
│              32 个可学习查询向量            │
└─────────────────────────────────────────┘
```

### 5.2 Flamingo

Flamingo 由 DeepMind 提出，是最早的大规模多模态少样本学习模型之一：

- **Perceiver Resampler**：将可变数量的视觉特征压缩为固定数量的 Token
- **门控交叉注意力层**：在冻结的 LLM 层间插入可训练的交叉注意力层
- **多图像/视频输入**：原生支持交错排列的图像-文本序列
- 展示了强大的少样本（Few-shot）学习能力

### 5.3 Qwen-VL

Qwen-VL 由阿里云团队开发，具有以下特点：

- 基于 Qwen 语言模型，使用 ViT-bigG 作为视觉编码器
- 支持图像、文本和边界框的多粒度理解
- 引入位置感知的视觉-语言对齐，支持区域级别的理解
- Qwen2-VL 进一步引入了 Naive Dynamic Resolution 和 M-RoPE（多模态旋转位置编码）

### 5.4 GPT-4V / GPT-4o

GPT-4V（GPT-4 with Vision）和 GPT-4o 是 OpenAI 的闭源多模态模型：

| 特性 | GPT-4V | GPT-4o |
|------|--------|--------|
| 模态 | 图像 + 文本 | 图像 + 文本 + 音频 |
| 架构 | 推测为早期融合 + 跨模态注意力 | 统一的原生多模态架构 |
| 推理能力 | 强大的视觉推理 | 更快的推理速度，多模态一体化 |
| 上下文长度 | 128K Token | 128K Token |

虽然架构细节未公开，但从其表现来看，GPT-4V/4o 在复杂视觉推理、图表理解、空间关系推理等方面展示了极强的能力，为自动驾驶场景的理解提供了重要参考。

---

## 6. 预训练策略

### 6.1 对比预训练

对比预训练通过拉近配对样本、推远非配对样本来学习对齐的多模态表示。

**优势**：

- 学习到的表示具有良好的对齐性和均匀性
- 支持零样本迁移和跨模态检索
- 训练效率较高，可扩展到大规模数据

**代表工作**：CLIP、ALIGN、SigLIP

SigLIP 将 InfoNCE 损失替换为 Sigmoid 损失，消除了对全局负样本的依赖：

$$\mathcal{L}_{SigLIP} = -\frac{1}{N}\sum_{i,j}\log \sigma(y_{ij}(\text{sim}(\mathbf{I}_i, \mathbf{T}_j) - b))$$

其中 $y_{ij} \in \{-1, +1\}$ 表示样本对是否匹配，$b$ 为偏置项。

### 6.2 生成式预训练

生成式预训练通过图像描述生成（Image Captioning）或视觉问答等任务训练模型：

- **自回归生成**：给定图像，逐 Token 生成文本描述
- **掩码语言建模**：遮盖部分文本，利用图像信息预测被遮盖的 Token
- **掩码图像建模**：遮盖部分图像块，利用上下文重建原始像素或特征

### 6.3 混合预训练

结合对比学习和生成学习的优势，现代 VLM 普遍采用混合预训练策略：

| 策略 | 对比学习 | 图像描述生成 | 图文匹配 |
|------|---------|------------|---------|
| BLIP | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| BLIP-2 | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| CoCa | $\checkmark$ | $\checkmark$ | |
| BEiT-3 | $\checkmark$ | $\checkmark$ | |

### 6.4 数据规模与多样性

预训练数据的规模和质量对 VLM 性能有决定性影响：

| 数据集 | 规模 | 来源 | 特点 |
|-------|------|------|------|
| LAION-5B | 58 亿对 | 网络爬取 | 大规模但噪声较多 |
| DataComp | 12.8 亿对 | 筛选的网络数据 | 经过质量过滤 |
| ShareGPT4V | 120 万条 | GPT-4V 生成 | 高质量详细描述 |
| nuScenes-QA | 46 万条 | 自动驾驶数据集 | 领域特定 |

研究表明，数据质量往往比数据数量更重要。经过精心过滤的中等规模数据集可能优于未经筛选的超大规模数据集。

---

## 7. 视觉编码器的自动驾驶适配

### 7.1 高分辨率处理需求

自动驾驶场景对视觉编码器有特殊的高分辨率需求。远处的交通标志、行人和小型障碍物在标准分辨率（如 224×224 或 336×336）下可能仅占数个像素，难以被准确识别。

常见的高分辨率适配方案：

- **直接提升分辨率**：通过位置编码插值，将 ViT 应用于更高分辨率图像（如 1024×1024），但计算量呈二次增长
- **分块编码**：将高分辨率图像分割为多个子图，分别编码后融合，如 InternVL 的动态分辨率方案
- **多尺度特征提取**：在不同分辨率层级提取特征，融合局部细节和全局上下文

位置编码的双线性插值方案：

$$\mathbf{E}_{pos}^{high} = \text{Interpolate}(\mathbf{E}_{pos}^{low}, (H', W'))$$

其中 $(H', W')$ 为新的网格尺寸。

### 7.2 多视角图像处理

自动驾驶车辆通常配备 6-8 个环视摄像头，需要同时处理多个视角的图像。多视角处理的关键挑战包括 Token 数量膨胀（6 个视角可产生数千个视觉 Token）、视角间空间关系建模、以及计算效率控制。常用解决方案：

```
多视角处理流程:
  ┌────────┐  ┌────────┐  ┌────────┐
  │ 前视图  │  │ 左前视图│  │ 右前视图│  ...
  └───┬────┘  └───┬────┘  └───┬────┘
      │           │           │
      ▼           ▼           ▼
  ┌─────────────────────────────────┐
  │     共享视觉编码器 (ViT)         │
  └─────────────┬───────────────────┘
                │
  ┌─────────────▼───────────────────┐
  │  Token 压缩 / Perceiver Resampler│
  └─────────────┬───────────────────┘
                │
  ┌─────────────▼───────────────────┐
  │     空间位置编码融合              │
  └─────────────┬───────────────────┘
                │
                ▼
          统一视觉表示
```

### 7.3 时序建模

自动驾驶需要理解交通场景的时序演化。处理视频输入的常见方法：

- **帧采样**：从视频中均匀采样关键帧，独立编码后拼接
- **时序注意力**：在空间注意力之外增加时序注意力层，建模帧间关系
- **时序位置编码**：为不同时刻的视觉 Token 添加时间信息

时空分解注意力（Divided Space-Time Attention）的计算方式：

$$\mathbf{z}^{(s)} = \text{SpatialAttn}(\mathbf{z}) + \mathbf{z}$$

$$\mathbf{z}^{(st)} = \text{TemporalAttn}(\mathbf{z}^{(s)}) + \mathbf{z}^{(s)}$$

这种分解策略将时空复杂度从 $O((T \cdot N)^2)$ 降低到 $O(T \cdot N^2 + N \cdot T^2)$，其中 $T$ 为帧数，$N$ 为每帧的 Token 数。

---

## 8. 模型规模与能力涌现

### 8.1 VLM 的缩放定律

与大语言模型类似，VLM 也遵循缩放定律（Scaling Laws）。模型性能与参数规模、数据规模和计算量之间存在幂律关系：

$$L(N, D) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_{\infty}$$

其中 $N$ 为参数量，$D$ 为数据量，$\alpha_N, \alpha_D$ 为缩放指数，$L_{\infty}$ 为不可约损失。

对于 VLM 而言，缩放需要同时考虑视觉编码器和语言模型的规模平衡：

| 视觉编码器 | 语言模型 | 总参数量 | 典型表现 |
|-----------|---------|---------|---------|
| ViT-B (86M) | 7B | ~7B | 基础视觉理解能力 |
| ViT-L (300M) | 7-13B | ~7-13B | 较强的视觉语言推理 |
| ViT-G (1B) | 13-34B | ~14-35B | 复杂场景理解和推理 |
| InternViT-6B | 20-76B | ~26-82B | 高级视觉推理和规划 |

### 8.2 涌现能力

随着模型规模增大，VLM 会展现出一些在小模型中不存在的**涌现能力**：

**空间推理能力**

- 小规模模型（<7B）：可以识别物体，但难以准确判断相对空间关系
- 中等规模（7B-13B）：能够描述基本的空间关系（左/右、前/后）
- 大规模（>30B）：可以进行复杂的三维空间推理，估计距离和遮挡关系

**因果推理能力**

- 理解交通事件的因果链条（如"因为前车急刹，所以需要减速"）
- 预测行为后果（如"如果现在变道，可能与右侧车辆碰撞"）

**思维链（Chain-of-Thought）推理**：大规模 VLM 可以进行多步视觉推理，例如观察到前方红灯和正在过马路的行人后，依次进行场景分析、因果推理、最终做出保持等待的决策。

### 8.3 能力涌现的量化分析

研究者通过多个基准测试追踪了不同规模 VLM 的能力变化：

| 能力维度 | <3B | 3B-7B | 7B-13B | 13B-34B | >34B |
|---------|-----|-------|--------|---------|------|
| 物体识别 | 中等 | 良好 | 优秀 | 优秀 | 优秀 |
| 场景描述 | 基础 | 中等 | 良好 | 优秀 | 优秀 |
| 空间推理 | 较弱 | 基础 | 中等 | 良好 | 优秀 |
| 因果推理 | 无 | 较弱 | 基础 | 中等 | 良好 |
| 多步规划 | 无 | 无 | 较弱 | 基础 | 中等 |
| 常识推理 | 无 | 较弱 | 中等 | 良好 | 优秀 |

值得注意的是，对于自动驾驶场景中至关重要的空间推理和因果推理能力，通常需要 13B 以上的模型规模才能初步具备。这也是当前将 VLM 部署到车端面临的主要挑战之一。

---

## 参考资料

1. Dosovitskiy, A. et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
2. Radford, A. et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.
3. Liu, H. et al. "Visual Instruction Tuning." NeurIPS 2023.
4. Chen, Z. et al. "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks." CVPR 2024.
5. Li, J. et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models." ICML 2023.
6. Alayrac, J.B. et al. "Flamingo: a Visual Language Model for Few-Shot Learning." NeurIPS 2022.
7. Bai, J. et al. "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond." arXiv 2023.
8. Touvron, H. et al. "Training data-efficient image transformers & distillation through attention." ICML 2021.
9. Liu, Z. et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
10. Zhai, X. et al. "Sigmoid Loss for Language Image Pre-Training." ICCV 2023.
11. Kaplan, J. et al. "Scaling Laws for Neural Language Models." arXiv 2020.
12. Wei, J. et al. "Emergent Abilities of Large Language Models." TMLR 2022.
