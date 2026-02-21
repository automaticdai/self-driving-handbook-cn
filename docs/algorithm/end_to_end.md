# 端到端自动驾驶

端到端（End-to-End）自动驾驶是一种将传感器输入直接映射到车辆控制输出的统一深度学习框架，与传统的模块化管线（感知→预测→规划→控制）形成鲜明对比。随着深度学习能力的飞跃和数据规模的膨胀，端到端方法正从实验室走向量产。


## 模块化管线 vs 端到端

| 维度 | 模块化管线（Traditional） | 端到端（End-to-End） |
| --- | --- | --- |
| 系统架构 | 感知→预测→规划→控制，各模块串行 | 统一神经网络，直接输出控制量 |
| 中间表示 | 明确（目标列表、轨迹、地图） | 隐式（神经网络内部特征） |
| 可解释性 | 强，每个模块可独立调试 | 弱，"黑盒"特性 |
| 错误传播 | 上游模块误差在下游级联放大 | 端到端联合优化，避免中间量误差 |
| 泛化能力 | 受限于人工规则和中间表示质量 | 数据驱动，可拟合复杂分布 |
| 训练方式 | 各模块独立训练，可用有标注数据 | 需要大规模真实驾驶数据或仿真 |
| 工程复杂度 | 模块间接口定义复杂，集成调试繁琐 | 架构简洁，但数据管理复杂 |

两种路线并非绝对对立。业界趋势是**"端到端骨干 + 可解释辅助监督头"**的混合方案：保留端到端的优化能力，同时通过辅助任务（如中间感知监督）提升可解释性和数据效率。


## 发展历程

### 第一阶段：行为克隆萌芽（2015–2019）

**NVIDIA DAVE-2（2016）——端到端驾驶先驱：**

直接从单摄像头图像预测转向角，是第一个在真实道路验证的端到端系统：

```
摄像头图像（3×66×200）
    │
5 个卷积层（特征提取）
    │
3 个全连接层
    │
转向角输出（单个标量）
```

以约 72 小时人类驾驶视频为训练数据，在高速公路测试有效。局限：仅输出转向角，没有纵向控制；遇到训练分布外场景容易失败。

**ChauffeurNet（Waymo/Google, 2019）：**

将感知结果渲染为俯视语义图像（Road Map + Agent Box），再进行端到端规划。规避了原始传感器的复杂性，加入了对抗训练使模型能处理长尾场景（如交通事故、逆行车辆）。

### 第二阶段：BEV 感知 + 模块化辅助（2020–2022）

BEV（Bird's Eye View，鸟瞰视角）感知范式兴起，成为端到端系统的标准中间表示：

**BEVFormer（Li et al., ECCV 2022）：**
- 跨摄像头、跨时间帧的 Transformer 注意力机制
- 利用空间可变形注意力将多视角图像特征投影到统一 BEV 网格

**BEVDet（Huang et al., 2021）：**
- 基于 LSS（Lift-Splat-Shoot）方法，通过深度估计将 2D 特征提升到 3D

**UniAD（Hu et al., CVPR 2023 最佳论文）：**

将感知（追踪、在线地图）、预测和规划统一在单一 Transformer 网络中，端到端优化：

```
多摄像头 → BEV 编码器 → [追踪头] → [在线地图头] → [运动预测头] → [规划头] → 轨迹
                                ↕ 跨任务注意力（Query 交互）
```

UniAD 证明了联合优化有助于规划性能，开创了"以规划为导向的感知"研究范式。

### 第三阶段：大规模端到端量产（2023–至今）

**Tesla FSD V12（2023）——首个量产端到端：**

Tesla 宣布将传统 C++ 模块化代码（超过 300,000 行）替换为统一的端到端神经网络，处理从 8 路摄像头到车辆控制的全流程：
- 输入：8 路摄像头原始图像帧序列
- 输出：方向盘转角、油门、制动控制量
- 规模：数千万参数，需要 HW4 FSD 芯片支持

**DriveVLM（Tian et al., Wayve/清华，2024）：**
将视觉语言模型（VLM）引入端到端驾驶，实现场景理解和自然语言可解释性：
- VLM 负责场景分析和高层决策生成（用文字描述驾驶意图）
- 轨迹生成网络负责将文字决策转化为具体轨迹


## 关键技术

### 占用网络（Occupancy Network）

传统目标检测输出稀疏的 3D 边界框，受限于预定义类别（无法检测"奇怪的障碍物"）。**占用网络**将三维空间离散化为体素网格，预测每个体素的占据状态和语义：

$$O_{xyz} \in \{0, 1\} \times \text{Class}\ =\ f_\theta(I_1, I_2, \ldots, I_N)$$

**优点：**
- 表达任意形状的障碍物（不受边界框形状限制）
- 适合开放世界感知（处理从未见过的物体类型）

**代表工作：** Tesla Occupancy（2022 AI Day）、Occ3D（清华，2023）、SurroundOcc（旷视，2023）。

### 知识蒸馏与特权学习

端到端视觉模型的挑战之一：纯摄像头无法直接获取精确深度，而 LiDAR 可以。**特权学习**（Privileged Learning）让学生网络（仅摄像头）向教师网络（含 LiDAR）学习：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(\hat{y}, y) + \lambda \, \mathcal{L}_{\text{KD}}(f_{\text{student}},\ f_{\text{teacher}})$$

其中 $\mathcal{L}_{\text{KD}}$ 为特征级或输出级蒸馏损失。推理时仅用摄像头，无需 LiDAR，但获得了接近 LiDAR 水平的感知能力。


## 世界模型（World Model）

### 定义与核心思想

世界模型（World Model）是一种通过**自监督学习**构建的环境内部表征，能够在隐空间预测未来状态，从而在不执行实际动作的情况下支持"想象式"规划。其核心思路受认知科学启发：人类驾驶员在行动前会在脑中模拟可能的场景。

形式上，世界模型由两部分组成：

$$z_t \sim p_\theta(z_t \mid z_{t-1},\ a_{t-1}) \quad \text{（状态转移模型）}$$
$$\hat{x}_t \sim q_\theta(\hat{x}_t \mid z_t) \quad \text{（解码器，将隐状态还原为观测）}$$

其中 $z_t$ 为 $t$ 时刻的隐状态，$a_{t-1}$ 为上一步动作，$\hat{x}_t$ 为预测的观测（图像帧）。训练目标为最大化观测的对数似然：

$$\mathcal{L} = \mathbb{E}\left[\sum_t \log q_\theta(\hat{x}_t \mid z_t)\right] - \beta\, D_{\text{KL}}\!\left(p_\theta(z_t) \| \mathcal{N}(0, I)\right)$$

### GAIA-1（Wayve, 2023）

GAIA-1 是首个专门为自动驾驶设计的大规模生成式视频世界模型：

- **参数规模**：约 9B 参数（视频生成 Transformer）
- **训练数据**：数千小时英国城市道路驾驶视频
- **输入条件**：当前摄像头帧 + 驾驶动作（转向角、速度）+ 文本描述（如"变道至左侧"）
- **输出**：未来驾驶视频帧序列（60 fps，时间连续）

GAIA-1 的关键能力：
1. **反事实推理**：模拟"如果此刻急打方向会怎样"
2. **长尾场景生成**：可条件生成罕见事故、施工路段等场景
3. **行为验证**：在部署前对规划轨迹进行"心理演练"

### DriveDreamer 与 WoVogen

**DriveDreamer（Wang et al., 2023）：**

以结构化道路信息（高精地图、交通框标注）为条件，利用扩散模型生成逼真的驾驶视频：

```
输入条件:
  ├── 高精地图（车道线、路口）
  ├── 3D 交通参与者边界框序列
  └── 自车轨迹

扩散模型 (U-Net + 时序注意力)

输出: 多摄像头同步驾驶视频
```

**WoVogen（World-Volume Generation）：**

在三维体积空间中进行场景生成，能够保证多摄像头视角的几何一致性，克服了 DriveDreamer 多视角不一致的问题。

### DreamerV3 与 RSSM

DreamerV3 是通用世界模型的代表，其核心是**循环状态空间模型（Recurrent State Space Model, RSSM）**：

$$h_t = f_\phi(h_{t-1},\ z_{t-1},\ a_{t-1}) \quad \text{（确定性循环状态）}$$
$$z_t \sim p_\phi(z_t \mid h_t) \quad \text{（随机隐状态，先验）}$$
$$z_t \sim q_\phi(z_t \mid h_t,\ x_t) \quad \text{（后验，基于实际观测）}$$

RSSM 将世界状态分解为确定性分量（GRU 维护的历史记忆）和随机分量（Categorical 或 Gaussian 分布），在保留长时依赖的同时支持多样化的未来预测。

DreamerV3 在自动驾驶中的应用：
- 完全在"想象空间"中训练 RL 规划器，无需大量真实环境交互
- 在 Minecraft、Atari 等多任务上达到 SOTA，正向驾驶任务迁移

### 世界模型的两大应用

**数据增强：**

世界模型可将现有真实数据"变形"为新场景：改变天气（晴天→暴雨）、改变交通密度、插入新的参与者，从而低成本扩充长尾数据：

$$x_{\text{aug}} = G_\theta(z_{\text{real}},\ c_{\text{new}}) \quad c_{\text{new}} \in \{\text{rain}, \text{night}, \text{fog}, \ldots\}$$

**基于模型的规划（Model-Based Planning）：**

在世界模型的隐空间中搜索最优行动序列，避免真实世界试错：

$$a_{0:H}^* = \arg\max_{a_{0:H}} \sum_{t=0}^{H} \gamma^t r(z_t, a_t), \quad z_{t+1} \sim p_\theta(z_{t+1} \mid z_t, a_t)$$

其中 $r$ 为奖励函数（如行驶进度、舒适性、无碰撞），$\gamma$ 为折扣因子，$H$ 为规划时域。


## 具身智能与驾驶

### VLM 驱动的驾驶推理

视觉语言模型（Vision-Language Model, VLM）将大规模语言模型的常识推理能力引入驾驶感知，实现开放词汇的场景理解：

**DriveVLM（2024）：**

采用"慢思考-快执行"双系统架构：
- **慢系统（VLM 推理）**：理解复杂场景语义，输出驾驶决策描述（文字）
- **快系统（轨迹网络）**：将文字决策转化为毫秒级控制指令

**DriveLM（Wang et al., ECCV 2024）：**

将驾驶推理建模为**图形视觉问答（Graph VQA）**任务：构建感知→预测→规划的有向推理图，每个节点对应一个问答对：

```
Q: "前方行人是否有穿越意图？"
A: "是，行人朝向道路，速度约 1.2 m/s"
  ↓
Q: "如何响应？"
A: "减速至 10 km/h，准备礼让"
  ↓
Q: "目标轨迹是什么？"
A: "在 2 s 内减速，保持当前车道"
```

### 思维链（Chain-of-Thought）驾驶推理

普通端到端模型直接输出控制量，缺乏中间推理过程。CoT 驾驶要求模型在生成控制指令前显式输出推理步骤：

$$\text{输入图像} \rightarrow \underbrace{\text{场景描述} \rightarrow \text{风险分析} \rightarrow \text{决策意图}}_{\text{思维链（可见文字）}} \rightarrow \text{控制指令}$$

CoT 推理的优势：
1. **可解释性**：可审查每一步推理是否合理
2. **推理质量**：复杂场景（如建筑工地、非常规交通）下决策准确率更高
3. **标注友好**：文字标注成本低于密集轨迹标注

### 语言作为中间表达

传统端到端中间表示为 BEV 特征图（数值张量）。语言中间表示将其替换为自然语言描述，实现人类可读的中间过程：

| 阶段 | 传统表示 | 语言表示 |
| --- | --- | --- |
| 场景描述 | BEV 占据图 + 目标列表 | "前方 15m 有一辆公共汽车正在靠站，右侧有骑行者" |
| 风险分析 | 碰撞概率热图 | "公共汽车可能突然停车，骑行者可能横穿" |
| 规划意图 | 候选轨迹集合 | "减速并向左移动半个车道以保持安全距离" |
| 控制输出 | $(\delta, a)$ 数值 | 轨迹点序列 + 速度曲线 |

语言中间表达的局限：推理延迟高（LLM 推理 > 500 ms），难以满足实时控制要求，通常需要异步或缓存机制。

### 具身智能的接地问题

将语言模型与真实驾驶世界对齐（Grounding）面临根本性挑战：

1. **空间接地**：语言模型天然缺乏精确空间感知（"前方 15 m"对 LLM 是模糊概念）
2. **时序接地**：驾驶场景以 10 Hz 变化，而 LLM 推理延迟以秒计
3. **分布漂移**：LLM 在网络文本上预训练，驾驶场景描述与预训练分布差异大
4. **幻觉问题**：LLM 可能"虚构"不存在的障碍物或交通规则

解决方向：多模态对齐训练（视觉 token 与语言 token 联合训练）、视觉感知模块与语言推理模块的专门接口设计、以真实驾驶数据进行指令微调。


## 数据扩展律（Scaling Laws）

### 自动驾驶中的数据扩展现象

Scaling Laws 最初由 OpenAI 在语言模型研究中发现：模型性能随参数量 $N$、数据量 $D$、算力 $C$ 呈幂律增长：

$$L(N, D) \propto N^{-\alpha} + D^{-\beta} + L_\infty$$

自动驾驶领域同样观察到类似现象：随着训练数据规模增加，端到端模型在各类场景上的干预率（Human Intervention Rate）持续下降，且未见明显饱和趋势。

### 特斯拉 FSD V12 的数据规模

Tesla 是目前端到端自动驾驶数据规模最大的公司：

- **车队规模**：全球超过 600 万辆具备数据采集能力的车辆
- **年采集里程**：数十亿英里
- **视频帧数**：训练 FSD V12 使用数百亿图像帧（估计）
- **标注方式**：自动标注为主，人工复核为辅
- **算力投入**：约 10,000 块 H100 GPU 组成的 Dojo 超算集群

Tesla Elon Musk 公开表示，FSD V12 相比 V11 的核心改进来自于将训练数据从 10 亿帧扩展到 100 亿帧以上（10 倍数据量提升），而非模型架构创新。

### 视频预训练 → 驾驶微调范式

借鉴 LLM 的"预训练 + 微调"范式，驾驶世界模型采用两阶段训练：

**第一阶段：大规模视频预训练**

在互联网视频（YouTube 行车记录仪、街景视频）上进行自监督预训练，学习物理世界的通用视觉动态：

$$\mathcal{L}_{\text{pretrain}} = -\sum_t \log p_\theta(\hat{x}_t \mid x_{<t})$$

视频 token 预测（类似语言模型的下一个 token 预测）。

**第二阶段：驾驶场景微调**

在真实或合成驾驶数据上进行有监督微调，引入驾驶特定的动作条件和奖励信号：

$$\mathcal{L}_{\text{finetune}} = \mathcal{L}_{\text{pred}} + \lambda_1 \mathcal{L}_{\text{action}} + \lambda_2 \mathcal{L}_{\text{safety}}$$

这一范式的优势：视频预训练赋予模型对光照、天气、物理运动的强先验，大幅降低驾驶阶段的数据需求。

### 合成数据与真实数据配比

合成数据（来自仿真引擎或生成模型）成本低但存在领域偏差（Sim-to-Real Gap），真实数据昂贵但质量高：

| 数据来源 | 成本 | 数量上限 | 真实性 | 典型用途 |
| --- | --- | --- | --- | --- |
| 真实采集 | 高 | 受车队规模限制 | 高 | 核心训练集 |
| CARLA 仿真 | 低 | 无上限 | 中（视觉差距大）| 规则学习、结构性场景 |
| 神经渲染（NeRF）| 中 | 有限（基于真实场景重建）| 很高 | 数据增强、视角扩展 |
| 生成模型（扩散）| 低-中 | 较大 | 中-高 | 长尾场景补充 |

业界普遍经验：合成数据与真实数据的最优比例约为 **3:1 至 10:1**（合成数据更多），但过高比例的合成数据会导致性能下降（合成数据与真实数据在特征分布上存在差异）。


## 端到端安全性

### 黑盒不可解释性问题

端到端神经网络的核心安全挑战在于其**不可解释性（Black-Box Nature）**：

- 无法事先枚举所有失效场景
- 在训练分布外（Out-of-Distribution, OOD）的行为无法预测
- 调试困难：当系统做出错误决策时，难以定位原因

典型失效案例：对抗样本（Adversarial Example）——在路牌上贴一张人眼不可察觉的贴纸，可导致分类器完全失效，而基于规则的感知系统通常不会有此问题。

### 形式化验证的局限

形式化验证（Formal Verification）在传统软件中证明程序满足规约（Specification），但在深度神经网络上面临根本性困难：

- **参数空间维度过高**：数十亿参数的网络状态空间无法穷举
- **输入空间无界**：高维图像输入空间（224×224×3 = 150,528 维）的覆盖不可能完备
- **当前 SOTA**：仅能验证极小型网络（< 10,000 参数）在有限输入扰动下的局部鲁棒性

神经网络验证工具（ERAN、Marabou、α-β-CROWN）的实用限制：网络层数 < 20 层，输入扰动半径 $\epsilon < 0.01$（L∞ 范数），与量产网络差距数个数量级。

### 神经符号方法（Neural-Symbolic）

神经符号方法将神经网络的感知能力与符号逻辑的可验证性结合：

**架构分层：**

```
原始传感器输入
      │
[神经感知层] → 符号化中间表示（抽象场景图）
      │
[符号推理层] → 规则引擎 + 时序逻辑约束
      │
[控制输出层] → 经验证的动作指令
```

**优势：** 符号层可进行形式化推理（如"如果行人在斑马线内，必须减速"）

**劣势：** 神经感知到符号表示的转换（Perception-to-Symbol）本身是个神经网络，仍有不确定性；符号规则无法覆盖所有长尾场景

代表工作：DeepProbLog（结合概率逻辑与深度学习）、Neuro-Symbolic Concept Learner（NS-CL）。

### Safety Layer：RSS 与 CBF 安全过滤器

最实用的安全方案是在端到端输出上叠加独立的安全过滤层，对不安全的控制指令进行修正或拒绝：

**责任敏感安全（Responsibility-Sensitive Safety, RSS）：**

由 Intel Mobileye 提出，定义了一套形式化的安全驾驶规则集（如最小安全距离、响应时间约束），构成可证明安全的驾驶包络：

$$d_{\text{min,rear}} = v_r t_{\text{resp}} + \frac{v_r^2}{2a_{\text{max,brake}}} - \frac{v_f^2}{2a_{\text{min,brake}}}$$

若端到端输出的控制指令会违反 RSS 约束，则用 RSS 计算的安全指令替代。

**控制障碍函数（Control Barrier Function, CBF）：**

CBF 是一种基于李雅普诺夫方法的实时安全约束，将安全集 $\mathcal{C}$ 定义为状态空间中的不变集：

$$h(x) \geq 0 \Rightarrow x \in \mathcal{C} \quad \text{（安全区域）}$$

$$\dot{h}(x, u) \geq -\gamma h(x) \quad \text{（CBF 条件，确保系统不会离开安全集）}$$

实时 QP 过滤器：在满足 CBF 条件的前提下，寻找最接近端到端输出的安全控制指令：

$$u_{\text{safe}} = \arg\min_u \|u - u_{\text{e2e}}\|^2 \quad \text{s.t.} \quad \dot{h}(x, u) \geq -\gamma h(x)$$

CBF QP 的求解时间 < 1 ms，可在 1 kHz 控制频率下运行，不影响系统实时性。


## 典型开源系统

### UniAD（上海人工智能实验室，2023）

UniAD（Unified Autonomous Driving）是 CVPR 2023 最佳论文，首次将感知、预测、规划统一在单一 Transformer 网络中端到端优化：

**系统架构：**

```
多摄像头输入
    │
BEV 编码器（BEVFormer）
    │
┌───┴───┐
追踪头   地图头
（TrackFormer）（MapFormer）
     │
  运动预测头（MotionFormer）
     │
  占用预测头（OccupancyFlow）
     │
  规划头（PlannerMLP）
     │
  规划轨迹输出
```

所有模块通过**跨任务 Transformer 注意力**共享特征，上游任务（追踪、地图）为规划头提供丰富语义上下文。

**nuScenes 开放数据集性能：**

| 指标 | UniAD | 传统模块化基线 | 提升 |
| --- | --- | --- | --- |
| 规划碰撞率（L2=2s）| 0.48% | 0.87% | -45% |
| 规划 L2 偏差（3s）| 0.88 m | 1.45 m | -39% |
| 追踪 AMOTA | 0.359 | 0.293 | +22% |

### VAD（向量化场景表示端到端，2023）

VAD（Vectorized scene representation for efficient Autonomous Driving）以向量化表示替代栅格 BEV 特征，大幅降低计算量：

**核心创新：**
- 用向量（折线、多边形）而非密集栅格表示车道线、代理轨迹
- 向量交叉注意力替代空间卷积，参数更少，速度更快
- 引入场景约束损失：规划轨迹须满足地图拓扑约束（不逆行、不压实线）

**性能对比（nuScenes val）：**

| 指标 | VAD-Base | UniAD | VAD-Tiny（轻量版）|
| --- | --- | --- | --- |
| 碰撞率（1s）| 0.17% | 0.20% | 0.21% |
| L2（3s）| 0.72 m | 0.88 m | 0.83 m |
| 推理速度 | 16.8 FPS | 1.8 FPS | 33.5 FPS |

VAD-Tiny 以接近 UniAD 的性能实现约 18 倍的速度提升，更适合量产部署。

### SparseDrive（稀疏表示端到端，2024）

SparseDrive 将稀疏 3D 表示引入端到端框架，进一步降低计算复杂度：

**核心思想：** 仅在有意义的 3D 位置（目标、车道节点）维护稀疏特征，而非对整个 BEV 空间进行密集计算：

$$\mathbf{F}_{\text{sparse}} = \{(p_i, f_i)\}_{i=1}^{M}, \quad M \ll H \times W$$

其中 $p_i \in \mathbb{R}^3$ 为空间位置，$f_i \in \mathbb{R}^C$ 为对应特征，$M$ 为稀疏 token 数（约 200–500 个），远小于密集 BEV 特征图的 token 数（通常 2500–10000 个）。

**性能表现（nuScenes val）：**

| 指标 | SparseDrive-S | SparseDrive-B | VAD-Base |
| --- | --- | --- | --- |
| L2（1s/2s/3s）| 0.32/0.56/0.78 m | 0.31/0.54/0.74 m | 0.54/0.72/0.94 m |
| 碰撞率（3s）| 0.05% | 0.04% | 0.06% |
| 推理延迟 | 39 ms | 58 ms | 比较基准 |

### CARLA Leaderboard 性能对比

| 系统 | 公开时间 | Driving Score | Route Completion | Infraction Rate |
| --- | --- | --- | --- | --- |
| LAV（Learning from All Vehicles）| 2022 | 61.8 | 94.1% | 0.71 |
| TCP（Trajectory-guided Control Prediction）| 2022 | 75.9 | 95.6% | 0.80 |
| UniAD（CARLA 适配版）| 2023 | 78.4 | 96.8% | 0.82 |
| VAD | 2023 | 81.2 | 97.3% | 0.83 |
| DriveVLM-Dual | 2024 | 85.7 | 98.1% | 0.88 |

注：Driving Score = Route Completion × Infraction Score，满分为 100。数值来源于各论文报告，评测版本和传感器配置可能不完全一致。


## 数据飞轮

端到端自动驾驶的核心竞争力是**数据飞轮**（Data Flywheel）：

```
更多行驶里程
      │
      ▼
更多真实驾驶数据
      │
      ▼
更好的端到端模型
      │
      ▼
更安全 / 更智能的驾驶
      │
      ▼
更多用户接受 → 更多订阅收入 → 更多部署车辆
      │
      └──────────────────────────────────────┘
                    （循环飞轮）
```

**关键数据技术：**

| 技术 | 描述 | 目的 |
| --- | --- | --- |
| 影子模式（Shadow Mode） | 实车运行时记录自动驾驶"如果接管会怎么做" | 低风险大规模评估与数据收集 |
| 自动标注（Auto-Labeling） | 离线用多帧 LiDAR 重建 3D 点云，为视觉数据提供伪标注 | 降低人工标注成本 |
| 场景挖掘（Scene Mining） | 从海量数据中自动检索困难场景（变道干扰、鬼探头）进行重点训练 | 覆盖长尾分布 |
| 对抗数据生成 | 用仿真或 GAN 生成罕见危险场景 | 提升边角场景覆盖 |


## 大模型与具身智能

最前沿的研究将**大语言模型（LLM）**和**视觉语言模型（VLM）**引入自动驾驶：

**GPT-Driver（Mao et al., 2023）：**
将运动规划问题建模为自然语言生成任务，用 GPT-3.5 直接输出结构化轨迹：
- 优点：天然可解释（"因为前方有行人，所以我减速至 20 km/h"）
- 挑战：推理延迟 > 1 s，远超实时控制要求

**DriveVLM（2024）：**
结合 VLM 的常识推理能力（理解"婚礼车队"、"施工人员"等语义）和高效轨迹生成网络：
- 双系统设计：慢速 VLM（决策层）+ 快速轨迹网络（执行层）
- 覆盖"什么是前方物体、应如何响应"的语义推理

**趋势展望：**
- 感知、预测、规划、控制的边界将进一步模糊
- 世界模型将成为自动驾驶的"大脑"
- 多模态大模型将带来更强的零样本泛化和语言可解释性


## 参考资料

1. M. Bojarski et al. End to End Learning for Self-Driving Cars. NVIDIA, arXiv:1604.07316, 2016.
2. M. Hu et al. Planning-Oriented Autonomous Driving (UniAD). CVPR Best Paper, 2023.
3. A. Hu et al. GAIA-1: A Generative World Model for Autonomous Driving. arXiv:2309.17080, 2023.
4. Tesla. AI Day Technical Presentations, 2021–2023.
5. Y. Tian et al. DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models. arXiv:2402.12289, 2024.
6. S. Wang et al. DriveLM: Driving with Graph Visual Question Answering. ECCV, 2024.
7. J. Jiang et al. VAD: Vectorized Scene Representation for Efficient Autonomous Driving. ICCV, 2023.
8. Z. Sun et al. SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation. arXiv:2405.19620, 2024.
9. D. Hafner et al. Mastering Diverse Domains through World Models (DreamerV3). arXiv:2301.04104, 2023.
10. W. Wang et al. DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. arXiv:2309.09777, 2023.
11. S. Shalev-Shwartz et al. On a Formal Model of Safe and Scalable Self-driving Cars (RSS). arXiv:1708.06374, 2017.
12. A. D. Ames et al. Control Barrier Functions: Theory and Applications. European Control Conference, 2019.
