# 行为预测

行为预测（Behavior Prediction）是自动驾驶软件栈中感知与规划之间的关键桥梁。自动驾驶系统不仅需要知道周围障碍物的当前状态，还需要预测它们在未来数秒内的运动意图和轨迹，才能做出安全合理的规划决策。


## 预测的必要性

以高速公路变道场景为例：车辆仅凭当前感知结果（邻车位置和速度）无法判断该车是否即将变道。如果无法预测邻车未来 3 秒的运动，规划模块就无法决定是否应该减速让行。

**预测的内在挑战：**
- **行为多样性**：同一情景下，不同驾驶员可能采取截然不同的行为（激进/保守/礼让）
- **社会交互性**：多个交通参与者相互博弈，一方行为影响另一方决策
- **长尾分布**：大部分场景常见且容易预测，少数极端场景（急刹、鬼探头）最危险但样本稀少
- **意图不可观测**：驾驶员意图是内部状态，只能通过行为间接推断


## 预测的分类

### 按时间尺度

| 时间范围 | 典型应用 | 精度要求 | 主要方法 |
| --- | --- | --- | --- |
| 短期（0–2 s） | 碰撞检测、紧急制动 AEB | 高（厘米级） | 物理模型、简单外推 |
| 中期（2–5 s） | 变道决策、跟车规划 | 中（亚米级） | 深度学习、基于地图 |
| 长期（5–10 s） | 路口决策、超车规划 | 低（意图级别） | 意图识别、生成模型 |

### 按方法论

| 类别 | 代表方法 | 优势 | 劣势 |
| --- | --- | --- | --- |
| 基于物理 | 常速模型（CV）、CTRV | 实时、无需训练、可解释 | 不考虑意图、道路结构 |
| 基于机器学习 | LSTM、GAN、Transformer | 数据驱动、场景适应性强 | 可解释性差、泛化需大量数据 |
| 基于意图 | 贝叶斯、隐马尔可夫模型 | 语义明确、可解释 | 意图识别本身困难 |
| 基于交互 | 社会力模型、图神经网络 | 显式建模多体交互 | 计算量大，扩展性有限 |


## 基于物理的预测

最简单的预测方法假设障碍物保持当前运动状态，无需训练，适合短期预测。

### 常速模型（CV，Constant Velocity）

假设目标以当前速度做匀速直线运动：

$$\begin{bmatrix} x_{t+1} \\ y_{t+1} \end{bmatrix} = \begin{bmatrix} x_t + v_x \Delta t \\ y_t + v_y \Delta t \end{bmatrix}$$

适用于直线行驶的短期预测（< 1 s），在路口转弯等场景误差迅速累积。

### 常转率常速模型（CTRV，Constant Turn Rate and Velocity）

适用于转弯中的车辆，在极坐标下保持速度 $v$ 和转率 $\dot{\psi}$ 不变：

$$\begin{pmatrix} x \\ y \\ \psi \\ v \\ \dot{\psi} \end{pmatrix}_{t+\Delta t} = \begin{pmatrix} x + \frac{v}{\dot{\psi}}\bigl(\sin(\psi + \dot{\psi}\Delta t) - \sin\psi\bigr) \\ y + \frac{v}{\dot{\psi}}\bigl(-\cos(\psi + \dot{\psi}\Delta t) + \cos\psi\bigr) \\ \psi + \dot{\psi}\Delta t \\ v \\ \dot{\psi} \end{pmatrix}$$

当 $\dot{\psi} \approx 0$ 时退化为匀速模型。CTRV 是无迹卡尔曼滤波（UKF）目标跟踪的常用过程模型。


## 基于深度学习的预测

### 序列模型（LSTM）

将历史轨迹作为时间序列输入，预测未来位置：

```
历史轨迹 [(x₁,y₁), ..., (xₙ,yₙ)]
         │
    LSTM × n 层
         │
    未来轨迹 [(x̂₁,ŷ₁), ..., (x̂ₘ,ŷₘ)]
```

- **优点**：简单有效，适合单体长期运动建模
- **缺点**：不考虑与其他智能体的交互；不利用道路拓扑信息

### 社会化模型

**Social LSTM（Alahi et al., CVPR 2016）：**

引入"社会池化层"（Social Pooling），将空间邻近行人的隐藏状态汇聚后融合，让模型感知社会交互：

$$h_i^t = \text{LSTM}\bigl(h_i^{t-1},\ e_i^t,\ P_i^t\bigr)$$

其中 $P_i^t = \text{MaxPool}\left(\{h_j^{t-1} : j \in \mathcal{N}(i)\}\right)$ 为邻居隐藏状态的池化。

**Social GAN（Gupta et al., CVPR 2018）：**

用生成对抗网络生成多条多样化轨迹，覆盖行人运动的多模态分布。引入多样性损失鼓励生成不同的可能未来。

### 图神经网络（GNN）

**Trajectron++ / GRIP++：**

将交通场景建模为图（每个智能体为节点，时空关系为边），用图神经网络传递交互信息：

$$h_i^{(l+1)} = \sigma\!\left(W^{(l)} \cdot \text{Aggregate}\bigl(\{h_j^{(l)} : j \in \mathcal{N}(i)\}\bigr)\right)$$

图结构能够灵活表示任意数量的参与者和异构交互（车-车、车-行人、车-自行车）。

### 基于地图的预测

道路结构对车辆运动有极强约束：车辆几乎不会偏离车道，路口行为遵循交通规则。

**TNT（Target-driveN Trajectories，Zhao et al., CoRL 2020）：**
1. 先预测目标终点分布（路口后会去哪里）
2. 再为每个候选终点生成完整轨迹
3. 联合对目标点和轨迹评分，输出 Top-K 候选

**VectorNet（Gao et al., CVPR 2020）：**

用多段折线（Polyline）统一表征地图元素（车道线、红绿灯）和智能体历史轨迹，用全局 Transformer（类 PointNet）进行图级特征融合，避免将地图光栅化为图像的信息损失。


## 基于地图的预测深化

### 车道图（Lane Graph）构建

基于地图的预测方法的核心是将 HD Map 转化为车道图（Lane Graph），供神经网络编码拓扑关系：

- **节点**：每个车道段（Lane Segment）对应一个节点，携带中心线采样点、车道宽度、限速等属性
- **边**：两类有向边：（1）前驱/后继连接（车道沿行驶方向延续）；（2）相邻连接（可变道的左右邻车道）
- **节点特征向量**：将中心线多段折线编码为固定维度向量，例如将 $N$ 个采样点 $(x_i, y_i, \theta_i)$ 通过 MLP 或 PointNet 聚合为 $\mathbf{f}_{\text{lane}} \in \mathbb{R}^d$

```
车道图示意（路口场景）：

  L1 ──▶ L3 ──▶ L5（直行）
          │
          └──▶ L4 ──▶ L6（右转）
  L2 ──▶ L7（合流）
```

车道图的构建使预测模型能够"感知"道路拓扑，从而直接在候选车道上推断智能体意图（选择哪条路线），而不是在无约束的二维空间中预测。

### VectorNet 详细架构

VectorNet（Gao et al., CVPR 2020）提出用统一的向量表示法处理地图和轨迹，分两阶段编码：

**阶段一：子图网络（SubGraph Network）**

每个地图元素（车道线、边界线、人行横道）和每个智能体的历史轨迹都被表示为一条多段折线（Polyline）。对折线中第 $i$ 条线段，特征向量定义为：

$$\mathbf{v}_i = \left[ d_i^s,\ d_i^e,\ a_i,\ j_i \right]$$

其中 $d_i^s, d_i^e$ 分别为线段起止点全局坐标，$a_i$ 为该线段的属性（车道类型、速度等），$j_i$ 为折线编号（用于区分不同元素）。

子图网络对每条折线内部的线段做局部聚合（类 PointNet 操作）：

$$\mathbf{p}_{\text{poly}} = \phi\!\left(\text{MaxPool}\left(\{g(\mathbf{v}_i) : i \in \text{poly}\}\right)\right)$$

其中 $g(\cdot)$ 为逐线段的 MLP，$\phi(\cdot)$ 为最终编码层。每条折线输出一个固定维度的折线级特征向量。

**阶段二：全局图 Transformer**

将所有折线特征向量视为图节点，用自注意力（Self-Attention）进行全局交互：

$$\mathbf{A} = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

全局图允许任意智能体与任意地图元素之间建立注意力连接，从而编码"哪辆车在哪条车道"的语义关系。最终每个智能体节点的特征用于解码未来轨迹分布。

**VectorNet 的优势：**
- 无需光栅化，避免像素级表示的空间分辨率损失
- 计算量与地图元素数量成线性关系，而非图像尺寸的平方
- 天然支持不规则形状的地图元素

### TNT 目标驱动轨迹预测详细流程

TNT（Target-driveN Trajectories）将预测分解为三个级联子任务：

**步骤一：候选目标点采样**

沿车道图的每条候选路线，均匀采样若干目标位置点 $\{\tau_k\}_{k=1}^{K}$（通常 $K=1000$），并通过分类器预测每个候选点的概率：

$$p(\tau_k \mid x) = \text{softmax}\!\left(f_\theta(\mathbf{z}_{\text{agent}},\ \mathbf{z}_{\tau_k})\right)$$

其中 $\mathbf{z}_{\text{agent}}$ 为智能体编码，$\mathbf{z}_{\tau_k}$ 为候选目标点的位置编码。

**步骤二：以目标为条件的轨迹生成**

对每个高概率候选目标点 $\tau_k$，生成从当前状态到 $\tau_k$ 的完整轨迹：

$$\hat{y}^{(k)} = g_\phi\!\left(\mathbf{z}_{\text{agent}},\ \tau_k\right)$$

$g_\phi$ 为自回归或 MLP 解码器，输出每个时刻的位置分布（高斯）。

**步骤三：联合评分与 NMS**

对所有 $(τ_k, \hat{y}^{(k)})$ 对计算联合得分，用非最大抑制（NMS）过滤相似候选，输出 Top-K 多模态预测结果。

### HiVT：层次向量 Transformer

HiVT（Hierarchical Vector Transformer，Zhou et al., CVPR 2022）在 VectorNet 基础上引入层次注意力机制，达到 nuScenes 数据集 SOTA 性能：

**局部编码阶段：**

以每个智能体为中心建立局部坐标系，对该智能体附近的地图元素和邻近智能体做局部注意力（Local Attention），提取相对运动和几何特征：

$$\mathbf{h}_i^{\text{local}} = \text{LocalTransformer}\!\left(\mathbf{e}_i,\ \{\mathbf{e}_j : j \in \mathcal{N}_r(i)\}\right)$$

其中 $\mathcal{N}_r(i)$ 为智能体 $i$ 半径 $r$ 范围内的邻居集合。局部注意力天然具有旋转和平移等变性（Equivariance），提升模型泛化能力。

**全局聚合阶段：**

在局部特征基础上，用全局 Transformer 捕获远距离的智能体间依赖（例如前车减速对后车的影响）：

$$\mathbf{h}_i^{\text{global}} = \text{GlobalTransformer}\!\left(\mathbf{h}_1^{\text{local}},\ \ldots,\ \mathbf{h}_N^{\text{local}}\right)$$

HiVT 的分层结构降低了全局注意力的计算复杂度（从 $O(N^2)$ 局部化为 $O(N \cdot |\mathcal{N}_r|)$），同时保持了全局场景理解能力。


## 多智能体联合预测

### 边际预测 vs 联合预测

目前主流预测系统输出**边际预测（Marginal Prediction）**：对每个智能体独立预测其未来轨迹分布，忽略智能体之间的相关性。

$$p(y_1, y_2, \ldots, y_N \mid x) \approx \prod_{i=1}^N p(y_i \mid x) \quad \text{（边际预测假设）}$$

边际预测的问题在于：多个智能体的预测结果在物理上可能相互冲突（两辆车预测为同时占据同一位置），导致规划层收到"幽灵碰撞"警告，产生过度保守的决策。

**联合预测（Joint Prediction）**直接建模多个智能体的联合未来分布：

$$p(y_1, y_2, \ldots, y_N \mid x)$$

联合预测保证预测结果**场景一致性**（Scene Consistency）——智能体之间的预测轨迹不会相互穿越或碰撞。例如路口汇入场景中，联合预测能够正确建模"A 先行、B 等待"或"B 先行、A 等待"两种联合模式。

### 预测的交互感知

真实交通行为存在强烈的反应性（Reactivity）：若智能体 A 开始变道，智能体 B 会减速让行。这种条件依赖无法被边际预测捕捉。

**条件预测（Conditional Prediction）：**

给定智能体 A 的轨迹假设 $y_A$，预测智能体 B 的条件分布：

$$p(y_B \mid x,\ y_A)$$

通过枚举 A 的多种行为假设，可以评估"如果 A 这样做，B 会怎么反应"的因果关系，为规划模块的博弈推理提供依据。

**MCMS（Multi-agent Conditional Motion Simulation）：**

MCMS 将联合预测分解为序列化条件预测——先采样 Agent 1 的轨迹，以此为条件采样 Agent 2，依此类推：

$$p(y_1, y_2, \ldots, y_N) = p(y_1) \cdot p(y_2 \mid y_1) \cdot p(y_3 \mid y_1, y_2) \cdots$$

每个条件分布由独立的轨迹生成网络建模，通过注意力机制接收已采样智能体的轨迹作为条件输入。MCMS 在保持场景一致性的同时，避免了指数级的联合状态空间搜索。

### 联合预测的计算复杂度挑战

联合预测面临的核心难题是**组合爆炸**：若每个智能体有 $K$ 种模态，$N$ 个智能体的联合模态数为 $K^N$，在实际场景（$N=10$，$K=6$）时联合模态数达 $6^{10} \approx 6000$ 万，完全枚举不可行。

常用应对策略：

| 策略 | 思路 | 计算代价 |
| --- | --- | --- |
| 近似因子分解 | 按依赖强度分组，组内联合、组间独立 | 中等 |
| 采样近似 | 用 MCMC 或自回归采样近似联合分布 | 可控 |
| 学习一致性后处理 | 先边际预测，再用图模型约束一致性 | 低 |
| 端到端联合生成 | Transformer 同时输出所有智能体轨迹 | 高（单次前向） |


## 多模态预测

真实场景中未来是**多模态**的——同一情景下存在多种合理结局（直行、左转、右转）。单输出均值预测会导致模型在各模式之间折中，产生不合理的平均轨迹。

**高斯混合模型（GMM）：**

$$p(\hat{y}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\hat{y};\ \mu_k,\ \Sigma_k)$$

其中 $K$ 为模式数量，$\pi_k$ 为各模式权重（$\sum \pi_k = 1$）。

**条件变分自编码器（CVAE）：**

通过随机隐变量 $z$ 参数化不同的未来模式，从条件分布 $p(y \mid x, z)$ 采样生成多条轨迹：

$$p(y \mid x) = \int p(y \mid x, z) \, p(z \mid x) \, dz$$

**代表性工作：**

| 工作 | 机构 | 核心贡献 |
| --- | --- | --- |
| MultiPath | Google | 锚点轨迹 + 高斯混合，快速推理 |
| Wayformer | Waymo | 统一 Transformer，多模态注意力 |
| MTR（Motion Transformer） | 上海 AILab | 稀疏运动查询 + 全局意图定位 |
| MotionDiffuser | — | 扩散模型生成多模态轨迹分布 |


## 评估指标

| 指标 | 全称 | 计算公式 | 说明 |
| --- | --- | --- | --- |
| ADE | 平均位移误差 | $\frac{1}{T}\sum_{t=1}^T \|\hat{y}_t - y_t\|_2$ | 预测轨迹所有时刻的平均误差 |
| FDE | 最终位移误差 | $\|\hat{y}_T - y_T\|_2$ | 预测终点与真实终点的距离 |
| minADE@K | 最优 K 条中最小 ADE | $\min_{k} \text{ADE}(\hat{y}^{(k)})$ | 多模态评估，衡量最优猜测 |
| minFDE@K | 最优 K 条中最小 FDE | $\min_{k} \text{FDE}(\hat{y}^{(k)})$ | 同上，终点精度 |
| MR（漏检率） | Miss Rate | 预测终点与真实终点 > 2 m 的比例 | 安全相关，关注危险漏报 |


## 运动预测评估基准

### nuScenes Prediction Challenge

nuScenes 数据集（nuTonomy，2020）包含波士顿和新加坡城市场景，预测任务要求预测周围车辆未来 6 秒（12 帧 @ 2Hz）轨迹，评测指标为 minADE@5、minFDE@5 和 Miss Rate（终点误差 > 2m 视为漏检）。

下表列出部分代表性方法的排行榜成绩（数值越低越好）：

| 方法 | minADE@5 (m) | minFDE@5 (m) | Miss Rate |
| --- | --- | --- | --- |
| CV 基线（匀速） | 1.90 | 4.42 | 0.76 |
| CoverNet | 1.96 | 4.31 | 0.64 |
| MTP | 1.72 | 3.93 | 0.62 |
| WIMP | 1.47 | 3.33 | 0.55 |
| HiVT | **1.02** | **1.89** | **0.47** |
| MTR | 0.97 | 1.71 | 0.44 |

HiVT 和 MTR 等基于 Transformer 的方法相比早期 LSTM 基线将 minFDE@5 降低了约 50%，体现了地图感知和多智能体交互建模的重要性。

### Waymo Open Motion Dataset (WOMD)

WOMD 是目前规模最大的运动预测数据集之一，包含约 10 万个高质量场景片段。评测要求预测多类型道路使用者（车辆、行人、自行车）未来 8 秒轨迹，核心指标为：

- **Soft minFDE**（软最小终点误差）：对 Top-K 预测按置信度加权的 FDE
- **minADE**（最小平均轨迹误差）：K=6 条候选中最优者
- **Overlap Rate**：预测轨迹之间的物理碰撞率，衡量联合预测一致性

WOMD 要求同时预测最多 8 个感兴趣智能体（Agents of Interest），强调联合预测一致性，显著区别于单体评测框架。

### Argoverse 2 Motion Forecasting

Argoverse 2（Argo AI，2023）聚焦城市交叉口场景，提供更详细的 HD Map 标注（包括人行横道、停车标志、中央分隔线）。预测任务为未来 6 秒（60 帧 @ 10Hz），主要指标：

- **minADE@K**（$K \in \{1, 6\}$）
- **minFDE@K**
- **brier-minFDE**：将置信度纳入 FDE 计算，惩罚对错误模式赋予高权重的模型

Argoverse 2 的 10Hz 标注频率（vs nuScenes 的 2Hz）使其更适合评测短期精细轨迹预测。

### Top-K 预测多样性 vs 准确性权衡

多模态预测面临一个根本性权衡：增加预测条数 $K$ 能覆盖更多可能未来（提高召回率），但代价是规划层需要处理更多候选，计算和决策成本上升。

对于固定 $K$，模型需要在两个目标之间平衡：

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{acc}}}_{\text{准确性：minFDE 最小化}} + \lambda \underbrace{\mathcal{L}_{\text{div}}}_{\text{多样性：轨迹间距离最大化}}$$

过度追求多样性会导致预测轨迹"发散"（包含大量不合理的低概率轨迹），而过度追求准确性则会退化为单峰预测（所有 $K$ 条轨迹聚集在同一模式）。实践中通常用 NMS 后处理过滤冗余候选，同时用置信度得分筛选高质量预测。


## 预测结果在规划中的使用

### 预测轨迹作为规划约束

预测模块输出的多条候选轨迹（通常 Top-5 或 Top-10）传递给规划模块，作为动态障碍物的未来占用区域：

$$\mathcal{O}(t) = \bigcup_{k=1}^K \text{BoundingBox}\!\left(\hat{y}^{(k)}_t,\ \theta^{(k)}_t\right) \oplus \mathcal{B}_{\text{safe}}$$

其中 $\oplus$ 表示 Minkowski 和，$\mathcal{B}_{\text{safe}}$ 为安全膨胀区域。规划模块生成的轨迹须在所有时刻 $t$ 避开 $\mathcal{O}(t)$。

在实际工程中，通常取置信度最高的几条预测轨迹（而非全部 $K$ 条）用于碰撞检测，以控制规划计算量。

### 预测不确定性在碰撞检测中的处理

预测模型通常输出每个时刻的位置分布，可表示为**概率椭圆（Prediction Uncertainty Ellipse）**：

$$\mathbf{y}_t \sim \mathcal{N}(\mu_t,\ \Sigma_t)$$

碰撞风险概率可通过计算自车轨迹与预测分布的重叠积分得到：

$$P_{\text{collision}}(t) = \int_{\mathcal{A}_{\text{ego}}(t)} \mathcal{N}(y;\ \mu_t,\ \Sigma_t) \, dy$$

实际中常用保守近似：将概率椭圆扩展为固定置信水平（如 $2\sigma$，对应 95% 概率区域）的确定性边界框，再执行传统碰撞检测。

### 保守预测 vs 乐观预测的规划影响

| 预测策略 | 对规划的影响 | 风险 |
| --- | --- | --- |
| 保守预测（假设最坏情况） | 规划产生大量减速/等待 | 舒适性差、路口通行效率低 |
| 乐观预测（假设最有可能情况） | 规划更积极、通行效率高 | 若预测错误，碰撞风险上升 |
| 多模态加权 | 对高概率模态积极、低概率模态保守 | 计算复杂度高 |

自动驾驶系统通常在**安全临界场景**（TTC < 3s）采用保守策略，在**正常跟车场景**采用乐观策略，通过场景识别动态切换。

### 预测-规划联合优化

传统"先预测、后规划"的两步法存在根本性缺陷：**预测误差无法被规划纠正**。规划模块的输出（自车未来轨迹）会影响其他智能体的实际行为，但在两步法中这种影响无法反馈给预测模块。

联合预测-规划的优化目标：

$$\min_{\xi_{\text{ego}}} \mathcal{C}_{\text{plan}}(\xi_{\text{ego}}) \quad \text{s.t.} \quad p(\xi_{\text{agent}} \mid x,\ \xi_{\text{ego}}) \cdot \text{Safety}(\xi_{\text{ego}},\ \xi_{\text{agent}}) \geq \epsilon$$

其中 $\xi_{\text{ego}}$ 为自车规划轨迹，$\xi_{\text{agent}}$ 为条件于自车行为的其他智能体预测轨迹。

端到端的 ImitationPlanning 和 PLUTO 等框架尝试用单一网络同时输出预测和规划结果，从根本上消除两步法的误差耦合问题。


## 语言增强预测

### 行为意图的语言描述

大语言模型（LLM）的兴起为轨迹预测带来了新的研究方向：用自然语言描述驾驶场景中的行为意图，辅助传统预测模型：

- "该车正在低速行驶并频繁停车，可能正在寻找停车位"
- "前方行人正在看手机，注意力不在行驶方向上"
- "该车连续三次触碰车道线，疑似疲劳驾驶"

语言描述充当了从观测状态到行为意图的"桥梁"，将隐性的驾驶员意图显式化。基于语言意图的预测模型（如 DriveVLM）将文本意图编码作为额外条件输入轨迹解码器：

$$\hat{y} = f_\theta(\mathbf{z}_{\text{agent}},\ \mathbf{z}_{\text{map}},\ \mathbf{z}_{\text{lang}})$$

其中 $\mathbf{z}_{\text{lang}}$ 为语言意图描述的文本嵌入，由 CLIP 或 LLM 的文本编码器提取。

### 大模型辅助罕见场景预测

传统预测模型在**长尾场景**（如救护车高速穿越路口、施工区单行道、洪水积水绕行）中因训练样本稀少而失效。大模型具备丰富的世界知识，可以通过以下方式辅助罕见场景预测：

**思维链推理（Chain-of-Thought Reasoning）：**

```
输入：场景描述 + 当前状态
→ LLM："该场景为施工区，右侧车道封闭，前方车辆必须
         合并到左车道，预计在 50 m 处开始变道行为。"
→ 意图识别：变道（右 → 左）
→ 轨迹生成网络：基于意图约束生成轨迹
```

**数据增强（零样本生成训练数据）：**

用 LLM 生成罕见场景的语言描述，结合仿真器合成对应场景数据，弥补真实数据长尾不足的问题。

### 常识推理在预测中的应用

人类驾驶员依赖大量**隐性常识**进行预测，例如：

- 车辆不会无故停在高速公路中间（除非故障）
- 儿童放学时间学校附近行人密度高、行为随机性强
- 在超市停车场，车辆倒车概率远高于道路场景
- 消防车旁边的车辆很快会移走

这些常识无法从轨迹数据中直接学习，但可以通过 LLM 的知识提取得到。**场景常识约束（Commonsense Constraint）**作为软约束加入预测目标函数，使模型输出在语义上更合理：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{traj}} + \beta \cdot \mathcal{L}_{\text{commonsense}}$$

其中 $\mathcal{L}_{\text{commonsense}}$ 由 LLM 对预测轨迹的合理性打分反向传播得到，是将大模型知识蒸馏进预测网络的一种方式。


## 参考资料

1. A. Alahi et al. Social LSTM: Human Trajectory Prediction in Crowded Spaces. CVPR, 2016.
2. A. Gupta et al. Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks. CVPR, 2018.
3. T. Zhao et al. TNT: Target-driveN Trajectory Prediction. CoRL, 2020.
4. J. Gao et al. VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation. CVPR, 2020.
5. H. Shi et al. Motion Transformer with Global Intention Localization and Local Movement Refinement. NeurIPS, 2022.
6. Z. Zhou et al. HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction. CVPR, 2022.
7. C. Ettinger et al. Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset. ICCV, 2021.
8. B. Wilson et al. Argoverse 2: Next Generation Acceleration of Research in HD Mapping and Motion Forecasting. NeurIPS Datasets, 2023.
9. X. Jia et al. DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models. arXiv, 2024.
10. Y. Hu et al. PLUTO: Pushing the Limit of Imitation Learning-based Planning for Autonomous Driving. arXiv, 2024.
