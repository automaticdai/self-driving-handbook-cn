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

业界常用 **nuScenes Prediction**、**Argoverse**、**Waymo Open Motion** 等公开数据集评测。


## 参考资料

1. A. Alahi et al. Social LSTM: Human Trajectory Prediction in Crowded Spaces. CVPR, 2016.
2. A. Gupta et al. Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks. CVPR, 2018.
3. T. Zhao et al. TNT: Target-driveN Trajectory Prediction. CoRL, 2020.
4. J. Gao et al. VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation. CVPR, 2020.
5. H. Shi et al. Motion Transformer with Global Intention Localization and Local Movement Refinement. NeurIPS, 2022.
