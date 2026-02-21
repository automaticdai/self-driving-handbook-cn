# 激光点云数据处理

激光雷达（LiDAR，Light Detection and Ranging）是自动驾驶感知系统的核心传感器之一。它通过发射激光脉冲并测量反射时间（ToF，Time of Flight）来获取三维空间中物体的精确距离信息，进而生成**点云（Point Cloud）**数据。点云以毫米级精度描述车辆周围的三维几何结构，为目标检测、语义分割、定位建图等上游任务提供不可替代的基础信息。

与摄像头相比，激光雷达不受光照条件影响，可在夜间、强光及低对比度场景下稳定工作；与毫米波雷达相比，激光雷达的角分辨率高出数十倍，能够精确描绘物体形状轮廓。目前主流车载激光雷达包括机械旋转式（Velodyne HDL-64E）、固态式（Intel RealSense L515）和混合固态式（Livox Avia），线数从16线到128线不等。

---

## 1. 点云数据表示

### 1.1 无序点集（Raw Point Set）

激光雷达输出的原始数据是无序点集，每个点通常包含以下字段：

| 字段 | 含义 |
|------|------|
| $x, y, z$ | 三维笛卡尔坐标（单位：米） |
| intensity | 激光反射强度（0–255） |
| ring | 激光束编号（线号） |
| time | 相对时间戳（同帧内各点的采集偏差） |

点云的核心挑战在于其**无序性**：点的排列顺序没有固定的空间含义，无法直接应用标准二维卷积。一帧典型点云（64线，10 Hz）包含约 $1.2\times10^5$ 个点。

### 1.2 体素表示（Voxel）

将三维空间均匀划分为边长为 $v$ 的体素网格。落入同一体素的点被聚合为一个特征向量：

$$
\mathbf{f}_{voxel} = \frac{1}{|P_v|} \sum_{p_i \in P_v} \mathbf{p}_i
$$

其中 $P_v$ 为落入该体素的点集。体素表示将无序点云转化为规则的三维张量，可直接应用3D卷积。典型体素尺寸为 $0.1\,\text{m} \times 0.1\,\text{m} \times 0.1\,\text{m}$。

### 1.3 距离图像（Range Image / 球面投影）

将三维点云通过球面投影映射到二维图像空间。对点 $(x, y, z)$，其在距离图像中的像素坐标为：

$$
u = \left\lfloor \frac{1}{2}\left(1 - \frac{\arctan(y/x)}{\pi}\right) W \right\rfloor
$$

$$
v = \left\lfloor \left(1 - \frac{\arcsin(z/r) + f_{down}}{f}\right) H \right\rfloor
$$

其中 $r = \sqrt{x^2+y^2+z^2}$ 为点到原点的距离，$f = f_{up} + f_{down}$ 为雷达垂直视场角，$W$、$H$ 为距离图像的宽高。距离图像保留了点云的空间邻域关系，可直接应用二维卷积，是 RangeNet++ 等方法的基础。

### 1.4 Pillar 柱状表示

将点云在 $x$-$y$ 平面划分为大小为 $p_x \times p_y$ 的柱形区域（Pillar），每根 Pillar 在 $z$ 轴方向不限高度。每个点用9维特征 $(x, y, z, r, x_c, y_c, z_c, x_p, y_p)$ 表示，其中下标 $c$ 表示与 Pillar 内点中心的偏差，下标 $p$ 表示与 Pillar 中心的偏差。PointPillars 方法基于此表示实现极快的推理速度。

### 1.5 各表示方法对比

| 表示方法 | 空间精度 | 内存占用 | 推理速度 | 典型应用 |
|---------|---------|---------|---------|---------|
| 无序点集 | 最高 | 低 | 慢 | PointNet, Point-RCNN |
| 体素 | 高（受体素尺寸限制） | 高（含大量空体素） | 中 | VoxelNet, SECOND |
| 距离图像 | 中（有投影失真） | 低 | 快 | RangeNet++, SalsaNet |
| Pillar | 中 | 中 | 最快 | PointPillars |

---

## 2. 点云预处理

### 2.1 地面点分割

地面点分割是点云处理的第一步，目的是去除地面点以降低后续检测的干扰。

**RANSAC 平面拟合（Random Sample Consensus）**

RANSAC 通过随机采样迭代估计平面模型 $ax + by + cz + d = 0$，判断内点（Inlier）的条件为点到平面距离小于阈值 $\epsilon$：

$$
\text{dist}(p_i, \pi) = \frac{|ax_i + by_i + cz_i + d|}{\sqrt{a^2+b^2+c^2}} < \epsilon
$$

每次迭代随机选取3个点拟合平面，统计内点数量，保留内点最多的平面作为地面模型。设内点比例为 $w$，所需迭代次数为：

$$
N = \frac{\log(1-p)}{\log(1-w^3)}
$$

其中 $p$ 为期望的成功概率（通常取 0.99）。

**渐进形态学滤波（Progressive Morphological Filter，PMF）**

PMF 模拟地形测量中的滤波思路，通过多尺度开运算（形态学腐蚀后膨胀）逐步分离地面点与非地面点。滤波窗口尺寸从小到大递增，适应复杂地面起伏，常用于室外大范围场景。

### 2.2 降采样

**体素网格滤波（Voxel Grid Filter）**

将空间划分为体素网格，每个非空体素保留一个代表点（通常为质心）。降采样比例可通过调节体素边长 $v$ 控制。适用于均匀降采样，计算复杂度约为 $O(N \log N)$。

**最远点采样（Farthest Point Sampling，FPS）**

FPS 迭代地选择距已选点集最远的点，保证采样点在空间上的均匀分布。设已采样点集为 $S$，下一个采样点为：

$$
p^* = \arg\max_{p_i \notin S} \min_{p_j \in S} d(p_i, p_j)
$$

FPS 比随机采样保留更多边界和细节信息，是 PointNet++、SA-SSD 等方法的核心组件，但计算复杂度为 $O(N^2)$，常用 GPU 加速实现。

### 2.3 去噪

**统计离群值去除（Statistical Outlier Removal，SOR）**

对每个点 $p_i$，计算其 $k$ 近邻点的平均距离 $\bar{d}_i$。若 $\bar{d}_i$ 超过全局均值 $\mu$ 加上标准差 $\sigma$ 的倍数：

$$
\bar{d}_i > \mu + \alpha \cdot \sigma
$$

则将 $p_i$ 判定为噪声点并剔除。参数 $k$（近邻数）和 $\alpha$（标准差倍数）需根据场景调整。

### 2.4 坐标变换与外参标定

激光雷达输出的点云坐标基于传感器自身坐标系，需通过外参矩阵 $\mathbf{T}$ 变换到车辆坐标系（以后轴中心为原点）。变换关系为：

$$
\mathbf{p}_{vehicle} = \mathbf{T}_{lidar}^{vehicle} \cdot \mathbf{p}_{lidar} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

其中 $\mathbf{R} \in SO(3)$ 为旋转矩阵，$\mathbf{t} \in \mathbb{R}^3$ 为平移向量。外参标定通常使用棋盘格靶标或球形靶标，通过多帧优化求解。

---

## 3. 基于深度学习的 3D 目标检测

### 3.1 PointNet

PointNet（Qi et al., CVPR 2017）是第一个直接处理无序点集的深度学习框架，解决了点云无序性问题的核心思路是使用**对称函数（Symmetric Function）**——最大池化（Max Pooling）。

全局特征聚合公式为：

$$
f(x_1, \ldots, x_n) = \gamma\!\left(\max_{i=1\ldots n}\{h(x_i)\}\right)
$$

其中 $h: \mathbb{R}^3 \to \mathbb{R}^K$ 为共享权重的多层感知机（MLP），$\max$ 为逐维最大池化，$\gamma$ 为后续 MLP。由于 $\max$ 操作对输入点的排列不变，整个网络具有排列不变性。

**T-Net（空间变换网络）**

PointNet 引入 T-Net 预测一个 $3\times3$ 或 $64\times64$ 的变换矩阵，对输入点或特征进行对齐，以提高网络对旋转的鲁棒性。正则化约束变换矩阵接近正交矩阵：

$$
L_{reg} = \|\mathbf{I} - \mathbf{A}\mathbf{A}^T\|_F^2
$$

### 3.2 PointNet++

PointNet++（Qi et al., NeurIPS 2017）引入分层局部特征提取，解决了 PointNet 无法捕获局部几何结构的问题。

**集合抽象层（Set Abstraction，SA）**

SA 层包含三步：
1. **FPS 采样**：选取 $N'$ 个中心点
2. **球形邻域查询**：对每个中心点以半径 $r$ 查找邻域点
3. **PointNet 局部编码**：在每个邻域内应用 PointNet 提取局部特征

**多尺度分组（Multi-Scale Grouping，MSG）**

MSG 使用多个不同半径 $r_1 < r_2 < \ldots < r_K$ 并行提取局部特征，并将多尺度特征拼接：

$$
\mathbf{f}_{i}^{MSG} = \text{concat}\left[\text{PointNet}(B(p_i, r_1)), \ldots, \text{PointNet}(B(p_i, r_K))\right]
$$

其中 $B(p_i, r)$ 为以 $p_i$ 为中心、半径为 $r$ 的邻域球。MSG 对点云密度变化（远近不同）具有更强的鲁棒性。

### 3.3 VoxelNet

VoxelNet（Zhou & Tuzel, CVPR 2018）将点云体素化后端到端学习三维特征，核心组件为**体素特征编码层（Voxel Feature Encoding，VFE）**。

VFE 层对体素内每个点 $p_i$ 与体素质心 $\bar{p}$ 的差值拼接为增强特征：

$$
\tilde{p}_i = [x_i, y_i, z_i, r_i, x_i - \bar{x}, y_i - \bar{y}, z_i - \bar{z}]
$$

经 MLP 和逐元素最大池化后与局部特征拼接，再经多层 VFE 得到体素特征张量。随后用3D稀疏卷积和区域候选网络（RPN）完成检测。

### 3.4 SECOND

SECOND（Yan et al., Sensors 2018）的关键创新是引入**稀疏卷积（Sparse Convolution）**，仅对非空体素计算，大幅降低计算量和内存开销。

**SubM 卷积（Submanifold Sparse Convolution）**

SubM 卷积只在输入非空位置处产生输出，保持稀疏性不退化：

$$
\text{output}_{x} = \begin{cases} \sum_{k} W_k \cdot \text{input}_{x+k} & \text{若 } x \in \Omega_{in} \\ 0 & \text{其他} \end{cases}
$$

其中 $\Omega_{in}$ 为输入非空位置集合。SECOND 将 VoxelNet 推理速度提升约5倍，成为后续众多方法的骨干网络。

### 3.5 PointPillars

PointPillars（Lang et al., CVPR 2019）采用 Pillar 表示，将三维问题转化为二维，实现实时检测（>60 FPS）。

**PointPillar 特征提取流程：**
1. 将点云划分为 $H\times W$ 个 Pillar（鸟瞰图像素）
2. 每个 Pillar 内保留最多 $P$ 个点，不足则补零
3. 用简化版 PointNet 对每个 Pillar 提取特征向量 $\mathbf{c} \in \mathbb{R}^C$
4. 将特征向量"散射"回 $H\times W\times C$ 的伪图像
5. 用标准二维卷积骨干（SSD 风格）处理伪图像并输出检测结果

Pillar 的 $x$-$y$ 尺寸通常为 $0.16\,\text{m}$，端到端延迟可低至 16 ms。

### 3.6 CenterPoint

CenterPoint（Yin et al., CVPR 2021）摒弃传统的锚框（Anchor）设计，采用**中心点热力图**表示目标，框架更简洁且对旋转不敏感。

**高斯热力图（Gaussian Heatmap）**

对每个真值目标，以其 BEV 中心点 $(x_c, y_c)$ 为圆心，在热力图上叠加二维高斯分布：

$$
Y_{xy} = \exp\!\left(-\frac{(x-x_c)^2+(y-y_c)^2}{2\sigma^2}\right)
$$

其中 $\sigma$ 由目标尺寸自适应决定：$\sigma = \frac{1}{3}\min(w, l) / r$，$r$ 为输出步长。

检测头包括：热力图头（分类）、偏移量头（亚像素中心修正）、高度头（$z$ 坐标）、尺寸头（$w, l, h$）、朝向头（$\sin\theta, \cos\theta$）和速度头（用于跟踪）。

### 3.7 主流方法对比

| 方法 | 输入表示 | 3D mAP（KITTI Car, Mod） | 推理速度 | 参数量 |
|------|---------|------------------------|---------|-------|
| PointNet++ | 点集 | ~70% | ~5 FPS | ~1.7 M |
| VoxelNet | 体素 | 65.11% | ~4 FPS | ~4.3 M |
| SECOND | 稀疏体素 | 76.48% | ~20 FPS | ~5.3 M |
| PointPillars | Pillar | 74.99% | >60 FPS | ~4.8 M |
| CenterPoint | 体素/Pillar | 78.4%（nuScenes） | ~25 FPS | ~6.1 M |

---

## 4. 多帧点云融合

### 4.1 时序累积与 Ego-Motion 补偿

单帧点云稀疏性限制了远距离目标的检测能力。将多帧历史点云累积可显著增加点密度，但需先进行**自车运动（Ego-Motion）补偿**，将历史帧点云变换到当前帧坐标系：

$$
\mathbf{p}^{(t)}_i = \mathbf{T}^t_{t_k} \cdot \mathbf{p}^{(t_k)}_i, \quad k = 1, \ldots, K
$$

其中 $\mathbf{T}^t_{t_k}$ 为从历史时刻 $t_k$ 到当前时刻 $t$ 的位姿变换矩阵，通过 GPS/IMU 或激光里程计获取。累积帧数通常为 3–10 帧。

### 4.2 4D 点云（加入时间维度）

将时间戳作为第四维度附加到点云坐标，构成 $(x, y, z, \Delta t)$ 的4D点云。运动目标因时间偏移而形成"拖尾"形状，可用于：
- 区分静态背景与动态目标
- 估计目标运动速度
- 提升对快速运动目标的检测鲁棒性

4D 方法在 nuScenes 等数据集上已成为主流做法，部分方法（如 PillarNet）通过 4D 时序特征显著提升远处小目标的召回率。

### 4.3 点云地图增量更新

在自动驾驶系统中，需对高精点云地图进行实时增量更新，以反映环境变化（施工区域、临时障碍物等）。常用策略包括：

- **TSDF（Truncated Signed Distance Function）**：将空间建模为有符号距离场，支持增量融合新点云
- **OctoMap**：基于八叉树的概率占用地图，内存高效且支持动态更新
- **动静分离**：先识别并剔除动态物体点云（行人、车辆），再更新静态背景地图

---

## 5. 激光 SLAM

### 5.1 ICP（迭代最近点）

ICP（Besl & McKay, 1992）是激光点云配准的经典算法，目标是求解变换 $(\mathbf{R}, \mathbf{t})$ 最小化两帧点云间的距离：

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_{i=1}^{N} \|\mathbf{p}^{source}_i - (\mathbf{R}\mathbf{p}^{target}_{c(i)} + \mathbf{t})\|^2
$$

其中 $c(i)$ 为点 $\mathbf{p}^{source}_i$ 在目标点云中的最近邻。算法交替执行"最近邻匹配"和"最小二乘求解"直至收敛。ICP 对初始值敏感，常需借助 NDT 或 IMU 提供初始位姿估计。

**ICP 变体：**
- **Point-to-Plane ICP**：最小化源点到目标点所在切平面的距离，收敛速度更快
- **Generalized ICP（GICP）**：将点云建模为局部协方差分布，兼具 Point-to-Point 和 Point-to-Plane 的优点

### 5.2 NDT（正态分布变换）

NDT（Biber & Strasser, 2003）将参考点云划分为体素网格，在每个体素内拟合点的正态分布 $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$。对源点云中的点 $\mathbf{p}$，其得分为：

$$
s(\mathbf{p}) = \exp\!\left(-\frac{(\mathbf{p} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\mathbf{p} - \boldsymbol{\mu}_k)}{2}\right)
$$

优化目标为最大化所有点的总得分，使用牛顿法迭代求解。NDT 对点云密度变化更鲁棒，是 Autoware 等开源自动驾驶平台的核心定位算法。

### 5.3 LOAM（激光里程计与建图）

LOAM（Zhang & Singh, RSS 2014）通过提取两类几何特征进行配准：

**边缘点（Edge Points）**：局部曲率 $c$ 较大的点，位于物体棱边处：

$$
c = \frac{1}{|S| \cdot \|\mathbf{p}_i\|} \left\| \sum_{j \in S, j \neq i} (\mathbf{p}_j - \mathbf{p}_i) \right\|
$$

**平面点（Planar Points）**：局部曲率 $c$ 较小的点，位于地面或墙面。

LOAM 将里程计（高频，10 Hz）与建图（低频，1 Hz）解耦，里程计用边缘-边缘和平面-平面匹配快速估计位姿，建图用更精细的优化更新地图。

### 5.4 LeGO-LOAM

LeGO-LOAM（Shan & Englot, IROS 2018）针对地面车辆场景对 LOAM 进行了优化：

1. **地面分离**：先分割地面点，仅对地面点提取平面特征，对非地面点提取边缘特征
2. **两步优化**：先用地面平面点优化 $z$、$\text{roll}$、$\text{pitch}$，再用边缘点优化 $x$、$y$、$\text{yaw}$，降低计算量
3. **回环检测**：集成基于欧氏距离的位姿图回环检测，减少长时间累积漂移

LeGO-LOAM 在嵌入式平台（如 Jetson TX2）上可实时运行，被广泛用于低速园区自动驾驶场景。

### 5.5 LIO-SAM（激光惯性里程计）

LIO-SAM（Shan et al., IROS 2020）将激光雷达与 IMU 紧耦合，在**因子图（Factor Graph）**框架下联合优化：

**系统包含四类因子：**
- **IMU 预积分因子**：通过预积分在关键帧之间传播高频 IMU 测量
- **激光里程计因子**：相邻关键帧间的相对位姿约束（由特征匹配得到）
- **GPS 因子**：当 GPS 信号可用时提供绝对位置约束
- **回环因子**：检测到回环时加入回环约束

IMU 预积分在两关键帧 $i$、$j$ 之间累积旋转、速度和位移的增量：

$$
\Delta\mathbf{R}_{ij} = \prod_{k=i}^{j-1}\text{Exp}\!\left(\tilde{\boldsymbol{\omega}}_k \Delta t\right), \quad \Delta\mathbf{v}_{ij} = \sum_{k=i}^{j-1}\Delta\mathbf{R}_{ik}\tilde{\mathbf{a}}_k\Delta t
$$

通过 GTSAM 库的增量平滑算法（iSAM2）在线求解因子图，实现精确且实时的激光惯性里程计。

---

## 6. 语义点云分割

### 6.1 RangeNet++

RangeNet++（Milioto et al., IROS 2019）基于球面投影的距离图像进行语义分割，流程为：
1. 将点云投影为距离图像（$H\times W$，通常为 $64\times2048$）
2. 用 2D 语义分割网络（DarkNet-53 骨干）预测每像素类别
3. 使用 KNN 后处理将语义标签传播回三维点云（修正投影失真导致的误差）

RangeNet++ 在 SemanticKITTI 数据集上达到 52.2% mIoU，推理速度约 24 FPS（在 RTX 2080Ti 上），是实时点云分割的代表性方法。

### 6.2 Cylinder3D

Cylinder3D（Zhu et al., CVPR 2021）采用**柱坐标（Cylindrical Coordinates）**划分点云，将点云表示为 $(\rho, \varphi, z)$ 空间中的体素，比笛卡尔体素更适应激光雷达的近密远疏分布特性。

核心创新：**非对称残差网络（Asymmetrical Residual Block）**在高度方向使用较小的卷积核，在俯仰方向使用较大的卷积核，匹配点云的各向异性分布。Cylinder3D 在 SemanticKITTI 排行榜上长期名列前茅，mIoU 达 65.9%。

### 6.3 全景点云分割（Panoptic Segmentation）

全景分割同时预测**语义类别**（所有点的类别）和**实例 ID**（可数目标的实例区分），是点云理解的最高层次任务。

**Panoptic-PolarNet**（Zhou et al., CVPR 2021）基于极坐标 BEV 表示：
- **Stuff 分支**：预测道路、植被等不可数类别的语义标签
- **Thing 分支**：预测车辆、行人等可数类别的实例中心热力图，结合聚类生成实例 ID
- **全景融合**：合并两个分支的输出，生成最终全景标注

全景评估指标 **Panoptic Quality（PQ）** 定义为：

$$
PQ = \underbrace{\frac{\sum_{(p,g)\in TP} IoU(p,g)}{|TP|}}_{\text{Recognition Quality (RQ)}} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{\text{Segmentation Quality (SQ)}}
$$

---

## 7. 评估指标

### 7.1 3D 目标检测指标

**3D IoU（三维交并比）**定义为两个三维旋转框体积之交与之并的比值：

$$
IoU_{3D} = \frac{V_{pred} \cap V_{gt}}{V_{pred} \cup V_{gt}}
$$

判定检测成功的常用阈值为 $IoU \geq 0.5$（行人/自行车）和 $IoU \geq 0.7$（车辆）。

**平均精度（Average Precision，AP）**通过在召回率-精度曲线下积分计算：

$$
AP = \int_0^1 p(r)\,dr \approx \sum_{k} p(r_k) \Delta r_k
$$

**平均 AP（mAP）**为所有目标类别 AP 的均值。

### 7.2 主流数据集评估指标对比

| 数据集 | 主要指标 | IoU 阈值 | 类别数 | 备注 |
|--------|---------|---------|-------|------|
| KITTI | 3D AP / BEV AP | 0.7（车）/ 0.5（行人） | 3 | 按难易分 Easy/Mod/Hard |
| nuScenes | NDS, mAP | 0.5–0.5 m（距离阈值） | 10 | 含速度和属性评估 |
| Waymo Open | mAPH（含朝向） | IoU 0.7 / 0.5 | 3+2 | 分近远距离分别统计 |

**nuScenes 检测分数（NDS）**综合考虑多项指标：

$$
NDS = \frac{1}{10}\left[5 \cdot mAP + \sum_{m \in \mathcal{M}} (1 - \min(1, e_m))\right]
$$

其中 $\mathcal{M}$ 包含平移误差（ATE）、尺度误差（ASE）、朝向误差（AOE）、速度误差（AVE）和属性误差（AAE）共5项。

### 7.3 语义分割指标

**平均交并比（mIoU）**为各类别 IoU 的均值，是点云语义分割最通用的评估指标：

$$
mIoU = \frac{1}{C}\sum_{c=1}^{C}\frac{TP_c}{TP_c + FP_c + FN_c}
$$

其中 $C$ 为类别总数，$TP_c$、$FP_c$、$FN_c$ 分别为类别 $c$ 的真正例、假正例和假负例点数。

---

## 参考资料

1. Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation**. *CVPR 2017*.
2. Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). **PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space**. *NeurIPS 2017*.
3. Lang, A. H., Vora, S., Caesar, H., Zhou, L., Yang, J., & Beijbom, O. (2019). **PointPillars: Fast Encoders for Object Detection from Point Clouds**. *CVPR 2019*.
4. Yan, Y., Mao, Y., & Li, B. (2018). **SECOND: Sparsely Embedded Convolutional Detection**. *Sensors, 18(10)*.
5. Yin, T., Zhou, X., & Krahenbuhl, P. (2021). **Center-based 3D Object Detection and Tracking**. *CVPR 2021*.
6. Zhang, J., & Singh, S. (2014). **LOAM: Lidar Odometry and Mapping in Real-time**. *RSS 2014*.
7. Shan, T., Englot, B., Meyers, D., Wang, W., Ratti, C., & Rus, D. (2020). **LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping**. *IROS 2020*.
