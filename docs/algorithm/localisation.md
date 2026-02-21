# 地图与定位

自动驾驶车辆需要在全局和局部两个尺度上回答"我在哪里"这个问题。定位（Localization）精度直接决定了轨迹规划和控制的可行性——自动驾驶通常要求厘米级定位精度，远超普通导航 GPS 的米级精度。车辆一旦偏离车道超过 20 cm，即可能引发安全事故。


## 定位方法体系

### GNSS/RTK 定位

全球导航卫星系统（GNSS，Global Navigation Satellite System）通过测量卫星信号传播时间计算接收机位置。现有四大系统：美国 GPS、中国北斗（BDS）、欧洲 Galileo、俄罗斯 GLONASS。

| 定位方式 | 精度 | 技术原理 | 局限 |
| --- | --- | --- | --- |
| 单点 GPS | 1–5 m | 测量伪距，受大气层和多径效应影响 | 无法满足自动驾驶需求 |
| DGPS（差分 GPS） | 0.5–1 m | 已知参考站广播误差修正 | 精度仍不足 |
| RTK（实时动态差分） | 1–2 cm | 传输载波相位差分，实时解算整周模糊度 | 需密集基站、信号良好环境 |

#### 伪距观测方程

GNSS 定位的核心观测量为**伪距**（Pseudorange）。接收机到第 $i$ 颗卫星的伪距观测方程为：

$$\rho_i = \|\mathbf{r} - \mathbf{r}^{(i)}\| + c\,\delta t - c\,\delta t^{(i)} + T_i + I_i + \epsilon_i$$

其中：
- $\rho_i$：接收机到第 $i$ 颗卫星的伪距观测值（m）
- $\mathbf{r}$：接收机位置向量（未知量）
- $\mathbf{r}^{(i)}$：第 $i$ 颗卫星位置向量（由星历计算得到）
- $c$：光速（$\approx 3\times10^8$ m/s）
- $\delta t$：接收机时钟误差（未知量）
- $\delta t^{(i)}$：卫星时钟误差（由导航电文给出）
- $T_i$：对流层延迟（约 2–20 m，天顶方向约 2.3 m）
- $I_i$：电离层延迟（约 1–50 m，可用双频消除）
- $\epsilon_i$：多径效应和接收机噪声（约 0.1–1 m）

四颗卫星可解算出接收机三维位置 $(x,y,z)$ 和时钟误差 $\delta t$ 四个未知量；多余观测量通过最小二乘提高精度。

#### 差分原理

差分 GPS 利用位置已知的**参考站**（基准站）计算误差修正量，向用户播发：

$$\Delta\rho_i = \rho_i^{\text{ref,obs}} - \rho_i^{\text{ref,calc}} = c\,\delta t + T_i + I_i + \epsilon_i$$

用户接收机用同一卫星的差分修正量消除大气误差和时钟误差，将精度从米级提升至分米级（DGPS）或厘米级（RTK 载波相位差分）。

**GNSS 的主要局限：**
- 城市峡谷：高楼、隧道、立交桥遮挡信号，产生多径效应
- 更新频率低：通常仅 1–10 Hz，无法满足实时控制需求
- 需配合 IMU 进行航位推算（Dead Reckoning）填补信号丢失间隔


### 惯性导航（IMU）与预积分

惯性测量单元（IMU，Inertial Measurement Unit）集成三轴加速度计和三轴陀螺仪，通过积分计算相对位移和姿态。

#### 连续时间运动方程

IMU 连续时间积分模型（忽略噪声）：

$$\dot{\mathbf{p}} = \mathbf{v}, \quad \dot{\mathbf{v}} = \mathbf{R}\,(\mathbf{a}_m - \mathbf{b}_a) - \mathbf{g}, \quad \dot{\mathbf{R}} = \mathbf{R}\,\lfloor\boldsymbol{\omega}_m - \mathbf{b}_g\rfloor_\times$$

其中 $\mathbf{p}$ 为位置，$\mathbf{v}$ 为速度，$\mathbf{R} \in SO(3)$ 为旋转矩阵，$\mathbf{a}_m$ 为加速度计读数，$\boldsymbol{\omega}_m$ 为陀螺仪读数，$\mathbf{b}_a / \mathbf{b}_g$ 为零偏，$\mathbf{g}$ 为重力向量。

#### 离散预积分因子

直接积分要求每次状态更新后重新从第 $i$ 帧积分到第 $j$ 帧，计算量巨大。**预积分（Pre-integration）** 将积分量从绝对参考帧中解耦，定义在两帧之间的相对变化量：

$$\Delta\mathbf{R}_{ij} = \prod_{k=i}^{j-1} \mathrm{Exp}\bigl((\boldsymbol{\omega}_k - \mathbf{b}_g)\,\Delta t\bigr)$$

$$\Delta\mathbf{v}_{ij} = \sum_{k=i}^{j-1} \Delta\mathbf{R}_{ik}\,(\mathbf{a}_k - \mathbf{b}_a)\,\Delta t$$

$$\Delta\mathbf{p}_{ij} = \sum_{k=i}^{j-1} \left[\Delta\mathbf{v}_{ik}\,\Delta t + \frac{1}{2}\Delta\mathbf{R}_{ik}\,(\mathbf{a}_k - \mathbf{b}_a)\,\Delta t^2\right]$$

其中 $\mathrm{Exp}(\cdot): \mathfrak{so}(3) \to SO(3)$ 为李群指数映射，$\Delta t$ 为 IMU 采样间隔。

预积分量 $(\Delta\mathbf{R}_{ij}, \Delta\mathbf{v}_{ij}, \Delta\mathbf{p}_{ij})$ 可在两关键帧之间**离线计算并缓存**，当零偏估计更新时，只需对缓存量进行线性修正（一阶 Jacobian 近似），避免重新积分。这使 IMU 预积分成为图优化（Factor Graph）框架中的高效约束因子。

| 等级 | 陀螺仪漂移 | 加速度计偏差 | 典型应用 |
| --- | --- | --- | --- |
| 消费级 | > 10 °/h | > 1 mg | 手机姿态、步行导航 |
| 战术级 | 0.1–10 °/h | 0.1–1 mg | 无人机、机器人 |
| 导航级 | < 0.01 °/h | < 0.05 mg | 自动驾驶、航空航天 |

- **优点**：高频（100–1000 Hz）、全环境可用（不依赖外部信号）
- **缺点**：积分误差累积（漂移），单独使用数秒后误差即达数米


### 高精地图匹配定位

将实时传感器数据与预先建立的高精度地图对比，通过最大化匹配度确定车辆位姿：

**高精地图通常包含三层：**
1. **点云层**：激光扫描的三维点云地图，用于 LiDAR 实时匹配
2. **特征层**：路面标线、灯杆、护栏等几何特征，用于快速匹配
3. **语义层**：车道线拓扑、信号灯位置、限速等道路属性

**国内主要高精地图厂商：** 百度地图、高德地图、四维图新（与 HERE 合资）、Momenta 众包地图。

**局限：** 地图覆盖率和更新频率是制约高精地图定位普及的核心瓶颈；施工区域、新开通道路等场景地图滞后严重。


## SLAM（同步定位与建图）

SLAM（Simultaneous Localization and Mapping，同步定位与建图）在未知或部分已知环境中同步估计车辆位姿和构建地图，是自动驾驶建图与定位的关键技术。

### 问题建模

SLAM 本质上是在线贝叶斯推断问题，联合估计：
- 车辆历史轨迹 $x_{1:t}$
- 地图 $m$
- 在传感器观测 $z_{1:t}$ 和控制输入 $u_{1:t}$ 条件下的后验概率：

$$p(x_{1:t}, m \mid z_{1:t}, u_{1:t}) \propto p(z_t \mid x_t, m) \cdot p(x_t \mid x_{t-1}, u_t) \cdot p(x_{1:t-1}, m \mid z_{1:t-1}, u_{1:t-1})$$


## 激光 SLAM

基于 LiDAR 点云的 SLAM 是自动驾驶的主流建图路线。

### LOAM 详解

**LOAM（Lidar Odometry and Mapping in Real-time，Zhang & Singh, RSS 2014）** 是影响深远的激光 SLAM 系统，核心思想是将复杂的 6-DOF 点云配准问题分解为**特征提取 + 两步优化**。

#### 特征点提取

LOAM 从每帧点云中提取两类几何特征点，依据每个点的**局部曲率**来判别：

对于第 $i$ 个点，定义曲率评分：

$$c_i = \frac{1}{|S|\,\|p_i\|} \left\| \sum_{j \in S,\, j \neq i} (p_j - p_i) \right\|$$

其中 $S$ 为同一扫描线上以 $i$ 为中心的邻域点集（通常取前后各 5 个点）：

| 特征类型 | 判定条件 | 物理含义 |
| --- | --- | --- |
| **边缘点（Edge Points）** | $c_i > c_{\text{th,high}}$（曲率大） | 物体边缘、杆状物体 |
| **平面点（Planar Points）** | $c_i < c_{\text{th,low}}$（曲率小） | 地面、墙面、地板 |

为保证特征均匀分布，LOAM 将每条扫描线分为若干子区间，每个子区间最多选取固定数量的边缘点和平面点。

#### 两步优化

LOAM 将 SLAM 分解为运行在**不同频率**的两个模块，实现里程计与建图的解耦：

**Step 1：激光里程计（LiDAR Odometry，10 Hz）**

在相邻两帧之间，利用边缘点和平面点分别建立点到线/点到面的距离约束：

- 边缘点 $p$ 到上一帧对应线段 $(\mathbf{a}, \mathbf{b})$ 的距离：

$$d_e = \frac{\|(p - \mathbf{a}) \times (p - \mathbf{b})\|}{\|\mathbf{a} - \mathbf{b}\|}$$

- 平面点 $p$ 到上一帧对应平面 $(\mathbf{a}, \mathbf{b}, \mathbf{c})$ 的距离：

$$d_p = \frac{|(p - \mathbf{a}) \cdot \left[(\mathbf{b}-\mathbf{a}) \times (\mathbf{c}-\mathbf{a})\right]|}{|(\mathbf{b}-\mathbf{a}) \times (\mathbf{c}-\mathbf{a})|}$$

通过最小化总约束残差（Levenberg-Marquardt 非线性最小二乘）求解帧间变换 $\mathbf{T}_{k,k+1} \in SE(3)$，频率 10 Hz，用于实时位姿估计。

**Step 2：激光建图（LiDAR Mapping，1 Hz）**

将里程计估计的位姿和特征点云融合进全局地图（Voxel Map），并用全局地图的特征对里程计位姿进行精化（Map Refinement），频率 1 Hz。

此**解耦设计**使里程计轻量运行、响应快，而建图精化离线慢速运行，整体实现实时运行的 cm 级精度。

#### LOAM 系统指标（KITTI 数据集）

| 指标 | LOAM 原版 | LeGO-LOAM | LIO-SAM |
| --- | --- | --- | --- |
| 平均平移误差 | 0.78% | 1.12% | 0.64% |
| 平均旋转误差 | 0.35°/100m | 0.57°/100m | 0.27°/100m |
| 运行频率 | 10 Hz | 10 Hz | 10 Hz |
| CPU 占用 | 高（全量特征） | 低（地面分割优化） | 中（IMU 辅助） |


### LIO-SAM 详解

**LIO-SAM（Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping，Shan et al., IROS 2020）** 是当前最广泛使用的激光-IMU 紧耦合 SLAM 系统，在 LOAM 基础上引入因子图优化框架和 IMU 预积分。

#### 四因子图架构

LIO-SAM 将所有传感器约束统一建模为**因子图（Factor Graph）**中的边，在 GTSAM 框架上实时求解：

```
                     ┌────────────────────────────────────────────────────┐
                     │              LIO-SAM 因子图                        │
                     │                                                    │
  x_i ──[IMU预积分]──► x_{i+1} ──[IMU预积分]──► x_{i+2} ──...           │
   │                       │                       │                     │
[LiDAR里程计因子]    [LiDAR里程计因子]       [LiDAR里程计因子]            │
   │                                                                      │
[GPS因子]（当GNSS可用时添加绝对位置约束）                                 │
   │                                                                      │
[回环因子]（检测到回环时添加相对位姿约束）                                 │
                     └────────────────────────────────────────────────────┘
```

**四类因子说明：**

| 因子类型 | 数据来源 | 频率 | 约束类型 |
| --- | --- | --- | --- |
| **IMU 预积分因子** | IMU（100–500 Hz） | 高频 | 相邻关键帧间相对运动约束（ $\Delta\mathbf{R}_{ij}, \Delta\mathbf{v}_{ij}, \Delta\mathbf{p}_{ij}$） |
| **LiDAR 里程计因子** | LiDAR 特征匹配（10 Hz） | 中频 | 相邻关键帧间 6-DOF 相对位姿 |
| **GPS 因子** | GNSS/RTK（1–10 Hz） | 低频 | 绝对位置约束（3-DOF，含不确定性） |
| **回环检测因子** | Scan Context 描述子 | 按需 | 历史帧间相对位姿约束，消除累积误差 |

**实时运行策略：** LIO-SAM 使用**边缘化（Marginalization）**策略，将超出滑动窗口的旧因子转化为先验因子后丢弃，保持因子图规模恒定，实现 10 Hz 实时运行。在 128线 LiDAR + 中等 CPU（i7-9750H）配置下，LIO-SAM 整体耗时约 **50–80 ms/帧**。

**Scan Context 回环检测：** 将点云编码为极坐标扇区统计直方图，计算两帧点云的环境相似度，平均描述子计算时间 < 5 ms，检测准确率 > 95%（KITTI 数据集）。


## 视觉 SLAM 补充

### ORB-SLAM3 详解

**ORB-SLAM3（Campos et al., IEEE T-RO 2021）** 是视觉 SLAM 领域最完整的开源系统，支持单目、双目、RGB-D 以及视觉-IMU 融合。

#### ORB 特征提取

ORB（Oriented FAST and Rotated BRIEF）特征具有旋转不变性和快速提取优势：

1. **关键点检测（FAST角点）：** 在每层图像金字塔（8层，缩放因子 1.2）上检测 FAST 角点，通过像素圆弧亮度对比筛选角点
2. **主方向计算：** 利用灰度矩计算关键点主方向：$\theta = \mathrm{atan2}(m_{01}, m_{10})$，其中 $m_{pq} = \sum_{x,y} x^p y^q I(x,y)$
3. **BRIEF 描述子：** 以主方向旋转预定义的测试点对，计算 256-bit 二进制描述子

#### PnP 位姿求解

视觉 SLAM 的**重定位**和**初始化**依赖 **PnP（Perspective-n-Point）** 问题求解：给定 $n$ 对 2D-3D 点对应 $\{(u_i, v_i) \leftrightarrow (X_i, Y_i, Z_i)\}$，求解相机位姿 $(\mathbf{R}, \mathbf{t})$：

$$\lambda_i \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix} = \mathbf{K} \left[ \mathbf{R} \mid \mathbf{t} \right] \begin{bmatrix} X_i \\ Y_i \\ Z_i \\ 1 \end{bmatrix}$$

其中 $\mathbf{K}$ 为相机内参矩阵，$\lambda_i$ 为深度。ORB-SLAM3 使用 **EPnP + RANSAC** 求解，最少 4 点即可求解，RANSAC 迭代处理外点，鲁棒性强。

#### 词袋回环检测

ORB-SLAM3 采用 **DBoW2（Bag of Words）** 进行回环候选帧检索：

1. 离线阶段：用大量图像训练视觉词汇表（Vocabulary Tree，10万词汇）
2. 在线阶段：将当前帧 ORB 描述子量化为词汇向量 $\mathbf{v}_t$
3. 检索历史帧中向量相似度最高的帧作为回环候选：$\text{score}(t, t') = \|\mathbf{v}_t - \mathbf{v}_{t'}\|_1$
4. 几何验证：用 Essential Matrix 筛除错误回环
5. 通过 Pose Graph 优化消除回环累积误差

**ORB-SLAM3 关键性能指标（EuRoC 数据集 MH_01）：**

| 模式 | ATE（m） | 运行频率 |
| --- | --- | --- |
| 单目 | 0.016 | 实时（30 Hz） |
| 双目 | 0.009 | 实时（30 Hz） |
| 单目-IMU | 0.008 | 实时（30 Hz） |

### 其他视觉 SLAM 系统

| 系统 | 类型 | 传感器 | 核心方法 |
| --- | --- | --- | --- |
| ORB-SLAM3 | 特征点法 | 单目/双目/RGB-D/+IMU | ORB 特征 + DBoW2 词袋回环 |
| DSO | 直接法 | 单目 | 光度一致性联合优化，无特征提取 |
| VINS-Mono | 视觉惯性里程计 | 单目+IMU | 紧耦合预积分 + 滑动窗口优化 |
| SVO | 半直接法 | 单目 | 快速特征跟踪 + 深度滤波 |
| OpenVINS | 视觉惯性 | 单目/多目+IMU | MSCKF 滤波器 |

**视觉 SLAM 主要挑战：**
- 无纹理区域（白墙、沥青路面）特征匮乏
- 光照剧烈变化（隧道出入口、日落逆光）
- 运动模糊（急转弯高速行驶）


## 融合定位架构

单一传感器定位均有不足，自动驾驶采用多传感器深度融合实现鲁棒定位：

```
GNSS/RTK ──────────────────────────┐
                                    │
 LiDAR 扫描匹配 ─────────────────── ┤  融合状态估计  ──► 6-DOF 全局位姿
                                    │  (EKF / 因子图)    + 不确定性协方差
 Camera 视觉里程计 ──────────────── ┤
                                    │
 IMU (100–1000 Hz) ────────────────┘
          │
          └──► 高频姿态预测（帧间外推）
```

**松耦合 vs 紧耦合：**

| 架构 | 描述 | 优势 | 劣势 |
| --- | --- | --- | --- |
| 松耦合（Loose Coupling） | 各传感器独立输出位姿后融合 | 实现简单，模块化 | 中间量损失信息 |
| 紧耦合（Tight Coupling） | 直接融合原始观测量（伪距、特征点） | 精度更高，能在 GNSS 退化时鲁棒 | 实现复杂 |

**扩展卡尔曼滤波（EKF）** 是最常用的实时融合框架，将非线性系统线性化后进行状态估计。大型系统中常采用**因子图优化**（GTSAM、Ceres Solver），支持多传感器增量批量优化。


## 定位评估指标与行业标准

### ATE / RPE 定义

#### 绝对轨迹误差（ATE，Absolute Trajectory Error）

ATE 衡量估计轨迹与真值轨迹的全局对齐程度。首先用 Umeyama 方法将估计轨迹与真值轨迹在 $SE(3)$ 上对齐，然后计算均方根误差：

$$\text{ATE} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \left\| \mathbf{p}_t^{\text{gt}} - \mathbf{p}_t^{\text{est}} \right\|^2}$$

其中 $\mathbf{p}_t^{\text{gt}}$ 为第 $t$ 帧真值位置，$\mathbf{p}_t^{\text{est}}$ 为估计位置（对齐后）。ATE 反映**全局漂移**大小。

#### 相对位姿误差（RPE，Relative Pose Error）

RPE 衡量固定时间窗口（或固定距离间隔）内的相对位姿估计误差，更能反映**局部精度**和漂移速率：

$$\text{RPE}(\Delta) = \frac{1}{T-\Delta} \sum_{t=1}^{T-\Delta} \left\| \left(\mathbf{T}_t^{\text{gt}}\right)^{-1} \mathbf{T}_{t+\Delta}^{\text{gt}} \ominus \left(\mathbf{T}_t^{\text{est}}\right)^{-1} \mathbf{T}_{t+\Delta}^{\text{est}} \right\|$$

其中 $\mathbf{T}_t \in SE(3)$ 为完整位姿（旋转+平移），$\ominus$ 表示李代数上的相对差，$\Delta$ 为固定时间步长（KITTI 通常取 100–800 m 距离间隔）。

### KITTI 定位榜单

KITTI（Karlsruhe Institute of Technology and Toyota Technological Institute）数据集是激光 SLAM 最权威的公开评测基准（11 条训练序列，11 条测试序列，总里程约 40 km）：

| 方法 | 平均平移误差（%） | 平均旋转误差（°/100m） | 传感器 |
| --- | --- | --- | --- |
| LOAM（2014） | 0.78 | 0.35 | LiDAR |
| LeGO-LOAM（2018） | 1.12 | 0.57 | LiDAR |
| LIO-SAM（2020） | 0.64 | 0.27 | LiDAR+IMU |
| MULLS（2021） | 0.54 | 0.23 | LiDAR |
| CT-ICP（2022） | 0.42 | 0.18 | LiDAR |
| ORB-SLAM3（2021） | 1.28 | 0.52 | 双目相机 |
| VINS-Mono（2018） | 0.91 | 0.41 | 单目+IMU |

### 行业量产定位精度要求

| 场景 / 功能 | 横向精度要求 | 纵向精度要求 | 可用性要求 |
| --- | --- | --- | --- |
| 高速 NOA（导航辅助驾驶） | < 30 cm | < 50 cm | > 99% |
| 城市 NOA（城市领航） | < 20 cm | < 30 cm | > 98%（开放道路） |
| L4 Robotaxi（限定区域） | **< 10 cm** | < 20 cm | > 99.9% |
| 自动泊车（APA/AVP） | < 5 cm | < 5 cm | > 99.5%（停车场） |
| 精准停靠（公交/地铁站） | < 3 cm | < 5 cm | > 99.9% |

**横向 10 cm 精度的工程含义：** 标准车道宽 3.5 m，车宽约 1.8 m，单侧余量约 0.85 m。10 cm 横向定位误差约占单侧余量的 12%，与控制误差（约 5 cm）叠加后仍有足够安全余量，是 L4 系统的最低可接受精度门槛。

### 综合定位评估指标

| 指标 | 全称 | 计算方式 | 量产典型要求 |
| --- | --- | --- | --- |
| ATE | 绝对轨迹误差 | $\sqrt{\frac{1}{T}\sum_t \|\mathbf{p}_t - \hat{\mathbf{p}}_t\|^2}$ | < 10 cm（L4 场景） |
| RPE | 相对位姿误差 | 相邻帧位姿变化误差均值 | < 0.1% 行驶距离 |
| 横向误差 | 车道中心偏离量 | $|d_t|$ 均值 | < 10 cm（L4），< 20 cm（L2+） |
| 定位成功率 | 满足精度要求比例 | 横向误差 < 阈值的帧比例 | > 99.9% |
| 可用性 | 系统正常定位的时间比例 | 正常运行时长 / 总时长 | > 99.95% |
| 初始化时间 | 首次定位收敛耗时 | 从上电到定位精度达标的时间 | < 30 s（城市开放场景） |


## 参考资料

1. J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time. RSS, 2014.
2. T. Qin, P. Li, and S. Shen. VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator. IEEE T-RO, 2018.
3. C. Cadena et al. Past, Present, and Future of Simultaneous Localization and Mapping: Toward the Robust-Perception Age. IEEE T-RO, 2016.
4. T. Shan et al. LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping. IROS, 2020.
5. C. Campos et al. ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multi-Map SLAM. IEEE T-RO, 2021.
6. 百度 Apollo. 自动驾驶高精地图与定位技术白皮书, 2022.
