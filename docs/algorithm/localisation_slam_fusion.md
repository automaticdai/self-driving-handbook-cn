# SLAM 与融合定位：方法、架构与工程实现

本页聚焦定位主算法链路，从地图构建与定位（SLAM）到多源融合的工程实现。

---

## 1. SLAM 问题定义

SLAM（Simultaneous Localization and Mapping）同时解决**定位**（我在哪？）和**建图**（环境是什么样？）两个相互依赖的问题。

**状态空间：**

$$\mathbf{x}_t = \{p_x, p_y, p_z, \phi, \theta, \psi\}^T \quad \text{（6-DOF 位姿）}$$

**贝叶斯框架：**

$$p(\mathbf{x}_t, \mathbf{m} \mid \mathbf{z}_{1:t}, \mathbf{u}_{1:t})$$

其中 $\mathbf{m}$ 为地图，$\mathbf{z}$ 为观测，$\mathbf{u}$ 为控制输入（IMU）。

---

## 2. 激光 SLAM

### 2.1 点云配准原理

激光 SLAM 的核心是点云配准——寻找两帧点云之间的变换矩阵 $T$，使重叠区域的点尽量重合。

**ICP（Iterative Closest Point）：**

$$T^* = \arg\min_T \sum_{i} \| p_i - T q_i \|^2$$

迭代步骤：
1. 为源点云 $Q$ 中每个点找最近邻 $P$ 中的点
2. 最小化点对距离求解变换 $T$
3. 应用变换，重复直到收敛

**NDT（Normal Distribution Transform）：**

将目标点云划分为体素格，每个格内建立高斯分布；用最大化目标点在分布中的概率来估计变换：

$$T^* = \arg\max_T \sum_i \exp\left(-\frac{(T p_i - \mu_k)^T \Sigma_k^{-1} (T p_i - \mu_k)}{2}\right)$$

NDT 对点云噪声更鲁棒，在自动驾驶定位中广泛使用。

### 2.2 主流激光 SLAM 系统

| 系统 | 特点 | 典型应用 |
| --- | --- | --- |
| LOAM | 边特征+面特征，实时性好 | 早期 L4 系统 |
| LeGO-LOAM | 轻量化 LOAM，地面分割 | 低功耗平台 |
| LIO-SAM | 紧耦合 LiDAR-IMU，因子图后端 | 城区复杂场景 |
| HDL Graph SLAM | 图优化，多闭环检测 | 建图 |

---

## 3. 视觉 SLAM

### 3.1 特征法视觉 SLAM（ORB-SLAM 系列）

流程：

```
图像帧
    │
    ├─ 1. 提取 ORB 特征点（角点 + 二进制描述子）
    ├─ 2. 特征匹配（Hamming 距离）
    ├─ 3. 估计相机运动（PnP 求解 + RANSAC）
    ├─ 4. 三角化新路标点
    ├─ 5. 局部 BA（Bundle Adjustment）优化
    └─ 6. 回环检测（DBoW2）+ 全局优化
```

**优缺点：**

| 优点 | 缺点 |
| --- | --- |
| 特征稳定，对光照变化有一定鲁棒性 | 纹理缺乏场景退化（白墙、夜间） |
| 绝对尺度可由立体视觉或 RGBD 获取 | 单目存在尺度不确定性 |
| 回环检测成熟 | 实时性受限于特征数量 |

### 3.2 直接法视觉 SLAM（DSO）

直接最小化光度误差（Photometric Error）：

$$E = \sum_i \sum_{j \in \text{obs}(i)} \left\| I_j(\pi(\mathbf{T}, d_i)) - I_{\text{ref}}(p_i) \right\|_\gamma$$

无需特征提取，对纹理弱场景更好，但对曝光变化和运动模糊敏感。

---

## 4. 视觉-惯性紧耦合（VIO）

### 4.1 IMU 预积分理论

IMU 输出角速度 $\omega$ 和加速度 $a$，积分得到姿态和速度：

$$\Delta R_{ij} = \prod_{k=i}^{j-1} \text{Exp}((\tilde{\omega}_k - b_g)\Delta t)$$

$$\Delta v_{ij} = \sum_{k=i}^{j-1} \Delta R_{ik} (\tilde{a}_k - b_a) \Delta t$$

$$\Delta p_{ij} = \sum_{k=i}^{j-1} \left[\Delta v_{ik}\Delta t + \frac{1}{2}\Delta R_{ik}(\tilde{a}_k - b_a)\Delta t^2\right]$$

预积分可以将 IMU 量重复利用，避免在优化迭代中重复积分。

### 4.2 主流 VIO 系统

| 系统 | 方法 | 特点 |
| --- | --- | --- |
| MSCKF | 多状态约束 EKF | 计算高效，适合实时 |
| VINS-Mono | 优化 + 滑动窗口 | 精度高，开源活跃 |
| ORB-SLAM3 | 特征法 + IMU 紧耦合 | 多传感器支持全 |
| OKVIS | 非线性优化，紧耦合 | 精度高，开源 |

---

## 5. 因子图融合架构

### 5.1 因子图基础

因子图将融合定位表述为最大后验（MAP）估计问题：

$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_i \| e_i(\mathbf{x}) \|_{\Sigma_i}^{-2}$$

图中包含：

- **变量节点**：待估计状态（位姿、速度、偏差）
- **因子节点**：对变量施加约束的量测（IMU、GPS、LiDAR 匹配）

**常用框架：**

| 框架 | 语言 | 特点 |
| --- | --- | --- |
| GTSAM | C++/Python | 功能完整，学术界主流 |
| g2o | C++ | 图优化，性能高 |
| Ceres | C++ | 通用非线性最小二乘 |

### 5.2 自动驾驶典型融合因子

```
因子图节点与因子示意：

x_0 ──IMU预积分因子── x_1 ──IMU预积分因子── x_2
 │                      │                      │
GNSS先验因子        LiDAR匹配因子         回环检测因子
                        │
                   地图先验约束
```

### 5.3 边缘化（Marginalization）

为保持实时性，滑动窗口内的旧状态通过边缘化移除，保留其约束对后续状态的影响（Prior Factor）。

---

## 6. GNSS/IMU 紧耦合

### 6.1 误差状态卡尔曼滤波器（ESKF）

状态向量：位置、速度、姿态、IMU 偏差（加速度计零偏 $b_a$、陀螺仪零偏 $b_g$）。

**预测步（IMU 积分）：**

$$\hat{\mathbf{x}}_{k|k-1} = f(\mathbf{x}_{k-1}, \mathbf{u}_k)$$

**更新步（GNSS 量测）：**

$$K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}$$

$$\mathbf{x}_k = \hat{\mathbf{x}}_{k|k-1} + K_k(\mathbf{z}_k - H\hat{\mathbf{x}}_{k|k-1})$$

---

## 7. 关键工程问题

### 7.1 时间同步

多传感器时间戳对齐是融合精度的基础约束：

| 传感器 | 典型同步方式 | 精度要求 |
| --- | --- | --- |
| 相机 | 硬件触发（FSYNC）+ 时戳记录 | < 1 ms |
| LiDAR | PPS 信号 + NMEA 句子同步 | < 0.1 ms |
| IMU | 硬件中断时戳 | < 0.1 ms |
| GNSS | PPS 秒脉冲标定 | < 1 μs |

时间同步偏差 > 10 ms 会显著影响 VIO 和激光-视觉联合定位精度。

### 7.2 外参标定

```
需要标定的外参关系：
  相机 ↔ LiDAR（旋转 + 平移，6-DOF）
  LiDAR ↔ IMU（旋转 + 平移 + 时间偏移）
  相机 ↔ 相机（多相机联合标定）
  IMU ↔ 车体坐标系

标定方式：
  出厂标定：棋盘格 + 靶标，工厂环境高精度
  在线标定：利用运动约束自动校正外参漂移
  地标校验：使用固定地标周期性验证标定质量
```

### 7.3 场景退化检测

某些场景下传感器无法提供有效约束：

| 退化场景 | 原因 | 处理方式 |
| --- | --- | --- |
| 长隧道 | GNSS 失效，视觉纹理单一 | 提高 IMU 权重，LiDAR 主导 |
| 暴风雪 | LiDAR 大量噪声点 | 点云滤波，降低 LiDAR 权重 |
| 玻璃幕墙 | LiDAR 穿透，视觉反射 | 点云噪声过滤，依赖地图先验 |
| 十字路口开阔区域 | 激光点云特征稀少 | 依赖 GNSS + 地图匹配 |

---

## 8. 定位输出接口建议

```yaml
localization_output:
  header:
    timestamp: 1234567890.123
    frame_id: "map"
  pose:
    position: {x, y, z}        # UTM 坐标系或地图坐标系
    orientation: {qw, qx, qy, qz}
  velocity:
    linear: {vx, vy, vz}
    angular: {wx, wy, wz}
  covariance: [6x6 矩阵]       # 位置+姿态协方差
  confidence: 0.95             # 0-1，综合置信度
  mode:
    status: "NORMAL"           # NORMAL/DEGRADED/RELOCATING/FAILED
    active_sensors: ["gnss", "lidar", "camera"]
    map_match_score: 0.87      # 地图匹配质量分数
```

---

## 9. 回环检测

回环检测识别车辆是否到达之前访问过的位置，用于消除长程漂移：

| 方法 | 原理 | 适用 |
| --- | --- | --- |
| DBoW2 | 视觉词袋，图像相似度检索 | 视觉 SLAM |
| Scan Context | 激光点云极坐标描述符 | 激光 SLAM |
| NetVLAD | 深度学习场景描述符 | 视觉，光照鲁棒 |
| Intensity ScanContext | 点云反射强度信息加入 | 提升区分性 |

回环检测后需要通过 RANSAC + PnP 验证几何一致性，避免误检测导致定位突变。
