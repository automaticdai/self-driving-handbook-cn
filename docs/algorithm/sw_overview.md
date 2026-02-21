# 系统软件框架总览

![软件系统架构图](assets/1565719380310_2.png)

自动驾驶软件栈是一个高度复杂的实时系统，需要将传感器输入在 100–200 ms 内转化为车辆控制指令，同时保证功能安全和系统可靠性。整个系统可以分为五大核心模块，形成**感知—预测—规划—控制**的完整闭环。


## 整体架构

```
传感器原始数据（Camera / LiDAR / Radar / IMU / GNSS）
                         │
                    时间同步 & 外参标定
                         │
              ┌──────────▼──────────┐
              │    感知（Perception）│
              │  目标检测 │ 语义分割 │
              │  目标跟踪 │ 车道检测 │
              └──────────┬──────────┘
                         │ Object List / Occupancy
              ┌──────────▼──────────┐
              │    预测（Prediction）│  ◄── 定位（Localization）
              │  行为意图 │ 轨迹预测 │       HD 地图（HD Map）
              └──────────┬──────────┘
                         │ Predicted Trajectories
              ┌──────────▼──────────┐
              │    规划（Planning）  │  ◄── 路由（Routing）
              │  行为决策 │ 运动规划 │
              └──────────┬──────────┘
                         │ Reference Trajectory
              ┌──────────▼──────────┐
              │    控制（Control）   │
              │  横向控制 │ 纵向控制 │
              └──────────┬──────────┘
                         │ 转向角 / 油门 / 制动指令
                    线控系统（By-Wire）
                         │
                    车辆物理执行
```


## 五大核心 ADAS 功能模块

ADAS（Advanced Driver Assistance Systems，高级驾驶辅助系统）是自动驾驶的基础，也是量产 L1/L2 级系统的核心：

| 功能 | 英文全称 | 缩写 | 核心描述 | 典型级别 |
| --- | --- | --- | --- | --- |
| 自适应巡航控制 | Adaptive Cruise Control | ACC | 自动保持与前车安全距离，纵向速度控制 | L1 |
| 自动紧急制动 | Autonomous Emergency Braking | AEB | 检测前方碰撞风险，自动施加制动力 | L1 |
| 辅助车道保持 | Lane Keeping Assist | LKA | 检测车道线，偏离时施加横向纠正 | L1 |
| 行人保护系统 | Pedestrian Protection System | PPS | 检测行人、骑行者，触发 AEB 或警告 | L1 |
| 交通标志识别 | Traffic Sign Recognition | TSR | 识别限速、禁令等交通标志并提示 | L2 |
| 车道变换辅助 | Lane Change Assist | LCA | 辅助/自动执行变道动作 | L2 |
| 自动泊车辅助 | Automated Parking Assist | APA | 自动控制转向完成泊车，驾驶员控制油门制动 | L2 |


## 中间件框架

中间件负责模块间的消息传递、任务调度和硬件抽象，是软件架构的"神经系统"：

### ROS 2（Robot Operating System 2）
- 基于 **DDS（Data Distribution Service）** 的发布-订阅通信，支持 QoS 配置
- 核心概念：节点（Node）、话题（Topic）、服务（Service）、动作（Action）
- 优势：活跃开源社区，丰富的工具链（RViz、rosbag、ros2 launch）
- 主要用途：学术原型、Autoware 等开源方案
- 劣势：实时性不及专用中间件，大型系统通信开销大

### Cyber RT（百度 Apollo 自研）
- 专为自动驾驶设计，基于 **协程（Coroutine）** 调度，降低线程切换开销
- **组件（Component）化**：各模块独立编译、热插拔，支持不同团队并行开发
- 时间触发（Timed）与数据触发（Data-driven）混合模型
- 性能较 ROS 有显著提升，适合高频传感器融合

### AUTOSAR Adaptive（自适应 AUTOSAR）
- 汽车级标准中间件（ISO 17458），面向服务的通信架构（SOA）
- 使用 **SOME/IP** 协议实现服务发现与调用，兼容车载以太网（100BASE-T1 / 1000BASE-T1）
- 支持 OTA 功能更新，POSIX 操作系统接口
- 被各大 Tier1 和主机厂广泛用于量产 L2+/L3 车型
- 典型实现：Vector、ETAS、Elektrobit、华为（GoAhead）


## 数据流水线

```
原始传感器数据
    │
    ├─ Camera（30–60 Hz）────► 预处理（去畸变、曝光补偿）→ 深度神经网络推理
    ├─ LiDAR（10–20 Hz）────► 地面去除 → 体素化/降采样 → 3D 检测/配准
    ├─ Radar（10–20 Hz）────► CFAR 检测 → 目标提取 → 追踪滤波
    └─ IMU（100–1000 Hz）───► 预积分 → 与 GNSS 融合定位（EKF）
              │
              ▼
      时间同步（硬件时钟，精度 < 1 ms）
      空间对齐（传感器外参标定）
              │
              ▼
    多模态感知融合（特征级 or 目标级）
              │
              ▼
    环境模型（World Model）：动态目标 + 静态地图 + 自车位姿
              │
              ▼
    预测模块：障碍物意图分类 + 未来轨迹分布（3–5 s）
              │
              ▼
    决策规划：行为选择（变道/跟车）→ 轨迹优化（约束求解）
              │
              ▼
    控制模块：LQR/MPC 横向控制 + PID 纵向控制
              │
              ▼
    线控执行：CAN/以太网指令发送 → 执行器响应（< 10 ms）
```


## 系统性能要求

端到端延迟对自动驾驶安全至关重要。以 100 km/h 速度行驶时，200 ms 延迟对应约 5.6 m 的前进距离。

| 模块 | 典型处理频率 | 端到端延迟要求 | 关键瓶颈 |
| --- | --- | --- | --- |
| 传感器采集 | 10–60 Hz | — | 接口带宽（Camera 4K 需 > 10 Gbps） |
| 感知（检测+追踪） | 10–25 Hz | < 50 ms | 神经网络推理，GPU 利用率 |
| 定位更新（IMU 融合） | 100 Hz | < 10 ms | 卡尔曼滤波实时性 |
| 预测 | 10 Hz | < 50 ms | 多模态轨迹生成 |
| 规划（轨迹优化） | 10 Hz | < 100 ms | 优化求解器（QP/NLP） |
| 控制 | 100 Hz | < 10 ms | 实时操作系统调度 |
| **端到端总延迟** | — | **< 200 ms** | 需全链路时延预算管理 |


## 主流开源软件栈

| 软件栈 | 维护方 | 语言 | 中间件 | 特点 |
| --- | --- | --- | --- | --- |
| Autoware.Universe | Autoware Foundation | C++ | ROS 2 | 模块化全栈，适合 L4 研究 |
| Apollo | 百度 | C++ / Python | Cyber RT | 完整工具链 + 云服务，生产级 |
| Openpilot | Comma.ai | Python / C++ | — | 量产车 OBD 改造，纯视觉 |
| CARMA Platform | FHWA（美联邦公路局） | C++ | ROS 2 | V2X 协同驾驶专注 |


## 参考资料

1. S. Paden et al. A Survey of Motion Planning and Control Techniques Talked About in the Meeting Room. IEEE T-ITS, 2016.
2. Autoware Foundation. Autoware Architecture Design Document, 2023.
3. 百度 Apollo. Cyber RT Technical Reference Manual, 2019.
4. AUTOSAR Consortium. AUTOSAR Adaptive Platform Specification, 2022.
