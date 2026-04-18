# 自动驾驶术语表

本页汇总全书高频术语，涵盖自动化等级、系统架构、传感器、算法与安全标准等核心概念。建议在阅读系统、硬件、算法章节前先快速浏览。

---

## 阅读建议

先建立以下三条主线认知，再深入各章节：

```
安全主线：ODD → 降级 → TOR → MRC
数据流：感知 → 预测 → 规划 → 控制 → 执行
功能安全：危害分析 → ASIL 评级 → 安全机制 → 验证
```

---

## 1. 自动化等级

| 术语 | 全称 | 简要说明 |
| --- | --- | --- |
| **L0** | Level 0 | 无自动化，驾驶员全权控制 |
| **L1** | Level 1 | 单一功能辅助（如 ACC 或 LKA，不可同时）|
| **L2** | Level 2 | 组合功能辅助（ACC + LKA），驾驶员需持续监控 |
| **L3** | Level 3 | 特定条件下系统驾驶，驾驶员须在系统请求时接管（TOR）|
| **L4** | Level 4 | 特定 ODD 内完全自动，无需驾驶员（ODD 外仍需介入）|
| **L5** | Level 5 | 任何条件下完全自动，不需要驾驶员 |
| **ODD** | Operational Design Domain | 自动驾驶可安全运行的设计边界（道路类型、天气、速度等）|

---

## 2. 驾驶安全核心术语

| 术语 | 全称 | 简要说明 |
| --- | --- | --- |
| **TOR** | Takeover Request | 系统发出的接管请求，要求驾驶员在规定时间内恢复控制 |
| **MRC** | Minimum Risk Condition | 系统无法继续驾驶时执行的最小风险处置（减速靠边停车）|
| **MRM** | Minimum Risk Maneuver | 执行 MRC 的具体操作序列（减速→靠边→停车→上报）|
| **FTTI** | Fault Tolerant Time Interval | 从故障发生到安全机制响应的最大允许时间 |
| **SOTIF** | Safety Of The Intended Functionality | 功能本身的预期局限导致的危害（ISO 21448），区别于随机硬件故障 |

---

## 3. 功能安全（ISO 26262）

| 术语 | 含义 |
| --- | --- |
| **ASIL** | Automotive Safety Integrity Level，功能安全等级：A/B/C/D（D 最严格）|
| **ASIL-D** | 最高功能安全等级，用于线控转向、AEB 等关键系统 |
| **ASIL 分解** | 将 ASIL-D 需求分配到两个独立的 ASIL-B 路径，等效满足要求 |
| **HARA** | Hazard Analysis and Risk Assessment，危害分析与风险评估 |
| **Safety Goal** | 安全目标，描述系统不得违反的最高层安全需求 |
| **FTA** | Fault Tree Analysis，故障树分析 |
| **FMEA** | Failure Mode and Effects Analysis，失效模式与影响分析 |
| **Lockstep** | 锁步执行：两个 CPU 核运行同样指令并比较输出，检测随机错误 |
| **ECC** | Error Correction Code，内存纠错码，检测和纠正内存位翻转 |

---

## 4. 硬件与传感器

| 术语 | 全称 | 简要说明 |
| --- | --- | --- |
| **ECU** | Electronic Control Unit | 车载电子控制单元（小型专用控制器）|
| **DCU** | Domain Control Unit | 域控制器，管理某一功能域（智驾域、底盘域）|
| **SoC** | System on Chip | 片上系统，集成 CPU+GPU+NPU+ISP 的高算力芯片 |
| **MCU** | Microcontroller Unit | 微控制器，执行实时安全监控（安全岛）|
| **NPU** | Neural Processing Unit | 神经网络处理器，专用于 AI 推理加速 |
| **LiDAR** | Light Detection and Ranging | 激光雷达，通过激光脉冲测距，输出 3D 点云 |
| **GNSS** | Global Navigation Satellite System | 全球卫星导航系统（GPS/BDS/GLONASS/Galileo）|
| **RTK** | Real-Time Kinematic | 实时动态差分定位，厘米级定位精度 |
| **IMU** | Inertial Measurement Unit | 惯性测量单元，测量加速度和角速度 |
| **ISP** | Image Signal Processor | 图像信号处理器，负责 RAW 图像预处理 |
| **GMSL** | Gigabit Multimedia Serial Link | 车载高速串行摄像头传输接口（Maxim/ADI）|
| **ToF** | Time of Flight | 飞行时间测距原理（激光/超声波）|
| **FMCW** | Frequency Modulated Continuous Wave | 调频连续波，雷达测距与测速方法 |

---

## 5. 感知与定位

| 术语 | 含义 |
| --- | --- |
| **Point Cloud** | 点云，激光雷达输出的三维点集合，每个点含 XYZ 坐标和反射强度 |
| **BEV** | Bird's-Eye View，俯视鸟瞰视角，常用于多摄像头特征融合 |
| **Sensor Fusion** | 传感器融合，将多种传感器数据整合为统一环境模型 |
| **SLAM** | Simultaneous Localization and Mapping，即时定位与地图构建 |
| **VIO** | Visual-Inertial Odometry，视觉惯导里程计 |
| **LIO** | LiDAR-Inertial Odometry，激光惯导里程计 |
| **NDT** | Normal Distributions Transform，正态分布变换，点云匹配算法 |
| **ICP** | Iterative Closest Point，迭代最近点，点云配准算法 |
| **mAP** | mean Average Precision，目标检测的平均精度均值 |
| **IoU** | Intersection over Union，检测框与真值框的交并比 |
| **HD Map** | High Definition Map，高精地图，包含车道级别几何和语义信息 |
| **OpenDRIVE** | 一种通用的 HD Map 数据格式（XML 结构）|

---

## 6. 规划与控制

| 术语 | 含义 |
| --- | --- |
| **Trajectory** | 轨迹，包含时序位置与速度约束的运动序列 |
| **Frenet Frame** | 弗莱纳坐标系，沿参考路径的纵向（s）和横向（d）坐标系 |
| **MPC** | Model Predictive Control，模型预测控制，滚动优化策略 |
| **LQR** | Linear Quadratic Regulator，线性二次调节器，常用于横向控制 |
| **PID** | Proportional-Integral-Derivative，比例积分微分控制器 |
| **jerk** | 加加速度（加速度的变化率），乘坐舒适性关键指标（单位：m/s³）|
| **QP** | Quadratic Programming，二次规划，轨迹优化常用方法 |
| **FSM** | Finite State Machine，有限状态机，行为决策常用建模方法 |
| **Behavior Tree** | 行为树，比 FSM 更具模块化的行为规划结构 |
| **AEB** | Automatic Emergency Braking，自动紧急制动 |
| **ACC** | Adaptive Cruise Control，自适应巡航控制 |
| **LKA** | Lane Keeping Assist，车道保持辅助 |
| **TJA** | Traffic Jam Assist，拥堵辅助 |

---

## 7. 线控执行

| 术语 | 含义 |
| --- | --- |
| **EPS** | Electric Power Steering，电动助力转向 |
| **SbW** | Steer-by-Wire，线控转向（无机械连接）|
| **EHB** | Electro-Hydraulic Brake，电子液压制动 |
| **EMB** | Electro-Mechanical Brake，电子机械制动（去液压）|
| **ABS** | Anti-lock Braking System，防抱死制动系统 |
| **ESC** | Electronic Stability Control，电子稳定控制 |
| **EPB** | Electronic Parking Brake，电子驻车制动 |
| **ETC** | Electronic Throttle Control，电子节气门控制 |
| **CAN FD** | Controller Area Network Flexible Data-Rate，高速车载总线协议 |
| **AUTOSAR** | Automotive Open System Architecture，汽车开放系统架构标准 |

---

## 8. 通信与网络安全

| 术语 | 含义 |
| --- | --- |
| **V2X** | Vehicle-to-Everything，车辆与外界（车/路/人/网）的通信 |
| **V2V** | Vehicle-to-Vehicle，车对车直接通信 |
| **V2I** | Vehicle-to-Infrastructure，车对路侧基础设施通信 |
| **V2N** | Vehicle-to-Network，车对云网络通信 |
| **DSRC** | Dedicated Short Range Communications，专用短程通信（5.9 GHz）|
| **C-V2X** | Cellular V2X，基于蜂窝（LTE/5G）的 V2X 通信 |
| **RSU** | Roadside Unit，路侧单元（通信 + 边缘计算设备）|
| **SPAT** | Signal Phase and Timing，信号相位与剩余时间消息 |
| **PKI** | Public Key Infrastructure，公钥基础设施（证书信任体系）|
| **SCMS** | Security Credential Management System，安全证书管理系统 |
| **OTA** | Over-the-Air，空中升级（远程软件更新）|
| **HSM** | Hardware Security Module，硬件安全模块（密钥存储与加密）|
| **ISO/SAE 21434** | 汽车网络安全工程标准 |

---

## 9. 测试与验证

| 术语 | 含义 |
| --- | --- |
| **SIL** | Software in the Loop，软件在环测试（纯软件仿真）|
| **HIL** | Hardware in the Loop，硬件在环测试（真实 ECU + 仿真环境）|
| **MIL** | Model in the Loop，模型在环测试（控制模型仿真）|
| **ATE** | Absolute Trajectory Error，绝对轨迹误差（定位评估指标）|
| **RPE** | Relative Pose Error，相对位姿误差（定位评估指标）|
| **MTBF** | Mean Time Between Failures，平均故障间隔时间 |
| **MTTR** | Mean Time To Recovery，平均恢复时间 |
| **AEC-Q100** | 汽车级芯片可靠性测试标准（温循、振动、ESD 等）|
| **NCAP** | New Car Assessment Programme，新车评价规程 |
| **DTC** | Diagnostic Trouble Code，诊断故障码 |

---

## 10. 商业运营

| 术语 | 含义 |
| --- | --- |
| **Robotaxi** | 无人驾驶出租车（L4 自动驾驶商业运营）|
| **ROC** | Remote Operations Center，远程运营中心 |
| **RAS** | Remote Assistance System，远程协助系统 |
| **MPKD** | Miles Per Kilometre Disengagement / 每千公里干预次数 |
| **NPS** | Net Promoter Score，净推荐值（用户满意度指标）|
| **Geofencing** | 地理围栏，限定自动驾驶运营区域的虚拟边界 |

---

## 11. 端到端与大模型

| 术语 | 全称 | 简要说明 |
| --- | --- | --- |
| **E2E** | End-to-End | 端到端驾驶，从传感器原始输入直接输出轨迹或控制指令 |
| **Modular E2E** | — | 模块化端到端：保留感知/预测/规划子网络，但联合训练 |
| **One-Model E2E** | — | 单模型端到端：一个大网络直接产出轨迹，无显式中间表示 |
| **Occupancy Network** | — | 占据网络，预测三维空间每个体素的占据概率与语义（Tesla/华为）|
| **World Model** | — | 世界模型，学习环境动态并在潜空间中推演未来（GAIA、DriveDreamer）|
| **VLM** | Vision-Language Model | 视觉语言模型，联合理解图像与文本（用于场景解释、决策）|
| **VLA** | Vision-Language-Action | 视觉-语言-动作模型，输出可执行动作（驾驶控制/机器人）|
| **LLM** | Large Language Model | 大语言模型，驱动推理与自然语言交互 |
| **MoE** | Mixture of Experts | 专家混合，稀疏激活以降低推理成本 |
| **CoT** | Chain of Thought | 思维链，显式输出中间推理步骤 |
| **RAG** | Retrieval-Augmented Generation | 检索增强生成，借外部知识库补足模型参数 |
| **SFT** | Supervised Fine-Tuning | 监督微调 |
| **RLHF** | Reinforcement Learning from Human Feedback | 基于人类反馈的强化学习 |
| **Diffusion Policy** | — | 扩散策略，用扩散模型采样多模态轨迹 |
| **Imitation Learning** | — | 模仿学习，从专家轨迹学习驾驶策略 |
| **Token** | — | 大模型处理的最小语义单元（文字/图像 patch）|

---

## 12. 数据与训练

| 术语 | 含义 |
| --- | --- |
| **Data Pipeline** | 数据闭环：采集 → 脱敏 → 标注 → 训练 → 评测 → 回灌 |
| **Auto-Labeling** | 自动标注，用大模型/离线高精算法生成训练标签 |
| **Corner Case** | 长尾场景，低频但高风险的驾驶情况（施工、异形障碍物）|
| **Shadow Mode** | 影子模式，在量产车后台运行新模型并对比人类驾驶以挖掘 corner case |
| **Active Learning** | 主动学习，优先标注模型不确定的样本 |
| **Domain Gap** | 领域差异，训练与部署分布不一致导致的性能下降 |
| **Data Augmentation** | 数据增广（旋转、遮挡、光照等）|
| **Scenario Mining** | 场景挖掘，从量产车日志中筛选目标场景 |
| **HIL Replay / Log Replay** | 日志回灌，把真实采集数据注入仿真或新算法重放评估 |
| **Ground Truth** | 真值，由高精算法/人工标注得到的参考答案 |

---

## 13. 仿真与评测

| 术语 | 含义 |
| --- | --- |
| **Sim-to-Real** | 仿真到实车的迁移问题（sim2real gap）|
| **Digital Twin** | 数字孪生，对真实物理世界的高保真数字复刻 |
| **Scenario DSL** | 场景领域特定语言（如 OpenSCENARIO、Scenic）|
| **OpenSCENARIO** | ASAM 发布的场景描述标准（XML/XOSC）|
| **OpenDRIVE** | ASAM 发布的道路描述标准（XODR）|
| **NDS** | Navigation Data Standard，导航地图数据标准 |
| **CARLA / LGSVL / VTD** | 常见开源/商用自动驾驶仿真器 |
| **PEM** | Perception Error Model，感知误差模型（在仿真中注入感知噪声）|
| **Collision Rate** | 碰撞率，安全评测核心指标 |
| **Route Completion** | 路线完成度，CARLA Leaderboard 常用指标 |
| **nuScenes / Waymo Open / Argoverse** | 常见感知与预测公开数据集 |
| **nuPlan / Bench2Drive** | 常见规划与端到端基准 |

---

## 14. 芯片与部署

| 术语 | 含义 |
| --- | --- |
| **TOPS** | Tera Operations Per Second，每秒万亿次运算（NPU 算力单位，注意精度）|
| **INT8 / FP16 / BF16** | 推理常用数值格式；低精度减小带宽与算力开销 |
| **Quantization** | 量化，将 FP32 权重压缩到 INT8/INT4 |
| **PTQ / QAT** | 训练后量化 / 量化感知训练 |
| **Sparsity** | 稀疏化，剪枝零权重以加速推理 |
| **Knowledge Distillation** | 知识蒸馏，用大模型教小模型 |
| **TensorRT / OpenVINO / TVM** | 常见推理优化工具链 |
| **Safety Island** | 安全岛，SoC 内独立的 ASIL-D 监控核（锁步 MCU）|
| **Hypervisor** | 虚拟化管理程序，实现智驾域/座舱域隔离 |
| **Zonal Architecture** | 区域控制架构，按位置划分 ECU（取代传统功能域布线）|
