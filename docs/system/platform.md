# 仿真平台与数据集

仿真平台和公开数据集是自动驾驶开发的"虚拟试验场"，能够大幅降低测试成本和安全风险。由于真实路测成本极高、危险边缘场景（Corner Cases）难以在现实中复现，仿真系统可以在虚拟环境中高效验证感知、规划和控制算法，加速开发迭代。业界普遍认为自动驾驶系统需要数百亿公里的验证里程，仿真是实现这一目标的唯一可行途径。

公开数据集则为算法训练和基准测试提供了标准化的评测环境，推动了学术界和工业界的技术进步。两者相辅相成：仿真平台负责生成合成数据和闭环测试，公开数据集提供真实世界的分布参考，共同构成自动驾驶技术验证的完整生态。


## 仿真平台详解

### CARLA（开源，基于UE4）

[CARLA](http://carla.org/)（Car Learning to Act）是当前学术界最广泛使用的开源自动驾驶仿真器，由巴塞罗那自治大学与英特尔实验室联合开发，基于虚幻引擎4（Unreal Engine 4）构建，提供高质量的视觉效果和物理仿真。

**传感器仿真能力：**

- **摄像头**：RGB、深度（Depth）、语义分割（Semantic Segmentation）、实例分割
- **LiDAR**：可配置线数、旋转频率、水平/垂直分辨率，支持Ray-Cast模型
- **雷达（Radar）**：毫米波雷达，输出点云和速度信息
- **GPS / IMU**：提供位置和惯性测量数据，支持误差模型注入
- **车道线检测传感器**、**语义LiDAR**、**碰撞/车道侵入传感器**

**Python API接口：**

CARLA提供完整的Python API，用户可通过脚本化方式控制仿真世界：

```python
import carla

client = carla.Client('localhost', 2000)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# 设置天气
weather = carla.WeatherParameters(
    cloudiness=80.0,
    precipitation=20.0,
    sun_altitude_angle=30.0
)
world.set_weather(weather)
```

**地图与场景：**

CARLA内置Town01至Town12共12张地图，涵盖城市街道、高速公路、乡村道路等多种场景。地图采用OpenDRIVE格式描述，支持自定义地图导入。

**场景运行器（Scenario Runner）：**

与CARLA配套的Scenario Runner工具基于OpenSCENARIO标准，支持结构化场景描述和自动执行，是CARLA Challenge（CVPR自动驾驶挑战赛）的核心测试框架。

**适合场景：**
- 学术算法验证（感知、规划、控制）
- 强化学习（Reinforcement Learning）智能体训练
- 端到端（End-to-End）自动驾驶研究
- 边缘场景（Corner Case）数据合成

---

### LGSVL / SVL Simulator

[LGSVL Simulator](https://www.lgsvlsimulator.com/)由LG电子硅谷研究院开发，基于Unity游戏引擎，以与Apollo和Autoware的深度集成著称。

**核心特性：**

- **引擎**：Unity，支持跨平台部署
- **接口**：原生支持ROS/ROS2和百度Cyber RT，可直接对接Apollo/Autoware软件栈
- **传感器**：可通过配置文件灵活定义传感器套件（摄像头、LiDAR、雷达、GPS、IMU）
- **真值（Ground Truth）传感器**：提供3D检测框、语义地图等理想传感器输出

**现状：**

LG电子于2022年宣布停止对SVL Simulator的官方支持，但由于其与Apollo/Autoware生态的良好兼容性，仍在工业界和学术界被广泛使用。社区维护的分支版本持续提供更新。

---

### AirSim（微软，基于UE4）

[AirSim](https://github.com/microsoft/AirSim)（Aerial Informatics and Robotics Simulation）由微软研究院开发，基于虚幻引擎4，支持无人机和地面车辆双模式仿真。

**核心特性：**

- **双模式**：无人机（Multirotor）和汽车（Car）模式，适合多场景研究
- **物理引擎**：基于PhysX，物理建模精确，包括空气动力学和地面接触模型
- **传感器**：摄像头（多视角、深度、分割）、IMU、气压计、磁力计
- **合成数据生成**：天气随机化、光照变化、对象随机放置，支持域随机化

**现状：**

微软于2022年将AirSim开源社区版移交给外部维护（Colosseum分支），官方Microsoft AirSim版本已停止更新。

---

### MATLAB / Simulink

MathWorks的MATLAB/Simulink工具链是控制工程和ADAS算法开发的行业标准环境。

**Automated Driving Toolbox：**

- 提供车道检测、目标检测、跟踪算法的参考实现
- 支持传感器融合（摄像头+LiDAR+雷达）仿真
- 内置Driving Scenario Designer，可图形化设计测试场景
- 代码自动生成（Code Generation），可直接部署到嵌入式平台

**HIL集成：**

MATLAB/Simulink与硬件在环（Hardware-in-the-Loop，HIL）系统深度集成，主要合作伙伴包括：
- **dSPACE**：SCALEXIO、MicroAutoBox HIL平台
- **ETAS**：LABCAR HIL系统
- **NI**：CompactRIO HIL环境

---

### NVIDIA DRIVE Sim（基于Omniverse）

[NVIDIA DRIVE Sim](https://developer.nvidia.com/drive/simulation)基于NVIDIA Omniverse平台，提供工业级传感器物理仿真，是商业自动驾驶开发的主流选择之一。

**核心能力：**

- **光线追踪（Ray Tracing）传感器仿真**：基于物理的摄像头、LiDAR、雷达仿真，精确模拟光学特性
- **大规模并行仿真**：支持在数千GPU上同时运行仿真实例，实现加速测试
- **合成数据生成（Synthetic Data Generation）**：域随机化，自动生成带标注的训练数据
- **数字孪生（Digital Twin）**：与真实地图数据（HD Map）融合，重建真实道路环境

---

### 内部仿真平台

业界头部公司普遍构建了面向自身需求的专有仿真系统：

- **Waymo Carcraft**：基于真实采集数据重建的仿真环境，每年可运行超过100亿公里的等效模拟里程
- **百度 Apollo仿真**：与Apollo开放平台配套，支持云端大规模仿真和场景库管理
- **特斯拉 Dojo仿真**：基于Shadow Mode采集的真实数据，闭环训练和验证FSD神经网络

---

### 仿真平台对比

| 平台 | 开源/商业 | 物理精度 | 传感器类型 | 接口 | 典型用途 |
|------|-----------|----------|-----------|------|----------|
| CARLA | 开源 | 中高 | 摄像头/LiDAR/雷达/GPS/IMU | Python API / ROS | 学术研究，RL训练 |
| LGSVL/SVL | 开源（停更） | 中 | 摄像头/LiDAR/雷达/GPS | ROS / Cyber RT | Apollo/Autoware集成测试 |
| AirSim | 开源（停更） | 高 | 摄像头/IMU/气压计 | Python / ROS | 无人机+车辆，合成数据 |
| MATLAB/Simulink | 商业 | 中 | 摄像头/LiDAR/雷达 | MATLAB API / Code Gen | 控制算法，HIL集成 |
| NVIDIA DRIVE Sim | 商业 | 极高 | 摄像头/LiDAR/雷达 | Omniverse API | 工业级传感器仿真 |
| PreScan (Siemens) | 商业 | 高 | 摄像头/LiDAR/雷达/V2X | MATLAB / FMI | ADAS测试，V2X |
| SUMO | 开源 | 低（交通流） | 无 | TraCI / libsumo | 宏观交通流仿真 |


## 公开数据集

### KITTI

[KITTI数据集](https://www.cvlibs.net/datasets/kitti/)由德国卡尔斯鲁厄理工学院（KIT）与丰田美国技术研究院联合发布（2012年），是自动驾驶领域最具影响力的早期基准数据集之一。

**采集平台：**
- **传感器**：Velodyne HDL-64E LiDAR + 2对立体摄像头（灰度/彩色）+ GPS/IMU
- **采集地点**：德国卡尔斯鲁厄市区及高速公路

**规模与标注：**
- 原始数据：约400GB，覆盖多种天气和时段
- 训练序列：21个，测试序列：29个（无标签）
- 标注：7481帧3D标注（车辆、行人、自行车）

**基准任务：**
- 3D目标检测（3D Object Detection）
- 目标追踪（Multi-Object Tracking）
- 光流估计（Optical Flow）
- 深度估计（Depth Estimation）
- 视觉里程计（Visual Odometry）

---

### nuScenes

[nuScenes](https://www.nuscenes.org/)由Motional（原nuTonomy）发布（2020年），是当前全球使用最广泛的自动驾驶数据集之一，具有完整的全向传感器套件。

**采集平台：**
- **传感器**：6个摄像头（360°覆盖）+ 1个32线LiDAR + 5个毫米波雷达 + GPS/IMU
- **采集地点**：波士顿和新加坡

**规模与标注：**
- 1000个场景（每段约20秒）
- 1400万个3D标注框（23类目标）
- 关键帧标注频率：2Hz，原始数据：20Hz

**基准任务：**
- 3D目标检测（nuScenes Detection Score，NDS）
- 多目标追踪（AMOTA指标）
- 运动预测（Trajectory Prediction）
- 地图分割（Map Segmentation）

---

### Waymo Open Dataset

[Waymo Open Dataset](https://waymo.com/open/)由Waymo于2019年发布，以极高的标注质量和传感器配置见长。

**采集平台：**
- **传感器**：5个LiDAR（含顶部激光雷达）+ 5个摄像头（前向+侧向）
- **采集地点**：凤凰城、旧金山、山景城等美国城市

**规模与标注：**
- 1000个场景（感知子集）
- 2D/3D精细标注，4类目标（车辆、行人、自行车、摩托车）
- Waymo Motion Dataset：超过10万个场景，用于运动预测

**基准任务：**
- 3D目标检测
- 2D目标检测
- 领域适应（Domain Adaptation）
- 运动预测（Waymo Motion Prediction Challenge）

---

### Argoverse 2

[Argoverse 2](https://www.argoverse.org/av2.html)由Argo AI（现已解散，数据集持续开放）发布，以高精地图集成和运动预测任务闻名。

**核心特性：**
- **传感器**：7个摄像头（环视）+ 2个LiDAR
- **高精地图（HD Map）**：与场景数据强绑定，提供车道级地图标注
- **运动预测数据集**：25万个场景片段，聚焦长尾驾驶行为
- **3D目标检测数据集**：1000个场景，26类目标

---

### ONCE（中国，一汽集团）

[ONCE数据集](https://once-for-auto-driving.github.io/)（One millioN sCEnes）由一汽集团发布，是国内规模最大的自动驾驶感知数据集之一。

**核心特性：**
- **规模**：100万帧，约15000个采集片段
- **传感器**：7个摄像头（前向+环视）+ 1个LiDAR
- **标注**：100万帧LiDAR标注，覆盖车辆、行人、自行车、交通锥等
- **场景多样性**：中国城市、郊区、高速公路，包含多种天气和时段
- **特色**：包含大量中国特有交通参与者（电动自行车、三轮车）

---

### OpenDV / DriveX

OpenDV及类似的大规模互联网视频数据集代表了数据驱动自动驾驶的新方向。

**核心理念：**
- 从互联网（YouTube、行车记录仪视频平台）大规模爬取驾驶视频
- 无需精细3D标注，用于端到端（End-to-End）驾驶模型预训练
- 规模可达数万小时，远超传统数据集

---

### 数据集对比

| 数据集 | 规模 | 传感器 | 标注类型 | 主要任务 | 许可 |
|--------|------|--------|----------|----------|------|
| KITTI | ~15K帧 | 摄像头/LiDAR/GPS | 3D框 | 检测/追踪/深度/里程计 | CC BY-NC-SA |
| nuScenes | 1000场景 | 6摄/LiDAR/5雷达 | 3D框+属性 | 检测/追踪/预测 | CC BY-NC-SA |
| Waymo Open | 1000场景 | 5摄/5LiDAR | 2D+3D框 | 检测/追踪/预测 | Waymo专有 |
| Argoverse 2 | 250K场景 | 7摄/2LiDAR | 3D框+地图 | 检测/运动预测 | CC BY-NC-SA |
| ONCE | 1M帧 | 7摄/LiDAR | 3D框 | 检测 | 学术使用 |
| OpenDV | 数万小时视频 | 摄像头 | 无/弱标注 | E2E预训练 | 混合 |


## 场景描述语言

### OpenSCENARIO

[OpenSCENARIO](https://www.asam.net/standards/detail/openscenario/)是ASAM（汽车仿真和测量系统协会）制定的场景描述标准，采用XML格式描述驾驶场景的动态行为。

**核心概念：**
- **故事板（Storyboard）**：场景的顶层结构，包含Init初始化和Story故事
- **演员（Actor）**：场景中的参与者（自车、NPC车辆、行人）
- **事件（Event）**：基于触发条件（距离、时间、速度）激活的动作
- **动作（Action）**：速度变化、车道变换、路点跟踪等驾驶行为

**应用：**
- CARLA Scenario Runner
- openPASS仿真框架
- dSPACE/ETAS HIL测试系统

---

### OpenDRIVE

[OpenDRIVE](https://www.asam.net/standards/detail/opendrive/)是ASAM制定的道路网络描述标准，定义了高精地图的路网拓扑结构。

**描述内容：**
- **车道（Lane）**：车道类型、宽度、连接关系
- **道路拓扑（Road Topology）**：路段、交叉口、匝道
- **交通信号（Traffic Signs）**：信号灯、限速标志、停止线
- **表面特性**：路面摩擦系数、超高（Superelevation）

CARLA的所有内置地图均以OpenDRIVE格式存储，可导出用于其他工具。

---

### Scenic

[Scenic](https://scenic-lang.org/)是加州大学伯克利分校开发的概率场景描述语言，采用类Python语法，专门用于生成长尾场景。

**核心特性：**
- **概率分布**：可将场景参数定义为概率分布，自动采样生成多样化场景
- **空间关系**：提供`ahead of`、`behind`、`left of`等直觉性空间描述符
- **与仿真器集成**：支持CARLA、GTA V、Webots等多种仿真环境
- **反事实场景**：通过蒙特卡洛采样生成罕见但关键的测试场景

```python
# Scenic示例：前方有一辆慢速卡车
ego = Car
truck = Car ahead of ego by (10, 30),
        with speed uniform(0, 5)  # 随机速度0-5 m/s
require (distance to truck) < 20
```

---

### 场景库管理

规模化仿真需要系统化管理场景库（Scenario Database）：

- **参数化场景（Parameterized Scenarios）**：定义场景模板，通过参数空间采样生成变体
- **场景分类**：按ODD（Operational Design Domain）、危险类型、天气条件分类存储
- **场景演化**：基于真实路测数据中的异常事件，自动提取并扩充场景库
- **覆盖率追踪**：记录哪些场景类别已被充分测试，指导新场景生成优先级


## 闭环评估（Closed-Loop Evaluation）

### 开环 vs 闭环的本质区别

| 评估方式 | 描述 | 优点 | 缺点 |
|----------|------|------|------|
| 开环（Open-Loop） | 在离线日志数据上回放，比较预测输出与真值标注 | 成本低，可重复 | 忽略因果反馈，存在分布偏移 |
| 闭环（Closed-Loop） | 在仿真中实时执行，系统动作影响后续环境状态 | 真实反映系统能力 | 需要高质量仿真器，成本高 |

**分布偏移（Distribution Shift）问题：**

开环评估存在根本性缺陷：被测系统的输出不会影响场景演化。一个规划器即使在关键时刻给出了错误决策，开环日志依然会继续回放"专家"行为，掩盖了系统的真实失误。这一现象称为协变量偏移（Covariate Shift），导致开环指标（如L2位移误差）与真实驾驶性能相关性较低。

---

### nuPlan闭环规划基准

[nuPlan](https://nuplan.org/)是Motional发布的首个大规模闭环规划基准，专门针对规划器的闭环评估。

**特性：**
- 1500小时真实驾驶日志（来自美国、新加坡、匹兹堡等城市）
- 基于日志回放的反应式仿真（Reactive Simulation）
- **评估指标**：无碰撞率、舒适性、进度完成率、交通规则遵从率
- 支持基于规则（IDM）、基于学习（ML Planner）等多种规划器接入

---

### Waymo WOMD运动预测挑战

[Waymo Open Motion Dataset（WOMD）](https://waymo.com/open/data/motion/)挑战赛聚焦多智能体运动预测，是运动预测领域最权威的竞赛之一。

- 超过10万个场景，覆盖复杂交互行为
- 评估指标：minADE（最小平均位移误差）、minFDE（最终位移误差）、Miss Rate
- 每年举办，推动了Transformer-based预测模型的快速发展

---

### PDM评估框架

PDM（Privileged Driver Model）是一种基于特权信息（Privileged Information）的闭环评估框架。

- **核心思想**：用访问完整场景状态（如其他智能体真实轨迹）的特权规划器作为上界参考
- 将被测系统的闭环性能与特权规划器对比，量化"与最优解的差距"
- 已被多个学术工作（如UniAD、VAD）用于验证端到端规划性能


## 合成数据生成（Synthetic Data）

### 域随机化（Domain Randomization）

域随机化是提升仿真数据多样性、减小"仿真到真实（Sim-to-Real）"差距的核心技术：

- **纹理随机化**：随机替换路面、建筑、车辆外表面的纹理贴图
- **光照随机化**：随机改变太阳方向角、强度、色温，模拟不同时段和天气
- **天气随机化**：雨量、雾浓度、雪深、镜头光晕等参数化控制
- **对象放置随机化**：障碍物、行人、临时标识的位置和朝向随机化
- **传感器参数随机化**：摄像头曝光、白平衡、LiDAR噪声模型随机化

---

### NeRF场景重建

神经辐射场（Neural Radiance Field，NeRF）提供了从真实采集数据到可重渲染虚拟场景的新范式：

1. **采集**：在真实道路上采集多视角摄像头和LiDAR数据
2. **重建**：用NeRF或3D Gaussian Splatting拟合场景的隐式表示
3. **合成**：在重建场景中修改天气、时段、障碍物，生成带标注的训练数据
4. **优势**：合成数据与真实世界高度一致，减小域差距

代表工作：Street Surfer、EmerNeRF、UniSim、OmniMatting。

---

### 扩散模型驾驶视频生成

基于扩散模型（Diffusion Model）的生成方法可以直接生成逼真的驾驶场景视频：

- **UniSim（NVIDIA）**：基于神经闭环传感器仿真，通过扩散模型合成多摄像头、任意轨迹下的驾驶视频
- **WoVogen**：World Volume-aware Video Generation，保持3D一致性的驾驶视频生成
- **DriveDreamer / DriveX**：基于文本/条件控制的驾驶场景生成，支持罕见场景合成

---

### 合成数据配方

实践中，合成数据与真实数据的混合比例（Recipe）对模型性能至关重要：

| 训练阶段 | 真实数据比例 | 合成数据比例 | 典型用途 |
|----------|-------------|-------------|----------|
| 预训练 | 30%–50% | 50%–70% | 建立基础感知能力 |
| 微调 | 70%–90% | 10%–30% | 适配真实数据分布 |
| 长尾增强 | 50% | 50%（针对长尾） | 提升边缘场景性能 |

合成数据在检测类任务中通常可提升2–5%的AP，对长尾类别（如施工区域、异型障碍物）提升更为显著。


## 测试验证框架

### 基于场景的测试（Scenario-Based Testing）

基于场景的测试（Scenario-Based Testing，SBT）已成为自动驾驶功能安全验证的主流方法论，取代了传统的基于里程的测试方式。

**核心流程：**

1. **场景空间定义**：基于ODD（Operational Design Domain）划定测试范围
2. **场景参数化**：将具体场景抽象为参数空间（如车速、跟车距离、能见度）
3. **场景采样**：系统化采样测试场景，确保参数空间覆盖
4. **自动执行**：批量在仿真器中运行，收集通过/失败结果
5. **失败分析**：对失败场景进行根因分析，反馈至算法改进

---

### 覆盖率指标（Coverage Metrics）

衡量测试充分性的关键指标：

- **ODD覆盖度**：已测试场景覆盖ODD参数空间的比例
- **场景类别覆盖率**：已测试场景类别数 / 总定义场景类别数
- **代码覆盖率（软件层面）**：针对规划/控制代码的分支覆盖
- **决策覆盖率**：自动驾驶系统关键决策点（超车、让行、避障）的覆盖比例

---

### 关键场景（Critical Scenario）识别

从海量路测数据中自动挖掘关键/罕见场景：

- **数据挖掘方法**：基于碰撞风险指标（TTC、ETTC）从日志中筛选高风险时刻
- **异常检测**：用无监督学习识别与正常驾驶分布偏离的片段
- **对抗搜索（Adversarial Search）**：用强化学习或进化算法在仿真中自动搜索导致系统失败的场景参数组合
- **场景库扩充**：将挖掘到的关键场景纳入场景库，循环迭代提升系统鲁棒性

---

### 加速仿真（Accelerated Simulation）

为实现高效的大规模仿真验证，业界采用多种加速手段：

- **重要性采样（Importance Sampling）**：优先采样高风险、低概率场景，提升测试效率
- **自适应场景生成**：基于历史失败结果，动态调整场景采样分布
- **GPU并行仿真**：在多GPU集群上并行运行数千个仿真实例（如NVIDIA DRIVE Sim、Isaac Sim）
- **简化物理模型**：对非关键仿真步骤降低物理精度，换取速度

---

### 百亿英里等效测试

Waymo的Carcraft仿真系统是目前规模最大的自动驾驶仿真验证体系之一：

- **规模**：每年运行超过**100亿英里**的等效模拟里程（截至2023年数据）
- **方法**：基于真实采集数据重建的高精度场景，配合对抗场景生成
- **意义**：Waymo Rider-Only（无安全员）商业运营的安全验证基础

对比真实路测：Waymo截至2024年累计真实路测里程约3500万英里，仿真里程与真实路测比例超过250:1。


## 参考资料

1. Dosovitskiy, A. et al. "CARLA: An Open Urban Driving Simulator." *Conference on Robot Learning (CoRL)*, 2017.
2. Caesar, H. et al. "nuScenes: A multimodal dataset for autonomous driving." *CVPR*, 2020.
3. Sun, P. et al. "Scalability in Perception for Autonomous Driving: Waymo Open Dataset." *CVPR*, 2020.
4. Karnchanachari, N. et al. "nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles." *ICRA Workshop*, 2022.
5. Rempe, D. et al. "UniSim: A Neural Closed-Loop Sensor Simulator." *CVPR*, 2023.
6. Fremont, D. J. et al. "Scenic: A Language for Scenario Specification and Scene Generation." *PLDI*, 2019.
