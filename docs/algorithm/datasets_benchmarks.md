# 公开数据集与评测基准

自动驾驶算法的进步离不开高质量公开数据集与统一评测基准：前者为模型提供训练燃料，后者为不同方法提供横向对比的共同标尺。本节汇总感知、预测、规划与端到端四大方向中最常用的资源，并给出选择建议与使用陷阱。

---

## 1. 为什么数据集和基准重要？

- **可复现性**：没有统一测试集，论文结果难以比较，工程优化容易被运气误导；
- **长尾覆盖**：单一公司数据即使量级再大，也无法覆盖所有地理、天气、文化场景；
- **公平对比**：benchmark 的 leaderboard 能快速暴露新方法的真实边界；
- **工程脚手架**：成熟数据集带完整 SDK、标注工具、评测脚本，省下大量基础建设。

!!! warning "常见陷阱"
    - **训练集污染测试集**：同一路段不同时间采集的数据相互泄漏，导致指标虚高；
    - **标注一致性差**：多家标注商标准不同，拼接数据集时需重新校准；
    - **指标口径不同**：同为 mAP，IoU 阈值、类别权重略有差异就会出现 5~10% 的偏差；
    - **基准过拟合**：社区对特定榜单过度优化（"nuScenes-number chasing"），忽视真实场景。

---

## 2. 感知数据集

感知是自动驾驶数据最丰富的领域。下面按"数据体量 × 传感器复杂度"由低到高梳理。

### 2.1 KITTI（2012，卡尔斯鲁厄理工/丰田 TRI）

- **里程碑意义**：自动驾驶领域的"MNIST"，几乎所有经典 3D 检测论文都报告 KITTI 分数；
- **配置**：1 × 64 线 LiDAR（Velodyne HDL-64E）、双目摄像头、GPS/IMU；
- **规模**：约 15 k 帧训练 + 7.5 k 帧测试，仅覆盖德国卡尔斯鲁厄郊区晴天；
- **任务**：2D/3D 检测、语义分割、光流、里程计、跟踪、场景流；
- **局限**：数据量小、场景单一、标注密度低，2020 年后已不再是主力训练集。

### 2.2 Waymo Open Dataset（2019 起，Waymo）

- **规模**：1 200 段 20 秒的场景 + 1 150 个验证段（Perception v1.4，2024 年更新至 v2 包含更多天气）；
- **传感器**：5 × LiDAR（自研 Honeycomb 近程 + Top LiDAR）、5 × 高分辨率摄像头、时钟精对齐；
- **标注**：3D 边界框、语义分割、关键点、车道线、交互关系；
- **官方子挑战**：Motion Prediction、Occupancy & Flow、Sim Agents、End-to-End Driving；
- **特点**：传感器最齐全、标注工程质量最高；下载需同意 Waymo license，不可商用。

!!! note "2026 年 Waymo 不再举办正式挑战赛"
    Waymo 已宣布 **2026 年不举办正式的 Challenges**，但 Waymo Open Dataset 的 **leaderboard 仍然完全开放**，鼓励研究者继续提交以对比模型。这对科研节奏有实际影响：过去"每年 CVPR 前冲榜"的固定周期不再存在，评测更接近持续提交模式。

### 2.3 nuScenes（2019，Motional 前身 nuTonomy）

- **规模**：1 000 段 20 秒场景，约 1.4 M 图像 + 390 k LiDAR 扫描；
- **传感器**：6 × 摄像头（360°）、1 × 32 线 LiDAR、5 × 毫米波雷达；
- **场景**：波士顿 + 新加坡，含夜间/雨天；
- **独特价值**：毫米波雷达数据公开最全，BEV 时代（BEVFormer、PETR）的事实标准；
- **衍生**：nuImages（图像检测）、nuScenes-Occupancy（2023）、nuScenes-QA（多模态问答）。

### 2.4 Argoverse 1/2（2019/2022，Argo AI / Uber ATG）

- **Argoverse 1**：专注运动预测（324 k 场景）与 3D 跟踪；
- **Argoverse 2**：1 000 段感知场景 + 250 k 运动预测场景 + 1 000 个 HD Map 覆盖城市；
- **价值**：运动预测赛道主力，地图信息丰富，城市样本跨 6 个美国都市。

### 2.5 Cityscapes / BDD100K / Mapillary（图像感知）

- **Cityscapes**（2016）：5 k 精细像素级语义分割，德国城市，学术界分割基准；
- **BDD100K**（2018，伯克利）：100 k 视频 + 100 k 图像，10 类任务，覆盖美国多州；
- **Mapillary Vistas**（2017）：25 k 全球众包图像，100+ 国家，多文化场景。

### 2.6 NVIDIA Physical AI Autonomous Vehicles（2025）

NeurIPS 2025 随 Alpamayo 模型族一同发布，是目前规模最大的开放自动驾驶数据集：

| 维度 | 规模 |
| --- | --- |
| 时长 | 约 **1 700 小时**（官方另有 1 727 小时口径） |
| 地理覆盖 | **25 个国家、2 500+ 城市** |
| 片段 | **306 152** 段，每段 20 秒 |
| 传感器 | 全部片段含多摄像头；298 326 段含 LiDAR；160 761 段含毫米波雷达 |
| 体积 | 约 **133 TB**（约为 Waymo Open Dataset 的 3 倍） |

!!! success "最重要的一点是许可，而不是规模"
    Waymo Open、nuScenes 等主流数据集均 **限研究用途**，这使得学术成果难以直接进入产品。该数据集按其许可协议 **可用于商业与非商业 AV 开发**，并托管在 Hugging Face 上。对需要落地的团队而言，这一条比"3 倍于 WOD"的体量更有价值。
    
    但仍需注意：跨 25 个国家的数据意味着法规与隐私口径差异极大，实际使用前应确认目标市场的数据合规要求（尤其中国的数据出境规定）。

### 2.7 中国本土数据集

| 数据集 | 发布者 | 特色 |
| --- | --- | --- |
| **Apollo Scape** | 百度 | 2018 年发布，包含车道、3D 跟踪、轨迹、立体匹配多任务 |
| **ONCE** | 华为诺亚 | 1 M 场景、无标注日志，主打自监督预训练 |
| **ZOD**（Zenseact Open Dataset） | Zenseact | 2023，含大量欧洲高速场景，但包含中国合作数据 |
| **DAIR-V2X** | 清华 + 百度 | 首个车路协同数据集，含 RSU LiDAR 视角，C-V2X 研究必备 |
| **OpenLane / OpenLane-V2** | 上海 AI Lab | 车道线拓扑 + 交通要素，BEV 车道线主力基准 |

---

## 3. 运动预测与规划基准

### 3.1 Waymo Open Motion Dataset（WOMD）

- 1.1 M 个 9 秒场景，每个预测未来 8 秒；
- Motion Prediction Challenge：Top-K 轨迹 minADE/minFDE + Miss Rate + Soft mAP；
- Sim Agents Challenge（2023 起）：让模型同时预测所有交通参与者轨迹，用于闭环仿真；
- End-to-End Driving Challenge（2024 起）：从传感器直接输出轨迹。

### 3.2 Argoverse 2 Motion Forecasting

- 250 k 场景，每段 5 秒观察 + 6 秒预测；
- 指标与 WOMD 类似，更突出稀有交互（非保护左转、环岛）；
- 允许在 HD Map 上训练，被 Wayformer/Scene Transformer 等工作广泛使用。

### 3.3 nuPlan（2023，Motional）

- 首个大规模**规划闭环**基准：1 300 h 数据，1 300 k 场景，8 座城市；
- 支持开环（Open-Loop，预测轨迹与真值对比）和闭环（Closed-Loop，把规划器放进仿真看碰撞/偏离）两种评测；
- 评分综合碰撞率、可行驶区间、舒适度、进度、交通规则遵守；
- 后起之秀 **nuPlan-R**（2024）增加重规划对抗扰动。

### 3.4 Bench2Drive / CARLA Leaderboard

- **CARLA Leaderboard 2.0**（2023–）：在 CARLA 仿真器中跑 10 个城市 × 多天气 × 挑战场景，统计路线完成度、违章、碰撞；
- **Bench2Drive**（2024，上海 AI Lab）：在 CARLA 上提供 44 个技能场景 + 220 个闭环路线，是当前端到端学术工作的主流闭环基准。

---

## 4. 端到端与大模型专用基准

| 基准 | 任务 | 特点 |
| --- | --- | --- |
| **WOD-E2E**（CVPR 2026） | 长尾端到端驾驶 | 专门挖掘罕见高风险事件；以人类偏好评分替代专家轨迹对齐 |
| **NAVSIM v2**（CoRL 2025） | 伪闭环端到端评测 | 两阶段协议 + 3DGS 反事实视角合成 |
| **CARLA / Bench2Drive** | 端到端驾驶 | 仿真闭环，允许密集交互 |
| **nuScenes-OpenScene** | 开放闭环驾驶 | 以 nuScenes 为基础构造虚拟闭环 |
| **DriveLM** | 视觉-语言推理 | 场景问答 + 因果链，验证 VLM 驾驶能力 |
| **NuScenes-QA / LingoQA** | 驾驶问答 | 评估 VLM 对交通环境的理解 |
| **CODA / Corner Case Bench** | 长尾检测 | 特意采集不常见障碍物（横倒锥桶、异形车辆）|

### 4.1 WOD-E2E：把评测焦点移到长尾

WOD-E2E（Waymo Open Dataset for End-to-End Driving，CVPR 2026）不是"更大的数据集"，而是**刻意偏斜的数据集**：

- **规模**：4 021 段行车片段，约 12 小时——比通用数据集小得多
- **构造方式**：从 Waymo **640 万英里** 真实日志中挖掘，按人工定义的 **11 类长尾事件** 分类，并要求语料中出现频率 **低于 0.03%**
- **输入**：360° 视觉 + 详细车辆状态
- **关键指标**：**Rater Feedback Score（RFS）**——用人类标注的偏好评分衡量轨迹质量，**而非与记录的专家轨迹对齐**

!!! tip "RFS 解决的是一个真实的评测缺陷"
    传统开环指标（L2 误差、与专家轨迹的偏差）隐含假设"记录的人类轨迹是唯一正确答案"。但在长尾场景中，合理反应往往有多个，且人类司机自己的反应也不一定最优——此时逼近专家轨迹反而会惩罚正确的保守行为。RFS 用人类评分给多种可接受轨迹打分，绕开了这个问题。这也是为什么 **在长尾评测上，开环 L2 误差几乎失去意义**。

### 4.2 NAVSIM v2：用 3DGS 做"伪闭环"

闭环评测最真实但需要仿真器与反应式交通流；开环评测便宜但无法体现误差累积。NAVSIM（NeurIPS 2024）与 **NAVSIM v2**（CoRL 2025）提出的 **伪仿真（Pseudo-Simulation）** 取两者折中：

- **两阶段协议**：第二阶段的合成观测 **依赖于第一阶段的策略行为**，从而体现"决策改变了后续所见"
- **3DGS 反事实视角**：当策略偏离记录轨迹后，用 3D 高斯泼溅合成偏离位置上的相机视角——这是从日志数据近似闭环的关键
- **指标**：两阶段扩展 PDMS（**EPDMS**），新增交通灯合规（TLC）、行驶方向合规（DDC）、车道保持（LK）与扩展舒适性（EC）

伪仿真的意义在于：它让"闭环级别的鲁棒性评测"不再必须依赖完整仿真器，而可以直接建立在真实日志之上。

---

## 5. 选择数据集的实用建议

### 按任务匹配

| 任务 | 推荐首选 | 补充 |
| --- | --- | --- |
| 3D 目标检测 | Waymo Open + nuScenes | 华为 ONCE（预训练）、DAIR-V2X（车路协同）|
| BEV 车道线 | OpenLane-V2 | nuScenes + Argoverse 2 地图 |
| 运动预测 | WOMD + Argoverse 2 | INTERACTION（高交互性）|
| 规划闭环评测 | nuPlan | Bench2Drive（仿真）、NAVSIM v2（伪闭环）|
| 端到端 | NAVSIM v2 + Bench2Drive | WOD-E2E（长尾）、真实车队日志（自有）|
| 长尾/极端场景 | WOD-E2E | CODA、世界模型生成场景 |
| 需商用许可的大规模预训练 | NVIDIA Physical AI AV | 自有车队数据 |
| VLM 驾驶问答 | DriveLM + LingoQA | CODA（长尾）|

### 工程化注意事项

1. **许可合规**：Waymo Open、nuScenes 均限研究用途，商业落地前务必核对 license；需要商用的大规模预训练数据可考虑 NVIDIA Physical AI AV 数据集；
2. **数据泄漏**：同一 trip 的帧不要跨训练/验证；时间顺序采样而非随机切分；
3. **多数据集联合训练**：需统一坐标系、类别映射、标注密度，一般以 nuScenes 为基准对齐；
4. **合成与真实混合**：CARLA/Bench2Drive 适合探索算法上限，必须叠加真实域数据避免 sim2real 崩塌；
5. **定期刷新基准**：leaderboard 每年更新任务与指标，2019 年的 mAP 分数与 2024 年不可直接比较。

---

## 6. 行业评测与安全报告

除学术基准外，监管方与运营商也公开了一些宏观评测：

- **California DMV Disengagement Report**（每年 2 月）：加州路测公司按自愿报告披露 MPD（每千英里接管数）；
- **Waymo Safety Report / Safety Impact**：Waymo 持续公开全无人驾驶事故统计与对比数据；
- **NHTSA Standing General Order**：强制 Level 2+ 系统上报涉事事故，公开数据库可查询特斯拉/Cruise/Waymo 案例；
- **中国强标体系**：GB/T 40429 分级、GB/T 44721 测试场景、GB 44495 网络安全，是国内上市必查项。

---

## 7. 小结

- **感知**：nuScenes + Waymo Open 是事实标准；中国场景用 DAIR-V2X、ONCE、Apollo Scape 补齐；
- **预测**：WOMD 与 Argoverse 2 是旗帜；INTERACTION 专攻高交互；
- **规划/端到端**：nuPlan（真实日志）+ Bench2Drive（仿真闭环）+ **NAVSIM v2**（伪闭环）是当下组合拳；长尾能力单独用 **WOD-E2E** 评测；
- **规模与许可**：**NVIDIA Physical AI AV**（约 1 700 小时、25 国、可商用）改变了"大数据集只能做研究"的局面；
- **VLM/语言**：DriveLM、LingoQA、NuScenes-QA 是重点新基准；
- **监管**：California DMV、NHTSA、中国强标提供真实运营安全视角，是技术之外的重要评测渠道。

学术基准可以衡量进步，但最终落地仍需真实数据飞轮：读者在使用这些公开资源时，应始终把它们作为"起点"而非"终点"。

---

## 参考资料

1. Geiger, A., Lenz, P., Urtasun, R. Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. CVPR 2012.
2. Sun, P. et al. Scalability in Perception for Autonomous Driving: Waymo Open Dataset. CVPR 2020.
3. Caesar, H. et al. nuScenes: A multimodal dataset for autonomous driving. CVPR 2020.
4. Wilson, B. et al. Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting. NeurIPS Datasets 2021.
5. Caesar, H. et al. nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles. CVPR 2022.
6. Jia, X. et al. Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving. NeurIPS 2024.
7. California DMV. Autonomous Vehicle Disengagement Reports, annual.
8. NHTSA. Standing General Order on Crash Reporting. 2021–.
9. Xu, K. et al. WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios. CVPR 2026. [arXiv:2510.26125](https://arxiv.org/abs/2510.26125)
10. Dauner, D. et al. NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking. NeurIPS 2024；及 Pseudo-Simulation for Autonomous Driving. CoRL 2025.
11. NVIDIA. *PhysicalAI-Autonomous-Vehicles Dataset*. Hugging Face / NeurIPS 2025.
12. Waymo. *Challenges Overview — Waymo Open Dataset*. 2026（说明 2026 年不举办正式挑战赛）.
