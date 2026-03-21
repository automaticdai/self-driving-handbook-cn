# 交通场景生成

交通场景生成（Traffic Scenario Generation）是自动驾驶仿真系统的核心环节。高质量、高覆盖率的测试场景是验证自动驾驶系统安全性的前提条件。据估计，自动驾驶系统需要行驶超过 $10^{11}$ 公里才能在统计意义上证明其安全性优于人类驾驶员，纯靠实车路测不可行，因此仿真场景生成成为必经之路。

本章介绍场景生成的主要方法，包括：场景描述语言、基于规则的生成、数据驱动挖掘、对抗性生成、角落案例枚举、参数化与变异、复杂度度量，以及大规模场景管理。


## 1. 场景描述语言

场景描述语言为仿真场景提供标准化、可复现的表达方式，是场景生成流水线的基础。

### 1.1 OpenSCENARIO

OpenSCENARIO 是 ASAM 组织制定的基于 XML 的场景描述标准，被 CARLA、esmini、VTD 等主流仿真器广泛支持。

**核心概念：**

| 概念 | 说明 |
| --- | --- |
| Entities | 场景中的参与者（主车、NPC 车辆、行人等） |
| Actions | 参与者可执行的动作（变道、加速、跟车等） |
| Events | 触发动作的条件（距离触发、时间触发、碰撞触发等） |
| Story | 由多个 Act 组成的剧情，每个 Act 包含多个 Maneuver |
| Storyboard | 顶层容器，包含 Story 和全局触发条件 |

**示例：Cut-in 场景**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="1" date="2025-01-01"
              description="Cut-in scenario" author="example"/>
  <RoadNetwork>
    <LogicFile filepath="Town04.xodr"/>
  </RoadNetwork>
  <Entities>
    <ScenarioObject name="Ego">
      <Vehicle name="vehicle.tesla.model3" vehicleCategory="car"/>
    </ScenarioObject>
    <ScenarioObject name="Adversary">
      <Vehicle name="vehicle.bmw.grandtourer" vehicleCategory="car"/>
    </ScenarioObject>
  </Entities>
  <Storyboard>
    <Story name="CutInStory">
      <Act name="CutInAct">
        <ManeuverGroup name="AdversaryManeuvers">
          <Actors selectTriggeringEntities="false">
            <EntityRef entityRef="Adversary"/>
          </Actors>
          <Maneuver name="LaneChange">
            <Event name="CutInEvent" priority="overwrite">
              <Action name="LaneChangeAction">
                <PrivateAction>
                  <LateralAction>
                    <LaneChangeAction>
                      <LaneChangeActionDynamics dynamicsShape="sinusoidal"
                                                 value="3.0"
                                                 dynamicsDimension="time"/>
                      <LaneChangeTarget>
                        <RelativeTargetLane entityRef="Ego" value="0"/>
                      </LaneChangeTarget>
                    </LaneChangeAction>
                  </LateralAction>
                </PrivateAction>
              </Action>
              <StartTrigger>
                <ConditionGroup>
                  <Condition name="DistanceTrigger"
                             conditionEdge="rising" delay="0">
                    <ByEntityCondition>
                      <TriggeringEntities triggeringEntitiesRule="any">
                        <EntityRef entityRef="Adversary"/>
                      </TriggeringEntities>
                      <EntityCondition>
                        <RelativeDistanceCondition entityRef="Ego"
                            relativeDistanceType="longitudinal"
                            value="20.0" rule="lessThan"/>
                      </EntityCondition>
                    </ByEntityCondition>
                  </Condition>
                </ConditionGroup>
              </StartTrigger>
            </Event>
          </Maneuver>
        </ManeuverGroup>
      </Act>
    </Story>
  </Storyboard>
</OpenSCENARIO>
```

### 1.2 GeoScenario

GeoScenario 基于 GeoJSON 格式，将场景直接绑定到真实地理坐标系，适合基于高精地图的场景表达。其优势在于与地理信息系统（GIS）工具链天然兼容。

### 1.3 Scenic 概率编程语言

Scenic 是由 UC Berkeley 开发的概率编程语言，专为自动驾驶场景生成设计。其核心思想是将场景描述为一个带约束的概率分布，支持从分布中采样生成具体场景实例。

```python
# Scenic 示例：生成一个前方有障碍车辆的场景
param map = localPath('town04.xodr')
param carla_map = 'Town04'

ego = new Car on road
leading = new Car on road,
    ahead of ego by Range(10, 30),
    with speed Range(5, 15)

require leading can see ego
require (distance to leading) > 8
```

**Scenic 的核心特性：**

- **空间关系运算符**：`ahead of`、`behind`、`left of`、`visible from` 等
- **概率分布**：`Range(a, b)`、`Normal(mu, sigma)`、`Uniform(a, b, c)`
- **软硬约束**：`require`（硬约束）、`require[p]`（软约束，以概率 $p$ 满足）
- **可组合性**：可导入和继承已有场景进行扩展


## 2. 基于规则的场景生成

基于规则的方法通过预定义的驾驶行为模板和参数范围，系统性地生成测试场景。

### 2.1 典型场景模板

| 场景类别 | 关键参数 | 典型参数范围 |
| --- | --- | --- |
| 跟车（Car Following） | 初始车距、主车速度、前车减速度 | 车距 10–50 m，速度 30–120 km/h，减速度 0–8 m/s² |
| 切入（Cut-in） | 切入车相对距离、切入时间、切入角度 | 距离 5–30 m，时间 1–5 s，角度 5°–30° |
| 交叉路口（Intersection） | 交叉角度、对向车速度、信号灯状态 | 角度 60°–120°，速度 20–60 km/h |
| 障碍物避让 | 障碍物类型、偏移量、可见距离 | 偏移 0–3 m，可见距离 20–100 m |
| 行人横穿 | 横穿速度、起始位置、遮挡物 | 速度 1–6 m/s，距路口 0–20 m |
| 超车 | 对向车距离、同向慢车速度、可用车道宽度 | 对向距离 50–300 m，慢车速度 30–60 km/h |

### 2.2 参数化生成流程

```
场景模板定义 → 参数空间划分 → 采样策略选择 → 场景实例化 → 仿真执行 → 结果评估
```

常用采样策略包括：

- **网格采样**：对每个参数均匀离散化后做全组合，覆盖全面但指数爆炸
- **拉丁超立方采样（LHS）**：保证每个参数维度上的均匀覆盖，效率远高于网格采样
- **Sobol 序列**：低差异序列，兼顾均匀性和随机性


## 3. 数据驱动的场景挖掘

从真实道路测试数据中自动发现有价值的测试场景，是提升仿真场景真实性和覆盖率的重要途径。

### 3.1 关键事件检测

从行车记录中检测具有测试价值的片段：

| 检测指标 | 定义 | 阈值示例 |
| --- | --- | --- |
| TTC（Time To Collision） | 按当前速度到碰撞的时间 | TTC < 3 s |
| THW（Time Headway） | 车头时距 | THW < 1.0 s |
| PET（Post-Encroachment Time） | 两车先后经过同一冲突点的时间差 | PET < 1.5 s |
| 急刹车 | 纵向减速度超过阈值 | $a_x < -4 \text{ m/s}^2$ |
| 急转向 | 横向加速度超过阈值 | $a_y > 3 \text{ m/s}^2$ |
| 偏离车道 | 横向偏移超过车道宽度比例 | 偏移 > 0.3 × 车道宽度 |

### 3.2 场景聚类

将检测到的关键事件聚类为代表性场景类别：

**K-Means 聚类：**

将每个场景片段表示为特征向量 $\mathbf{x}_i = [v_{ego}, v_{obj}, d, \theta, a_{lat}, ...]$，优化目标为最小化类内距离：

$$J = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

**DBSCAN 密度聚类：**

适合发现任意形状的场景簇，不需要预设簇数。通过邻域半径 $\varepsilon$ 和最小样本数 $MinPts$ 两个参数控制，能自动将稀疏区域的场景标记为噪声点（离群场景）。

### 3.3 行为克隆生成交通参与者

利用真实驾驶数据训练交通参与者的行为模型：

$$\pi_\theta(a_t | s_t) = \arg\max_\theta \sum_{t=1}^{T} \log \pi_\theta(a_t^{expert} | s_t)$$

其中 $s_t$ 为交通环境状态，$a_t^{expert}$ 为真实驾驶员的动作，$\pi_\theta$ 为参数化策略网络。训练后的模型可驱动仿真中的 NPC 车辆，生成更真实的交通流。


## 4. 对抗性场景生成

对抗性场景生成旨在主动搜索可能导致自动驾驶系统失效的场景，是安全验证的关键手段。

### 4.1 优化问题建模

将对抗性场景搜索建模为优化问题：

$$\mathbf{x}^* = \arg\max_{\mathbf{x} \in \mathcal{X}} \; R(\mathbf{x})$$

$$\text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \quad i = 1, ..., m$$

其中：

- $\mathbf{x}$ 为场景参数向量（车辆位姿、速度、天气、行人行为等）
- $\mathcal{X}$ 为合理的场景参数空间
- $R(\mathbf{x})$ 为风险度量函数（如碰撞概率、最小 TTC 的负值）
- $g_i(\mathbf{x})$ 为自然性约束（保证生成的场景物理合理）

### 4.2 搜索算法

**遗传算法（GA）：**

将场景参数编码为染色体，通过选择、交叉、变异操作进行进化搜索：

```
初始化种群 P₀ (N 个随机场景)
for generation = 1, 2, ..., G:
    评估适应度: fitness(xᵢ) = R(xᵢ)
    选择: 基于适应度轮盘赌选择
    交叉: 单点/多点交叉生成子代
    变异: 以概率 pₘ 对参数高斯扰动
    更新种群: P_{t+1} = 选择(父代 ∪ 子代)
返回最优个体
```

**贝叶斯优化：**

使用高斯过程（GP）作为代理模型，通过采集函数（Acquisition Function）平衡探索与利用：

$$\alpha_{EI}(\mathbf{x}) = \mathbb{E}\left[\max(R(\mathbf{x}) - R^+, 0)\right]$$

其中 $R^+$ 为当前已知最优风险值。贝叶斯优化在场景参数维度较低（$< 20$维）时效率极高。

### 4.3 强化学习对抗智能体

训练一个对抗性智能体控制 NPC 车辆，目标为最大化与被测自动驾驶系统（ADS）的碰撞概率：

$$\max_{\pi_{adv}} \mathbb{E}_{\tau \sim \pi_{adv}} \left[\sum_{t=0}^{T} \gamma^t r_t \right]$$

其中奖励函数设计需平衡对抗性和自然性：

$$r_t = \underbrace{w_1 \cdot r_{collision}}_{\text{碰撞奖励}} + \underbrace{w_2 \cdot r_{nearness}}_{\text{接近奖励}} - \underbrace{w_3 \cdot r_{unnature}}_{\text{非自然惩罚}}$$

常用算法包括 PPO、SAC、MADDPG（多智能体场景）。非自然惩罚项约束 NPC 的行为不偏离真实交通流太远，避免生成无意义的"自杀式"碰撞场景。


## 5. 角落案例枚举

角落案例（Corner Case）是指在正常驾驶中极少出现、但可能导致系统失效的极端场景。

### 5.1 角落案例分类

| 类别 | 描述 | 示例 |
| --- | --- | --- |
| 感知失效 | 传感器无法正确识别目标 | 反光路面导致相机过曝、黑色车辆在夜间不可见 |
| 预测失效 | 目标行为超出预测模型覆盖范围 | 行人突然折返、车辆逆行 |
| 遮挡场景 | 关键目标被其他物体遮挡 | "鬼探头"——行人从停靠车辆后突然出现 |
| 极端天气 | 恶劣天气降低传感器性能 | 暴雨、大雾、强逆光、冰雪路面 |
| 罕见物体 | 训练数据中缺乏的目标类别 | 掉落货物、动物、特种车辆、施工设备 |
| 基础设施异常 | 道路/标志/信号异常 | 临时改道、信号灯故障闪烁、车道线磨损 |
| 多重组合 | 多个边缘条件同时出现 | 雨天 + 弯道 + 行人横穿 + 遮挡 |

### 5.2 生成方法

- **知识驱动**：由安全专家根据事故数据库（如 NHTSA、GIDAS）梳理角落案例清单
- **规则穷举**：将环境因素（天气、光照、道路类型）和交通因素（参与者类型、行为模式）做组合穷举
- **反向推理**：从已知系统缺陷（如特定条件下的检测失败）反向构造触发场景
- **GAN 生成**：训练生成对抗网络，在传感器输入空间中生成对抗性样本（如对抗性纹理、雨滴模拟）


## 6. 场景参数化与变异

### 6.1 参数化场景模板

将场景抽象为模板 + 参数的形式：

$$\mathcal{S} = \mathcal{T}(\mathbf{p}), \quad \mathbf{p} = [p_1, p_2, ..., p_n] \in \mathcal{P}$$

其中 $\mathcal{T}$ 为场景模板，$\mathbf{p}$ 为参数向量，$\mathcal{P}$ 为参数空间。例如"Cut-in 场景"模板的参数空间为：

| 参数 | 符号 | 范围 | 分布 |
| --- | --- | --- | --- |
| 主车速度 | $v_{ego}$ | [60, 120] km/h | 均匀分布 |
| 切入车初始纵向偏移 | $\Delta x$ | [10, 40] m | 均匀分布 |
| 切入车初始横向偏移 | $\Delta y$ | [3.0, 4.5] m | 正态分布 |
| 切入持续时间 | $t_{lc}$ | [1.5, 5.0] s | 均匀分布 |
| 切入车最终速度差 | $\Delta v$ | [-20, 10] km/h | 正态分布 |
| 天气 | $w$ | {晴, 阴, 小雨, 大雨, 雾} | 离散分布 |

### 6.2 组合测试

当参数空间维度较高时，全组合测试不可行。**覆盖数组（Covering Array）**提供了一种高效替代方案：

- **$t$-维覆盖**：保证任意 $t$ 个参数的所有值组合至少出现一次
- 典型选择 $t = 2$（成对覆盖）或 $t = 3$（三维覆盖）
- 用例数从全组合的 $O(\prod_i |D_i|)$ 降低到 $O(\max_i |D_i|^t \cdot \log n)$

### 6.3 重要性采样

对场景参数空间进行非均匀采样，将更多仿真资源集中在高风险区域：

$$\hat{P}_{fail} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[f(\mathbf{x}_i) \in \text{Fail}] \cdot \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}$$

其中 $p(\mathbf{x})$ 为场景的真实分布，$q(\mathbf{x})$ 为重要性分布（偏向于高风险区域采样）。通过交叉熵方法（Cross-Entropy Method）可以自适应地更新 $q(\mathbf{x})$：

1. 从当前提议分布 $q_k$ 中采样 $N$ 个场景
2. 执行仿真，评估风险 $R(\mathbf{x}_i)$
3. 选取风险值排名前 $\rho$% 的精英样本
4. 用精英样本拟合更新后的提议分布 $q_{k+1}$
5. 重复直到收敛


## 7. 场景复杂度度量

为了评估场景的测试价值和区分难度，需要定义场景复杂度的量化指标。

### 7.1 关键性度量指标

| 指标 | 公式 / 定义 | 说明 |
| --- | --- | --- |
| 最小 TTC | $TTC_{min} = \min_t TTC(t)$ | 整个场景中最危险时刻的碰撞时间 |
| 最小距离 | $d_{min} = \min_t d_{ego, obj}(t)$ | 主车与最近障碍物的最小距离 |
| 必要减速度 | $a_{req} = \frac{v_{rel}^2}{2 \cdot d}$ | 避免碰撞所需的最小制动减速度 |
| 反应时间裕度 | $\Delta t = TTC - t_{react}$ | 碰撞时间减去系统反应时间 |
| 责任敏感安全（RSS）距离 | 基于 Mobileye RSS 模型 | 主车是否保持了 RSS 安全距离 |

### 7.2 综合复杂度评分

将多维指标归一化后加权求和：

$$C(\mathcal{S}) = \sum_{j=1}^{M} w_j \cdot \tilde{m}_j(\mathcal{S})$$

其中 $\tilde{m}_j$ 为第 $j$ 个归一化指标，$w_j$ 为权重。权重可以通过层次分析法（AHP）或专家打分确定。

**场景难度分级参考：**

| 等级 | 综合评分 | 典型特征 |
| --- | --- | --- |
| L1（简单） | 0.0–0.2 | 直道跟车、无交互 |
| L2（一般） | 0.2–0.4 | 简单变道、低速交叉 |
| L3（中等） | 0.4–0.6 | 多车交互、中速切入 |
| L4（困难） | 0.6–0.8 | 密集交通、遮挡、恶劣天气 |
| L5（极端） | 0.8–1.0 | 多重角落案例叠加 |


## 8. 大规模场景管理

工业级自动驾驶仿真测试涉及数百万甚至数十亿个场景，需要系统化的管理方案。

### 8.1 场景数据库架构

```
┌─────────────────────────────────────────────────┐
│                 场景数据库                        │
├──────────┬──────────┬──────────┬────────────────┤
│  元数据层  │  场景层   │  结果层   │   索引层       │
├──────────┼──────────┼──────────┼────────────────┤
│ - 场景ID   │ - 模板    │ - 通过/失败│ - 标签索引     │
│ - 创建时间  │ - 参数    │ - 关键指标 │ - 全文检索     │
│ - 来源     │ - 地图    │ - 日志路径 │ - 向量检索     │
│ - 版本     │ - 天气    │ - 复现种子 │ - 参数范围索引  │
│ - 标签     │ - 参与者  │ - 覆盖度   │               │
└──────────┴──────────┴──────────┴────────────────┘
```

### 8.2 版本控制

场景随 ADS 版本迭代而演化，版本管理要点：

- **场景版本化**：每个场景有唯一 ID 和版本号，修改参数后递增版本
- **基线管理**：维护各版本 ADS 对应的回归测试场景集（Baseline Suite）
- **差异追踪**：新版本场景集与旧版本的增量对比，追踪新增/删除/修改的场景
- **可复现性**：记录随机种子、仿真器版本、地图版本等，确保结果可复现

### 8.3 标签与检索

多维标签体系支持灵活的场景检索：

| 标签维度 | 示例值 |
| --- | --- |
| 功能场景（ODD） | 高速公路、城市道路、停车场 |
| 天气条件 | 晴天、雨天、雾天、雪天、夜间 |
| 参与者类型 | 乘用车、卡车、两轮车、行人、动物 |
| 关键行为 | 切入、急刹、逆行、闯红灯、鬼探头 |
| 测试结果 | 通过、失败、边界 |
| 来源 | 规则生成、数据挖掘、对抗生成、专家编写 |
| 关键性等级 | L1–L5 |

典型查询示例：

```sql
SELECT scenario_id, parameters, criticality_score
FROM scenarios
WHERE odd_tag = '城市道路'
  AND weather IN ('雨天', '雾天')
  AND behavior_tag = '行人横穿'
  AND criticality_level >= 'L3'
  AND test_result = '失败'
ORDER BY criticality_score DESC
LIMIT 100;
```


## 参考资料

1. ASAM OpenSCENARIO V1.1 User Guide. ASAM e.V., 2022.
2. Fremont D J, Dreossi T, Ghosh S, et al. Scenic: A Language for Scenario Specification and Scene Generation. *PLDI*, 2019.
3. Feng S, Feng Y, Yu C, et al. Testing Scenario Library Generation for Connected and Automated Vehicles, Part I: Methodology. *IEEE Transactions on Intelligent Transportation Systems*, 2021, 22(3): 1555–1567.
4. Feng S, Feng Y, Sun H, et al. Testing Scenario Library Generation for Connected and Automated Vehicles, Part II: Case Studies. *IEEE Transactions on Intelligent Transportation Systems*, 2021, 22(3): 1568–1581.
5. Ding W, Chen B, Li B, et al. A Survey on Safety-Critical Driving Scenario Generation — A Methodological Perspective. *IEEE Transactions on Intelligent Transportation Systems*, 2023.
6. Klischat M, Althoff M. Generating Critical Test Scenarios for Automated Vehicles with Evolutionary Algorithms. *IEEE Intelligent Vehicles Symposium*, 2019.
7. Sun C, Schwartz M, Bhatt S, et al. Corner Cases for Visual Perception in Automated Driving: Some Guidance on Detection Approaches. *arXiv preprint arXiv:2102.05897*, 2021.
8. Riedmaier S, Ponn T, Ludwig D, et al. Survey on Scenario-Based Safety Assessment of Automated Vehicles. *IEEE Access*, 2020, 8: 87456–87477.
9. Shalev-Shwartz S, Shammah S, Shashua A. On a Formal Model of Safe and Scalable Self-driving Cars. *arXiv preprint arXiv:1708.06374*, 2017.
10. Menzel T, Bagschik G, Maurer M. Scenarios for Development, Test and Validation of Automated Vehicles. *IEEE Intelligent Vehicles Symposium*, 2018.
