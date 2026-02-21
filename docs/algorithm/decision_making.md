# 决策 (Decision Making)

## 1. 开篇介绍

决策系统是自动驾驶的"大脑"，决定了一个系统的性质是自动系统（Automated Systems）还是自主系统（Autonomous Systems）。在自动驾驶软件栈中，决策模块位于感知与规划之间，承上启下，是整个系统智能化程度的核心体现。

### 1.1 感知→预测→决策→规划链条

自动驾驶软件栈可以抽象为以下处理链：

```
传感器原始数据
      ↓
感知（Perception）：目标检测、语义分割、占据栅格
      ↓
预测（Prediction）：周围车辆/行人未来轨迹预测
      ↓
决策（Decision Making）：选择驾驶行为策略
      ↓
规划（Planning）：生成具体可执行轨迹
      ↓
控制（Control）：执行油门/制动/转向指令
```

感知模块提供对当前世界状态的理解，预测模块估计环境的未来演化，决策模块在此基础上确定自车应采取的高层行为（如变道、跟车、礼让），最终由规划模块将这一行为决策转化为精确的时空轨迹。

### 1.2 决策的难点

相比感知和控制，决策面临以下特有挑战：

- **不确定性叠加**：传感器噪声、预测误差、他车意图未知，决策需要在多重不确定性下保证安全。
- **多智能体交互**：路口、合流等场景中，自车行为会影响他车行为，形成反应循环，难以用单智能体模型描述。
- **长时序依赖**：一次超车决策的安全性需要考虑数秒乃至十几秒的未来演化，而非仅看当前时刻。
- **场景多样性**：从高速直道到复杂城区，驾驶场景的组合几乎是无限的，难以穷举所有情况进行规则编码。
- **可解释性与安全性**：决策系统的输出必须可追溯、可验证，以满足功能安全标准（如 ISO 26262）的要求。


## 2. 决策层次结构

自动驾驶决策通常采用三层架构，不同层次在时间尺度、抽象程度和求解方法上各有侧重。

### 2.1 任务规划（Mission Planning）

任务规划在地图层面确定从起点到终点的最优路线，时间尺度为分钟到小时级别。主要输入包括高精度地图（HD Map）、实时交通信息和用户目的地。常用算法有 Dijkstra 最短路径和 A* 启发式搜索。

任务规划的核心是代价函数设计，综合考虑行驶距离、预计时间、道路等级和通行费等因素：

$$C_{route} = w_d \cdot d + w_t \cdot t + w_r \cdot r_{toll} + w_e \cdot e_{energy}$$

其中 $d$ 为距离，$t$ 为预计行驶时间，$r_{toll}$ 为通行费，$e_{energy}$ 为能耗估计，$w_*$ 为对应权重。

### 2.2 行为规划（Behavior Planning）

行为规划是决策的核心层，在道路场景中选择合适的驾驶行为，时间尺度为秒级。典型行为包括：车道保持（Lane Keeping）、变道（Lane Change）、跟车（Car Following）、超车（Overtaking）、汇入（Merging）、礼让（Yielding）和停车（Stopping）。

行为规划的输入来自感知和预测模块，输出的是离散化的高层行为指令，供运动规划模块进一步细化。

### 2.3 动作规划（Motion Planning）

动作规划将行为规划的高层指令转化为时间上连续的可执行轨迹 $\{(x_t, y_t, \theta_t, v_t)\}_{t=0}^{T}$，需要满足车辆运动学约束、舒适性约束（曲率连续、加速度有界）和障碍物约束。

### 2.4 三层架构示意图

```
┌─────────────────────────────────────────────────────────┐
│              任务规划（Mission Planning）                │
│   输入：地图 + 目的地    输出：参考路线（Road Graph）    │
│   时间尺度：分钟～小时   方法：A*, Dijkstra             │
└─────────────────────────┬───────────────────────────────┘
                          │ 参考路线
┌─────────────────────────▼───────────────────────────────┐
│              行为规划（Behavior Planning）               │
│   输入：感知 + 预测      输出：驾驶行为指令              │
│   时间尺度：秒级         方法：FSM, MPDM, POMDP, RL     │
└─────────────────────────┬───────────────────────────────┘
                          │ 行为指令
┌─────────────────────────▼───────────────────────────────┐
│              运动规划（Motion Planning）                 │
│   输入：行为指令 + 环境  输出：时空轨迹                  │
│   时间尺度：毫秒～秒     方法：Frenet, Lattice, MPC      │
└─────────────────────────────────────────────────────────┘
```


## 3. 有限状态机（FSM）

有限状态机是最早用于自动驾驶行为决策的方法，凭借其结构清晰、可解释性强的特点，至今仍被广泛应用于工业系统。

### 3.1 状态定义

一个典型的高速公路场景 FSM 包含以下状态：

| 状态名称 | 描述 |
| --- | --- |
| 车道保持（Lane Keep） | 在当前车道内以目标速度行驶 |
| 跟车（Car Follow） | 检测到前车，以安全车距跟随 |
| 变道准备（Prepare LC） | 评估目标车道可行性 |
| 变道执行（Execute LC） | 执行横向变道机动 |
| 超车（Overtake） | 超越慢速前车后返回原车道 |
| 停车（Stop） | 减速并停止于停止线或障碍物前 |
| 紧急制动（Emergency） | 触发 AEB，以最大减速度制动 |

### 3.2 状态转移条件（触发器）

状态转移由条件触发器驱动，例如：

- **车道保持 → 跟车**：前方 $d < d_{safe}$，且相对速度 $\Delta v < 0$（前车更慢）
- **跟车 → 变道准备**：前车速度 $v_{lead} < v_{target} - \Delta v_{thresh}$，且持续时间超过 $t_{thresh}$
- **变道准备 → 变道执行**：目标车道间隙 $gap > gap_{min}$，且无后方快速接近车辆
- **任意状态 → 紧急制动**：TTC（碰撞时间）$< T_{emergency}$

### 3.3 优缺点分析

**优点：**
- 逻辑清晰，每个状态和转移条件均可解释和审计
- 开发和调试周期短，适合快速工程落地
- 可直接编码交通规则，与法规合规性对齐

**缺点：**
- **状态爆炸**：场景复杂度增加时，状态数量呈指数级增长，难以维护
- **转移条件脆弱**：阈值参数需要大量人工调优，泛化能力差
- **无法处理不确定性**：FSM 假设感知结果确定，无法建模传感器噪声
- **多智能体局限**：难以显式建模他车的响应行为

### 3.4 层次 FSM（HFSM）

为缓解状态爆炸问题，可将 FSM 组织为层次结构（Hierarchical FSM）。高层 FSM 管理抽象行为（如高速行驶、城区行驶、停车场行驶），每个高层状态内部嵌套一个子 FSM 处理具体操作。

```
高层 FSM
├── 高速公路模式
│   ├── 子FSM：车道保持 / 跟车 / 变道
├── 城区模式
│   ├── 子FSM：直行 / 转弯 / 让行 / 路口通过
└── 紧急模式
    └── 子FSM：紧急制动 / 靠边停车
```

### 3.5 行为树（Behavior Tree）与 FSM 对比

行为树（Behavior Tree, BT）是另一种常用的决策表示形式，通过组合控制节点（Sequence、Selector、Parallel）和叶节点（条件、动作）构建决策逻辑。

| 特性 | 有限状态机（FSM） | 行为树（BT） |
| --- | --- | --- |
| 模块化 | 差（全局状态转移） | 强（子树可复用） |
| 可扩展性 | 差（状态爆炸） | 好（局部修改） |
| 并发处理 | 困难 | 原生支持（Parallel 节点） |
| 调试难度 | 低（状态明确） | 中等 |
| 工业应用 | 广泛（Waymo 早期） | 增长（ROS 2 生态） |


## 4. 基于规则的决策

### 4.1 交通规则编码

交通规则可以被系统化地编码为一组约束和优先级规则，常见规则包括：

- **限速规则**：$v \leq v_{limit}(road\_type, weather, visibility)$
- **安全车距规则**（2秒原则）：$d_{safe} \geq v_{ego} \cdot t_{headway}$，其中 $t_{headway} \approx 2s$
- **优先权规则**：右侧优先、直行优先于转弯、主路优先于辅路
- **礼让规则**：在无保护转弯时必须等待间隙（Gap Acceptance）足够大才可通行

间隙接受模型（Gap Acceptance Model）用于判断是否可以插入某个间隙：

$$P(accept) = \begin{cases} 1 & \text{若 } gap \geq gap_{critical} \\ 0 & \text{若 } gap < gap_{critical} \end{cases}$$

更精细的概率模型可以用 Logistic 函数建模：

$$P(accept \mid gap) = \frac{1}{1 + e^{-\beta(gap - \mu)}}$$

其中 $\mu$ 为临界间隙均值，$\beta$ 为陡峭度参数。

### 4.2 势场评估（Potential Field）

人工势场法将道路环境建模为势能场，自车受到目标点的引力和障碍物的斥力共同作用：

$$U_{total}(q) = U_{att}(q) + U_{rep}(q)$$

引力场（指向目标）：

$$U_{att}(q) = \frac{1}{2} k_{att} \|q - q_{goal}\|^2$$

斥力场（远离障碍物）：

$$U_{rep}(q) = \begin{cases} \frac{1}{2} k_{rep} \left(\frac{1}{\rho(q)} - \frac{1}{\rho_0}\right)^2 & \text{若 } \rho(q) \leq \rho_0 \\ 0 & \text{若 } \rho(q) > \rho_0 \end{cases}$$

其中 $\rho(q)$ 为自车到最近障碍物的距离，$\rho_0$ 为影响范围阈值，$k_{att}$、$k_{rep}$ 为权重系数。决策方向沿负梯度方向 $F = -\nabla U_{total}$。

### 4.3 基于代价函数的决策

对候选行为 $\{a_1, a_2, \ldots, a_n\}$ 分别计算综合代价，选择代价最低的行为：

$$a^* = \arg\min_{a_i} C(a_i) = \arg\min_{a_i} \left[ w_s \cdot C_{safety} + w_e \cdot C_{efficiency} + w_c \cdot C_{comfort} + w_r \cdot C_{rule} \right]$$

其中各分项代价分别量化安全风险、行驶效率（与期望速度的偏差）、乘坐舒适度（加速度/急动度）和交规遵守程度。


## 5. MPDM（多策略并行决策模型）

MPDM（Multipolicy Decision Making）是斯坦福大学提出的一种面向城区驾驶的决策框架，通过对有限策略集合进行前向仿真来选择最优策略。

### 5.1 策略集合定义

MPDM 使用一组参数化的闭环策略（Closed-loop Policy）来描述候选行为，典型策略集合包括：

| 策略 $\pi_i$ | 描述 |
| --- | --- |
| $\pi_{LK}$ | 车道保持（Lane Keep），以目标速度巡航 |
| $\pi_{LCL}$ | 向左变道（Lane Change Left） |
| $\pi_{LCR}$ | 向右变道（Lane Change Right） |
| $\pi_{ACC}$ | 加速（Accelerate）至速度上限 |
| $\pi_{DEC}$ | 减速（Decelerate）并准备停车 |

### 5.2 前向仿真评估

对每个候选策略 $\pi_i$，MPDM 利用预测模块对场景进行前向仿真，得到一条预测轨迹 $\tau_i = \{s_0, s_1, \ldots, s_T\}$，其中他车行为同样采用闭环策略建模（通常假设他车也执行某一预定策略）。

### 5.3 策略选择（最大期望回报）

策略价值通过折扣累积奖励计算：

$$V(\pi_i) = \sum_{t=0}^{T} \gamma^t R(s_t, a_t)$$

其中 $\gamma \in (0,1]$ 为折扣因子，$R(s_t, a_t)$ 为即时奖励函数，综合安全、效率和舒适度：

$$R(s_t, a_t) = r_{safety}(s_t) + \lambda_e \cdot r_{efficiency}(s_t) + \lambda_c \cdot r_{comfort}(a_t)$$

最终选择价值最高的策略：

$$\pi^* = \arg\max_{\pi_i} V(\pi_i)$$

MPDM 的计算复杂度为 $O(|\Pi| \cdot N_{sim} \cdot T)$，其中 $|\Pi|$ 为策略数量，$N_{sim}$ 为仿真步数，满足实时性要求。


## 6. POMDP（部分可观测马尔可夫决策过程）

POMDP 是处理不确定性决策的理论框架，相比 MDP，它显式地建模了感知的不完整性，更贴近真实驾驶场景。

### 6.1 形式化定义

一个 POMDP 由七元组 $(S, A, T, R, \Omega, O, \gamma)$ 定义：

| 符号 | 名称 | 含义 |
| --- | --- | --- |
| $S$ | 状态空间 | 环境的真实状态（含他车意图等隐变量） |
| $A$ | 动作空间 | 自车可执行的行为集合 |
| $T(s'\|s,a)$ | 状态转移函数 | 在状态 $s$ 执行动作 $a$ 后转移到 $s'$ 的概率 |
| $R(s,a)$ | 奖励函数 | 在状态 $s$ 执行动作 $a$ 的即时奖励 |
| $\Omega$ | 观测空间 | 传感器可观测到的量（带噪声） |
| $O(o\|s',a)$ | 观测函数 | 到达状态 $s'$ 后观测到 $o$ 的概率 |
| $\gamma$ | 折扣因子 | 对未来奖励的折扣（$\gamma \in [0,1)$） |

在驾驶场景中，隐状态通常包括他车驾驶员的意图（如是否准备变道）、行人的目的地等，这些量无法通过传感器直接观测。

### 6.2 信念状态更新

由于真实状态不可直接观测，POMDP 维护一个**信念状态**（Belief State）$b(s)$，表示对当前状态的概率分布。当执行动作 $a$ 并观测到 $o$ 后，信念状态按贝叶斯规则更新：

$$b'(s') = \eta \cdot O(o \mid s', a) \sum_{s \in S} T(s' \mid s, a) \cdot b(s)$$

其中 $\eta$ 为归一化常数，确保 $\sum_{s'} b'(s') = 1$。

信念状态 $b(s)$ 是一个连续的概率分布，POMDP 的求解等价于在信念空间上找到最优策略 $\pi^*: b \rightarrow a$。

### 6.3 近似求解算法

精确求解 POMDP 是 PSPACE-hard 问题，实际应用中使用近似算法：

- **DESPOT（Determinized Sparse Partially Observable Tree）**：通过对随机性进行确定化采样，构建稀疏信念树，时间复杂度大幅降低，适合实时决策。
- **POMCP（Partially Observable Monte-Carlo Planning）**：基于蒙特卡洛树搜索（MCTS）的在线规划算法，用粒子滤波近似信念状态。
- **QMDP**：一种快速近似方法，假设下一步后状态完全可观测，用于得到策略上界。

### 6.4 POMDP 在路口博弈中的应用

在无保护左转场景中，对向来车的意图（减速礼让 vs. 保持速度通过）是关键隐状态。POMDP 框架可以：

1. 建模来车在"减速"和"直行"意图下的速度分布（观测模型）
2. 根据历史观测更新来车意图的信念概率
3. 在高不确定性时选择保守行为（等待），在信念集中时选择通过

这使得决策系统能够在不确定条件下做出安全且不过度保守的决策。


## 7. 基于学习的决策

### 7.1 行为克隆（Imitation Learning）

行为克隆（Behavior Cloning, BC）从人类驾驶数据中学习决策策略，将其建模为监督学习问题：

$$\min_\theta \mathbb{E}_{(s,a) \sim \mathcal{D}_{human}} \left[ \mathcal{L}(f_\theta(s), a) \right]$$

其中 $\mathcal{D}_{human}$ 为人类驾驶数据集，$f_\theta$ 为参数化的策略网络，$\mathcal{L}$ 为损失函数（分类行为用交叉熵，连续控制用 MSE）。

行为克隆的主要问题是**协变量漂移（Covariate Shift）**：训练数据来自专家轨迹，而测试时策略的错误会导致状态分布偏离训练分布，错误不断累积。DAgger 算法通过迭代地在策略执行轨迹上查询专家标注来缓解这一问题。

### 7.2 逆强化学习（IRL）

逆强化学习（Inverse Reinforcement Learning）从专家示范中推断隐含的奖励函数，再用该奖励函数进行强化学习。相比直接克隆行为，IRL 能更好地泛化到新场景。

**最大熵 IRL**（Maximum Entropy IRL）假设专家策略在奖励函数 $R(s,a;\theta)$ 下具有最大熵分布，求解：

$$\max_\theta \sum_{(s,a) \in \mathcal{D}} \log P_\theta(\tau) = \max_\theta \sum_{(s,a) \in \mathcal{D}} R(s,a;\theta) - \log Z(\theta)$$

其中 $Z(\theta) = \sum_\tau \exp\left(\sum_t R(s_t, a_t; \theta)\right)$ 为配分函数。梯度为：

$$\nabla_\theta \mathcal{L} = \mathbb{E}_{\mathcal{D}}[\nabla_\theta R] - \mathbb{E}_{\pi_\theta}[\nabla_\theta R]$$

即专家特征期望与策略特征期望之差。

### 7.3 强化学习（RL）

深度强化学习通过与（仿真）环境交互，学习最大化长期累积奖励的策略：

$$J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

常用算法：

- **PPO（Proximal Policy Optimization）**：限制策略更新幅度，训练稳定，适合连续动作控制
- **SAC（Soft Actor-Critic）**：最大熵强化学习，鼓励探索，适合高维连续动作空间
- **TD3（Twin Delayed Deep Deterministic Policy Gradient）**：解决 Q 值高估问题，适合确定性策略

### 7.4 奖励函数设计

奖励函数是 RL 系统的核心，通常综合以下分项：

$$r_t = r_{safe} + w_e \cdot r_{eff} + w_c \cdot r_{comf} + w_r \cdot r_{rule}$$

| 分项 | 正奖励条件 | 负奖励（惩罚）条件 |
| --- | --- | --- |
| $r_{safe}$ | TTC 充足，无碰撞风险 | 碰撞：$-100$；TTC $<$ 阈值：$-10$ |
| $r_{eff}$ | 车速接近限速 | 过慢行驶；不必要停车 |
| $r_{comf}$ | 加速度/急动度在舒适范围内 | 急刹车；急转向 |
| $r_{rule}$ | 遵守交通信号和路权规则 | 闯红灯；逆行；违规变道 |

奖励塑形（Reward Shaping）时需注意：过强的安全惩罚会使策略极度保守，过强的效率奖励会导致激进驾驶。


## 8. 博弈论与交互决策

### 8.1 纳什均衡在交通中的应用

在多车交互场景中，每辆车的最优策略依赖于其他车辆的策略选择，形成策略耦合。纳什均衡（Nash Equilibrium）描述了这样一个状态：在均衡点，任何一方单独改变策略都不能提高自身收益。

对于两车博弈（自车 $e$，他车 $o$），纳什均衡 $(\pi_e^*, \pi_o^*)$ 满足：

$$V_e(\pi_e^*, \pi_o^*) \geq V_e(\pi_e, \pi_o^*), \quad \forall \pi_e$$
$$V_o(\pi_e^*, \pi_o^*) \geq V_o(\pi_e^*, \pi_o), \quad \forall \pi_o$$

在合流场景中，博弈矩阵可能出现"鹰鸽博弈"（Hawk-Dove Game）结构：双方都激进则可能碰撞，双方都保守则效率低下，纳什均衡给出理性的混合策略。

### 8.2 主从博弈（Stackelberg Game）

在许多实际驾驶场景中，博弈双方并非同时做决策，而是存在领导者（Leader）和跟随者（Follower）的层次结构。主从博弈（Stackelberg Game）刻画这种场景：

- **领导者**（通常为自车）率先宣告并承诺其策略 $\pi_e$
- **跟随者**（他车）观察到领导者策略后，选择最优响应 $\pi_o^*(\pi_e)$
- 领导者预见跟随者的响应，选择自身最优策略：

$$\pi_e^* = \arg\max_{\pi_e} V_e\left(\pi_e, \pi_o^*(\pi_e)\right)$$

主从博弈适用于超车场景：自车（领导者）通过加速表明超车意图，他车（跟随者）预计会礼让，从而形成安全的交互结果。

### 8.3 意图感知决策

意图感知决策将他车意图的推断与自车决策紧密耦合，形成闭环：

1. **意图推断**：基于他车的历史轨迹和当前状态，推断其驾驶意图（如是否准备变道），得到意图分布 $P(intent_o)$
2. **条件规划**：对每种可能的他车意图分别规划自车最优响应
3. **期望最优化**：取期望意义下的最优策略，或采用最坏情况（Minimax）准则保证安全

$$a_e^* = \arg\max_{a_e} \sum_{intent} P(intent_o) \cdot V_e(a_e \mid intent_o)$$


## 9. 风险评估

### 9.1 碰撞时间（TTC）和碰撞余量（TTE）

**碰撞时间**（Time To Collision, TTC）是最常用的危险度量指标，定义为在当前相对速度下两车碰撞所需的时间：

$$TTC = \frac{d_{rel}}{v_{rel}} = \frac{x_{lead} - x_{ego} - l_{ego}}{v_{ego} - v_{lead}}$$

其中 $d_{rel}$ 为车辆间净距，$v_{rel} = v_{ego} - v_{lead}$ 为相对速度（仅在追赶时有意义，即 $v_{ego} > v_{lead}$）。

**碰撞余裕时间**（Time to Escape, TTE）在障碍物来自侧向时使用，量化脱离危险区域所需时间。

决策安全阈值参考：

| TTC 值 | 危险等级 | 建议决策动作 |
| --- | --- | --- |
| $TTC > 4s$ | 安全 | 保持当前状态 |
| $2s < TTC \leq 4s$ | 警告 | 轻微减速或变道评估 |
| $1s < TTC \leq 2s$ | 危险 | 立即减速，禁止变道 |
| $TTC \leq 1s$ | 紧急 | 触发 AEB |

### 9.2 责任敏感安全（RSS）在决策中的集成

RSS（Responsibility-Sensitive Safety）由 Mobileye 提出，通过数学化定义安全距离，为决策提供形式化的安全保证。

**纵向安全距离**（后车对前车的安全要求）：

$$d_{safe}^{lon} = \left[v_{rear} \cdot \rho + \frac{v_{rear}^2}{2 a_{min,rear}} - \frac{v_{front}^2}{2 a_{max,front}}\right]^+$$

其中 $\rho$ 为反应时间，$a_{min,rear}$ 为后车最大制动加速度，$a_{max,front}$ 为前车最大制动加速度，$[\cdot]^+$ 表示取正值部分。

**横向安全距离**（相邻车道间）：

$$d_{safe}^{lat} = \mu + \left[\frac{(v_{lat,1} + v_{lat,2})^2}{2 a_{lat,min}}\right]^+$$

其中 $\mu$ 为最小横向间距，$v_{lat}$ 为横向速度分量。

RSS 在决策中的集成方式：将 RSS 安全条件作为硬约束，过滤掉所有违反 RSS 的候选动作，仅在安全动作集合中进行优化选择。

### 9.3 风险地图（Risk Map）

风险地图将场景中各点的危险程度以栅格形式表示，综合考虑以下因素：

- **碰撞概率**：基于对周围障碍物轨迹的预测，计算自车到达某位置时的碰撞概率
- **历史事故数据**：路口、盲区等高风险区域有更高的先验风险值
- **交通规则违规代价**：违反交规的区域（如对向车道）被赋予高风险值

风险地图可以与运动规划无缝集成，规划算法在最小化代价的同时，自动避开高风险区域：

$$C_{path} = \int_0^L \left[ w_{risk} \cdot Risk(x(s), y(s)) + w_{len} \right] ds$$

其中 $s$ 为路径弧长参数，$Risk(\cdot)$ 为风险地图在该位置的风险值。


## 参考资料

1. Shalev-Shwartz, S., Shammah, S., & Shashua, A. (2017). On a Formal Model of Safe and Scalable Self-driving Cars. *arXiv:1708.06374*.
2. Cunningham, A. G., Galceran, E., Eustice, R. M., & Olson, E. (2015). MPDM: Multipolicy Decision-Making in Dynamic, Uncertain Environments for Autonomous Driving. *IEEE ICRA 2015*.
3. Somani, A., Ye, N., Hsu, D., & Lee, W. S. (2013). DESPOT: Online POMDP Planning with Regularization. *NeurIPS 2013*.
4. Ziebart, B. D., Maas, A., Bagnell, J. A., & Dey, A. K. (2008). Maximum Entropy Inverse Reinforcement Learning. *AAAI 2008*.
5. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
6. Fisac, J. F., Bronstein, E., Stefansson, E., Sadigh, D., Sastry, S. S., & Dragan, A. D. (2019). Hierarchical Game-Theoretic Planning for Autonomous Vehicles. *IEEE ICRA 2019*.
