# 路径规划

路径规划（Path Planning）是自动驾驶的核心模块之一，负责在感知到的环境中为车辆生成安全、高效、舒适的行驶轨迹。规划问题通常分为三个层次：全局路由、行为规划和运动规划。

## 规划层次架构

自动驾驶的规划系统通常采用层次化架构，每层专注于不同时间尺度的决策：

| 层次 | 输入 | 输出 | 时间尺度 | 代表算法 |
| --- | --- | --- | --- | --- |
| 全局路由（Routing） | 地图 + 起终点 | 路网级路径（路段序列） | 分钟级 | Dijkstra、A*、CH |
| 行为规划（Behavior） | 感知结果 + 高精地图 | 驾驶行为（变道/跟车/超车） | 秒级 | FSM、MPDM、POMDP |
| 运动规划（Motion） | 行为指令 + 障碍物 | 可执行轨迹（位置+速度+时间） | 100 ms 级 | 多项式、RRT*、MPC |

各层之间形成异步协同：全局路由提供粗粒度目标，行为规划确定驾驶动作，运动规划生成精确轨迹。


## 全局路由算法

全局路由在有向路网图 $G=(V,E)$ 上寻找从起点 $s$ 到终点 $t$ 的最优路径，边权通常为行驶时间或距离。

### Dijkstra 算法

基于贪心的最短路径算法，逐步从已访问节点集合中提取代价最小的节点：

$$d[v] = \min_{u \in \text{closed}} \left( d[u] + w(u, v) \right)$$

- **时间复杂度**：$O(V^2)$（朴素实现）或 $O(E \log V)$（优先队列）
- **优点**：保证最优解，适合静态图
- **缺点**：不利用启发信息，大规模路网效率低

### A* 算法

在 Dijkstra 基础上引入启发函数 $h(n)$ 估计当前节点到目标的代价，优先扩展"最有希望"的节点：

$$f(n) = g(n) + h(n)$$

其中 $g(n)$ 为起点到 $n$ 的实际代价，$h(n)$ 为启发估计（需满足**可接纳性**：$h(n) \leq h^*(n)$）。

常用启发函数：
- **欧氏距离**：$h(n) = \sqrt{(x_n - x_t)^2 + (y_n - y_t)^2}$（连续空间）
- **曼哈顿距离**：$h(n) = |x_n - x_t| + |y_n - y_t|$（网格空间）

**启发函数对搜索效率的影响：**

启发函数的质量直接决定 A* 的效率与正确性之间的权衡：

- 当 $h(n) = 0$ 时，A* 退化为 Dijkstra，保证最优但搜索范围最大
- 当 $h(n) = h^*(n)$（完美启发）时，A* 只扩展最优路径上的节点，效率最高
- 当 $h(n) > h^*(n)$（不可接纳）时，A* 不再保证最优解，但搜索速度更快（Weighted A*）

在道路网络中，欧氏距离是严格可接纳的启发函数（因为实际行驶距离不可能小于直线距离）。但在城市中欧氏距离与实际行驶时间相关性较弱，可改用"以最高限速行驶欧氏距离的时间"作为更紧致的启发估计。

### Contraction Hierarchies（CH）

面向超大规模路网（百万级节点，如全国导航图），Dijkstra 和 A* 的单次查询仍需数百毫秒，无法满足实时性要求。Contraction Hierarchies（Geisberger et al., 2008）通过**预处理**极大加速查询：

**预处理阶段（离线，一次性执行）：**

按节点重要度从低到高逐一"收缩"（Contraction）：删除节点 $v$，若 $v$ 的某对邻居 $(u, w)$ 通过 $v$ 的路径是唯一最短路，则添加**捷径边**（Shortcut）$u \to w$，权重为 $w(u,v) + w(v,w)$。重要度通常综合考虑以下因素：

- **边差分**（Edge Difference）：添加的捷径数 - 删除的原始边数
- **已收缩邻居数**：邻居中已被收缩的节点越多，当前节点越适合收缩
- **覆盖原始路径数**：捷径代表的原始边越多，说明该节点重要度越低

预处理后形成层次图（Hierarchy）：底层为原始路网，高层为高速公路骨干网。

**查询阶段（在线，毫秒级）：**

从起点向上搜索（沿层次图向高层扩展），从终点也向上搜索，两个搜索在顶层汇合：

$$d(s, t) = \min_{v} \left( d_{\uparrow}(s, v) + d_{\uparrow}(t, v) \right)$$

由于高层节点稀疏，双向 CH 查询通常只需扩展数千个节点，比 Dijkstra 快 **1000 倍以上**，是 Google Maps、高德地图等商业导航系统的标准后台算法。

### 城市场景路网表示

自动驾驶中的全局路由不仅需要路网连通性，还需要丰富的语义信息以支持行为规划。两种主流 HD Map 路网格式：

**Lanelet2（Poggenhans et al., IROS 2018）：**

以车道（Lanelet）为基本单元，每个 Lanelet 定义为：

- 左右边界线（折线）
- 交通规则（限速、优先级、禁止变道）
- 与相邻 Lanelet 的拓扑关系（前驱、后继、左邻、右邻）

Lanelet2 被广泛用于学术研究和欧洲自动驾驶系统，提供 C++ 和 Python 接口，支持基于车道级拓扑图的路径规划。

**Apollo HD Map（百度）：**

采用 Protobuf 格式定义道路元素，包含：

- Road（道路段）→ Section（路段分组）→ Lane（单条车道）的层次结构
- 精确的车道边界线和中心线（采样点间隔约 0.1–1 m）
- 信号灯、停止线、人行横道、限速区域的关联关系

Apollo 规划模块在车道图（Lane Graph）上执行 A* 搜索，输出全局参考线（Reference Line），供后续行为规划和运动规划使用。


## 行为规划深化

行为规划（Behavior Planning）负责在全局路由的指引下，根据实时感知结果决定当前应采取的驾驶行为（跟车、变道、超车、礼让等）。

### 场景识别与状态机

结构化道路场景可以用有限状态机（Finite State Machine，FSM）建模，每个状态对应一种驾驶模式：

```
                  ┌─────────────────────────────────────────┐
                  ▼                                         │
[跟车（Follow）] ──变道条件满足──▶ [变道准备（LCPrep）] ──完成──▶ [变道执行（LC）]
      │                                                              │
      │ 前方无车                                                    │ 完成
      ▼                                                              ▼
[自由行驶（FreeDrive）]                                     [跟车（Follow）]
      │
      │ 检测到路口
      ▼
[路口减速（Intersection）] ──通过──▶ [自由行驶]
```

主要状态及触发条件：

| 状态 | 进入条件 | 行为输出 |
| --- | --- | --- |
| 自由行驶 | 前方无障碍，车道畅通 | 保持参考速度 |
| 跟车 | 前车距离 < 安全阈值 | 智能跟车（IDM 模型） |
| 变道准备 | 目标车道满足间隙条件 | 开启转向灯，横向移动 |
| 路口决策 | 进入路口感知区域 | 减速，等待通行权 |
| 合流 | 进入汇入区 | 与主路车辆协商间隙 |
| 紧急避险 | TTC < 阈值 | AEB / 紧急制动 |

FSM 的优点是逻辑清晰、行为可预期，但难以处理状态之间的平滑过渡和模糊边界（例如"跟车"和"路口减速"同时触发时的优先级）。

### 代价地图（Cost Map）

行为规划通常在**代价地图**（Cost Map）上进行候选轨迹评分，代价地图将多类约束叠加为标量场：

$$C(x, y) = w_{\text{obs}} \cdot C_{\text{obs}}(x,y) + w_{\text{lane}} \cdot C_{\text{lane}}(x,y) + w_{\text{vel}} \cdot C_{\text{vel}}(x,y) + \cdots$$

**障碍物代价 $C_{\text{obs}}$：**

对感知到的障碍物（含预测未来位置）周围设置代价场，通常用高斯分布：

$$C_{\text{obs}}(x,y) = \sum_{i} A_i \exp\!\left(-\frac{(x - x_i)^2}{2\sigma_{x,i}^2} - \frac{(y - y_i)^2}{2\sigma_{y,i}^2}\right)$$

障碍物中心代价最高，向外衰减，$A_i$ 随障碍物尺寸和类别调整（行人代价高于固定物体）。

**车道中心线代价 $C_{\text{lane}}$：**

偏离参考车道中心线的横向距离代价，保证车辆在正常行驶时保持居中：

$$C_{\text{lane}}(x,y) = \left(\frac{d_\perp(x,y,\ \text{centerline})}{\text{LaneWidth}/2}\right)^2$$

**速度一致性代价 $C_{\text{vel}}$：**

惩罚与周围交通流速度的偏差（过快或过慢都会增大风险）：

$$C_{\text{vel}}(v) = \left(\frac{v - v_{\text{flow}}}{v_{\text{flow}}}\right)^2$$

### 基于搜索的行为规划：MCTS 用于路口决策

路口决策（多车博弈）的复杂性超出了 FSM 的建模能力。蒙特卡洛树搜索（Monte Carlo Tree Search，MCTS）通过在线仿真模拟未来场景演化，为路口决策提供理论支撑：

**MCTS 基本步骤（迭代执行）：**

1. **选择（Selection）**：从当前节点按 UCB1 策略向下遍历到叶节点：
   $$a^* = \arg\max_a \left[ Q(s,a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}} \right]$$
2. **扩展（Expansion）**：在叶节点处扩展一个未访问的行为选项（直行/左转/右转/等待）
3. **模拟（Simulation）**：从扩展节点运行随机 Rollout 策略，估计其他智能体的可能反应
4. **反向传播（Backpropagation）**：将模拟结果（成功通行/冲突）反馈更新所有祖先节点的 $Q$ 值和 $N$ 值

MCTS 的优点是不需要预先枚举所有可能场景，可以在有限计算预算内对最有希望的行为进行重点模拟，适合无保护左转、无信号路口等高不确定性场景。


## 局部运动规划

运动规划需要在连续空间中，在障碍物、车辆动力学、舒适性等约束下生成可执行轨迹。

### 基于采样的方法

适合高维构型空间，无需显式建模障碍物形状。

**RRT（快速随机扩展树，Rapidly-exploring Random Tree）：**

```
初始化: T ← {x_start}
循环:
    x_rand ← 随机采样（以一定概率偏向目标）
    x_near ← T 中离 x_rand 最近的节点
    x_new  ← Steer(x_near, x_rand, δ)   // 向 x_rand 延伸步长 δ
    if CollisionFree(x_near, x_new):
        T.add(x_new)
        if Reached(x_new, x_goal): return path
```

**RRT*（渐近最优 RRT）：**

在 RRT 基础上增加 **rewire** 步骤——对 $x_{\text{new}}$ 附近节点检查是否可以通过 $x_{\text{new}}$ 获得更优路径，并重新连接。理论上样本数趋于无穷时，RRT* 收敛到全局最优解。

**适用场景**：停车场泊车、窄道通行、构型空间维度高的场景。

### Frenet 坐标系

高速公路和结构化道路规划常在 **Frenet 坐标系**下进行，将运动分解为：

- **纵向 $s$**：沿参考线（车道中心线）方向的弧长
- **横向 $d$**：垂直于参考线方向的偏移量

优点：自然地将"行进方向控制"与"横向保持"解耦，简化约束表达。在 Frenet 坐标系下，变道目标即为 $d$ 从当前车道切换到相邻车道。

### 多项式轨迹（JMT）

最小化加加速度（Jerk Minimizing Trajectory）的五次多项式，是 Frenet 坐标系下的经典方法：

$$s(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$

6 个系数由边界条件唯一确定：

$$\begin{bmatrix} s_0 \\ \dot{s}_0 \\ \ddot{s}_0 \end{bmatrix}, \quad \begin{bmatrix} s_T \\ \dot{s}_T \\ \ddot{s}_T \end{bmatrix}$$

同样的方法对横向 $d(t)$ 做独立多项式规划，再合并为完整轨迹。加加速度（Jerk）代价：

$$\mathcal{C}_j = \int_0^T \left( \dddot{s}^2 + \dddot{d}^2 \right) dt$$

### 基于优化的轨迹规划

将轨迹规划建模为约束优化问题：

$$\min_{\xi} \int_0^T \left[ w_j \dddot{s}^2 + w_v (v - v_{\text{ref}})^2 + w_d d^2 \right] dt$$

约束条件：
- $v_{\min} \leq v(t) \leq v_{\max}$（速度限制）
- $a_{\min} \leq a(t) \leq a_{\max}$（舒适性/动力学）
- 不与障碍物碰撞

常用求解器：OSQP（二次规划）、Ipopt（非线性规划）。

### 基于势场的方法

将规划空间建模为势能场：

$$U_{\text{total}} = U_{\text{att}}(q) + U_{\text{rep}}(q)$$

- **引力场**：$U_{\text{att}} = \frac{1}{2} \xi \|q - q_{\text{goal}}\|^2$，目标点产生吸引力
- **斥力场**：$U_{\text{rep}} = \frac{1}{2} \eta \left(\frac{1}{\rho} - \frac{1}{\rho_0}\right)^2$，障碍物产生排斥力

车辆沿负梯度方向运动：$F = -\nabla U$。

- **优点**：实时性好，实现简单
- **缺点**：易陷入**局部极小值**（障碍物之间的势谷），不保证到达目标


## 参数化轨迹优化

参数化方法将轨迹表示为有限参数集合（多项式系数、控制点），将无限维的函数优化转化为有限维的参数优化，是工程实现中最常用的轨迹生成框架。

### Minimum Jerk / Minimum Snap 轨迹

最小化 Jerk（加加速度）和 Snap（加加加速度）是常用的轨迹平滑目标，分别对应四阶和六阶积分代价：

$$\min_{c} \int_0^T \left[\dddot{p}(t)\right]^2 dt \quad \text{（Minimum Jerk）}$$

$$\min_{c} \int_0^T \left[p^{(4)}(t)\right]^2 dt \quad \text{（Minimum Snap）}$$

对于分段多项式轨迹（分为 $M$ 段，每段为 $n$ 次多项式），最小化代价等价于二次规划（QP）问题。以 Minimum Snap 为例，设第 $m$ 段的系数向量为 $c_m \in \mathbb{R}^{n+1}$，则：

$$\mathbf{c}^* = \arg\min_{\mathbf{c}} \mathbf{c}^\top Q \mathbf{c} + \lambda \|\mathbf{c}\|^2$$

其中 $Q$ 为由微分算子构造的半正定代价矩阵：

$$Q_{ij} = \int_0^T \frac{d^k p_i(t)}{dt^k} \cdot \frac{d^k p_j(t)}{dt^k} \, dt$$

$k=3$（Jerk）或 $k=4$（Snap），正则项 $\lambda \|\mathbf{c}\|^2$ 用于防止过拟合和数值病态。

约束矩阵 $M$ 将系数向量 $\mathbf{c}$ 映射到端点状态（位置、速度、加速度），保证轨迹满足初末状态约束和连接点处的平滑性（Continuity）约束：

$$M \mathbf{c} = \mathbf{b}$$

其中 $\mathbf{b}$ 包含所有边界条件。将约束代入 QP 可得显式解，无需迭代求解器，适合实时计算。

### CHOMP（梯度下降轨迹优化）

CHOMP（Covariant Hamiltonian Optimization for Motion Planning，Ratliff et al., 2009）将轨迹表示为离散时间序列 $\xi = [q_1, q_2, \ldots, q_T]$，通过梯度下降最小化：

$$\mathcal{U}(\xi) = \mathcal{F}_{\text{smooth}}(\xi) + \lambda \mathcal{C}_{\text{obs}}(\xi)$$

**平滑项：**

$$\mathcal{F}_{\text{smooth}}(\xi) = \frac{1}{2} \xi^\top A^\top A \xi$$

其中 $A$ 为有限差分矩阵，$A^\top A$ 近似轨迹的曲率积分，保证生成轨迹的平滑性。

**障碍物代价项：**

$$\mathcal{C}_{\text{obs}}(\xi) = \int_0^1 \sum_{i} c_{\text{obs}}\!\left(q_i(\tau)\right) \|\dot{q}_i(\tau)\| \, d\tau$$

其中 $c_{\text{obs}}$ 为障碍物代价函数（由距离场计算），$\|\dot{q}\|$ 为速度项（保证代价对时间参数化不变）。

CHOMP 的关键创新是使用**协变梯度**（Covariant Gradient）而非欧氏梯度更新轨迹，保证梯度更新方向在轨迹空间中保持几何意义，避免普通梯度下降因不同维度量纲差异导致的收敛问题。

### 贝塞尔曲线与 B 样条

**贝塞尔曲线（Bezier Curve）：**

$n$ 次贝塞尔曲线由 $n+1$ 个控制点 $P_0, P_1, \ldots, P_n$ 定义：

$$B(t) = \sum_{i=0}^n \binom{n}{i} (1-t)^{n-i} t^i P_i, \quad t \in [0, 1]$$

性质：曲线通过端点 $P_0$ 和 $P_n$，但不一定通过中间控制点（仅被"吸引"）。三次贝塞尔曲线（$n=3$）常用于路径生成，因其形状直观、计算简单。

**B 样条（B-Spline）：**

B 样条是贝塞尔曲线的推广，通过节点向量（Knot Vector）$\{t_0, t_1, \ldots, t_{m}\}$ 实现局部控制：

$$C(t) = \sum_{i=0}^n N_{i,k}(t) P_i$$

其中 $N_{i,k}(t)$ 为 $k$ 阶 B 样条基函数（由 Cox-de Boor 递推公式计算）。B 样条的关键优点是**局部修改性**：移动某个控制点 $P_i$ 只影响曲线在其对应基函数支撑区间内的形状，不影响其他部分。这一特性在轨迹实时调整中非常有用——当障碍物变化时，只需修改局部控制点，无需重新规划整条轨迹。

在自动驾驶中，均匀三次 B 样条（Uniform Cubic B-Spline）常用于路径平滑，配合等间隔控制点可得到 $C^2$ 连续的光滑曲线（位置、速度、加速度均连续）。


## ST 图速度规划

运动规划通常先规划路径形状，再规划速度剖面（Speed Profile）。ST 图方法将速度规划转化为二维图中的路径搜索问题，兼顾直觉性和计算效率。

### ST 图定义

**ST 图（Speed-Time Graph，速度-时间图）：**

- **横轴**：时间 $t$，范围通常为规划时域 $[0, T]$（$T = 8$ 秒）
- **纵轴**：纵向行驶距离 $s$，范围为 $[0, S_{\max}]$（$S_{\max}$ 为最大规划距离，如 200 m）
- **障碍物表示**：每个动态障碍物根据其预测轨迹在 ST 图中映射为一个**矩形禁区**，矩形的时间范围对应障碍物占据路段的时间窗口，纵向范围对应障碍物在参考线上的投影区间

```
s (m)
▲
|         ████ 障碍物 B（慢车）
|   ██████████
|         ████████ 障碍物 A（过路行人）
|   ████████
|
└─────────────────────────────▶ t (s)
0    2    4    6    8
```

速度规划目标是在 ST 图中从 $(t=0, s=0)$ 到 $(t=T, s=s_{\text{goal}})$ 找一条**平滑单调递增曲线**，完全避开所有矩形禁区。

曲线斜率 $\frac{ds}{dt} = v(t)$ 即为瞬时速度，斜率变化率 $\frac{d^2s}{dt^2} = a(t)$ 为加速度，通过约束斜率范围可同时控制速度和加速度边界。

### DP 粗规划

由于 ST 图中的路径约束是非凸的（障碍物矩形将可行域分割为多个不连通区域），通常先用**动态规划（Dynamic Programming）**进行粗规划，确定越过每个障碍物的方式（"超过"还是"等待"）：

**DP 状态定义：** 在 ST 图上等间隔划分格点，状态为 $(t_i, s_j)$，状态转移代价包含：

$$\text{cost}(s_j, t_i \to s_{j'}, t_{i+1}) = w_v (v - v_{\text{ref}})^2 + w_a a^2 + w_j \text{jerk}^2 + \infty \cdot \mathbb{1}[\text{碰撞}]$$

DP 输出粗粒度速度曲线，确定对每个障碍物的处理策略（先行或等待）。

### QP 细化

在 DP 确定的拓扑结构（绕障顺序）基础上，用**二次规划（Quadratic Programming）**对速度曲线进行精细平滑：

$$\min_{s_0, s_1, \ldots, s_N} \sum_{i} \left[ w_v (v_i - v_{\text{ref}})^2 + w_a a_i^2 + w_j j_i^2 \right]$$

约束条件（线性化为 QP 标准形式）：
- $0 \leq v_i \leq v_{\max,i}$（逐时刻速度上界，考虑限速和障碍物 TTC）
- $a_{\min} \leq a_i \leq a_{\max}$（加速度舒适性约束）
- $s_i \notin [s_{\text{obs},\min}(t_i),\ s_{\text{obs},\max}(t_i)]$（障碍物禁区约束，线性化为半平面约束）

QP 求解时间通常在 10–30 ms，满足实时规划要求。

### 速度约束的来源

| 约束类型 | 公式 | 来源 |
| --- | --- | --- |
| 限速约束 | $v(t) \leq v_{\text{limit}}(s(t))$ | HD Map 车道限速标注 |
| 舒适加速度 | $-3.0 \leq a(t) \leq 2.0\ (\text{m/s}^2)$ | 乘员舒适性标准 |
| 紧急制动 | $a_{\min} = -8.0\ \text{m/s}^2$（硬约束） | 车辆动力学极限 |
| TTC 约束 | $\text{TTC}(t) = \frac{s_{\text{obs}}(t) - s_{\text{ego}}(t)}{v_{\text{ego}} - v_{\text{obs}}} \geq 3\ \text{s}$ | 安全跟车距离 |
| 曲率限速 | $v(t) \leq \sqrt{a_{\text{lat,max}} / \kappa(s)}$ | 横向加速度 < 0.3g |

### 百度 Apollo EM Planner

Apollo EM Planner（Fan et al., ICRA 2018）是工业界最具影响力的开源规划框架之一，其速度规划模块采用 DP + QP 两阶段方法：

**E 步（Expectation）：** 在 SL 图（路径）和 ST 图（速度）上交替进行 DP 搜索，得到粗粒度路径和速度方案。

**M 步（Maximization）：** 以 DP 结果为初值，在各自的 QP 问题中进行精细优化，输出平滑可执行轨迹。

两步交替迭代（类 EM 算法），通常 2–3 次迭代即可收敛。整个规划周期（含感知输入处理）控制在 100 ms 以内。


## 动态障碍物处理

动态障碍物（行人、车辆）使规划问题变为时空规划（Spatio-Temporal Planning）：

- **预测+规划（两步法）**：先用预测模块估计障碍物未来轨迹，再规划避让路径。解耦简单但预测误差会导致保守规划
- **联合时空规划**：在时空空间（$s, d, t$）中联合搜索，可以协商让行
- **ORCA（Optimal Reciprocal Collision Avoidance）**：多智能体实时避障算法，假设其他智能体也在避让，适合低速行人密集场景


## 泊车规划

泊车是低速构型空间规划的典型场景，需要支持倒车。

- **Reeds-Shepp 路径**：允许前进和后退的最短路径解析解，适合已知障碍物的无约束最优泊车轨迹
- **Hybrid A***：在连续（$x, y, \theta$）构型空间中进行离散搜索，结合车辆运动约束（最小转弯半径）找到可执行路径，斯坦福 DARPA 团队提出，广泛用于自动泊车
- **基于优化的泊车**：将泊车规划建模为非线性优化，适合更复杂的停车位形状


## 实时性保障

自动驾驶规划系统对延时有严格要求：规划输出必须在感知数据到达后 100 ms 内完成，否则控制层将因数据过时而采用降级策略。

### 规划周期与计算预算

标准规划频率为 **10 Hz**（每 100 ms 输出一次轨迹），计算预算分配参考：

| 子模块 | 典型耗时 | 备注 |
| --- | --- | --- |
| 参考线生成 | 5–10 ms | 从 HD Map 提取并平滑 |
| 行为决策 | 5–15 ms | FSM 或 MCTS |
| 路径 DP | 10–20 ms | SL 图粗搜索 |
| 路径 QP | 5–15 ms | 样条优化 |
| 速度 DP | 10–20 ms | ST 图粗搜索 |
| 速度 QP | 5–15 ms | 速度曲线平滑 |
| 碰撞检测与后处理 | 5–10 ms | 多帧验证 |
| **合计** | **~70–105 ms** | 余量用于系统调度开销 |

超出预算的帧将触发**降级策略**（见下文）。

### 热启动（Warm Start）

规划是一个时序连续的过程，相邻两帧的最优轨迹通常差异很小。热启动利用上一帧的规划结果作为当前帧优化的初始值：

**路径规划热启动：**

将上一帧的最优路径在时间上平移 $\Delta t$（当前规划周期时长），截取 $[s_{\text{current}}, s_{\text{horizon}}]$ 段作为新一轮 QP 的初值。由于 QP 是凸优化，初值主要影响收敛速度，热启动通常使迭代次数从 50 次减少到 10 次以内。

**速度规划热启动：**

类似地，将上一帧 ST 图中的速度曲线平移 $\Delta t$ 作为新一帧 DP 的初始粗解，跳过 DP 的完整搜索，直接进入 QP 精细优化。在环境未发生剧烈变化时，热启动可节省约 30–50% 的规划时间。

### 降级策略

当规划模块因超时、求解失败或感知数据异常无法输出有效轨迹时，必须有**降级（Fallback）策略**保证车辆安全：

```
规划执行流程（含降级）：

┌─────────────────────────────────────────────────────┐
│ 尝试完整规划（DP + QP，~80 ms）                      │
│   ├── 成功 → 输出新轨迹                              │
│   └── 失败/超时                                      │
│         ├── 使用上帧轨迹继续执行（轨迹时移，~1 ms）  │
│         │     └── 连续 N 帧失败（N=3）→              │
│         └── 请求紧急停车（Comfortable Stop）          │
│               └── 紧急停车失败 → AEB 介入            │
└─────────────────────────────────────────────────────┘
```

各级降级策略说明：

| 降级级别 | 触发条件 | 响应动作 | 持续时限 |
| --- | --- | --- | --- |
| Level 1：使用缓存轨迹 | 单帧规划超时 | 时移上帧轨迹，继续执行 | 最多 3 帧（300 ms） |
| Level 2：舒适停车 | 连续多帧规划失败 | 沿当前车道安全减速停车 | 直到恢复正常 |
| Level 3：紧急制动 | 舒适停车路径被障碍物阻挡 | AEB 介入，最大制动 | 单次响应 |
| Level 4：接管请求 | 长时间无法恢复规划 | 向驾驶员发出接管警报 | 取决于 ODD |

### 计算资源分配

在嵌入式自动驾驶计算平台（如英伟达 DRIVE AGX、地平线 Journey 5）上，规划模块的计算资源分配通常遵循以下原则：

**CPU 核心绑定（CPU Affinity）：**

规划线程绑定到专用物理核心（避免 OS 调度导致的不确定延时），通常为：

```bash
# 将规划进程绑定到 CPU 2–3 核
taskset -c 2,3 planning_node
```

**优先级设置：**

规划线程设置为实时优先级（SCHED_FIFO，priority = 80），高于感知和预测模块，保证在计算平台负载高峰时规划任务不被抢占。

**内存预分配：**

规划模块在初始化时预分配所有中间变量的内存，避免运行时动态 malloc 导致的延时不确定性。ST 图格点数组、QP 矩阵等固定大小的数据结构均在启动时分配完毕。

**并行化策略：**

路径规划和速度规划在 DP 阶段可并行执行（路径 DP 与速度 DP 在前一帧结果基础上同时启动），QP 阶段则串行执行（速度 QP 依赖路径 QP 的输出）。典型的并行化实现可将整体规划时间压缩 20–30%。


## 规划系统的核心挑战

| 挑战 | 描述 | 主要应对思路 |
| --- | --- | --- |
| 实时性 | 需在 100 ms 内完成全部规划 | 暖启动、分层规划、并行计算 |
| 动态环境 | 障碍物持续运动，地图随时变化 | 滚动时域规划（类 MPC 思路） |
| 不确定性 | 感知误差、行人意图不确定 | 概率规划、鲁棒性优化 |
| 舒适性 | 避免急加减速、频繁变道 | Jerk 约束、轨迹平滑 |
| 交通法规 | 不得违反信号灯、限速、优先权规则 | 将规则编码为硬约束或高惩罚代价 |
| 长尾场景 | 罕见危险场景难以覆盖 | 仿真生成、对抗测试、数据积累 |


## 参考资料

1. S. Werling et al. Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame. ICRA, 2010.
2. D. Dolgov et al. Path Planning for Autonomous Vehicles in Unknown Semi-structured Environments. IJRR, 2010.
3. S. M. LaValle and J. J. Kuffner. Randomized Kinodynamic Planning. IJRR, 2001.
4. W. Ziegler et al. Trajectory Planning for Bertha — A Local, Continuous Method. IEEE Intelligent Vehicles, 2014.
5. M. Pivtoraiko and A. Kelly. Efficient Constrained Path Planning via Search in State Lattices. ISER, 2008.
6. R. Geisberger et al. Contraction Hierarchies: Faster and Simpler Hierarchical Routing in Road Networks. WEA, 2008.
7. F. Poggenhans et al. Lanelet2: A High-Definition Map Framework for the Future of Automated Driving. ITSC, 2018.
8. H. Fan et al. Baidu Apollo EM Motion Planner. arXiv, 2018.
9. N. Ratliff et al. CHOMP: Gradient Optimization Techniques for Efficient Motion Planning. ICRA, 2009.
10. D. Mellinger and V. Kumar. Minimum Snap Trajectory Generation and Control for Quadrotors. ICRA, 2011.
