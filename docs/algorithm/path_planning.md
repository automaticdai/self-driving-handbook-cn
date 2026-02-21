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

### 道路网络专用加速算法

大规模路网（百万节点）需要专用算法：

- **Contraction Hierarchies (CH)**：预处理时按重要度收缩节点，构建多级快速路径。查询速度比 Dijkstra 快 1000 倍以上，是导航 APP（Google Maps、高德）的标准后台算法
- **ALT**：基于预计算的地标节点（Landmark）和三角不等式加速 A*，适合大图多查询场景


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


## 速度规划（ST 图）

运动规划通常先规划路径形状，再规划速度剖面（Speed Profile）。

**ST 图（Speed-Time Graph）：**

以时间 $t$ 为横轴、纵向行驶距离 $s$ 为纵轴，障碍物在 ST 图中对应矩形禁区。速度规划目标是在 ST 图中从 $(0, 0)$ 到 $(T, s_{\text{goal}})$ 找一条平滑曲线，避开所有障碍区。

斜率 $\frac{ds}{dt}$ 即为瞬时速度，斜率变化率为加速度。

**约束条件：**
- $0 \leq \dot{s}(t) \leq v_{\max}$（速度上下界）
- $a_{\min} \leq \ddot{s}(t) \leq a_{\max}$（加减速舒适性）
- 不进入障碍物对应的 ST 矩形区域（安全约束）


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
