# 局部轨迹与速度规划

本页关注"给定行为意图后，如何生成可执行、可控、可舒适的轨迹"，涵盖坐标系选择、轨迹生成方法与速度规划。

---

## 1. 坐标系基础

### 1.1 笛卡尔坐标系

全局 $(x, y, \theta)$ 坐标系，直观但在结构化道路场景中不便于分解横纵向约束。

### 1.2 Frenet 坐标系

Frenet 坐标以参考路径为基准，将运动分解为：

- **纵向（$s$）**：沿参考路径的弧长位移
- **横向（$l$）**：垂直于参考路径的偏移距离

**坐标变换：**

已知参考路径 $\{x_r(s), y_r(s), \theta_r(s)\}$，车辆笛卡尔坐标 $(x, y)$ 的 Frenet 坐标：

$$s = \arg\min_s \sqrt{(x - x_r(s))^2 + (y - y_r(s))^2}$$

$$l = (x - x_r(s))\sin\theta_r(s) - (y - y_r(s))\cos\theta_r(s)$$

**Frenet 运动方程：**

$$\dot{s} = \frac{v\cos(e_\psi)}{1 - l\kappa_r}$$

$$\dot{l} = v\sin(e_\psi)$$

其中 $\kappa_r$ 为参考路径曲率，$e_\psi = \psi - \psi_r$ 为航向误差。

!!! warning "Frenet 坐标局限"
    在大曲率弯道（$l\kappa_r \to 1$）或路径不连续（路口、绕行）场景，Frenet 坐标变换会退化。工程上需要切换回笛卡尔坐标或做特殊处理。

---

## 2. 轨迹生成方法

### 2.1 Lattice 采样规划

在 Frenet 空间生成候选轨迹族：

```
横向轨迹族：
  l_end ∈ {-3.5m, -1.75m, 0, +1.75m, +3.5m}（相对当前位置）
  T_end ∈ {3s, 5s, 8s}（规划时域）

纵向轨迹族：
  v_end ∈ {0, 5, 10, 15, ..., v_max} km/h
  s_end = f(v_end, T_end)
```

对每条候选轨迹计算代价并选取最优：

$$J_{\text{traj}} = w_{\text{safety}} \cdot C_{\text{safety}} + w_{\text{comfort}} \cdot C_{\text{comfort}} + w_{\text{efficiency}} \cdot C_{\text{efficiency}}$$

### 2.2 多项式轨迹

用多项式参数化横向和纵向运动，在 Frenet 坐标中常用五次多项式：

$$l(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$

**端点约束（6 个约束恰好确定 6 个系数）：**

$$l(0) = l_0, \quad \dot{l}(0) = \dot{l}_0, \quad \ddot{l}(0) = \ddot{l}_0$$

$$l(T) = l_f, \quad \dot{l}(T) = \dot{l}_f, \quad \ddot{l}(T) = \ddot{l}_f$$

**最小 jerk 轨迹**（三次多项式）：最小化 $\int_0^T \dddot{l}^2 \, dt$，数学上导出五次多项式形式。

### 2.3 贝塞尔曲线与 B 样条

| 方法 | 优点 | 局限 |
| --- | --- | --- |
| 贝塞尔曲线 | 直观控制点操作，凸包性质 | 局部修改影响全局形状 |
| B 样条 | 局部支撑，调整单段不影响全局 | 参数化复杂 |
| 分段多项式 | 灵活约束，工程常用 | 分段连接处需保证 $C^2$ 连续 |

### 2.4 基于优化的轨迹生成

将轨迹生成表述为优化问题（QP 或 NLP）：

**目标函数：**

$$\min_{\mathbf{u}} \sum_{k=0}^{N} \left[ \|\mathbf{x}_k - \mathbf{x}_{\text{ref}}\|_Q^2 + \|\mathbf{u}_k\|_R^2 + \|\Delta\mathbf{u}_k\|_P^2 \right]$$

**约束（硬约束）：**

$$\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k) \quad \text{（动力学）}$$

$$|\delta_k| \leq \delta_{\max}, \quad |a_k| \leq a_{\max} \quad \text{（执行器限制）}$$

$$d(\mathbf{x}_k, \text{obstacles}) \geq d_{\text{safe}} \quad \text{（碰撞避免）}$$

---

## 3. S-T 图速度规划

### 3.1 S-T 图定义

S-T 图以时间为横轴、路径弧长为纵轴，将动态障碍物表示为时空占用区域：

```
s（路径弧长）
↑
│        /forbidden/
│      ██████████
│    ██████████
│  ██████████       ← 障碍物 A（前车）
│
│          ████████ ← 障碍物 B（横穿）
│
└──────────────────→ t（时间）
```

规划目标：找到一条从 $(0, s_0)$ 到 $(T, s_{\text{end}})$ 的平滑曲线，避开所有禁止区域。

### 3.2 DP 粗搜索

在 S-T 图上进行动态规划，离散化时间和速度状态：

```python
# 伪代码
for t in time_steps:
    for v in velocity_states:
        s = integrate(v, dt)
        cost = J_prev + g(v, v_prev) + h(s)  # g: jerk代价, h: 约束代价
        if not in_forbidden_region(s, t):
            dp[t][v] = min(dp[t][v], cost)
```

### 3.3 QP 速度优化

在 DP 粗解基础上，用 QP 求解平滑速度曲线：

**变量：** $s_0, s_1, ..., s_N$（各时刻弧长）

**目标：** 最小化 jerk

$$\min \sum_{k=0}^{N-2} \left(\frac{s_{k+2} - 2s_{k+1} + s_k}{\Delta t^2} - a_{k+1}\right)^2$$

**约束：**

- 速度边界：$0 \leq \frac{s_{k+1}-s_k}{\Delta t} \leq v_{\max}$
- 加速度边界：$a_{\min} \leq \frac{s_{k+1}-2s_k+s_{k-1}}{\Delta t^2} \leq a_{\max}$
- 障碍物时空间隔：$s_k \notin \text{forbidden}(t_k)$

---

## 4. 约束体系

### 4.1 硬约束 vs 软约束

| 类型 | 处理方式 | 典型场景 |
| --- | --- | --- |
| 硬约束 | 必须满足，违反则拒绝解 | 碰撞避免、执行器限幅 |
| 软约束（惩罚函数） | 违反则增加代价 | 舒适性、车道中心对齐 |
| 软约束（松弛变量） | 引入 $\epsilon \geq 0$ 松弛 | 无可行解时允许轻微违反 |

```
碰撞约束（软化示例）：
  d(x, obstacle) + ε ≥ d_safe
  ε ≥ 0, 目标函数加入 w_slack · ε²
  → 不可行时允许小量违约，但代价极大，仅用于数值鲁棒性
```

### 4.2 完整约束集

```
[车辆动力学]
  |a| ≤ a_max, |jerk| ≤ jerk_max
  |δ| ≤ δ_max, |δ_rate| ≤ δ_rate_max

[碰撞避免]
  与所有障碍物保持安全距离

[交通规则]
  v ≤ v_limit（限速）
  停止线前减速停车
  禁止越实线

[舒适性]
  |ay| ≤ ay_max（侧向加速度）
```

---

## 5. 在线求解器选型

| 求解器 | 类型 | 特点 | 适用 |
| --- | --- | --- | --- |
| OSQP | QP | 无矩阵分解，热启动快 | 速度规划、轨迹优化 |
| qpOASES | QP | 精确有效集方法，稳定 | 控制层 MPC |
| IPOPT | NLP | 内点法，处理非线性约束 | 非线性轨迹优化 |
| ACADO | NLP/QP | 适合嵌入式实时 MPC | 低延迟控制 |

!!! tip "热启动的价值"
    相邻规划周期的解相似，利用上一周期解作为热启动（Warm Start）可将 QP 求解时间降低 3–10 倍，是实时规划的关键优化。

---

## 6. 实时性与降级

### 6.1 计算时间监控

```python
t_start = clock()
trajectory = planner.solve(state, obstacles)
t_elapsed = clock() - t_start

if t_elapsed > budget_ms:
    # 求解超时
    trajectory = fallback_trajectory  # 上周期安全轨迹
    alert("planning_timeout")
```

### 6.2 降级策略层级

```
正常：全约束优化轨迹（高舒适性，最优性）
    ↓ 超时或求解失败
热启动快速解：减少迭代次数
    ↓ 仍失败
上周期安全轨迹延续（最多 0.5 秒）
    ↓ 持续无解
最小风险轨迹：保持车道，逐步减速
    ↓ 极端情况
MRC：靠边停车
```

---

## 7. 验证场景集设计

| 场景类型 | 描述 | 验证重点 |
| --- | --- | --- |
| 急 cut-in | 前方突然插入低速车辆 | 速度规划快速响应，无碰撞 |
| 鬼探头 | 路边突然出现行人 | 紧急制动轨迹生成 |
| 前车急刹 | 前车 TTC < 2 s 急制动 | AEB 级别轨迹 |
| 静态障碍绕行 | 路中有锥桶或停车 | 横向轨迹绕行安全性 |
| 多障碍物场景 | 前后左右多车并行 | 组合约束下的可行解 |
| 高曲率弯道 | 曲率 > 0.05 m⁻¹ | 速度自动降低，轨迹可跟踪 |
| 隧道低速跟车 | 视觉受限、窄道 | 保守策略、更大安全余量 |

---

## 8. 输出接口规范

轨迹输出建议包含以下字段：

```yaml
trajectory:
  header:
    timestamp: 1234567890.123
    frame_id: "map"
    valid_duration: 5.0  # 秒
  points:
    - s: 0.0
      l: 0.0
      x: 100.0
      y: 200.0
      theta: 1.57
      v: 8.33      # m/s
      a: 0.0       # m/s²
      kappa: 0.01  # 曲率
      timestamp_offset: 0.0
    - ...
  confidence: 0.95
  fallback_type: "KEEP_LANE"   # 无解时的降级策略标识
  failure_code: 0              # 0=正常，非0=错误码
```
