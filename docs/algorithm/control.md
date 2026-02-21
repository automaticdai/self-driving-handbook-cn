# 控制算法

控制（Control）模块是自动驾驶软件栈的最后一环，负责将规划模块输出的期望轨迹转化为具体的车辆执行指令（转向角、油门开度、制动力），并通过线控系统作用于车辆。控制模块对实时性和精度要求极高，通常以 100 Hz 频率运行。


## 控制问题定义

**输入：**
- 期望轨迹：$\{(x_d(t),\ y_d(t),\ v_d(t),\ \psi_d(t))\}$（参考位置、速度、航向）
- 当前车辆状态：$(x,\ y,\ v,\ \psi,\ \delta)$（实际位置、速度、航向、前轮转角）

**输出：**
- 横向控制指令：前轮转角 $\delta_{\text{cmd}}$（或转向力矩）
- 纵向控制指令：加速度 $a_{\text{cmd}}$（油门/制动百分比）

**控制目标：**
$$\min_{u} \left[ e_y^2 + e_\psi^2 + (v - v_d)^2 + w_j \dddot{s}^2 \right]$$

其中 $e_y$ 为横向偏差，$e_\psi$ 为航向偏差，最后一项为加加速度舒适性代价。


## 车辆动力学模型

控制器设计基于车辆模型，常用两种单车（Bicycle）模型：

### 运动学单车模型（低速适用）

忽略轮胎侧偏力，适用于低速（< 30 km/h）和小曲率场景：

$$\dot{x} = v \cos\psi$$
$$\dot{y} = v \sin\psi$$
$$\dot{\psi} = \frac{v \tan\delta}{L}$$

其中 $L$ 为前后轴距，$\delta$ 为前轮转角。前轮转角与方向盘转角之间有固定传动比关系。

### 动力学单车模型（高速适用）

在高速或急转弯场景，轮胎侧偏角（Slip Angle）$\alpha$ 不可忽略，需引入轮胎侧向力：

$$m\ddot{y}_{\text{body}} + mv\dot{\psi} = F_{yf} + F_{yr}$$
$$I_z \ddot{\psi} = l_f F_{yf} - l_r F_{yr}$$

线性轮胎模型（Fiala 模型小角度近似）：
$$F_{yf} \approx -C_{af} \alpha_f, \quad F_{yr} \approx -C_{ar} \alpha_r$$

其中 $C_{af},\ C_{ar}$ 为前后轮侧偏刚度（需通过车辆实测标定）。


## 车辆运动学模型补充

### 阿克曼转向几何（Ackermann Steering Geometry）

真实车辆有两个前轮，转弯时内外轮转角不同。阿克曼几何确保四轮均以同一瞬时圆心为中心转动，避免轮胎刮蹭：

$$\cot\delta_o - \cot\delta_i = \frac{w}{L}$$

其中 $\delta_o$、$\delta_i$ 分别为外轮和内轮转角，$w$ 为轮距，$L$ 为轴距。单车模型以等效前轮转角 $\delta$ 近似：

$$\frac{1}{R} = \frac{\tan\delta}{L}$$

$R$ 为转弯半径。阿克曼关系在底层转向机构中由梯形连杆实现，上层控制器通常只需下发等效单车转角指令。

### 轮胎侧偏角详解

**侧偏角**（Slip Angle）$\alpha$ 定义为轮胎实际行驶方向与车轮平面方向之间的夹角，是产生侧向力的根本原因：

前轮侧偏角：
$$\alpha_f = \delta - \arctan\!\left(\frac{\dot{y} + l_f \dot{\psi}}{v_x}\right) \approx \delta - \frac{\dot{y} + l_f \dot{\psi}}{v_x}$$

后轮侧偏角：
$$\alpha_r = -\arctan\!\left(\frac{\dot{y} - l_r \dot{\psi}}{v_x}\right) \approx -\frac{\dot{y} - l_r \dot{\psi}}{v_x}$$

其中 $v_x$ 为纵向速度，$\dot{y}$ 为车体坐标系横向速度，$l_f$、$l_r$ 分别为质心到前后轴距离。

### 侧偏刚度与轮胎特性

侧偏刚度 $C_{\alpha f}$、$C_{\alpha r}$（单位：N/rad）描述轮胎在小侧偏角范围内的线性响应：

$$F_{yf} = -C_{\alpha f} \cdot \alpha_f, \quad F_{yr} = -C_{\alpha r} \cdot \alpha_r$$

| 轮胎类型 | 前轮 $C_{\alpha f}$ (N/rad) | 后轮 $C_{\alpha r}$ (N/rad) |
| --- | --- | --- |
| 普通乘用车 | 80,000 – 100,000 | 100,000 – 120,000 |
| 运动型轮胎 | 120,000 – 150,000 | 140,000 – 170,000 |
| 卡车 | 200,000 – 400,000 | 250,000 – 500,000 |

侧偏刚度随负载、温度、气压变化，精确值需通过滚鼓试验台实测标定。当侧偏角超过约 5°–8° 时，轮胎进入非线性饱和区，线性模型失效，需使用 Pacejka Magic Formula。


## 横向误差动力学

将车辆相对于参考轨迹的误差建模为状态空间系统，是 LQR/MPC 控制器设计的基础。

### 误差状态定义

设参考轨迹曲率为 $\kappa_r$，定义误差状态向量：

$$\mathbf{e} = \begin{bmatrix} e_y \\ \dot{e}_y \\ e_\psi \\ \dot{e}_\psi \end{bmatrix}$$

其中：
- $e_y$：横向位置偏差（垂直于参考轨迹方向）
- $\dot{e}_y$：横向偏差变化率
- $e_\psi = \psi - \psi_r$：航向角偏差
- $\dot{e}_\psi$：航向偏差变化率

### 线性化状态空间方程

在参考轨迹附近线性化，得到连续时间状态空间方程：

$$\dot{\mathbf{e}} = A\,\mathbf{e} + B\,\delta + C\,\kappa_r$$

状态矩阵 $A$、输入矩阵 $B$、参考前馈矩阵 $C$：

$$A = \begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & -\dfrac{2C_{\alpha f}+2C_{\alpha r}}{mv_x} & \dfrac{2C_{\alpha f}+2C_{\alpha r}}{m} & -\dfrac{2C_{\alpha f}l_f - 2C_{\alpha r}l_r}{mv_x} \\
0 & 0 & 0 & 1 \\
0 & -\dfrac{2C_{\alpha f}l_f - 2C_{\alpha r}l_r}{I_z v_x} & \dfrac{2C_{\alpha f}l_f - 2C_{\alpha r}l_r}{I_z} & -\dfrac{2C_{\alpha f}l_f^2 + 2C_{\alpha r}l_r^2}{I_z v_x}
\end{bmatrix}$$

$$B = \begin{bmatrix} 0 \\ \dfrac{2C_{\alpha f}}{m} \\ 0 \\ \dfrac{2C_{\alpha f}l_f}{I_z} \end{bmatrix}, \quad
C = \begin{bmatrix} 0 \\ \dfrac{2C_{\alpha f}l_f - 2C_{\alpha r}l_r}{mv_x} - v_x \\ 0 \\ -\dfrac{2C_{\alpha f}l_f^2 + 2C_{\alpha r}l_r^2}{I_z v_x} \end{bmatrix}$$

矩阵中各参数的物理意义：$m$ 为整车质量，$I_z$ 为绕垂直轴的转动惯量，$v_x$ 为纵向速度。矩阵元素随速度 $v_x$ 变化，因此需要进行增益调度（Gain Scheduling）。

### 离散化

控制器实现时需对连续方程进行离散化（采样周期 $T_s = 0.01$ s）：

$$\mathbf{e}_{k+1} = A_d\,\mathbf{e}_k + B_d\,\delta_k + C_d\,\kappa_{r,k}$$

$$A_d = e^{AT_s} \approx I + AT_s + \frac{(AT_s)^2}{2}, \quad B_d = \int_0^{T_s} e^{A\tau} B\, d\tau$$


## 横向控制算法

### PID 控制

以横向偏差 $e_y$ 为误差，输出转向角修正量：

$$\delta = K_p e_y + K_i \int_0^t e_y \, d\tau + K_d \dot{e}_y$$

- **优点**：实现简单，参数调节直观
- **缺点**：不含预瞄（Previewing），高速下"亡羊补牢"式响应；忽略车辆动力学；参数对速度敏感

### Pure Pursuit（纯追踪算法）

在参考轨迹上选取前瞻点（Look-ahead Point），计算使车辆以圆弧到达该点的前轮转角：

$$\delta = \arctan\!\left(\frac{2L\sin\alpha}{l_d}\right)$$

其中 $l_d$ 为前瞻距离（通常取 $l_d = k \cdot v$，随速度自适应），$\alpha$ 为车辆航向与前瞻点方向的夹角。

- **优点**：几何意义直观，实现简单，对噪声鲁棒
- **缺点**：前瞻距离的选取影响大；低速时精度差，高速时稳定性下降

### Stanley 控制器

斯坦福大学 DARPA 大挑战参赛车辆（Stanley）使用，综合考虑航向误差和横向误差：

$$\delta = \psi_e + \arctan\!\left(\frac{k \cdot e_{fa}}{v + \epsilon}\right)$$

其中：
- $\psi_e$：车辆航向与轨迹切线方向的偏差角
- $e_{fa}$：前轴中心到最近轨迹点的横向距离（带符号）
- $k$：增益参数
- $\epsilon$：防止低速除零的小量

**控制解析：** 当横向偏差大时，第二项产生额外纠偏转角；当偏差消除后，仅保留航向对准项，确保平滑追踪。

### LQR 线性二次调节器

将横向控制建模为**线性最优控制**问题，最小化无限时域代价：

$$J = \int_0^\infty \left(\mathbf{e}^T Q \mathbf{e} + u^T R u\right) dt$$

状态量 $\mathbf{e} = [e_y,\ \dot{e}_y,\ e_\psi,\ \dot{e}_\psi]^T$，控制量 $u = \delta$，$Q \geq 0$、$R > 0$ 为权重矩阵。

求解代数黎卡提方程（ARE）得到最优反馈增益矩阵 $K$：

$$u^* = -K\mathbf{e}$$

$$A^T P + PA - PBR^{-1}B^T P + Q = 0, \quad K = R^{-1}B^T P$$

**特点：** 系统级最优，自然处理多状态耦合；需要准确的车辆动力学模型；权重矩阵 $Q,R$ 需调参（直观意义：$Q$ 决定对偏差的惩罚，$R$ 决定对转向幅度的惩罚）。百度 Apollo 早期版本采用 LQR 横向控制器。


## LQR 实现细节

### 权重矩阵调参指导

$Q$ 矩阵为对角矩阵，各元素对应各状态量的惩罚权重；$R$ 为标量（单输入系统）：

$$Q = \begin{bmatrix} q_{e_y} & 0 & 0 & 0 \\ 0 & q_{\dot{e}_y} & 0 & 0 \\ 0 & 0 & q_{e_\psi} & 0 \\ 0 & 0 & 0 & q_{\dot{e}_\psi} \end{bmatrix}, \quad R = r_\delta$$

调参原则：
- **增大** $q_{e_y}$：加强横向位置跟踪，但可能导致转向振荡
- **增大** $q_{e_\psi}$：加强航向对齐，改善高速稳定性
- **增大** $r_\delta$：抑制转向幅度，提升乘坐舒适性，但跟踪精度下降
- **规则**：$q_{e_y} / r_\delta$ 的比值越大，系统越激进；建议从 $Q = \text{diag}(1, 0, 1, 0)$，$R = 1$ 开始调试

### 离散 LQR 求解（Python 伪代码）

```python
import numpy as np
from scipy.linalg import solve_discrete_are

def solve_lqr_discrete(Ad, Bd, Q, R):
    """
    求解离散时间 LQR 最优增益矩阵
    Ad, Bd: 离散化状态空间矩阵
    Q, R:   权重矩阵
    返回: K (增益矩阵), P (黎卡提方程解)
    """
    # 求解离散代数黎卡提方程 (DARE)
    # Ad^T P Ad - P - Ad^T P Bd (R + Bd^T P Bd)^{-1} Bd^T P Ad + Q = 0
    P = solve_discrete_are(Ad, Bd, Q, R)

    # 最优增益矩阵
    K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
    return K, P

def lateral_control_lqr(state_error, K, kappa_r, v_x):
    """
    state_error: [e_y, e_y_dot, e_psi, e_psi_dot]
    K:           LQR 增益矩阵 (1 x 4)
    kappa_r:     参考轨迹曲率（前馈项）
    v_x:         当前纵向速度
    """
    # 反馈控制
    delta_fb = -K @ state_error

    # 曲率前馈（补偿稳态误差）
    delta_ff = np.arctan(kappa_r * (lf + lr))

    return float(delta_fb) + delta_ff
```

### 速度自适应增益调度表

由于 $A$ 矩阵随速度 $v_x$ 变化，需要在不同速度点离线计算 LQR 增益，运行时插值：

| 速度 (km/h) | $k_{e_y}$ | $k_{\dot{e}_y}$ | $k_{e_\psi}$ | $k_{\dot{e}_\psi}$ | 备注 |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.30 | 0.08 | 1.20 | 0.25 | 低速停车场景 |
| 30 | 0.45 | 0.12 | 1.50 | 0.35 | 城区低速 |
| 60 | 0.55 | 0.18 | 1.80 | 0.50 | 城区主干道 |
| 100 | 0.60 | 0.22 | 2.10 | 0.65 | 高速公路 |
| 120 | 0.58 | 0.25 | 2.30 | 0.75 | 高速极限工况 |

注：表中数值为示例，实际需根据车辆参数标定。增益并非单调递增，高速时 $k_{e_y}$ 略有下降（避免过激响应）。


## MPC 模型预测控制

MPC（Model Predictive Control，模型预测控制）是当前自动驾驶中最先进的控制框架，**统一**处理横纵向控制，并能自然处理复杂约束。

### 问题公式

在每个控制周期，求解有限时域 $N$ 步的在线优化问题：

$$\min_{u_0, \ldots, u_{N-1}} \sum_{k=0}^{N-1} \left[(\mathbf{x}_k - \mathbf{x}_k^{\text{ref}})^T Q (\mathbf{x}_k - \mathbf{x}_k^{\text{ref}}) + u_k^T R u_k\right] + (\mathbf{x}_N - \mathbf{x}_N^{\text{ref}})^T P_f (\mathbf{x}_N - \mathbf{x}_N^{\text{ref}})$$

**约束条件：**

$$\mathbf{x}_{k+1} = f(\mathbf{x}_k, u_k) \quad \text{（车辆动力学约束）}$$
$$u_{\min} \leq u_k \leq u_{\max} \quad \text{（执行器物理限制：转角、加速度上下界）}$$
$$|\Delta u_k| = |u_k - u_{k-1}| \leq \Delta u_{\max} \quad \text{（舒适性：限制控制量变化率）}$$

只执行第一步优化结果 $u_0^*$，下一时刻重新求解（**滚动时域**）。

### MPC 完整约束集

实际工程中，MPC 通常包含以下完整约束集：

**硬约束（Hard Constraints）：**
$$-\delta_{\max} \leq \delta_k \leq \delta_{\max} \quad \text{（前轮最大转角，典型值 ±30°）}$$
$$a_{\min} \leq a_k \leq a_{\max} \quad \text{（纵向加速度，如 -8 m/s² 至 +3 m/s²）}$$
$$|\dot{\delta}_k| \leq \dot{\delta}_{\max} \quad \text{（转向速率限制，防止方向盘急打）}$$
$$0 \leq v_k \leq v_{\max} \quad \text{（速度约束）}$$

**软约束（Soft Constraints）用于障碍物规避：**

将障碍物区域建模为软约束，引入松弛变量 $\xi_k \geq 0$：

$$d(\mathbf{x}_k,\ \text{obstacle}) \geq d_{\text{safe}} - \xi_k$$
$$\min \cdots + \rho \sum_{k=0}^{N-1} \xi_k^2 \quad \text{（惩罚松弛变量，} \rho \gg 1 \text{）}$$

软约束允许在极端情况下轻微违反约束（如紧急避让），但通过大惩罚系数 $\rho$ 使违反代价极高，兼顾安全性和求解可行性。

### 实时求解器对比

| 求解器 | 类型 | 典型求解时间 | 适用场景 | 开源 |
| --- | --- | --- | --- | --- |
| OSQP | 二次规划（QP）| 1–5 ms（N=10）| 线性 MPC，嵌入式部署 | 是 |
| ACADOS | 非线性 MPC（SQP/RTI）| 5–20 ms（N=20）| 非线性 MPC，高精度场景 | 是 |
| HPIPM | 结构化 QP | 0.5–3 ms | 高速线性 MPC | 是 |
| IPOPT | 通用非线性规划 | 50–200 ms | 离线验证，非实时 | 是 |
| qpOASES | 活跃集 QP | 2–8 ms | 小规模在线 QP | 是 |

### 计算负荷分析

以典型配置为例（10 Hz 控制频率，预测时域 N=10 步，每步 0.1 s）：

- **状态维度**：$\mathbf{x} \in \mathbb{R}^6$（$x, y, \psi, v, \delta, a$）
- **控制维度**：$u \in \mathbb{R}^2$（$\delta_{\text{cmd}}, a_{\text{cmd}}$）
- **决策变量总数**：$N \times n_u = 10 \times 2 = 20$
- **约束数量**：约 80–120 个不等式约束
- **QP 求解时间（OSQP）**：约 2–5 ms，满足 100 ms 控制周期要求
- **CPU 占用**：单核 10–30%（ARM Cortex-A72 级别处理器）

当预测时域增加到 N=30 或引入非线性动力学时，需要 GPU 加速或并行 SQP 方法。

### MPC 优势与挑战

**优势：**
- **统一框架**：横纵向联合优化，避免解耦控制的相互干扰
- **前瞻规划**：预测时域 $N$ 步，提前应对曲率变化和速度限制
- **约束处理**：自然地将加速度、转角、Jerk 等限制编码为硬约束
- **适应性**：模型可在线更新，处理负载变化、路面摩擦系数变化

**挑战：**
- **计算量大**：需实时求解 QP（线性 MPC）或 NLP（非线性 MPC）
- **模型精度**：模型不准确（如未标定的轮胎参数）会影响控制质量，需鲁棒 MPC 扩展


## 纵向控制详解

### 带前馈的 PID 速度控制

基础 PID 仅有反馈项，存在稳态误差和响应滞后。加入**速度前馈**和**加速度前馈**可显著改善跟踪性能：

$$a_{\text{cmd}} = \underbrace{K_p e_v + K_i \int_0^t e_v\, d\tau + K_d \dot{e}_v}_{\text{反馈项}} + \underbrace{a_{\text{ref}} + K_{ff} \dot{a}_{\text{ref}}}_{\text{前馈项}}$$

其中 $e_v = v_d - v$ 为速度误差，$a_{\text{ref}}$ 为参考轨迹加速度，$K_{ff}$ 为前馈增益。

**油门/制动分配：**

$$\text{throttle} = \begin{cases} \text{Throttle\_MAP}(a_{\text{cmd}}, v) & a_{\text{cmd}} > 0 \\ 0 & a_{\text{cmd}} \leq 0 \end{cases}$$

$$\text{brake} = \begin{cases} 0 & a_{\text{cmd}} > 0 \\ \text{Brake\_MAP}(|a_{\text{cmd}}|, v) & a_{\text{cmd}} \leq 0 \end{cases}$$

MAP 表通过车辆标定试验获得，需覆盖不同速度、负载、坡度工况。

### 自适应巡航控制（ACC）双模式

ACC 系统在两种模式间无缝切换：

**速度模式（Speed Mode）：**

前方无目标车辆或目标距离超过安全距离时激活，目标为维持设定速度 $v_{\text{set}}$：

$$a_{\text{cmd}} = K_p (v_{\text{set}} - v) + K_i \int (v_{\text{set}} - v)\, dt$$

**跟车模式（Distance Mode）：**

检测到前车且距离低于安全阈值时激活，同时控制距离和相对速度：

$$a_{\text{cmd}} = K_{p,d}(d - d_{\text{ref}}) + K_{d,d} \Delta v$$

其中 $d$ 为实际车距，$\Delta v = v_{\text{lead}} - v$ 为相对速度（前车速度减本车速度），$d_{\text{ref}}$ 为目标距离：

$$d_{\text{ref}} = \tau_h \cdot v + d_{\min}$$

时间间距 $\tau_h$（Time Headway）通常设为 1.5–2.5 s，$d_{\min}$ 为静止时最小安全距离（约 3–5 m）。

**模式切换判断：**

$$\text{模式} = \begin{cases} \text{跟车模式} & d_{\text{current}} < \tau_h \cdot v + d_{\text{switch}} \\ \text{速度模式} & \text{otherwise} \end{cases}$$

为避免频繁切换，设置迟滞区间（$d_{\text{switch}}$ 略大于 $d_{\min}$）。

### 电动车再生制动力矩融合

纯电动和混合动力车辆可利用电机进行再生制动，需在摩擦制动和再生制动之间分配：

$$T_{\text{brake, total}} = T_{\text{regen}} + T_{\text{friction}}$$

$$T_{\text{regen}} = \min\left(T_{\text{regen, max}},\ \eta_{\text{regen}} \cdot \left|\frac{P_{\text{charge, max}}}{\omega_{\text{motor}}}\right|\right)$$

分配策略：优先使用再生制动（能量回收），不足部分由摩擦制动补充。在 ABS 介入、电池 SOC 过高或低温场景下，须降低甚至关闭再生制动。

**控制框图：**

```
目标减速度 a_cmd (< 0)
       │
       ▼
  [再生制动分配器]
   ├─ T_regen → 电机控制器 → 能量回收 → 电池
   └─ T_friction → 液压制动系统 → 摩擦片
```

### Jerk 限制与乘坐舒适性

**加加速度**（Jerk，$j = \dot{a}$）是影响乘坐舒适性的关键指标。过大的 Jerk 会造成乘客不适和机械冲击。

**舒适性标准：**
- 正常行驶：$|j| \leq 2\ \text{m/s}^3$
- 紧急制动上限：$|j| \leq 10\ \text{m/s}^3$
- ISO 2631 标准：长时间暴露于 $> 0.315\ \text{m/s}^2$ RMS 振动时有不适感

**Jerk 限制实现（速率限制器）：**

$$a_{k+1} = a_k + \text{clip}\left(a_{\text{cmd}} - a_k,\ -j_{\max} T_s,\ +j_{\max} T_s\right)$$

其中 $T_s$ 为控制周期，$j_{\max} = 2\ \text{m/s}^3$。这等价于对加速度指令施加一阶斜坡速率限制。


## 控制器性能评估

### 跟踪误差指标

**横向跟踪 RMSE（Root Mean Square Error）：**

$$\text{RMSE}_{e_y} = \sqrt{\frac{1}{N}\sum_{k=1}^{N} e_{y,k}^2}$$

**纵向速度跟踪 RMSE：**

$$\text{RMSE}_{v} = \sqrt{\frac{1}{N}\sum_{k=1}^{N} (v_k - v_{d,k})^2}$$

工程指标要求（典型值）：
- 城区（< 60 km/h）：横向 RMSE $< 0.15$ m，速度 RMSE $< 0.5$ km/h
- 高速（60–120 km/h）：横向 RMSE $< 0.25$ m，速度 RMSE $< 1.0$ km/h

### 乘坐舒适性指标

**横向加速度 RMS：**

$$a_{y,\text{RMS}} = \sqrt{\frac{1}{T}\int_0^T a_y^2(t)\, dt}$$

**最大纵向 Jerk：**

$$j_{\max} = \max_{t} |\dot{a}(t)|$$

| 指标 | 优秀 | 可接受 | 不舒适 |
| --- | --- | --- | --- |
| 横向加速度 RMS | $< 0.5\ \text{m/s}^2$ | $< 1.5\ \text{m/s}^2$ | $> 2.0\ \text{m/s}^2$ |
| 纵向 Jerk 最大值 | $< 2\ \text{m/s}^3$ | $< 5\ \text{m/s}^3$ | $> 8\ \text{m/s}^3$ |
| 转向角速率 RMS | $< 5°/\text{s}$ | $< 15°/\text{s}$ | $> 25°/\text{s}$ |

### 稳定性裕度分析

对于线性控制系统（PID、LQR），可通过频域方法分析稳定裕度：

- **增益裕度（Gain Margin）**：使系统恰好不稳定所需的额外增益倍数，一般要求 $> 6$ dB
- **相位裕度（Phase Margin）**：使系统恰好不稳定所需的额外相位滞后，一般要求 $> 30°$

对于非线性 MPC，通过蒙特卡洛仿真（参数扰动、初始状态扰动）评估鲁棒性，要求在 95% 置信水平下满足跟踪误差要求。

### 算法横向对比

| 控制算法 | 横向 RMSE（城区）| 横向 RMSE（高速）| 计算时间 | 约束处理 | 工程复杂度 |
| --- | --- | --- | --- | --- | --- |
| PID | 0.25–0.40 m | 0.40–0.80 m | < 0.1 ms | 无 | 低 |
| Pure Pursuit | 0.20–0.35 m | 0.30–0.60 m | < 0.1 ms | 无 | 低 |
| Stanley | 0.15–0.25 m | 0.25–0.50 m | < 0.1 ms | 无 | 低 |
| LQR | 0.10–0.20 m | 0.15–0.30 m | 0.5–2 ms | 无（需后处理饱和限制）| 中 |
| MPC（线性）| 0.08–0.15 m | 0.10–0.20 m | 2–10 ms | 完整约束 | 高 |
| MPC（非线性）| 0.05–0.12 m | 0.08–0.15 m | 10–50 ms | 完整非线性约束 | 很高 |

注：数值为典型场景参考值，实际性能高度依赖于车辆标定质量和参数整定水平。


## 端到端控制

近年来，基于学习的控制方法逐渐受到关注：

**行为克隆（Imitation Learning）：**

直接从摄像头图像学习转向角，NVIDIA DAVE-2（2016）是先驱工作：

$$(\delta, a) = \pi_\theta(I_{\text{camera}},\ v,\ \text{GPS route})$$

**强化学习（RL）控制：**

在仿真器中通过奖励信号学习驾驶策略（Wayve、Comma.ai 等公司），优势是无需人工设计控制律；劣势是仿真到现实的迁移困难，安全性难以保证。


## 参考资料

1. J. Kong et al. Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design. IEEE IV, 2015.
2. R. Rajamani. Vehicle Dynamics and Control. 2nd ed., Springer, 2012.
3. J. Snider. Automatic Steering Methods for Autonomous Automobile Path Tracking. CMU Technical Report, 2009.
4. M. Bojarski et al. End to End Learning for Self-Driving Cars. NVIDIA, arXiv:1604.07316, 2016.
5. B. Paden et al. A Survey of Motion Planning and Control Techniques for Self-driving Urban Vehicles. IEEE T-ITS, 2016.
6. E. Guiggiani. The Science of Vehicle Dynamics. Springer, 2018.
7. B. Stellato et al. OSQP: An Operator Splitting Solver for Quadratic Programs. Mathematical Programming Computation, 2020.
8. R. Verschueren et al. acados: A Modular Open-Source Framework for Fast Embedded Optimal Control. Mathematical Programming Computation, 2022.
