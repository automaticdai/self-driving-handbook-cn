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


## 纵向控制算法

### PID 速度控制

以速度误差为被控量：

$$a = K_p (v_d - v) + K_i \int_0^t (v_d - v) d\tau + K_d \frac{d(v_d - v)}{dt}$$

根据计算出的加速度需求，通过**逆模型**（Inverse Model / 标定 MAP）转化为油门踏板开度（加速）或制动压力（减速）。

### 自适应巡航控制（ACC）

在纯速度控制之上增加前车跟随逻辑：

- **速度模式**：前方无目标车辆或过远时，保持目标速度（类 PID）
- **跟车模式**：检测到前车时，维持安全跟车距离（类 PD，以相对速度和车距为输入）

两模式切换基于：$d_{\text{current}} < d_{\text{safety}} = \tau_h \cdot v + d_{\min}$（时间间距策略，$\tau_h$ 通常取 1.5–2.5 s）。


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

### MPC 优势
- **统一框架**：横纵向联合优化，避免解耦控制的相互干扰
- **前瞻规划**：预测时域 $N$ 步，提前应对曲率变化和速度限制
- **约束处理**：自然地将加速度、转角、Jerk 等限制编码为硬约束
- **适应性**：模型可在线更新，处理负载变化、路面摩擦系数变化

### MPC 挑战
- **计算量大**：需实时求解 QP（线性 MPC）或 NLP（非线性 MPC），常用求解器：OSQP（开源 QP）、ACADOS（嵌入式非线性 MPC）
- **模型精度**：模型不准确（如未标定的轮胎参数）会影响控制质量，需鲁棒 MPC 扩展


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
