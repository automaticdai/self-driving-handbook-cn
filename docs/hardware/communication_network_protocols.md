# 网络与协议：总线、以太网、拓扑与服务通信

本页聚焦车内网络的数据承载能力与协议选择，从传统总线到高速以太网，涵盖拓扑演进与服务化通信架构。

---

## 1. 传统总线技术

### 1.1 CAN 与 CAN FD

CAN（Controller Area Network）是汽车领域最成熟的总线技术，具备多主仲裁、差分信号、故障自隔离等特性。

| 特性 | CAN 2.0 | CAN FD |
| --- | --- | --- |
| 最高速率 | 1 Mbps | 8 Mbps（数据段） |
| 帧格式 | 标准/扩展帧 | 扩展帧 + 可变数据长度 |
| 数据字段 | 最大 8 字节 | 最大 64 字节 |
| 向下兼容 | — | 支持与 CAN 节点共存 |
| 典型用途 | 底盘控制、OBD | 域间状态上报、ADAS 辅助链路 |

**帧结构示意（CAN FD）：**

```
| SOF | Arbitration ID | BRS | ESI | DLC | Data (0-64B) | CRC | ACK | EOF |
```

- **BRS**（Bit Rate Switch）：控制数据段切换到更高波特率
- **ESI**（Error State Indicator）：标识发送节点错误状态
- **CRC**：CAN FD 使用更强的 17/21 位 CRC

!!! note "仲裁机制"
    CAN 采用非破坏性位仲裁（CSMA/CD-like），显性位（0）优先于隐性位（1）。ID 越小优先级越高，控制帧通常分配低 ID。

### 1.2 LIN 总线

LIN（Local Interconnect Network）适用于低速、低成本场景：

- 单线半双工，主从架构（一主多从）
- 最高速率：20 kbps
- 典型用途：车窗升降、座椅调节、车身灯光、雨刮器控制

LIN 成本极低，但不适合高实时性场景，不具备多主能力。

### 1.3 FlexRay

FlexRay 是面向底盘安全的高速确定性总线：

- 最高速率：10 Mbps（双通道 20 Mbps）
- 时间触发（TDMA）调度，抖动极小
- 支持冗余通道，适合制动、转向等安全关键功能
- 成本较高，已逐渐被 Automotive Ethernet 取代

---

## 2. 车载以太网

### 2.1 标准演进

| 标准 | 速率 | 线缆 | 特点 |
| --- | --- | --- | --- |
| 100BASE-T1 | 100 Mbps | 单对非屏蔽双绞线 | ADAS 感知数据、中低带宽场景 |
| 1000BASE-T1 | 1 Gbps | 单对非屏蔽双绞线 | 域间骨干、摄像头数据 |
| 2.5GBASE-T1 | 2.5 Gbps | 单对 | 多摄像头汇聚 |
| 10GBASE-T1 | 10 Gbps | 单对 | 高带宽感知、中央计算平台 |
| MultiGBASE-T1 | 可变 | 单对 | 速率自适应 |

与消费以太网相比，车载以太网物理层采用单对非屏蔽双绞线，体积小、重量轻，适合整车布线约束。

### 2.2 带宽需求估算

```
单路摄像头（8MP，30fps，YUV420）≈ 720 Mbps（未压缩）
经 H.265/H.264 压缩后：约 10–50 Mbps
16 路摄像头总带宽：200–800 Mbps（压缩后）

128 线 LiDAR（点频 10M/s，每点 XYZ+反射+时戳 20B）≈ 1.6 Gbps（峰值）
实际传输（UDP 包，部分稀疏）：约 200–400 Mbps

毫米波雷达：约 10–50 Mbps（目标列表级别）
```

!!! warning "带宽规划原则"
    感知数据不要与控制数据共享物理链路，或至少配置 QoS 隔离。预留 30%+ 冗余带宽，避免突发流量导致控制链路阻塞。

---

## 3. 网络拓扑演进

### 3.1 传统分布式 ECU 架构

```
传感器 → ECU₁ ─┐
传感器 → ECU₂ ─┤─ CAN Bus ─ 网关 ─ OBD/云
传感器 → ECU₃ ─┘
```

- 特点：各 ECU 功能独立，线束复杂（部分车型线束重量超 50 kg）
- 问题：扩展困难，算法升级需同步多 ECU

### 3.2 域控制器架构

```
┌─────────────────────────────────────────────────────┐
│  智驾域         底盘域         座舱域         车身域  │
│  (ADAS DCU)    (Chassis DCU)  (IVI DCU)    (Body DCU) │
└────┬────────────────┬───────────┬──────────────┬─────┘
     │                │           │              │
     └──── 中央网关 (Central Gateway) ─ 以太网骨干 ┘
```

- 同域功能集中，跨域通过网关交换
- 线束显著减少，算法迭代更灵活

### 3.3 中央计算 + 区域控制器（Zonal Architecture）

```
              ┌─────────────────────────┐
              │    中央计算平台（CCU）   │
              │  (CPU + GPU/NPU + HSM)  │
              └──────────┬──────────────┘
                         │ 车载以太网骨干
          ┌──────────────┼──────────────┐
     ┌────┴────┐   ┌─────┴────┐   ┌────┴────┐
     │ 区域控制器  │   区域控制器   │  区域控制器
     │  (Zone A)  │   (Zone B)    │  (Zone C)
     └────┬────┘   └─────┬────┘   └────┬────┘
     传感器/执行器    传感器/执行器    传感器/执行器
```

- 区域控制器按物理位置（前/后/左/右）划分，就近接入传感器和执行器
- 极大简化线束，支持软件定义车辆（SDV）

---

## 4. SOME/IP 服务通信

SOME/IP（Scalable Service-Oriented Middleware over IP）是 AUTOSAR 定义的车载服务化通信协议。

### 4.1 核心概念

| 概念 | 说明 |
| --- | --- |
| Service | 提供一组方法和事件的逻辑单元 |
| Method（RPC） | 客户端请求，服务端响应 |
| Event | 服务端主动推送，客户端订阅接收 |
| Field | 可读写的配置项，带 Getter/Setter/Notifier |
| Service Discovery（SD） | 服务发现：Offer/Subscribe/StopOffer |

### 4.2 服务发现流程

```
Client                              Server
  |                                   |
  |──── Find Service ─────────────>  |
  |<─── Offer Service ──────────────  |
  |──── Subscribe EventGroup ──────>  |
  |<─── Subscribe Acknowledge ─────   |
  |<══════ Event Notification ════════ | (周期或变化触发)
```

### 4.3 序列化（Serialization）

SOME/IP 消息格式（头部 16 字节）：

```
| Service ID (2B) | Method ID (2B) | Length (4B) |
| Client ID (2B)  | Session ID (2B)| Proto (1B)  |
| Interface Ver (1B) | Msg Type (1B)| Return Code (1B) |
| Payload (variable) |
```

!!! tip "工程建议"
    在接口 IDL（Interface Definition）中严格定义数据类型和版本号，避免序列化兼容性问题。跨域通信的方法调用建议增加超时和重试机制。

---

## 5. DDS 在自动驾驶中的应用

DDS（Data Distribution Service）是 OMG 标准的发布订阅中间件，ROS 2 默认使用 DDS 传输层。

| 特性 | SOME/IP | DDS |
| --- | --- | --- |
| 标准来源 | AUTOSAR | OMG |
| 适用场景 | 量产 E/E 架构 | 研发/原型/ROS 2 |
| QoS 策略 | 有限 | 丰富（Reliability/Durability/Liveliness 等） |
| 发现机制 | 静态配置/动态 SD | 动态 RTPS 发现 |
| 工具链生态 | AUTOSAR 工具链 | ROS 2/Cyber RT 生态 |

DDS 的 QoS 策略是其优势所在：

```yaml
# ROS 2 QoS 示例
qos:
  reliability: BEST_EFFORT    # 或 RELIABLE
  durability: VOLATILE        # 或 TRANSIENT_LOCAL
  history: KEEP_LAST
  depth: 10
  deadline: 100ms
  liveliness: AUTOMATIC
```

---

## 6. TSN（时间敏感网络）

TSN 是一组 IEEE 802.1 标准的集合，赋予标准以太网确定性传输能力。

### 6.1 核心标准

| 标准 | 功能 |
| --- | --- |
| IEEE 802.1AS | 精确时钟同步（gPTP），精度 <1 μs |
| IEEE 802.1Qbv | 时间感知整形（TAS），门控调度 |
| IEEE 802.1Qbu | 帧抢占，减少高优先级帧等待 |
| IEEE 802.1Qcc | 集中式网络配置（CNC/CUC 模型） |
| IEEE 802.1CB | 帧复制与消除（FRER），冗余路径 |

### 6.2 时间感知整形（TAS）原理

```
时间轴：
|←── 125 μs 周期 ──→|
|── 控制流窗口 ──|── 尽力而为流 ──|

Gate Control List（GCL）控制各队列开关：
队列 7（控制）：开  关  开  关  ...
队列 0（尽力）：关  开  关  开  ...
```

控制流在专用时间窗口内发送，保证最坏情况时延。

### 6.3 时钟同步精度

gPTP 基于 IEEE 1588v2，在汽车网络中通常可实现：

- 端到端同步精度：< 1 μs（局域网内）
- 对于多传感器时间对齐（相机/LiDAR/Radar），通常要求同步精度 < 100 μs

---

## 7. 网络安全基础

车载网络的安全隔离建议：

- 区分安全关键网络（Safety-Critical）与信息娱乐网络（Infotainment）
- 使用防火墙/网关严格过滤跨域流量
- 关键控制指令走 SecOC 认证（详见通信安全页面）

---

## 8. 网络诊断与监控

### 8.1 关键监控指标

| 指标 | 说明 | 建议阈值 |
| --- | --- | --- |
| 端到端时延 P95 | 控制链路关键指标 | < 5 ms（底盘控制） |
| 端到端时延 P99 | 长尾时延控制 | < 10 ms |
| 时延抖动（Jitter） | 周期任务时序稳定性 | < 500 μs |
| 丢包率 | 网络可靠性 | < 0.001%（控制链路） |
| 总线/链路利用率 | 过载预警 | < 60%（控制） |
| CAN 错误帧率 | 物理层健康 | 告警阈值：> 0.01% |

### 8.2 诊断工具

- **CAN 诊断**：CANalyzer、CANoe（Vector）；车端使用 UDS（ISO 14229）
- **以太网诊断**：Wireshark（开发）、车端轻量抓包（tcpdump）
- **DDS 诊断**：FastDDS Monitor、ros2 topic echo/hz

---

## 9. 设计建议

1. **关键控制链路**优先使用确定性网络（TSN/FlexRay/CAN FD）。
2. **感知大流量与控制小流量分网**，或使用 TSN QoS 严格隔离优先级。
3. **提前规划网关算力**：中央网关需要处理大量协议转换和过滤规则，避免成为系统瓶颈。
4. **网络带宽预留 30%** 以上冗余，应对突发场景（故障重传、日志上报）。
5. **接口版本化管理**：SOME/IP 服务 ID 和 Method ID 应统一维护，防止多团队开发冲突。

---

## 参考标准

- ISO 11898（CAN 标准）
- IEEE 802.3bw/bp（100BASE-T1 / 1000BASE-T1）
- AUTOSAR SOME/IP 协议规范
- IEEE 802.1AS/Qbv/CB（TSN 标准族）
- OPEN Alliance TC（BroadR-Reach/100BASE-T1 推广联盟）
