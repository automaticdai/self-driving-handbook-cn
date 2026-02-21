# 中间件与部署：ROS2/Cyber RT/AUTOSAR 与云边协同

本页关注框架"跑起来"的基础设施能力，涵盖中间件选型、车端部署、云边协同与可观测性体系。

---

## 1. 中间件选型对比

### 1.1 ROS 2

ROS 2（Robot Operating System 2）是机器人与自动驾驶研究领域最广泛使用的中间件框架。

**核心概念：**

| 概念 | 说明 |
| --- | --- |
| Node | 最小计算单元，运行一个独立进程或线程 |
| Topic | 发布/订阅通信信道（异步，无状态） |
| Service | 请求/响应通信（同步） |
| Action | 长时间任务（带反馈的异步 RPC） |
| Parameter | 运行时可配置的节点参数 |
| QoS | 通信质量策略（可靠性/历史/时效性） |

**QoS 策略示例：**

```yaml
# 高实时性感知数据：允许丢弃旧帧
perception_qos:
  reliability: BEST_EFFORT  # 不重传，避免积压
  durability: VOLATILE       # 不缓存历史消息
  history: KEEP_LAST
  depth: 1

# 控制指令：必须可靠
control_qos:
  reliability: RELIABLE
  durability: VOLATILE
  deadline: 20ms             # 超时告警
```

**ROS 2 适用场景：**

- 快速原型开发与算法验证
- 多机器人系统集成
- 与 Carla/SUMO 仿真器集成
- 研究机构与早期项目

### 1.2 Cyber RT（百度 Apollo）

Cyber RT 是百度 Apollo 自研的高性能实时中间件，针对自动驾驶场景优化。

**核心特点：**

- 基于协程（Coroutine）调度，比线程更轻量
- 内置共享内存（Zero-Copy）大消息传输
- Component 设计模式：每个 Component 绑定输入 Channel，响应式触发
- 内置可视化工具（Cyber Visualizer）和调试工具

```cpp
// Cyber RT Component 示例（伪代码）
class PerceptionComponent : public cyber::Component<Image, PointCloud> {
public:
    bool Proc(const std::shared_ptr<Image>& image,
              const std::shared_ptr<PointCloud>& cloud) override {
        // 处理逻辑
        auto result = detector_.Detect(*image, *cloud);
        writer_->Write(result);
        return true;
    }
private:
    Detector detector_;
    std::shared_ptr<Writer<ObjectList>> writer_;
};
```

**与 ROS 2 对比：**

| 维度 | ROS 2 | Cyber RT |
| --- | --- | --- |
| 调度模型 | 线程池 | 协程（更低上下文切换开销） |
| 时延 | 中等 | 更低（点对点 ~1 ms 以内） |
| 安全认证 | 无官方车规认证 | 与 Apollo 商业版集成 |
| 生态 | 极丰富（ROS 2 社区） | Apollo 生态圈 |
| 调试工具 | rqt, ros2 topic | Cyber Visualizer, Recorder |

### 1.3 AUTOSAR Adaptive

AUTOSAR Adaptive（AP）是面向高计算能力 ECU 的车规化软件平台标准。

**核心服务：**

| 服务 | 说明 |
| --- | --- |
| `ara::com` | 服务化通信（基于 SOME/IP 或自定义传输） |
| `ara::exec` | 进程执行管理与生命周期控制 |
| `ara::diag` | 车载诊断接口（UDS 协议） |
| `ara::log` | 统一日志框架（DLT 格式） |
| `ara::per` | 持久化存储管理 |
| `ara::phm` | 平台健康管理（模块故障检测） |

**AUTOSAR AP 适用场景：**

- 量产车规化平台（ISO 26262 合规流程）
- 多供应商协同开发（接口标准化）
- 底盘/安全功能与智驾功能的协同

!!! note "研发 vs 量产的常见做法"
    许多企业在研发阶段使用 ROS 2 + Cyber RT，量产阶段迁移到 AUTOSAR AP 或自研量产框架。两套栈并行维护是目前行业痛点之一。

---

## 2. DDS 通信层

DDS（Data Distribution Service）是 ROS 2 的底层传输协议，提供丰富的 QoS 策略：

| DDS 实现 | 特点 | 适用 |
| --- | --- | --- |
| Fast DDS（eProsima） | ROS 2 默认，功能完整 | 通用 |
| CycloneDDS（Eclipse） | 轻量，延迟低 | 嵌入式/边缘 |
| Connext DDS（RTI） | 商业支持，高可靠性 | 量产安全关键 |
| OpenDDS | 开源，符合标准 | 政府/国防项目 |

---

## 3. 共享内存与零拷贝

大消息（点云、图像）通过网络序列化传输代价极高，共享内存是核心优化手段：

```
传统模式：
  Publisher → 序列化 → 网络栈 → 反序列化 → Subscriber
  （多次内存拷贝，典型延迟：10–50 ms for 1 MB）

共享内存模式：
  Publisher → 写入 SHM → 通知（轻量消息） → Subscriber 读取 SHM
  （零拷贝，典型延迟：< 1 ms for 1 MB）
```

**实现方式：**

- POSIX Shared Memory：`shm_open` + `mmap`
- GPU-CPU 统一内存（CUDA Unified Memory）：适合 GPU 推理直接输出到 CPU 消费场景

---

## 4. 车端部署分层

### 4.1 按实时性分层

```
┌─────────────────────────────────────────────────────┐
│              强实时层（< 10 ms 时延约束）            │
│  控制执行 │ AEB 安全监控 │ 传感器驱动               │
│  调度策略：SCHED_FIFO，隔离 CPU 核                  │
├─────────────────────────────────────────────────────┤
│              准实时层（10–100 ms 时延约束）           │
│  感知推理 │ 预测 │ 规划决策 │ 定位更新              │
│  调度策略：SCHED_FIFO 中等优先级，GPU 并行           │
├─────────────────────────────────────────────────────┤
│              非实时层（> 100 ms 容忍）               │
│  地图更新 │ 模型热切换 │ 参数下发                   │
│  调度策略：SCHED_OTHER，共享 CPU 资源               │
├─────────────────────────────────────────────────────┤
│              离线层（异步，不影响主链路）             │
│  日志压缩 │ 数据上传 │ 统计分析 │ 后台诊断          │
│  调度策略：SCHED_IDLE，带宽限流                      │
└─────────────────────────────────────────────────────┘
```

### 4.2 容器化部署

Docker/OCI 容器在车端的应用限制：

- **优点**：环境隔离，部署标准化，快速回滚
- **限制**：
  - 实时性：容器内核调度不完全隔离，需要额外配置
  - GPU 直通：需要 NVIDIA Container Toolkit
  - 存储：车规级 eMMC/SSD 的容量和寿命约束

---

## 5. 云边协同架构

### 5.1 职责划分

```
┌─────────────────────────────────────────────────────┐
│                     云端（Cloud）                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ 模型训练  │  │ 仿真评估  │  │  数据挖掘/标注   │  │
│  │（GPU集群）│  │（闭环仿真）│  │  （批量处理）    │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │ OTA/模型分发
                       │ 数据回传（触发片段）
┌──────────────────────↓──────────────────────────────┐
│                     车端（Edge）                     │
│  在线推理 │ 实时决策 │ 本地安全兜底 │ 边缘数据预处理 │
└─────────────────────────────────────────────────────┘
```

**核心原则：**

- **云端增强，车端闭环自主可运行**
- 车端不依赖云端实时响应，网络中断不影响驾驶安全
- 云端负责离线优化，车端负责在线推理

### 5.2 数据回传策略

```
触发回传的条件（示例）：
  ├─ 接管事件（人工干预）
  ├─ 感知置信度持续低于阈值
  ├─ 规划超时或求解失败
  ├─ 传感器异常（跳变、丢帧）
  └─ 特定地理围栏区域（标记为重点采集区）

回传数据格式：
  - 触发前后 N 秒的传感器原始数据
  - 系统日志（感知/预测/规划输出）
  - 车辆状态（速度、加速度、转角）
```

---

## 6. 可观测性三支柱

### 6.1 指标（Metrics）

使用 Prometheus + Grafana 构建指标监控：

```yaml
# 关键指标示例（Prometheus 格式）
perception_latency_ms{module="camera", percentile="p99"} 65.3
planning_timeout_count_total 0
control_publish_hz 100.2
trajectory_valid_ratio 0.997
localization_error_m{axis="lateral"} 0.045
```

### 6.2 日志（Logging）

结构化日志设计：

```json
{
  "timestamp": 1234567890.123,
  "level": "WARN",
  "module": "planning",
  "frame_id": "f_001234",
  "event": "planning_timeout",
  "details": {
    "elapsed_ms": 87.3,
    "budget_ms": 50.0,
    "fallback": "prev_trajectory"
  }
}
```

**日志存储策略：**

- 车端循环缓冲区（Ring Buffer）：保留最近 N 分钟完整日志
- 触发事件后，快照上传到云端
- 关键事件（接管、碰撞、故障）永久保留

### 6.3 链路追踪（Tracing）

跨模块追踪单帧数据的处理路径：

```
frame_id: f_001234
  ├─ camera_driver: +0ms (t=0)
  ├─ perception_start: +5ms
  ├─ bev_inference: +35ms (GPU)
  ├─ fusion: +45ms
  ├─ prediction: +62ms
  ├─ planning_start: +65ms
  ├─ planning_end: +88ms
  └─ control_publish: +90ms
```

使用 OpenTelemetry 标准，可对接 Jaeger、Zipkin 等可视化工具。

---

## 7. 发布管理

### 7.1 版本规范

```
<major>.<minor>.<patch>-<build>
  major: 重大架构变更（接口不兼容）
  minor: 新功能（向后兼容）
  patch: Bug修复
  build: CI 构建号

示例：2.5.3-20250601.001
```

### 7.2 灰度发布流程

```
内部测试车（10辆）
    ↓ 通过：时延/指标/接管率达标
受控用户（0.1%）
    ↓ 通过：7天故障率无上升
扩大灰度（5% → 30%）
    ↓ 通过：一键回滚验证OK
全量
```

### 7.3 发布前回归清单

- [ ] 全量仿真回归（Baseline 场景 > 5000 个）
- [ ] 接管率未超过基线 10%
- [ ] 关键时延指标 P99 未回退
- [ ] 一键回滚验证通过
- [ ] 安全 DTC 无新增
- [ ] 配置与模型版本绑定正确
