# 成像与标定：光学基础、ISP、内外参与误差治理

本页聚焦摄像头"看得清、看得准"的基础能力，涵盖光学成像原理、图像信号处理管线、内外参标定方法与运营期误差治理。

---

## 1. 光学成像基础

### 1.1 薄透镜成像模型

车载摄像头的标准成像模型基于薄透镜（Pinhole Camera）：

$$\frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}$$

其中 $f$ 为焦距，$d_o$ 为物距，$d_i$ 为像距。实际成像遵循透视投影：

$$\begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = \frac{1}{Z} K \begin{pmatrix} X \\ Y \\ Z \end{pmatrix}, \quad K = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}$$

其中 $K$ 为相机内参矩阵，$(c_x, c_y)$ 为主点，$f_x, f_y$ 为像素焦距（单位：像素）。

### 1.2 视场角（FOV）

FOV 决定了摄像头的覆盖范围与几何畸变权衡：

$$\text{FOV}_h = 2\arctan\left(\frac{W_{sensor}}{2f}\right)$$

| 应用场景 | 推荐 FOV | 典型焦距（1/2.8" 传感器） |
| --- | --- | --- |
| 前视长焦（远距目标） | 25–35° | 8–12 mm |
| 前视标准（近场障碍） | 60–80° | 3–5 mm |
| 环视鱼眼（泊车） | 180–200° | 1.2–1.6 mm |
| 车内 DMS（驾驶员监测） | 90–120° | 2–3 mm |

### 1.3 像素尺寸与感光能力

像素尺寸直接影响低照度性能（感光量子效率）：

$$\text{SNR} \propto \sqrt{A_{pixel} \cdot t_{exp} \cdot L}$$

其中 $A_{pixel}$ 为像素面积，$t_{exp}$ 为曝光时间，$L$ 为场景亮度。

| 像素尺寸 | 感光量（相对） | 典型应用 |
| --- | --- | --- |
| 1.0 μm | 1× | 高分辨率、日间 |
| 2.1 μm | 4.4× | 平衡夜间与分辨率 |
| 4.0 μm | 16× | 高端 DMS/前视 |

---

## 2. ISP 管线要点

### 2.1 标准 ISP 处理流程

```
RAW 图像（Bayer 格式）
    │
    ↓ 黑电平校正（Black Level Correction）
消除传感器暗电流偏移
    │
    ↓ 坏点补偿（Defective Pixel Correction）
替换异常像素值
    │
    ↓ 去噪（Noise Reduction）
  ├─ 空域：双边滤波、BM3D
  └─ 时域：多帧 TNR（Temporal Noise Reduction）
    │
    ↓ 去马赛克（Demosaicing）
Bayer 插值 → RGB 图像
    │
    ↓ 白平衡（Auto White Balance）
消除色温偏差（日光 5600K / 钨丝 3200K）
    │
    ↓ 自动曝光（Auto Exposure Control）
多区域测光 + 反馈调节（增益 × 曝光时间）
    │
    ↓ 色彩校正（Color Correction Matrix, CCM）
传感器光谱响应 → 标准色彩空间
    │
    ↓ Gamma / Tone Mapping
非线性压缩（适配显示或神经网络输入范围）
    │
    ↓ 畸变校正（Lens Distortion Correction）
使用内参 + 畸变系数（k1, k2, p1, p2, k3）
    │
输出图像（RGB / YUV / ISP 压缩格式）
```

### 2.2 畸变模型

常用径向-切向联合畸变模型（Brown-Conrady）：

$$x_{distorted} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2p_1 xy + p_2(r^2 + 2x^2)$$

其中 $r^2 = x^2 + y^2$，$k_1, k_2, k_3$ 为径向畸变系数，$p_1, p_2$ 为切向畸变系数。

鱼眼镜头使用等距（Equidistant）或等立体角模型：

$$r_{img} = f \cdot \theta \quad \text{（等距模型）}$$

### 2.3 ISP 性能权衡

| 处理模块 | 画质提升 | 算力开销 | 时延引入 |
| --- | --- | --- | --- |
| 多帧 TNR | 强 | 高 | +10–30 ms |
| HDR 融合 | 强（高对比场景）| 中 | +3–10 ms |
| 超分辨率 | 中 | 高 | +20–50 ms |
| 基础去噪+去马赛克 | 基础 | 低 | < 5 ms |

---

## 3. 内参标定（Intrinsic Calibration）

### 3.1 棋盘格标定法（Zhang 方法）

最常用的内参标定方法，利用多张棋盘格图像求解：

```
步骤：
1. 准备高精度棋盘格（格间距误差 < 0.1 mm）
2. 采集 20–50 张不同角度、位置的棋盘格图像
3. 提取角点（亚像素精度：cornerSubPix）
4. 优化求解内参 K 和畸变系数 d
5. 计算重投影误差（Reprojection Error）

通过标准：
  - 重投影误差 < 0.5 pixel（高精度需求 < 0.3 pixel）
  - 标定图像覆盖视场边缘和中心
```

**重投影误差计算：**

$$e_{reproj} = \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{p}_i - \hat{\mathbf{p}}_i \|^2$$

其中 $\mathbf{p}_i$ 为实测角点，$\hat{\mathbf{p}}_i$ 为根据标定结果重投影的角点。

### 3.2 焦距温漂

镜头焦距随温度变化（镜片膨胀）：

$$\Delta f \approx \alpha_{thermal} \cdot f \cdot \Delta T$$

其中 $\alpha_{thermal}$ 为热膨胀系数。-40°C 到 85°C 范围内，焦距漂移可达 0.5–2%，对长焦镜头影响尤其明显。

工程对策：

- 出厂多温度点标定（-40/0/25/85°C）
- 运行期根据温度传感器插值修正 $f_x, f_y$

---

## 4. 外参标定（Extrinsic Calibration）

### 4.1 相机到车体坐标系

外参 $\mathbf{T}_{cam}^{vehicle}$ 描述摄像头在车体坐标系下的位姿（6 DOF：3 个平移 + 3 个旋转）。

**靶板法（多平面角点）：**

```
1. 在停车场布置 3–5 个已知位置的靶板
2. 用高精度 RTK 测量靶板角点坐标（mm 精度）
3. 通过 PnP（Perspective-n-Point）求解 T_cam^vehicle
4. 验证：多靶板联合重投影误差 < 2 pixel
```

**道路直线约束法（在线标定）：**

```
利用路面车道线的直线性约束（3D 直线应投影为 2D 直线）：
1. 在高速直线路段采集视频
2. 提取车道线检测结果
3. 拟合直线并估计俯仰角（Pitch）和横滚角（Roll）
4. 适合大批量车辆的工厂线标定或在线持续校准
```

### 4.2 多相机联合标定

对于环视系统（4–8 路相机），需要统一到同一坐标系：

```
参考坐标系（通常为车体 IMU 坐标系）
    │
    ├─ T_front → Front Camera
    ├─ T_rear  → Rear Camera
    ├─ T_left  → Left Camera
    └─ T_right → Right Camera

约束：相邻相机 FOV 重叠区域应保持几何一致性
验证：重叠区物体不应出现位置跳变
```

### 4.3 Camera-LiDAR 联合标定

将相机与激光雷达坐标系对齐：

$$\mathbf{p}_{cam} = T_{lidar}^{cam} \cdot \mathbf{p}_{lidar}$$

**棋盘格联合标定：**

```
1. 放置大型棋盘格（≥ 60 cm × 80 cm）
2. 相机提取棋盘格角点（像素坐标）
3. LiDAR 从平面点云中拟合棋盘格平面（法向量 + 中心点）
4. 最小化 3D–2D 对应误差求解 T_lidar^cam

精度要求：
  - 平移误差 < 2 cm
  - 旋转误差 < 0.5°
```

---

## 5. 误差治理与在线监控

### 5.1 运行期外参漂移原因

| 原因 | 漂移量级 | 检测方式 |
| --- | --- | --- |
| 振动疲劳（安装螺钉松动）| 0.3–2° | 周期性靶板检测 |
| 碰撞/轻微剐蹭 | 1–10° | 实时一致性检测 |
| 温度冲击（塑料支架变形）| 0.1–0.5° | 多温度点比对 |
| 镜头污染（折射效应改变）| 间接影响焦距等效值 | 重投影误差监控 |

### 5.2 在线外参健康监控

```python
# 在线外参监控伪代码
class CalibrationMonitor:
    def check_consistency(self, camera_detections, lidar_objects):
        for obj in lidar_objects:
            # 将 LiDAR 目标投影到图像坐标
            proj_bbox = project_to_image(obj, T_lidar_cam, K)
            # 与图像检测框匹配
            matched = find_matching_detection(proj_bbox, camera_detections)
            if matched:
                iou = compute_iou(proj_bbox, matched.bbox)
                self.iou_history.append(iou)

        mean_iou = np.mean(self.iou_history[-100:])  # 滑动窗口
        if mean_iou < 0.4:  # 阈值告警
            trigger_recalibration_alert()
```

### 5.3 标定质量指标

| 指标 | 健康值 | 告警阈值 |
| --- | --- | --- |
| 内参重投影误差 | < 0.5 pixel | > 1.0 pixel |
| Camera-LiDAR IOU | > 0.6 | < 0.4 |
| 外参漂移（旋转） | < 0.3° | > 0.5° |
| 外参漂移（平移） | < 1 cm | > 3 cm |
| 跨相机重叠区一致性 | < 2 pixel 偏差 | > 5 pixel 偏差 |

### 5.4 标定基准库管理

```
标定记录 = {
    "vehicle_id": "VH-12345",
    "calibration_date": "2025-10-01T08:30:00Z",
    "software_version": "2.3.1",
    "intrinsic": {
        "fx": 1234.56, "fy": 1234.78, "cx": 960.1, "cy": 540.2,
        "distortion": [-0.32, 0.12, 0.001, -0.0002, -0.04]
    },
    "extrinsic": {...},
    "reprojection_error": 0.42,
    "temperature_at_calibration": 22.5,
    "calibration_target": "checkerboard_9x6_30mm",
    "operator": "auto_calibration_v1.2"
}
```

---

## 6. 工厂标定流程（量产）

### 6.1 量产标定站布局

```
标定站（工厂线）布置示意：

┌─────────────────────────────────────────────┐
│  ████████ 靶板 A（前）████████              │
│                                             │
│ 靶板 B ← ← 车辆停止位置 → → 靶板 C        │
│（左前）                      （右前）       │
│                                             │
│  ████████ 靶板 D（左）████████              │
│  ████████ 靶板 E（右）████████              │
│                                             │
│  ████████ 靶板 F（后）████████              │
└─────────────────────────────────────────────┘

步骤：
1. 车辆驶入定位槽（机械定位精度 < 1 mm）
2. 相机自动采集多路靶板图像
3. 软件自动计算内外参
4. 结果存储到 ECU + 云端备份
5. 单车标定时间目标：< 5 分钟
```

### 6.2 出厂回归验证

每辆出厂车必须通过以下验证：

- [ ] 前视摄像头重投影误差 < 0.5 pixel
- [ ] 环视拼接图无明显接缝偏移（< 5 pixel）
- [ ] Camera-LiDAR 对齐误差 < 2 cm（目标位置误差）
- [ ] 多温度点（0°C / 25°C / 85°C）焦距参数验证
- [ ] 畸变校正后直线应无明显弯曲

---

## 7. 运营期标定管理建议

1. **建立标定版本管理**：每次标定结果带版本号 + 时间戳，支持按时段回溯历史标定
2. **OTA 更新后强制回归**：ECU 软件更新后必须重新验证标定一致性
3. **高风险场景提高检频率**：长途运营车（每月一次）、运营事故后（每次必查）
4. **跨传感器一致性告警**：当 Camera-LiDAR IOU 连续 100 帧低于阈值，自动上报
5. **镜头污染监控**：利用图像清晰度（拉普拉斯方差）检测遮挡，联动清洗指令
