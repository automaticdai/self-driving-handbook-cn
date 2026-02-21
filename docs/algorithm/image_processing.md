# 图像感知与处理

视觉感知是自动驾驶系统的核心能力之一。摄像头作为最接近人类视觉的传感器，以低成本、高分辨率和丰富语义信息的优势，在自动驾驶感知栈中扮演着不可替代的角色。本章系统梳理从传统方法到前沿深度学习算法的主要技术路线，涵盖目标检测、语义分割、车道线检测、深度估计以及 BEV（鸟瞰视角）感知范式。


## 1. 视觉感知在自动驾驶中的地位

### 1.1 相机与 LiDAR 的互补性

自动驾驶感知系统通常融合多种传感器。相机与 LiDAR 各有优劣，形成天然互补：

| 维度 | 相机 | LiDAR |
| --- | --- | --- |
| 语义信息 | 丰富（颜色、纹理、文字） | 稀少 |
| 深度精度 | 单目需估计，精度较低 | 直接测量，精度高（cm 级） |
| 分辨率 | 高（百万像素级） | 低（点云稀疏） |
| 恶劣天气 | 受雨雾影响大 | 受影响相对小 |
| 成本 | 低（百至千元） | 高（数万至数十万元） |
| 夜间性能 | 需主动光源辅助 | 主动发射，不受光照影响 |

主流量产方案（如 Tesla FSD）倾向纯视觉路线，依赖大规模数据和强力算法；而 Waymo 等则坚持相机+LiDAR+雷达的多模态融合路线，以追求极致安全冗余。

### 1.2 图像感知的核心任务

```
原始图像
    |
    +---> 目标检测 (2D/3D 框)
    |
    +---> 语义/实例/全景分割
    |
    +---> 车道线检测
    |
    +---> 深度估计
    |
    +---> BEV 特征聚合
             |
             +---> 融合规划决策
```

| 任务 | 输入 | 输出 | 典型应用 |
| --- | --- | --- | --- |
| 目标检测 | 图像 | 2D/3D 边界框 + 类别 + 置信度 | 车辆、行人、交通标志 |
| 语义分割 | 图像 | 逐像素类别标签 | 可行驶区域、道路边界 |
| 实例分割 | 图像 | 逐像素类别 + 实例 ID | 区分同类别不同个体 |
| 全景分割 | 图像 | 语义 + 实例统一输出 | 场景完整理解 |
| 车道线检测 | 图像 | 车道线参数或掩码 | 车道保持、变道辅助 |
| 深度估计 | 单目/双目图像 | 逐像素深度图 | 距离感知、3D 重建 |


## 2. 目标检测

### 2.1 两阶段检测器：Faster R-CNN

Faster R-CNN（Ren et al., NeurIPS 2015）是两阶段检测的里程碑，由区域提议网络（Region Proposal Network，RPN）和 Fast R-CNN 检测头组成。

**RPN 网络原理**

RPN 共享主干特征图，在每个空间位置上预定义 $k$ 个 anchor（不同尺度和长宽比的参考框）。对于特征图上的每个位置，RPN 输出：

- 每个 anchor 的二分类分数（是否含目标）
- 相对于 anchor 的回归偏移量 $(\Delta x, \Delta y, \Delta w, \Delta h)$

**Anchor 机制**

设 anchor 中心为 $(x_a, y_a)$，宽高为 $(w_a, h_a)$，预测的真实框为：

$$x = x_a + w_a \cdot \Delta x, \quad y = y_a + h_a \cdot \Delta y$$

$$w = w_a \cdot e^{\Delta w}, \quad h = h_a \cdot e^{\Delta h}$$

RPN 的训练损失为分类损失与回归损失之和：

$$L_{RPN} = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* L_{reg}(t_i, t_i^*)$$

其中 $p_i^*$ 为 anchor 的真值标签，$t_i$ 为预测的回归量，$L_{reg}$ 为 Smooth L1 损失。

### 2.2 单阶段检测器

#### YOLO 系列（YOLOv8）

YOLO（You Only Look Once）将检测问题转化为单次前向传播的回归问题，速度极快。YOLOv8 是 Ultralytics 推出的最新版本，采用解耦检测头（Decoupled Head），将分类和回归分支分离，并引入 DFL（Distribution Focal Loss）进行边界框精细回归。

YOLOv8 网络结构示意：

```
Input (640x640)
    |
[CSPDarknet Backbone (C2f模块)]
    |
[PANet Neck (多尺度特征融合)]
    |
[Decoupled Head]
    +-- 分类分支 --> 类别概率
    +-- 回归分支 --> bbox坐标 (DFL)
```

#### SSD（Single Shot MultiBox Detector）

SSD 在多个特征图层（不同分辨率）上同时预测，天然适应多尺度目标。大目标由浅层大感受野特征图负责，小目标由深层高语义特征图负责。

#### RetinaNet 与 Focal Loss

RetinaNet（Lin et al., ICCV 2017）提出 **Focal Loss** 解决单阶段检测器中正负样本严重不均衡问题。标准交叉熵损失对易分类负样本（大量背景）赋予同等权重，导致训练被背景主导。Focal Loss 对易分类样本降权：

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

其中 $p_t$ 为模型对真实类别的预测概率，$\gamma \geq 0$ 为聚焦参数（论文取 2），$\alpha_t$ 为类别平衡权重。当 $\gamma = 0$ 时退化为标准交叉熵。聚焦因子 $(1-p_t)^\gamma$ 使得对于易分类样本（$p_t \to 1$）损失接近 0，训练集中于难分类样本。

### 2.3 Transformer 检测器

#### DETR（Detection Transformer）

DETR（Carion et al., ECCV 2020）首次实现端到端目标检测，彻底去除 anchor 设计和 NMS 后处理。

**二分匹配损失**

DETR 预测固定数量 $N$ 个目标查询（object query），通过匈牙利算法在预测集合与真值集合之间寻找最优二分匹配：

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)})$$

匹配代价包含分类代价和框回归代价（L1 + GIoU）。匹配确定后，训练损失为：

$$\mathcal{L}_{Hungarian} = \sum_{i=1}^{N} \left[ -\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{c_i \neq \varnothing} \mathcal{L}_{box}(b_i, \hat{b}_{\hat{\sigma}(i)}) \right]$$

#### DINO（DETR with Improved DeNoising Anchor Boxes）

DINO 在 DETR 基础上引入对比去噪训练（CDN）和混合匹配策略，大幅提升收敛速度和检测精度，在 COCO 上达到 63.3 AP，是当前最强的 Transformer 检测器之一。

### 2.4 3D 目标检测：BEV 相机方案

#### BEVDet

BEVDet 将各相机的透视特征通过 LSS（Lift-Splat-Shoot）方法投影到统一的 BEV 空间，再在 BEV 特征图上执行 3D 检测。

#### PETR（Position Embedding Transformer）

PETR 将 3D 位置编码注入图像特征，使得 Transformer cross-attention 能够直接在 3D 空间中建立全局上下文关联，无需显式的视图变换。

PETR 的核心思路：对图像特征图的每个像素，生成对应的 3D 位置编码 $PE_{3D}$，并与图像特征拼接后送入解码器：

$$F_{PETR} = F_{img} \oplus \text{MLP}(PE_{3D})$$

其中 $PE_{3D}$ 由相机内外参将图像坐标反投影到 3D 空间中得到，cross-attention 机制使每个 object query 能够聚合来自多视角的相关特征。


## 3. 语义分割与实例分割

### 3.1 FCN（全卷积网络）

FCN（Long et al., CVPR 2015）将 VGG 等分类网络的全连接层替换为卷积层，实现逐像素预测。通过上采样（转置卷积或双线性插值）恢复空间分辨率，并引入跳跃连接（skip connection）融合浅层细节信息。

### 3.2 DeepLabV3+

DeepLabV3+ 的核心创新是 **ASPP（Atrous Spatial Pyramid Pooling，空洞空间金字塔池化）**模块。空洞卷积在不增加参数量的前提下扩大感受野，ASPP 以多个不同膨胀率（dilation rate）的空洞卷积并行处理特征，捕获多尺度上下文信息：

```
输入特征图
    |
    +--[1x1 Conv]--+
    |              |
    +--[3x3 Atrous r=6]--+
    |                    |
    +--[3x3 Atrous r=12]-+--> Concat --> 1x1 Conv --> 分割输出
    |                    |
    +--[3x3 Atrous r=18]-+
    |              |
    +--[Global Avg Pool]-+
```

DeepLabV3+ 在 Encoder-Decoder 结构中，将 ASPP 模块的输出与低层特征融合，得到精细的分割边界。

设空洞卷积的膨胀率为 $r$，卷积核大小为 $k$，则感受野为：

$$RF = k + (k-1)(r-1)$$

### 3.3 Mask R-CNN 实例分割

Mask R-CNN（He et al., ICCV 2017）在 Faster R-CNN 的基础上增加掩码预测分支。关键改进是引入 **RoIAlign** 替代 RoIPool，通过双线性插值消除量化误差，保留精确的空间对应关系。

三个并行输出分支：

```
RoIAlign 特征
    |
    +--[分类头]--> 类别概率
    |
    +--[回归头]--> 边界框坐标
    |
    +--[掩码头 (FCN)]--> 二值掩码 (28x28)
```

总损失为三项之和：

$$L = L_{cls} + L_{box} + L_{mask}$$

其中 $L_{mask}$ 为逐像素的二元交叉熵损失，仅对真值类别的掩码分支反传梯度。

### 3.4 全景分割（Panoptic Segmentation）

全景分割统一了语义分割（无限类：道路、天空等）和实例分割（可数类：车辆、行人等），每个像素分配唯一的 $(category\_id, instance\_id)$ 对。

**全景质量指标（PQ）**定义为：

$$PQ = \underbrace{\frac{\sum_{(p,g) \in TP} IoU(p,g)}{|TP|}}_{\text{分割质量 SQ}} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{\text{识别质量 RQ}}$$

### 3.5 行驶区域分割（Drivable Area Segmentation）

行驶区域分割是自动驾驶中最直接的感知输出，判断图像中哪些区域可供车辆行驶。常见方案：

- **二分类语义分割**：将像素分为可行驶 / 不可行驶两类
- **多类别区分**：可行驶区域、停车区域、限制区域等
- **结合 BEV 输出**：在俯视图中输出行驶区域，便于路径规划


## 4. 车道线检测

### 4.1 传统方法：Hough 变换

Hough 变换将图像空间中的直线检测转化为参数空间中的峰值检测。对于图像中的每个边缘点 $(x_i, y_i)$，在参数空间 $(\rho, \theta)$ 中对所有满足

$$\rho = x_i \cos\theta + y_i \sin\theta$$

的参数对进行投票。参数空间中的峰值对应图像中共线的点集，即直线。

传统流程：灰度化 → Canny 边缘检测 → ROI 裁剪 → 霍夫变换 → 直线拟合

局限：仅能检测直线，弯道和复杂场景下效果差。

### 4.2 基于分割的方法：LaneNet

LaneNet（Wang et al., 2018）将车道线检测转化为实例分割问题：

- **二分类分支**：区分车道线像素与背景（语义分割）
- **嵌入分支**：为每条车道线的像素学习区分性嵌入向量，后处理时用聚类算法分离不同车道线

```
输入图像
    |
[共享 Encoder (ENet)]
    |
    +--[二分类分支]--> 前景/背景掩码
    |
    +--[嵌入分支]--> 像素嵌入向量
                         |
                    [DBSCAN 聚类]
                         |
                    分离后的车道线实例
                         |
                    [三次样条拟合] --> 车道线曲线
```

### 4.3 基于参数曲线的方法

**三次多项式**

将车道线建模为以车辆坐标系中 $y$ 为自变量的三次多项式：

$$x(y) = a_0 + a_1 y + a_2 y^2 + a_3 y^3$$

**Bezier 曲线**

$n$ 阶 Bezier 曲线由 $n+1$ 个控制点 $\mathbf{P}_0, \ldots, \mathbf{P}_n$ 定义：

$$\mathbf{B}(t) = \sum_{i=0}^{n} \binom{n}{i} (1-t)^{n-i} t^i \mathbf{P}_i, \quad t \in [0, 1]$$

车道线检测网络直接回归 Bezier 控制点，绕过逐像素分割的繁琐后处理，推理效率高。

### 4.4 基于 Anchor 的方法：LaneATT

LaneATT（Tabelini et al., CVPR 2021）预定义大量从图像底部延伸至顶部的折线锚（lane anchor），网络对每个锚预测：

- 是否包含车道线的分类分数
- 相对于锚的侧向偏移量

通过注意力机制聚合全局特征，解决车道线遮挡问题，实现高精度实时检测。

### 4.5 关键评估指标

| 指标 | 定义 | 说明 |
| --- | --- | --- |
| F1-Score | $\frac{2 \cdot P \cdot R}{P + R}$ | 精准率与召回率的调和平均 |
| IoU | $\frac{TP}{TP + FP + FN}$ | 预测区域与真值区域的交并比 |
| Accuracy | 正确像素/总像素 | 分割像素准确率（受类别不均衡影响） |


## 5. 深度估计

### 5.1 单目深度估计：MonoDepth2

单目深度估计从单张图像预测深度图，是高度欠定的问题（同一图像可能对应无数深度方案）。MonoDepth2（Godard et al., ICCV 2019）采用**自监督学习**方案，利用视频帧间的视图合成损失代替昂贵的深度标注。

**自监督损失**

给定目标帧 $I_t$ 和相邻帧 $I_{t'}$，通过预测的深度 $D_t$ 和相对位姿 $T_{t \to t'}$ 将 $I_{t'}$ 重投影合成 $\hat{I}_t$：

$$\hat{I}_t = I_{t'} \langle \text{proj}(D_t, T_{t \to t'}, K) \rangle$$

重投影损失结合 L1 损失与 SSIM（结构相似性）：

$$L_{re} = \alpha \frac{1 - \text{SSIM}(I_t, \hat{I}_t)}{2} + (1 - \alpha) \| I_t - \hat{I}_t \|_1$$

论文取 $\alpha = 0.85$。此外，MonoDepth2 引入最小化重投影损失（min reprojection loss）处理遮挡问题。

### 5.2 双目立体匹配：SGM 算法

双目相机通过两个水平基线对齐的相机获取同一场景的两幅图像，基于视差 $d$ 计算深度：

$$Z = \frac{f \cdot B}{d}$$

其中 $f$ 为焦距，$B$ 为基线长度，$d = x_L - x_R$ 为同名点的水平像素差。

**SGM（Semi-Global Matching）算法**

SGM 在代价聚合阶段从多方向（通常 8 或 16 个方向）累积匹配代价，平衡精度与效率：

$$S(p, d) = C(p, d) + \min \begin{cases} L_r(p-r, d) \\ L_r(p-r, d-1) + P_1 \\ L_r(p-r, d+1) + P_1 \\ \min_{d'} L_r(p-r, d') + P_2 \end{cases}$$

其中 $C(p,d)$ 为像素 $p$ 在视差 $d$ 下的匹配代价（如 Census 变换），$P_1$、$P_2$ 为惩罚系数，$P_2 > P_1$ 以惩罚大的视差变化。

### 5.3 深度补全（Sparse-to-Dense）

LiDAR 点云投影到图像平面后得到稀疏深度图（通常仅 5% 像素有效），深度补全任务将稀疏深度与 RGB 图像融合输出稠密深度图。

```
RGB 图像  + 稀疏深度图
    |              |
[RGB Encoder]  [深度 Encoder]
    |              |
    +----[Concat]--+
          |
    [Decoder + 跳跃连接]
          |
    稠密深度图输出
```

常用方法包括 FuseNet、CSPN（Convolutional Spatial Propagation Network）等，CSPN 通过学习传播系数在邻域像素间迭代更新深度估计，显著改善边界处的深度质量。


## 6. BEV 感知范式

### 6.1 IPM（逆透视变换）

逆透视变换（Inverse Perspective Mapping）基于**平坦地面假设**，将透视图像变换为俯视图：

设图像平面点 $\mathbf{u} = (u, v, 1)^T$，相机内参矩阵 $K$，相机到地面的外参矩阵（旋转 $R$、平移 $t$），地面点 $\mathbf{X} = (X, Y, 0)^T$，则变换关系为：

$$\lambda \mathbf{u} = K [R | t] \mathbf{X}$$

令 $Z=0$，可解出地面点坐标。IPM 计算简单，适合结构化道路，但在坡道、起伏路面上会产生较大误差。

### 6.2 BEVFormer：时空注意力机制

BEVFormer（Li et al., ECCV 2022）引入可变形注意力（Deformable Attention）从多相机特征中聚合 BEV 特征，支持时序信息融合。

**空间注意力**

对 BEV 查询位置 $q$ 对应的 3D 参考点，投影到各相机图像平面采样特征：

$$\text{SCA}(Q_p, F_t) = \frac{1}{|V_{hit}|} \sum_{i \in V_{hit}} \sum_{j=1}^{N_{ref}} \text{DeformAttn}(Q_p, \mathcal{P}(p, i, j), F_t^i)$$

其中 $V_{hit}$ 为能看到该 3D 点的相机集合，$\mathcal{P}(p, i, j)$ 将 3D 参考点投影到第 $i$ 个相机的 2D 位置。

**时序注意力**

BEVFormer 保留前一时刻的 BEV 特征 $B_{t-1}'$，通过自车位姿变换对齐后与当前 BEV 查询融合：

$$\text{TSA}(Q_p, \{Q, B_{t-1}'\}) = \sum_{V \in \{Q, B_{t-1}'\}} \text{DeformAttn}(Q_p, p, V)$$

时序融合使 BEVFormer 能够推断被遮挡目标的存在，并估计速度信息。

### 6.3 LSS（Lift, Splat, Shoot）

LSS（Philion & Fidler, ECCV 2020）提供了一种优雅的相机到 BEV 变换方式：

**Lift（提升）**：对每个图像像素，预测其在各离散深度箱 $d \in D$ 上的概率分布 $\alpha_d$，将 2D 图像特征"提升"为 3D 特征云：

$$c_d(u, v) = \alpha_d \cdot f(u, v)$$

**Splat（溅射）**：将所有相机的 3D 特征云通过体素化（Voxel Pooling）投影到 BEV 平面，使用 Cumsum trick 高效实现：

$$BEV(x, y) = \sum_{d, u, v: \Pi(u,v,d) \in (x,y)} c_d(u, v)$$

**Shoot（射击）**：在 BEV 特征图上完成下游任务（目标检测、分割等）。

LSS 的核心优势是深度预测完全由数据驱动，可以端到端训练，无需 LiDAR 监督信号。

### 6.4 多任务统一 BEV 输出

现代 BEV 感知框架（如 BEVFusion、UniAD）在同一 BEV 特征图上联合输出多任务结果：

```
多相机图像 [N x C x H x W]
        |
[视图变换（LSS / BEVFormer）]
        |
BEV 特征图 [C x X x Y]
        |
        +--[3D 目标检测头]--> 车辆/行人/障碍物 3D 框
        |
        +--[车道线检测头]--> BEV 车道线多段线
        |
        +--[可行驶区域头]--> BEV 分割掩码
        |
        +--[地图元素头]--> 道路边界、人行横道
```

多任务共享 BEV 特征，大幅降低计算开销，且各任务之间的特征共享有助于相互约束和提升。


## 7. 评估指标

### 7.1 指标汇总

| 指标 | 全称 | 计算公式 | 适用任务 |
| --- | --- | --- | --- |
| mAP | mean Average Precision | $\frac{1}{C} \sum_c \text{AP}_c$ | 目标检测 |
| mIoU | mean Intersection over Union | $\frac{1}{C} \sum_c \frac{TP_c}{TP_c + FP_c + FN_c}$ | 语义分割 |
| F1 | F1-Score | $\frac{2PR}{P+R}$ | 车道线检测 |
| ATE | Average Translation Error | $\frac{1}{N}\sum_i \|\hat{t}_i - t_i\|_2$ | 3D 目标检测 |
| NDS | nuScenes Detection Score | 综合 mAP + 5 项属性误差 | 3D 目标检测（nuScenes） |
| PQ | Panoptic Quality | $SQ \times RQ$ | 全景分割 |

### 7.2 指标说明

**mAP（目标检测）**：对每个类别计算 Precision-Recall 曲线下面积（AP），再对所有类别取平均。COCO 数据集在 IoU 阈值 0.5 至 0.95（步长 0.05）上取平均，记为 $\text{AP}^{COCO}$。

**mIoU（语义分割）**：交并比（Intersection over Union）衡量预测掩码与真值掩码的重叠程度。对 $C$ 个类别取平均得到 mIoU，是语义分割最主要的评测指标。

**ATE（3D 检测）**：衡量预测 3D 框中心与真值中心的欧氏距离，单位为米。nuScenes 还定义了 ASE（尺寸误差）、AOE（方向误差）、AVE（速度误差）、AAE（属性误差）。

**NDS（nuScenes 检测分数）**：综合多项误差的归一化评测分数：

$$NDS = \frac{1}{10} \left[ 5 \cdot mAP + \sum_{mTP \in \mathbb{TP}} (1 - \min(1, mTP)) \right]$$


## 8. 参考资料

1. S. Ren, K. He, R. Girshick, J. Sun. **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**. NeurIPS, 2015.

2. T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár. **Focal Loss for Dense Object Detection (RetinaNet)**. ICCV, 2017.

3. N. Carion, F. Massa, G. Synnaeve et al. **End-to-End Object Detection with Transformers (DETR)**. ECCV, 2020.

4. Z. Li, W. Wang, H. Li et al. **BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers**. ECCV, 2022.

5. J. Philion, S. Fidler. **Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D**. ECCV, 2020.

6. C. Godard, O. Mac Aodha, M. Firman, G. J. Brostow. **Digging Into Self-Supervised Monocular Depth Estimation (MonoDepth2)**. ICCV, 2019.
