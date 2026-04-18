# 自动驾驶技术指南

*A Guide to Autonomous Driving — 中文*

本手册系统介绍自动驾驶技术的历史、现状与发展趋势，覆盖从基础概念到核心算法、从硬件系统到工程落地的完整知识体系。

![Image result for google autonomous car](docs/google_av.png)

**在线阅读：** <https://yfrobotics.github.io/self-driving-handbook-cn>

## 章节一览

1. **概述** — 定义、SAE 分级、发展历史、术语表
2. **系统** — 车辆架构、V2X、高精地图、功能安全、法规
3. **硬件** — 计算平台（CCU）、线控、车载通信、传感器、摄像头
4. **算法** — 感知、融合、定位、规划、预测、控制、端到端
5. **仿真测试** — 仿真平台、场景生成、Sim-to-Real
6. **视觉语言大模型** — VLM 基础、场景理解、决策规划、部署
7. **实例** — Apollo、Waymo、Tesla、中国玩家、Robotaxi

## 本地构建

```bash
pip install -r requirements.txt
mkdocs serve           # 本地预览，监听 http://127.0.0.1:8000
mkdocs build --strict  # 严格构建（CI 使用）
```

## 贡献

欢迎通过 Issue 或 Pull Request 参与。详见[贡献指南](docs/how-to-contribute.md)与[书写规范](docs/standard.md)。

## 版权声明

![cc-by-sa-4.0](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)

本维基遵循 [知识共享署名-相同方式共享 4.0 国际协议（CC BY-SA 4.0）](https://creativecommons.org/licenses/by-sa/4.0/deed.zh-Hans)。
