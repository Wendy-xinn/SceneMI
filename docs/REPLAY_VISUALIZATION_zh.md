# 无场景 replay 2000：EGO / EXO Viser 检查

本机入口：<http://localhost:8842/>。当前权重是
`runs/scene_joint128_replay/candidates/step_002000.pt`，SHA256 为
`a46122d40c572be8ac82b55ca7d421e7c64a4de690c7ae369091064ba1b9260c`。
此为当前场景适配的父模型。其完整验证没有通过全部保持能力门槛，不能称为有效 best。

共九个 validation 案例。每个数据集按既有 CUDA 验证结果选择 EGO 根位置误差的
窗口中位数和第 90 百分位附近各一例，再选择 EXO gap 的第 90 百分位附近一例。
选例在重新生成前固定，清单为 `runs/scene_joint128_replay/visualization/CASES.json`。
这是有意检查高误差的选例，不能作为随机平均表现；独立 test 未使用。

| 案例 | 数据集 | 路径 | 本次预览根平均误差 |
| --- | --- | --- | ---: |
| 01 / 02 | EgoBody | EGO 中位 / 高误差 | 208 / 382 mm |
| 03 | EgoBody | EXO gap 高误差 | 85 mm |
| 04 / 05 | TRUMANS | EGO 中位 / 高误差 | 96 / 147 mm |
| 06 | TRUMANS | EXO gap 高误差 | 89 mm |
| 07 / 08 | GRAIL | EGO 中位 / 高误差 | 126 / 252 mm |
| 09 | GRAIL | EXO gap 高误差 | 97 mm |

蓝色为 GT，紫色为模型预测，浅绿色为 EXO 合成估计初始化。EGO 只接收相机，不提供
人体初始化或个人真值体形。场景仅供显示，未输入模型；EXO 初始化也是合成前端，
不能解释成真实 tracker 的性能。两个模式的窗口独立列出，未把不同窗口伪装成配对。

展开对应 EGO / EXO 图层，开启“骨盆行走轨迹”，播放或拖动帧滑块。场景、输入相机、
输入参考人体可分别切换。坐标保留真实全局关系：未逐帧居中、根对齐、缩放或贴地。
GT、预测与场景的共同坐标变换通过逐关节一致性检查；所有九例的显示网格与实际 FK
最大坐标差均低于 0.0000005 米。手指和面部使用显示用默认姿态，只评价共享的 22 关节。

界面同时显示根平均误差、首末帧误差和根误差向量的首尾变化。静态偏移和时间变化应分开看。
例 07 根误差从约 95 mm 到 389 mm，误差向量变化约 428 mm；例 09 首帧约 721 mm、
末帧约 35 mm，需结合其前缀缺失段检查恢复过程。单一首尾差也不能替代整段轨迹检查。

每例 128 帧、30 Hz，约 4.27 秒。**只能检查窗口内误差，不能证明长序列不累计漂移。**
本次导出使用 CPU DDIM 固定 seed 777，CPU 随机数流不同于原 CUDA 验证；观察输入的
全部历史身份字段均复核一致，但本次预测不冒充原 CUDA 验证预测，界面指标按本次输出重算。

准备阶段通过既有 CPU 资源封装执行，6 GiB RAM、CPU 12–13，未使用 GPU。
回放服务为 3 GiB RAM、CPU 10，继续使用既有 MotionViewer，无模型推理。
场景训练未重启、未修改依赖或权重。导出九例约占 350 MiB，可视化仍在使用，暂时保留。

稳定命令入口（准备和回放均需既有 CPU cgroup 封装）：

```bash
python -m scenemi view replay prepare
python -m scenemi view replay serve --port 8842
```

案例资产、哈希、指标及服务信息分别在该 run 的 `visualization/PREPARED_CASES.json`、
`STATUS.json`、`SERVER.json`。交付检查包含 HTTP 200 和匹配 Viser 版本子协议的
WebSocket 连接及二进制场景接收。
