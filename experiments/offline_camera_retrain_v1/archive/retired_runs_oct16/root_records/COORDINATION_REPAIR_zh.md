# 全身协调修复 v1（2026-10-02）

## 目标与边界

不是给行走硬编码“异侧手脚同相”，而是从各自真实动作学习全身关系。
模型输入、骨架表示、mask 混训比例和 ego 可见场景采样不变；推理不使用 GT
接触、动作标签或事后改动轨迹。原有权重和实验均保留。

新增 `coordination_v1` = `gait_v2` + 全身短窗关系损失。包含头/躯干、双上肢、
双下肢共 12 个节段向量（手相对肩，脚相对髋等），不是只分析脚。

- 每窗 32 帧（20 Hz 下 1.6 秒），窗口间隔 16 帧。
- 去除节段静态均值后，比较滞后 0 / 4 / 8 帧的跨节段向量协方差。
- 用各节段活动能量和 3 cm 正则项归一化，避免近静止噪声主导相关性。
- 另外比较活动幅度，防止仅靠缩小动作获得好看的相关性分数。
- 权重：关系 MSE × 0.05，幅度 MSE（m²）× 5；乘以 diffusion alpha_bar，
  减少高噪声时强迫某个参考动作模式的问题。
- 没有预设手脚应同相还是反相；拿东西、双手协作、弯腰等以各自 GT 为准。

这是统计关系辅助监督，并非完整的人类自然性判别器；不严格对任意相位移动不变，
也不能保证新动作都自然。低活动幅度下 Pearson 相关性不可靠，不能单独验收。
测试包括非行走双手协作、躯干运动、相位错误、幅度坍缩、静止与短序列、
全局旋转/任意逐帧平移不变性、噪声门控、有限梯度和原数据管线。

## 正在进行的配对实验

目录：`runs/coordination_paired_oct02`。

- 起点：`foot_fix_long20k_oct02/gait_v2/evaluations/step_020000/inference.pt`
  （累计 54,000 步，原 20k 延续训练及评估已成功完成）。
- 两组：`coordination_v1` 与 `gait_v2`，各加 1,000 步。
- 相同 seed=20261007、batch=4、lr=3e-5、float32、额外低噪声比例 0.25，
  `trumans_male_v2` 骨架；相同优化器重置与学习率 schedule。
- 原始运行代码保存在 `source_snapshot`；`status.json` 记录训练和最终评估状态。
- 结束后自动在 24 个验证窗口 × 两个采样 seed（777/778）上比较起点及两个分支。
  这些窗口已用于此前 20k 模型的检查，因此不是全新的盲测集；当前用于严格配对。
- 最终对比标签 `original` 指本轮 54k 起点，不是最初 30k 模型。
- 首次完成短程对照之前，不宣称顺拐已修复或整体自然性已提高。

## 验收与重现

`evaluate_sampling.py` 自动记录全身关系/幅度误差，并保留头轨迹、脚滑、
脚相对骨盆活动幅度、速度/加速度和支撑代理指标。关系指标本身也是训练项，
必须结合独立动作播放、幅度和其他指标，不能把优化训练项等同于泛化自然性。

独立只读分析程序 `analyze_coordination.py` 可从现有 gallery 提取腕部活动、
手脚相位和按 GT 路径/脚间距定义的行走代理子集；其余窗口没有人工动作标签，
不能直接称为“非行走动作集”。输出文件拒绝覆盖。

```bash
PYTHONPATH=. /home/wenxin/miniconda3/envs/scenemi/bin/python -m unittest \
  experiments.offline_camera_retrain_v1.test_coordination \
  experiments.offline_camera_retrain_v1.test_gait \
  experiments.offline_camera_retrain_v1.test_pipeline

PYTHONPATH=. /home/wenxin/miniconda3/envs/scenemi/bin/python \
  -m experiments.offline_camera_retrain_v1.analyze_coordination \
  --gallery experiments/offline_camera_retrain_v1/runs/coordination_paired_oct02/final_fresh_validation/gallery.html \
  --output experiments/offline_camera_retrain_v1/runs/coordination_paired_oct02/final_coordination_audit.json
```

网页入口（本机服务运行时）：
`http://127.0.0.1:8766/coordination_paired_oct02/index.html`
