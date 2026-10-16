# 三数据集、不同录制的扩展验证

本次按用户要求，在原版主线持续训练期间，独立验证 TRUMANS、EgoBody、RICH 各 10 个窗口。30 个窗口来自 30 个不同 recording/sequence；EgoBody 包含 5 个 camera_wearer 和 5 个 interactee，录制不重复。每个窗口为 192 帧、30 Hz，起点取相应合法索引的中位位置；录制按固定 seed 的 SHA256 排序选择，不按预测结果挑选。

全部 clean/noisy 预测使用同一个冻结的 **12,600 步** checkpoint，共 60 次采样；两种输入对应同一窗口，不算作 60 个独立窗口。保持主线的 ego/head-only、scene ON、DDIM20、软头部约束和原有物理修正设置。模型、优化器、学习率计划和训练 RNG 均未修改，主线仍以 55,000 步为目标。

这是扩展开发 VAL，不是全量 VAL 或独立 TEST；包含一个此前固定验证的 TRUMANS 窗口。将来比较时应复用相同窗口、输入协议、seed 和采样设置，不能将这里的 30 窗均值与原固定 3 窗均值直接相减判断训练进步。

- [当前进度与指标页面](index.html)：本地服务 `http://localhost:8794/`，每 8 秒更新。
- [最终或进度报告](REPORT_zh.md)、[完整汇总](SUMMARY.json)：只有全部 60 次采样与 runner 最终检查完成，才标记最终结果。
- [抽样计划](PLAN.json)、[运行协议](PROTOCOL.json)、[冻结 checkpoint 证据](FROZEN_CHECKPOINT.json)。
- [独立 CPU 验收范围](VERIFY_README_zh.md)、[最近验收](VERIFY.json)：完成后对全部保存动作重新计算指标。
- [同窗 12k/12600 对照](CASE6_MATCHED_REVIEW.json)：区分配置/统计匹配与直接输入数组核对的证据。

`runner.py` 独立、可续跑，模型权重只读；派生 surface 缓存写在本目录，共享 static map 只读，缺失时在内存重建。CPU 线程为 2，CUDA 分配器上限为总显存的 25%；与训练共用 GPU，不能视为吞吐隔离。60 个动作 NPZ 和 evaluation-only checkpoint 保留本地，Git 仅备份代码与紧凑实验记录。

脚步稳定性重点报告 GT 足部速度低于 0.2 m/s 时的预测足速，并单列 GT 移动窗口，避免坐姿/静态窗口降低整体足速均值。这个低速区间是运动学代理，不能当成真实接触标签或场景穿透率。
