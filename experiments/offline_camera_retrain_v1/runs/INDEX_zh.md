# 当前实验入口

当前主线是 `native30_gait_v2_ego55k_oct15`；55k 是目标步数，实际进度看训练状态。保留原模型、输入、监督、Adam 和 RNG，持续训练，不因诊断或评估自动暂停、重启。

| 要看什么 | 入口 |
|---|---|
| 当前训练进度 | [STATUS.json](native30_gait_v2_ego55k_oct15/continuous_training_oct16/STATUS.json) |
| 已保存的 best、选择指标和步数 | [best.json](native30_gait_v2_ego55k_oct15/training/best.json) · [自动保存状态](native30_gait_v2_ego55k_oct15/training/BEST_GUARDIAN_STATUS.json) |
| 总 loss、各监督贡献、训练时长 | [TensorBoard](http://localhost:6006/#custom_scalars) · [损失说明](native30_live_monitor_oct16/LOSS_GUIDE_zh.md) |
| RICH 全部合法窗口验证 | [实时报告](native30_live_monitor_oct16/rich_full_validation_oct16/REPORT_zh.md) · [进度](native30_live_monitor_oct16/rich_full_validation_oct16/STATUS.json) · [网页](http://localhost:8795/) |
| 方法有没有用、证据到哪一步 | [证据与比较方案](../ops/EVIDENCE_STATUS_zh.md) |
| 正确的旧55k对照 | [结果](native_dynamic_scene20_contact_55k_oct07/RESULTS_zh.md) · `native_dynamic_scene20_contact_55k_oct07/last.pt` |
| 固定/轮换验证截图与脚步检查 | [截图索引](native30_live_monitor_oct16/visuals/INDEX.html) · [脚步复核](http://localhost:8793/) |
| EgoGVAE / EgoEgo / EgoAllo 比较协议 | [已核对的训练、划分与输入差异](native30_live_monitor_oct16/rich_full_validation_oct16/PROTOCOL_REVIEW_zh.md) |

`training/last.pt` 保留优化器用于续训；`training/best.pt` 为模型副本，用于评估。best 暂按同一固定 RICH VAL 窗口的 clean/noisy World MPJPE 均值选择，不是全 RICH 最优。2026-10-16 19:04 已用实际存在的15,100步权重和同一步真实评估初始化，分数150.290 mm。历史12,000步该指标更低，但权重已覆盖，不能补称已保存。[保存机制与限制](../ops/CHECKPOINT_SELECTION_zh.md)。

当前 RICH 验证固定15,100步、206窗/27条录制、clean/noisy共412次预测；完成及独立复核前不把局部均值当最终结果。本轮是 VAL，不是外部论文使用的 RICH TEST。旧55k有历史官方TEST重叠，旧/新还存在FPS、输入和训练预算差别，不能直接拿历史表格宣布新版胜出。

2026-10-16晚间清理：44个退役目录、165份旧Python脚本移到 [retired_runs_oct16](../archive/retired_runs_oct16/README_zh.md)；删除3份退役版本中间权重和14,282份可重建预测数组，共6.411 GiB。44个目录归档本身不释放磁盘。当前训练的365份封存依赖清理前后SHA一致。[实际执行记录](../ops/cleanup_oct16_evening/EXECUTED.json)。此前29个历史试验的归档和50.06 GiB清理另见 [旧清理记录](run_inventory_oct16/CLEANUP_PLAN_zh.md)。

其余保留目录是活跃监控、仍有比较价值的旧EXO/控制组或实际依赖。尤其 `native30_final_review_oct16/io_recovery.py`、`native30_physics_schedule_probe_oct15/run_probe.py`、`ego_exo_surface_continue_oct11/TRAIN_camera_mounts.json` 仍在使用。历史架构和推导见 [原研究记录](exo_occlusion_review_oct12/EXPERIMENT_AND_ARCHITECTURE_zh.md)。新增运维工具统一放在 `../ops/`，不再逐次散落到 runs。
