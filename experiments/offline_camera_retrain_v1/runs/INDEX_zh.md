# 当前实验入口

当前主线 `native30_gait_v2_ego55k_oct15` 已于 **2026-10-17 20:49:24 正常完成 55,000 步**，worker 返回码 0。最终判断是训练拟合接近平台、验证轻微回退，脚部支撑问题仍未解决；保留 41k 作为主要开发候选、55k 作为完整续训状态，暂不延长、不自动接 EXO。统一入口：[完成报告](native30_gait_v2_ego55k_oct15/final_review/REPORT_zh.md) · [训练与固定验证曲线](native30_gait_v2_ego55k_oct15/final_review/TRAINING_CURVES.png)。

用户随后授权的新阶段分开记录：[冻结41k的RICH TEST](rich_test_ego41k/README_zh.md) 已准备335窗/670次预测，前2次通过独立CPU复算，全量任务等待GPU，尚非全TEST结论；[共享扩散EXO适配](native30_ego_exo_adapt_v1/README_zh.md) 已通过实际EXO/mixed反传、隐藏GT隔离及12次固定VAL基线，开始1000更新、EXO80%/EGO20%的新阶段。实时进度看下表，实际更新记录在该阶段training/trace.jsonl。两个任务串行使用GPU。

| 要看什么 | 入口 |
|---|---|
| 55k 完成状态 | [STATUS.json](native30_gait_v2_ego55k_oct15/continuous_training_oct16/STATUS.json) |
| RICH TEST实时进度 | [STATUS.json](rich_test_ego41k/STATUS.json) · [逐窗与汇总](rich_test_ego41k/SUMMARY.json) |
| EXO适配实时进度 | [STATUS.json](native30_ego_exo_adapt_v1/STATUS.json) · [实际预检](native30_ego_exo_adapt_v1/prepared/input_checks.json) |
| 55k实际训练时长 | [31小时06分更新 / 46小时24分总跨度](native30_gait_v2_ego55k_oct15/final_review/TRAINING_DURATION.json) |
| 41k实际训练时长 | [23小时16分更新 / 37小时28分至快照可用](native30_gait_v2_ego55k_oct15/final_review/TRAINING_DURATION_41K.json) |
| Viser同步动作回放 | [打开六个案例](http://localhost:8800/) · [案例身份与步数](native30_live_monitor_oct16/visuals/viser/CASES.json)；GT/预测、脚轨迹、完整扫描/可见场景可分别切换 |
| 已保存的 best、选择指标和步数 | [best.json](native30_gait_v2_ego55k_oct15/training/best.json) · [自动保存状态](native30_gait_v2_ego55k_oct15/training/BEST_GUARDIAN_STATUS.json) |
| 总 loss、各监督贡献、训练时长 | [TensorBoard](http://localhost:6006/#custom_scalars) · [损失说明](native30_live_monitor_oct16/LOSS_GUIDE_zh.md) |
| 已完成的 15,100 步 RICH 全部合法窗口验证 | [报告](native30_live_monitor_oct16/rich_full_validation_oct16/REPORT_zh.md) · [412 份独立 CPU 验收](native30_live_monitor_oct16/rich_full_validation_oct16/VERIFY_COMPLETE.json) · [网页](http://localhost:8795/) |
| 方法有没有用、证据到哪一步 | [证据与比较方案](../ops/EVIDENCE_STATUS_zh.md) |
| 正确的旧55k对照 | [结果](native_dynamic_scene20_contact_55k_oct07/RESULTS_zh.md) · `native_dynamic_scene20_contact_55k_oct07/last.pt` |
| 固定/轮换验证截图与脚步检查 | [截图索引](native30_live_monitor_oct16/visuals/INDEX.html) · [脚步复核](http://localhost:8793/) |
| EgoGVAE / EgoEgo / EgoAllo 比较协议 | [已核对的训练、划分与输入差异](native30_live_monitor_oct16/rich_full_validation_oct16/PROTOCOL_REVIEW_zh.md) |

`training/last.pt` 为 **55k**，保留完整 Adam、sampler/counts 与全部 RNG；`training/best.pt` 为 **41k 模型副本**，仅用于评估。best 按同一固定 RICH VAL 窗口的 clean/noisy World MPJPE 均值选择，不是全 RICH 最优。两份权重及 365 份源封印已实核一致；[身份与恢复审计](native30_gait_v2_ego55k_oct15/final_review/CHECKPOINT_AUDIT.json)、[保存机制与限制](../ops/CHECKPOINT_SELECTION_zh.md)。

15,100 步的 RICH 全部合法窗口验证已完成：206 窗/27 条录制、clean/noisy 共 412 次预测，独立 CPU 重算通过。最新 41k/55k 比较另覆盖 14 条不同录制的 VAL 窗口（12 RICH＋1 TRUMANS＋1 EgoBody），主比较关闭推理物理修正，5 个核心窗补报物理开启，共 76 次预测，独立 CPU 重算通过。后者不是最终权重的全 RICH 验证，两者均不是外部论文 RICH TEST；旧55k另有历史官方TEST重叠及FPS、输入、预算差异，尚不能宣布新版胜过旧55k或论文方法。

2026-10-16晚间清理：44个退役目录、165份旧Python脚本移到 [retired_runs_oct16](../archive/retired_runs_oct16/README_zh.md)；删除3份退役版本中间权重和14,282份可重建预测数组，共6.411 GiB。44个目录归档本身不释放磁盘。当前训练的365份封存依赖清理前后SHA一致。[实际执行记录](../ops/cleanup_oct16_evening/EXECUTED.json)。此前29个历史试验的归档和50.06 GiB清理另见 [旧清理记录](run_inventory_oct16/CLEANUP_PLAN_zh.md)。

2026-10-17进一步清理：实际删除106个根目录退役入口和1份重复审计脚本，根目录Python文件352→246；原55k的365项精确封存源码集中保留在[恢复包](../archive/source_bundles/README_zh.md)，当前EXO98项、TEST95项活动依赖清理前后均验证一致。[逐文件执行记录](../ops/ROOT_SCRIPT_RETIREMENT_EXECUTED.json)。这次没有删除权重、数据或预测数组，剩余模块尚未全部归并。

其余保留目录是活跃监控、仍有比较价值的旧EXO/控制组或实际依赖。尤其 `native30_final_review_oct16/io_recovery.py`、`native30_physics_schedule_probe_oct15/run_probe.py`、`ego_exo_surface_continue_oct11/TRAIN_camera_mounts.json` 仍在使用。历史架构和推导见 [原研究记录](exo_occlusion_review_oct12/EXPERIMENT_AND_ARCHITECTURE_zh.md)。新增运维工具统一放在 `../ops/`，不再逐次散落到 runs。
