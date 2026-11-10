# 当前实验入口

更新：2026-10-21。本索引区分已有模型、新 AMASS 配置和已退役记录；不把准备中的配置称为完成训练。

| 当前内容 | 入口与状态 |
|---|---|
| 新 AMASS 128 帧阶段 | [协议与准备进度](../../../docs/AMASS_RETRAIN_zh.md) · [配置](../../../configs/amass128.yaml)；尚未开始训练 |
| 当前统一模型完整 noisy 验证 | [总报告](native30_final_metrics_v1/full_validation_noisy/OVERALL_REPORT_zh.md) · [指标表](native30_final_metrics_v1/full_validation_noisy/METRIC_TABLES_zh.md) · [状态](native30_final_metrics_v1/full_validation_noisy/STATUS.json) |
| 完整 clean 验证 | [总报告](native30_final_metrics_v1/full_validation/OVERALL_REPORT_zh.md) |
| EXO natural 10k | [冻结配置](native30_exo_natural10k_v1/config.json) · [状态](native30_exo_natural10k_v1/STATUS.json)；10,000 更新已完成，保留完整恢复权重 |
| EGO 场景融合 control | [选定模型合同](native30_ego_scene_fusion_v2/SELECTED_MODEL.json) · [配对报告](native30_ego_scene_fusion_v2/REPORT_zh.md)；新增 attention 未被采纳，保留原结构 control |
| 原 EGO 41k / 55k | [完成报告](native30_gait_v2_ego55k_oct15/final_review/REPORT_zh.md) · [权重审计](native30_gait_v2_ego55k_oct15/final_review/CHECKPOINT_AUDIT.json)；41k 用于开发评价，55k 为完整恢复状态 |
| RICH 官方 TEST 41k | [报告](rich_test_ego41k/REPORT_zh.md) · [状态](rich_test_ego41k/STATUS.json)；670/670 预测完成 |
| GRAIL SMPL-X 处理 | [实时 STATUS](grail_data_v1/STATUS.json) · [说明](grail_data_v1/README_zh.md)；CPU 持续处理，以实时文件为准 |
| 保留的候选与旧官方模型 | `native30_checkpoint_review_v1/`、`native_dynamic_scene20_contact_55k_oct07/`；保留资产，不自动继续训练 |
| 当前可视化 | [Viser 8800](http://localhost:8800/) · [已准备案例](native30_final_metrics_v1/full_validation_noisy/visuals/PREPARED_CASES.json) |

完整 noisy 验证已经生成 7,082/7,082 预测，并记录数值例外；原严格数值门未全部通过，不能把完成预测写成所有检查通过。具体有效性和限制以总报告为准。可视化观感、遮挡生成质量和论文可比性仍需分别验证。

## 保留与退役范围

完整保留的八组目录：

- `grail_data_v1`
- `native30_final_metrics_v1`
- `native30_exo_natural10k_v1`
- `native30_gait_v2_ego55k_oct15`
- `native30_ego_scene_fusion_v2`
- `rich_test_ego41k`
- `native30_checkpoint_review_v1`
- `native_dynamic_scene20_contact_55k_oct07`

其他 27 个目录已按批准清单移除旧权重、预测和可重建缓存，保留配置、指标与审计等小记录。它们不再是可直接恢复的模型；旧运行状态仅作历史证据。`native30_final_review_oct16/io_recovery.py`、`native30_physics_schedule_probe_oct15/run_probe.py` 和旧 TRAIN mount 文件等仍是冻结依赖，因此保留其路径；不为清空目录破坏现有模型。

[逐项清理与复验](../../../docs/maintenance/README_zh.md) · [历史源码恢复包](../archive/source_bundles/README_zh.md) · [全部目录的保留分类](../../../docs/maintenance/run_retention.json)。历史网页若引用已删除数组，不再作为查看入口。新增实验使用集中配置与模块 CLI，不在 runs 放新 Python 脚本。
