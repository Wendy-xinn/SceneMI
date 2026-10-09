# 实验保留与清理索引（2026-10-09）

汇报训练预算是 **55,000 步**。本轮清理保留以下主线及其完整权重、评估、可视化和代码依赖。

| 主线 | 保留目录 / 说明 |
|---|---|
| 原始30k | `runs/scene5hz_long30k_b4_oct02`；`control30k_fullval_oct03` |
| 脚步约束后的55k | `runs/coordination_paired_oct02/gait_v2`；从54k追加1k的原脚步损失分支；保留整个配对目录及 `control55k_*` 评估 |
| 脚步修复过程 | `foot_fix_paired_oct02`、`foot_fix_template_oct02`、`foot_fix_support_oct02`、`foot_fix_long20k_oct02`；保留用于追溯30k→34k→54k→55k |
| 10月7日三数据集汇报55k | `canonical_smpl_3dataset_20hz_55k_oct07`；`report55k_fullval_oct07` |
| 原生20Hz汇报对照55k | `canonical_smpl_3dataset_native20_faceout_55k_oct07`；`report55k_native20_fullval_oct07` |
| 动态场景/接触修复55k（当前主线） | `native_dynamic_scene20_contact_55k_oct07`；含 `RESULTS_zh.md`、全量验证及场景作用示例 |
| 转身尝试 | `turn_repair_oct08`、`turn_generalization_oct09`、`turn_balance_oct09`、`soft_body_turn_robust_oct09`；全部保留 |

其他尚不能确认无用的坐标/数据对照、输入审计和可视化验收也暂时保留。所有 `data/` 缓存、共享训练/评估/导出/测试代码及清理前未提交的修改均保留。

本轮移出短pilot、500步配对pilot及RNG失败版、native body/接触/场景集成入口试验、被正式55k替代的动态5k试训、两次未产出权重的三数据集启动尝试，以及6个独立旧脚本和相关日志。详细路径见 [清理清单](CLEANUP_20261009.json)。旧文档中的这些路径作为历史记录保留，其内容已按用户要求永久删除。

仓库外的临时归档已按用户要求永久删除，不再提供恢复。删除内容约 12.86 GiB；详细条目见清理清单。
