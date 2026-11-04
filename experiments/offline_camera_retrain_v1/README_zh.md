# 既有 EGO/EXO 实验与兼容实现

本目录保存已训练的 30 Hz / 192 帧模型、数据导出、评估和运维工具。新 AMASS 128 帧阶段从仓库顶层 [scenemi/](../../scenemi/) 和 [configs/amass128.yaml](../../configs/amass128.yaml) 进入；[代码导航](../../docs/CODE_MAP_zh.md) 说明两者边界。

## 常用入口

- [当前模型、评测和 GRAIL 状态](runs/INDEX_zh.md)
- [完整 noisy 验证总报告](runs/native30_final_metrics_v1/full_validation_noisy/OVERALL_REPORT_zh.md)
- [指标表](runs/native30_final_metrics_v1/full_validation_noisy/METRIC_TABLES_zh.md)
- [运维说明](ops/README_zh.md)
- [本轮清理记录](../../docs/maintenance/README_zh.md)

已有模型、报告和 GRAIL 任务保留原路径。运行配置以 run 内冻结文件为准，不把顶层 AMASS YAML 用于旧训练器。新配置通过验证不表示真实数据已准备完成。

## 目录职责

| 位置 | 职责 |
|---|---|
| `ops/` | 既有可复用 CLI、资源检查、评估、可视化与 GRAIL；新任务继续复用 |
| `runs/` | 八组受保护的模型/评测/数据任务，以及因冻结路径需要保留的历史小记录 |
| `data/`、`configs/` | 旧阶段的有效数据与配置资产，不能当实验输出清理 |
| `archive/` | 历史记录及精确源码恢复包；退役权重和可重建数组已删除 |
| 根目录 `*.py` | 67 个封存兼容文件；新实现不再向这里平铺 |
| 顶层 `tests/legacy/` | 本轮迁移的 22 个回归测试和恢复的旧配置夹具 |

本轮永久删除 157 个退役源文件及其清单内生成资产，根源码 246 → 89；再将 22 个非封存测试移至顶层，根源码变为 67。保留的源码/模型 212 项已逐一 SHA256 复验；还有一个测试属于封存依赖，保留原路径。当前模型依赖解除前，不为目录整洁改写其封印。

历史审计描述的是清理前状态。退役运行的权重和预测已不存在，旧菜单、旧 resumable 标记不代表仍可直接运行；查看当前资产请使用新索引和清理记录。
