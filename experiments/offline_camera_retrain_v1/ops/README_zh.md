# 实验运维

当前实验、曲线、验证和权重统一从 [runs/INDEX_zh.md](../runs/INDEX_zh.md) 进入。

- `checkpoint_guardian.py`：独立保存一个按验证指标选择的模型副本；[选择口径](CHECKPOINT_SELECTION_zh.md)。
- `scene_causal_check.py`：完整RICH验证及独立复核通过后，配对区分学习式场景条件与物理修正的贡献；[证据说明](EVIDENCE_STATUS_zh.md)。
- `cleanup_oct16_evening/`：此次清理的精确文件清单、依赖核验和实际执行记录。

工具放在本目录；权重和预测仍留本地实验目录。已退役工具随其完整实验移到 archive，保留历史重现信息。
