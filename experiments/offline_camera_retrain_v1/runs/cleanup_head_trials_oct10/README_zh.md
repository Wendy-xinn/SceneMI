# 已完成头部短训清理

本次释放18.827GB。范围仅限typed_head_track、bounded_head_adaptation和position_adapter已结束短训：移除从未用于评估的best.pt，对失败的最终权重移除优化器张量；逐个验证保留模型张量完全相同。

保留全部配置、结果、逐窗预测、随机状态及审计记录。bounded/joint_all当前用作对照和warm-start，保留完整优化器。正式55k及源数据依赖不在清理范围。

精简后的last.pt支持推理、既有审计和init-from，不支持继续resume；内部storage_role已明确标记。逐文件大小见cleanup.json。清理是幂等的，但不要重复执行覆盖最初清理记录。
