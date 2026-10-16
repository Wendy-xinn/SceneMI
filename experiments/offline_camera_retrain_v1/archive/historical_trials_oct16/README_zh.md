# 历史试验归档

这里保存 29 个已完成、已停止或已取消的历史试验，减少 `runs` 中的查找负担。归档不等于全部失败，成功诊断、早期基线及必要候选也按原状态保存。

当前入口见 [runs/INDEX_zh.md](../../runs/INDEX_zh.md)。旧模型准确身份是 `native_dynamic_scene20_contact_55k_oct07/last.pt`；新版持续主线为 `native30_gait_v2_ego55k_oct15`，两者仍在原 `runs` 目录。EXO、当前配对对照、主线依赖和活跃回放也留原路径。

[MOVED.json](MOVED.json) 逐项记录原路径、新路径、完成证据和 Git 中小记录的移动映射。目录使用同一文件系统的原子 rename，文件内容及模型/优化器状态保持原样。原配置、日志中的绝对路径作为历史证据保留，没有重写源码指纹。

历史配置和结论可以直接阅读；记录内部保留的原始跨目录链接可按 MOVED 查找。重跑使用原绝对路径的历史协议前，先确认没有同名新试验，再从项目根目录恢复相应目录：

```bash
mv experiments/offline_camera_retrain_v1/archive/historical_trials_oct16/<目录名> experiments/offline_camera_retrain_v1/runs/<目录名>
```

此次删除的是另行审计的冗余周期权重、与 last 模型相同的 best 以及损坏的中断 ZIP；最终模型、指标、配置、身份和种子保留。历史中间训练步权重删除后不能直接重跑，清单见 [CLEANUP_EXECUTED.json](../../runs/run_inventory_oct16/CLEANUP_EXECUTED.json)。主线仍需的 359 项 root/model Python 源码保持原位，未因名称看似陈旧而删改。
