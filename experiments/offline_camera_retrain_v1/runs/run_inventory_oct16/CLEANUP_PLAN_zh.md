# 10月16日 runs 清理盘点与执行建议

只读盘点完成。清理前 84 个一级目录，逻辑 143.09 GiB、实际块占用 143.20 GiB。此文及清单尚未删除或移动任何文件。根盘当时只余约16 GiB；checkpoint占用属于磁盘，不是训练显存。

## 先保住对比主线

用户所说的旧55k准确身份是 `native_dynamic_scene20_contact_55k_oct07/last.pt`，步数55000，EXO协议记录文件SHA `9d771510c16af5d71b8d4f8ae6c4269517814ebb382a0884e84a2ddb4f707363`。两个 `canonical_smpl_*55k` 是更早坐标/规范版本，不能误当当前旧55k。正确旧55k保留完整Adam/RNG，当前新版 `native30_gait_v2_ego55k_oct15` 是唯一持续训练主线；不因清理中断、换loss、重置学习率。旧55k历史TEST重叠及不同采样/输入限制保留，不能直接将其历史指标当正式同协议新旧胜负。

## 可立即清理的第一批

1. `CANDIDATE_PERIODIC_SNAPSHOTS.json`：68份已完成阶段采样的 `evaluations/step_*/inference.pt`，无优化器，合计 30.77 GiB。保留每个run对应last模型及各步指标/身份/seed；删除这些文件后，历史指标仍在，但不能任意复跑某个中间训练步权重。`RETAINED_MODELS_AUDIT.json`记录全部关联last的模型SHA和有限性，满足先核验保留模型的要求。
2. `CANDIDATE_DUPLICATE_BEST.json`：15份best与同目录last同step、全部模型tensor SHA一致，可删best，合计 18.44 GiB。包括旧55k best==last，旧last不动。两个canonical55k的best52000与last55000不同，不在本批。
3. `CORRUPT_CHECKPOINTS.json`：turn_balance旧目录两份损坏ZIP，已无法torch.load；没有可恢复用途。其余模型/小记录保留，不将损坏文件当正常模型归档。

前两批互不重叠，直接释放约 49.21 GiB。删除操作需另写执行前后文件统计、磁盘余量与成功/失败清单；本审计未执行删除。

## 其余失败/废弃尝试

优先先清快照与重复best；余下已结束且不计划恢复的失败候选按照AGENTS去Adam，保留模型、全部配置、评估、身份、seed和RNG。不要只凭run目录名判断失败；如 `native30_overfit_diagnosis` 的TRAIN过拟合验证通过，属于完成诊断而非失败。四臂IK-off暂停2979不是完成，若不计划继续需明确记录放弃恢复后再决定去Adam。当前主线、旧55k、control、有用候选不去Adam。

`report55k_fullval_oct07` 与 `report55k_native20_fullval_oct07` 是更早canonical模型的全量生成数组，各约3.1 GiB，可在保留小汇总/逐窗指标/身份后清理其可重建motions/*.npz；这会丢失原生成逐帧数组，应在删除记录标注。正确旧55k的 `full_validation/motions` 和 `scene_effect_demo` 暂留，便于对比与证明场景作用。当前主线validation_latest、当前对比数组、旧EXO回放和8000步正被server读取的数组均不入这批。

## runs 外归档与代码

`ARCHIVE_PLAN.json`逐项给出 73 个可移出runs的历史目录，目标 `experiments/offline_camera_retrain_v1/archive/failed_trials_oct16/`。归档名称只是收纳，不改变每项原有成功/失败/未完成状态。先清大文件，再移动保留下来的compact记录与所需evaluation-only模型，写MOVED.json及恢复命令。当前保留目录如下：

- `native_dynamic_scene20_contact_55k_oct07`：用户所指旧55k：55000步，20Hz绝对x0；EXO冻结先验依赖其last.pt。历史含官方TEST重叠，仅开发对照，不能写成干净官方benchmark。
- `native30_gait_v2_ego55k_oct15`：唯一持续训练主线，30Hz参考残差，目标55000；last.pt、Adam/RNG、prepared、config与监控必须留原路径。
- `legacy55k_exo_long_oct15`：旧55k独立EXO7831步对照、可恢复模型和8791回放。
- `legacy55k_exo_residual_oct15`：EXO匹配scene/no-scene/参考保护对照与long起点，保留最小对照模型及协议。
- `native30_final_review_oct16`：当前主线仍导入io_recovery.py；本轮control/lean/foot/IK-off开发对照及结论。
- `native30_physics_schedule_probe_oct15`：当前主线run_control.foot_metrics_source严格SHA引用run_probe.py，不得删移该脚本。
- `native30_gait_v2_arch_review_oct15`：当前主线工程准备来源证据，prepared只作provenance。
- `ego_exo_surface_continue_oct11`：全量主线不读取该旧3k权重；config.diagnostic_mount_defaults仍指向TRAIN_camera_mounts.json，保留小JSON与来源。旧3k可作为历史对照归档。
- `native30_fresh_ego_oct14`：新版旧目标10000步对照；当前8788/8789两个可视化进程读取此目录，未停进程前不得移动回放。
- `native30_gait_restore_probe_oct15`：保存旧55k精确复跑/配方、100+100监督配对结果及control10100，有用的架构/监督比较。
- `exo_occlusion_review_oct12`：当前主文档与研究结论索引。

当前effective_config严格封存359份root/model Python源码；其中很多看起来是旧诊断脚本，但删掉或改动会使主线重启来源核验失败，不能趁训练中泛删根目录.py。runs里的source_snapshot副本可整体压缩/移入归档，不改源码内容；只删可重建__pycache__。历史run的absolute paths/源码SHA原封保留，运行历史协议前按MOVED恢复到原路径。主文档可更新历史相对链接或指向归档索引，但不能为省目录修改活跃源码和有效配置。

源数据归档、manifest、raw30/native缓存、contact标签、body/SMPL资产以及active scene/runtime caches均不清。当前data_sha256没有runs内数组，然而main仍直接加载final_review/io_recovery.py与physics_schedule_probe/run_probe.py；这两脚本及SHA协议不可移动。
