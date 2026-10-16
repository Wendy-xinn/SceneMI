# 当前实验入口

当前重点是旧55k与新版持续训练的对比。目录名含55k只表示目标或历史命名；实际步数以checkpoint/训练状态为准。

| 用途 | 入口 | 说明 |
|---|---|---|
| 新版持续主线 | [训练状态](native30_gait_v2_ego55k_oct15/continuous_training_oct16/STATUS.json) | 30Hz、参考残差扩散、官方划分；持续目标55000，保持原模型/输入/监督/Adam/RNG |
| 曲线与验证截图 | [监控目录](native30_live_monitor_oct16/) | TensorBoard、各监督项、训练步数/时长；固定验证截图保存于visuals |
| 旧55k对照 | [结果](native_dynamic_scene20_contact_55k_oct07/RESULTS_zh.md) | 正确旧模型是此目录last.pt：55000步、20Hz绝对x0，保留可恢复状态和必要场景demo |
| 旧55k＋独立EXO | [状态](legacy55k_exo_long_oct15/STATUS.json) | 7831步EXO残差修正，旧扩散冻结先验；与场景条件扩散须区分 |
| 新版近期配对比较 | [阶段报告](native30_final_review_oct16/INTERIM_REPORT_zh.md) | control/精简/逐脚监督/IK-off，不覆盖持续主线；io_recovery.py仍被主线使用 |
| 脚步监督/旧配方复核 | [复核目录](native30_gait_restore_probe_oct15/) | 旧55k复跑、10100步control及监督配对证据 |
| 研究记录 | [架构和实验主文档](exo_occlusion_review_oct12/EXPERIMENT_AND_ARCHITECTURE_zh.md) | 历史结论与限制 |
| 清理和恢复 | [清理计划](run_inventory_oct16/CLEANUP_PLAN_zh.md) | 原始占用、保留模型核验、实际删除记录及历史目录归档计划 |

旧55k具有历史官方TEST重叠，旧/新还存在FPS、条件、样本预算等差别；历史表格可帮助诊断，正式胜负须同协议重评。两个更早canonical_smpl的55k目录不等同上述正确旧55k。

主线依赖 `native30_physics_schedule_probe_oct15/run_probe.py` 的脚部监控函数及 `native30_final_review_oct16/io_recovery.py`，这些目录不能作为失败记录整删。部分准备来源与mount JSON也保留小证据。root/model源码已有359项SHA封存，训练中不泛删旧.py。

29个已完成/取消/拒绝的历史试验目录已移至 [historical_trials_oct16](../archive/historical_trials_oct16/)，实际路径与恢复映射见 [MOVED.json](../archive/historical_trials_oct16/MOVED.json)。归档不等同失败，配置、指标、身份、seed、代码和剩余模型保留。重跑历史绝对路径协议前，把对应目录恢复到原 `runs/<name>`，不改原配置/源码SHA。本轮删除85份冗余/损坏权重，释放50.06 GiB；没有再去除控制组或保留候选的Adam。源数据、人体资产和活跃场景缓存保留。
