# 10月16日 runs 清理记录

已删除 85 份文件，释放 **50.06 GiB**；29个已完成/取消/拒绝的历史试验目录已逐项同文件系统rename至 `archive/historical_trials_oct16/`。原始盘点84个目录、逻辑143.09 GiB/磁盘143.20 GiB；删除后根盘空闲约65.1 GiB。归档移动本身不释放磁盘，释放量来自删除冗余和损坏文件。

## 删除了什么

- 68份历史周期 `evaluations/step_*/inference.pt`，均无优化器；各步指标、身份、seed及各run最终last保留。
- 15份与同目录last同step、模型tensor SHA相同的best；仅删重复副本。
- turn_balance内2份损坏ZIP checkpoint，已无法正常读取。

[执行记录](CLEANUP_EXECUTED.json)逐项保存原路径、大小、类别和成功删除结果；[保留模型审计](RETAINED_MODELS_AUDIT.json)核验11份last的完整模型SHA及有限性，[重复best审计](CANDIDATE_DUPLICATE_BEST.json)保存15对模型一致证据。[清理后来源和训练核验](POST_CLEANUP_SOURCE_AND_TRAINING_AUDIT.json)用于检查来源SHA及实际主线进展。历史中间步权重已删，原指标仍在，不能再直接加载这些中间权重。

## 保留的重点

正确旧55k是 `native_dynamic_scene20_contact_55k_oct07/last.pt`，55000步、20Hz绝对x0；EXO协议记录原文件SHA `9d771510c16af5d71b8d4f8ae6c4269517814ebb382a0884e84a2ddb4f707363`。该last及完整Adam/RNG不变，两个canonical_smpl的55k不是此模型。旧55k存在历史官方TEST重叠，旧/新条件、FPS和样本预算不同；正式比较须同协议重评。

新版 `native30_gait_v2_ego55k_oct15` 是持续主线，目标55000；训练配置、模型、输入、监督、Adam/RNG和LR计划未因清理改变。旧EXO long7831及最小匹配对照、当前control/有用候选、fresh10k参考和正在读取的回放保留。新TensorBoard/验证截图目录 `native30_live_monitor_oct16` 保留原路径。

主线仍直接导入 `native30_final_review_oct16/io_recovery.py` 和 `native30_physics_schedule_probe_oct15/run_probe.py`，不可删移。full-data准备来源和diagnostic mount小JSON保留证据。root/model源码359项SHA封存，未删除或修改旧.py；否则会影响严格来源恢复核验。源数据、manifest、raw/native人体与观察数组、contact标签、SMPL/body资产和活跃场景/runtime caches均未清理。

## 历史试验已归档

[ARCHIVE_PLAN.json](ARCHIVE_PLAN.json)记录29个首批归档对象及其完成/取消/拒绝证据；[MOVED.json](../../archive/historical_trials_oct16/MOVED.json)记录实际移动路径与时间。含control/baseline的历史目录、状态不明确目录均未扩大移动。归档不等于失败，成功的诊断也按原状态保留。

配置、指标、身份、seed、源码和剩余模型整体保留。所有原协议绝对路径与源码内容未重写；重跑某历史试验前，按MOVED映射把对应目录从 `archive/historical_trials_oct16/<name>` 恢复至原 `runs/<name>`。此动作与被删除的冗余中间权重不同：归档目录可原样移回。

## 不再扩大压缩

本轮**没有进一步去除任何保留checkpoint的Adam**。128步late_four虽未通过身体坐标脚位门槛，其模型及约0.914 GiB优化器亦保留；[SMALL_REJECTED_MODEL_REVIEW.json](SMALL_REJECTED_MODEL_REVIEW.json)标注action=deferred。current128仍作控制组；soft_body/neck_parent曾明确推荐为有用候选，bounded/joint_all是位置adapter基线。short500和被用户停止的cached300缺少充分放弃恢复证据，也保留。空间已经释放，主线优先，不为额外省空间扰动控制组和可恢复状态。

两个早期canonical全量评估目录约6.16 GiB生成npz只列过候选清单，**未执行删除**。正确旧55k的full_validation原数组和scene_effect_demo也保留。
