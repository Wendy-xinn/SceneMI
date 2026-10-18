# 实验运维

当前实验、曲线、验证和权重统一从 [runs/INDEX_zh.md](../runs/INDEX_zh.md) 进入。

EGO 场景与主干融合实验复用以下模块入口，案例身份放在不可覆盖的 PLAN 中，不在 runs 复制训练或评估脚本：

- `python -m experiments.offline_camera_retrain_v1.ops.scene_attention_trial {prepare,preflight,train,audit}`：同一 warm-start 的原结构/场景 cross-attention 配对训练，`train --arm control|attention --stop-step 500 --resume` 恢复包含优化器和 RNG 的单个 `last.pt`。
- `python -m experiments.offline_camera_retrain_v1.ops.ego_scene_review {prepare,evaluate,recheck,report} --out <目录>`：同观测/参考/噪声的场景消融；新候选用新目录和显式 `--models <JSON清单>`，声明 checkpoint 路径、SHA256、步数、架构和场景 arms。`evaluate --resume` 恢复已保存的逐窗结果，`recheck` 在独立 CPU 进程重读原数据复算。
- `python -m experiments.offline_camera_retrain_v1.ops.scene_feedback_policy {prepare,evaluate,recheck,compare} --out <目录>`：复用上项评估，只改变学习式反馈的许可或开关，保留场景条件；与已完成原策略做精确输入配对。

本轮设计、完成结果与实验边界见 [EGO 场景/attention 报告](../runs/native30_ego_scene_attention_trial/REPORT_zh.md)。未运行完整 TEST，短程比较不自动替换正式模型；封印源码在其训练和依赖评估完成前保持不变。

统一运行 `python -m experiments.offline_camera_retrain_v1.ops --help`。子命令 `status`、`view`、`best`、`scene-check` 分别读取状态、打开Viser、保存best和执行登记的配对检查；`motion-check`、`support-check` 执行动作约束实验。已有后台任务不需要重复启动。

- `motion_viewer.py`：同步回放已有预测、GT和场景，案例来自数据清单；以后换例子复用同一入口。[Viser](http://localhost:8800)。
- `checkpoint_guardian.py`：独立保存一个按验证指标选择的模型副本；[选择口径](CHECKPOINT_SELECTION_zh.md)。
- `scene_causal_check.py`：完整RICH验证及独立复核通过后，配对区分学习式场景条件与物理修正的贡献；[证据说明](EVIDENCE_STATUS_zh.md)。
- `rich_test.py`：`ops rich-test prepare/evaluate/recheck` 使用VAL事先选定的冻结权重评估官方RICH TEST，逐窗保存、可恢复、独立CPU复算；[本轮协议](../runs/rich_test_ego41k/README_zh.md)。
- `adapt.py`：`ops adapt prepare/preflight/train/status/recheck` 在同一共享扩散模型上做有限预算的EGO/EXO适配；准备与实际反传预检分开，验证前后使用相同清单；[本轮配置与边界](../runs/native30_ego_exo_adapt_v1/README_zh.md)。
- `cleanup_oct16_evening/`：此次清理的精确文件清单、依赖核验和实际执行记录。

工具放在本目录；权重和预测仍留本地实验目录。已退役工具随其完整实验移到 archive，保留历史重现信息。

2026-10-17进一步实际删除106个已退役根入口及1份完全重复的历史审计脚本，根目录Python文件352→246；其余模块没有宣称全部重构完成。[执行记录](ROOT_SCRIPT_RETIREMENT_EXECUTED.json)包含逐文件身份及清理前后活动EXO98项/TEST95项封印复验。[单个源码恢复包](../archive/source_bundles/README_zh.md)保留原55k的365项源码；当前仍封存使用的`runs/native30_final_review_oct16/io_recovery.py`保留原路径，迁移须在后续新阶段准备时完成。

# 冻结权重的动作约束实验

使用 `python -m experiments.offline_camera_retrain_v1.ops motion-check` 运行同权重、同输入和同 seed 的头部锚定/弱重力方向先验配对检查；`--prepare-only` 先登记清单。`motion_ablation.py` 负责实验、`support_constraints.py` 负责无GT的候选约束、`support_metrics.py` 负责只在评价阶段读取GT的指标。案例与结果放 `runs/native30_live_monitor_oct16/support_ablation/`，不往runs复制Python脚本。

本轮四臂的重力项在原头部约束之前应用；它是有死区与角度限幅的root直立先验，不是牛顿重力或接触动力学。选取了弯腰、俯卧撑和瑜伽录制的窗口，但实际最大root倾角未超过60°，尚未验证倒立安全；不能把倾角变小当作成功。当前训练、既有全RICH评估与场景检验不改配置。

`python -m experiments.offline_camera_retrain_v1.ops support-check --out <新目录>` 在已有baseline输出上运行CPU可见支撑关节投影，不重新采样或训练；原结果目录拒绝覆盖。当前候选已因垂直抖动增加而否决。40次配对采样及4次CPU后处理的完整结论见[决策记录](../runs/native30_live_monitor_oct16/support_ablation/DECISION_zh.md)，[同权重Viser](http://localhost:8801/)保留四组回放。

`python -m experiments.offline_camera_retrain_v1.ops motion-trace` 对已冻结的19k权重记录原始FK、学习反馈、整身贴头、IK和物理修正阶段，排除proposal和物理候选内部约束。首轮4次采样和1次无监控控制已完成，默认结果目录不可覆盖。该诊断只解释同一采样轨迹中的阶段变化，不替代删除组件的完整消融。[阶段报告](../runs/native30_live_monitor_oct16/foot_stage_diagnostics/REPORT_zh.md)、[训练趋势与下一阶段建议](../runs/native30_live_monitor_oct16/foot_stage_diagnostics/TRAINING_AND_NEXT_STAGE_zh.md)。

脚步配对短训复用 `ops foot-trial`，实现放在 `paired_foot_trial.py`，不修改封存的训练器。`prepare` 登记实验，`run` 顺序执行两臂，`status` 读取状态，`audit` 验证1000步实际输入、监督、场景几何、Adam/RNG/sampler恢复和原55k学习率。已有后台任务无需重复启动。[本次状态](../runs/native30_live_monitor_oct16/paired_foot_trial/STATUS.json)；[固定起点与预算](../runs/native30_live_monitor_oct16/paired_foot_trial/PARENT.json)。

`python -m experiments.offline_camera_retrain_v1.ops.foot_trial_evaluation --out <评价目录> --models <模型清单> --only <已完成模型>` 从清单验证checkpoint确切步数后冻结，并分阶段恢复评价。三模型统一physics OFF主比较，五个核心窗口另交叉ON；保留GT幅度、双足运动RMS、stride、脚速、根/脚高度及同一可见局部表面指标。14窗含12个不同RICH录制和两个其他数据集录制，属于开发验证，不代替全RICH或TEST成绩。[本次评价报告](../runs/native30_live_monitor_oct16/paired_foot_trial/evaluation/REPORT_zh.md)。
