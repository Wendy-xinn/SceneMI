# 实验运维

当前实验、曲线、验证和权重统一从 [runs/INDEX_zh.md](../runs/INDEX_zh.md) 进入。

统一运行 `python -m experiments.offline_camera_retrain_v1.ops --help`。子命令 `status`、`view`、`best`、`scene-check` 分别读取状态、打开Viser、保存best和执行登记的配对检查；`motion-check`、`support-check` 执行动作约束实验。已有后台任务不需要重复启动。

- `motion_viewer.py`：同步回放已有预测、GT和场景，案例来自数据清单；以后换例子复用同一入口。[Viser](http://localhost:8800)。
- `checkpoint_guardian.py`：独立保存一个按验证指标选择的模型副本；[选择口径](CHECKPOINT_SELECTION_zh.md)。
- `scene_causal_check.py`：完整RICH验证及独立复核通过后，配对区分学习式场景条件与物理修正的贡献；[证据说明](EVIDENCE_STATUS_zh.md)。
- `cleanup_oct16_evening/`：此次清理的精确文件清单、依赖核验和实际执行记录。

工具放在本目录；权重和预测仍留本地实验目录。已退役工具随其完整实验移到 archive，保留历史重现信息。

# 冻结权重的动作约束实验

使用 `python -m experiments.offline_camera_retrain_v1.ops motion-check` 运行同权重、同输入和同 seed 的头部锚定/弱重力方向先验配对检查；`--prepare-only` 先登记清单。`motion_ablation.py` 负责实验、`support_constraints.py` 负责无GT的候选约束、`support_metrics.py` 负责只在评价阶段读取GT的指标。案例与结果放 `runs/native30_live_monitor_oct16/support_ablation/`，不往runs复制Python脚本。

本轮四臂的重力项在原头部约束之前应用；它是有死区与角度限幅的root直立先验，不是牛顿重力或接触动力学。选取了弯腰、俯卧撑和瑜伽录制的窗口，但实际最大root倾角未超过60°，尚未验证倒立安全；不能把倾角变小当作成功。当前训练、既有全RICH评估与场景检验不改配置。

`python -m experiments.offline_camera_retrain_v1.ops support-check --out <新目录>` 在已有baseline输出上运行CPU可见支撑关节投影，不重新采样或训练；原结果目录拒绝覆盖。当前候选已因垂直抖动增加而否决。40次配对采样及4次CPU后处理的完整结论见[决策记录](../runs/native30_live_monitor_oct16/support_ablation/DECISION_zh.md)，[同权重Viser](http://localhost:8801/)保留四组回放。
