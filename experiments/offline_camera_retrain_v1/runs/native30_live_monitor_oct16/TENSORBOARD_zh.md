# TensorBoard 只读监控

`tensorboard_bridge.py` 是独立 CPU sidecar，只读训练已有的 JSONL 和验证截图。
没有修改训练源码、监督系数、配置、checkpoint、优化器、随机状态或训练进程。

当前回填两个模型：

- `new_native30`：正在持续训练的新版，原始 optimizer step 作为横轴。
- `old55k`：`native_dynamic_scene20_contact_55k_oct07`，原始记录到 55000 步。
- 对应 `__active_seconds` run：总 loss 的横轴改为累计训练循环秒数。整数秒内有多个真实更新时可以出现相同横轴秒数；原始 optimizer step 不变。

建议先打开 **Custom Scalars**：预设看板按进度、扩散与身体监督、观测与场景、固定验证、脚步、轮换验证、速度与时长排列，避开大量逐序列明细标签。看板只引用原曲线，不改变其数值或权重。Scalars 中可按以下标签筛选：

- `train/loss`、`train/gradient_norm`、`train/lr`。
- `train/terms/base/*`、`train/terms/observation/*`、`train/terms/geometry/*`：新版全部已记录监督项。
- `train/losses/*`：旧版原始 loss 字段。
- `validation/fixed/*`、`validation/rotating/*`：固定窗口趋势和轮换覆盖分开。
- `feet/fixed/*`、`feet/rotating/*`：脚位、速度、幅度与左右脚关系。
- `time/train_update_hours`：逐步真实 `update_s` 的累计小时；包括数据准备、前反传和 Adam 更新，不包括独立验证。
- `time/train_compute_hours`、`time/train_preparation_hours`：相应计时累计值。
- `time/validation_hours`：原 `foot_monitor.jsonl` 的真实验证调用耗时累计值，只记一次。
- `time/active_loop_hours`、`time/active_loop_minutes`：原 `elapsed_s` 按恢复会话累计，是已记录循环计时的下界。trace 在每步验证前写入；若某会话在最后一次验证后退出，该次验证及后续保存不进入最后一条 trace，因此不能把跨会话相加视为完整训练墙钟。`__active_seconds` 镜像有相同限制。精确耗时优先查看逐步 update/compute/preparation 累计与独立 validation 累计；不猜测离线暂停或完整数据校验等启动耗时。

loss 数值和层级完全来自原 JSON。`raw_x0`、`gait_raw_*` 等名称照原样保留；部分 total 已包含原实现的权重，因此桥不会再乘系数。名字中的 raw 可能指修正前的预测，不能一律解释成“未乘权重”。不同架构的 loss 定义、采样和 batch 不同，不能用总 loss 大小判断哪个模型更好。旧版没有逐步 update/compute/preparation 秒数日志，因此不会补造这些曲线。

历史 trace 没有逐步绝对时间戳。TensorBoard event 的 wall_time 在有原始 `checked_at` 时保留原值，其他事件使用明确标记的导入时间；不要把 TensorBoard 内置 WALL/RELATIVE 横轴当成历史训练时长。真正耗时应看 `time/*` 或 `__active_seconds` run。`provenance/walltime_is_source_absolute` 显示这个区别。

Images 中 `visual/fixed/{dataset}/{clean,noisy}` 展示每个已捕获验证 step 的截图。
每张图的场景可用性与输入可见性以截图和 `visuals/step_*/MANIFEST.json` 为准。

启动（使用现有 scenemi 环境，不需要安装包）：

```bash
/home/wenxin/miniconda3/envs/scenemi/bin/python -u /home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_live_monitor_oct16/tensorboard_bridge.py

/home/wenxin/miniconda3/envs/scenemi/bin/python /home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_live_monitor_oct16/tensorboard_compat_server.py --logdir /home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_live_monitor_oct16/tensorboard --host 127.0.0.1 --port 6006
```

现有 TensorBoard 2.16.2 的 Custom Scalars 接口使用了新版 protobuf 已删除的 JSON 参数。本地 `tensorboard_compat_server.py` 只在 TensorBoard 进程内映射这个参数；没有改安装包，也不接触训练。普通曲线与图片仍走原 TensorBoard 服务。

`--once` 回填到当前末尾后退出，默认每 30 秒读取增量，减少不可变 event 小文件数量。运行 PID 记录在 `BRIDGE_LAUNCH.json`，最新 offset、真实累计计时和进度记录在 `BRIDGE_STATUS.json`。

SQLite 将已读 offset、时钟和 event 内容一次提交，再将不可变批次原子写成 TensorBoard 文件。未完成行等待下一次读取；提交或写文件中途退出均可以恢复，不重复导入记录。恢复后原 trace 若被训练 worker 截断重放，桥会自动重建 generation，时钟重新从当前完整源日志累计，`SessionLog START(step=0)` 清除 TensorBoard 中失效的尾段，原因记在 `REBUILD_EVENTS.jsonl`。为让已打开的 TensorBoard reader 能继续前进，旧不可变文件保留在事件目录，显示内容由新 generation 的 START 清理；不会把两段训练曲线拼接成当前历史。

`test_tensorboard_bridge.py` 包含真实 TensorBoard reader 集成测试，覆盖原 loss 数值、断点不重复、两处写入崩溃、部分行、会话计时重置、源文件回滚以及正在打开的 reader 清理旧点。`BACKFILL_AUDIT.json` 记录一次实际回填和读取核验。
