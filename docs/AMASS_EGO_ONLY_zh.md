# AMASS EGO-only 对照

2026-10-23 北京时间 00:48 从随机权重开训，05:50 完成 55k 更新，全量验证选模也已结束。用户明确选择从零训练，而非从统一模型续训。CPU/CUDA 的真实 batch 8、输入隔离及逐位恢复均通过，23 项配置、模型、数据和指标回归检查通过。训练保留可完整恢复的 `last.pt`，独立全量验证选模保留一个 `selection/best.pt`。

完整 [验证候选表](../runs/amass_ego128/METRIC_TABLES_zh.md) 和 [测试报告](../runs/amass_ego128/evaluation/REPORT_zh.md) 已整理。完整验证集选定 50k 后，固定测试清单 344 窗口的全局 MPJPE 为 **150.18 mm**、标准 PA 96.30 mm、head 63.92 mm；最终 55k 的测试 MPJPE 为 **155.07 mm**。688 份预测数组及 11 项指标通过独立 NumPy 复算。论文引用值与本机结果分列，三个官方基线的同清单、同传感器重跑仍未完成。

验证选模仍为 337 条原始动作、2,222 个窗口：50k 动作等权 MPJPE 149.38 mm，55k 为 153.35 mm。40k 后出现平台附近波动，不把 50k 比 40k 的微小差值解读为显著进步。验证的家族宏平均不能与测试的窗口均值混用。

## 与统一版的关系

配置为 [amass_ego128.yaml](../configs/amass_ego128.yaml)。相对于已完成的统一版，唯一训练配置变化是 `mode_probabilities: {ego: 1.0, exo: 0.0}`。仍为 AMASS train-only、128 帧、30 Hz、batch 8、55k 更新、相同种子、学习率、损失与预测形状接口；场景关闭，相机仍为双眼中点前方 2 cm。不给模型真实 rest、betas 或人体关节。没有从旧检查点加载模型、优化器或随机状态。

这比较的是**同一总更新预算分配给 EGO 或 EGO/EXO 混合**的结果。EGO-only 的 EGO 样本约为混训的两倍；模式相关的随机数消耗也使后续样本顺序不同。即使 EGO-only 更好，也不能单独归因为 EXO 的负迁移。要隔离这一原因，仍需控制 EGO 样本量的额外实验。这里只报告单训练种子的对照，不宣称已证明结构优劣。

固定开发监测保留旧版的四个 val 动作、12 个条件采样，包含 EXO 评测；**EXO 不参与任何正式优化更新**。这些少量监控仍不能作为全验证收敛或视觉质量结论。正式训练不读取 test，不自动运行 GRAIL 或下一阶段。

## 全验证集选模

用户要求扩大选模验证范围后，新增 [selection.py](../scenemi/selection.py) 和 [固定协议](../configs/amass_ego_selection.json)。不修改已封存、正在运行的训练代码及随机状态。独立进程从 11k 固定快照开始，随后按 15k、20k、…、55k 检查已保存权重；尚未接入选模前已被覆盖的权重无法补评。训练停止或评估跟不上时，以实际捕获的步数为准。

覆盖 AMASS **全部 337 条合格 val 原始动作、4 个子库、2,222 个 128 帧窗口**。步长 128，追加末尾对齐窗口以覆盖所有帧，不重复计镜像增强。固定 yaw=0、DDIM 20 步、seed=777、batch=4、每窗口一次采样；不是 best-of-k。主分数是每条动作先平均其窗口，再对动作等权的全局 MPJPE；同时记录各子库、根、头、标准 PA、速度和 jerk 指标。不用 test 选模型。

**“完整”的准确范围是全部合格序列和全部时间帧，不是穷举全部窗口起点。**
2026-10-27 对冻结清单逐序列核验：262,776 / 262,776 帧被至少一个窗口覆盖，
没有未覆盖时间段；EGO 选模清单与当前 EXO 续训、从零联合训练的封存清单完全一致。
若改用步长 64 加尾窗，为 3,935 窗口；穷举步长 1 的全部合法起点为 219,977 窗口，
约为当前计算量的 99 倍。后者相邻窗口共享 127 帧，不代表 99 倍独立验证数据。
当前结果没有验证改变窗口起点后候选排名是否稳定，不能将全帧覆盖理解为该项稳定性保证。
上述规模核验没有更改运行中训练或选模协议。
逐项证据：[验证覆盖审计](../runs/amass128/VALIDATION_COVERAGE.json)。

11k、15k、20k、25k、30k、35k、40k、45k、50k、55k 的全验证均已完成。完整日志、实际权重 SHA、每窗口来源和误差在 `runs/amass_ego128/selection/`。`BEST.json` 是已完整评估候选中的最佳；`best.pt` 保留完整优化器和 RNG，可恢复。快照以硬链接固定文件版本，训练原子替换 `last.pt` 不会改变评估内容。仅保留一个 best 和一个 last，不累计编号权重。

```bash
# 本次选模已完成。仅在确有未完成候选时恢复；文件锁拒绝重复实例。
python -m scenemi.parallel_resources run --device cuda \
  --memory-gib 6 --cpu-count 2 --allocator-gib 4 -- \
  python -u -m scenemi.selection --watch
```

选模进程只与已识别的本用户 SceneMI 任务共享 GPU；未知或另一发行版的 GPU PID 会拒绝准入。训练的旧少量监测日志仍存在，但不参与最佳权重判断。

与旧混训版做固定预算消融时，主比较仍使用双方最终 55k；新分支的验证最佳作为额外结果单列。旧混训中间权重已被原子覆盖，无法重选它的验证最佳；不能将新 best 对旧 final 的差异全部归因为 EGO/EXO 配比。

## 入口与恢复

`scenemi/ego_training.py` 是固定 EGO-only 协议的薄适配器，复用封存的模型、损失、数据、优化器、保存与正式训练循环。它校验配置仅有上述一项差异，重新核验统一版源码、数据索引和最终检查点身份。旧训练依赖不修改，输出固定隔离到 `runs/amass_ego128`，输入复用已审计的 `runs/amass128/index/complete.json`，不重做 AMASS。

当前任务已经完成，以下是协议复现/中断恢复命令，**不要重复启动已完成任务或覆盖已有结果**。使用 `scenemi` 环境，在仓库根目录执行：

```bash
python -m scenemi.ego_training config

python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 10 --cpu-count 2 -- \
  python -u -m scenemi.ego_training preflight --device cpu

python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cuda --memory-gib 12 --cpu-count 4 -- \
  python -u -m scenemi.ego_training preflight --device cuda

python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cuda --memory-gib 12 --cpu-count 4 -- \
  python -u -m scenemi.ego_training train --device cuda

# 仅在已有任务停止后，恢复本分支的 last.pt；不读取统一版权重。
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cuda --memory-gib 12 --cpu-count 4 -- \
  python -u -m scenemi.ego_training train --device cuda --resume
```

此适配入口沿用原资源包装器的独占启动检查，训练进程仍安装 6 GiB PyTorch 分配上限；不绕过拒绝或干预其他进程。`INITIALIZATION.json` 记录 step 0、空优化器及未加载检查点；`CONTROL.json` 保存统一版身份与比较边界，`CONTRACT.json` 保存新运行的源码/配置/环境封印。改源码后不能将其伪装成原任务的无缝恢复。

## 后续场景数据

本次 AMASS 训练不混入 GRAIL。另已从 694 条坐下训练候选中，按固定 seed 20261023、椅子家族轮转选出 400 条录制，覆盖 149 个椅子家族。为每条最早合格的 128 帧窗口生成一个全椅表面采样的合成场景条件，400 份文件及来源掩码已核验；所选录制共保留 9,539 个重叠合格窗口，尚未为全部窗口接入训练读取器。

这批条件只用于后续合成增强，不是真实相机已经看到全椅的证据，也不是 400 种独立椅子。说明见 [GRAIL 支撑与椅子来源](data/GRAIL_SUPPORT_V2_zh.md)。场景分支使用独立的原生体形适配和读取器，进度见 [原生场景分支](SCENE_NATIVE_zh.md)。坐下/蹲下歧义与真实观测泛化仍需独立验证。
