# 四数据集正式无场景主线

2026-10-28 用户要求先产出正式模型，停止扩展结构和超参数短试验。本阶段授权训练不等于短筛已通过：历史 AMASS 保持和部分 EXO 指标失败继续保留，最终由完整验证判定，不更改原门槛。

- 初始化：已结束的 AMASS 从零 EGO/EXO 混训，完整验证选出的 100k，SHA256 `5145b2e65ac7792ddee225e90e97148c410578c81096c90f4fdb098018a2dafe`。
- 配置：`configs/scene_joint128_replay.yaml`；输出：`runs/scene_joint128_replay`。
- 每窗 128 帧；每次更新 128 个窗口，EGO 64 / EXO 64；总计 2,000 次更新、256,000 个抽样窗口。
- AMASS/EgoBody/TRUMANS/GRAIL 抽样比例为 25%/12.5%/28.125%/34.375%；监督权重分别为 50%/8.3333%/18.75%/22.9167%。采用已完成短筛的保守主干学习率，并保留原生体形和 EXO 适配层学习率。
- EGO 相机连续、体形由观测预测；EXO 使用估计初始化、可靠度、遮挡和短时出画模拟；不输入场景。使用当前筛选范围和 TRUMANS 移动相机；不加入 RICH，不打开独立 test。
- 复用短筛的按窗口索引确定的数据和扩散噪声流，在线预取，不落盘数十 GiB 的训练缓存。断点包含共享主干、适配器、优化器、随机状态和日志偏移。

为先产出模型，先完成训练并保存 8,192、65,536、256,000 窗口后的候选，再依次评估初始权重和这三个候选。最终 best 只在完整评估且通过既有门槛的候选中选择，同时保留 last；若没有候选通过，`BEST_CANDIDATE.json` 明确记录最佳分数候选及失败项，不伪称合格 best。这不保证未评估的训练步中不存在更优权重。

完整验证覆盖固定的全部合格 validation 序列/范围，步长 128 加对齐尾窗：EGO 5,693 窗口，EXO 每条件 5,762 窗口，四条件共计 28,741 次模型窗口评估。分数据集、family 等权聚合，保留全局 MPJPE、PA-MPJPE、根误差和 EXO 可见/修复区域误差。评估按数据集/条件保存可恢复分片，断开后保留已完成的分片。

```bash
PYTHON=/home/wenxin/miniconda3/envs/scenemi/bin/python
$PYTHON -m scenemi.replay.training launch --task prepare --device cpu
$PYTHON -m scenemi.replay.training launch --task train --device cuda
```

准备和开训均经原有 cgroup、CPU、RAM 和共享 GPU 准入；GPU 保留 4 GiB 物理空闲及 1 GiB CUDA 余量。未识别 GPU 进程仍须按仓库共享规则处理，不干预其他任务。训练预计约 1.6–2 小时，完整验证另计；新批量指标实现的全流程速度须实际运行后更新。

用户随后明确授权本次正式任务“按余量共享”，不绑定变化的外部 PID。授权与精确配置、100k 父权重绑定，保存在输出目录的 `SHARING_AUTHORIZATION.json`。本次启动使用 `python -m scenemi.replay.sharing launch --task train --device cuda`；它只替换这次启动的 PID 拒绝规则，保留所有内存余量和 cgroup 检查，不修改已经通过 CPU 验收的训练源码。独立后台服务为 `scenemi-four-corpus-replay.service`，WSL 的 `/usr/lib/wsl/lib` 已加入服务 PATH；服务记录、共享准入和额外启动器源码归档均保存在同一输出目录。

进度文件：`STATUS.json` 记录准备阶段；`training/STATUS.json` 记录真实更新和验证阶段；`selection/step_*/STATUS.json` 记录完整评估进度。训练运行时不修改封存的代码、配置或数据清单。

## 2026-10-28 训练结束与验证恢复

正式优化已完成 2,000 次更新、256,000 个抽样窗口。保留 `training/last.pt` 和 64 / 512 / 2,000 更新的三个候选。训练结束后的自动评估在加载 AMASS 初始权重时失败：该父权重尚无 24 个原生人体/相机适配参数，不能直接严格加载到四数据集模型。三个训练候选均具有完整参数，不需要重新训练。

验证入口为 `python -m scenemi.replay.validation_sharing --ego-batch 16 --exo-batch 32`，后台服务 `scenemi-four-corpus-validation.service`。它核验原训练封存文件，用同一训练随机种子重建完整的初始原生适配器，并严格加载三个完整候选。原训练配置、权重和源码不修改；新增验证执行协议及源码独立保存于 `VALIDATION_PROTOCOL.json`、`validation_sources.tar.gz`。

验证批量候选由 EGO 4 / EXO 8 增至 16 / 32。按原逻辑 batch 4 / 8 生成相同 DDIM 噪声，再合并计算，包含原逻辑尾批的实际大小，避免直接扩大随机张量导致噪声分配变化。先在四来源 EGO 与 EXO tracking / occlusion / gap 共 16 组真实窗口上比较原采样器和合批采样器：最大关节差须不超过 0.1 mm，逐窗口 MPJPE / PA-MPJPE / 根误差差异须不超过 0.05 mm。证据：`VALIDATION_BATCH_BENCHMARK.json`；只有全部通过才开始完整验证。

完整验证仍依次覆盖初始模型与三个候选，每个模型 28,741 个条件窗口，共 114,964 次；EGO 20 步、EXO 8 步 DDIM，数据划分、窗口、合成输入、指标及选择门槛保持原协议。每批更新 `training/STATUS.json`，每个数据集/条件完成后保存可恢复的指标分片。`completed_checkpoint_windows` 是当前检查点的验证进度，与已经完成的 2,000 次训练更新分别记录。最终结果与选择见 `VALIDATION_RUN.json` 和 `selection/BEST*.json`；不能用 batch 性能检查推断模型精度已有收益。

北京时间 17:34，16 组合批一致性检查全部通过，已开始完整验证。最大关节差 0.00420 mm，逐窗口 MPJPE / PA-MPJPE / 根误差的最大差 0.000403 mm；9 项验证入口、共享准入及批量指标测试通过。验收证据在 `VALIDATION_ACCEPTANCE.json`。该记录只代表验证调度可用，不代表最终模型通过保持能力门槛，也不代表完整验证已经结束。
