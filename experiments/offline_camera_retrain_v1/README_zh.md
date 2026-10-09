# 离线相机轨迹与可见场景条件的 SceneMI 重训

## 目标

从随机权重训练 SceneMI 的 AdaGN Temporal U-Net、全局占据网格 ViT 和逐帧 BPS 分支。推理输入是带显式时空 mask 的可变身体部位轨迹和累计可见静态场景，输出是相机首帧锚定坐标系中的全局 SMPL 动作。当前主要模式把完整相机轨迹放在头部槽位，但同一模型也训练骨盆、稀疏双腕、混合部位和短历史全身条件。此任务不使用原版的整帧关键帧补全协议。

动作张量共 201 维：3 维根平移、22×6 维以单位旋转为中心的残差旋转、22×3 维辅助全局关节。训练监督包括 x0、旋转矩阵、FK 关节位置/速度/加速度、头部轨迹、真值静止脚的滑移约束，以及辅助关节与 FK 的一致性。

## 与后续 tracker 的模型统一（当前决策）

最终交付链路统一采用 ProtoMotions tracker 使用的 **SMPL**，不能把 SMPL-X 和 SMPL 的模板直接混在最终模型里。tracker 的实际刚体顺序是 `SMPL_MUJOCO_NAMES`（24 个刚体，包含 `L_Hand/R_Hand`），而本目录的 201 维扩散表示是 22 个身体关节的紧凑表示，适合当前 SceneMI 训练，但本身不是可直接喂给 tracker 的完整 SMPL 参数。

因此当前 55k/30k 模型应视为“混合 SMPL(-X) 关节表示”的实验模型，而不是最终 tracker-compatible checkpoint。正式版本应按以下顺序处理：

1. 以 `/home/wenxin/projects/ProtoMotions/data/smpl/SMPL_{MALE,FEMALE,NEUTRAL}.pkl` 及其 24 刚体顺序作为唯一 canonical SMPL 资产；
2. 将 TRUMANS、EgoBody 和 RICH 全部转换/拟合到这套 SMPL（RICH 当前导出仍是精确 SMPL-X，不能直接作为最终 SMPL 监督）；
3. 明确 22→24 的输出适配（手部叶节点和 tracker 的 MJCF 顺序），通过 FK/mesh/头脚位置审计后，再从头训练统一版本。

在上述转换完成前，可以继续用当前 checkpoint 做 SceneMI 的场景、mask 和脚步消融；但不应把它的输出直接宣称为 tracker 的最终输入。

## 时间窗口协议

Viser 中看到的 RICH 文件是完整动作片段，不是训练窗口：例如当前三个预览分别为 63、126、80 帧（5 Hz）。它们没有被强行补到 128 帧。当前实验代码为了做长度消融仍支持 64/128/192 三种窗口，验证主协议使用 128 帧，即 5 Hz 下 25.6 秒。

正式加入 RICH 时，建议把所有足够长的动作切成固定 128 帧窗口并使用重叠起点；短于 128 帧的动作不重复最后一帧填充，避免给模型制造人为静止段。后续如果把窗口扩大到 256 帧，应作为新的长度协议和 checkpoint 单独验证，而不是把不同长度混成一个未记录的训练设置。

每个可控关节输入 3 维位置、6 维全局朝向和 1 维 mask。训练 mask 概率为：头部 55%、头部＋稀疏骨盆 15%、头部＋稀疏双腕 10%、骨盆 8%、稀疏双腕 4%、随机混合稀疏部位 3%、短历史全身 5%。包含短历史和随机混合后，约 87% 样本含某种头部观测，同时保留不含头部的训练模式。

当前场景不是点云 cross-attention：全局分支把累计可见表面栅格化为 24×48×48 occupancy 后用 SceneMI ViT 编码；局部分支在 67 个相机中心固定锚点上查询逐帧 BPS，再用 MLP 编码。两者都只依赖相机轨迹和可见地图，不使用真值身体位置查询场景。TRUMANS 加入稳定区间起点的可见动态物体快照；EgoBody 使用整段轨迹模拟可见表面的并集。

完整扫描 mesh 只用于离线模拟 ego 深度，不作为模型输入。候选表面必须进入相机视锥并通过 64×48 像素 z-buffer 后才进入累计地图；当前每 4 帧投影一次（20 Hz 动作下为 5 Hz）。occupancy 只将已观察表面记为 1，未观察区域保持 0（unknown），不会再把未观察区域编码为确认的自由空间。训练入口会检查每条序列的可见性协议、4 帧步长、候选点数和局部查询半径，拒绝非 ego 投影来源或不完整的场景文件。

训练时有 10% 概率同时关闭全局 occupancy 和局部 BPS，而不是只丢弃其中一个场景分支。验证固定比较完整场景与完整移除场景，避免把随机 dropout 当成场景收益。

## 训练前硬性检查

在 SceneMI 仓库根目录运行；必须设置 `PYTHONPATH=.`：

```bash
PYTHONPATH=. /home/wenxin/miniconda3/envs/scenemi/bin/python -m unittest \
  experiments.offline_camera_retrain_v1.test_pipeline -v

PYTHONPATH=. /home/wenxin/miniconda3/envs/scenemi/bin/python -m \
  experiments.offline_camera_retrain_v1.audit_visible_maps

PYTHONPATH=. /home/wenxin/miniconda3/envs/scenemi/bin/python -u \
  experiments/offline_camera_retrain_v1/diagnose_overfit.py --steps 120
```

单元测试检查零旋转残差是否严格解码为单位旋转，以及三组数据的干净真值 FK 是否落在数据审计阈值内。固定批过拟合用于确认完整模型、损失和优化器能够显著降低同一高噪声探针的误差。

## 分阶段训练

2026-10-02 凌晨已启动 `runs/scene5hz_long30k_b4_oct02`：随机初始化、30,000 步、batch=4、500 步 warmup、按完整 30k 计划余弦衰减；头部为主的混合 mask 与 5 Hz ego 可见场景保持启用。后台会话为 `tmux attach -t scenemi_oct02`。`status.json` 记录进程和结束状态，`console.log`/`training_log.jsonl` 记录进度、学习率及梯度范数。

每 1,000 步保存并自动运行 DDIM-20（每域 4 个固定 128 帧样本，五种条件），产物位于 `evaluations/step_XXXXXX/metrics.json` 和 `gallery.html`；首轮评估完成后 `index.html` 自动汇总历次结果。每 5,000 步额外保留一个仅供推理的 `inference.pt`，滚动 `last.pt` 则包含续训所需优化器和随机状态。每次评估失败会让训练明确报错并保留最近 checkpoint。

原 500-step 场景模型已用更新后的指标生成 `baseline500/metrics.json` 与 `baseline500/gallery.html`。新评估用原始 GT 关节计算运动幅度，额外报告根轨迹长度、水平脚速、头路径相对误差；头关节到相机的距离包含解剖偏移，不等同于严格控制误差。可视化为骨架诊断，没有场景网格，不能替代真实场景碰撞检查。

首个 1,000-step 自动评估已成功完成并恢复训练。相同 12 个 DDIM 验证片段中，旧 500-step / 新 1,000-step 的头部误差为 47.20 / 44.39 cm，生成/GT 关节速度比为 0.20 / 0.38，真值静止脚滑为 0.170 / 0.325 cm/帧。新模型去除控制后头部误差为 47.74 cm，表明开始使用控制输入；动作幅度提高但脚滑仍需改善。两次训练的 batch 和学习率计划不同，这组数值用于跟踪进展，不能解释为只增加步数的严格对照。

先运行短 pilot，不直接启动长训：

```bash
PYTHONPATH=. /home/wenxin/miniconda3/envs/scenemi/bin/python -u \
  experiments/offline_camera_retrain_v1/train.py \
  --output experiments/offline_camera_retrain_v1/runs/pilot_1000 \
  --steps 1000 --save-every 500 --validation-samples 4
```

场景消融使用相同 seed 和独立的条件随机数流，确保序列、长度、扩散噪声和身体 mask 配对一致：

```bash
PYTHONPATH=. /home/wenxin/miniconda3/envs/scenemi/bin/python -u -m \
  experiments.offline_camera_retrain_v1.train \
  --output experiments/offline_camera_retrain_v1/runs/paired_full \
  --steps 1000 --save-every 500 --validation-samples 4

PYTHONPATH=. /home/wenxin/miniconda3/envs/scenemi/bin/python -u -m \
  experiments.offline_camera_retrain_v1.train \
  --output experiments/offline_camera_retrain_v1/runs/paired_no_scene \
  --steps 1000 --save-every 500 --validation-samples 4 --disable-scene
```

检查点通过后再从随机初始化、用新目录运行 5k/10k；不要把已按 500/1000 总步数完成余弦衰减的短 pilot 直接续成正式长训，否则学习率会发生不合理跳变。只有固定验证曲线、完整 DDIM 样本和条件消融都合理时才启动 30k。30k 是检查点，不是假定的最终收敛步数。对于原计划总步数不变、只是意外中断的训练，使用：

```bash
PYTHONPATH=. /home/wenxin/miniconda3/envs/scenemi/bin/python -u \
  experiments/offline_camera_retrain_v1/train.py \
  --output experiments/offline_camera_retrain_v1/runs/formal_5000 \
  --resume experiments/offline_camera_retrain_v1/runs/formal_5000/last.pt \
  --steps 5000 --save-every 500 --validation-samples 4
```

完整生成和头部/脚步/支撑面代理指标：

```bash
PYTHONPATH=. /home/wenxin/miniconda3/envs/scenemi/bin/python -u \
  experiments/offline_camera_retrain_v1/evaluate_sampling.py \
  --checkpoint experiments/offline_camera_retrain_v1/runs/pilot_1000/best.pt \
  --output experiments/offline_camera_retrain_v1/runs/pilot_1000/ddim20_validation64.json \
  --length 64 --ddim-steps 20 --samples-per-group 4
```

评估同时运行完整条件、移除场景、移除所有稀疏控制，以及骨盆／双腕控制。短训中条件差异很小是正常的；长训后如果差异仍接近零，应停止并检查模型是否忽略条件。

## 当前已验证结果

- 统一 mask 版完整配置 smoke 已通过前向、反向、checkpoint、验证和条件消融。
- 固定三域 batch 分别使用头部、骨盆和稀疏双腕条件；过拟合 120 步后，高噪声探针总损失从 0.430 降到 0.038，FK MPJPE 从 49.8 cm 降到 12.2 cm，头部误差从 44.5 cm 降到 5.7 cm。
- 50-step 随机 mask／随机全数据 pilot 的双样本验证平均 FK MPJPE 为 78.20 cm；场景与控制消融均已产生非零差异。它只验证混训链路，不能代表生成质量。
- 5 Hz ego 可见地图审计通过：TRUMANS 469/40/58 条、EgoBody 213/99/68 条与源清单严格一致。TRUMANS 可见点数相对旧 1 Hz 地图的中位倍率为 1.82–1.90；EgoBody 因原地图已含密集相机射线，中位倍率为 1.03–1.07。
- 500-step 场景／无场景配对 pilot 已完成。固定时刻去噪验证中场景组 MPJPE 为 50.89 cm、无场景为 48.89 cm，但场景组真值静止脚滑移低 16%。DDIM-20、每域 4 个 128 帧样本中，场景组 MPJPE 为 47.94 cm、无场景为 48.31 cm，头部误差低 0.86 cm，真值静止脚滑移低约 4.9%，支撑悬浮低约 4.2 mm。
- 上述 500-step 模型仍明显欠训练：生成关节速度约为真值的 20%，而且在同一个场景模型中开关场景只改变约 0.01 cm MPJPE，说明当前结果主要验证训练链路，尚不能证明模型已学会强场景依赖。正式结论必须来自更长训练和可视化。
- 旧的独立 9 维相机条件 200-step pilot 与当前 mask 架构不兼容，只保留数值报告作历史诊断，不可续训。

## 2026-10-02 脚步修复

新训练请显式使用 `--skeleton-profile trumans_male_v2`。审计发现 TRUMANS 关节归档实际对应 male SMPL-X，而此前导出因 gender 为空用了 neutral，造成约 5 cm 脚端监督冲突。旧 checkpoint 和历史对照仍按 `archived` 读取，原始缓存没有覆盖。

脚部新增监督、配对试验、模板审计和未见窗口结果见 [FOOT_REPAIR_zh.md](FOOT_REPAIR_zh.md)。`gait_v1` 已改善迈步幅度但未解决支撑滑动；`gait_v2` 的相位无关支撑、低噪声训练和 float32 组合正在验证，不应仅凭训练损失宣布自然度达标。相应选项为 `--loss-profile gait_v2 --precision float32 --low-noise-fraction 0.25`。

`--init-from` 仅加载权重、重置优化器和学习率日程，适合独立的修复试验。`--resume` 才恢复完整状态，并要求模板、损失及精度协议一致。新脚步试验没有修改原 30k 权重、mask 比例或 ego 可见场景协议。

## 已知边界

静态场景和头部轨迹无法唯一决定真实动作。模型目标是生成合理、轨迹一致且利用场景的动作，而不是逐帧复原唯一 GT。另一名人物和显著移动的动态物体目前不是条件输入，因此不能期待恢复具体交互动作。支撑面指标目前用 held-out 真值脚高度作评估代理，不适合楼梯等多高度表面；正式物理评估仍需可靠场景 SDF/局部支撑面。
