# AMASS EXO 去噪修复训练

## 当前续训：仅 EXO，主干冻结

2026-10-27 用户确认从已选边界 6k 继续额外 20k。新配置为
[`amass_exo_continue.yaml`](../configs/amass_exo_continue.yaml)，新入口为
[`exo_continuation.py`](../scenemi/exo_continuation.py)，输出
[`runs/amass_exo_continue128`](../runs/amass_exo_continue128)。
该模型已累计 EXO 36k＋边界 6k；共享 EGO-only 55k 权重保持冻结，
只训练边界编码器与 EXO 内部注入层。误开的 warm 联合 1,350 步从不作为初始化。

每 5k 同步完整验证全部 337 个 family / 每条件 2,222 个窗口，使用原 EXO
三条件区域分数与输入保留门槛选 best，并每 500 步原子覆盖可恢复 last。
复用父模型 step 0 完整结果前，CPU/CUDA 均已验证逐位精确加载父权重。
学习率 3e-5、batch 8，优化器重置；采样、损失、遮挡与缺帧分布不变。
EGO 哨兵更新前后逐元素相等；这项检查覆盖 4 个固定窗口，不能称为新做的完整 EGO 评测。

当前启动状态见 `STARTUP_STATUS.json`，开训后见 `training/STATUS.json`。
断开 IDE 与机器/WSL 重启不同；重启后必须核对 last、资源准入和用户服务，再显式恢复。
不恢复已停用的外部周期检查器，不自动启动三数据集。
后续是否解冻主干或训练场景分支需要依据该阶段目标另定；本轮冻结不代表永久冻结。

## 原 36k 及边界短对照的历史协议

2026-10-27，用户授权从 EGO-only 55k 开始夜间训练，希望北京时间当日 09:00 前获得是否进入三数据集阶段的判断。该训练与旧 EXO 15k、两个 2k 时间修复短试分开，输出 [`runs/amass_exo_denoise128`](../runs/amass_exo_denoise128)。当前状态以 [`training/STATUS.json`](../runs/amass_exo_denoise128/training/STATUS.json) 为准；不得在训练前声称收敛或取得收益。

## 起点、输入与学习路径

- 固定原 EGO-only 55k，SHA256 `50a061e0d40fd0acd866b3806620cfe98255394f1e4c32c7828a542aae4a3305`。原去噪器和 EGO 路径冻结；新优化器不加载旧 EXO 15k 或时间修复 2k 权重。
- 仍使用 AMASS train，128 帧、30 Hz，场景关闭。按历史协议均匀抽 family，再随机原始/镜像、完整窗口起点和 yaw，有放回；不把它称作固定 64 步长实验。train/val/test 划分不变，test 数组不读。
- 输入为明确声明的**合成完整原生姿态、体形估计**，含时间相关姿态/根误差、肢体遮挡、8–32 帧整人缺失。不是实测前置模型输出，输入比旧“部分关节生成”丰富；实际 ZIP 参数重建和可靠度校准待后续核验。
- 缺失估计先设为 NaN，再仅依据已有估计邻帧做插值；原始缺帧观测清零。标签、个人真值 rest/betas 不进入推理 payload。理想准确估计仅用于保持能力检查，不能表述为实际推理已知真值体形。
- 固定未训练时间候选混合作为锚点，包括保留、3/5/9/17 帧时间核与可用邻帧延续。固定 logits `[0,0,0,1,1,-1]`，不加载其 2k 训练权重；因而并非纯零修正起点，必须比较相同锚点与已训练时间修复对照。

实现 [`exo_denoise.py`](../scenemi/exo_denoise.py)：把估计动作、当前扩散状态、可靠度与时间信号编码后，用零初始化投影注入 U-Net 三个编码层、两个中间层和两个解码层。参考前向不注入；适配前向注入，两者使用同一冻结去噪器、状态和公共观测。两次输出的差经固定变换形成根平移和局部 SO(3) 修正，再重做 FK。**所有可学习修正必须经过 U-Net，没有直接时间输出头绕过它。** 这仍需实验才能证明预训练动作知识有额外价值，不能用结构上的依赖代替预训练/随机先验的训练消融。

根平移修正上限 0.5 m、局部旋转上限 1.2 rad，再乘不可靠程度；锚点自身沿用 0.25 m/0.6 rad 的固定限制。理想完全可靠帧直接返回原估计。根可靠度与局部旋转可靠度分别处理。实际可靠度如果误判，保留规则不会自动识别该错误。

训练在既有扩散时间表 0–200 步，监督干净 x0；可靠部分将状态拉回同噪声下的估计锚点。推理从加噪锚点开始，8 步 DDIM，每步依据可靠度约束回估计；每步都重新查询去噪器。损失包括全身与根位置、旋转、速度、可靠部分保留，以及按真实合成遮挡/缺帧 mask 归一的区域损失。离线处理允许使用未来邻帧，不称作因果实时模型。

## 预算与选模

| 项目 | 固定设置 |
| --- | --- |
| 训练预算 | 最多 36,000 更新，batch 8；AdamW 1e-4，300 步 warmup，按 36k 余弦衰减至 0.1 倍 |
| 时间预算 | 北京时间 07:30 后停止新增更新，对当前 last 做完整验证并汇总，为 09:00 检查预留时间；实际更新数与停止原因明确记录，不能称为固定 36k 已完成 |
| 可训练参数 | 1,085,824，仅条件编码与内部投影 |
| last | 每 500 更新原子覆盖，含优化器、数据/CPU/CUDA RNG、已提交日志长度 |
| 验证 | step 0 及每 3k，同步完整验证：337 个源动作 family、每条件 2,222 窗口，四条件共 8,888 次修复采样 |
| 窗口与随机数 | 验证 stride 128 加完整尾窗，固定未镜像/yaw0；输入 RNG=777+case_id，采样 seed777，batch8 |
| best 分数 | 含噪跟踪、遮挡区域、缺帧区域 MPJPE 各先 family 等权，再三条件等权平均；同分保留较早候选 |
| 保留门槛 | 优先选择理想准确输入保持、其他条件全窗口/可见区域/根误差相对初始化不退化超过 1 mm 的候选；若无合格候选，明确报告未通过 |
| 进入后续阶段门槛 | 在上述约束下，遮挡与缺帧区域分别优于初始化、简单时间平滑和上一轮**已训练**时间候选对照；用户追加授权后，仅通过该门槛且下一阶段接口验收有效时，允许自动接续三数据集无场景适配 |

全部候选同步评测，训练等验证完成；不存在异步选择漏掉候选。仅保留一个可恢复 best 和一个 last，不保存每个候选的权重副本。选择使用验证集，不能据此保证所有未保存更新都更差。

五个对照列共享输入、人体解码、窗口和坐标：初始化、5 帧简单时间平滑、固定未训练时间锚点、已训练时间修复、新去噪修复。已训练和未训练时间对照复用已有逐窗口记录，逐例验证 identity 完全相同，并封存记录 SHA；不是重新采样另一套输入。各候选保存全身 MPJPE、根误差、根相对 MPJPE、PA-MPJPE、可见/受损区域与时序指标；最终报告按 family 配对 bootstrap 区间。

## 验收、资源与恢复

实际 CPU/CUDA batch8 验收包括：精确恢复的下一更新、冻结权重哈希、原 EGO 固定窗口逐元素保持、私有标签隔离、理想可靠帧保持、缺失观测清零、FK/SO(3) 一致、完整 DDIM 有限，以及更新后输出对当前扩散状态确有依赖。EGO 哨兵只有 4 个窗口，不当作重新完成全 EGO 验证。

准备验收发现日志时间戳影响逐行恢复比较，已将时间戳移到正式训练日志，修复后重做验收；原证据留在 `preparation_revision_001`。随后在正式开训前补充根轨迹不退化门槛，验收与封存重新完成，旧准备证据单独保留。用户追加授权增加夜间训练后，预算由 18k 改为最多 36k、预声明 07:30 时间停止，并重新做最终 CPU/CUDA 验收和真实更新/采样计时。训练开始后不再改依赖封存。

沿用用户明确授权的 AMASS EXO 按余量共享：5.5 GiB PyTorch 分配上限，额外 1 GiB CUDA、2 GiB 空闲余量；10 GiB RAM、2 CPU 核。资源准入不能通过则不启动，不干预其他进程。运行通过用户级 systemd 服务启动，IDE 断开后继续；机器/WSL 故障仍需从最近原子 last 恢复。

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 10 --cpu-count 2 -- \
  python -m scenemi.exo_denoise_training preflight --device cpu
python -m scenemi.denoise_runtime run \
  --device cuda --memory-gib 10 --cpu-count 2 --allocator-gib 5.5 -- \
  python -m scenemi.exo_denoise_training preflight --device cuda
python -m scenemi.denoise_runtime run \
  --device cuda --memory-gib 10 --cpu-count 2 --allocator-gib 5.5 -- \
  python -m scenemi.exo_denoise_training train --device cuda
# 精确恢复：同一 train 命令增加 --resume；仍须资源准入。
```

固定配置 [`amass_exo_denoise.yaml`](../configs/amass_exo_denoise.yaml)，训练与报告入口 [`exo_denoise_training.py`](../scenemi/exo_denoise_training.py)，资源入口 [`denoise_runtime.py`](../scenemi/denoise_runtime.py)。最终汇总写入 `RESULT.json` 和 `SUMMARY_zh.md`，选模进度 `selection/STATUS.json`、`selection/BEST.json`；全部输入身份、源代码、参考指标和基座 SHA 见 `PROTOCOL.json`。

## 夜间检查与条件接续

用户随后授权延长训练、定时检查及在条件合适时进入三数据集。配置 [`exo_overnight.json`](../configs/exo_overnight.json)，执行入口 [`monitoring.py`](../scenemi/monitoring.py)；用户级 `scenemi-exo-overnight-check.timer` 每 30 分钟执行一次，记录 AMASS 的 `OVERNIGHT_STATUS.json` 和 `OVERNIGHT_CHECKS.jsonl`。检查包括服务状态、逐批验证心跳、训练源文件哈希。07:00 前最多两次按原封存参数精确恢复；非有限损失或完整性失败不自动重试。定时脚本不会自行设计新模型、改超参数，也不保证训练一定取得正收益。用户服务可在 IDE 断开后继续；机器或 WSL 重启会丢失这类临时服务，仍须重新核查和恢复。

AMASS 完成后，自动流程还会核对最终报告、BEST 和所选权重哈希是否一致、质量门槛是否通过。只有三数据集诊断 CPU/CUDA 验收通过且准备源文件未变，才允许启动 `scenemi-scene-exo-repair.service`。失败则明确写出原因，不通过更改门槛强行进入后续训练。

下一阶段配置 [`scene_exo_repair.yaml`](../configs/scene_exo_repair.yaml)，入口 [`scene_repair_training.py`](../scenemi/scene_repair_training.py)。只读当前选择合同的 EgoBody / TRUMANS / GRAIL train/validation，排除 RICH/test；TRUMANS 使用已完成的移动相机数据。此阶段是 **EXO-only、无场景的原生输入分布适配基线**，尚未启动场景分支。原生 SMPL / SMPL-X 10 维形状各自解码，GRAIL 使用声明的公共固定 G1 模板，不将它们补维后解释为 SMPL-H 16 维形状。个人估计仍是人工噪声模拟，模板类别暂以来源模板模拟上游估计，真实前置模型的类别与可靠度仍需核验。

诊断检查可使用独立固定的 AMASS 中间权重，仅用于提前核验接口，记录 `diagnostic_only`，不能作为正式初始化或选择证据。正式接续必须等待 AMASS 完成并通过门槛，将完整验证选中的 best 复制到独立的 `initialization.pt`，随后对这份最终初始化重新做 CPU/CUDA 验收，再训练。全部 GPU 子任务仍须按 5.5 GiB 分配上限及额外余量准入；任何阶段拒绝或失败都会停在该阶段，不干预其他任务。

原生适配额外要求每 128 帧至少 34 帧有真实来源的可用估计，以保证随机删除最长 32 帧后仍能以两个可用估计建立初始化。该筛选不等同于动作质量筛选；源序列和窗口起点的前后数量写入 `PROTOCOL.json/eligibility`。其诊断列表覆盖全部剩余验证序列，每条件 **3,540 窗口**，分割连续允许起点区间、stride128 加完整尾窗。按三来源等权、来源内 family 等权；保存各来源指标及相同聚合的配对区间。训练最多 24k、每 3k 同步完整验证并选 best/last，检查点门槛同时约束总体、可见区域和根轨迹不退化；与初始化、简单平滑、固定时间锚点及 AMASS 迁移起点比较。

准备状态见 [`READINESS.json`](../runs/exo_preparation/three_corpus_repair/READINESS.json)；正式接续后的分阶段进度见 `runs/scene_exo128_noscene_repair/PIPELINE_STATUS.json`，训练进度见其 `training/STATUS.json`。当前序列范围合同和适配额外筛选分别记录，不能将源数据表的全部帧数误称为修复模型实际用到的全部窗口。

## 36k 后的边界条件短对照

2026-10-27 用户要求停止自动检查，`scenemi-exo-overnight-check.timer/service` 已停止，证据在 AMASS 原 run 的 `AUTOMATION_STOPPED.json`；后续短对照不重启周期检查，也不自动进入三数据集。原 36k 在 06:16 完成，best/last 相同，SHA256 `fe08afbd8abcf732cb939b53e3bd0d6329386be1a5b67d17eebae26f0637bdd9`。

用户随后授权在当天 11:00 前尝试缺口改进。配置 [`amass_exo_completion.yaml`](../configs/amass_exo_completion.yaml)，公共实现 [`exo_completion.py`](../scenemi/exo_completion.py)，输出 `runs/amass_exo_completion128/{control,boundary}`。两组都从已完成的 36k 修复权重开始，底层 EGO-only 55k 冻结，重置新适配阶段优化器；各额外 6k、batch8、lr3e-5、100 步 warmup。数据、噪声/缺失分布、损失、DDIM 和完整验证列表均相同，重置模型构造消耗的 RNG 后再更新，以使两组数据、扩散噪声和 timestep 一致。原权重及源码封存不变。

control 保持原编码器；boundary 增加显式边界条件：到最近可用估计的左右时间距离、是否存在该侧观测、根置信度、相对根位移、从真实可用估计计算的边界速度，以及局部相对旋转。插值帧不充当边界锚点。新编码器末投影为零，初始采样与原 36k 完全相同；所有学习修正仍经冻结 U-Net 内部产生。此轮只检验显式边界条件，不同时改变窗口、目标函数或实际出画分布，收益须由对照验证。

step0 的全验证记录复用父 36k，前提是 CPU/CUDA 真实 batch 的完整 8 步采样初值等价检查通过；明确标记复用，不冒充重新采样。额外 3k 与最终更新均完整评测 337 family / 每条件 2222 窗口，选 best/last；最终报告还在相同最终更新预算下直接比较 boundary 与 control，并保存配对 family 区间。10:20 后不新增训练更新，预留完整验证及汇总时间。若共享资源变化或验收失败，按实际状态报告，不保证正收益或准点。

截至该短试准备时，原模型的综合验证分数 30k / 33k / 36k 为 109.187 / 109.756 / 108.953 mm，后 6k 最优改善约 0.21%；缺帧区域 30k 的 170.124 mm 仍略优于 36k 的 170.545 mm。训练平均损失仍下降，因此更准确的结论是收益放缓、尚未证实充分收敛，不能将“预算结束”称作“已完全收敛”。

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 10 --cpu-count 2 -- \
  python -m scenemi.exo_completion preflight --variant boundary --device cpu
python -m scenemi.exo_completion run \
  --device cuda --memory-gib 10 --cpu-count 2 --allocator-gib 5.5 -- \
  python -m scenemi.exo_completion preflight --variant boundary --device cuda
python -m scenemi.exo_completion run \
  --device cuda --memory-gib 10 --cpu-count 2 --allocator-gib 5.5 -- \
  python -m scenemi.exo_completion train --variant boundary --device cuda
# control 分支将 --variant 换成 control；两组均通过匹配验收再开训。
```

训练结束时的一次性汇总写入根目录 `STATUS.json`、`COMPARISON.json` 和 `SUMMARY_zh.md`；每组 `training/STATUS.json` 与 `selection/step_*/STATUS.json` 分别记录更新和逐批验证心跳。真实相机视野出画、重入画跳变和更长缺失的评测不由本轮替代。
