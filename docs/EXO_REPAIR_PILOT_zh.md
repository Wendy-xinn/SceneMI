# EXO 初始化修复短试

2026-10-26 晚，用户授权修改后训练验证，希望北京时间 10 月 27 日 00:30 前得到可行性判断。配置 [`amass_exo_repair.yaml`](../configs/amass_exo_repair.yaml)，输出 `runs/amass_exo_repair128`。这是独立试验，保留原 AMASS EXO 15k 及 EGO-only 55k。

**最终进展（北京时间 2026-10-27 00:06）：两个独立 2k 短试及完整验证均已结束。自由残差候选失败；时间候选混合版通过预设筛查，适合作为初始化修复基线。尚未验证扩散先验具有实质收益。** 完整数字和限制见本文末尾“最终结果”。

## 模型与输入

- 冻结原 EGO-only 55k；使用其去噪器提供冻结动作特征，原 EGO 模式直接走原路径。
- EXO 显式接收完整估计初始动作、估计 rest、局部姿态/全局轨迹可靠度、跟踪是否存在和合成可见性。61 帧感受野的时间卷积分支预测有界修正：根平移最多 0.25 m，局部旋转最多 0.6 rad，再按不可靠程度缩放。旋转以 SO(3) 修正并重新 FK，不逐关节拼接。
- 最后一层零初始化；修正起点为零。整个帧可靠时直接返回原初始化，即使分支已训练也保持该帧。可靠姿态与不可靠根轨迹分别控制，根位置变化仍会共同移动人体。
- 在既有扩散时间表的 0–200 步范围训练 x0；采样从估计初始化前向加噪后开始，8 步确定性 DDIM 返回修复动作。并未训练第二个独立扩散先验。
- 新增可训练参数 **671,173**。冻结特征不参与反向传播。

本短试的输入比原“只有部分关节位置”的 EXO 15k **更丰富**，不能把其数值变化当作只改一个模块的公平消融。还需要初始化相同的插值、时间平滑对照，以及之后的先验特征消融。

## AMASS 合成前置估计

当前模拟原生 SMPL-H 姿态和体形估计；非 identity 条件的 betas 加 σ=0.15 噪声，姿态加时间相关 0.015–0.04 rad 扰动，根轨迹加 0.01–0.04 m 时间相关误差。遮挡区域的估计姿态另加 0.12–0.25 rad 扰动。并非实测视频估计误差。

训练条件概率：理想准确估计 20%、一般跟踪 35%、局部遮挡估计 25%、整人跟踪缺失 20%。理想条件专用于检验准确输入不被改坏；不能表述成实际推理已知真值体形。其他条件使用明确加噪后的估计体形，不使用个人真值 rest 解码输出。

局部遮挡时，前置完整估计仍存在，只降低该部分可靠度；整人缺失为 8–32 帧，原始观测清零，初始动作只从邻近**已有估计**做平移插值与旋转 Slerp。首尾缺失保持最近估计。进入插值前，缺失估计先设为 NaN，以核验没有用隐藏干净动作填空。时间轴保持 128 帧、30 Hz；使用未来邻帧时属于离线窗口处理。

可靠度来自声明的合成噪声强度、缺失和遮挡标记，**不从已实现的 GT 误差计算**。实际部署的可靠度代理还需接入与校准。完整参数初始化也尚未从真实 ZIP 网格重建；本轮不代表 ZIP 已能直接运行该新分支，更不代表真实视频收益。已有仅关节条件路径仍可消费网格回归关节。

实现：[`exo_initialization.py`](../scenemi/data/exo_initialization.py)、[`exo_repair.py`](../scenemi/exo_repair.py)、[`exo_repair_training.py`](../scenemi/exo_repair_training.py)。

## 固定训练和验收

| 项目 | 设置 |
| --- | --- |
| 数据 | AMASS train；EgoAllo 划分；不读 test 数组；场景关闭 |
| 取样 | 历史等概率 family，再镜像/随机完整 128 帧/yaw；有放回 |
| 预算 | 固定 2,000 更新、batch8、AdamW 3e-4、100 更新 warmup，之后余弦降至 0.2 倍 |
| checkpoint | 每 250 更新覆盖原子 `training/last.pt`；固定最终 2k 评测，不根据中途结果挑 best |
| 子集评测 | 0/1k/2k；四个验证语料各 16 条动作，每条一个中间窗口，共 64 窗口；四种条件各评一次 |
| 最终评测 | 全部 337 条验证动作、2,222 窗口；四种条件共 8,888 个修复采样，另评相同输入的初始化和时间平滑 |
| 资源 | 4 GiB allocator、额外 1 GiB CUDA、2 GiB 空闲余量；8 GiB RAM、2 CPU 核；沿用用户本轮按余量共享授权，不修改其他进程 |

根位置与姿态、干净关节位置/速度监督，加上可靠部分的保留项。输入/标签严格分离；缺失的前置估计不会通过原始观测字段进入模型。

固定两个对照：`initial`（完整估计；缺失段已按上述规则插值）和 `temporal_smoothing`（5 帧 `[1,2,3,2,1]/9` 滤波，按相同不可靠程度混合，并重新 FK）。三者共享初始化、估计骨架、mask、窗口和坐标。

预先登记的候选门槛：identity 最大位移不超过 0.001 mm；其他条件整体及可见部分相对初始化不退化超过 1 mm；局部遮挡、整帧缺失区域分别优于**两个**对照。按 family 等权平均。这个门槛是本轮候选筛查，不自动构成统计显著性或论文最终结论；若失败，按原门槛报告，不事后改评测区域。

CPU/CUDA 真 batch8 已通过：精确恢复、冻结先验及 EGO 固定窗口保持、私有标签隔离、可靠帧原样保持、缺失原始估计清零、FK/SO(3) 一致和 DDIM 有限。CUDA 峰值约 0.57 GiB。插值单元检查另外核验隐藏 payload 不影响结果、已有估计逐元素保留、边界处理和空跟踪拒绝。

## 运行与状态

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 8 --cpu-count 2 -- \
  python -m scenemi.exo_repair_training preflight --device cpu
python -m scenemi.initialized_runtime run \
  --device cuda --memory-gib 8 --cpu-count 2 --allocator-gib 4 -- \
  python -m scenemi.exo_repair_training preflight --device cuda
python -m scenemi.initialized_runtime run \
  --device cuda --memory-gib 8 --cpu-count 2 --allocator-gib 4 -- \
  python -m scenemi.exo_repair_training train --device cuda
```

精确恢复使用同一训练命令加 `--resume`，仍先通过资源准入。运行开始后，封存源文件、配置及初始化不能修改。

进度 `STATUS.json`、`training/STATUS.json`；中途评测 `validation/step_*/RESULT.json`；最终结果 `full_validation/RESULT.json`；CPU/CUDA 证据 `preflight/{cpu,cuda}.json`；初始化、源码和全部验证身份 `PROTOCOL.json`。旧训练的最终结果见 [EXO 15k 报告](AMASS_EXO_RESULTS_zh.md)。

## 第二个同预算候选：明确的时间修正方向

第一个分支在完整验证中未优于对照，尤其局部遮挡没有收益。随后登记同样 2k、相同数据分布和验证身份的候选 [`amass_exo_basis.yaml`](../configs/amass_exo_basis.yaml)，输出 `runs/amass_exo_basis128`。

它仍围绕完整估计动作修复，但将自由预测的方向改为可解释的候选：保留原估计、3/5/9/17 帧三角时间核，以及根据已跟踪邻帧速度构造的边界延续或区间三次插值。姿态方向在 SO(3) 切空间计算，根位移单独处理，沿用相同可靠度门控与幅度上限；候选不咨询 GT。冻结扩散特征参与预测候选混合权重。

新候选并非零修正初始化；固定初始 logits 为 `[0,0,0,1,1,-1]`，其未经训练的混合必须另作对照，不能把先验规则本身的收益归因于学习。完整未训练基线按数学等价关系在 CPU 计算：初始 readout 权重全零，输出只取决于固定 bias 和候选，因而此时扩散特征与采样噪声不影响输出。该计算逐例对照实际 step0 DDIM 结果，允差 0.01 mm。

实现 [`exo_basis.py`](../scenemi/exo_basis.py)，通过 [`exo_basis_training.py`](../scenemi/exo_basis_training.py) 在封存训练器创建合同前显式替换模型工厂。新后端、组合入口、实际 launcher 均进入依赖封存；原训练器保持不变。CPU/CUDA 同样通过精确恢复、可靠帧保持和 FK 检查。第二版的首次 CPU 准备证据保存在 `preparation_revision_001`；正式训练前补齐 launcher 封存与默认配置后，重新完成 CPU/CUDA 验收。

```bash
python -m scenemi.basis_runtime run \
  --device cuda --memory-gib 8 --cpu-count 2 --allocator-gib 4 -- \
  python -m scenemi.exo_basis_training train --config configs/amass_exo_basis.yaml --device cuda
```

[`exo_repair_report.py`](../scenemi/exo_repair_report.py) 汇总完整验证及 family 配对 bootstrap 区间；[`exo_basis_report.py`](../scenemi/exo_basis_report.py) 额外比较训练后与未训练候选。不能只凭训练后胜过单一平滑就证明训练有效。

另有固定权重的 [`exo_feature_probe.py`](../scenemi/exo_feature_probe.py)：在 CPU 上，使用相同输入、噪声、权重与 batch，比较正常冻结扩散特征和将该特征置零的采样结果。这是推理压力检查，不能代替从训练开始移除先验的正式消融。

源身份中继承的 `synthetic_exo_pattern`、`visible_fraction` 等字段仅记录旧采样函数内部的源定位过程。两轮真正输入都由新的完整估计、可靠度和 `estimate_gap_or_occlusion` 重建；评测使用实际 `joint_visible` / `estimate_available` 张量，不用继承字段划分区域。

## 最终结果

两个候选分别从同一个冻结 EGO-only 55k 先验出发，各训练固定 2k；第二个没有使用第一个的权重。第一版于 10 月 26 日 23:47 完成，第二版于 10 月 27 日 00:06 完成。每条件使用全部 **337 个源动作 family、2,222 个验证窗口**，四条件共 8,888 个模型采样；按 family 等权汇总，未使用独立 test。两版使用相同合成初始化分布、验证身份和候选门槛；它们仍属于验证集上的方法开发，不能替代独立测试。

### 完整窗口指标

以下均为世界坐标下的全窗口、全身 MPJPE，单位 mm，越低越好。

| 条件 | 初始化 | 5 帧时间平滑 | 第一版自由残差 | 第二版候选混合 |
| --- | ---: | ---: | ---: | ---: |
| 理想准确估计 | 0.000 | 0.000 | 0.000 | 0.000 |
| 含噪跟踪 | 61.181 | 60.275 | 61.309 | 60.385 |
| 局部遮挡 | 62.817 | 61.673 | 62.963 | 61.480 |
| 整人缺帧 | 83.729 | 82.986 | 83.938 | 81.642 |

第一版未通过预设门槛，按仓库规则保留实验记录、移除无复用计划的权重；文件 SHA 和删除前证据见 [`CHECKPOINT_RETIREMENT.json`](../runs/amass_exo_repair128/CHECKPOINT_RETIREMENT.json)。第二版保留可恢复的 [`training/last.pt`](../runs/amass_exo_basis128/training/last.pt)，没有依据中途验证结果挑选 checkpoint。

### 遮挡与缺帧区域，以及学习的增益

这里的局部遮挡只评实际遮挡关节/帧，整人缺帧只评实际缺失帧；不能与上表全窗口指标混用。缺帧初始化已经使用估计邻帧插值，不是把空白位置当零。

| 评测区域 | 初始化 | 5 帧时间平滑 | 第二版未训练 | 第二版训练 2k |
| --- | ---: | ---: | ---: | ---: |
| 局部遮挡 | 121.324 | 110.933 | 105.254 | **99.112** |
| 整人缺帧 | 183.234 | 183.292 | 180.745 | **174.937** |

固定的时间候选规则本身已经提供收益；训练后相对同结构未训练版仍有额外收益。按源动作 family 配对 bootstrap，5,000 次重采样、固定种子 20261026，差值为“训练后减对照”，95% percentile 区间如下：

| 区域 | 相对时间平滑差值 / mm | 95% 区间 | 相对同结构未训练差值 / mm | 95% 区间 |
| --- | ---: | --- | ---: | --- |
| 局部遮挡 | -11.821 | [-12.498, -11.153] | -6.142 | [-6.608, -5.661] |
| 整人缺帧 | -8.355 | [-13.802, -2.786] | -5.808 | [-11.049, -0.906] |

遮挡区域相对平滑有 324/337 个 family 改善；缺帧区域只有 166/337 改善，平均收益受部分较大改善影响，**不能声称所有短缺帧都更好**。这些区间只反映当前验证源动作的重采样不确定性，没有覆盖训练种子、合成噪声模型或真实前置估计的变化。

普通含噪跟踪时，第二版 60.385 mm 比时间平滑 60.275 mm 略差，也比同结构未训练的 59.838 mm 差。因此它是面向遮挡/缺失的修复候选，还不能认为在所有观测条件下都优于对照。缺帧条件的根误差为初始化 71.792、平滑 71.362、修复 69.364 mm，改善有限，尚不能据此宣称已解决真实深度或长期轨迹漂移。

### 保留能力与扩散作用

- 理想准确输入在全部验证窗口中保持原样；这由显式可靠度保留通路保证，不是学到“识别准确输入”。真实可靠度错误时仍可能修改不该改的部分。
- 冻结先验源权重未变，EGO 模式绕过新 EXO 分支；CPU/CUDA 验收及训练后 4 个固定 EGO 窗口均与原模型逐元素一致。本轮没有重新跑完整 EGO 验证，不能把 4 窗口检查写成全验证。
- 64 个 family 的配对推理检查中，将冻结扩散特征置零只引起约 **0.004–0.005 mm** 的平均输出变化，遮挡区域误差差约 0.005 mm、缺帧约 -0.002 mm。当前没有证据证明扩散特征发挥实质作用；该检查也不等价于重新训练一个无先验消融。
- 当前可支持的结论是：**围绕前置估计、按可靠度保留观测、学习选择时间修正，可以改善这套合成遮挡/缺帧验证。** 不能将收益表述为已经验证成功的“扩散先验修复”，也不是场景信息收益。

下一步应以这个候选为强时间修复对照，让去噪器实际预测受约束的姿态与根轨迹残差，再验证其是否超过同初始化的时间对照。真实视频仍需核验参数初始化、单位/重力/时间对齐和可靠度；本轮没有把 ZIP 中的网格直接当作原生姿态参数，也没有实际视频精度结果。先验、场景或统一模型的论文结论均需各自独立对照。

可查阅的完整记录：

- [`第二版汇总`](../runs/amass_exo_basis128/SUMMARY_zh.md)、[`完整逐窗口指标`](../runs/amass_exo_basis128/full_validation/metrics.jsonl)。
- [`学习增益与区间`](../runs/amass_exo_basis128/LEARNING_EFFECT.json)、[`未训练完整对照`](../runs/amass_exo_basis128/untrained_full_validation/RESULT.json)。
- [`冻结特征置零检查`](../runs/amass_exo_basis128/feature_probe/RESULT.json)、[`CUDA 验收`](../runs/amass_exo_basis128/preflight/cuda.json)。
- [`协议和依赖封存`](../runs/amass_exo_basis128/PROTOCOL.json)、[`配置`](../configs/amass_exo_basis.yaml)、[`训练完成状态`](../runs/amass_exo_basis128/training/STATUS.json)。
