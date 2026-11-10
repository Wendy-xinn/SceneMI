# 旧场景1000步：三数据集版本诊断

本实验是四数据集无场景replay第2000步与其后额外适配的旧场景第1000步之间的
版本比较。验证范围只有EgoBody、TRUMANS、GRAIL，不含AMASS；训练量、EXO结构
及内参处理未构成单变量控制，因此不能据此证明场景模块的净收益。
四数据集无场景训练自身的step0/64/512/2000验证另见
[四数据集无场景验证结果](FOUR_CORPUS_VALIDATION_RESULTS_zh.md)。

用户在 2026-10-30 明确要求完整数据集对照，不使用小样本结果替代。
配置固定为 `configs/scene_joint128_comparison.yaml`，评估入口为
`scenemi.replay.scene_comparison`，完整执行与论文表格汇总入口为
`scenemi.replay.comparison_reporting`。不恢复训练、不修改原模型实现或权重。

## 权重与完整范围

场景 adapter 为第 1000 步，SHA256
`deccf721d4170ca8970ebe1cfaaacf73fbd66f5a671905023cb647c077530f18`。
无场景 replay 为第 2000 步，SHA256
`a46122d40c572be8ac82b55ca7d421e7c64a4de690c7ae369091064ba1b9260c`。

使用原 `runs/scene_joint128_replay/VALIDATION_CASES.json` 的全部三原生数据集窗口，
清单 SHA256 `117b8d42308ed59019b09502d0926be553131f955cd2f5f9ac0169720d9d616c`。
每个接受起点区间按 128 帧步长覆盖，保留末尾对齐窗口；独立测试数组不开启。

| 数据集 | EGO 窗口 | EXO 窗口/条件 | EXO 条件数 |
|---|---:|---:|---:|
| EgoBody | 401 | 470 | 4 |
| TRUMANS | 540 | 540 | 4 |
| GRAIL | 2530 | 2530 | 4 |
| 合计 | 3471 | 3540 | 4 |

EXO 条件为 identity / tracking / occlusion / gap。完整验证共有 17,631 组，
另有原来的九例定向诊断，共 17,640 组。诊断例不进入整体统计。
这里“完整”指此次场景阶段训练的三个原生数据集的全部既定验证窗口，
不包含训练数据，也不将父模型 AMASS 的保留验证冒充为已完成。

## 内参差异的处理

2026-10-30 进一步检查导出器和实际数组后更正：`exo_intrinsics` 已经是
`[fx/W, fy/H, cx/W, cy/H]`，不是此前误称的像素内参。`NativeRepairDataset.sample`
直接返回此值，无场景 replay 使用该导出归一化值；`SceneExoDataset.sample`
又除以 256，属于重复缩小。EGO 的另一个源数组 `camera_intrinsics_128x96`
保存像素值，两个 EGO 入口对它除以 256 是原模型既有的特征缩放，不能据此类推 EXO。
场景模型训练与本次推理均使用重复缩小后的 EXO 输入，存在与父模型的输入分布差异。
该操作发生在网络条件特征中，不会直接缩放已经导出的世界坐标场景点或人体。

全量保留三个 EXO 输出：

1. `no_scene`：原无场景 checkpoint，对导出归一化内参额外 `/256`，属于输入干预。
2. `scene_1000`：场景 checkpoint，沿用该场景训练中的额外 `/256` 输入。
3. `historical_no_scene`：同一无场景 checkpoint，使用正确的原导出归一化值，与父模型训练一致。

前两路除场景几何外，公开观测、初始化、原生格式和目标数组逐项严格相等，
每组保存输入哈希；第三路单独保留历史分布。额外缩小后的无场景输出是固定权重的
预处理干预，不伪称该 checkpoint 按此新口径训练过。原始无场景训练和历史结果不覆写。

三个 EXO 核验窗口切换内参后，MPJPE 变化均小于 0.01 mm；这是执行核验，
不是全量结论。全量将直接汇总此影响，不能仅依据前三例忽略预处理问题。

独立 CPU 固定权重诊断 `runs/scene_joint128_comparison/intrinsics_diagnosis` 已完成：
三个数据集各一个预先固定窗口，分别覆盖四种 EXO 条件，共 12 个条件窗口、24 个
模型/条件结果。切换额外 `/256` 与原导出值，场景模型最大单窗 MPJPE 变化为
0.02272 mm，最大逐关节位置变化为 3.56815 mm（GRAIL）；不能用均值很小代替峰值判断，
也不能将此诊断宣称为完整验证。当前证据不要求丢弃 1000 步 adapter 或重训无场景父模型，
但不应原样继续错误输入；恢复应在独立运行中修正 EXO 特征单位并先验收。

完整 GPU 评估的原代码/配置/协议已封存，已完成这个已有版本的计算，没有热改输入。
其中历史文本和 JSON 字段的 `pixel` 字样存在旧误标，解读必须结合输出目录中的
`INTRINSICS_UNIT_CORRECTION.json`；三个实际输入值及全部数值结果均保留。

## 采样、指标与统计

同设备、batch=1、seed=777；EGO 使用 20 步 DDIM，EXO 使用 8 步、起始 timestep=200。
输入扰动保持原 `777 + case_id` 种子。两模型在此设置下重新成对采样，
不把原 CUDA 大 batch 验证数值与本次 CPU 结果混接。
场景历史只截至窗口起点，当前点云只使用逐帧实测 first-hit，不用未来场景或 GT 人体作模型条件。

复用原标准全局 MPJPE、PA-MPJPE、root、MPJVE 与 jerk 指标；另存首末/最坏帧根误差、
高度偏差、EGO 相机残差、缺失区域误差，以及排除缺帧边界的跟踪段加速度和 jerk。
不对全局指标做根对齐、贴地或逐帧修正。PA-MPJPE 明确保留其自身对齐约定。

先按独立源家族汇总，再对三个数据集等权；不同 EXO 条件分别报告。
配对置信区间使用独立家族 bootstrap（固定 2,000 次），不把相邻重叠窗口视作独立样本。
区间不涵盖重新训练与扩散多 seed 不确定性。

## 执行与证据

CPU 资源限额为 8 GiB RAM、4 核（10–13），不占用其他 WSL 用户的 GPU。
原 Python 环境导入 NumPy 时发生过段错误；本次使用已经按原包校验的同版本
NumPy 1.24.4 私有副本 `/tmp/scenemi-v2-runtime-overlay`，不修改原环境。
运行时版本、导入路径及核心二进制 SHA256 保存为 `RUNTIME_NUMPY.json`。

完整配置和清单为 `PROTOCOL.json`；实际导入源码及环境为 `EVALUATOR_SEAL.json`，
源码归档为 `evaluator_sources.tar.gz`。逐组保存 JSON、全局关节数组、输入和输出哈希；
每组同时完成两个主输出后才提交记录，可从中断处继续，不跳过失败案例。

执行输出目录：`runs/scene_joint128_comparison/step_001000_full_cpu`。
`STATUS.json` 是模型评估进度，`PIPELINE_STATUS.json` 是完整流程状态。
全部完成后，自动写出 `PAIRED_COMPARISON.json` 和 `REPORT_zh.md`，
包含同内参主表、历史内参对照、逐家族配对区间及九例诊断。
没有完成前的部分结果不得作为全量验证结论。

```bash
PYTHONPATH=/tmp/scenemi-v2-runtime-overlay \
PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/tmp/scenemi-comparison-pycache \
taskset -c 10,11,12,13 \
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 8 --cpu-count 4 -- \
  python -u -m scenemi.replay.comparison_reporting pipeline --device cpu
```

## GPU 批量执行核验

用户已明确允许本次完整对照共享 GPU，授权原文与配置/权重身份记录在各新输出目录的
`SHARING_AUTHORIZATION.json`。沿用 4 GiB PyTorch allocator 上限、额外 1 GiB CUDA
开销、至少 4 GiB 物理空闲，以及 8 GiB RAM / 四核 cgroup；不操作其他 WSL 任务。

`scenemi.replay.comparison_acceleration` 将每个窗口原本在 CPU 上由 seed 777
生成的相同噪声复制到批次，保持逻辑 batch=1 的采样语义。直接 CUDA batch=4 在
GRAIL EXO 首窗出现最大 0.1564 mm 的关节差异，超过预先固定的 0.1 mm 核验线；
batch=8 在 EXO 局部平面特征分解时触及 4 GiB allocator 限额。两次失败原始记录
保留在 `step_001000_full_cuda/BATCH_BENCHMARK.batch4.json` 和 `.batch8.json`。
这些核验均未启动正式 CUDA 全量评估，未提升显存上限或放宽误差线。

`scenemi.replay.comparison_geometry` 为独立的推理包装层：网络在 CUDA 上按批运行，
原 `relation_features` 在 CPU 上计算，仅缓存不随 DDIM 步变化的观测点和 KD 树；
每一步仍使用当前预测的 anchors/rotations。它不改动训练实现或已有权重。
配置为 `configs/scene_joint128_comparison_hybrid.yaml`，结果目录为
`runs/scene_joint128_comparison/step_001000_full_hybrid`。正式运行必须先通过全部
三数据集 EGO/EXO 的 CPU 对照和 batch=1 对照核验，各最大关节差异均不超过 0.1 mm。
完整 CPU 评估在核验期间继续；两套设备的逐例结果不会混入同一评估封印。

混合路径的 GRAIL EXO batch=8 与同设备逐窗结果仍相差最大 0.10434 mm，
因此最终配置 `configs/scene_joint128_comparison_gpu.yaml` 明确声明
`batch_overrides: {exo_grail: 1}`：该分组逐窗 CUDA，其余五个 mode/source 分组
物理 batch=8。核验和正式调度都读取同一覆盖表，0.1 mm 核验线保持不变。
最终执行目录为 `runs/scene_joint128_comparison/step_001000_full_gpu`。
报告中的 batch=1 表示每窗独立的原始噪声语义；实际执行 batch 还记录在
`EVALUATOR_SEAL.json`、配置和每条记录的 `execution_batch_size` 中。

北京时间 2026-10-30 00:59 左右观察到 WSL 重启，原 CPU/GPU 核验进程均退出。
后续Windows System1001和WER记录确认同期存在0xA蓝屏及主机重启；
00:58:50为重启后的事件登记时间，Event6008报告此前非正常关机为00:55:50。
故此处中断涉及主机系统崩溃，不能只归因于Python文件读取。
恢复前对 CPU 落盘记录逐条检查协议、评估器封印、公共输入一致性和预测数组 SHA256，
604 对通过，9 条损坏记录移到 `interrupted_corrupt` 后重新计算，详见
`step_001000_full_cpu/RESTART_RECOVERY.json`。原 numpy 私有副本的 870 个文件
按已有清单逐一重建并核验 SHA256，未修改原 conda 环境。

## 完整结果已完成

北京时间2026-10-30 07:09完成全部17640/17640对：3471个EGO窗口、3540个EXO窗口
各四种条件，另有九例诊断不计入主聚合。正式结果为
`runs/scene_joint128_comparison/step_001000_full_gpu/PAIRED_COMPARISON.json`，
可读结果为同目录`REPORT_zh.md`。CPU旧任务已由此GPU完整结果接替，不把其部分结果混入。

EGO三来源等权MPJPE为194.29→183.17mm；主要收益来自GRAIL，TRUMANS退化，
PA-MPJPE基本不变且jerk变差。主表EXO tracking/occlusion/gap的等权MPJPE分别
改善1.63/1.70/1.19mm，不能据此宣称消除漂移、抖动或缺失区域问题。

此处是旧场景1000步与无场景父版本2000步的比较，不是修正版场景训练结果。
EXO主表的无场景输入额外除以256，属输入干预；父版本的原生导出归一化内参结果
保存在`historical_no_scene`及报告第二张表中，单位误标已经在可读报告修正。
原JSON键名和封存reporter保持原样，解释时必须携带`INTRINSICS_UNIT_CORRECTION.json`。

## 论文解释范围

这是两个已有训练版本的完整原生验证对照，额外适配步数与 EXO 提案结构也不同，
不能直接称为纯场景输入消融；EXO 的 frontend 是合成估计扰动，不能写成真实 tracker 结果。
无场景父版本此前没有通过 AMASS EGO 保留门槛，本对照不会撤销这一失败记录。
验证结果可以用于研究诊断、消融规划和如实报告；最终独立泛化结论仍需冻结方案后的独立测试。
