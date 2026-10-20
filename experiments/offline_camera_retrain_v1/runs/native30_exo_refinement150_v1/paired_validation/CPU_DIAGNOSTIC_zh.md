当前 EXO10k 的 CPU 场景配对诊断（仅3个窗口）

2026-10-20 21:36 完成。模型为当前 natural10k step10000，冻结全部权重，0次优化更新，仅官方 VAL，没有读取 TEST。
原计划为20个窗口；GPU启动失败后，经root明确授权，按原先封存的身份顺序，每数据集只取首个主面板窗口。没有按GT、误差或可视化效果选择。每窗192帧、30Hz，clean/noisy各跑sceneON/OFF，共12次CPU推理。
这是明确标注的部分诊断，不能当成固定100窗、完整VAL或当前最终指标，也不用于重新选checkpoint。

| 数据集 | 窗口身份 | 起始帧 | CPU DDIM种子 |
|---|---|---:|---:|
| rich | ParkingLot2_014_eating2 | 256 | 3876338614 |
| trumans | 2023-02-18@22-06-51 | 576 | 4185193926 |
| egobody | recording_20220218_S23_S02_01_interactee_1551-3951_master_full | 2176 | 3933759198 |

下表都是遮挡/不可用位置关节的FK位置误差，单位mm；每格对应同一窗。OFF和ON采用完全一致的CPU输入与种子。

| 数据 | 输入 | 参考误差 | 场景OFF | 场景ON |
|---|---|---:|---:|---:|
| rich | clean | 8.92 | 24.04 | 20.86 |
| rich | noisy | 15.41 | 50.02 | 46.03 |
| trumans | clean | 63.50 | 140.69 | 111.37 |
| trumans | noisy | 78.73 | 123.11 | 127.87 |
| egobody | clean | 2.16 | 4.65 | 4.61 |
| egobody | noisy | 31.28 | 18.21 | 19.84 |

场景确实影响当前10k的输出：本小面板6个条件里4个遮挡指标下降、2个升高。TRUMANS clean较明显改善，TRUMANS/EgoBody noisy反而退化，因此不能称EXO场景收益已普遍证实。它也没有解决好参考被改坏的问题；以上只有EgoBody noisy的模型输出优于输入参考。
可见关节、足速、加速度与完整逐窗指标保存在rows.jsonl。sceneOFF同时关闭学习式全局/patch/局部查询/反馈，解析物理引导在ON和OFF下始终关闭，不能将总差值归因于单一编码器。

独立CPU复核通过：12个输出的NumPy/SciPy FK与原torch指标交叉，遮挡MPJPE最大差0.000034mm，加速度最大差0.0541mm/s²；同窗clean或noisy的ON/OFF在GT、骨架、观测、mask、参考上逐元素完全一致。OFF禁止调用guard均为0。EGO锚点本次未跑，CPU_RECHECK中的空EGO_retention不代表新验证了EGO。
同一个数字seed在CPU和CUDA不保证同一个初始扩散噪声；因此本次CPU ON/OFF是严格配对，与历史GPU预测只作描述性核对。重跑与历史raw_metrics差异逐行保留，没有要求CPU/CUDA逐值相同。
两个新150步模型未完成，本目录没有它们的预测或指标。没有自动替换或promote任何模型。

该诊断复用ops/refinement_evaluation.py，保留原20窗PLAN和显式CPU_PARTIAL_SELECTION.json；root批准的进程内身份过滤没有编辑封存源文件。自动REPORT_zh.md描述原20窗设计，实际范围以本说明、CPU_PARTIAL_SELECTION和逐行数据为准。
