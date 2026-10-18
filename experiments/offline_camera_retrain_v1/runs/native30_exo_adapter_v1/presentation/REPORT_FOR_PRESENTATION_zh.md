今晚可直接汇报的五句话见 [口播稿](TALK_zh.md)。本次更新只转录已完成、经过独立CPU复核的记录，没有新增模型推理或修改训练/评估协议。

EGO主训练输入已审计：完整55k的110,000个样本均为EGO，EXO为0；真实EGO为15,273个，伪EGO为94,727个。真实EGO使用EgoBody wearer的PV相机轨迹，伪EGO使用目标自己的虚拟头相机；另有已知身体信息和可见场景输入。后续EXO adapter是独立适配阶段，不应写成EGO主训练混入了EXO。见 [离线输入审计](EGO_TRAINING_INPUT_AUDIT_zh.md)。

最新v2开发验证摘要（已完成1000步，整体工程验收仍未通过）：

v2沿用相同41k冻结基座、adapter初值与1000步预算，将训练采样调整为三数据集等配额。在固定9个EXO开发窗上，隐藏关节误差的等数据集宏均值clean从114.534降至80.956 mm、改善29.32%；noisy从135.566降至121.297 mm、改善10.53%。3个固定EGO窗口×clean/noisy的保存输出逐值不变。这是固定开发面板上的结果，未证明完整EGO TEST逐值不变或EXO总体泛化。

| 数据 | 输入 | 隐藏关节mm：前→后 | 隐藏改善率 | 可见关节mm：前→后 | GT低速时3D脚速m/s：前→后 |
|---|---|---:|---:|---:|---:|
| trumans | clean | 194.533 → 130.359 | +32.99% | 11.070 → 8.934 | 0.750 → 0.521 |
| trumans | noisy | 217.708 → 171.064 | +21.43% | 51.543 → 40.561 | 0.700 → 0.552 |
| egobody | clean | 68.190 → 36.400 | +46.62% | 9.073 → 4.462 | 0.231 → 0.190 |
| egobody | noisy | 79.052 → 63.353 | +19.86% | 34.014 → 32.288 | 0.286 → 0.242 |
| rich | clean | 80.879 → 76.108 | +5.90% | 7.670 → 1.379 | 0.301 → 0.264 |
| rich | noisy | 109.939 → 129.476 | -17.77% | 39.293 → 29.988 | 0.310 → 0.271 |

RICH clean的隐藏误差改善5.90%，但noisy从109.939增至129.476 mm（退化17.77%），超过不退化门槛。RICH双足运动幅值误差clean为16.816→23.101 mm、noisy为18.985→24.389 mm，两组均未通过；可见关节与GT低速3D脚速门槛六组均通过。虽然宏平均改善超过5%，整体验收仍是failed。v2 checkpoint SHA256为`25d78ae22422345d8ac6cefe84a120bc510e6504bd15a39bd856be9f46e2e95d`。独立CPU复核检查480个主指标值，最大重算差6.10e-5，输入数组匹配，未用GPU；脚部运动指标另有独立公式交叉检查。

可见/隐藏按输入位置可用mask划分；数据集内按有效关节帧数加权，三数据集宏平均等权。GT低速3D脚速按有效低速区间数加权，双足幅值/时间差分指标按窗口等权。EXO输入仍为GT位置结合first-hit可见性与置信度的tracker模拟，使用已知body profile，尚未验证真实tracker输出、未知体型或真实尺度漂移。

扩大EXO验证已预先固定64个不同录制，三模型×clean/noisy共384次生成，新55个录制与旧9个回归录制分开统计；该扩大验证尚未完成，不能提前写成v2已在64个录制上取得收益。

以下保留原RICH TEST口径与历史v1结果，供对照；所有现有图片和视频均来自v1，不是v2可视化。

RICH TEST主表（冻结41k EGO，等窗口平均）：

| 输入 | World MPJPE mm | PA-MPJPE mm | Root-relative MPJPE mm | GT低速时预测水平脚速 m/s |
|---|---:|---:|---:|---:|
| clean | 155.672 | 113.944 | 146.876 | 0.279 |
| noisy | 169.478 | 121.101 | 155.043 | 0.337 |

协议：30Hz/192帧/步长64/DDIM20；头部观测、已知体型、该相机录制内实际观测场景，场景与物理修正开启。World不对齐；PA为逐帧22关节相似变换；root-relative减各自骨盆。CSV另列52录制等权宏均值。窗口可重叠，335窗不等于335个独立样本；缺失拟合附近的窗口被排除。当前三数据集训练/场景输入协议尚未与AMASS-only论文协议对齐，不能直接声称优于EgoGVAE、EgoEgo或EgoAllo。

脚速代理：在GT左右脚关节(10/11)三维速度<0.2 m/s的相邻帧区间计算预测脚的水平(XZ)速度，再等权平均窗口。它不是真实接触、官方HSC、足底速度或mesh穿模率；EXO验收使用同一GT低速掩码下的三维预测脚速，二者不能混报。

历史v1 EXO前后对照（开发VAL，每数据集3个不同录制窗口；下表不是v2）：

| 数据 | 输入 | 隐藏关节mm：前→后 | 隐藏改善率 | 可见关节mm：前→后 | 可见改善率 |
|---|---|---:|---:|---:|---:|
| trumans | clean | 194.533 → 135.718 | +30.2% | 11.070 → 9.631 | +13.0% |
| trumans | noisy | 217.708 → 179.675 | +17.5% | 51.543 → 40.079 | +22.2% |
| egobody | clean | 68.190 → 36.242 | +46.9% | 9.073 → 4.631 | +49.0% |
| egobody | noisy | 79.052 → 65.816 | +16.7% | 34.014 → 34.209 | -0.6% |
| rich | clean | 80.879 → 89.021 | -10.1% | 7.670 → 0.800 | +89.6% |
| rich | noisy | 109.939 → 141.591 | -28.8% | 39.293 → 31.795 | +19.1% |

改善率=(前−后)/前，负值为退化。可见/隐藏按输入位置可用掩码划分关节帧（conditioned/unconditioned），不等于整个人的语义遮挡分类。可见/隐藏误差在数据集内按有效关节帧数加权，EXO脚速按GT低速脚区间数加权；双足RMS/stride误差为等窗口平均，CSV已分别标明。双足RMS衡量各自骨盆朝向下、左右脚相对运动的去均值幅值；stride比较6/12帧间隔的变化量，不能当成真实步长或接触率。历史v1三数据集等权隐藏宏均值clean改善24.05%，noisy改善4.82%，后者未达预注册5%门槛。RICH可见关节贴合虽改善，隐藏姿态与双足幅值仍退化。

范围：EXO输入当前为GT关节位置结合first-hit可见性/置信度，并按既定noisy协议加入扰动（ideal_GT_position_tracker_simulation_with_firsthit_confidence），并非真实人体tracker输出。实验使用已知body profile，尚未验证未知体型或真实尺度漂移的修正；隐藏补全的改善不能据此表述为已解决真实tracker误差或尺度问题。

9个EXO窗仅为工程筛查，未完成EXO全验证/TEST或显著性检验。每阶段另有3个固定EGO窗，12窗×clean/noisy=24次生成，前后共48次，其中EXO36次。EGO TEST来自原冻结41k；EXO结果来自其内部adapter适配，两任务不混成总分。


文件与证据（所有主入口可在解压后的目录中离线打开）：

- [RICH TEST完整61项表与说明](rich_test_ego41k_results/results/FULL_RESULTS_zh.md)、[全部指标CSV](rich_test_ego41k_results/results/ALL_METRICS.csv)、[逐录制CSV](rich_test_ego41k_results/results/PER_RECORDING.csv)、[逐序列CSV](rich_test_ego41k_results/results/PER_SEQUENCE.csv)、[670次逐窗口CSV](rich_test_ego41k_results/results/PER_WINDOW.csv)、[覆盖范围](rich_test_ego41k_results/results/COVERAGE.csv)、[独立完整结果ZIP](RICH_TEST_ALL_RESULTS.zip)。这些结果属于冻结41k EGO，不是最终55k或EXO。
- [EGO输入审计阅读版](EGO_TRAINING_INPUT_AUDIT_zh.md)、[原始MD证据](evidence/EGO_TRAINING_INPUT_AUDIT_zh.md)、[原始JSON证据](evidence/EGO_TRAINING_INPUT_AUDIT.json)。原始审计内的绝对源码路径是原机定位，离线阅读版已转成文字，不随材料打包源码。
- [v2六组开发验证CSV](EXO_V2_VAL_BY_DATASET.csv)、[v2验收](evidence/EXO_V2_ACCEPTANCE.json)、[v2前后记录](evidence/EXO_V2_COMPARISON.json)、[v2独立CPU复核](evidence/EXO_V2_CPU_RECHECK.json)。
- [RICH主表CSV](RICH_TEST_MAIN.csv)、[历史v1分数据集CSV](EXO_VAL_BY_DATASET.csv)、[历史v1隐藏宏均值CSV](EXO_HIDDEN_MACRO.csv)、[历史v1验收](evidence/EXO_ACCEPTANCE.json)、[历史v1 CPU复核](evidence/EXO_CPU_RECHECK.json)。V1和V2原始证据不互相覆盖；历史文件身份和当前SHA映射见 [材料来源记录](PRESENTATION_PROVENANCE.json)。
- [历史v1数据图 PNG](visuals/EXO_hidden_metrics.png) / [SVG](visuals/EXO_hidden_metrics.svg)、[历史v1视频图片索引](visuals/index.html)、[历史v1三案例图](visuals/EXO_examples.png)。
- 历史v1视频：[TRUMANS](visuals/case06_clean.mp4)、[EgoBody](visuals/case09_clean.mp4)、[RICH失败例](visuals/case10_noisy.mp4)。视频同机位、同尺度，无额外对齐或贴地；选例不能代表总体。
- [Viser新版v2交互查看](http://localhost:8803/) 对应本轮1000步适配，可在线检查适配前/后；与历史v1的8802分别提供。两个Viser入口均需运行中的本地服务及端口转发，属于在线资源，不能离线打开；包内现有静态图和视频仍为历史v1。
- [Viser历史v1交互查看](http://localhost:8802/) 需要运行中的本地服务及端口转发，不属于离线材料；使用case6、case9及case10/noisy的适配前/后菜单。
- [五句口播](TALK_zh.md)；压缩包从 [index.html](index.html) 打开。GIF仅留原本地目录，精简包排除GIF、模型权重和运动数组。
