今晚可直接汇报的三句话：

1. 冻结41k EGO扩散模型已完成RICH TEST评估，覆盖335个合法窗口、52个录制、53条目标人体序列，clean/noisy的World MPJPE为155.7/169.5 mm，结果已独立CPU复核。
2. EXO采用扩散网络内部的专用adapter，冻结原主干、更新1000步；9个开发验证窗中，TRUMANS隐藏关节误差降低30.2%/17.5%，EgoBody降低46.9%/16.7%，3个固定EGO验证窗的保存输出逐值保持不变。
3. RICH的EXO隐藏关节误差反而增加10.1%/28.8%，双足运动幅值误差也变大，因此本轮工程验收未通过；今晚展示前后对照和失败例，暂不宣称已经解决遮挡与滑步。

RICH TEST主表（冻结41k EGO，等窗口平均）：

| 输入 | World MPJPE mm | PA-MPJPE mm | Root-relative MPJPE mm | GT低速时预测水平脚速 m/s |
|---|---:|---:|---:|---:|
| clean | 155.672 | 113.944 | 146.876 | 0.279 |
| noisy | 169.478 | 121.101 | 155.043 | 0.337 |

协议：30Hz/192帧/步长64/DDIM20；头部观测、已知体型、该相机录制内实际观测场景，场景与物理修正开启。World不对齐；PA为逐帧22关节相似变换；root-relative减各自骨盆。CSV另列52录制等权宏均值。窗口可重叠，335窗不等于335个独立样本；缺失拟合附近的窗口被排除。当前三数据集训练/场景输入协议尚未与AMASS-only论文协议对齐，不能直接声称优于EgoGVAE、EgoEgo或EgoAllo。

脚速代理：在GT左右脚关节(10/11)三维速度<0.2 m/s的相邻帧区间计算预测脚的水平(XZ)速度，再等权平均窗口。它不是真实接触、官方HSC、足底速度或mesh穿模率；EXO验收使用同一GT低速掩码下的三维预测脚速，二者不能混报。

EXO前后对照（开发VAL，每数据集3个不同录制窗口）：

| 数据 | 输入 | 隐藏关节mm：前→后 | 隐藏改善率 | 可见关节mm：前→后 | 可见改善率 |
|---|---|---:|---:|---:|---:|
| trumans | clean | 194.533 → 135.718 | +30.2% | 11.070 → 9.631 | +13.0% |
| trumans | noisy | 217.708 → 179.675 | +17.5% | 51.543 → 40.079 | +22.2% |
| egobody | clean | 68.190 → 36.242 | +46.9% | 9.073 → 4.631 | +49.0% |
| egobody | noisy | 79.052 → 65.816 | +16.7% | 34.014 → 34.209 | -0.6% |
| rich | clean | 80.879 → 89.021 | -10.1% | 7.670 → 0.800 | +89.6% |
| rich | noisy | 109.939 → 141.591 | -28.8% | 39.293 → 31.795 | +19.1% |

改善率=(前−后)/前，负值为退化。可见/隐藏按输入位置可用掩码划分关节帧（conditioned/unconditioned），不等于整个人的语义遮挡分类。可见/隐藏误差在数据集内按有效关节帧数加权，EXO脚速按GT低速脚区间数加权；双足RMS/stride误差为等窗口平均，CSV已分别标明。双足RMS衡量各自骨盆朝向下、左右脚相对运动的去均值幅值；stride比较6/12帧间隔的变化量，不能当成真实步长或接触率。三数据集等权隐藏宏均值clean改善24.05%，noisy改善4.82%，后者未达预注册5%门槛。RICH可见关节贴合虽改善，隐藏姿态与双足幅值仍退化。

范围：EXO输入当前为GT关节位置结合first-hit可见性/置信度，并按既定noisy协议加入扰动（ideal_GT_position_tracker_simulation_with_firsthit_confidence），并非真实人体tracker输出。实验使用已知body profile，尚未验证未知体型或真实尺度漂移的修正；隐藏补全的改善不能据此表述为已解决真实tracker误差或尺度问题。

9个EXO窗仅为工程筛查，未完成EXO全验证/TEST或显著性检验。每阶段另有3个固定EGO窗，12窗×clean/noisy=24次生成，前后共48次，其中EXO36次。EGO TEST来自原冻结41k；EXO结果来自其内部adapter适配，两任务不混成总分。

文件与证据：

- [RICH主表CSV](RICH_TEST_MAIN.csv)：原始[汇总副本](evidence/RICH_TEST_SUMMARY.json)、[CPU复核副本](evidence/RICH_TEST_CPU_RECHECK.json)，670/670，原公式差0、独立公式最大差5.68e-14。
- [EXO分数据集CSV](EXO_VAL_BY_DATASET.csv)、[隐藏宏均值CSV](EXO_HIDDEN_MACRO.csv)：原始[验收副本](evidence/EXO_ACCEPTANCE.json)、[CPU复核副本](evidence/EXO_CPU_RECHECK.json)。480个主指标值重算最大差6.10e-5，匹配输入数组。副本与源文件SHA相同，来源映射见[PRESENTATION_PROVENANCE.json](PRESENTATION_PROVENANCE.json)。
- 本材料只转抄已验证记录并计算改善百分比，未新增GPU采样、训练或重跑指标。今晚可视化应展示输入可见点、适配前/后、GT，以及RICH失败例。

直接用于汇报的素材：

- [数据图 PNG](visuals/EXO_hidden_metrics.png) / [SVG](visuals/EXO_hidden_metrics.svg)。
- [视频与图片索引](visuals/index.html)、[三案例汇总](visuals/EXO_examples.png)。
- [TRUMANS前后MP4](visuals/case06_clean.mp4)、[EgoBody前后MP4](visuals/case09_clean.mp4)、[RICH失败MP4](visuals/case10_noisy.mp4)。视频同机位、同尺度，无额外对齐或贴地。
- [Viser交互查看](http://localhost:8802/)；使用case6、case9及case10/noisy对应的适配前/后菜单。
- [五句口播稿](TALK_zh.md)。压缩包解压后从[index.html](index.html)打开，静态图、MP4和证据均使用包内相对路径（GIF保留在本地目录，不随精简压缩包分发）；Viser仍需要服务运行及本地端口转发。

后续动作（尚无新结果）：已准备同41k基座、同adapter初值、同1000步预算的三数据集等配额实验；扩大EXO面板已固定为64个不同录制，其中新55、旧9回归单列。等配额v2与扩大面板尚无可汇报的完成结果，不能把计划写成已有收益。
