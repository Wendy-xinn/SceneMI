现有10k EXO失效的根节点/局部姿态离线归因

Three highest noisy hidden-MPJPE RICH EXO cases in existing step10000 fixed panel; diagnostic selection only, not a new benchmark panel.

GT替换仅作离线oracle分析；不属于可运行推理方案、不参与gate或模型选择。没有加载模型、训练或新增推理。

| case | recording | 原hidden | 参考hidden | 换GT root T | 换GT root R | 换GT root T+R | 换GT非root局部 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Pavallion_006_yoga1 | 585.78 | 95.26 | 561.37 | 649.58 | 625.46 | 216.72 |
| 6 | LectureHall_018_wipingtable1 | 135.98 | 167.91 | 120.21 | 147.54 | 138.25 | 136.30 |
| 4 | ParkingLot1_005_overfence2 | 106.70 | 133.16 | 102.05 | 134.21 | 130.08 | 49.56 |

局部关节SO3增量、root平移/旋转误差分位数、可见比例、单关节GT替换诊断以及输入SHA详见同名JSON。Root和局部效应会相互抵消，替换收益不可相加。
