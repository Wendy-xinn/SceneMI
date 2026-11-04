冻结 EXO 场景分支定位；仅官方 VAL，原10k权重不变，不训练、不读TEST。

24个EXO窗口、3个EGO锚点 × clean/noisy × 2种策略，计划上限108次预测；当前60次。
全部保持use_scene=True、局部查询/BPS/region/feedback；策略和适用模式由PLAN逐项声明。
Fixed12 dev + independent recording-disjoint rotating12 confirmation + three EGO retention anchors. Dev chooses a policy before reading candidate confirmation.

| 策略/输入/数据 | 遮挡mm | 可见mm | 支撑期足速m/s | 加速度mm/s² |
|---|---:|---:|---:|---:|
| exo_patch_second_pass/scene_on/clean/rich | 35.719 | 2.505 | 0.146 | 1146.098 |
| exo_patch_second_pass/scene_on/clean/trumans | 150.731 | 4.070 | 0.169 | 1501.605 |
| exo_patch_second_pass/scene_on/clean/egobody | 23.647 | 2.009 | 0.127 | 1521.691 |
| exo_patch_second_pass/scene_on/noisy/rich | 203.808 | 36.267 | 0.258 | 3818.123 |
| exo_patch_second_pass/scene_on/noisy/trumans | 198.288 | 26.781 | 0.198 | 2064.851 |
| exo_patch_second_pass/scene_on/noisy/egobody | 45.127 | 26.783 | 0.145 | 2766.853 |
| original/scene_on/clean/rich | 35.758 | 2.469 | 0.144 | 1137.787 |
| original/scene_on/clean/trumans | 150.777 | 4.059 | 0.169 | 1503.510 |
| original/scene_on/clean/egobody | 23.639 | 2.008 | 0.127 | 1521.644 |
| original/scene_on/noisy/rich | 203.530 | 36.092 | 0.259 | 3829.740 |
| original/scene_on/noisy/trumans | 198.678 | 26.772 | 0.198 | 2067.371 |
| original/scene_on/noisy/egobody | 45.121 | 26.757 | 0.145 | 2765.800 |

指标须经独立CPU复核；不自动推广任何策略或checkpoint。
