冻结 EXO 配对验证；仅官方 VAL，不训练、不读取 TEST。

原PLAN预选20个EXO候选窗口；本次PRIMARY实际评估12个（每数据集4个不同录制）× clean/noisy × 有/无场景，另3个EGO保持锚点×clean/noisy。三个模型共162次预测，剩余8个EXO候选未评估。实际范围见PRIMARY_SELECTION.json和DIRECTION_REVIEW_zh.md。
以下为每窗口等权均值；P90是窗口均值的P90，不能作为逐关节P90。未完成组保留实际窗口数。

| 模型/场景/输入/数据 | 窗口 | 遮挡均值 mm | 遮挡 P90 mm | 可见 mm | 支撑期足速 m/s | 加速度 mm/s² | 比参考差≥50mm窗 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base10k/scene_on/clean/rich | 4 | 35.758 | 57.089 | 2.469 | 0.144 | 1137.787 | 0 |
| base10k/scene_on/clean/trumans | 4 | 150.777 | 240.839 | 4.059 | 0.169 | 1503.510 | 0 |
| base10k/scene_on/clean/egobody | 4 | 23.639 | 53.162 | 2.008 | 0.127 | 1521.644 | 0 |
| base10k/scene_on/noisy/rich | 4 | 203.530 | 437.628 | 36.092 | 0.259 | 3829.740 | 2 |
| base10k/scene_on/noisy/trumans | 4 | 198.678 | 337.114 | 26.772 | 0.198 | 2067.371 | 0 |
| base10k/scene_on/noisy/egobody | 4 | 45.121 | 76.653 | 26.757 | 0.145 | 2765.800 | 0 |
| base10k/scene_off/clean/rich | 4 | 38.453 | 57.475 | 1.790 | 0.159 | 1184.598 | 0 |
| base10k/scene_off/clean/trumans | 4 | 176.617 | 293.200 | 4.566 | 0.244 | 1680.276 | 0 |
| base10k/scene_off/clean/egobody | 4 | 27.319 | 63.005 | 2.451 | 0.149 | 1605.263 | 0 |
| base10k/scene_off/noisy/rich | 4 | 75.369 | 91.519 | 25.848 | 0.190 | 2614.784 | 1 |
| base10k/scene_off/noisy/trumans | 4 | 214.815 | 363.319 | 27.845 | 0.286 | 2251.521 | 0 |
| base10k/scene_off/noisy/egobody | 4 | 46.777 | 88.877 | 25.111 | 0.174 | 2792.932 | 0 |
| control/scene_on/clean/rich | 4 | 35.412 | 56.860 | 2.453 | 0.140 | 1075.346 | 0 |
| control/scene_on/clean/trumans | 4 | 153.429 | 252.851 | 4.208 | 0.164 | 1448.394 | 0 |
| control/scene_on/clean/egobody | 4 | 23.277 | 52.460 | 1.914 | 0.126 | 1498.284 | 0 |
| control/scene_on/noisy/rich | 4 | 196.626 | 431.727 | 34.198 | 0.240 | 3521.161 | 1 |
| control/scene_on/noisy/trumans | 4 | 190.590 | 327.053 | 26.140 | 0.186 | 1966.865 | 0 |
| control/scene_on/noisy/egobody | 4 | 45.184 | 75.686 | 26.880 | 0.145 | 2744.074 | 0 |
| control/scene_off/clean/rich | 4 | 37.590 | 56.884 | 1.798 | 0.155 | 1121.252 | 0 |
| control/scene_off/clean/trumans | 4 | 171.210 | 290.449 | 4.639 | 0.235 | 1622.817 | 0 |
| control/scene_off/clean/egobody | 4 | 26.928 | 62.179 | 2.419 | 0.148 | 1590.709 | 0 |
| control/scene_off/noisy/rich | 4 | 70.984 | 83.486 | 24.102 | 0.183 | 2449.327 | 0 |
| control/scene_off/noisy/trumans | 4 | 206.554 | 354.099 | 27.055 | 0.279 | 2171.286 | 0 |
| control/scene_off/noisy/egobody | 4 | 46.895 | 88.156 | 25.406 | 0.174 | 2769.079 | 0 |
| nondegradation/scene_on/clean/rich | 4 | 35.633 | 57.199 | 2.442 | 0.138 | 1043.956 | 0 |
| nondegradation/scene_on/clean/trumans | 4 | 156.410 | 261.951 | 4.229 | 0.165 | 1440.822 | 0 |
| nondegradation/scene_on/clean/egobody | 4 | 23.781 | 53.867 | 1.851 | 0.128 | 1452.128 | 0 |
| nondegradation/scene_on/noisy/rich | 4 | 192.015 | 428.430 | 33.166 | 0.233 | 3385.071 | 1 |
| nondegradation/scene_on/noisy/trumans | 4 | 190.311 | 328.710 | 26.076 | 0.181 | 1947.911 | 0 |
| nondegradation/scene_on/noisy/egobody | 4 | 44.728 | 75.871 | 26.476 | 0.146 | 2642.152 | 0 |
| nondegradation/scene_off/clean/rich | 4 | 37.197 | 56.065 | 1.790 | 0.154 | 1091.841 | 0 |
| nondegradation/scene_off/clean/trumans | 4 | 172.504 | 295.608 | 4.649 | 0.232 | 1611.487 | 0 |
| nondegradation/scene_off/clean/egobody | 4 | 27.236 | 62.826 | 2.363 | 0.149 | 1558.243 | 0 |
| nondegradation/scene_off/noisy/rich | 4 | 67.539 | 78.084 | 22.881 | 0.175 | 2332.012 | 0 |
| nondegradation/scene_off/noisy/trumans | 4 | 206.317 | 354.867 | 27.003 | 0.272 | 2159.373 | 0 |
| nondegradation/scene_off/noisy/egobody | 4 | 46.410 | 87.753 | 25.081 | 0.173 | 2682.701 | 0 |

场景OFF同时关闭学习场景编码、局部查询与反馈；本版解析物理引导始终关闭。
原模型重跑与历史指标的差异逐行保存；CPU_RECHECK通过后才确认指标与输入一致性。
