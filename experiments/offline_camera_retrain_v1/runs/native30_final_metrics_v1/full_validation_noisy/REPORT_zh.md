冻结模型正式指标；未完成结果不能作为完整测试集结果。

split=full-validation；完成7082/7082；physics OFF；模型由VAL预先确定。

| mode/dataset/regime/scene | 完成/计划 | World MPJPE mm | 遮挡 mm | 可见 mm |
|---|---:|---:|---:|---:|
| ego/rich/noisy/scene_on | 206/206 | 179.436 | 186.552 | 25.798 |
| ego/rich/noisy/scene_off | 206/206 | 226.344 | 235.630 | 25.806 |
| ego/trumans/noisy/scene_on | 962/962 | 112.262 | 116.269 | 25.335 |
| ego/trumans/noisy/scene_off | 962/962 | 154.189 | 160.045 | 27.170 |
| ego/egobody/noisy/scene_on | 638/638 | 139.564 | 144.372 | 35.237 |
| ego/egobody/noisy/scene_off | 638/638 | 167.768 | 173.870 | 35.284 |
| exo/rich/noisy/scene_on | 206/206 | 32.507 | 81.533 | 24.300 |
| exo/rich/noisy/scene_off | 206/206 | 29.758 | 73.395 | 23.775 |
| exo/trumans/noisy/scene_on | 709/709 | 79.134 | 172.022 | 28.477 |
| exo/trumans/noisy/scene_off | 709/709 | 79.688 | 175.213 | 28.217 |
| exo/egobody/noisy/scene_on | 820/820 | 34.244 | 62.940 | 25.389 |
| exo/egobody/noisy/scene_off | 820/820 | 36.874 | 68.912 | 25.997 |

Native192/30Hz/S64,22 common joints,known body profile,synthetic/noisy sensor observations,observed static recording union; not a matched AMASS-only paper protocol.
CPU_RECHECK通过后才确认完整指标；TEST不用于改模型或调系数。
