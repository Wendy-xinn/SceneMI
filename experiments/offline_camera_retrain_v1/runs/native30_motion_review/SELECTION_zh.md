# 统一动作展示选例

仅复用已保存的41k EGO和EXO adapter v2预测，没有新训练或扩散推理。选例为有意的定性展示，不能代替全体验证。

EGO库存：TRUMANS 2个独立录制、EgoBody 2个、RICH 3个；EXO为TRUMANS 2个、EgoBody 2个、RICH 3个。EgoBody wearer静止且误差较大，明确保留库存限制；第三个RICH EGO为原地动态。

|案例|GT水平路径 m|clean World MPJPE mm|说明|
|---|---:|---:|---|
|TRUMANS EGO ／ 2023-02-18@22-11-49 ／ 41k|1.295|112.06|物理引导 on|
|TRUMANS EGO ／ 2023-02-17@10-53-45 ／ 41k|1.527|112.13|物理引导 on|
|EGOBODY EGO ／ recording_20210921_S11_S10_01 ／ 41k|0.111|225.75|物理引导 on|
|EGOBODY EGO ／ recording_20211002_S03_S18_04 ／ 41k|3.040|84.54|物理引导 on|
|RICH EGO ／ LectureHall_020_wipingchairs1 ／ 41k|2.072|109.09|物理引导 on|
|RICH EGO ／ Pavallion_002_sidebalancerun ／ 41k|1.426|130.40|物理引导 off|
|RICH EGO ／ LectureHall_003_wipingchairs1 ／ 41k|5.327|136.22|物理引导 off|
|TRUMANS EXO ／ 2023-02-18@16-55-22|1.200|137.84|EXO前后同输入/GT/rest完全一致|
|TRUMANS EXO ／ 2023-02-18@22-17-40|1.679|17.77|EXO前后同输入/GT/rest完全一致|
|EGOBODY EXO ／ recording_20220215_S21_S22_01|1.414|12.30|EXO前后同输入/GT/rest完全一致|
|EGOBODY EXO ／ recording_20211002_S03_S18_01|1.521|18.96|EXO前后同输入/GT/rest完全一致|
|RICH EXO ／ Pavallion_006_yoga1 ／ RICH失败对照|1.301|17.79|EXO前后同输入/GT/rest完全一致|
|RICH EXO ／ LectureHall_020_wipingchairs1|1.344|16.30|EXO前后同输入/GT/rest完全一致|
|RICH EXO ／ ParkingLot1_004_phonetalk2 ／ 固定窗口补充|0.190|18.80|EXO前后同输入/GT/rest完全一致|

EXO“输入参考mesh”来自实际reference_motion。它与“41k直接EXO输入的扩散输出”是两个不同图层，不应互相替代。

|EXO录制|条件|输入参考未提供部位误差 mm|原41k误差 mm|适配后误差 mm|
|---|---|---:|---:|---:|
|2023-02-18@16-55-22|clean|146.41|296.02|152.89|
|2023-02-18@16-55-22|noisy|174.39|328.46|184.89|
|2023-02-18@22-17-40|clean|97.78|85.93|77.12|
|2023-02-18@22-17-40|noisy|112.89|123.78|104.87|
|recording_20220215_S21_S22_01|clean|83.36|44.13|16.97|
|recording_20220215_S21_S22_01|noisy|112.79|54.27|46.25|
|recording_20211002_S03_S18_01|clean|157.05|87.72|49.30|
|recording_20211002_S03_S18_01|noisy|165.95|101.67|73.45|
|Pavallion_006_yoga1|clean|80.95|106.31|106.25|
|Pavallion_006_yoga1|noisy|117.61|195.46|314.04|
|LectureHall_020_wipingchairs1|clean|83.81|72.20|80.39|
|LectureHall_020_wipingchairs1|noisy|100.78|86.35|85.24|
|ParkingLot1_004_phonetalk2|clean|15.28|71.85|35.75|
|ParkingLot1_004_phonetalk2|noisy|56.41|68.18|25.77|

不能只比较“原41k”与“适配后”就断言超过输入：TRUMANS case6 clean适配结果相较原41k改善，但未超过输入参考；RICH cases10/11有退化，仍保留。

原始GT、rest、clean/noisy observations、observation_meta、reference_motion在EXO适配前后逐元素一致，检查详见SELECTION.json。

RICH TEST更好移动例为case16、237、225；原npz无identity_json，尚未用于该清单。
