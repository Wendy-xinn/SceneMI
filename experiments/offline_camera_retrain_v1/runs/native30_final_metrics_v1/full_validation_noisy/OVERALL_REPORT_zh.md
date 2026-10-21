冻结原始模型：跨数据集指标汇总。

完成 7082/7082 次预测；状态 sampling_complete_CPU_status_separate。
checkpoint step=10000，SHA256=3247383297ca2d7129755c6cd96c82eb2009cb2758aa8e22b6e5d0acb04c66b5。

数值格式：窗口加权 overall / 等数据集 macro。缺少任一计划数据集时 macro 为 —；partial 不代表完整验证集。

| 输入/观测/场景 | 完成/计划窗 | 数据集覆盖 rich,trumans,egobody | World MPJPE mm | 未提供部位 mm | 已提供部位 mm |
|---|---:|---|---:|---:|---:|
| ego/noisy/scene_on (complete) | 1806/1806 | rich 206/206, trumans 962/962, egobody 638/638 | 129.569 / 143.754 | 134.214 / 149.065 | 28.886 / 28.790 |
| ego/noisy/scene_off (complete) | 1806/1806 | rich 206/206, trumans 962/962, egobody 638/638 | 167.216 / 182.767 | 173.551 / 189.849 | 29.881 / 29.420 |
| exo/noisy/scene_on (complete) | 1735/1735 | rich 206/206, trumans 709/709, egobody 820/820 | 52.382 / 48.628 | 110.409 / 105.498 | 26.521 / 26.055 |
| exo/noisy/scene_off (complete) | 1735/1735 | rich 206/206, trumans 709/709, egobody 820/820 | 53.525 / 48.773 | 113.566 / 105.840 | 26.640 / 25.996 |
| combined_modes/noisy/scene_on (complete) | 3541/3541 | rich 412/412, trumans 1671/1671, egobody 1458/1458 | 91.749 / 94.836 | 122.643 / 124.577 | 27.727 / 27.138 |
| combined_modes/noisy/scene_off (complete) | 3541/3541 | rich 412/412, trumans 1671/1671, egobody 1458/1458 | 111.511 / 114.927 | 144.395 / 145.845 | 28.293 / 27.489 |

有无场景配对：仅同窗口、同观测、同seed、同完整input SHA且ON/OFF均已提交。Δ=ON−OFF；负值只表示该指标数值下降。

| 输入/观测 | 配对/计划窗 | 输入不匹配 | Δ World mm | Δ 未提供 mm | Δ 已提供 mm |
|---|---:|---:|---:|---:|---:|
| ego/noisy (complete) | 1806/1806 | 0 | -37.647 / -39.013 | -39.337 / -40.784 | -0.995 / -0.630 |
| exo/noisy (complete) | 1735/1735 | 0 | -1.143 / -0.145 | -3.157 / -0.342 | -0.119 / 0.059 |
| combined_modes/noisy (complete) | 3541/3541 | 0 | -19.761 / -20.091 | -21.751 / -21.267 | -0.566 / -0.350 |

原整体CPU校验：not_completed_for_this_snapshot；原失败记录和阈值保留。
CPU保存指标重算：passed；重算 7082/7082，最大差 0.0；ON/OFF完整配对 3541。
独立NumPy FK严格校验：failed_strict；原阈值未改。
原CUDA输入精确重建：passed；3541/3541 个窗口，覆盖 7082/7082 份场景结果。
独立数值异常 foot_joint_speed_gt_low_m_s：最大差 0.0002716587819679983，原容差 0.0001。
足速边界补证：6 份结果、3 个重叠窗口在GT足速0.2m/s硬边界出现选帧差异；使用原选帧时独立最大差 3.763114201166262e-06m/s。[CPU_FOOT_BOUNDARY_AUDIT.json](CPU_FOOT_BOUNDARY_AUDIT.json) 仅解释该异常，不将原严格失败改为通过。
原整体CPU校验尝试状态：failed；分阶段报告不覆盖原门。
全部评估指标及有效样本加权数值见 OVERALL_SUMMARY.json；不自动选择模型，不声明与其他论文输入/训练协议一致。
EGO与EXO按不同输入协议分列；若启用combined_modes，该合并只作异质输入描述。

Native192/30Hz/S64,22 common joints,known body profile,synthetic/noisy sensor observations,observed static recording union; not a matched AMASS-only paper protocol.
