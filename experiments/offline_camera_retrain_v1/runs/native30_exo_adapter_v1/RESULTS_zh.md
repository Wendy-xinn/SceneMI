本轮已完成 1000 次内部 adapter 更新；工程验收未通过。不自动替换正式模型。
本报告只使用本轮固定开发验证：24 次匹配预测，未使用 TEST 选择模型或调参。
本次启动：2026-10-18T02:08:04.903207+08:00；最终复核完成：2026-10-18T03:08:28.321897+08:00。
实际更新累计 34.5 分钟；本次启动至最终 CPU 复核完成 60.4 分钟，后者包含准备、等待和验证。

|输入|隐藏关节 MPJPE：适配前 → 后（mm）|等数据集宏平均门槛|
|---|---:|---|
|clean|114.534 → 86.994|通过（至少改善 5%）|
|noisy|135.566 → 129.028|失败（至少改善 5%）|

各数据集内按有效关节数加权；脚部 RMS/stride 误差按窗口平均。低速脚指标使用 GT 关节速度代理，不是实测接触。
逐数据集门槛：隐藏关节最多增加 max(5mm,5%)；可见关节 max(1mm,2%)；低速脚速度 max(0.01m/s,5%)；双足 RMS/stride 误差 max(5mm,10%)。

|数据集/输入|隐藏关节 mm|可见关节 mm|低速脚速度 m/s|双足运动 RMS 误差 mm|stride 误差 mm|
|---|---:|---:|---:|---:|---:|
|trumans/clean（3窗）|194.533 → 135.718 ✓|11.070 → 9.631 ✓|0.750 → 0.519 ✓|90.587 → 17.641 ✓|116.299 → 83.508 ✓|
|trumans/noisy（3窗）|217.708 → 179.675 ✓|51.543 → 40.079 ✓|0.700 → 0.574 ✓|132.500 → 21.865 ✓|117.534 → 88.820 ✓|
|egobody/clean（3窗）|68.190 → 36.242 ✓|9.073 → 4.631 ✓|0.231 → 0.185 ✓|23.901 → 15.515 ✓|31.585 → 23.309 ✓|
|egobody/noisy（3窗）|79.052 → 65.816 ✓|34.014 → 34.209 ✓|0.286 → 0.238 ✓|36.711 → 31.562 ✓|45.734 → 41.860 ✓|
|rich/clean（3窗）|80.879 → 89.021 失败|7.670 → 0.800 ✓|0.301 → 0.267 ✓|16.816 → 30.967 失败|26.740 → 24.718 ✓|
|rich/noisy（3窗）|109.939 → 141.591 失败|39.293 → 31.795 ✓|0.310 → 0.268 ✓|18.985 → 29.463 失败|35.065 → 39.882 ✓|

未通过项：验收.groups[rich/clean].hidden_no_material_regression_pass；验收.groups[rich/clean].motion_guards.bilateral_motion_rms_error_mm.passed；验收.groups[rich/noisy].hidden_no_material_regression_pass；验收.groups[rich/noisy].motion_guards.bilateral_motion_rms_error_mm.passed；验收.hidden_macro[noisy].passed

EGO 保存输出逐值一致：True；非零 adapter 的 EGO 旁路预检：True；无人体动作标签的独立 EXO 加载预检：True。
冻结主干 SHA：`4d789e1e63f1681885c000b1d1795ba444b5a3eee3fd6af225fc9b7b0da6008f`；最终完整 checkpoint SHA：`d410954559d4d9c4847aad295dab7f51f9ce24348de43c1e8d97d4998a8586bd`。
最终独立部署加载：[已通过](DEPLOYMENT_LOAD_CHECK.json)。

尺度诊断仅解释本轮 EXO 开发窗口：每窗口一个固定刚体/相似变换，GT 对齐不用于修改预测。表中先按数据集平均，再对数据集等权；RMS 是平方误差开方，MPJPE 是欧氏距离平均，两者不可替代。

|阶段/输入|世界 RMS mm|GT 刚体 RMS mm|GT 相似 RMS mm|可见拟合尺度|
|---|---:|---:|---:|---:|
|baseline/clean|77.289|71.953|71.782|1.001|
|baseline/noisy|99.886|87.082|86.453|1.011|
|final/clean|57.389|54.596|54.351|0.999|
|final/noisy|95.228|88.881|88.079|1.000|

[当前 Viser](http://localhost:8802/)。原始证据：[验收](ACCEPTANCE.json)、[匹配比较](COMPARISON.json)、[独立 CPU 复核](CPU_RECHECK.json)、[尺度诊断](ALIGNMENT_DIAGNOSTICS.json)。
