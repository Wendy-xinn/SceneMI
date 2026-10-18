# 500步 EGO 场景 attention：独立配对验证

结论：**这次 attention500 没有优于同预算 control500，不推广为长训版，正式41k保持原样。** 宏平均World相对control增加 clean 0.130mm / noisy 0.060mm，仅3/12、4/12窗口改善。差异很小，但没有支持额外结构收益的方向性证据。这只评价当前具体实现的500步 pilot，不能宣称所有attention方案无效或模型已收敛。

两臂均从41k续训500步、全参数更新、physics OFF。96个新预测与旧41k的48个同窗预测配对；12个独立录制，三数据集各4，每个录制各clean/noisy。两名审计分别核对了输入/身份与指标：递归输入指纹、观察、reference、GT、shape/rest、identity和分母跨模型精确一致；144个artifact SHA通过；所有guide调用为0；sceneoff的三个guard和geometry查询均为0。CPU独立公式复算World/PA/root-relative/head/足高/足速/速度/加速度，最大差异2.17e-5（对应指标单位），未初始化CUDA。

## 有场景结果：matched control 是主对照

所有指标均为越低越好。表中41k也关闭显式物理solver，保留所有learned scene路径；因此推理开关一致。

| 数据集/输入 | World mm：41k / control / attention | attention−control mm | World改善窗口 | PA mm：control→attention | root-relative mm：control→attention |
|---|---:|---:|---:|---:|---:|
| rich/clean | 119.09 / 128.64 / 128.76 | +0.128 | 2/4 | 94.15 → 94.26 | 121.14 → 121.30 |
| rich/noisy | 129.86 / 151.16 / 151.29 | +0.126 | 1/4 | 104.04 → 104.14 | 139.74 → 139.86 |
| trumans/clean | 112.38 / 101.27 / 101.39 | +0.123 | 1/4 | 82.81 → 82.83 | 96.31 → 96.43 |
| trumans/noisy | 128.39 / 120.44 / 120.49 | +0.048 | 1/4 | 86.11 → 86.08 | 106.65 → 106.78 |
| egobody/clean | 156.53 / 156.10 / 156.24 | +0.138 | 0/4 | 97.25 → 97.18 | 140.33 → 140.45 |
| egobody/noisy | 170.41 / 170.61 / 170.61 | +0.006 | 2/4 | 103.16 → 103.11 | 151.45 → 151.48 |
| macro/clean | 129.33 / 128.67 / 128.80 | +0.130 | 3/12 | 91.40 → 91.42 | 119.26 → 119.39 |
| macro/noisy | 142.88 / 147.40 / 147.46 | +0.060 | 4/12 | 97.77 → 97.78 | 132.61 → 132.70 |

两组500步续训都在RICH回退、TRUMANS改善，因此宏平均不能替代目标数据集的判断；这也不能归为attention独有问题。相对41k的变化同时包含继续训练、fresh Adam及训练时关闭solver等因素，这份对照不能分离它们。单独拿attention500对未续训41k，会把继续训练收益误认成新增结构收益。

## 头部与脚部时间质量

| 数据集/输入 | 头部GT误差 mm，control→attention | 足高误差 mm，control→attention | GT低速期水平足速 m/s，control→attention | 加速度误差 m/s²，control→attention |
|---|---:|---:|---:|---:|
| rich/clean | 0.066 → 0.066 | 38.32 → 38.18 | 0.2103 → 0.2106 | 6.750 → 6.749 |
| rich/noisy | 22.366 → 22.365 | 34.47 → 34.31 | 0.2282 → 0.2283 | 7.751 → 7.750 |
| trumans/clean | 0.078 → 0.078 | 33.11 → 33.44 | 0.1245 → 0.1249 | 2.485 → 2.487 |
| trumans/noisy | 27.439 → 27.460 | 33.87 → 34.26 | 0.1335 → 0.1335 | 2.588 → 2.583 |
| egobody/clean | 14.507 → 14.506 | 42.05 → 42.00 | 0.1123 → 0.1127 | 3.436 → 3.440 |
| egobody/noisy | 30.729 → 30.771 | 49.72 → 49.67 | 0.1351 → 0.1354 | 3.822 → 3.822 |
| macro/clean | 4.884 → 4.884 | 37.83 → 37.87 | 0.1491 → 0.1494 | 4.224 → 4.225 |
| macro/noisy | 26.844 → 26.866 | 39.35 → 39.41 | 0.1656 → 0.1657 | 4.720 → 4.718 |

相比41k，两组500步的宏平均足高和低速足速较好，但World、root-relative及PA的变化并不一致；不能仅凭脚部指标改好就整体替换。新增attention相对control的脚部差异也很小，没有一致优势。

## 录制分组与场景开关

dev/confirmation每数据集各2个不同录制，均来自验证集。负的World差表示attention更低。场景on=learned_only；off=none，保留原观察/IK/reference。

| 数据集/输入 | dev ΔWorld mm | confirmation ΔWorld mm | control：off→on World mm | attention：off→on World mm | 场景交互差 mm |
|---|---:|---:|---:|---:|---:|
| rich/clean | -0.040 | +0.296 | 153.25 → 128.64 | 153.69 → 128.76 | -0.313 |
| rich/noisy | -0.025 | +0.278 | 186.03 → 151.16 | 186.09 → 151.29 | +0.060 |
| trumans/clean | +0.088 | +0.158 | 148.58 → 101.27 | 149.36 → 101.39 | -0.659 |
| trumans/noisy | +0.083 | +0.014 | 166.65 → 120.44 | 166.99 → 120.49 | -0.298 |
| egobody/clean | +0.196 | +0.079 | 166.50 → 156.10 | 166.82 → 156.24 | -0.187 |
| egobody/noisy | +0.161 | -0.149 | 189.37 → 170.61 | 189.54 → 170.61 | -0.160 |
| macro/clean | +0.081 | +0.178 | 156.11 → 128.67 | 156.62 → 128.80 | -0.386 |
| macro/noisy | +0.073 | +0.048 | 180.68 → 147.40 | 180.87 → 147.46 | -0.133 |

场景交互差=(attention on−off)−(control on−off)，要与绝对on/off数值一起看：更大的场景收益也可能来自off结果变差。sceneoff虽旁路新增分支，两臂主干经过不同训练，off输出仍可能不同。

## 实现与解释边界

- 训练真实性见同级trial的UPDATE_SANITY_500.json：500条实际输入及前后RNG一致，末端sampler/RNG状态一致；原315个参数张量两臂均更新，attention新增8个张量/148800参数全部更新；模型及Adam moments有限。场景分支Adam计数444与444个sceneon步骤相符，不要求所有条件参数都计500。
- 架构审计确认新增local K/V使用raw region_context，未采用旧路径的alpha_bar²幅度门控。比较包含条件通路/强度变化，不能只归因于融合位置；Q为全身时间bottleneck特征，非逐关节token或时序self-attention。
- 500步只是固定horizon1500的早期pilot，不是收敛结论。没有自动延长、覆盖正式模型或把它选为最终长训版。
- 配对胜率只描述这个12录制面板。clean/noisy共享录制，96条预测不能当作96个独立样本。dev/confirmation虽录制互斥，均为VAL，没有使用TEST。
- 足速是在GT足关节三维速度<0.2m/s时的水平速度，GT低速不等于真实接触。足高误差相对GT，不是场景穿模率。
- 全部数据集、输入条件、panel、World/PA/root-relative/head/foot/时间指标、场景开关、交互与逐窗正负结果均保存在AUDIT_FINDINGS.json。此审计记录是在清理前对原始保存输出/权重的核验；未采用attention资产如后续删除，由独立CLEANUP记录封存精确身份和依赖检查。

保留策略：本次未采用attention500；root已启动其训练last.pt、评估hardlink及48条attention预测的records-only清理，保留可恢复的control500、48条control预测及旧41k的72条基线预测。精确文件身份、依赖检查与执行状态以 [CLEANUP.json](CLEANUP.json) 为准。本文指标审计及原CPU_RECHECK均为清理前证据。
