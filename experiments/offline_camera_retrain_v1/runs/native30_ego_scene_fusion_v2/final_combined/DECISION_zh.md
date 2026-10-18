# 场景融合最终采纳判定

判定：**reject**。基于43个独立录制窗口、clean/noisy及两个预注册DDIM种子；先在录制内均值，再进行配对bootstrap。

| 对照 | RICH World差值 mm | 95% CI mm | 相对改善 | 全部门槛 |
|---|---:|---|---:|---|
| control | 0.151 | [-0.127, 0.454] | -0.09% | 未通过 |
| base41k | -1.369 | [-5.673, 2.784] | 0.79% | 未通过 |

control 未通过：RICH_mean_World_relative_improvement；RICH_bootstrap_upper_delta_mm；rich/clean/World；rich/noisy/World。

base41k 未通过：RICH_mean_World_relative_improvement；RICH_bootstrap_upper_delta_mm。

负差值表示融合模型误差更低；需同时对原41k和同预算control满足1%改善、置信上界及各项守卫。完整每项阈值、观测头误差监测和证据SHA见JSON。此结果不能替代完整窗口验证或官方test指标，也不自动替换保留模型。
