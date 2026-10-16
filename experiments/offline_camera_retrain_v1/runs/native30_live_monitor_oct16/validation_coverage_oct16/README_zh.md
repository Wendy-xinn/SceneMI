# 当前验证覆盖与扩展清单

当前每1000步的监控来自官方 VAL：TRUMANS、EgoBody、RICH各1个固定窗和1个轮换窗，共6个不同的192帧/30Hz窗口。每窗分别用clean/noisy输入评估，共12次DDIM20采样；每5000步给固定窗另加scene-OFF，共18次，仍只有6个不同窗口。fixed用于同窗趋势；rotating用于扩大观察范围，不可把相邻不同窗口的误差当同窗学习曲线。

截至12k，累计见过39个不同窗口，但它们使用不同训练步的模型预测，不能拼成当前checkpoint的39窗评估。原有fixed三窗继续保留。

实际索引含1806个合法ego窗口：TRUMANS962、EgoBody638、RICH206，来自101个有效source bundle、76个录制。窗口长192帧、起点步长64帧，邻窗重叠2/3；这是当前网格的完整VAL，不是所有原始帧起点，未补短窗口或尾段。

已准备[EVALUATION_PLAN.json](EVALUATION_PLAN.json)的两个独立方案：

- `one_per_recording`：每个合法录制1窗，76窗。录制内按group/sequence排序取一source，再取其中位合法起点。
- `one_per_bundle`：每个合法source/目标角色片段1窗，101窗，覆盖全部76个录制；各source取中位合法起点。

两个方案均在读取预测结果前按索引规则确定，使用固定可复现seed、ego head-only、scene-ON和clean/noisy；它们是全录制/全source覆盖的扩展子集，不是1806窗全量。

常规6窗实测总耗时中位约142秒，折算每窗clean/noisy约23.6秒：76窗约30分钟、101窗约40分钟、1806窗约11.9小时。这是现有监控速度的线性估算，未计入新增窗口缓存准备、启动及共用GPU的减速；不是已完成评估的计时结果。

建议先在同一个冻结checkpoint上评101窗，按dataset→recording→source分层汇总，同时报告窗口均值和尾部误差，避免长录制/多角色重复支配总分。后续选定checkpoint再评全部1806窗。指标包括全身/下肢位置、方向、双脚局部运动、GT低速期滑步和root漂移；GT低速仅是运动学代理，不代表真实触地。

**状态：仅完成CPU清单核验，尚未启动GPU评估，也没有复制checkpoint或改动主线。** 真正执行时需冻结同一个已提交checkpoint及其SHA256，独立保存逐窗结果和断点，不写主线日志。当前只发现旧20Hz全量评估器，不能直接用于新版；需要复用新版builder/sample/metrics的独立runner。

复现：在仓库根目录使用scenemi环境运行 `prepare_plan.py`，核验manifest和索引SHA256后重建清单。没有模型导入、推理或训练更新。
