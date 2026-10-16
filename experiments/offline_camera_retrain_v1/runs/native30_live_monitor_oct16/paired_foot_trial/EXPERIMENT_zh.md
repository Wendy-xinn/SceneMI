# 脚步配对短训

用户授权先验证关闭独立物理修正参与训练的方案。原55k训练持续运行，不修改主线、GT监督、头部锚定、IK、学习式场景路径或监督系数。

固定完整24500步父checkpoint，保留模型、Adam、sampler和全部RNG。control保留物理修正，no_physics关闭；两臂各1000步至25500，学习率继续使用55000步总计划，不重置优化器。

逐步审计动作监督、观测、噪声、t、场景几何、录制身份、scene dropout和LR；两臂最终sampler/RNG也须一致。短训统一不插入周期验证，结束后固定配对评价。

14个不同录制窗口（RICH12、TRUMANS1、EgoBody1），均为开发VAL；含已有失败例与俯卧撑/瑜伽反例。三个模型在相同输入/seed下统一关闭物理修正评价，5核心窗另交叉开启。共114次DDIM20，不是TEST或新的全RICH成绩。

同时看World/PA、头部误差、脚水平/垂直速度、低速期脚高、根高、姿态高度、双足运动幅度/stride/时序、同一可靠可见局部表面的GT校准偏差。不能把脚不动或强行压低root当作修复；稀疏表面锚点比例包含合法摆脚，不等于全mesh悬空/穿模率。

运行状态见STATUS.json及两臂STATUS.json。持久后续进程按PIPELINE_LAUNCH.json记录的代码，在训练审计通过后自动完成统一验证与独立CPU复算；报告写evaluation/REPORT_zh.md，补充脚部证据写evaluation/CPU_RECHECK_all.json。不自动替换主线或启动exo。

已确认control有真实有限更新，学习率符合原计划；原365份源码、父checkpoint和主训练保持不变。父模型38例已完成并独立CPU复算通过。两臂训练及最终效果结论仍在等待。
