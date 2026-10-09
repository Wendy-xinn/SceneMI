# 修复后55k长训

> **已完成**：55000步及12913窗口全量验证全部结束，结论见 [RESULTS_zh.md](RESULTS_zh.md)。

启动时间：2026-10-07T22:42:07+08:00。用户已授权启动。从头训练，不加载旧55k或5k权重。

配置继承本次通过的5k：原生SMPL/SMPL-X、22区域接触头、完整20Hz场景、temporal_valid_v1动态窗口、时长平方根采样（TRUMANS/EgoBody/RICH约62.06%/26.25%/11.69%），batch4、latent256、float32、seed2026、lr1e-4、coordination_v1、接触权重0.1。总步数55000，因此学习率衰减按55k进度运行；每5000步保存及固定DDIM20评估。

后台监督进程通过独立session运行，IDE或本次会话结束不会中断。实际配置在config.json生成后查看；精确启动参数见launch_command.json。

明天查看：

- job_status.json：training / paired_history_evaluation / full_validation / best_checkpoint_sampling / completed；失败显示failed及日志位置。
- training_log.jsonl：最新训练步数、loss、验证指标和阶段采样退出码。training_stdout.log包含启动审核及完整输出。
- evaluations/step_*/metrics.json：每5k固定有/无场景对照，含MPJPE、WA-MPJPE、RICH接触指标。
- last.pt：最后保存的可续训权重；best.pt：固定去噪验证MPJPE最优权重，并非全量生成评估最优。
- history_reset_verification.json：55k完成后同噪声的有场景/无场景/历史reset对照。
- full_validation/metrics.json：最后权重三域完整合法验证窗口的配对生成评估；status.json报告覆盖进度。
- best_sampling.json：best权重固定样本配对评估。

训练正常结束后按上述顺序自动评估；任何阶段失败都会记录状态并停止后续阶段。全量评估采用连续头部条件、DDIM20，128帧合法起点全覆盖，仅不足128帧的序列改用64帧；该范围不代表所有窗口长度。生成动作保存于full_validation/motions，供后续mesh/动态碰撞与接触几何验收。自动全量评估当前只含运动误差及足部代理指标，不把它当成真实动态碰撞或mesh接触认证。

训练粗估约5小时，启动与训练后评估另计。当前场景收益仍待正式结果确认。原可视化8780保留。
