# RICH 全合法窗口验证

已按用户要求启动独立全量验证，主线训练继续向 55,000 步推进。冻结模型为 **15,100 步**，全部窗口使用这同一份权重；原优化器、训练 RNG、配置和源码保持不变。

评估集合是当前工程官方 `validation/ego` 索引中的全部 RICH 条目：**206 个窗口，27 条有合法窗口的序列，clean/noisy 共 412 次采样**。窗口为 192 帧、30 Hz，起点步长 64 帧，相邻窗口重叠 2/3。RICH 原始 VAL 的 28 条序列共 17,661 帧已经接入，但 `ParkingLot1_002_stretching2` 只有 165 帧，没有合法的 192 帧窗口；其余序列有 856 个尾帧未被该网格覆盖。本次覆盖 16,640 个唯一源帧，不通过填充或补尾改变协议。

评估沿用主线的 ego/head-only、可见场景 ON、DDIM20、软头部约束及原物理修正设置，不增加 tracker。按录制和起点排序，身份 seed 与原固定/扩展验证一致，便于做同窗对照。窗口均值、有效样本计数加权和录制等权平均分别报告，避免把重叠窗或较长录制当成更多独立证据。

- [进度与指标页面](index.html)：本地服务 `http://localhost:8795/`。
- [状态](STATUS.json)、[运行协议](PROTOCOL.json)、[清单及源帧覆盖](PLAN.json)。
- [冻结权重证据](FROZEN_CHECKPOINT.json)：源模型和保存后的 compact 模型逐张量内容摘要相同。
- [结果汇总](SUMMARY.json)、[完整报告](REPORT_zh.md)：未收齐且完成最终检查前均标为进行中，不把部分序列当成全 RICH 结果。
- [训练监督趋势](LOSS_TREND_zh.md)：最近完整千步段的加权总项和各原始分项，训练下降与验证改善分开判断。
- [独立 CPU 验收说明](VERIFY_README_zh.md)：首批复算通过；后台等待任务在评估完成后自动复算全部 412 份动作，届时生成 `VERIFY_COMPLETE.json`。
- [与旧模型的同窗对照](MATCHED_12600_VS_CURRENT.md)：只比较旧 12,600 步评估中的相同 10 个 RICH 窗口，逐数组核对 GT/rest/头部输入/reference 相等；全量任务结束后自动生成最终比较，进行中不据此给总体进退结论。

新增标准动作指标复用项目的 `standard_motion_metrics`，显式指定 **fps=30**：全局 22 关节 MPJPE、逐帧减 pelvis 的 MPJPE、逐帧 proper similarity Procrustes 对齐的 PA-MPJPE、全局速度误差（mm/s）和加速度误差（mm/s²）。它们与原始方向、脚步和漂移指标同时保留。低速脚掩码来自 GT 足关节速度 <0.2 m/s，只是运动学代理，不是真实接触标签、论文 Foot sliding 或场景穿透率。

用户指定的比较方法是 **EgoGVAE、EgoEgo、EgoAllo**。EgoGVAE 论文 Table 2 的三者对比采用 AMASS 训练、RICH **TEST** 评估；当前模型混合数据训练包含 RICH TRAIN，本轮是 RICH **VAL**，并有可见场景等额外条件。因此本轮用来诊断主线、追踪同窗进展；正式跨方法比较需要统一 split、训练数据、头部设备坐标、场景/体型输入、帧率、窗口、关节和指标代码。依据：[EgoGVAE §4.2 / Table 2](https://arxiv.org/html/2607.27755v1)。详细核对见 `PROTOCOL_REVIEW_zh.md`。

实现是可断点续跑的独立进程，GPU 分配器上限为总显存的 25%，CPU 线程为 2、nice=10，派生 surface 缓存写入本目录；共享 static map 只读。训练与评估共用一张 GPU，速度会互相影响。全部保存动作及 compact evaluation-only checkpoint 保留本地，Git 只备份代码与紧凑记录。
