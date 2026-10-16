# 全 RICH 验证的独立 CPU 验收

`verify_results.py` 只读训练源码/配置、runner、计划、冻结 checkpoint、已提交 rows 和 motion 文件，只在本目录写 `VERIFY*.json` 及自身日志。它不运行扩散模型、不初始化 CUDA、不创建或修复数据缓存、不更新优化器。建议使用项目环境、单线程和低 CPU 优先级：

```bash
nice -n 15 /home/wenxin/miniconda3/envs/scenemi/bin/python verify_results.py --plan-only
nice -n 15 /home/wenxin/miniconda3/envs/scenemi/bin/python verify_results.py
nice -n 15 /home/wenxin/miniconda3/envs/scenemi/bin/python verify_results.py --require-complete
nice -n 15 /home/wenxin/miniconda3/envs/scenemi/bin/python verify_results.py --wait-complete
```

`--plan-only` 从已封存 manifest 与 `validation/ego` 合法起点索引独立重建全部 RICH 集合，逐一核对 `(bundle ordinal, start)`，要求计划无漏窗、无重复、没有其他数据集、训练集或 TEST。索引共有 206 个合法窗口，来自 27 个序列；manifest 的第 28 条 `ParkingLot1_002_stretching2` 只有 165 帧，不能形成 192 帧窗口。程序独立重算每个源序列被覆盖的帧并对照计划；不追加尾窗、不 padding。

普通模式检查读取时已完整提交的 JSONL 行，尾部尚未写完的一行只记录、不修改。所有已提交 motion 都检查 SHA、身份、源元数据和输入指纹、冻结步数、坐标约定、数组 shape 与有限值；另外在 CPU 重算最先提交的三个预测。完成模式要求所有 206 窗 × clean/noisy 共 412 条齐全，并重算全部 412 条的 raw、足部 extra、活动类型与 benchmark 指标。

benchmark 包括世界坐标 MPJPE、去根平移 MPJPE、每帧相似变换对齐 PA-MPJPE、根位置误差、关节速度误差和加速度误差；显式使用 **30 Hz**。计数分别核对 192×22、192、191×22、190×22。数值容差为相对 `1e-6`、绝对 `3e-5`；整数计数、None 和布尔值必须相等。这是同一项目正式指标函数的独立离线重算，用于验证落盘结果与评估记录一致；不等于独立实现或复现其他论文的指标协议。

同时核对原训练 359 个文件和连续训练 6 个文件的封存哈希、原配置和 effective config、plan/runner/checkpoint 哈希。冻结步数从元数据读取，绝不写死为 12,600 或 15,000。冻结时 runner 在完整源 checkpoint 仍被 hard link 固定时计算模型张量 digest，并重新载入 compact 再比较 digest；本脚本独立重算 compact 的 sorted-name/dtype/shape/bytes digest，要求三者相等、所有张量有限且只含模型/配置/step、不含优化器或 RNG。原完整 pinned inode 发布后已解除链接，本脚本不会把持续变化的训练 `last.pt` 冒充原冻结源。

验收结束还核对主线 worker 存活、目标仍为 55k、相对评估开始已推进以及封存代码配置未改动。大体积原始数组不会重新哈希或重建；原始数据的范围检查依据源 metadata 与其中声明的逐数组/输入哈希，保存的 GT 是离线重算指标的参考。

`VERIFY_PLAN.json` 只表示 cohort 验收通过；`VERIFY_PARTIAL.json` 只表示已提交的局部结果通过；`VERIFY_COMPLETE.json` 仅在 runner 和 PROTOCOL 都完成、全部覆盖与全量指标重算通过时产生。`VERIFY.json` 是最近一次通过的结果，必须读取其状态和时间。失败另写 `VERIFY_FAILURE.json`，不会抹掉历史通过记录。

`--wait-complete` 等待阶段只用标准库每 15 秒查看状态，不加载 torch/numpy。检测 STATUS 与 PROTOCOL 都完成后，exec 成一个新的 CPU 验收进程；完整验收仍独立要求 412 条精确覆盖。若 runner 明确失败，等待进程会记录失败并退出。

这里的“全量”指项目固定索引的全部合法 RICH VAL 窗口，包含重叠窗口。192 帧、stride 64 的相邻窗口共享 128 帧；412 条 clean/noisy 预测不是 412 个独立动作样本。应同时参考窗口加权和录制等权均值。22 个共同 SMPL/SMPL-X 关节、头部输入、场景可见性、帧率、划分与对齐约定需要与其他 ego 方法核对后，才能作为同协议横向比较；本验收不宣称它们已经一致。
