2026-10-18更新：主线55k已结束，现存`best.pt`是41k，仍只表示单个固定RICH窗口上的临时候选。用户指出这一选择面板过小；独立扩展验证已于11:33启动，状态在[runs/native30_checkpoint_review_v1/STATUS.json](../runs/native30_checkpoint_review_v1/STATUS.json)，比较现存41k与最终55k。固定面板为RICH全部206个合法VAL窗口/27个录制、TRUMANS50窗/33个录制、EgoBody50窗/16个录制，共306窗、1224次预测；各模型共用输入和种子，场景及原物理修正开启。主判断使用RICH录制宏平均World MPJPE的clean/noisy均值，同时检查脚速及双足运动误差；另报窗口均值、计数加权及三数据集等权宏平均。17项CPU回归、真实权重严格加载、真实首窗输入预检已通过。任务逐次落盘，可中断续跑，完成后自动独立CPU复算。在完整任务和复算通过前，不把41k称为全验证集最优，也不覆盖原best/last。过去已覆盖的历史权重不能从曲线恢复，本次只比较41k与55k，不等于追溯比较过全部训练步数。EXO内部适配实验另按预注册的第1000步终点评估，没有用12窗挑本轮best。

以下为原守护器的历史选择协议：

当前主线保留两个不同用途的文件：`training/last.pt` 含优化器及随机状态，可继续训练；`training/best.pt` 只保存模型，用于评估。后者由本目录的 `checkpoint_guardian.py` 独立生成，训练代码、配置和当前训练进程均不改动。

目前 best 的选择口径是临时监控口径：同一个预先固定的 RICH 验证窗口、scene ON、head-only ego，干净和扰动输入的原始世界坐标 MPJPE 算术平均，越低越好。轮换窗口不进入选择。`best.json` 中明确记录 `selection_status=provisional_fixed_rich_window`。这不是全 RICH 验证集最优，也不是对外论文比较结果；完整验证结果可用于之后确定更可靠的候选。

守护器每 2 秒读取完成的验证记录，在训练器原子发布 `last.pt` 后，用硬链接固定其 inode，并核对 checkpoint 的真实 step 与验证记录完全一致。保存时删除副本中的优化器，逐项比较源模型与保存模型的 tensor，并同时记录 tensor digest、完整源文件 SHA 和副本 SHA。只原子覆盖一个 `best.pt`，不堆积历史权重。保存失败不会修改训练器的 `last.pt`。

历史固定窗口的最低均值出现在 12,000 步，为 142.6163 mm；该步模型已被过去的 `last.pt` 覆盖，无法从指标恢复。因此 best 只表示守护器实际捕获的候选中的最低值，不能宣称保留了过去所有步数中的最优模型。现有 12,600 和 15,100 步独立评估快照继续保留，不能把它们和别的步数的指标配对。

初始 best 已从现存的 15,100 步快照保存：全 RICH 评估的 case 29 恰好是同一固定窗口，干净/扰动 MPJPE 为 143.1271/157.4527 mm，均值 150.2899 mm。保存前核对输入轨迹、观测置信度、GT、rest 和初始化 reference 与训练器固定窗口逐数组完全相等，scene、seed、帧数、采样步数和 head-only 条件一致，并在 CPU 重算这两条预测的 MPJPE。只使用 15,100 步自身的预测指标；没有给它配上 15,000 步权重的指标。独立验收的 317 个模型 tensor 全部与冻结快照相等，证据位于 `ops/checkpoint_guardian_oct16/ACCEPTANCE.json`。后续从可精确匹配验证的主线 checkpoint 中选择更低均值，继续覆盖同一个 best。

启动命令：

```bash
/home/wenxin/miniconda3/envs/scenemi/bin/python -u experiments/offline_camera_retrain_v1/ops/checkpoint_guardian.py
```

运行信息在 `ops/checkpoint_guardian_oct16/LAUNCH.json` 和主线 `training/BEST_GUARDIAN_STATUS.json`。保护的封存依赖来自主线 `prepared/effective_config.json` 的 359 个源文件和 `continuous_training_oct16/PROTOCOL.json` 的 6 个执行依赖，保存前后均校验。运行守护器需要本工具、主线验证记录、上述两个清单以及 `last.pt`；清理实验目录时应保留这些活动依赖。

安全测试：

```bash
/home/wenxin/miniconda3/envs/scenemi/bin/python -m unittest discover -s experiments/offline_camera_retrain_v1/ops -p test_checkpoint_guardian.py -v
```
