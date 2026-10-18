冻结模型配对评估：原 41k 与 balanced-v2 EXO 适配版，分别使用原参考构造和修复后的参考构造。没有优化器更新，没有使用 TEST，也没有自动替换正式 checkpoint。

主评估从已有 identity-only cohort 中按原顺序取三个数据集各 4 个新录制，每个录制一个 192 帧窗口；另保留用户指出的 EgoBody 抽搐例作独立回归。该回归例不参与主评估均值。每窗口评估 clean/noisy、两种参考、两个模型，共 104 次采样。

修复只对 EXO 的无旋转证据帧进行局部 SO(3) 插值，保留有效证据和原置信度。它使用整个离线窗口，EGO 路径保持原参考构造。权重在两种参考下均固定，因此结果同时反映现有适配权重面对新参考分布时的表现。

入口统一放在可复用的 `ops` 模块中：

```bash
python -m experiments.offline_camera_retrain_v1.ops.reference_review prepare
python -m experiments.offline_camera_retrain_v1.ops.reference_review evaluate --resume
python -m experiments.offline_camera_retrain_v1.ops.reference_review recheck
python -m experiments.offline_camera_retrain_v1.ops.reference_review_audit inputs
python -m experiments.offline_camera_retrain_v1.ops.reference_review_audit results
python -m experiments.offline_camera_retrain_v1.ops.reference_review_audit summarize
python -m experiments.offline_camera_retrain_v1.ops.reference_review_audit plot
```

已有 `PLAN.json` 时不能重新 prepare。原计划与其显式 source seals 保持不变；所有输出逐项落盘，可恢复。GPU 和 CPU 检查分进程执行。运行环境使用 `/home/wenxin/miniconda3/envs/scenemi/bin/python`。

最终指标看 `PAIRED_COMPARISON_zh.md` 和同名 JSON；逐窗口数据看 `rows.jsonl`。`CPU_RECHECK.json`、`INPUT_POLICY_AUDIT.json`、`PAIRED_RESULT_AUDIT.json` 分别记录指标复算、唯一输入变因及配对/历史复现核查。未完成前的 `SUMMARY.json` 仅是进度快照。

指标中的 hidden/visible 对应位置观测未提供/已提供，不是所有物理遮挡的等价标签。主报告使用等录制、等数据集均值；按有效关节数量加权结果另存。腕部转角 P95 和大跳变计数只用于诊断稳定性，不能替代姿态精度；低转角也可能意味着动作不足。原始观测来自 GT 模拟，参考到模型输出之间还包含 IK 和观测锚定，不能将全部收益归因于纯网络。

Viser 继续复用 `ops.motion_viewer` 与 `native30_motion_review`，没有为此次对照另写展示脚本。大数组和 mesh 缓存留在本地；代码、逐例指标和简要审计记录备份到已授权的 user-backup。
