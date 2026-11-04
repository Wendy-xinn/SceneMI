# AMASS 128 帧评测协议

当前评测使用 AMASS 阶段 A 的最终 55k 检查点，不改权重或采样参数，不用测试集选择模型。开发监控仅有 4 段 val 动作，不能代替这里的 test 评测。

## 输入、样本与复现

- 测试子库为 EgoAllo 划分的 `Transitions_mocap`、`SSM_synced`。本地有 114 个满足 128 帧条件的原始动作家族；公开脚本另需前一帧，一段恰好 128 帧的动作因此排除，实际使用 113 家族、226 个原始/镜像版本。原始与镜像版本仍视为同一家族。
- 128 帧、30 Hz、EGO 连续虚拟相机；场景关闭，无手/其他身体关节或真实形状输入。预测骨架解码预测，真实骨架只解码评测标签。
- 固定 20 步 DDIM，seed 777，batch 1，每窗口生成一次，无 best-of-K、头部对齐后处理或贴地修正。头部误差是生成结果，不把输入位置复制进结果。
- 复现 EgoAllo 公开 `5_eval_body_metrics.py` 的确定性窗口规则：按长度 129 过滤，长度总和除以 129 得到样本数，轮转文件列表，对每文件轮转起点。0b 已去首帧，再取 129 帧中的后 128 帧评分。相邻时间用于构造相对输入，各方法可以采用自身输入编码。
- 固定列表改为 NPZ 相对路径字典序。上游 0b 文件列表受 glob 和写入线程顺序影响，论文使用的随机窗口也没有随代码发布；因此这里是可复现的 **344 窗口评测**，不是论文原窗口的逐项重放，也不是遍历所有可能窗口/所有时间帧。其中 SSM_synced 28 窗口、Transitions_mocap 316 窗口。
- 窗口均值接近公开脚本的汇总方式；另报家族宏平均与家族标准误，避免把镜像及重复窗口当作独立样本。它们也不是多随机种子训练误差条。

每次结果目录保存 `PROTOCOL.json`（窗口清单、检查点/数据/代码哈希、限制）、`metrics.jsonl`（逐窗口身份和指标）、`RESULT.json`、`STATUS.json`。最终权重只保留原来的可恢复 `training/last.pt`。

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources check
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cuda --memory-gib 8 --cpu-count 4 -- \
  python -m scenemi.evaluation --paths configs/paths.local.yaml \
  --output runs/amass128/evaluation/ego128
```

此阶段评测沿用原独占准入。评测实现暂用稳定独立模块入口，避免修改仍需恢复训练的根 CLI 源码封印；没有新增按 run 命名的脚本。输出目录必须为空；`--limit` 只生成明确标记的诊断结果。

## 指标约定与公开实现差异

`scenemi/metrics.py` 接收米制、含 pelvis 的 22 个 SMPL-H 身体关节。MPJPE 不作每帧 root/头部/地面平移；它保留全局轨迹误差。窗口的输入、预测和标签应用同一刚体坐标变换，这不会改变 MPJPE。`root_error_mm`、`head_error_mm` 分别是 pelvis 与 head 的位置误差。`pa_mpjpe_mm` 对每帧预测做尺度、旋转和平移的最小二乘对齐，再与标签比较，使用 proper rotation，不允许镜像变换。

固定版本 EgoAllo 与 EgoGVAE 的 `compute_mpjpe(..., per_frame_procrustes_align=True)` 存在同一实现差异：调用 `procrustes_align(points_y=prediction, points_x=label)` 得到对齐后的 **label**，随后与原 label 比较。这和论文文字中的标准 PA-MPJPE 不等价。平移 10 cm 的同一骨架，标准 PA 应为 0，公开实现会得到 100 mm。本仓库通过解析上游函数并对 FP32 数值做独立测试核验；不会修改第三方仓库，也不能据此推断论文私有评测一定用了同一实现。

因此同时保存标准 `pa_mpjpe_mm` 与明确命名的 `released_pa_compat_mm`，后者只用于诊断源码行为，不能冒充标准 PA 结果。两者不择优，也不把任一个直接当成已对齐论文 PA 的证据。

`egoallo_gnd_any_contact` 复现 EgoAllo 的窗口内任意一帧任意足关节低于阈值的 0/1 指示，阈值踝 8 cm、脚尖 4 cm；它不是逐帧接触准确率。`ground_lowest22_mm` 是每帧双方最低关节高度差绝对值均值。EgoGVAE 的 Ground 文字使用最低点误差，与 EgoAllo 的 GND 指示不同。

额外记录 root-relative MPJPE、30 Hz MPJVE、jerk（m/s³）及 GT jerk 作为诊断。EgoGVAE 发布版 `model/metric.py` 引用了 `compute_mpjve`、`compute_jitter`、`ground_metrics_lowest`、`compute_foot_skate_gt_contact_batched`，但这些函数在本地固定版本的目标模块中没有定义；因此不声称复现了其 Ground/Jitter/Foot-sliding 数值口径，也不把本仓库诊断 jerk 与其表中 Jitter 混列。

## 论文参考与公平性边界

以下均为 **EgoGVAE 论文表 1 的报告值**，不是本机重跑的结果，输入长度均为 128。MPJPE/PA/head 单位为 mm。

| 方法 | MPJPE | 论文 PA-MPJPE | head 误差 |
|---|---:|---:|---:|
| EgoEgo | 167.4 | 145.8 | 54.9 |
| EgoAllo | 119.7 | 101.1 | 6.2 |
| EgoGVAE | 106.7 | 89.9 | 5.5 |

来源：[EgoGVAE §4.2–4.3 与表 1](https://arxiv.org/html/2607.27755v1#S4.SS2)。EgoAllo/EgoEgo 的上述位置指标也见 [EgoAllo 表 2](https://arxiv.org/html/2410.03665v2#S4.T2)。[EgoAllo 公开评测脚本](https://github.com/brentyi/egoallo/blob/0c20ae8c8d48be4ff682803ea523bcb00a6a708d/5_eval_body_metrics.py) 自述不是论文原脚本，其公开权重 128 帧 MPJPE 示例为 118.340，和论文 119.7 略有不同。

直接比较仍有以下未对齐项：

1. 旧统一模型为 AMASS **train-only、EGO/EXO 各约 50%**，新 EGO-only 对照也是 train-only。EgoAllo 下载的公开权重附带配置明确 train+val，EgoGVAE 公开代码选 train-only；这些证据不等于确认论文所有私有实验的流程。实际权重及选模证据见 [检查点审计](BASELINE_CHECKPOINTS_zh.md)。
2. 本模型相机使用两个眼睛顶点均值加光轴前方 2 cm；EgoAllo/EgoGVAE 使用四个瞳孔顶点 CPF，无 clearance。固定版本 EgoAllo 的 `EgoTrainingData.load_from_npz` 从 shaped rest 计算固定 head-to-CPF，再随头部刚体变换；它另有 posed mesh 的 CPF 函数，但本次核对的 AMASS 0b 不调用该函数。两点与四点的差异仍可能依赖体形，不能只减 2 cm 就宣称完全相同。保留训练时相机定义；后续应给所有方法同一传感器轨迹及相同已知标定，不能给基线额外真实 betas。
3. 本地本次尚未运行三者的官方权重；同一窗口、同一传感器条件下的完整基线重跑尚未完成。EgoEgo 需要比较其 head-to-body 阶段，不把视频到头部估计误差额外引入。
4. PA、Ground、Jitter 与 Foot sliding 的公开源码/论文差异需要在正式投稿表格前解决；暂以全球 MPJPE 和 head 误差作量级参考，不能宣布严格同条件排名。

固定源码：EgoAllo `0c20ae8c8d48be4ff682803ea523bcb00a6a708d`；EgoGVAE `1dca529bf1efd6b53bf81da9c769ea3483777ac3`。评测报告应分别标注“本文实测”和“论文引用”，包括上面这些限制。

## 相机约束的历史结论与当前边界

2026-10-22 根据用户提醒复核原始记录。之前场景阶段已经尝试过相机/头部约束，不能把“让生成身体对齐相机”当作尚未试过的修复，也不能凭 head 或全局 MPJPE 下降判断可视化已经改善。

- [10 月 11 日强轨迹约束验收](../experiments/offline_camera_retrain_v1/archive/historical_trials_oct16/hierarchical_constraint_oct11/ASSESSMENT_zh.md)：精确设置头部朝向并平移 root，保留了错误的骨盆转向，出现颈部补偿和脚滑；两个展示例的头相对颈 p95 角度为 106.9° / 141.6°。加入连续转向和支撑求解后，转向改善，但四次生成中三次 GT 支撑期脚滑退化，没有通过腿部联合验收。没有替换当时的原 55k 模型。
- [10 月 16 日完整采样消融](../experiments/offline_camera_retrain_v1/runs/native30_live_monitor_oct16/support_ablation/DECISION_zh.md)和[后续逐阶段诊断](../experiments/offline_camera_retrain_v1/runs/native30_live_monitor_oct16/foot_stage_diagnostics/REPORT_zh.md)：原始生成已经存在较高足速；软整身平移在这些窗口中有收益，直接删除反而更差。因此，历史证据不支持“所有相机约束都有害”或“删掉贴头即可修复脚滑”。这些是旧权重、旧数据窗口上的诊断，不是当前 AMASS 模型的因果消融。
- [当前 AMASS 16 窗诊断](../runs/amass128/diagnostics/ego_alignment/REPORT_zh.md)：仅改变整身 root 刚体变换，所有非 root 局部关节旋转保持，因此与旧的局部锁头方法不同。但完整绑定后 jerk 从 120.04 升至 287.45 m/s³，最低关节高度差从 43.74 升至 54.00 mm；仅平移也使这两项变差。最低关节高度差是相对 GT 的代理，不是真实足底穿透深度。该诊断没有经过逐例可视化验收，不代表动作质量提高。

当前两种校正均只保留为诊断，没有接入默认推理或启动新训练，正式 344 窗测试数字及原 55k 权重保持不变。相机符合程度、身体姿态与接触质量必须一起判断；不能因基线有解析放置，就认定给已训练网络加同类后处理一定有效。

后续若继续验证相机相关改动，应在同一权重、窗口和生成噪声下比较原输出与候选，覆盖纯转头、身体转向、行走、弯腰和坐下，并结合颈部/骨盆转向、支撑期脚滑、可用场景表面的接触及时间连续性逐例回放。现有记录尚未确定剩余误差的主因，也没有证明把约束接进训练就能解决；不能据此直接重训或再叠一个更强的头部约束。
