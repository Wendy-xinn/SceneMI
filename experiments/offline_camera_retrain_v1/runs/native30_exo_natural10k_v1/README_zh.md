# 独立 EXO 适配层 10k 训练

从已选定的 original41k + 3k EGO control 重新初始化 EXO LoRA；不加载此前 1k 适配层。共享生成器全冻结，EGO 结构性绕过适配层。旧 1k 是保留对照，不作为最终版。

- 10,000 次更新，batch 8，合计 80,000 窗口；rank16 / alpha16，FP32。
- LR 1e-4，warmup200，完整 10k cosine，floor0.2，fresh AdamW。
- 分析式物理优化关闭；原学习式场景路径、反馈及 anchor IK 保留。
- 每个 TRAIN 合法 EXO 窗口完整打乱后无放回采样，再进入下一轮。窗口192帧、stride64，窗口间仍可能有时间重叠。

| 数据集 | TRAIN 合法窗口 | TRAIN 比例 | VAL 合法窗口 | VAL 比例 | 每个100窗面板 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RICH | 394 | 2.572% | 206 | 11.873% | 12 |
| TRUMANS | 11603 | 75.743% | 709 | 40.865% | 41 |
| EgoBody（两角色合计） | 3322 | 21.685% | 820 | 47.262% | 47 |

比例按各自 split 的可用窗口数定义，不能把训练比例原样套到验证集。尽量跨 recording/角色选窗；这仍是开发子集，不能当作完整验证集的无偏估计。

每2k更新验证一次：固定100个 EXO 窗口选 best，另有100个轮换 EXO 窗口只监控泛化；各做 clean/noisy 两种输入。5组轮换共500个窗口互不重复，并与固定100不重叠。额外9个 EGO head-only 保持探针及已知 EgoBody 手腕抖动例子不进入 best 分数。开始训练前先在同一固定面板测零适配基线。

Best 分数：各数据集固定窗口的隐藏关节 MPJPE 均值，按真实 VAL 窗口比例加权，再平均 clean/noisy。每个 checkpoint 复用固定观察、噪声与 DDIM 种子；不使用 TEST 选模型。best_available 仅表示本轮已评估 checkpoint 中固定分数最低者，科学有效性仍要看分数据集、可见关节、时序/脚部指标及后续场景消融。验证自动检查 EGO 输出逐元素一致；完整预测数组保留，供训练后的 CPU 时序检查。

每100步原子保存完整 last；2k/4k/6k/8k/10k 保留完整里程碑；best 指向其中一个完整 checkpoint。均含共享生成器、适配层、AdamW、采样游标、全部 RNG 和来源信息，可独立加载/恢复。保留旧41k/55k/control及有用1k对照。

复用入口（在 SceneMI 根目录，使用 scenemi 环境）：

```bash
python -m experiments.offline_camera_retrain_v1.ops.exo_longrun status --run experiments/offline_camera_retrain_v1/runs/native30_exo_natural10k_v1
```

配置见 config.json；采样全程模拟与断点恢复证据见 SAMPLING_PREFLIGHT.json；CPU 测试见 TESTS.json；真实 GPU 检查见 prepared/input_checks.json；启动身份见 LAUNCH.json / JOB.json；训练状态见 STATUS.json；曲线文件位于 training/tensorboard。结束后独立 CPU 进程严格加载 best，并保存 CPU_MODEL_VERIFY.json；不会自动开始 TEST 或自动宣布论文最终模型。

按历史吞吐估计全流程约13–15小时（含初始基线及5次扩大验证），需要以本轮实际速度更新，不能将纯训练时间当成完成时间。
