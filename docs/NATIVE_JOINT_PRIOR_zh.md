# 三数据集 EGO/EXO 无场景运动先验适配

用户于2026-10-27明确要求现在启动三数据集、EGO/EXO共同学习并稍微解冻主干。本阶段属于运动先验适配；后续场景分支另行实验。

当前正式入口改为 batch32，吞吐实测与局限见 [batch 测试](NATIVE_BATCH_THROUGHPUT_zh.md)。旧 batch8 任务在 step0 EXO 验证遇到多个出画区间的索引越界，完成更新数为0；失败记录及权重保留在 `runs/scene_joint128_noscene`。修正会逐段处理缺帧，只使用真实可用锚点，已在 CPU/CUDA 复现旧错误并通过完整 DDIM 验收。AMASS scratch 未改动。新的 batch32 必须再次通过自身完整 CPU/CUDA 验收才开始参数更新。

- 初始化：EGO-only AMASS 55k共享主干，加EXO36k＋边界6k＋追加15k选中权重。追加20k的last没有替代best。
- 历史AMASS遮挡对照未通过结论保留：100.341 mm相对99.112 mm。这次明确授权的新联合适配不把旧的`ready_for_three_corpus_stage=false`改为true，不表示旧实验全面胜出。
- 配置：`configs/scene_joint128_noscene_batch32.yaml`；代码：`scenemi.native_joint_batched` 复用 `native_joint_training` 的优化器、损失和选模器，`native_batch` 提供可配置的实际前向批次，`geometry/temporal.py` 处理多个独立缺帧区间。保留原边界编码器，只有一个共享扩散U-Net，EXO差分的两次计算均参与主干梯度。
- 数据：严格使用`configs/scene_data_current.json`及已核验清单。EgoBody、TRUMANS、GRAIL等权，来源内family等权。TRUMANS EXO读取移动相机；GRAIL沿用原相机。排除RICH和独立test。
- 用户补充的EgoBody视频前置模型演示应选择独立测试录制；训练仍使用EgoBody的train划分，演示视频不回流到训练。真实前置相机估计的尺度、坐标、设备外参需按已有相机契约接入，不能把数据集轨迹指标当作真实视频效果。
- 本阶段训练不混入AMASS回放。AMASS仅用于下面的完整EGO验证；如遗忘超过容限，应单独记录并设计回放对照，不能无记录地改变这条已启动协议。
- 每次更新交替抽取EGO/EXO各16个样本，分成两个同模式前向批次，各占0.5损失权重。128帧、30Hz。训练保持已有全合法起点的有放回随机抽样；没有擅自改为64步长。
- EGO仅相机及已知设备配置；不输入个人体形/rest。使用已有原生SMPL/SMPL-X/G1形状解码设计，新形状头从零学习，体形标签只用于监督。EXO使用声明的合成原生前置估计、噪声、可见性和8–32帧缺失；不冒充真实上游估计。
- 无场景：不读取/输入场景点云、occupancy或补齐椅子条件。EGO加载器仅复用已筛选动作及相机。
- 部分解冻：U-Net中间层、解码端、输出端以及camera/body/shape观测编码器，学习率1e-5；原生形状头5e-5；EXO边界和注入层3e-5。U-Net下采样端、场景模块保持冻结。
- 6k更新，总采样量保持192,000窗口；原子last每125步；warmup为75步，学习率不放大。常规完整验证0、1500、3000、4500、6000步；额外预先声明500步完整验证。每次包含3471个EGO窗口、每条件3540个EXO窗口及全部2222个AMASS EGO窗口。合法区间步长128并补尾窗，覆盖全部合格验证序列/帧，不穷举所有起点。
- 选模：step0完整测量作为迁移基线；仅EGO step0在模型所有张量逐值一致、EGO路径不变、验证窗口/推理参数一致的证明后，复用旧任务已完成的评测。EXO step0使用修正后的多个区间处理重新测量，AMASS EGO也完整测量。EGO/EXO各0.5的相对误差；三来源内family等权。EGO各来源相对起点不退化，EXO每来源的完整/可见/root误差相对输入最多+1mm，identity逐值保持。AMASS EGO相对本阶段step0最多+5mm，为预先声明的遗忘容限，并非宣称完全不退化。所有分项都记录，不用综合提升掩盖单项退化。保留合格best和last；没有合格改进时如实报告。
- 启动必须CPU/CUDA真实三来源批次、梯度、精确恢复、标签隔离、完整DDIM验收通过；通过原准入/cgroup封装，5.5GiB PyTorch分配上限、10GiB RAM、两CPU，不干预另一个实验或其他用户。

进度：`runs/scene_joint128_noscene_batch32/STARTUP_STATUS.json`、准备完成后的 `training/STATUS.json`；完整日志：`training/service.log`。验证进度：`selection/step_*/ego/STATUS.json`、`exo/STATUS.json`及`amass_ego/`；选中权重：`selection/BEST.json`。初始完整验证阶段不计为梯度更新。batch增加通过吞吐验收，不表示已经证明模型性能相同。

入口示例（需通过资源准入）：

```bash
python -m scenemi.native_joint_batched run --device cuda --memory-gib 10 --cpu-count 2 --allocator-gib 5.5 -- python -m scenemi.native_joint_batched accept-and-train --config configs/scene_joint128_noscene_batch32.yaml
# 已有last时显式恢复，不重复执行全新训练：
python -m scenemi.native_joint_batched run --device cuda --memory-gib 10 --cpu-count 2 --allocator-gib 5.5 -- python -m scenemi.native_joint_batched train --config configs/scene_joint128_noscene_batch32.yaml --resume
```
