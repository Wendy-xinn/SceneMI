# EGO 原生体形与场景分支

本分支从已完成的 AMASS EGO/EXO 混训 55k 初始化；它与从零训练的 AMASS EGO-only 对照使用独立目录。训练、选模验证均仅包含 EgoBody、TRUMANS、GRAIL，排除 RICH；当前只训练 EGO。128 帧、30 Hz，动作 denoiser 保留原尺寸。配置见 [scene_ego128.yaml](../configs/scene_ego128.yaml)，实时状态位于 `runs/scene_ego128/`。只有真实 CPU/CUDA batch 8、输入隔离和精确恢复全部通过后才允许正式开训。

2026-10-23 的首版真实验收通过后，完整零步场景基线暴露了内参输入的单位/初始化问题。首版在 158 步停止并退役，其配置、源码包、身份和逐例基线保留在本地 `runs/scene_ego128_rejected_intrinsics/`；不把这些步数算入修正版。修正版使用独立 v2 contract，重新从同一个 AMASS 55k 初始化并重跑验收。两套 AMASS 任务的源封印未改变。证据摘要见 [验收记录](validation/native_scene_and_full_selection.json)，实时状态以本地 JSON 为准。

AMASS 从未使用图像内参，相关输入列保留着未训练的随机权重。场景数据输入必须将像素内参统一除以 256，并将未训练的内参/标定不确定性输入列置零初始化；此前未训练的 SMPL-X 格式列从已训练的 SMPL-family 列初始化。它们后续均可学习。真实 AMASS 的原先有效输入保持逐位等价。

同一权重、seed 和三个固定验证窗口的配对诊断中，原始像素内参约 `2007 mm`，仅缩放约 `661 mm`，完整初始化修复约 `655 mm`。这是小规模接口诊断；还不是全验证集的微调结果，更不能声称已经消除了跨数据集的差距。不会根据这三个窗口给出最终性能结论。

修正版已重新通过 CPU/CUDA batch 8、逐位恢复及 23 项回归检查，并完成固定 160 窗口的零步基线后开始正式更新。零步各数据集家族平均 MPJPE：EgoBody 377 mm、TRUMANS 636 mm、GRAIL 1187 mm；三数据集等权分数约 733 mm。GRAIL 的跨体形/动作分布差距仍大，需要适配后的验证证明改善。接口修复不等于这些误差已经解决。

## 原生体形

SMPL-H 16 维形状空间无法准确表达 TRUMANS/GRAIL 的原生 rest。新 [native_shape.py](../scenemi/geometry/native_shape.py) 保留独立 SMPL、SMPL-X 系数解码器，并为 GRAIL 使用官方已缩放的 G1 模板；不把 SMPL-H 系数直接复制或截断成 10 维。所有格式仍共享 AMASS 中性 pelvis 参考，换参考时同步修正 root 通道，保持世界关节不动。

输出格式是已知配置。个人系数、SMPL/SMPL-X 的 neutral/female/male 模板由允许的相机观测预测，真实 beta、模板类型和 rest 只进入监督。模板采用 logits 的 argmax，另有分类监督；不把真实性别作为输入。G1 是固定公开输出模板，系数为零、缩放已烘焙，不能重复乘 scale，也不能将其称为未知人体形状预测。

每类抽查 12 条真实训练源，共 48 条，原生 rest 最大关节差约 `2.38e-7 m`。这证明解码公式与源数据一致，不是训练后性能证明。虚拟相机继续采用双眼中点前方 2 cm；静止模板眼部偏置与旧导出从动作估计的固定偏置存在至多约 4.9 mm 的抽查差异。EgoBody PV 使用 TRAIN-only 共享外参，并传入其位置/旋转不确定性；不使用个人真实 head-to-camera 标定。

## 场景与来源

[scene_dataset.py](../scenemi/data/scene_dataset.py) 先等概率选三个数据集，再选物体/录制家族、录制、合格窗口，避免 GRAIL 重叠窗口数量直接压倒其他来源。原始动作和场景文件保持不变。源数组使用已有 SHA256 验证读取器和共享缓存。

输入为窗口开始前及开始帧已观测的静态表面历史，以及各帧实际 first-hit 场景点。owner=100 的目标人体剔除；看不到的几何不当成已知空闲空间。不把整段录制的未来静态并集作为窗口起点的历史。

GRAIL 地形仅取已筛选 EGO 支撑窗口，另接入选定 400 条坐下录制、149 个椅子家族的完整椅子合成增强。其完整椅子是假设此前可见的合成条件，不能作为真实可见性证据。每次读取仍按当前窗口开始帧重算真实历史，不把最早原型窗口的实际观测错误复用于所有窗口。验证使用原始观测，不补齐椅子、不按训练质量筛选挑选验证样本。

[scene_model.py](../scenemi/scene_model.py) 先产生 scene-free proposal，再查询全局静态 patch 和当前帧表面，交由共享 denoiser 细化。新场景输出投影初始为零。旧皮肤局部反馈需要真实个体皮肤资产，当前关闭；不能宣称已支持预测皮肤上的碰撞反馈。动作、相机、速度和足部运动监督保留；当前没有新增真实碰撞穿透损失。

应区分“预测人体附近的局部几何条件”和“接触门控的 root/rotation 显式反馈”。旧 [反馈消融](../experiments/offline_camera_retrain_v1/runs/native30_ego_scene_feedback_review/AUDIT_FINDINGS_zh.md) 在保留局部条件的情况下关闭显式反馈，没有显示稳定退化；仅 12 个独立录制窗口，不支持断言新分支无需局部条件，也不支持直接恢复旧反馈就会提高性能。旧面板中的 RICH 只作历史证据，当前训练及选模仍排除它。建议先验证由预测体形生成的局部皮肤条件，再单独消融显式反馈，并联合检查真实可见表面的接触、滑动、穿透和时间连续性。本 v2 分支是缺少这两条局部路径的对照，不能宣称与旧场景架构性能等价。

## 验证与运行

本分支按固定 seed 777、每数据集至多 32 个验证家族、每家族首尾两个窗口选模；按家族先平均，再对三个数据集等权平均全局 MPJPE。固定 DDIM 20 步，不用 test 或合成完整椅子选模型。零步基线与后续候选使用同一清单；同时保留每窗口和每数据集的根、头、PA、速度、jerk 指标。AMASS EGO-only 的选模另用完整 337 条 val、2,222 窗口，见 [该协议](AMASS_EGO_ONLY_zh.md)。

```bash
# 本机路径放入被 Git 忽略的 scene_paths.local.yaml。
cp configs/scene_paths.example.yaml configs/scene_paths.local.yaml

python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 12 --cpu-count 2 -- \
  python -u -m scenemi.scene_training preflight --device cpu

python -m scenemi.parallel_resources run --device cuda \
  --memory-gib 12 --cpu-count 2 --allocator-gib 8 -- \
  python -u -m scenemi.scene_training preflight --device cuda

python -m scenemi.parallel_resources run --device cuda \
  --memory-gib 12 --cpu-count 2 --allocator-gib 8 -- \
  python -u -m scenemi.scene_training train --device cuda
```

新训练保留一个原子 `last.pt` 和一个验证最佳 `selection/best.pt`，配置、源文件、资产及验证身份写入 contract。检查通过只说明可正确执行；性能是否提高必须由验证指标和可视化判断，不能保证微调绝不下降。

## 下一轮局部反馈

用户要求下一轮启用局部反馈后，另建 [预测皮肤局部反馈分支](SCENE_LOCAL_zh.md)，配置为 `configs/scene_ego128_local.yaml`，输出为 `runs/scene_ego128_local`。用户已接受 GRAIL 可视化，它通过更新后的 CPU/CUDA 真实 batch 8、输入隔离、逐位恢复、原生皮肤计算和受限反馈验收后，已从 AMASS EGO-only 选定 50k 正式开训。本文描述的全局场景 v2 对照与其源码封印保持不变，不能把新分支验收结果说成旧运行已启用局部反馈。

## 30k 完成与场景输入对照

本 v2 对照于北京时间 2026-10-23 17:21 完成 30,000 次更新。原有固定 160 个验证窗口、80 个家族的选模结果仍选中 25k；最终 30k 没有更好。以下数值为 mm，先按家族、再按数据集等权，未使用 test。

| 候选 | EgoBody MPJPE | TRUMANS MPJPE | GRAIL MPJPE | 三数据集等权 MPJPE |
| --- | ---: | ---: | ---: | ---: |
| 20k | 189.48 | 114.05 | 95.18 | 132.90 |
| **25k（已选）** | **178.57** | **114.50** | 58.16 | **117.08** |
| 30k | 182.63 | 122.15 | **55.59** | 120.13 |

最后 5k 总分回退约 2.6%，但 GRAIL 继续改善；这说明不同来源的最佳点不同，不能仅以训练 loss 推断收敛或继续延长。现有 160 窗口是原选模子集，不是全验证或独立测试。25k/30k 权重均保留以支持比较与恢复。

场景依赖诊断固定已选 25k、原 160 窗口、DDIM 20 和 seed 777，在同一权重下关闭全部全局/patch/逐帧场景路径；相机、内参、预测体形解码保持不变。由新入口运行，封存训练源文件不改：

```bash
python -m scenemi.runtime run --device cuda \
  --memory-gib 6 --cpu-count 2 --allocator-gib 4 -- \
  python -u -m scenemi.benchmark scene --policy configs/scene_input_benchmark.json
```

结果保存在本地 `runs/scene_ego128/evaluation/scene_input/`，包含固定协议、场景开启结果的三个复现检查、逐窗关闭场景预测、指标与报告。此诊断只测推理时的场景依赖；无场景模型还需从同一 AMASS 父权重、以同样的原生体形适配、数据和更新预算训练，并各自按固定验证策略选模。不能拿未做场景域适配的 AMASS 权重直接归因纯场景收益，也不能把旧混训父权重与新 EGO-only 父权重的差异算作局部反馈收益。

本次 160 窗口诊断已完成，三个场景开启窗口的复现检查通过；全部关闭场景数组已逐个校验哈希，并用独立 NumPy 欧氏距离和 SVD Procrustes 复算 MPJPE、标准 PA-MPJPE、root，最大指标差小于 `2e-13 mm`。紧凑记录见 [配对评测证据](validation/scene_input_comparison.json)。

| 数据集 | 无场景 MPJPE | 有场景 MPJPE | 无场景 PA | 有场景 PA | 无场景 root | 有场景 root |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EgoBody | 264.07 | 178.57 | 101.41 | 87.50 | 233.37 | 129.62 |
| TRUMANS | 140.05 | 114.50 | 72.03 | 67.05 | 117.08 | 93.87 |
| GRAIL | 101.16 | 58.16 | 25.48 | 25.36 | 93.57 | 48.30 |
| 三数据集等权 | 168.43 | 117.08 | 66.31 | 59.97 | 148.01 | 90.59 |

所有误差单位为 mm。总体 MPJPE 相对下降 30.49%，说明这套已训练权重确实使用场景路径。收益主要体现在全局位置，GRAIL 的标准 PA 几乎不变；不能仅凭此表声称改善真实足底接触、碰撞或坐下/蹲下辨别。这些问题仍需接触评估与固定输入的场景敏感性检查。
