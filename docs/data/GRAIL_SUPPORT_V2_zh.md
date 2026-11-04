# GRAIL EGO：脚底支撑复核与椅子观测来源

此版本继承 v1 的穿透、滑动、座面支撑和可见性限制，在此前保留的 10,910 条训练录制上增加脚底支撑与坐下前整椅观测检查。实现位于 `scenemi/data/grail_support.py`、`grail_chair.py`、`grail_filter.py`；配置为 `configs/grail_ego_support_v2.json`。旧版本封存源码和账本不改动，AMASS 训练不受此版本影响。

## 全量筛选结果（2026-10-22）

死机后从 2,712 条完整账本继续，全量 10,910 条已处理，无处理失败。原始数据未删除。

| 类别 | v1 保留序列 | v2 保留序列 | v1 窗口 | v2 窗口 |
|---|---:|---:|---:|---:|
| 路沿 | 1,179 | 804 | 138,014 | 56,492 |
| 斜坡 | 1,386 | 1,031 | 204,530 | 96,890 |
| 楼梯 p1 | 3,853 | 2,503 | 530,857 | 186,945 |
| 楼梯 p2 | 3,798 | 2,673 | 541,609 | 240,544 |
| 坐下（真实完整观测条件） | 694 | 0 | 16,573 | 0 |
| 总计 | **10,910** | **7,011** | **1,431,583** | **580,871** |

窗口起点步长为 1，存在大量重叠，不能视为独立动作数量。坐下的 694 条录制及 16,573 个 v1 合格接触窗口仍单列为合成增强候选，不计入真实完整观测清单。独立审核证据在本地 `runs/grail_ego_quality_v2/full/VALIDATION.json`；审核完成后精简记录保存于 `docs/validation/grail_support_v2.json`。

## 脚与地形

每只脚使用固定 G1 SMPL-X 原生皮肤顶点，选世界高度最低的 15%，向下查询水平或斜面支撑。仅接受 `abs(normal_y)>=0.7` 的三角面，排除靠近台阶立面造成的假接触。底部点中至少 20% 距支撑面不超过 3 cm 才算通过这一几何检查。该比例是顶点覆盖率，不是实际接触面积。

地形动作中，双脚均无上述支撑，或源估计接触概率 >0.8、脚中心速度 <=0.30 m/s 的脚无支撑，连续 8 帧即标记；事件前后各扩展 6 帧，排除所有重叠的 128 帧训练窗口。单脚摆动不因离地直接失败。坐下不要求双脚贴地，沿用 v1 的臀部/座面、脚部深入地面等限制。

已知楼梯 `straight_aged_wood_stairs_with_8_steps_no_handrails_no_turns__0002` 第 82 帧，新测得左右足底 q10 支撑间隙分别为 16.58/20.05 cm，3 cm 内支撑覆盖率均为 0；左脚速度 0.024 m/s、源接触概率 0.994，已触发持续异常筛选。真实样本回归证据在 `runs/grail_ego_quality_v2/pilot/CASE82_REGRESSION.json`。

## 真实观测条件

用户确认的标准是：同一录制坐下前椅子整体轮廓曾入画、座面有实际观测；坐下后允许出画并保留此前历史。相机仍为双眼中点前方 2 cm，随头部旋转，66.56°×40.49°，没有新增俯角或朝向椅子的控制。

用椅子网格凸包判断完整轮廓是否处于相机视锥。用已有 128×96 渲染记录检查物体像素 >=123，再查导出时经过人体遮挡的真实 first-hit 点是否落在臀部对应座面。每帧至少 4 个座面点，连续 8 帧，才能建立观测历史；窗口内至少有 16 帧已具备该历史。导出点每帧最多 512 个，这里的“点数”不是像素数。座面由稳定坐姿下的臀部支撑高度确定，仅用于离线质量筛选，不把未来人体或未观测网格输给模型。

对旧清单的 694 条坐下录制进行整椅视锥检查：693 条在整个录制中都未完整容纳整椅；仅 `bistro_wicker_chair_white_painted_woven__002` 的第 20–25 帧完整入画，稳定坐下始于第 95 帧。连续观测不足 8 帧。按此配置，真实完整观测条件下没有坐下样本通过；这是严格条件下的筛选结果，不代表相机从未看到任何椅子部分。

## 合成补充椅子的训练候选

实际部署仍只提供相机真正观测到的场景。用户提出为部分训练样本人为补充椅子，目的是让模型学习“椅子场景支持坐下”的条件关系。可以作为训练数据增强，但需要把补充点标为合成来源，不能改写为真实相机观测。

本版另外发布 `augmentation/sitting_candidates.jsonl`：保留 v1 合格的坐下接触窗口、源身份和观测审核结果，字段 `condition_provenance=proposed_synthetic_chair_observation`、`input_materialized=false`。它是待构造增强的候选，不是已生成点云或已接通训练的数据。不得用它替代真实观测评测清单，也不能绕过原有接触窗口限制。

后续增强应复用全局静态场景输入接口，保留实际相机与局部可见观测；补充椅子通过固定随机种子从表面采样，加入与真实输入相近的稀疏度、局部缺失和位置误差。增强比例、采样参数及训练接入尚未确定，不改动正在运行的 AMASS 阶段。

已实现独立增强原型 `scenemi/data/grail_augmentation.py`，参数在 `configs/grail_chair_augmentation.json`。两条指定训练椅子各选起点 0 的接触合格窗口，仅累计截至第 0 帧的真实静态观测；另外从原始椅子表面按面积采样 2,048 点，随机保留约 80%，加入 5 mm 标准差的位置扰动。噪声不是实测传感器标定，当前也没有模拟连续空间遮挡，不能声称已匹配真实前置场景模型。采样函数不读取人体真值或未来相机观测。

本地 `runs/grail_ego_quality_v2/augmentation_preview/PROTOTYPES.json` 记录早期两条原型的身份和种子；NPZ 通过 `synthetic_mask` 区分来源，相机和动作原文件不改动。配套 PNG 依次显示真实因果观测、补充椅子的条件、原动作/场景参考；图的第三栏不是模型生成结果。两条原型只用于检查，后续 400 条扩展见下节，当前均未接入正式训练。可复现命令：

来源掩码用于审计，后续接入时不能把它作为坐下类别的额外标签送给模型。真实场景条件的评测仍只读取实际可见观测。

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 2 --cpu-count 2 -- \
  env CUDA_VISIBLE_DEVICES= python -m scenemi.data.grail_augmentation \
  --selection runs/grail_ego_quality_v2/augmentation_preview/SELECTION.jsonl \
  --baseline runs/grail_ego_contact/full \
  --output runs/grail_ego_quality_v2/augmentation_preview --plot
```

“未观测到椅子”不能当作“已知没有椅子”。若同一坐下真值仅被随机增删椅子输入，模型仍可学习忽略场景。要检验动作歧义消解，应包含相近头部轨迹的坐下和蹲下真值，以及确实观测到椅子或空地的条件；至少对同一输入轨迹比较有场景、无场景的动作分布、座面接触和碰撞。合成条件下的效果与真实观测条件下的效果分别报告。

## 400 条坐下训练条件（2026-10-23）

按用户提出的约 400 条规模，从 694 条训练候选中固定 seed 20261023，轮转随机排列的椅子家族及家族内录制，选出 **400 条录制、149 个椅子家族**。其中 110 家族各 3 条、31 家族各 2 条、8 家族各 1 条，保留总共 9,539 个重叠的接触合格窗口。没有根据生成模型表现挑选，也没有增加 validation/test 样本。

每条选择最早接触合格的 128 帧窗口，沿用上述增强参数生成一份带来源掩码的全椅表面采样条件，共 400 份、655,356 个合成点。真实观测仅累计截至窗口起点；原相机、动作和标签不变。全量文件哈希、元数据身份、训练划分、窗口范围、有限值和真实/合成点拆分已核验，见本地 `runs/grail_ego_quality_v2/chair400/VALIDATION.json`。数据准备目录约 20 MiB，不新增模型权重。

本次完成的是这 400 条的代表窗口条件；其余合格窗口尚未接入场景训练读取器。原候选行中的 `input_materialized=false` 描述选取时状态，最新生成文件及状态以 `conditions/PROTOTYPES.json`、`READINESS.json` 为准。旧全库筛选审核不回写成新的生成证据。没有进行完整物理有效性或 400 条逐例人工视觉验收。

```bash
python -m scenemi.data.grail_selection \
  --candidates runs/grail_ego_quality_v2/full/augmentation/sitting_candidates.jsonl \
  --output runs/grail_ego_quality_v2/chair400 --count 400 --seed 20261023

python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 3 --cpu-count 2 -- \
  python -m scenemi.data.grail_augmentation \
  --selection runs/grail_ego_quality_v2/chair400/SELECTION.jsonl \
  --baseline runs/grail_ego_contact/full \
  --output runs/grail_ego_quality_v2/chair400/conditions
```

## 执行与审核

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 4 --cpu-count 2 -- \
  env CUDA_VISIBLE_DEVICES= python -m scenemi.data.grail_filter \
  --baseline runs/grail_ego_contact/full \
  --output runs/grail_ego_quality_v2/full --workers 2
```

`--selection <JSONL>` 只运行源 ID 子集，输出标记为诊断；正式清单要求全量扫描。运行封存明确的源码、配置、身体模型与上游清单哈希，逐条追加 `RECORDS.jsonl`，允许在相同身份下断点恢复。完成后发布真实观测 `train/windows.jsonl`、`train/excluded.jsonl` 及独立合成椅子候选。

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 2 --cpu-count 2 -- \
  env CUDA_VISIBLE_DEVICES= python -m scenemi.data.grail_filter_audit \
  --output runs/grail_ego_quality_v2/full
```

审核独立复算每条窗口交集、因果准备时刻、元数据身份、候选来源和已知楼梯回归，输出 `VALIDATION.json`。只有扫描 `state=complete` 且审核通过才表示清单可供后续数据接入；不等于物理有效性已证明或训练已经使用。EXO、原始验证/测试范围及原始文件保留。

## 最新训练接入与随机复核

400 条坐下条件现已接入 EGO 场景读取器；上文的“未接入”描述其生成时状态。旧全局 EGO 场景对照正在使用三数据集训练，下一轮将从 AMASS EGO-only 最佳权重初始化并启用预测皮肤局部反馈，见 [当前配置与验收](../SCENE_LOCAL_zh.md)。训练/选模排除 RICH。

最新数据检查页是 [localhost:8840](http://localhost:8840/)。固定 seed 2026102302，每类无放回选两个不同物体家族，家族内随机录制及合格窗口：路沿、斜坡、楼梯 p1/p2 各两条，另从选定 400 条坐下增强录制中抽两条，共十条。没有按可视化观感重新抽样。

页面默认循环所选 128 帧，可切换完整原序列。青色点只包含截至窗口起点的真实静态观测；橙色椅子仅是合成训练条件；灰色完整源场景用于检查。坐下样例不能算通过“相机真实完整看到椅子”的筛选。逐帧可见点仍来自导出数据，不将未来静态并集冒充当前输入。

案例、源哈希和窗口范围位于本地 `runs/grail_preview/support_v2_review/SELECTION.json`；数据审计在 `AUDIT.json`，逐例图层切换检查在 `VIEWER_VALIDATION.json`。原始动作没有贴地或物理修正，筛选通过不代表物理正确性已经证明。
