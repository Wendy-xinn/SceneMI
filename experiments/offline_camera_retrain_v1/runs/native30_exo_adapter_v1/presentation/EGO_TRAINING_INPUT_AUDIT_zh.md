本文件是原始输入审计的离线阅读副本，仅将原机绝对路径链接改成文字；原始逐字证据见 [MD](evidence/EGO_TRAINING_INPUT_AUDIT_zh.md) / [JSON](evidence/EGO_TRAINING_INPUT_AUDIT.json)。

# 原55k / 所选41k EGO主训练输入审计

审计时间：2026-10-18T13:55:48.204088+08:00。审计对象是 `native30_gait_v2_ego55k_oct15`，包括同一训练轨迹中选出的41k checkpoint；不包含后来另起的EXO适配实验。本次仅CPU只读核查，没有修改源码、训练配置或checkpoint。

**结论：原55k及其41k前缀实际只使用真实EGO与伪EGO输入，没有EXO人体观测输入。** 这里的EGO还包含场景和静态身体信息，不能简化为“只有头部一个张量”。

## 真实训练记录

逐行流式解析正式 `training/trace.jsonl` 得到以下计数，统计单位为训练样本窗口（batch size为2），不是独立录制数。未计入归档的uncommitted尾部。步骤1至55,000连续，所有样本均为`split=train`，没有缺失/未知mode。

|范围|更新步数|样本数|EGO样本|EXO样本|场景开启步数|场景关闭步数|
|---|---:|---:|---:|---:|---:|---:|
|所选41k对应前缀|41,000|82,000|82,000|0|36,862|4,138|
|完整55k|55,000|110,000|110,000|0|49,468|5,532|

|数据组|输入来源|前41k样本|全55k样本|
|---|---|---:|---:|
|TRUMANS|伪EGO：自己的虚拟头相机|51,440|69,005|
|EgoBody camera_wearer|真实EGO：实际PV相机轨迹|11,385|15,273|
|EgoBody interactee|伪EGO：自己的虚拟头相机|10,654|14,292|
|RICH|伪EGO：自己的虚拟头相机|8,521|11,430|

全55k共15,273个真实EGO样本、94,727个伪EGO样本；前41k分别为11,385和70,615。最终四组计数与已保存的 training/protocol.json（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_gait_v2_ego55k_oct15/training/protocol.json:1786`） 完全一致。直接记录见 training/trace.jsonl（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_gait_v2_ego55k_oct15/training/trace.jsonl:1`）；41k身份见 best.json（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_gait_v2_ego55k_oct15/training/best.json:6`）。这只是输入覆盖核查，不构成41k已在完整验证集最优的证据。

## 配置和共享manifest为什么容易误读

真实配置（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_gait_v2_ego55k_oct15/config.json:249`） 和 prepared sampling_plan（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_gait_v2_ego55k_oct15/prepared/effective_config.json:330`） 均限定`training_mode=ego`。`ego_probability=0.7`先用于生成候选计划，再由 training_plan（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/prepare_native30.py:141`） 过滤为EGO并重归一，因此本次实际EXO比例为0，而不是30%。

共享manifest同时保存了可供后续EXO实验使用的元数据。TRAIN行中，TRUMANS 469行和RICH 62行允许`ego,exo`；EgoBody的camera_wearer有93行ego-only、65行exo-only，interactee有107行ego-only、65行exo-only。`allowed_modes`表示可用输入模式，不表示本次已抽取。EgoBody PV片段和完整master流在 split构建（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_splits.py:55`） 中分别限定为ego-only/exo-only，采样器还会 检查请求mode是否可用（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_data.py:150`）。

## 真实EGO和伪EGO具体是什么

EgoBody wearer使用发布的真实PV相机轨迹，经时间同步和标定变换到身体/场景世界坐标；代码见 raw_pv_camera（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/export_native20_training_scenes.py:65`） 和 wearer选择分支（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_export.py:168`）。这里不等于已接入在线RGB/SLAM估计前端。

TRUMANS、RICH及EgoBody interactee没有使用该目标自己的真实PV流，而是根据原生身体拟合的头部位置、旋转与固定安装偏移构造 虚拟头相机（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_export.py:310`）。interactee伪EGO表示“相机装在interactee自己头上”，不是“wearer的PV从外部看interactee”。

## 人体和场景条件

`ego_head_only_probability=1.0`，EGO人体运动观测只选15号头部；sample_item（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/prepare_native30.py:203`） 决定该选择，随后 观测构造（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_observations.py:130`） 对其余关节mask并清零，头部由同一个可能带噪/丢失的相机流反解。因此没有将EGO模式下的全身GT关节保留为干净观测，也没有启用额外手腕观测。身体rest、类型、尺度及推理所需的静态几何资产仍属于输入。

场景dropout配置为0.1，实际49,468步开启、5,532步关闭。输入是EGO视角first-hit观察到的表面；目标自身owner100被排除（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_export.py:238`），其他人或物体的可见表面可以保留。静态表面使用同一观察者/世界/划分内、整条所提供recording的已观察表面并集，动态表面使用当前帧，未观察位置保持unknown。见 scene配置（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_gait_v2_ego55k_oct15/config.json:30`） 和 EGO场景加载（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_observations.py:122`）。这是离线整段recording观察记忆，不是严格因果前缀地图；也不是直接把完整GT扫描全部作为可见场景输入。

## GT监督与扩散带噪输入的边界

GT全身动态姿态不是干净条件输入。输入白名单（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_observations.py:20`） 排除了`motion`、contact truth和目标GT pose；当前默认reference由观测构造。

但不能说GT“只用于算loss”：本模型以 GT motion减reference作为训练残差目标（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_refinement.py:174`），并用它构造扩散训练的带噪状态`x_t`，再计算全身/脚步等监督；见 训练循环（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/train_native30.py:127`）。这是扩散训练本身的监督过程，不是提供干净全身conditioning。推理则 从固定seed的高斯随机噪声开始（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_refinement.py:235`）。另外，伪EGO相机和离线可见几何本来就是由数据集身体/场景拟合模拟生成的，不能把它们描述为真实传感器前端输出。

## 后续EXO适配是独立阶段

`native30_exo_adapter_v1`和`native30_exo_adapter_balanced_v2`均从所选41k EGO模型出发，冻结原主干，只用EXO样本更新显式EXO门控的内部adapter；EGO路径绕过adapter。当前v2的 mode配置（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_exo_adapter_balanced_v2/config.json:258`） 和 冻结基座适配协议（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_exo_adapter_balanced_v2/config.json:324`） 可复核。此前全参数混合EGO/EXO的`native30_ego_exo_adapt_v1`又是另一场实验。这些都不能倒推为原55k EGO主训练已经使用过EXO人体输入。

## 文件身份与复核范围

机器可读计数、步骤/split检查、源码行号及SHA见 TRAINING_INPUT_AUDIT.json（原机定位：`/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_gait_v2_ego55k_oct15/TRAINING_INPUT_AUDIT.json`）。配置SHA为`89fadfc6d1af8cf0c819e503ac646443b6c8d281ce46abb5f7e8c53828a796ad`，prepared effective配置SHA为`1d58e21e4070fa86e543782a6b74291828f5579b7a80f38c6d1512134ad495f8`，protocol SHA为`104171856168b49d18abe6ce7fd44dfdbaa7216fa4398c7ce70b9b4351417238`。

正式trace为800,273,916字节，55,000行，mtime为2026-10-17T20:46:16.293858+08:00。先前流式统计时没有同时计算trace SHA，本报告复用了该次完整统计，没有为补报告重复读取800MB文件；JSON明确将trace SHA标为null，并记录stat/mtime/inode，不伪造哈希。
