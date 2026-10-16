# RICH 全量验证范围与外部比较协议核对

核对日期：2026-10-16。只读检查数据、索引和评估实现；未修改训练代码、权重、配置或优化器状态，未运行 GPU 推理。

## 本轮“全量”的准确范围

本地 `RICH/val_body.zip` 包含 28 个序列、17,661 个 `.pkl` 人体帧。压缩包序列集合与 `RICH/extracted/val_body`、现有 prepared 数据和 raw30 manifest 完全相同；提取后 `.pkl` 数量也为 17,661，没有发现已下载 VAL 源序列缺失。这些 VAL 序列各只有一个标注目标，覆盖 13 名受试者、4 个场景。

固定的 `native30_official_windows64_v1/validation_ego.npy` 中，RICH 有 **206 个窗口，来自 27 个序列**。每窗 192 帧、30 Hz，起点为 `0, 64, 128, ...`，不追加末尾非网格窗口、不 padding。相邻窗口重叠 128 帧，因此不能当作 206 个独立动作样本。

| 范围 | 数量 |
|---|---:|
| 下载并接入的 RICH VAL 源序列 | 28 |
| 有合法 192 帧网格窗口的序列 | 27 |
| 全部合法 RICH ego 网格窗口 | 206 |
| 每窗 clean / noisy 各一次的预测总数 | 412 |
| 源人体帧 | 17,661 |
| 网格覆盖的唯一源帧 | 16,640（94.22%） |
| 含重叠的窗口帧出现次数 | 39,552 |

`ParkingLot1_002_stretching2` 只有 165 帧，无法形成 192 帧窗口；另外 27 条序列共有 856 个尾帧不在当前网格覆盖范围。这轮结果应称为“RICH 全部合法 192 帧网格窗口验证”，不能称为发布 VAL 每帧无遗漏评估。若后续需要覆盖全部 28 条及所有尾帧，应另登记短窗和尾窗协议，不能混入本轮主指标而不说明。

RICH TEST 没有接入本地当前管线，代码明确不以 VAL 替代 TEST。当前结果用于开发验证，不是独立测试集成绩。

## 当前指标是什么

`native30_validation.raw_metrics` 的 `mpjpe_mm` 是原始未加权 GT 上的 **22 个 native SMPL/SMPL-X 身体关节 FK 世界坐标误差**，含 pelvis。预测与 GT 只共同经过输入首帧相机位置和水平朝向的刚体坐标变换；没有单独用 GT 对预测做平移、旋转、尺度或 Procrustes 对齐。因此该变换不改变距离误差。不能把 `mpjpe_mm` 叫作去根平移的姿态误差。

本轮适合同时报告：

| 指标 | 解释 |
|---|---|
| World MPJPE，mm | 全局身体放置与姿态合并误差，22 关节含 pelvis |
| Root-relative MPJPE，mm | 每帧分别减去预测与 GT 的 pelvis，保留旋转和尺度差异 |
| PA-MPJPE，mm | 每帧 proper similarity 对齐后误差，含尺度；不能代替全局误差 |
| MPJVE，mm/s | 世界坐标一阶差分误差，明确使用 30 Hz |
| 加速度误差，mm/s² | 世界坐标二阶差分误差，明确使用 30 Hz |
| Pelvis 位置 / 朝向误差 | 分离全局放置和转身问题 |
| Head 位置 / 朝向误差 | 输入头轨迹遵从性，和全身质量分开看 |
| GT 低速足部时段的预测足速 | 滑步诊断；GT 足速 < 0.2 m/s 只是运动学代理，不等于官方接触标签 |

`standard_motion_metrics` 默认参数和 `METRIC_PROTOCOL` 文本来自旧 20 Hz 版本；本轮应显式传 `fps=30.0` 并在报告写明 30 Hz，不能直接照搬旧说明。时间差分只在每个窗口内部计算，不跨窗口边界。重叠窗口保留时，应同时提供窗口等权与序列等权汇总、逐序列结果和误差分布；长序列不应被误认为独立样本更多就必然更重要。

## 与其他 ego 方法比较的边界

用户指定的三个方法都可纳入比较。**EgoGVAE 论文 §4.2 明确采用 RICH TEST；其 Table 2 同时比较 EgoGVAE、EgoEgo、EgoAllo，均用 AMASS 训练、128 帧合成 head/CPF 输入。我们本轮为混合训练后的 RICH VAL，尚不能直接对这张表排名。** [官方论文 Table 2 / §4.2–4.3](https://arxiv.org/html/2607.27755v1)。

| 同一出处：EgoGVAE Table 2，RICH TEST | MPJPE，mm | PA-MPJPE，mm |
|---|---:|---:|
| EgoEgo | 210.7 | 180.1 |
| EgoAllo | 193.1 | 172.9 |
| EgoGVAE | 179.9 | 165.9 |

这些是论文参考数值，未在本机重跑。EgoAllo 原论文 v2 Table 2 的 128 帧 RICH 数值是另一组 176.2 / 160.1 mm；不能与后续重评混列而不注明出处。[EgoAllo v2 Table 2](https://arxiv.org/html/2410.03665v2#S4.T2)。

| 方法 | 输入与人体形状 | 时间 / 骨架 / 指标核对 | 当前公开评估入口 |
|---|---|---|---|
| EgoGVAE | head/CPF；含相对地板高度；无手观测、无场景编码；预测 16 维 SMPL-H shape | 30 Hz；128 帧；pelvis + 21 身体关节；World MPJPE，PA 另列 | [eval.py](https://github.com/DCVL-3D/EgoGVAE_release/blob/main/eval.py)、[metric.py](https://github.com/DCVL-3D/EgoGVAE_release/blob/main/model/metric.py)、[head 输入](https://github.com/DCVL-3D/EgoGVAE_release/blob/main/model/utils.py)；默认数据入口为 AMASS TEST，未找到与论文 RICH TEST 对应的完整 manifest |
| EgoEgo | 比较身体先验时仅 head 位置和朝向；原完整系统另有视频头轨迹估计；无手、无场景；原公开骨架使用零 shape | 原版 30 Hz、120 帧、22 关节；原版 `compute_metrics_for_smpl` 的 MPJPE **减 pelvis**，不能直接混入统一 World MPJPE 表；EgoGVAE 表中为 128 帧重评 | [eval_stage2.py](https://github.com/lijiaman/egoego_release/blob/main/eval_stage2.py)、[原版 metric](https://github.com/lijiaman/egoego_release/blob/main/kinpoly/scripts/eval_metrics_imu_rec.py)；原仓库没有 RICH 评估入口 |
| EgoAllo | body 对比仅 CPF；含相对地板高度；不使用手观测；预测 SMPL-H shape；ground 高度由数据或估计提供 | 30 Hz；原论文 32 / 128 帧；root + 21 身体关节；World MPJPE，PA 另列 | [5_eval_body_metrics.py](https://github.com/brentyi/egoallo/blob/main/5_eval_body_metrics.py)、[metric helper](https://github.com/brentyi/egoallo/blob/main/src/egoallo/metrics_helpers.py)；公开入口声明并非论文的完全相同脚本，默认 AMASS TEST |

三者的论文指标名称需要结合具体重评来源看；当前 World MPJPE 与 EgoGVAE / EgoAllo 的 22 关节世界坐标形式接近，不能一概说完全不同。正式比较仍需将同一批预测交给同一 evaluator，核对 body model、root、单位和对齐。官方代码逐文件内容哈希、未确认项见同目录 `BASELINE_COMPARISON_PROTOCOL.json`。

但本轮与论文表格仍有以下实际差异，尚不能直接做胜负比较：

1. **训练数据不同。** 我们混合训练含 RICH 的 62 条 TRAIN 序列；其 13 名 VAL 受试者和 4 个 VAL 场景均在本地 RICH TRAIN 出现。TRAIN / VAL 序列 ID 交集为空，遵循发布序列划分，但不是未见受试者或未见场景设置，也不是 AMASS-only 到 RICH 的跨域泛化。
2. **人体形状条件不同。** 当前推理输入包括目标的 `rest`、`body_type`、`body_scale`；这些来自目标 GT 人体 profile。外部方法可能预测形状或使用平均形状，不能忽略这种额外先验。
3. **场景条件不同。** 当前有几何合成的 ego 可见表面 / 静态记忆输入，并非只有一条头轨迹。应明确标为 head + observed scene + known body profile；需要 head-only 公平对照时，按同样窗口和 seed 另跑 scene-OFF。
4. **相机、地板与噪声不同。** RICH 没有真实头戴相机，各方法依赖合成 head/CPF。当前首帧相机 XYZ 重锚定后，首帧相机高度为 0，没有显式相对地板高度；EgoGVAE / EgoAllo 的 head 参数化含地板相对高度。当前 clean / noisy 两组应分开汇报；不能与实拍 SLAM 噪声混称同一设置。
5. **时间窗口与评估样本不同。** 本轮 192 帧、stride 64；外部方法可能用 32 / 128 帧、不同 stride、长度过滤、随机抽样、body model 转换或多次采样。公开 AMASS 评估入口不等于完整 RICH 数据清单；本次尚未复核到他们逐个 RICH 评估窗口身份，不能假定其 VAL/TEST 范围与我们相同。
6. **后处理和骨架需要配齐。** 当前有 soft head / IK 和局部物理修正，外部方法也可能有 guidance 或后处理。两边均应写明；22 关节同名也应核对具体 SMPL-H / SMPL-X regressor、身体尺度及 root 定义。

用于正式对比时，建议先固定一份 RICH 评估 manifest，让各方法读取同样的 head 轨迹、角色、帧数、可用场景和 body profile 约定，采用同一 metric evaluator，分别列原始输出和允许的后处理。当前这轮能先回答“全 RICH 网格的质量、长尾和动作类型弱点”，并为后续同协议重跑 baseline 提供固定样本。

## 本地证据路径

- 源包：`/home/wenxin/projects/RICH/val_body.zip`。
- 源标注：`/home/wenxin/projects/RICH/extracted/val_body`。
- 源 manifest：`experiments/offline_camera_retrain_v1/data/native30_official_memory_v3/manifest.jsonl`，SHA256 `0ad36607e705a6480368486e98d3b5dd4de69f2ec916a45b73f2f0d9f2c9c9cc`。
- 网格元数据：`data/native30_official_windows64_v1/dataset.json`，SHA256 `6ca16b858d5e1f1aa028094a18651ce74f0fe426434b1e90d4f20bdb84836e01`。
- `validation_ego.npy`，SHA256 `63c29a65e1c8bf83128a84c885d56b2ce56ccae4f661d6401e47334ea3e50f0e`。
- 具体本轮窗口及未覆盖原因：同目录 `PLAN.json` 的 `plan` 与 `source_coverage`。
- 数据读取 / 合成相机：`native30_data.py`、`native30_export.py`。
- 指标：`native30_validation.py`、`standard_motion_metrics.py`。
