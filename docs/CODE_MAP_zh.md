# 代码导航

## 当前 EGO 主线与模型阅读入口（2026-11-02）

`configs/native_ego4_from_scratch_v2.yaml` 对应的四源 EGO-only 已完成 14,000 步，选中最后一步。新实验使用独立的 `configs/ego_only_4datasets.yaml`，输出到 `runs/ego_only_4datasets`。

相机误差适配使用 [ego_only_camera_adaptation.yaml](../configs/ego_only_camera_adaptation.yaml)，从固定的 14k 父权重续训至 16k，保留完整优化器和随机状态。扰动、重建同一窗口坐标和干净相机监督隔离位于 [camera_noise.py](../scenemi/data/camera_noise.py)、[observations.py](../scenemi/observations.py)。默认从零训练配置与已完成运行不变。

统一入口是 `python -m scenemi {train,preflight,evaluate,view} --help`，训练实现在 `train.py`，评估实现在 `evaluate.py`，数据装配在 `data/ego.py`，权重读写在 `checkpoint.py`；旧 `native_ego4_training.py` 仅保留兼容导入，不为每个实验增加脚本。训练由用户手动发起，必须通过资源限额入口；不会自动续训。

| 阅读顺序 | 文件与定义 | 当前职责 |
|---|---|---|
| 1 | [train.py](../scenemi/train.py) / [evaluate.py](../scenemi/evaluate.py) | 分别负责训练循环、断点选模与独立验证；[data/ego.py](../scenemi/data/ego.py) 装配四个数据集，[checkpoint.py](../scenemi/checkpoint.py) 保存与严格加载完整权重。 |
| 2 | [pipeline.py](../scenemi/pipeline.py)，`MotionPipeline.forward` | 当前模型的顶层前向：公共观测、原生体形预测、相机安装关系、条件去噪、运动学重建。EGO 通过 `use_scene=False` 路径。 |
| 3 | [observation_encoder.py](../scenemi/observation_encoder.py)、[body.py](../scenemi/body.py)；[native_shape.py](../scenemi/geometry/native_shape.py)，`NativeBodyAdapter` | 观测的时间编码/池化与体形输出：SMPL-H 16 维、SMPL/SMPL-X 各自 10 维及模板选择，GRAIL G1 固定模板。 |
| 4 | [observations.py](../scenemi/observations.py)，`public_observations` / `body_condition` | 输入与标签隔离；EGO 清空外部人体关节观测，再按预测体形和安装关系从相机观测生成头部条件。 |
| 5 | [native30_model.py](../experiments/offline_camera_retrain_v1/native30_model.py)，`observation_condition` / `camera_condition` / `_core_pass` | 实际观测编码和去噪主干调用；当前仍位于历史 experiment 包中。 |
| 6 | [mdm_scene_unet.py](../model/mdm_scene_unet.py)，`TemporalUnet` / `MDM_Scene_UNET` | 底层时间 U-Net、条件注入和扩散步嵌入。 |
| 7 | `native30_model.py:reconcile_motion` / `sample_native30` | 旋转规范化与 FK 重建；DDIM 采样。 |

当前 EGO 计算主路径可以概括为：

```text
128 帧相机位姿、有效性/置信度等观测
  → 观测时间编码 → 体形/模板预测 → rest 骨架及头部安装偏移
  → 相机到头部条件转换 + 观测/相机编码
  → 时间 U-Net（同时输入 noisy motion 和扩散步）
  → 201D clean-motion 预测 → 旋转规范化、FK 重建
```

`scene_model.NativeScenePrior` 是 `MotionPipeline` 的兼容别名，旧参数键保持不变。可选场景编码放在 `scene.py`，U-Net 公共导出在 `unet.py`。底层仍保留 `MotionPipeline → AmassMotionPrior → Native30SceneMI → OfflineSceneMI → nn.Module` 兼容继承链；最后再通过 `core` 调用 `MDM_Scene_UNET/TemporalUnet`。这解释了为什么只看 `scenemi/models.py` 看不全当前架构。

训练损失在 [losses.py](../scenemi/losses.py) 的 `motion_loss` 和 `pipeline.py` 的 `native_motion_loss`；训练中的验证与独立评估共用 `evaluate.py:evaluate`，同时提供独立的 `python -m scenemi evaluate --checkpoint ... --output ...`。加载使用 checkpoint 保存的配置、安装参数和身体基底，严格检查全部参数键。训练续训仍要求契约与设备一致；可加载推理不意味着可以跨源码契约无条件原样续训。`scenemi/evaluation.py` 是旧 AMASS 协议评测，不能当作通用四源模型评测入口。

已有入口与环境文件：

- [__main__.py](../scenemi/__main__.py)：`python -m scenemi` 提供四源 EGO 训练、预检查、独立验证和共用 Viser 入口；AMASS 准备/索引入口保留。旧联合场景训练与历史评测仍使用原接口。
- [amass_prepare.py](../scenemi/data/amass_prepare.py)、[amass_dataset.py](../scenemi/data/amass_dataset.py)、[checked_scene_dataset.py](../scenemi/data/checked_scene_dataset.py)：AMASS 准备/索引及当前原生数据筛选读取；原生导出仍有历史实现依赖。
- [requirements-rtx5090.txt](../requirements-rtx5090.txt)、[requirements-rtx5090.lock.txt](../requirements-rtx5090.lock.txt)、[ENVIRONMENT-RTX5090.md](../ENVIRONMENT-RTX5090.md)：已有环境说明和依赖记录。它们不等于已验证的最终发布版安装流程；环境说明包含早期实验阶段的历史状态。

### 已核实的组织问题

- **配置透传已修复。** `model.latent_dim` / `dim_mults` 会传入模型构造；现有 256 / [1,2,4] 权重的参数树保持不变。训练提供终端进度、验证进度与 TensorBoard，记录损失项、学习率、梯度范数、速度和各数据集指标。
- **先构造场景模型再关闭场景。** 当前 EGO-only 仍实例化历史场景分支，冻结参数和关闭路径不等于这些模块没有创建或没有进入 `state_dict`。
- **条件接口隐含。** `Native30SceneMI._core_pass` 用 forward hook 注入条件，并临时替换 `scene_embedding.forward`；发布实现应明确传递条件张量。
- **历史兼容层仍保留。** 当前四源训练/验证已按文件职责拆分，和 Viser 一起使用统一 CLI；历史联合场景与封存实现未迁移。`view prepare --checkpoint ... --output ...` 导出每数据集一个固定验证窗口，`view serve --manifest ...` 校验并播放；不会在播放时推理。

### 权重依赖与代码模块化是两回事

`scene_ego128_residual` 和五组 `scene_ego128_conditioning` 的 checkpoint 只保存适配器参数，恢复时需要冻结基座的参数。这是当时的权重存储选择；可以导出包含基座的完整模型，模型模块仍能分别定义。

两种无场景续训属于从父权重继续优化，不能一概称为“只保存适配器”：例如 `continuation.py:save` 保存完整 `model.state_dict()`。此外，旧 `scene_repair.py:make_model` 还在构造时读取 AMASS 初始化权重，随后再加载完整无场景权重，造成额外历史路径依赖。应分别记录训练来源、checkpoint 内容和加载器实际读取的文件，不能用一条箭头混称。

当前四源 EGO 的 `checkpoint.py:checkpoint` 保存完整模型、优化器和随机状态，不加载那些旧 EGO/EXO/场景模型 checkpoint。完整权重仍需要相同架构定义与人体模型等外部资产。

## 历史实验入口（以下“当前/最新”均为当时记录）

最新三数据集联合无场景主线见 [原生联合先验适配](NATIVE_JOINT_PRIOR_zh.md)：`native_joint_training.py` 提供共享优化器、损失和验证；`native_joint_batched.py` 提供可配置批次与额外早期完整验证；`native_batch.py` 负责真实吞吐基准及批次采样；`geometry/temporal.py` 修复多个独立出画区间。`native_joint_data.py` 提供不读取场景的已核验 EGO 输入，`native_joint_runtime.py` 复用原准入与限额；当前配置为 `configs/scene_joint128_noscene_batch32.yaml`。AMASS 在该阶段仅作 EGO 验证监测，不参与训练；不启动外部周期监控。旧失败任务、封存源码和修正验收记录保留供追溯。

其 AMASS 初始化主线为 `scenemi/exo_continuation.py` / `configs/amass_exo_continue.yaml`：
从已选 EGO 55k＋EXO 42k 边界模型继续 EXO-only 20k，共享主干冻结，
每 5k 同步完整验证，保留 best/last。进度在
`runs/amass_exo_continue128/training/STATUS.json`；尚未启动时查看 `STARTUP_STATUS.json`。
误开的 warm 联合训练已停止并按授权退役权重，记录保留。

AMASS 从零交替对照及历史联合配置见 [联合训练协议](AMASS_JOINT_zh.md)：
`joint_model.py` 在同一 U-Net 的差分两路完整反传，`joint_training.py` 管理
严格 4＋4 交替采样、同步完整验证与包含共享主干的恢复权重，`joint_runtime.py`
保留并发准入限制。配置为 `configs/amass_joint.yaml`，从零实验实时状态在
`runs/amass_joint128/scratch/training/STATUS.json`。

最新 EGO-only 场景与无场景适配结果、检查点身份及验证口径见 [匹配对照报告](SCENE_COMPARISON_zh.md)。

当前 AMASS EXO 去噪修复与夜间条件接续见 [协议与进度](EXO_DENOISING_zh.md)：`exo_denoise.py` 定义冻结 U-Net 内部修复注入，`exo_denoise_training.py` 负责 AMASS 全验证选模；`data/native_repair.py` / `scene_repair_training.py` 接通当前三数据集原生输入的无场景适配。`monitoring.py` 读取 `configs/exo_overnight.json` 执行定时检查，下一阶段先固定 AMASS 最终 best 并重新验收，不从变化中的训练权重直接启动。根 CLI 仍保持原封存。

36k 后的等预算边界条件短对照由 `scenemi/exo_completion.py` / `configs/amass_exo_completion.yaml` 管理，复用原冻结模型和选模引擎；两组唯一方法差异为是否增加由真实可用估计计算的边界条件。用户已停止夜间周期检查，此短试只在训练结束时生成一次比较报告。

当前五组冻结先验实验见 [几何条件实验](SCENE_CONDITIONING_zh.md)：`geometry/scene_features.py` 只从实际观测构建关系与可靠性；`scene_adapter.py` 将零初始化残差注入冻结 U-Net；`scene_conditioning.py` 负责统一验收、固定预算训练和可恢复评测；参数统一放在 `configs/scene_conditioning.yaml`。进度用现有 `ops status --run runs/scene_ego128_conditioning` 汇总，不新增监控脚本。

数据规模、帧数及筛选口径见 [数据分布表](DATASET_DISTRIBUTION_zh.md)，由 `scenemi/data/scene_inventory.py` 按实际加载器和已封存索引复算。清理策略与最新保留状态见 [维护记录](maintenance/README_zh.md)。

`scenemi/data/integrity.py` 对封存 AMASS train/val 做只读哈希扫描，必要时调用原始
CPU 预处理函数在独立目录重建候选；只有输出完整 SHA256 与原记录一致才接受。
它不修改训练索引、不读取独立 test，也不更改模型训练入口。

场景退化诊断与修复见 [修复说明](SCENE_REPAIR_zh.md)：`scenemi/scene_components.py` 负责固定权重组件干预；`scene_residual.py` 定义冻结主干的有界场景修正；`scene_repair.py` 负责真实验收、适配训练、紧凑恢复与原口径选模。实验参数在 `configs/scene_ego128_residual.yaml`，复用入口 `python -m scenemi.benchmark scene components|repair`，不新增按日期命名的脚本。

`scenemi/comparison.py` 独立复算有/无场景的配对家族指标，支持等待训练选模完成后自动写比较报告；只使用 CPU，不控制训练或其他进程。

后续场景收益核验见 [核验协议](SCENE_FOLLOWUP_zh.md)：`scenemi/scene_followup.py` 运行固定权重场景干预及额外无场景训练预算对照，`scenemi/geometry/scene_quality.py` 对预定 GRAIL 配对动作做足底/座面几何诊断。配置集中在 `configs/scene_residual_followup.yaml`，入口暂用独立模块，保留已封印根 CLI。

`scenemi/scene_checks.py` 负责固定步数的提前完整评测和原生场景资产可用性审计；`configs/scene_checkpoint_3k.json` 固定当前无场景 3k 与场景 3k 的比较身份，CPU watcher 不改正在运行的训练。

三数据集共用的观测表面关系监督在 `scenemi/geometry/observed_relations.py`；`scenemi/scene_relations.py` 负责从已选 3k 场景修复权重分叉的局部监督试验、真实验收及预定小集的场景敏感性核验。配置为 `configs/scene_relations.yaml`，分支关系和结论限制见 [场景关系监督](SCENE_RELATIONS_zh.md)。

`scenemi/geometry/relation_audit.py` 只读核验预定三数据集窗口的观测平面覆盖、原生人体坐标与错配几何差异。`scenemi/continuation.py` / `configs/scene_noscene_continue.yaml` 延续独立无场景父模型的原 AdamW/RNG 与末期学习率，避免高学习率重启的对照退化。`scenemi/shared_runtime.py` 仅识别用户明确授权并绑定完整进程身份的其他本用户 GPU 作业，保留原准入器全部资源检查。

`scenemi/relation_evaluation.py` 对关系监督试验的固定最终权重做可恢复评测，逐例保存正确/空/错配场景指标与预测，并可追加原 176 窗口验证。该入口绕开原试验汇总中重复 `rows` 参数的错误，保留原训练源码封印。

## 新 AMASS 阶段

| 文件 | 职责 |
|---|---|
| `configs/amass128.yaml` | 128 帧、30 Hz、AMASS-only、场景关闭、EGO/EXO 两种条件 |
| `configs/amass_ego128.yaml`、`scenemi/ego_training.py` | 仅训练模式比例改变的从零 EGO-only 对照，复用封存训练循环并隔离输出 |
| `configs/amass_splits.json` | 固定 EgoAllo commit 的 13/4/2 子库划分 |
| `configs/paths.example.yaml` | 机器路径模板；本机使用被忽略的 `paths.local.yaml` |
| `scenemi/config.py` | 校验协议，拒绝旧混合数据权重、场景输入和错误划分 |
| `scenemi/data/amass.py` | 原始文件盘点、上游代码核验和预处理命令计划 |
| `scenemi/data/amass_prepare.py` | 实际调用官方 0a、CPU 约束、逐文件审计、SHA256 恢复和状态记录 |
| `scenemi/data/amass_floor.py` | 与原版标签一致的一维高度聚类，避免密集邻接图的内存峰值 |
| `scenemi/data/amass_prepare_bounded.py` | 19 来源逐字段等价性核验、原封存清单审计与显式恢复适配器 |
| `scenemi/data/amass_dataset.py` | 全量索引核验、128 帧窗口、EGO/EXO 合成观测、标签与输入隔离 |
| `scenemi/data/observation_gaps.py` | EXO 连续缺失片段采样、输入掩码；保持原始时钟和完整监督 |
| `scenemi/geometry/camera.py` | 虚拟相机固定外参、头部逆变换、CPF 转换 |
| `scenemi/geometry/body.py` | SMPL-H 原生形状、22 关节、201D 编码 |
| `scenemi/geometry/shape.py` | 可微 SMPL-H 形状、共同 pelvis 参考、预测相机外参 |
| `scenemi/observations.py` | 允许的观测字段、可见性清理、预测形状驱动的相机到头部转换 |
| `scenemi/models.py` | 原 native30 主干、场景关闭、仅观测驱动的静态形状编码器 |
| `scenemi/losses.py` | 动作、相机和形状监督，预测与真值分别做 FK |
| `scenemi/training.py` | 实际更新、诊断、逐位恢复检查、正式训练和 CPU 就绪等待器 |
| `scenemi/evaluation.py` | 封存 55k 模型的 AMASS EGO 128 帧测试、固定清单与逐窗口记录 |
| `scenemi/benchmark.py`、`configs/amass_benchmark.json` | 预先固定 EGO-only 50k/55k 身份，复用完整 344 窗测试清单并保存可独立复算的预测 |
| `scenemi/scene_evaluation.py`、`configs/scene_input_benchmark.json`、`scene_local_input_benchmark.json` | 由 `benchmark scene` 对旧全局/新局部场景检查点做固定权重开关诊断，复现原指标并保留逐窗预测；按各自验证权重汇总，不替代独立训练对照 |
| `scenemi/selection.py`、`configs/amass_ego_selection.json` | AMASS EGO-only 全 337 条 val、2,222 窗口选模，独立固定快照与最佳权重 |
| `scenemi/geometry/native_shape.py` | SMPL/SMPL-X 原生系数和预测模板、固定 GRAIL G1 解码、原生骨架审计 |
| `scenemi/scene_model.py`、`data/scene_dataset.py`、`scene_training.py` | 三数据集 EGO 场景适配、已观测历史与当前点、无真实体形输入、精确恢复验收 |
| `configs/scene_ego128.yaml`、`scene_paths.example.yaml` | 独立场景分支协议与本机路径模板；不包含 RICH |
| `scenemi/geometry/skin.py`、`scenemi/data/local_scene.py` | 用预测原生体形计算皮肤锚点，读取因果历史和逐帧真实表面 |
| `scenemi/local_model.py`、`scenemi/local_training.py` | 预测皮肤局部条件、门控限幅反馈、独立训练和真实数据验收 |
| `scenemi/scene_selection.py` | 固定分层验证清单、动作类别等权选模、每 1000 步候选保留、较大验证集候选筛选 |
| `configs/scene_ego128_local.yaml` | 用户已接受 GRAIL 的 EGO 局部反馈与两阶段选模配置 |
| `scenemi/scene_ablation.py`、`configs/scene_ego128_noscene.yaml` | 同一 AMASS EGO-only 50k、相同数据与选模的无场景适配对照，由 `benchmark train-control` 启动 |
| `scenemi/runtime.py` | 仅为上述新评测/反馈入口扩展进程注册表，保留封存资源检查 |
| `scenemi/parallel_resources.py` | 用户授权的自有 SceneMI 训练/评估并发，拒绝未知 GPU PID，逐进程限额 |
| `scenemi/metrics.py` | 全局/标准 PA/公开代码兼容位置误差，明确单位及对齐口径 |
| `scenemi/geometry/alignment.py` | 仅凭连续 EGO 相机与预测体形解析放置全局 root；不修改局部姿态 |
| `scenemi/diagnostics.py` | 固定随机验证集诊断，记录原输出与相机放置的收益和副作用 |
| `scenemi/resources.py` | 用户授权后的共享 GPU 准入和进程内显存分配上限；复用冻结 cgroup/CPU 包装器 |
| `scenemi/visualization.py` | 复用现有 MotionViewer 的原始 GRAIL 播放适配，默认显示完整场景 |
| `scenemi/data/grail_review.py` | 每类按物体家族随机抽取合格训练窗口，区分真实因果历史与合成补椅图层 |
| `scenemi/amass_visualization.py` | 随机 AMASS 原生 SMPL-H 窗口及实际 EGO/EXO 输入检查，复用 MotionViewer |
| `scenemi/data/grail_quality.py` | 只读 GRAIL 可见性、地形支撑与原始 24 Hz 对齐审核；不自动删样本 |
| `scenemi/data/grail_visibility.py` | 全量 EGO 可见性扫描、因果静态历史、128 帧训练窗口筛选；保留原评测范围 |
| `configs/grail_ego_visibility.json` | EGO 可见性门槛及敏感性检查档位 |
| `scenemi/data/grail_contact.py` | EGO 足部/座面接触审核、与可见性窗口取交集、逐条断点恢复 |
| `configs/grail_ego_contact.json` | 接触审核范围、持续时间、几何和滑脚门槛 |
| `scenemi/data/grail_support.py`、`grail_chair.py` | v2 脚底支撑与坐下前整椅/座面真实观测检查 |
| `scenemi/data/grail_filter.py`、`grail_filter_audit.py` | 继承 v1 质量限制，发布新版清单并独立复算窗口与来源 |
| `configs/grail_ego_support_v2.json` | v2 筛选参数，保留相机定义、原始数据和评测范围 |
| `scenemi/data/grail_augmentation.py` | 单独标注来源的合成椅子点云原型、因果真实观测前缀与对照图 |
| `scenemi/data/grail_selection.py` | 固定种子按椅子家族轮转选择已筛选训练录制，保留接触合格窗口 |
| `configs/grail_chair_augmentation.json` | 有独立来源标记的增强原型参数，选定 400 条只用于场景训练 |
| `tests/model/`、`tests/data/`、`tests/training/` | 划分、时钟、形状、标签隔离、混合模式反传和开训门槛 |

统一入口是 `python -m scenemi`。包含 `config`、`amass inventory/plan/prepare/verify-height-clustering/index/preflight/ready/train`、`ops`。`amass prepare --bounded-height-clustering` 显式启用已核验的低内存执行适配器。**训练接口已接通；全量数据、CPU 检查和正式 batch 的 GPU 检查通过才允许正式开训。** `--diagnostic` 仅用于已完成的 pilot，不能绕过全量数据要求。协议、命令和实际进度见 [AMASS 说明](AMASS_RETRAIN_zh.md)。

AMASS 统一版 55k 训练已完成；为保留完整恢复能力，根 CLI 与训练源码封印仍不修改。独立 EGO-only 对照使用 `python -m scenemi.ego_training`，见 [EGO-only 协议](AMASS_EGO_ONLY_zh.md)。评测使用 `python -m scenemi.evaluation`，见 [AMASS 评测口径](AMASS_EVALUATION_zh.md)。v1 GRAIL 也保留原源码封印。新增 GRAIL 筛选、审计、增强暂用上述独立模块的 `python -m scenemi.data.<module>` 入口。具体命令及真实/合成场景区分见 [GRAIL v2 说明](data/GRAIL_SUPPORT_V2_zh.md)。

```bash
python -m scenemi config
cp configs/paths.example.yaml configs/paths.local.yaml  # 首次配置；已有本机文件时不要覆盖
# 编辑本机路径后，以受限 CPU 执行只读文件盘点，output 必须是新路径。
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 1 --cpu-count 2 -- \
  python -m scenemi amass inventory --paths configs/paths.local.yaml \
  --output runs/amass128/raw_inventory.jsonl
```

## 已有模型与运维

从 [当前实验索引](../experiments/offline_camera_retrain_v1/runs/INDEX_zh.md) 查模型身份、配置和报告。维护工具集中在 `experiments/offline_camera_retrain_v1/ops/`；新入口通过 `scenemi ops` 转发，不复制代码。

```bash
python -m scenemi ops status
python -m scenemi ops view --help
python -m scenemi view grail --help
python -m scenemi view amass --help
python -m scenemi grail audit --help
python -m scenemi grail ego-scan --help
python -m experiments.offline_camera_retrain_v1.ops.grail_prepare --help
python -m experiments.offline_camera_retrain_v1.ops.resources check
```

当前完整验证的已准备案例在 `runs/native30_final_metrics_v1/full_validation_noisy/visuals/PREPARED_CASES.json`（相对于旧实验目录）。查看时显式传入该 manifest、`--config` 和同目录 `visuals/cache`，避免误用已退役的旧预览菜单。8800 端口已有查看器时直接使用，不重复启动。

旧实验根目录仍保留 **67 个 Python 文件**：66 个实现/入口及一个仍在源码封印中的测试。它们是现有模型和 GRAIL 的兼容依赖；本轮保留原内容与路径。其余测试移至 `tests/legacy/`，历史配置作为测试夹具保存。后续新功能放入 `scenemi/`，解除旧阶段依赖后再迁移核心实现，避免改坏现有 checkpoint 恢复。

## 验证与发布边界

2026-10-28 的 AMASS 回放准备和真实 batch 性能测试集中于 `scenemi/replay/`，使用 `python -m scenemi.replay check|run|benchmark|pilot`。它不会自动启动新正式训练。`stream.py` 固定跨 batch 一致的案例、扰动与扩散噪声，`optimization.py` 处理来源/模式 loss 和按样本计的学习率，`screen.py` 是明确标记为非完整验证的配置筛查，`pilot.py` 固定父权重并比较配置。`scenemi/metrics_batch.py` 保留原指标定义并批量计算；测试在 `tests/training/test_replay.py`、`tests/training/test_replay_stream.py` 和 `tests/evaluation/test_metrics_batch.py`。接续点、数据比例、完整验证范围和待验收边界见 [回放方案](REPLAY_PRIOR_zh.md)，吞吐见 `runs/replay_benchmark/THROUGHPUT.json`，首轮配置筛查见 `runs/replay_pilot/RESULT.json`，追加筛查进度见 `runs/replay_pilot_retention/STATUS.json`。

CPU 回归测试：

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 6 --cpu-count 2 -- \
  env CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -t .
```

这只限制新子进程，不修改共享 GPU 设置。GPU 工作必须通过资源检查，拒绝时不启动。AMASS 阶段已获用户明确共享授权，使用 `python -m scenemi.resources check|run --config configs/amass128.yaml`；10 月 23 日新增的自有训练/选模/场景分支并发使用 `scenemi.parallel_resources`，见 [场景协议](SCENE_NATIVE_zh.md)。其余任务仍遵循原独占策略。

`model/`、`train/` 保留原作者接口；`scenemi/` 是新阶段明确的数据与模型接口。当前模型仍依赖兄弟仓库 `diffusion-motion-inbetweening`，许可证人体模型和数据也必须另备；这次整理尚未消除这些外部依赖。发布时提交源码、配置、文档与紧凑证据，不提交机器路径、人体模型、原始动作、权重和缓存。参考 [备份范围](BACKUP_zh.md)。
