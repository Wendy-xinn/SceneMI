# AMASS 128 帧重训：阶段 A

2026-10-23 北京时间 00:48，独立的 **AMASS EGO-only 从零训练对照**已启动；只改变训练模式比例，其余配置保持一致。新进度见本地 `runs/amass_ego128/training/STATUS.json`，协议与恢复命令见 [EGO-only 说明](AMASS_EGO_ONLY_zh.md)。下文统一版 55k 及其完整恢复依赖保留，两者不共用输出目录。

最新状态：2026-10-22 北京时间 22:44 完成 55,000 次更新，保留可恢复的单个 `last.pt`；23:26 完成 EGO 测试清单的全部 344 个窗口（113 原始动作家族）。全局 MPJPE 为 175.22 mm、head 误差为 91.07 mm；三篇方法目前仅引用论文值，尚未完成同窗口同相机基线重跑。指标差异与复现命令见 [AMASS 评测协议](AMASS_EVALUATION_zh.md)，紧凑结果见 [测试验证记录](validation/amass_evaluation.json)。下文的启动、验收与处理记录保留为历史过程证据。

2026-10-22 全量 AMASS 0a 处理和文件身份校验已完成；加入 EXO 连续缺失后，正式 batch 8 的 CPU/CUDA 检查均通过。**17:33（北京时间）已从随机权重正式开训。** 启动与验收状态见本地 `runs/amass128/READINESS.json`，实时训练进度见 `runs/amass128/training/STATUS.json`，日志为 `training/train.log`。最新证据见 [连续缺失检查](validation/amass_observation_gaps.json)。原 192 帧模型保持独立可恢复。

2026-10-22 在 13,381 个已提交源文件处发现长序列高度聚类导致的内存回收停滞；新增通过真实数据逐字段对比的低内存执行适配器，从已审计清单恢复。原因、等价性核验与任务封存方式见 [长序列内存处理](data/AMASS_MEMORY_zh.md)。

2026-10-22 08:07（北京时间）原处理任务在 DanceDB 长片段触发自身 6 GiB cgroup 内存限制并被终止；最后提交 8,063 个源文件。SIGKILL 未触发 Python 异常处理，旧 STATUS 的 `running` 和 `failures=0` 不能证明进程存活。09:28 已按相同封印协议以 16 GiB / 4 核 CPU 预算启动恢复，先校验已提交输出 SHA256；不重做已提交源文件、不跳过未提交长片段。中断与恢复身份保存在本地 `preparation/INTERRUPTION_001.json`、`RESUME_LAUNCH_001.json`。就绪监控现已检查真实进程，进程消失或长时间未提交时不报告 ETA，恢复后重新计算本次调用的速度。

预测形状、数据适配、分离骨架损失、训练循环及恢复接口已经接通。全量索引上的 CPU 检查覆盖真实数据反传、20 步 DDIM、标签/缺失输入隔离及断点恢复；恢复后的下一步参数、优化器和随机状态逐位一致。这些是少量诊断更新，不能作为训练完成或生成质量的证据。[早期训练接口验收](validation/amass_training_setup.json) 和 [接口验证记录](validation/amass128_setup.json) 保留为历史证据。

## 已确定的协议

- 保留既有 native30 主干、256 latent、`[1,2,4]` 层宽及 201D 布局；窗口改为 128 帧、30 Hz，约 4.27 秒。新增静态形状编码器；AMASS root 改用统一 neutral 模板 pelvis 参考，避免依赖未知真实形状，详见 [坐标约定](COORDINATES_zh.md)。
- 阶段 A 从随机初始化训练 AMASS-only 动作先验；预测 absolute x0。旧混合数据权重和 reference-residual / IK refinement 不作为本次纯 AMASS 初始化。
- 当前配置在同一 AMASS 上随机切换 EGO/EXO 观测，各 50%；按 source family 等概率采样，再选择反射版本和完整窗口，不按窗口数量给长片段加权。EGO 保留虚拟相机输入，没有额外手腕真值。EXO 是合成身体部位/时间缺失，不能声称是真实相机遮挡。
- **用户已确认 EGO 推理只有场景和相机轨迹，没有身高或身体标定。** 形状编码器仅从允许的观测预测每窗口一组 16D SMPL-H betas，再得到 rest、相机外参和 FK。真实 betas、rest 仅作监督标签；按真实形状生成虚拟相机属于离线传感器模拟，真实头部外参不进入模型输入。形状估计可能有歧义，接口检查不能保证其精度。
- 用户进一步确认：上述相机输入限制针对 EGO；**EXO 可以提供估计的 3D 人体/关节**。EXO 观测须显式记录可见性、置信度/误差，不能把未观测的 GT 旋转补进条件。GT 形状在两种模式下都只作标签；前置模型实际预测的形状属于估计信息，不能与 GT 混为一谈。
- AMASS 没有真实外部相机，阶段 A 的 EXO 标记相机不可用，仅提供经过合成缺失掩码的 3D 关节位置，不提供旋转。窗口平移锚点只由可见关节计算。当前位置是干净 GT 的合成观测，尚未对真实前置模型误差做标定；这限制了实际 EXO 泛化的判断，场景阶段仍需真实估计输入的验证与适配。
- 场景 ViT、patch/local query、场景反馈不参与阶段 A；不要求场景数组，接口内部生成空占位。仅用动作监督，包括被遮挡关节的旋转、FK 位置、速度等，不制造场景接触标签。
- 不添加未经标定的位置/旋转数值噪声；EXO 在原身体部位掩码上加入连续整帧缺失，模拟出画/跟踪失败。EGO 相机轨迹始终连续。数值噪声仍需真实估计误差标定，不能沿用旧扰动参数。
- 正式配置为 `55,000` 更新、batch 8。最初数百次更新约 2 updates/s，对应约 8 小时；共享 GPU 负载和验证耗时会改变 ETA，实时值见 `training/STATUS.json`。该预算和早期 loss 不能保证收敛或生成质量，也不能据此承诺整个场景阶段的完成时间。

## EgoAllo 对齐的准确边界

官方仓库已克隆到兄弟目录 `../egoallo`，固定 commit `0c20ae8c8d48be4ff682803ea523bcb00a6a708d`。[官方源码](https://github.com/brentyi/egoallo/tree/0c20ae8c8d48be4ff682803ea523bcb00a6a708d)。划分来自 `src/egoallo/data/amass.py`，预处理入口是 `0a_preprocess_training_data.py` 和 `0b_preprocess_training_data.py`。

**本次确定 train-only：13 个 train 子库训练、4 个 val 选模、2 个 test 终评。** EgoGVAE 论文 §4.2 只说沿用 EgoAllo 划分；其[官方 train.py](https://github.com/DCVL-3D/EgoGVAE_release/blob/1dca529bf1efd6b53bf81da9c769ea3483777ac3/train.py#L49) 明确使用 `splits=("train",)`。本地 `../EgoGVAE_release` 固定至该 commit，13/4/2 成员与 EgoAllo 相同；第一步预处理经 AST 比较相同，仅两处注释不同。[源码证据](data/amass_protocol.json)。发布源码不能证明论文权重历史上一定如何训练。

EgoAllo 的 `1_train_motion_prior.py` 默认 **train+val**，以后重训时显式改为 train-only；官方 train+val 权重单列数据差异。定义保存在 `configs/amass_splits.json`。

反射版本与原动作、同序列窗口都保持相同 family/split。TEST 不用于选模型或调扰动。另需对齐 CPF/虚拟相机转换、是否已知形状、地板高度、世界/根对齐指标和生成采样数；只对齐子库名称还不足以完成公平比较。

混合观测训练可以报告统一模型的 EGO 结果：测试只给相机、采用相同 train-only 数据，并披露合成 EXO 条件。要归因“统一训练带来提升”仍需 EGO-only 消融。不能为了转换相机而额外读取测试形状或头部姿态真值；当前不声称已经完整复现原论文的 CPF 输入和评测。

## 本机数据盘点

原始 AMASS：`/mnt/e/Datasets/human/AMASS/data`；SMPL-H：`/mnt/e/Datasets/human/AGORA/smplh`。本机路径已放入忽略提交的 `configs/paths.local.yaml`。

| 名义划分 | 原始动作文件 | 上游文件名规则排除 |
|---|---:|---:|
| train | 13,592 | 1,194 |
| val | 364 | 8 |
| test | 140 | 0 |

以上不代表最终可训练序列/窗口数。完整文件身份在本地 `runs/amass128/raw_inventory.jsonl`；[紧凑盘点记录](data/amass_inventory.json) 保存上游五个源码的 SHA256。ProtoMotions 的已有转换不直接替代该协议。

已完成覆盖 19 子库的 pilot：45 个原动作及其反射，共 90 个 NPZ；另有 8 个文件按上游命名规则排除。各子库均找到在 0b 首帧裁剪后仍满足 128 帧的动作。独立 scipy/numpy FK 最大误差 `3.0142e-5 m`，相机/head/CPF 变换通过；原始帧 ID、neutral 映射和反射旋转经直接对照通过。**pilot 不等于全量已验证。**

全量逐文件检查有限值、字段形状、时钟、FK 和相机关系；异常停止并记录。原始数据不修改；已提交输出按 SHA256 验证恢复。代码、模型、性别映射和环境身份写在输出目录 `JOB.json`。输出估算约 14.6 GiB（不是上界），两倍估算加 20 GiB 余量的空间检查通过，运行时继续检查余量。启动身份、日志和 pilot 证据在本地 `runs/amass128/preparation/`。

## 准备与下一步

2026-10-22 全量处理与 CPU 验收已通过：14,096 个原始文件全部记账，12,823 条生成原始/镜像共 25,646 个 NPZ；另有 1,202 条按上游命名规则排除、71 条上游短片段。0b 首帧裁剪后满足 128 帧的原始动作 family 为 train **7,485**、val **337**、test **114**。三个划分相互隔离，正式训练仍只用 train。

`runs/amass128/READINESS.json` 已记录正式开训：全量原文件/输出哈希、完整索引、CPU/CUDA batch 8 真实数据更新、标签/缺失输入隔离和逐位恢复检查均通过。27 项回归检查通过。CUDA 验收峰值分配约 **1.54 GiB**、预留约 **1.64 GiB**；本进程的 PyTorch 分配上限保持 **6 GiB**。

用户已明确授权共享 GPU：使用 `scenemi.resources` 检查新进程预算，额外计入 1 GiB CUDA 运行时余量与 2 GiB 空闲余量，再复用原有 cgroup/CPU 限制启动。训练器在自身进程内设置 PyTorch 上限；CUDA 上下文等分配不在该上限内。准入只是启动时快照，不能预留共享资源。原冻结运维代码不变，其他用户的进程不受修改。授权及启动身份见本地 `training/LAUNCH.json`。

首轮 1,000 更新后的 12 条固定验证均产生有限值；对应正式 checkpoint 已在隔离 GPU 的 CPU 进程中读回，包含 183 项优化器状态和 CUDA 随机状态。记录为本地 `training/FIRST_CHECKPOINT.json`。它证明训练、验证与保存接口已正常运行，不代表生成质量已达标。

原始数据检查页：`http://localhost:8842/`。固定 seed=2026102201，随机选择 8 个不同 train 子库，每库均匀选择 family、原始/镜像和一个完整窗口；不按观看效果重抽。每例的原生 SMPL-H mesh 与实际训练标签的关节误差小于 0.001 mm；从正式索引独立重取 EGO/EXO 的全部输入逐项一致。紫色是源动作，不是模型预测。AMASS 不含场景，登高、坐下等原动作可能需要未记录的支撑物，参考地面也不作为阶段 A 输入。不能以原动作碰到或离开参考地面直接判定预处理错误。

EGO 输入为相机位置及旋转，不输入人体关节或真值体形；EXO 为带可用性掩码的 3D 关节位置，不提供旋转。阶段 A 的 EXO 是干净真值的合成缺失，部分模式保留全部关节，尚不是有误差的实际 3D 估计器。8 例源身份、未修改的公开输入、随机状态和浏览器播放验收保存于本地 `runs/amass128/preview/`。用户于 2026-10-22 确认保留当前“双眼中点前方 2 cm”的合成相机定义，本轮不改为额头位置，也不因此重做数据。GRAIL 的视场问题仍需单独核验。FOV 不影响当前 AMASS 轨迹；未来若修改安装外参，须协调读取器与相机逆变换，不必重新执行 AMASS 0a。

每次训练采样 EXO 窗口时，均匀随机选择 6 种预设身体部位组合之一（包括完整 22 关节）；再以 50% 概率选择左臂、右臂、左小腿或右小腿之一，随机连续缺失 8–32 帧。部位组合在同一 128 帧窗口内保持，短时缺失另外叠加；如果选到原本就缺失的部位，掩码不额外变化。这是按部位和时间段的合成缺失，不是每帧独立随机删点，也不来自场景遮挡。缺失只作用于输入，完整动作仍参与监督；EGO 始终只提供相机流。

EXO 另外以 **50% 概率**丢失一个完整观测片段：长度均匀取 **8–32 帧（约 0.27–1.07 秒）**，位置均匀选窗口开头、中间或结尾。另一半 EXO 窗口没有这种整帧缺失，仍使用上述关节掩码。EGO 整帧缺失概率为 **0**。原 128 帧时钟、完整动作和形状监督均保留；缺失处的输入值、可用性和置信度清零，不插值，不压缩时间轴，也不把不同动作拼接。开头缺失时仅用第一个有效观测建立坐标锚点，避免从不可用帧读取真值位置。该长度和比例是阶段 A 初始训练策略，尚不能代表真实跟踪失败的统计分布。

上述 8 例可视化保留无额外整帧缺失的原始输入，便于检查数据本身；训练时在线采样缺失，身份及缺失帧区间写入训练日志，原始 NPZ 不变。

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 4 --cpu-count 2 -- \
  python -m scenemi view amass prepare --paths configs/paths.local.yaml \
  --output runs/amass128/preview --count 8 --seed 2026102201
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 3 --cpu-count 2 -- \
  python -m scenemi view amass serve \
  --manifest runs/amass128/preview/PREPARED_CASES.json --port 8842
```

1. `../.venvs/amass-preprocess` 复用 Python 3.9/torch，仅补充 0a 兼容依赖，已实际跑通 pilot，原训练环境未变更。它是 **0a 子集环境**，不是完整 EgoAllo/EgoGVAE 重训环境；后者仍需独立 Python ≥3.12 依赖。
2. 复用未修改的官方 0a：先取中间 80% 再检查一秒短片段、neutral 形状映射、地板估计、速度边缘裁剪、30 Hz 重采样与反射，包括特殊 fps 修正规则。
3. 数据适配器已接入身体旋转/FK、相机与形状监督。0b 为相对 CPF 再丢掉第一帧，新适配器对身体同样只裁一次，记录原始帧 ID。短于 128 帧的 NPZ 保留，但不计完整窗口。0a 的手部/对齐数组边缘处理与身体不同；本次仅预测前 22 个身体关节。阶段 A 直接读取 0a NPZ，基线的 0b HDF5 另行处理。
4. 先完成全量文件校验，再运行当前代码/config/index 身份下的 CPU 与 CUDA 真实数据检查；CUDA 必须覆盖正式 batch 8。少量 pilot 的 `diagnostic.json` 不能启动正式训练，代码或配置变化会使既有检查失效。正式阶段从随机权重重新初始化，不使用诊断权重。
5. 阶段 A 权重单独保留。阶段 B 必须保留形状编码器、形状缓冲区及共同 pelvis 表示，再恢复场景分支训练；整个新 state-dict 不能直接严格加载到旧 `Native30SceneMI` 类。当前 stage-A 类明确拒绝场景输入，阶段 B 需要独立适配器及验证，尚未在此轮宣称完成。旧 192 帧训练器也不接受这份 YAML。

## 训练入口与运行记录

训练与诊断共用同一数据、损失、优化器和更新函数。只将允许的观测送入模型；完整动作和形状标签在损失计算时读取。记录源 family、反射、窗口、模式、噪声 timestep 和随机状态；checkpoint 仅保存一个原子 `training/last.pt`，同时保留完整优化器状态。`--resume` 要求代码、配置、数据索引及设备身份一致，恢复时截去未提交的日志尾部。

`READINESS.json` 的 `waiting_for_full_data` 表示仍在处理；`cpu_verified_gpu_preflight_pending` 表示尚需 GPU 检查；`training` 表示已经核实正式训练启动。此文件是最近核验的快照，实时更新数读取它指向的 `training/STATUS.json`。`gpu_admission` 保留实际启动准入结果。此前 CPU 就绪等待器已结束；现有完整索引已逐文件验过身份，修改训练代码后可使用它重新验收，无需重复全量索引扫描。**不会自动等待并占用他人的 GPU。**

以下命令使用训练环境 `/home/wenxin/miniconda3/envs/scenemi/bin/python`；先激活该环境再执行。已有后台任务时不要重复启动。

```bash
# 已有完整索引时重新验收训练代码和配置，不重建数据索引。
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 10 --cpu-count 2 -- \
  env CUDA_VISIBLE_DEVICES= python -u -m scenemi amass preflight \
  --paths configs/paths.local.yaml --index runs/amass128/index/complete.json --device cpu

# 用户明确授权共享 GPU 后，以配置中的分配上限和余量执行。
python -m scenemi.resources run --config configs/amass128.yaml \
  --device cuda --memory-gib 12 --cpu-count 4 -- \
  python -u -m scenemi amass preflight --paths configs/paths.local.yaml \
  --index runs/amass128/index/complete.json --device cuda

python -m scenemi.resources run --config configs/amass128.yaml \
  --device cuda --memory-gib 12 --cpu-count 4 -- \
  python -u -m scenemi amass train --paths configs/paths.local.yaml

# 中断后使用相同代码、配置和完整数据索引恢复权重、优化器与 RNG。
python -m scenemi.resources run --config configs/amass128.yaml \
  --device cuda --memory-gib 12 --cpu-count 4 -- \
  python -u -m scenemi amass train --paths configs/paths.local.yaml --resume
```

2026-10-22 主机死机后已核验 `training/last.pt` 的 ZIP CRC、封存源码与已提交日志，从第 46,000 步恢复。旧状态文件显示的 46,700 步包含未落盘更新，不作为恢复依据；未提交日志尾部由恢复入口截断。具体证据在本地 `training/RESTART_AUDIT.json` 与 `RESUMES.jsonl`，实时进度查看 `training/STATUS.json`、`updates.jsonl`。

验证监测固定 val 每子库一段，分别运行 EGO 连续相机、EXO 关节掩码、EXO 关节掩码叠加整帧缺失，共 12 条；DDIM 20 步、seed 777，每 1000 更新执行。EXO 两种条件使用相同源窗口、部位掩码和采样 seed，另报缺失帧 MPJPE；EGO 不制造缺帧。它用于快速监测，**不等同于完整 val/test 或论文评测协议**；训练期间不读取 test。正式 EGO/EXO 质量、手臂扭曲和缺失区域生成仍需训练后的分层评测与可视化。

打印待执行的上游命令（不会启动任务；示例环境路径须实际建立）：

```bash
python -m scenemi amass plan --paths configs/paths.local.yaml \
  --python /path/to/isolated/egoallo/bin/python
```

0a 从 EgoAllo 仓库工作目录运行，因为性别形状映射使用相对路径。CPU 新子进程可单独设 `CUDA_VISIBLE_DEVICES=`；0b 明确要求 CUDA。所有重任务仍通过资源入口，不绕过共享 GPU 准入。

全量任务已经完成，不要重复启动；以下仅保留低内存适配器的预处理入口，恢复操作与预算证据见 [长序列内存处理](data/AMASS_MEMORY_zh.md)：

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 16 --cpu-count 4 -- \
  /home/wenxin/projects/.venvs/amass-preprocess/bin/python -u -m scenemi \
  amass prepare --bounded-height-clustering --paths configs/paths.local.yaml \
  --inventory runs/amass128/raw_inventory.jsonl
```
