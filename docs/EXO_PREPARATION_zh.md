# EXO 数据与下一阶段训练准备

本文描述三来源 EXO 数据准备；AMASS EXO 适配已独立完成 15,000 步，见 `runs/amass_exo128/training/STATUS.json`，这一轮只使用 AMASS。TRUMANS 移动相机全量 train/validation 已于北京时间 2026-10-26 21:34 完成，502 条、1,098,057 帧，清单逐条匹配原计划。三来源 EXO 训练接口与最终质量筛选尚待验收，数据导出完成不表示该阶段已经开训。

## 相机来源

| 数据集 | 当前准备采用的 EXO | 说明 |
| --- | --- | --- |
| EgoBody | 保留真实标定的固定 Kinect 相机 | 固定相机也是有效的实际视频输入，不为形式统一而修改标定 |
| TRUMANS | 恢复的逐帧移动外参＋声明的合成内参/光学轴 | 全量 train/validation 已完成，不冒充原 RGB 完整标定 |
| GRAIL | 暂保留现有合成固定观察者 | 不是已恢复的原视频标定；移动增强须另行定义轨迹并重新做三维首命中，不能通过移动旧视频画面实现 |

用户随后决定 GRAIL 的 EXO 与 EGO 对齐动作和时间选择；后续阶段统一合同为 `configs/scene_data_current.json`。train 为 7,411 条、保留序列总计 2,170,711 帧、其中 1,563,865 帧可参与采样；val 质量检查后为 1,096 条、保留序列总计 300,496 帧、其中可用 222,429 帧。EXO 使用自身观测，不复制 EGO 的合成椅子补充点。旧 `configs/grail_exo_selection.yaml` 和 `runs/exo_preparation/grail_ego_aligned/` 保留原记录，其中 val 的 1,663 条、468,313 帧属于历史未筛选版本，不能作为后续新阶段的默认选择。

用户随后明确要求 GRAIL val/test 同样检查动作与场景的质量。既有接触和脚底支撑规则的全量复核入口为 `scenemi.data.grail_split_quality`，配置 `configs/grail_split_quality.json`，输出 `runs/grail_heldout_quality_v1/full/`；不按相机可见性删除干净动作，不添加椅子，不改原始划分或历史评测。复核后的时间选择共用于 EGO/EXO，使用时必须显式引用新版本；旧对齐清单和旧评测记录保留。详见 [质量协议](GRAIL_HELDOUT_QUALITY_zh.md)。

后续数据入口为 `scenemi.data.checked_scene_dataset.SceneSelection`，启动前必须通过 `CURRENT_SELECTION_CHECK.json` 的清单身份、帧数和 EGO/EXO GRAIL 时间范围校验；EXO 的 TRUMANS 指向完成后的移动相机 overlay。此为源数据选择检查，三来源 EXO 的完整模型观测适配仍待验收，不能由此声称已就绪或已开训。

现有 GRAIL 相机是对准人物起点附近的固定斜侧机位，不按整段动作范围居中，也不保证完整人体/场景始终入画。是否存在没有人体参考像素、没有可用关节的帧，见上述 `STATUS.json` 的逐划分统计；不能仅从展示视频的观看视角推断训练传感器的覆盖。

不同数据集无需都采用移动相机。需要统一的是世界坐标、相机位姿方向、时间戳、米制尺度、可见性与置信度含义。若为 GRAIL 增加移动视角，那是同一重建动作的新合成观测，与原始固定视频的 tracker 结果不能直接配对。

## 全量 TRUMANS 范围

仅处理 train/validation：469＋33 条录制，共 1,098,057 帧（30 Hz）。test 不进入本次准备。预计新增数组约 9.28 GiB；EGO、动作和原生人体等未变数组通过显式路径与 SHA256 借用原导出，避免重复复制大型缓存。

入口为 `python -m scenemi exo-camera plan|run`，实现位于 `scenemi/data/exo_export.py`。逐段前向 SMPL-X，逐帧重算静态场景、源动态物体与人体首命中；每一帧都核对人体关节与旧导出的世界坐标，并做首命中点的相机投影回环检查。可见关节仍为真值位置经遮挡遮罩后的模拟，还未加真实估计误差。

一条完整训练录制已验收：2,684 帧，人体关节与旧世界坐标导出最大误差为 0，投影回环最大误差约 0.000097 px；缺失位置/置信度置零、环境不含目标皮肤、遮挡计数分解及借用/新增文件哈希均通过。另有七项自动检查通过。记录见 `runs/exo_preparation/trumans_moving_pilot/ACCEPTANCE.json`。

全量版本已启动，通过 6 GiB RAM、两个 CPU 核的资源限额运行，不使用 GPU。首条完整录制含加载约耗时 48.9 秒、约 55 帧/秒，按此粗估全量约 5.5 小时；后续场景复杂度、磁盘访问和共享主机负载可能改变速度，以实时状态为准。完成的录制按内容哈希恢复；最终元数据原子提交，未完成的数组不会作为已完成序列加入清单。代码、配置或源清单改变时拒绝在同目录续跑。

```bash
python -m scenemi exo-camera plan --native-manifest experiments/offline_camera_retrain_v1/data/native30_official_memory_v3/manifest.jsonl --output runs/exo_preparation/trumans_moving
python -m experiments.offline_camera_retrain_v1.ops.resources run --device cpu --memory-gib 6 --cpu-count 2 -- python -m scenemi exo-camera run --native-manifest experiments/offline_camera_retrain_v1/data/native30_official_memory_v3/manifest.jsonl --camera-config configs/trumans_exo_moving.yaml --output runs/exo_preparation/trumans_moving
```

准备项快照：`runs/exo_preparation/STATUS.json`；完整录制验收：`runs/exo_preparation/trumans_moving_pilot/STATUS.json`；**全量导出实时进度**：`runs/exo_preparation/trumans_moving/STATUS.json`。全量清单完成前不能直接当作训练就绪。当前正在执行的导出源码封存于全量目录的 `PROTOCOL.json`，完成前不修改其依赖文件。

### 原生几何扩展异常后的恢复

北京时间 2026-10-26 14:31:29，长进程因 `pybind.cpython-39-x86_64-linux-gnu.so` 段错误（内核 signal 11，外层退出码 245）退出，未进入 Python 异常处理，状态因此停留在 `preparing`。发现时已完成 84 条、180,584 帧；第 85 条仅有未提交数组，不作为已完成数据。内核证据、旧状态、原日志和哈希保存在 `trumans_moving/recovery/INTERRUPTION_001.json` 等记录中。

恢复入口改为 `scenemi.data.exo_workers`：保留原导出源码、相机配置和已完成元数据；先逐个校验旧记录的文件哈希与输入语义，再让每条新录制运行于一个独立几何子进程。原生扩展异常由协调进程记录，单条最多尝试两次；不跳过失败数据。只有原子提交的元数据、全部哈希和关节/遮挡/环境语义都通过才接受完成。独立进程的启动会增加少量时间，以恢复后的实时估计为准。

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run --device cpu --memory-gib 6 --cpu-count 2 -- python -m scenemi.data.exo_workers resume --output runs/exo_preparation/trumans_moving
```

该恢复已启动。**不要并行运行旧长进程入口或再次启动协调进程。** 仍用同一个 `STATUS.json` 查看进度；`coordinator_pid`、`worker_pid`、`verified_sequences`、`worker_attempt`、`worker_incidents` 区分校验、重算和原生异常。恢复代码另行封存在 `recovery/PROTOCOL.json`，不覆盖原 `PROTOCOL.json`。

恢复的完整哈希扫描发现，原 84 条中有 9 条的部分衍生数组与提交时的哈希不一致；它们已从完成清单移出，元数据和异常文件保存在 `recovery/quarantine/`，安排完整重算。原始数据的借用数组哈希均通过。实际保留并通过哈希及观测语义复核的是 **75 条、161,111 帧**。第一条隔离录制已由独立进程重算并再次通过校验，目前恢复流程已继续推进。证据：`OUTPUT_INTEGRITY_SCAN.json`、`RECOVERY_ACCEPTANCE.json`。不将旧的“84 条已提交”直接等同于“84 条全部可用”。

15:22 再次停止于 `2023-01-15@22-17-15`：本地 `motion_world` 读取镜像与既定 SHA256 不符，原路径文件的哈希正确。通过 `scenemi.data.source_cache` 核查本次导出需要的全部 3,382 个不同输入数组：2,879 份已有镜像正确、1 份从验证过的原文件恢复、502 份缺少的移动外参镜像重新建立；原数据不写入。检查和异常字节证据见 `recovery/source_cache_audit/`。

后续恢复校验还发现读路径异常：一份 Linux 输出通过直接 I/O 读取时符合原哈希，普通读取不符；只对该文件重新验证读取缓存后，普通读取也恢复为原哈希，未重写文件。另有一份借用的 EGO 点云在原路径持续读到不同哈希，但已有本地镜像完全符合既定哈希；仅将该条 EXO 元数据中的借用路径改为此镜像，保留原元数据和原数据文件。没有更换期望哈希、放宽校验或改变观测数值。证据见 `OUTPUT_INTEGRITY_SCAN_002.json`、`READ_PATH_RECHECK.json`、`read_path_failure_20261026T074127Z/MIRROR_REBIND.json`。这些现象尚不能确定底层原因；发生哈希异常仍须停止，不能将文件的物理损坏作为未经验证的结论。

同日 18:51:52 后导出再次中断，最后提交 296 条、701,850 帧；内核记录 18:53:35 出现 WSL `Init has exited. Terminating distribution`，随后环境重新启动。旧状态、清单、日志与启动标识保存于 `recovery/interruption_20261026T105843Z/`。18:58 通过同一资源限额和封存恢复入口重新启动：完整复核 296 条已提交记录，再从未提交的 `2023-02-14@12-16-02` 重算。没有跳过录制，也没有更改相机、遮挡或划分协议。实时校验和处理状态仍以全量目录的 `STATUS.json` 为准。

该轮启动核查确认 295 条、700,325 帧通过；末条 `2023-02-14@12-14-59` 的元数据文件为零字节，不能算作有效提交。将该条衍生输出整体移至上述中断证据目录的 `incomplete_commit/`，保留文件身份后安排完整重算，同时重算原未完成的下一条。19:03 再次启动封存的恢复入口。不能仅凭重启前的 296 条计数宣称这些数据都已持久保存。

## 开训前仍需完成的接口验收

- 当前 `SceneDataset` 与场景训练配置仅支持 EGO；新增 EXO 适配入口，保持原封存实现和 EGO 基线可复核。EXO 使用允许的估计 3D 关节接口，不提供真值旋转或个人体形。
- 检查输入世界坐标、外参、原生 SMPL/SMPL-X/G1 形状解码及根轨迹的对齐；不能将相机运动误作人体运动。
- 明确 EXO 的时间相关噪声、深度/根轨迹漂移与连续跟踪缺失模拟；EGO 相机轨迹继续连续。几何遮挡模拟与真实 tracker 误差分别记录。
- 建立 EXO 的独立验证清单和覆盖统计；EGO 继续保留既有固定评测清单，避免混训时掩盖 EGO 退化。当前数据分布文档中的 EGO 筛选规模不代表 EXO 就绪规模。
- 从原无场景 30k 检查点验收真实样本的前向、反向、保存/恢复与显存占用，再确定正式 EGO/EXO 适配配置。场景收益实验在观测适配后进行，不能把 EXO 输入变化造成的收益当作场景收益。

当前尚未启动本文的三来源 EXO 训练，也未预设该阶段混训比例与更新次数；已运行的 AMASS EXO 适配单独记录。
