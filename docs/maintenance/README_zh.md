# 维护记录

## 2026-10-27 AMASS 数据完整性恢复

两条 AMASS 训练在北京时间 13:24 / 13:31 因预处理文件 SHA256 不一致停止。
`CMU/55/55_09_poses_reflect.npz` 和 `CMU/90/90_30_poses.npz` 同时存在 ZIP 成员 CRC
损坏；文件时间戳不能识别这类内容变化，原因尚未确定。
已核验原始 AMASS 数据和实际执行的原始预处理源码，用原 CPU 环境重建独立候选，
并以最初完整 SHA256 验证逐字节相同后原子恢复。损坏副本保留作审计证据。
未改 manifest、数据划分、训练配置或运行中封存源码，不通过跳过文件来恢复训练。

全部 train/val 的 15,644 个预处理文件另行只读扫描，独立 test 不读取。
检查点 CRC 和 98 / 99 个封存训练源文件核验通过；按各自原优化器及 RNG
从 scratch 9,500 / EXO-only 7,500 步恢复，未提交日志尾部由原训练器处理。

- [完整扫描进度与结果](../../runs/amass128/integrity_recovery/AUDIT.json)
- [原数据重建证据](../../runs/amass128/integrity_recovery/reprocessed/RESULT.json)
- [原子恢复记录](../../runs/amass128/integrity_recovery/RESTORE.json)
- [检查点与恢复记录](../../runs/amass128/integrity_recovery/RESTART.json)
- 扫描后复验结果：`runs/amass128/integrity_recovery/VERIFIED.json`

复用工具为 `scenemi/data/integrity.py`，CPU 重建仍通过原资源限额入口。
历史根 CLI 不由此工具执行，其封存差异在重建报告中明确记录；全部实际数据变换
依赖和原始数据仍须匹配，重建输出必须满足完整原始 SHA256。

同日下午完整性错误再次出现，涉及 Eyes Japan、BMLmovi、CMU 的其他文件。
不能根据资源管理器可打开目录判断 NPZ 全部成员有效，也尚不能确定是存储故障。
EGO55k 权重曾在训练中报哈希错误，但后续连续读取、直接读取和 ZIP CRC 均通过，
因此没有将该父权重替换成其他版本。

为隔离 `/mnt/e` 的读取路径，已建立
`runs/amass128/verified_data`：15,644 个 train/val 文件共 13,749,054,440 字节，
逐文件复制并再次读取，全部匹配原索引 SHA256；目录和文件设为只读。
发现的损坏文件使用原环境重建、完整 SHA256 一致的候选。
仅更改忽略提交的 `configs/paths.local.yaml` 中机器路径，不改训练配置、数据内容、
划分、指标或已封存源码。独立 test 数组没有读取或复制，其目录仅保留原路径链接。

边界编码器另外通过当前 EXO 10k 权重的真实 8 样本 CPU 前向、反向及优化一步检查：
原条件编码器、边界两层及注入层梯度均非零且有限，冻结主干无梯度且权重不变，
输入文件和正式训练检查点均未改变。观测关系的 `no_grad` 不阻止后续编码层学习。
6 项边界/联合协议测试通过。两条训练的外部文件调用跟踪在已记录区间未发现
以写模式打开输入；这些检查仍不代表已经确定所有完整性异常的根因。

- [本地副本完整校验](../../runs/amass128/integrity_recovery/recurrence/SNAPSHOT.json)
- [路径切换及只读设置](../../runs/amass128/integrity_recovery/recurrence/DATA_RELOCATION.json)
- [当前权重梯度检查](../../runs/amass128/integrity_recovery/recurrence/BOUNDARY_GRADIENT_AUDIT.json)
- [输入文件调用跟踪](../../runs/amass128/integrity_recovery/recurrence/TRACE_SUMMARY.json)
- [恢复与后续状态](../../runs/amass128/integrity_recovery/recurrence/STATUS.json)

## 2026-10-21 清理执行记录

用户明确授权“同意按清单永久删除”后，已按本地 `cleanup_plan.json` 执行；没有扩展删除范围。

| 项目 | 结果 |
|---|---:|
| 删除的精确文件 | 31,085 |
| 其中退役根源码 | 157 |
| 预计释放的实际分配空间 | 92.25 GiB |
| 删除后复验的保留依赖（源码及模型） | 212 |
| 全量保留的 run | 8 |
| 已清除生成资产、仅留记录的 run | 27 |
| 旧实验根 Python 文件 | 246 → 89 → 67 |

最后一次减少是将 22 个非封存回归测试迁移到顶层 `tests/legacy/`；不是另一次永久删除。一个仍在现有源码封印中的测试保留原路径。三个测试使用的旧配置从已有历史档案精确恢复为夹具；未用新协议替代旧协议。

## 精确证据

- [紧凑执行及复验摘要](cleanup_summary.json)：含全量清单 SHA256、执行记录 SHA256、释放空间和恢复包身份。
- [清理策略与保护集合](cleanup_policy.json)：全量保留目录、源文件保护身份、退役目录和退役源码。
- [现有 run 的分类](run_retention.json)：当前资产与历史小记录一一列明。
- 完整本地清单：`docs/maintenance/cleanup_plan.json`；逐文件执行报告：`docs/maintenance/cleanup_executed.json`。两份共约 29 MiB，包含 SHA256、inode、字节数等信息，留在本机且忽略提交；未只保留统计数。
- [源码恢复包说明](../../experiments/offline_camera_retrain_v1/archive/source_bundles/README_zh.md)。本轮删除的源文件可从同目录 `pre_amass_cleanup_source.tar.gz` 恢复：363 个成员、1,058,433 字节，SHA256 `b3d312755ecfea96c29da78fa15760bafe3ed92259e565b9d0da9d63ca83da06`。生成数组和权重没有为了恢复而另做副本。

## 保留依据

删除前对所有候选、保护文件和源码包进行 SHA256 校验，再按批准的逐文件列表 unlink；保留依赖在删除后再次复验。文件改变、路径越界或符号链接会阻止执行。工具为 `experiments/offline_camera_retrain_v1/ops/maintenance.py`，附清理前拒绝变更依赖、拒绝变更候选、保留指标与拒绝符号链接等测试。

保留当前 EXO 10k、EGO 41k/55k、选定 control、完整评测预测、候选评估、旧官方模型、原始数据、人体模型、场景缓存、接触标签和 GRAIL 任务。部分历史目录仍包含冻结代码和标定文件，因此保留路径与小记录；不把它们当作仍可恢复的模型。

旧报告、审计和 viewer 菜单描述清理前资产，不能据其中旧 `completed`、`resumable` 字段判断权重仍存在。当前查看入口和保留身份以 [实验索引](../../experiments/offline_camera_retrain_v1/runs/INDEX_zh.md) 为准。Linux 分区清理后约有 175 GiB 可用；正在运行的数据任务会继续改变余量。

维护与新代码组织要求已写入仓库 `AGENTS.md`，后续沿用同一配置、模块和记录规则。

## 2026-10-25 补充清理

用户再次明确授权清除不需要的旧资产，包括此前因缺少具体路径授权而被自动审核拒绝的无场景续训权重。已按精确清单删除 **74 个 checkpoint 文件/硬链接路径**，实际磁盘余量增加约 **75.78 GiB**，清理后可用约 **94.35 GiB**（系统 `df -h` 取整显示 95G）。

| 范围 | 删除路径数 | 保留 |
| --- | ---: | --- |
| `scene_ego128_local` | 31 | 选中 27k、最终可恢复 30k、全部选模指标 |
| `scene_ego128_noscene` | 31 | 选中 30k、最终可恢复 30k、全部选模指标 |
| `scene_ego128_residual` | 8 | 选中 3k、最终可恢复 8k、全部选模指标 |
| `scene_ego128_noscene_continue` | 4 | 被否决的高学习率重启试验只保留记录；此前被拒绝的就是这四条路径 |

AMASS 统一版 55k、EGO-only 选中 50k/最终 55k、旧正式模型、有效低学习率对照、当前五组实验、原始数据、人体模型、场景缓存及源码均保留。本轮没有删除 Python 源码；旧兼容层仍属于保留模型的复现保护范围。各 run 的 `ARTIFACT_RETENTION.json` 说明最新权重保留状态，历史候选清单和旧报告描述的是清理前集合。

删除前检查文件 SHA256、inode、大小、mtime、硬链接计数和直接配置引用。删除后 **113 个保护文件**及源码恢复包重新通过 SHA256 复验。清理工具的六项检查通过，覆盖依赖变化、候选变化、符号链接、精确路径与邻近文件保护。当前训练依赖未改变。

- [紧凑执行及复验摘要](artifact_cleanup_summary.json)。
- 本地精确策略、清单及逐文件执行报告：`artifact_cleanup_policy.json`、`artifact_cleanup_plan.json`、`artifact_cleanup_executed.json`；这些含机器路径和文件身份，留在本机，不提交。
- 继续复用 `ops/maintenance.py`，新增顶层 `runs/` 资产清理必须逐文件列举，不对整个 run 递归删除。
# 2026-10-27 AMASS 续训整理

用户授权退役误开的 warm 联合分支权重，以及已被完成模型替代的恢复快照。
本轮删除 3 个文件，释放约 2.40 GiB 磁盘空间；配置、逐更新日志、指标、原文件身份与哈希保留。
冻结 EGO 55k、已选 EXO 边界 42k、从零对照和当前续训依赖均保留。
不删除预处理数据、原生人体模型或移动相机缓存。

- [精确清单](amass_continuation_cleanup_plan.json)
- [依赖审计](amass_continuation_cleanup_dependencies.json)
- [执行记录](amass_continuation_cleanup_executed.json)
- [删除后核验](amass_continuation_cleanup_summary.json)

