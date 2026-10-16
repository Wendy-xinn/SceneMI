# SceneMI：头部轨迹与可见场景条件的动作生成

当前主线为30 Hz、192帧窗口的 reference-residual 扩散，恢复逐脚步态监督，持续目标55,000步。旧对照为 `runs/native_dynamic_scene20_contact_55k_oct07`，不是其他名字带55k的目录。

[当前实验、训练进度、best和评估入口](runs/INDEX_zh.md) · [方法有效性证据](ops/EVIDENCE_STATUS_zh.md) · [TensorBoard](http://localhost:6006/#custom_scalars) · [Viser动作回放](http://localhost:8800)

## 统一命令

在 SceneMI 仓库根目录，使用 `scenemi` 环境：

```bash
python -m experiments.offline_camera_retrain_v1.ops status
python -m experiments.offline_camera_retrain_v1.ops view
python -m experiments.offline_camera_retrain_v1.ops best --help
python -m experiments.offline_camera_retrain_v1.ops scene-check --help
```

后台已经有训练、best守护和登记的评估任务，查看状态即可。Viser读取已有预测，不执行新推理；默认打开已经冻结的案例。新增案例修改 `runs/native30_live_monitor_oct16/visuals/viser/CASES.json`，用 `view --manifest <案例清单>` 重新准备/启动同一个查看器，不复制Python文件。

## 目录职责

| 目录 | 内容 |
|---|---|
| `ops/` | 可复用的运维、评估复核和可视化工具；通过模块CLI调用 |
| `runs/` | 当前训练/评估的数据、配置、指标和本地模型；从索引进入 |
| `data/`、`configs/` | 数据入口与配置资产；不作为失败实验输出清理 |
| `archive/` | 已退役实验与历史记录；精确恢复路径随清理记录保存 |
| 根目录Python模块 | 当前封存实现和历史遗留实现，暂不可随意移动 |

此前根README描述的5 Hz pilot与旧实验索引已原样移到 [历史记录](archive/retired_runs_oct16/root_records/)，不再作为当前运行说明。当前配置以各run内部的冻结配置为准。

## 根源码整理的实际限制

旧准备程序 `code_hashes()` 将根目录全部352份Python文件纳入359项源码封存，并没有只收集真实依赖。直接删旧脚本会破坏当前训练恢复、best保存和全RICH评估的校验，因此本轮没有改动这些源码，也没有通过隐藏文件来冒充清理。

已经删除14份结束日志，归档24份旧配置/记录/启动文件。[执行记录](ops/cleanup_oct16_evening/ROOT_NONPY_EXECUTED.json)。106份明确关联退役实验的脚本已列入 [后续归档清单](ops/cleanup_oct16_evening/ROOT_DEFERRED_ARCHIVE_PLAN.json)。下一次实现版本应按显式依赖集合封存，保留当前完整源码恢复包后再移出旧脚本；不改写当前实验身份。[依赖审计](ops/cleanup_oct16_evening/ROOT_CLEANUP_REVIEW_zh.md)。
