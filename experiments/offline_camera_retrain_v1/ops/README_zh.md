# 既有实验运维入口

工具实现保留原路径和源码封印。统一从仓库根目录执行：

```bash
python -m scenemi ops --help
python -m scenemi ops status
python -m scenemi ops view --help
python -m experiments.offline_camera_retrain_v1.ops.grail_prepare --help
```

当前模型与报告从 [实验索引](../runs/INDEX_zh.md) 进入，新 AMASS 接口见顶层 [代码导航](../../../docs/CODE_MAP_zh.md)。

## 常用模块

| 模块 | 用途 |
|---|---|
| `resources` | 共享机器 GPU 准入、CPU/RAM 限额及有效限额核验 |
| `motion_viewer` / `ops view` | 读取已准备的预测、GT、场景；当前使用 final_metrics 的 PREPARED_CASES |
| `grail_prepare` | 可恢复的 GRAIL CPU 数据预处理；已有任务时不要重复启动 |
| `final_metric_summary`、`final_metric_recheck` | 已有完整指标的汇总与独立复算；具体 CLI 以各模块 --help 为准 |
| `rich_test` / `ops rich-test` | 官方独立 RICH TEST 的准备、评价与 CPU 复算 |
| `maintenance` | 明确清单的 plan / apply / verify；本轮结果在 docs/maintenance |

历史 trial、ablation、adapter 模块继续保留，因为现有模型/审计引用其实现；它们不是新的推荐训练入口。退役目录不再有权重和预测，旧报告中的 resume/菜单描述只作为清理前证据。不要用旧模块默认输出目录覆盖这些历史记录。

## 共享机器重任务

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources check
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 6 --cpu-count 2 -- <已有命令>
```

GPU 工作使用 `--device cuda`，检查被拒绝时不得绕过，也不自动排队抢占另一用户的任务。限额仅作用于新进程；设置失败不回退无限额执行。禁止重启 WSL、重置 GPU 或修改共享全局设置。

清理复验：

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 1 --cpu-count 2 -- \
  python -m experiments.offline_camera_retrain_v1.ops.maintenance verify \
  --plan docs/maintenance/cleanup_plan.json \
  --report docs/maintenance/cleanup_executed.json \
  --output docs/maintenance/cleanup_summary.json
```

完整账本留在本机，紧凑摘要纳入代码。后续案例使用 manifest，参数使用 config，新增通用能力放在命名清楚的现有模块内。
