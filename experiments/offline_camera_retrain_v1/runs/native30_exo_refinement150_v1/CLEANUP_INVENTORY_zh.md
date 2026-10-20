本次未采用 EXO 短训的清理清单

状态：盘点完成，尚未删除。候选共 176 文件、1,121,319,156 字节（1.044 GiB）。

| 路径 | 文件数 | 字节 | 建议 |
|---|---:|---:|---|
| control | 1 | 519,867,891 | 已失败、无复用计划；授权后删除 |
| nondegradation | 1 | 519,867,891 | 已失败、无复用计划；授权后删除 |
| paired_validation/motions | 12 | 5,784,758 | 已失败、无复用计划；授权后删除 |
| paired_validation_cuda/motions | 162 | 75,798,616 | 已失败、无复用计划；授权后删除 |

逐文件路径、SHA256、inode、硬链接计数以及引用详见 CLEANUP_INVENTORY.json。两份权重均与 RESULT.json 一致；174 份数组均与各自 rows.jsonl 的哈希一致，无孤立数组。

原10k完整 SHA256 已复核为 3247383297ca2d7129755c6cd96c82eb2009cb2758aa8e22b6e5d0acb04c66b5；文件保留。此前严格加载与 base10k 配对/CPU复核证据保留；本次仅做字节校验，没有加载模型或使用GPU。

只读 host /proc 检查了31个进程，未发现 trial worker、evaluator、viser 或打开的 trial 文件。静态外部引用仅为复用模块默认路径和暂停的选择决策，没有候选晋升或活动viewer菜单依赖。

guard_review_window_step010000 内只有2份 JSON/MD 记录，无数组，整个目录保留。

保留 45 份紧凑试验记录，包括配置、TRAIN窗口/种子/源hash、两臂trace、rows、CPU审计、指标、选择/结论及错误记录。历史审计均为清理前证据；清理不改其原始校验值。

最终删除结果及磁盘前后占用须记入 CLEANUP.json；本清单本身不表示已执行清理。
