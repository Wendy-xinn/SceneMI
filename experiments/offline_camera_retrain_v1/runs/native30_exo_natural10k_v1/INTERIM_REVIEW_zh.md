# EXO 10k 中期结果

记录时间：2026-10-20T11:15:41.461446+08:00。仅固定100个EXO验证窗口，场景开启，clean/noisy自然VAL加权。不是完整VAL或TEST。

| 模型／步数 | 隐藏关节 MPJPE（mm，越低越好） |
| --- | ---: |
| 输入 reference | 117.54 |
| 未适配 control | 171.55 |
| 2000（已完成） | 126.17 |
| 4000（当前best） | 115.10 |
| 6000（已完成） | 116.93 |

当前best对control改善32.91%，对reference改善2.07%。6k验证较4k略退，训练总loss仍缓降。

| 数据集 | 条件 | reference hidden | control hidden | best4k hidden（mm） |
| --- | --- | ---: | ---: | ---: |
| rich | clean | 36.23 | 61.25 | 41.80 |
| rich | noisy | 60.06 | 101.44 | 117.48 |
| trumans | clean | 171.33 | 222.81 | 156.56 |
| trumans | noisy | 198.40 | 287.10 | 197.25 |
| egobody | clean | 66.54 | 107.38 | 64.44 |
| egobody | noisy | 86.97 | 136.80 | 76.70 |

三个已完成里程碑的9个EGO探针×clean/noisy=18项均逐元素保持。可见关节与GT支撑阶段脚部速度代理指标在4k/6k改善，但未证明消除漂浮、穿模或腕部抖动。RICH noisy隐藏关节尚未改善，不能宣布各域有效。

10:59 WSL/Linux环境重启，trace止于6044，完整断点在6000。114份历史源码、配置、优化器、RNG、全部采样状态及模型张量已由独立CPU严格核对，恢复原10k实验，未提交44步会重算。实时进度以STATUS.json/JOB.json为准；本记录不会改变best选择。

恢复核验：已重新产生6001步之后的真实训练更新，状态为training。与重启前未提交尾部逐项重放核对见RESUME_VERIFY.json；4k best不变。
