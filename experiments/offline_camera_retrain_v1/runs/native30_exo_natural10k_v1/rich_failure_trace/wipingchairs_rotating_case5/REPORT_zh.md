补查非瑜伽案例 LectureHall_020_wipingchairs1，rotating step8000 case5，start704，原始source_first709，seed2751936227。只做带噪输入的 CPU 同初始噪声场景配对，零优化器更新，CUDA未初始化，4线程，峰值RSS2.01GiB，耗时33.6秒。轨迹和meta与历史GPU数组完全一致，参考最大特征差2.09e-6。

| 输入／预测 | 遮挡MPJPE mm | 可见MPJPE mm |
|---|---:|---:|
| 输入参考 | 144.68 | — |
| CPU sceneOFF | 280.09 | 22.42 |
| CPU sceneON | 302.07 | 21.64 |

该例关场景之后仍显著差于输入参考，因此RICH失败不是只有瑜伽案例，也不能全归因于场景。在sceneOFF里，左右膝局部旋转GT误差中位数从第一轮核心预测约37.7°/26.3°扩大到最后56.8°/52.3°，左右踝误差约293.0/369.8毫米；ON最后约324.4/393.7毫米。该证据支持先检查参考保真与扩散补偿稳定性，同时审查场景条件的个别灾难性触发。

历史GPU ON为367.42毫米，本轮CPU ON为302.07毫米，输出全身平均关节距离25.26毫米。CPU/CUDA即使同数字seed也不保证相同初始噪声，故不宣称逐值复现历史GPU。第二例没有另做observer/no-observer验证；原瑜伽案例的同代码验证单列在父目录observer_replay。

完整统计和预测仅本地保留。供备份记录：本目录SUMMARY.json、REPORT_zh.md、INVOCATION.txt。
