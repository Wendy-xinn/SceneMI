# 扩大 VAL 双 checkpoint 配对复核

状态：partial，4/1224 次预测。没有训练或自动替换 best。

RICH 为全部206窗/27个录制；TRUMANS、EgoBody各50窗，实际独立录制数另列。每窗 clean/noisy、原41k/最终55k成对评价。
输入固定头部观测、可见场景、已知体型；30Hz/192帧/DDIM20，原物理修正开启。TEST不参与。

| 模型 | 数据 | 输入 | 窗数 | 录制数 | World录制宏均值mm | PA录制宏均值mm | GT低速3D脚速m/s |
|---|---|---|---:|---:|---:|---:|---:|
| candidate41k | rich | clean | 1 | 1 | 136.594 | 78.940 | 0.450 |
| candidate41k | rich | noisy | 1 | 1 | 145.355 | 85.553 | 0.419 |
| final55k | rich | clean | 1 | 1 | 139.379 | 76.602 | 0.416 |
| final55k | rich | noisy | 1 | 1 | 141.785 | 80.311 | 0.399 |

主判断是RICH27录制WorldMPJPE宏均值的clean/noisy平均；同时检查脚速、双足运动RMS误差和stride误差。
部分结果不用于选checkpoint；GT低速只是运动学代理，不能作为真实接触或全mesh穿模率。完整输出须CPU_RECHECK通过。
