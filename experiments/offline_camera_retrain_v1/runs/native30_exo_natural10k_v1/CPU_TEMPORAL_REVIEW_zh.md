# EXO 10k 保存结果 CPU 配对审计

best 步数：10000
固定窗口的输入、GT、reference 已核对；各里程碑 EGO 输出精确保持。

| 数据集 | 条件 | 对照 | best hidden mm | 对照 hidden mm | 变化 mm |
|---|---|---|---:|---:|---:|
| rich | clean | control | 38.438 | 61.248 | -22.810 |
| rich | clean | input_reference | 38.438 | 36.230 | 2.208 |
| trumans | clean | control | 155.482 | 222.811 | -67.329 |
| trumans | clean | input_reference | 155.482 | 171.333 | -15.851 |
| egobody | clean | control | 61.855 | 107.385 | -45.530 |
| egobody | clean | input_reference | 61.855 | 66.541 | -4.686 |
| natural_weighted | clean | control | 97.335 | 149.075 | -51.740 |
| natural_weighted | clean | input_reference | 97.335 | 105.765 | -8.430 |
| rich | noisy | control | 114.181 | 101.439 | 12.742 |
| rich | noisy | input_reference | 114.181 | 60.058 | 54.123 |
| trumans | noisy | control | 193.156 | 287.104 | -93.947 |
| trumans | noisy | input_reference | 193.156 | 198.403 | -5.246 |
| egobody | noisy | control | 75.991 | 136.805 | -60.814 |
| egobody | noisy | input_reference | 75.991 | 86.967 | -10.977 |
| natural_weighted | noisy | control | 128.404 | 194.025 | -65.620 |
| natural_weighted | noisy | input_reference | 128.404 | 129.310 | -0.906 |

速度、加速度、脚部与腕部时序、各窗口分母及源文件 SHA 见 CPU_TEMPORAL_REVIEW.json。
这是开发验证集的事后审计，未更改 best，不自动认定科学验收通过。
