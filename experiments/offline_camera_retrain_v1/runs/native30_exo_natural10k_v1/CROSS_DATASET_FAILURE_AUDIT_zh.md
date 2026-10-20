# EXO 8k 跨数据集失效排查

仅以 CPU 逐窗读取既有 fixed 与 step-8000 rotating VAL 的 reference、prediction、GT、observations；无新推理或训练。正式固定和轮换各100个EXO源窗口，每个包含clean/noisy；固定case109历史左手探针单列。空隐藏掩码从对应指标排除；不是全VAL/TEST结论。

| 面板 | 数据集 | 输入 | 有隐藏点窗口数 | 隐藏误差 reference→8k (mm) | 恶化窗口 | 恶化>50mm | 可见误差 reference→8k (mm) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| fixed | rich | clean | 11 | 36.23 → 40.40 | 8 | 0 | 2.57 → 3.75 |
| fixed | rich | noisy | 12 | 60.06 → 115.08 | 9 | 3 | 36.02 → 31.16 |
| fixed | egobody | clean | 45 | 66.54 → 62.87 | 24 | 0 | 20.30 → 5.04 |
| fixed | egobody | noisy | 46 | 86.97 → 78.90 | 15 | 1 | 48.21 → 27.78 |
| fixed | trumans | clean | 41 | 171.33 → 153.85 | 16 | 2 | 23.60 → 7.02 |
| fixed | trumans | noisy | 41 | 198.40 → 190.53 | 18 | 3 | 51.78 → 29.87 |
| rotating | rich | clean | 12 | 56.12 → 59.39 | 5 | 1 | 3.05 → 2.31 |
| rotating | rich | noisy | 12 | 89.01 → 94.21 | 4 | 1 | 33.06 → 23.45 |
| rotating | egobody | clean | 45 | 66.38 → 62.72 | 23 | 2 | 17.94 → 4.41 |
| rotating | egobody | noisy | 45 | 96.21 → 83.34 | 14 | 2 | 47.62 → 27.87 |
| rotating | trumans | clean | 41 | 182.07 → 154.59 | 13 | 2 | 21.95 → 7.41 |
| rotating | trumans | noisy | 41 | 215.05 → 199.68 | 13 | 5 | 53.11 → 32.17 |

## 窗口和序列去重

| 数据集 | fixed | rotating | 相同源窗口交集 | 联合唯一源窗口 | 联合录制 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rich | 12 | 12 | 0 | 24 | 20 |
| egobody | 47 | 47 | 0 | 94 | 17 |
| trumans | 41 | 41 | 0 | 82 | 31 |
| all | 100 | 100 | 0 | 200 | 68 |

## 输入参考原本较好时的退化

下表合并两个正式面板的观测，reference隐藏误差<100mm。这批fixed/rotating没有相同源窗口，但包含同序列或时间重叠窗口，不能理解为独立完整测试。

| 数据集/输入 | 观测窗口 | reference→8k 隐藏(mm) | 恶化窗口 | 恶化>50mm |
| --- | ---: | ---: | ---: | ---: |
| rich/clean | 19 | 28.93 → 31.47 | 12 | 0 |
| rich/noisy | 19 | 50.77 → 85.43 | 12 | 3 |
| egobody/clean | 67 | 26.64 → 25.67 | 40 | 0 |
| egobody/noisy | 63 | 51.82 → 49.71 | 25 | 2 |
| trumans/clean | 21 | 65.88 → 73.59 | 14 | 0 |
| trumans/noisy | 14 | 73.18 → 87.45 | 9 | 2 |

## 局部旋转大幅改坏的明确病例

严格筛选：排除root；相对reference的局部旋转改写中位数>90°，同时相对GT的局部旋转误差中位数增加>60°。用于发现过度改写，不把合理的大动作纠正判错。

| 面板/数据集/输入 | 序列 @起始帧 | 隐藏误差 reference→8k (mm) | 明显错误改写关节 |
| --- | --- | ---: | --- |
| fixed/rich/noisy | Pavallion_006_yoga1 @640 | 95.26 → 594.56 | left_knee, right_knee |
| rotating/trumans/noisy | 2023-02-18@16-39-58 @1664 | 174.79 → 377.92 | right_knee |
| rotating/egobody/clean | recording_20220315_S21_S30_06 @256 | 184.65 → 270.62 | left_knee, right_knee |
| rotating/egobody/clean | recording_20220315_S21_S30_02 @512 | 176.87 → 261.81 | left_knee, right_knee |
| fixed/trumans/noisy | 2023-02-17@10-58-16 @832 | 391.05 → 412.05 | left_knee, right_knee |
| fixed/trumans/clean | 2023-02-17@10-58-16 @832 | 302.65 → 304.81 | left_knee, right_knee |
| fixed/trumans/clean | 2023-02-18@16-46-38 @1088 | 283.72 → 261.01 | right_knee |
| fixed/trumans/noisy | 2023-02-18@16-46-38 @1088 | 337.00 → 300.12 | left_knee, right_knee |

## 证据边界

三数据集都出现隐藏误差恶化，EgoBody/TRUMANS均值仍改善，所以均值掩盖了部分明显尾部失败。不能把“残差”理解成天然有界的小补偿。

已保存逐例root-relative隐藏误差以区分整体平移和局部姿态失真。仅凭最终数组不能将原因归到场景、主干或置信度；需相同checkpoint/窗口/种子的逐去噪步与scene开关诊断。所有精确panel identity、seed、source_first、指标与文件SHA256见JSON。

NumPy采用supervision.py相同旋转/FK公式；逐窗MPJPE与既有JSON对照最大误差 0.000035 mm。CPU峰值RSS 44.0 MiB；未导入CUDA。

额外敏感性核验：只保留隐藏关节帧点数≥192的窗口，noisy隐藏恶化>50mm仍为RICH 2/13（2序列）、EgoBody 3/66（3序列）、TRUMANS 8/82（7序列）。最终保存的6D旋转也已核查并记录：严格局部改写病例的第一轴/正交第二轴范数未出现接近零退化；输出已经过旋转正规化，这不能排除更早阶段的数值问题。
