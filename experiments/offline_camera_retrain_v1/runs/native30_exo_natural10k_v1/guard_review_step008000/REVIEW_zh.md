# EXO 参考旋转修正范围 CPU 对照

阶段8000；仅处理已保存的 EXO 输出；不训练、不重跑 DDIM/IK，不自动选最终方案。

| 面板 | 数据集 | 输入 | 上限° | hidden mm | visible mm | visible守卫 | foot守卫 |
|---|---|---|---:|---:|---:|---|---|
| fixed | rich | clean | 原输出 | 40.40 | 3.75 | True | True |
| fixed | rich | clean | 30.0 | 40.22 | 4.85 | False | True |
| fixed | rich | clean | 60.0 | 40.56 | 4.16 | True | True |
| fixed | rich | clean | 90.0 | 40.42 | 3.80 | True | True |
| fixed | rich | noisy | 原输出 | 115.08 | 31.16 | True | True |
| fixed | rich | noisy | 30.0 | 77.33 | 31.79 | True | True |
| fixed | rich | noisy | 60.0 | 91.73 | 30.02 | True | True |
| fixed | rich | noisy | 90.0 | 101.15 | 30.31 | True | True |
| fixed | egobody | clean | 原输出 | 62.87 | 5.04 | True | True |
| fixed | egobody | clean | 30.0 | 59.66 | 10.43 | False | True |
| fixed | egobody | clean | 60.0 | 59.59 | 6.01 | True | True |
| fixed | egobody | clean | 90.0 | 61.22 | 5.07 | True | True |
| fixed | egobody | noisy | 原输出 | 78.90 | 27.78 | True | True |
| fixed | egobody | noisy | 30.0 | 77.67 | 32.23 | False | True |
| fixed | egobody | noisy | 60.0 | 77.82 | 27.89 | True | True |
| fixed | egobody | noisy | 90.0 | 78.44 | 27.77 | True | True |
| fixed | trumans | clean | 原输出 | 153.85 | 7.02 | True | True |
| fixed | trumans | clean | 30.0 | 156.82 | 14.13 | False | True |
| fixed | trumans | clean | 60.0 | 153.15 | 9.23 | False | True |
| fixed | trumans | clean | 90.0 | 151.16 | 7.45 | True | True |
| fixed | trumans | noisy | 原输出 | 190.53 | 29.87 | True | True |
| fixed | trumans | noisy | 30.0 | 185.78 | 35.73 | False | True |
| fixed | trumans | noisy | 60.0 | 186.96 | 30.13 | True | True |
| fixed | trumans | noisy | 90.0 | 186.08 | 29.80 | True | True |
| rotating | rich | clean | 原输出 | 59.39 | 2.31 | True | True |
| rotating | rich | clean | 30.0 | 55.40 | 2.42 | True | True |
| rotating | rich | clean | 60.0 | 59.35 | 2.31 | True | True |
| rotating | rich | clean | 90.0 | 59.39 | 2.31 | True | True |
| rotating | rich | noisy | 原输出 | 94.21 | 23.45 | True | True |
| rotating | rich | noisy | 30.0 | 87.02 | 23.83 | True | True |
| rotating | rich | noisy | 60.0 | 92.35 | 23.34 | True | True |
| rotating | rich | noisy | 90.0 | 94.16 | 23.44 | True | True |
| rotating | egobody | clean | 原输出 | 62.72 | 4.41 | True | True |
| rotating | egobody | clean | 30.0 | 58.80 | 9.85 | False | True |
| rotating | egobody | clean | 60.0 | 58.29 | 4.71 | True | True |
| rotating | egobody | clean | 90.0 | 60.30 | 4.40 | True | True |
| rotating | egobody | noisy | 原输出 | 83.34 | 27.87 | True | True |
| rotating | egobody | noisy | 30.0 | 81.53 | 32.97 | False | True |
| rotating | egobody | noisy | 60.0 | 82.38 | 27.91 | True | True |
| rotating | egobody | noisy | 90.0 | 83.03 | 27.87 | True | True |
| rotating | trumans | clean | 原输出 | 154.59 | 7.41 | True | True |
| rotating | trumans | clean | 30.0 | 161.37 | 14.54 | False | False |
| rotating | trumans | clean | 60.0 | 156.55 | 10.03 | False | False |
| rotating | trumans | clean | 90.0 | 154.71 | 7.78 | True | True |
| rotating | trumans | noisy | 原输出 | 199.68 | 32.17 | True | True |
| rotating | trumans | noisy | 30.0 | 194.24 | 38.13 | False | True |
| rotating | trumans | noisy | 60.0 | 195.44 | 32.98 | True | True |
| rotating | trumans | noisy | 90.0 | 198.28 | 32.20 | True | True |

真实需要大幅修正的子集、P90、逐窗退化数、源身份与哈希见 REVIEW.json。
范围相对于参考局部旋转，不是限制膝盖的绝对弯曲角。GT仅用于指标与分组，投影函数不读取GT。
