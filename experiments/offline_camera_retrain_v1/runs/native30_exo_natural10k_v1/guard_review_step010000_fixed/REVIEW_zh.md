# EXO 参考旋转修正范围 CPU 对照

阶段10000；仅处理已保存的 EXO 输出；不训练、不重跑 DDIM/IK，不自动选最终方案。

| 面板 | 数据集 | 输入 | 上限° | hidden mm | visible mm | visible守卫 | foot守卫 |
|---|---|---|---:|---:|---:|---|---|
| fixed | rich | clean | 原输出 | 38.44 | 3.38 | True | True |
| fixed | rich | clean | 30.0 | 38.20 | 4.32 | True | True |
| fixed | rich | clean | 60.0 | 38.56 | 3.78 | True | True |
| fixed | rich | clean | 90.0 | 38.45 | 3.43 | True | True |
| fixed | rich | noisy | 原输出 | 114.18 | 30.20 | True | True |
| fixed | rich | noisy | 30.0 | 75.75 | 30.80 | True | True |
| fixed | rich | noisy | 60.0 | 89.69 | 28.90 | True | True |
| fixed | rich | noisy | 90.0 | 99.63 | 29.35 | True | True |
| fixed | egobody | clean | 原输出 | 61.86 | 4.95 | True | True |
| fixed | egobody | clean | 30.0 | 58.39 | 10.60 | False | True |
| fixed | egobody | clean | 60.0 | 58.12 | 5.99 | False | True |
| fixed | egobody | clean | 90.0 | 59.89 | 4.97 | True | True |
| fixed | egobody | noisy | 原输出 | 75.99 | 26.98 | True | True |
| fixed | egobody | noisy | 30.0 | 74.77 | 31.32 | False | True |
| fixed | egobody | noisy | 60.0 | 74.66 | 27.07 | True | True |
| fixed | egobody | noisy | 90.0 | 75.36 | 26.97 | True | True |
| fixed | trumans | clean | 原输出 | 155.48 | 7.03 | True | True |
| fixed | trumans | clean | 30.0 | 155.46 | 13.95 | False | True |
| fixed | trumans | clean | 60.0 | 153.28 | 9.28 | False | True |
| fixed | trumans | clean | 90.0 | 152.38 | 7.43 | True | True |
| fixed | trumans | noisy | 原输出 | 193.16 | 29.72 | True | True |
| fixed | trumans | noisy | 30.0 | 185.99 | 35.37 | False | True |
| fixed | trumans | noisy | 60.0 | 187.91 | 29.96 | True | True |
| fixed | trumans | noisy | 90.0 | 187.96 | 29.63 | True | True |

真实需要大幅修正的子集、P90、逐窗退化数、源身份与哈希见 REVIEW.json。
范围相对于参考局部旋转，不是限制膝盖的绝对弯曲角。GT仅用于指标与分组，投影函数不读取GT。
