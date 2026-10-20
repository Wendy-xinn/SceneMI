# EXO 参考旋转修正范围 CPU 对照

阶段8000；范围unobserved；仅处理已保存的 EXO 输出；不训练、不重跑 DDIM/IK，不自动选最终方案。

| 面板 | 数据集 | 输入 | 上限° | hidden mm | visible mm | visible守卫 | foot守卫 |
|---|---|---|---:|---:|---:|---|---|
| fixed | rich | clean | 原输出 | 40.40 | 3.75 | True | True |
| fixed | rich | clean | 30.0 | 39.96 | 3.75 | True | True |
| fixed | rich | clean | 60.0 | 40.40 | 3.75 | True | True |
| fixed | rich | clean | 90.0 | 40.40 | 3.75 | True | True |
| fixed | rich | noisy | 原输出 | 115.08 | 31.16 | True | True |
| fixed | rich | noisy | 30.0 | 88.54 | 31.16 | True | False |
| fixed | rich | noisy | 60.0 | 96.42 | 31.16 | True | False |
| fixed | rich | noisy | 90.0 | 103.41 | 31.16 | True | False |
| fixed | egobody | clean | 原输出 | 62.87 | 5.04 | True | True |
| fixed | egobody | clean | 30.0 | 59.19 | 5.04 | True | True |
| fixed | egobody | clean | 60.0 | 59.57 | 5.04 | True | True |
| fixed | egobody | clean | 90.0 | 61.22 | 5.04 | True | True |
| fixed | egobody | noisy | 原输出 | 78.90 | 27.78 | True | True |
| fixed | egobody | noisy | 30.0 | 77.47 | 27.78 | True | True |
| fixed | egobody | noisy | 60.0 | 77.89 | 27.78 | True | True |
| fixed | egobody | noisy | 90.0 | 78.45 | 27.78 | True | True |
| fixed | trumans | clean | 原输出 | 153.85 | 7.02 | True | True |
| fixed | trumans | clean | 30.0 | 154.23 | 7.02 | True | True |
| fixed | trumans | clean | 60.0 | 152.22 | 7.02 | True | True |
| fixed | trumans | clean | 90.0 | 151.12 | 7.02 | True | True |
| fixed | trumans | noisy | 原输出 | 190.53 | 29.87 | True | True |
| fixed | trumans | noisy | 30.0 | 184.28 | 29.87 | True | True |
| fixed | trumans | noisy | 60.0 | 186.88 | 29.87 | True | True |
| fixed | trumans | noisy | 90.0 | 186.12 | 29.87 | True | True |
| rotating | rich | clean | 原输出 | 59.39 | 2.31 | True | True |
| rotating | rich | clean | 30.0 | 55.37 | 2.31 | True | True |
| rotating | rich | clean | 60.0 | 59.35 | 2.31 | True | True |
| rotating | rich | clean | 90.0 | 59.39 | 2.31 | True | True |
| rotating | rich | noisy | 原输出 | 94.21 | 23.45 | True | True |
| rotating | rich | noisy | 30.0 | 87.40 | 23.45 | True | True |
| rotating | rich | noisy | 60.0 | 92.80 | 23.45 | True | True |
| rotating | rich | noisy | 90.0 | 94.21 | 23.45 | True | True |
| rotating | egobody | clean | 原输出 | 62.72 | 4.41 | True | True |
| rotating | egobody | clean | 30.0 | 58.19 | 4.41 | True | True |
| rotating | egobody | clean | 60.0 | 58.28 | 4.41 | True | True |
| rotating | egobody | clean | 90.0 | 60.31 | 4.41 | True | True |
| rotating | egobody | noisy | 原输出 | 83.34 | 27.87 | True | True |
| rotating | egobody | noisy | 30.0 | 80.98 | 27.87 | True | True |
| rotating | egobody | noisy | 60.0 | 82.40 | 27.87 | True | True |
| rotating | egobody | noisy | 90.0 | 83.04 | 27.87 | True | True |
| rotating | trumans | clean | 原输出 | 154.59 | 7.41 | True | True |
| rotating | trumans | clean | 30.0 | 159.13 | 7.41 | True | False |
| rotating | trumans | clean | 60.0 | 155.97 | 7.41 | True | False |
| rotating | trumans | clean | 90.0 | 154.65 | 7.41 | True | True |
| rotating | trumans | noisy | 原输出 | 199.68 | 32.17 | True | True |
| rotating | trumans | noisy | 30.0 | 192.38 | 32.17 | True | True |
| rotating | trumans | noisy | 60.0 | 195.38 | 32.17 | True | True |
| rotating | trumans | noisy | 90.0 | 198.29 | 32.17 | True | True |

真实需要大幅修正的子集、P90、逐窗退化数、源身份与哈希见 REVIEW.json。
范围相对于参考局部旋转，不是限制膝盖的绝对弯曲角。GT仅用于指标与分组，投影函数不读取GT。
unobserved仅使用输入观测元数据保护可见位置和全局旋转；按帧mask切换可能引入跳动，速度和加速度误差已记录。
