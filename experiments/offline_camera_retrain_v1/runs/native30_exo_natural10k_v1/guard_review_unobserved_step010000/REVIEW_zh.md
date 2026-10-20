# EXO 参考旋转修正范围 CPU 对照

阶段10000；范围unobserved；仅处理已保存的 EXO 输出；不训练、不重跑 DDIM/IK，不自动选最终方案。

| 面板 | 数据集 | 输入 | 上限° | hidden mm | visible mm | visible守卫 | foot守卫 |
|---|---|---|---:|---:|---:|---|---|
| fixed | rich | clean | 原输出 | 38.44 | 3.38 | True | True |
| fixed | rich | clean | 30.0 | 38.01 | 3.38 | True | True |
| fixed | rich | clean | 60.0 | 38.42 | 3.38 | True | True |
| fixed | rich | clean | 90.0 | 38.44 | 3.38 | True | True |
| fixed | rich | noisy | 原输出 | 114.18 | 30.20 | True | True |
| fixed | rich | noisy | 30.0 | 86.14 | 30.20 | True | False |
| fixed | rich | noisy | 60.0 | 94.38 | 30.20 | True | False |
| fixed | rich | noisy | 90.0 | 101.83 | 30.20 | True | False |
| fixed | egobody | clean | 原输出 | 61.86 | 4.95 | True | True |
| fixed | egobody | clean | 30.0 | 57.87 | 4.95 | True | True |
| fixed | egobody | clean | 60.0 | 58.09 | 4.95 | True | True |
| fixed | egobody | clean | 90.0 | 59.90 | 4.95 | True | True |
| fixed | egobody | noisy | 原输出 | 75.99 | 26.98 | True | True |
| fixed | egobody | noisy | 30.0 | 74.48 | 26.98 | True | True |
| fixed | egobody | noisy | 60.0 | 74.76 | 26.98 | True | True |
| fixed | egobody | noisy | 90.0 | 75.38 | 26.98 | True | True |
| fixed | trumans | clean | 原输出 | 155.48 | 7.03 | True | True |
| fixed | trumans | clean | 30.0 | 152.85 | 7.03 | True | True |
| fixed | trumans | clean | 60.0 | 152.40 | 7.03 | True | True |
| fixed | trumans | clean | 90.0 | 152.37 | 7.03 | True | True |
| fixed | trumans | noisy | 原输出 | 193.16 | 29.72 | True | True |
| fixed | trumans | noisy | 30.0 | 184.50 | 29.72 | True | True |
| fixed | trumans | noisy | 60.0 | 187.90 | 29.72 | True | True |
| fixed | trumans | noisy | 90.0 | 188.02 | 29.72 | True | True |
| rotating | rich | clean | 原输出 | 32.29 | 3.19 | True | True |
| rotating | rich | clean | 30.0 | 32.25 | 3.19 | True | True |
| rotating | rich | clean | 60.0 | 32.29 | 3.19 | True | True |
| rotating | rich | clean | 90.0 | 32.29 | 3.19 | True | True |
| rotating | rich | noisy | 原输出 | 61.09 | 30.37 | True | True |
| rotating | rich | noisy | 30.0 | 61.02 | 30.37 | True | True |
| rotating | rich | noisy | 60.0 | 61.09 | 30.37 | True | True |
| rotating | rich | noisy | 90.0 | 61.09 | 30.37 | True | True |
| rotating | egobody | clean | 原输出 | 57.29 | 4.67 | True | True |
| rotating | egobody | clean | 30.0 | 51.72 | 4.67 | True | True |
| rotating | egobody | clean | 60.0 | 52.70 | 4.67 | True | True |
| rotating | egobody | clean | 90.0 | 54.45 | 4.67 | True | True |
| rotating | egobody | noisy | 原输出 | 72.50 | 26.29 | True | True |
| rotating | egobody | noisy | 30.0 | 68.98 | 26.29 | True | True |
| rotating | egobody | noisy | 60.0 | 69.97 | 26.29 | True | True |
| rotating | egobody | noisy | 90.0 | 70.66 | 26.29 | True | True |
| rotating | trumans | clean | 原输出 | 140.02 | 6.66 | True | True |
| rotating | trumans | clean | 30.0 | 140.55 | 6.66 | True | True |
| rotating | trumans | clean | 60.0 | 135.31 | 6.66 | True | True |
| rotating | trumans | clean | 90.0 | 136.28 | 6.66 | True | True |
| rotating | trumans | noisy | 原输出 | 173.88 | 29.01 | True | True |
| rotating | trumans | noisy | 30.0 | 166.59 | 29.01 | True | True |
| rotating | trumans | noisy | 60.0 | 164.60 | 29.01 | True | True |
| rotating | trumans | noisy | 90.0 | 167.15 | 29.01 | True | True |

真实需要大幅修正的子集、P90、逐窗退化数、源身份与哈希见 REVIEW.json。
范围相对于参考局部旋转，不是限制膝盖的绝对弯曲角。GT仅用于指标与分组，投影函数不读取GT。
unobserved仅使用输入观测元数据保护可见位置和全局旋转；按帧mask切换可能引入跳动，速度和加速度误差已记录。
