# EXO 参考旋转修正范围 CPU 对照

阶段10000；范围unobserved；eligibility=window；仅处理已保存的 EXO 输出；不训练、不重跑 DDIM/IK，不自动选最终方案。

| 面板 | 数据集 | 输入 | 上限° | hidden mm | visible mm | visible守卫 | foot守卫 |
|---|---|---|---:|---:|---:|---|---|
| fixed | rich | clean | 原输出 | 38.44 | 3.38 | True | True |
| fixed | rich | clean | 30.0 | 38.44 | 3.38 | True | True |
| fixed | rich | clean | 60.0 | 38.44 | 3.38 | True | True |
| fixed | rich | clean | 90.0 | 38.44 | 3.38 | True | True |
| fixed | rich | noisy | 原输出 | 114.18 | 30.20 | True | True |
| fixed | rich | noisy | 30.0 | 92.35 | 30.20 | True | True |
| fixed | rich | noisy | 60.0 | 99.39 | 30.20 | True | True |
| fixed | rich | noisy | 90.0 | 105.71 | 30.20 | True | True |
| fixed | egobody | clean | 原输出 | 61.86 | 4.95 | True | True |
| fixed | egobody | clean | 30.0 | 60.07 | 4.95 | True | True |
| fixed | egobody | clean | 60.0 | 59.54 | 4.95 | True | True |
| fixed | egobody | clean | 90.0 | 60.59 | 4.95 | True | True |
| fixed | egobody | noisy | 原输出 | 75.99 | 26.98 | True | True |
| fixed | egobody | noisy | 30.0 | 75.51 | 26.98 | True | True |
| fixed | egobody | noisy | 60.0 | 75.09 | 26.98 | True | True |
| fixed | egobody | noisy | 90.0 | 75.51 | 26.98 | True | True |
| fixed | trumans | clean | 原输出 | 155.48 | 7.03 | True | True |
| fixed | trumans | clean | 30.0 | 156.98 | 7.03 | True | True |
| fixed | trumans | clean | 60.0 | 152.99 | 7.03 | True | True |
| fixed | trumans | clean | 90.0 | 152.44 | 7.03 | True | True |
| fixed | trumans | noisy | 原输出 | 193.16 | 29.72 | True | True |
| fixed | trumans | noisy | 30.0 | 190.47 | 29.72 | True | True |
| fixed | trumans | noisy | 60.0 | 188.88 | 29.72 | True | True |
| fixed | trumans | noisy | 90.0 | 188.17 | 29.72 | True | True |
| rotating | rich | clean | 原输出 | 32.29 | 3.19 | True | True |
| rotating | rich | clean | 30.0 | 32.29 | 3.19 | True | True |
| rotating | rich | clean | 60.0 | 32.29 | 3.19 | True | True |
| rotating | rich | clean | 90.0 | 32.29 | 3.19 | True | True |
| rotating | rich | noisy | 原输出 | 61.09 | 30.37 | True | True |
| rotating | rich | noisy | 30.0 | 61.09 | 30.37 | True | True |
| rotating | rich | noisy | 60.0 | 61.09 | 30.37 | True | True |
| rotating | rich | noisy | 90.0 | 61.09 | 30.37 | True | True |
| rotating | egobody | clean | 原输出 | 57.29 | 4.67 | True | True |
| rotating | egobody | clean | 30.0 | 53.30 | 4.67 | True | True |
| rotating | egobody | clean | 60.0 | 53.38 | 4.67 | True | True |
| rotating | egobody | clean | 90.0 | 54.64 | 4.67 | True | True |
| rotating | egobody | noisy | 原输出 | 72.50 | 26.29 | True | True |
| rotating | egobody | noisy | 30.0 | 70.79 | 26.29 | True | True |
| rotating | egobody | noisy | 60.0 | 70.83 | 26.29 | True | True |
| rotating | egobody | noisy | 90.0 | 70.97 | 26.29 | True | True |
| rotating | trumans | clean | 原输出 | 140.02 | 6.66 | True | True |
| rotating | trumans | clean | 30.0 | 140.50 | 6.66 | True | True |
| rotating | trumans | clean | 60.0 | 135.43 | 6.66 | True | True |
| rotating | trumans | clean | 90.0 | 136.28 | 6.66 | True | True |
| rotating | trumans | noisy | 原输出 | 173.88 | 29.01 | True | True |
| rotating | trumans | noisy | 30.0 | 168.17 | 29.01 | True | True |
| rotating | trumans | noisy | 60.0 | 164.96 | 29.01 | True | True |
| rotating | trumans | noisy | 90.0 | 167.25 | 29.01 | True | True |

真实需要大幅修正的子集、P90、逐窗退化数、源身份与哈希见 REVIEW.json。
范围相对于参考局部旋转，不是限制膝盖的绝对弯曲角。GT仅用于指标与分组，投影函数不读取GT。
unobserved仅使用输入观测元数据保护可见位置和全局旋转；速度和加速度误差已记录。
window取整窗eligible交集，没有逐帧开关，但参考抖动、动作投影仍可能损害时序；没有加入平滑。
