# EXO 隐藏关节路径实现审查

只读审查当前10k实验封存实现，另运行已有小型CPU契约测试35项，全部通过。没有加载实际checkpoint，没有启动GPU推理/新训练，没有修改封存源码。审查的13个实际依赖源码均与训练封存哈希一致，证据行号见JSON。

**目前没有查到6D旋转行列约定、reference/GT残差加减或隐藏GT监督掩码错误；已确认当前设计允许把好参考大幅改坏。**这不代表实际推理路径已经完整排除故障。

- 训练目标是GT-reference；推理将absolute-reference作为DDIM残差状态更新，最后再加reference。参考、GT、FK统一使用局部旋转矩阵前两列和identity-centred6D。
- reference+residual是201维欧氏加法，不是有角度上限的SO(3)补偿，没有保证相对reference只做小改动的机制。
- 膝盖位置可见，只约束上游骨骼能否把膝盖放对；膝盖自己的旋转决定踝/脚位置。当这些后代全不可见时，观测IK对此膝旋转的strength恰为0。
- 参考旋转先乘rotation confidence；token还乘position/rotation confidence的最大值。完全隐藏关节会两次弱化；但错例右膝位置可见，其旋转不能简单描述成0.05×0.05。
- 显式场景反馈每次x0调用上限0.12rad，不限制U-Net本身的残差，也不限制DDIM整条轨迹的总旋转变化。
- 隐藏GT仍受raw-x0、x0、FK位置、全局旋转监督，没有被可见性掩码删掉。当前只训练内部LoRA，其余原始参数冻结；是否容量不足尚未证实。

身份说明：当前+3k control是原架构，额外scene_fusion_v2试验已reject，没有附加中层cross-attention模块。当前保留原有patch/local/relation attention、条件注入和learned feedback，不能误把历史fusion模块当作当前执行路径。

下一次诊断应保留同checkpoint、同输入fingerprint、同种子，比较scene ON/OFF，记录每步proposal、raw x0、反馈后、IK后的局部膝旋转，再判断错误最早出现在哪一段。同时检查实际6D两列范数/夹角与零残差直通。

现有ops/motion_diagnostics.py的观察器思路可复用，但CLI固定旧19k/EGO并要求physicsON，不能直接用于当前8k。采样API和扩展要求已写入JSON；本次没有执行新推理。
