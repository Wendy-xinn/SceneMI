# 修正内参与分支学习率后的场景重训

用户于 2026-10-30 明确要求重新训练场景分支。本轮配置为
`configs/scene_joint128_conditioning_corrected.yaml`，入口为
`python -m scenemi.scene_joint_training`，输出目录为
`runs/scene_joint128_conditioning_corrected`。

从无场景 replay step 2000 原父权重重新初始化，场景阶段从 step 0 开始训练
3000 步；不加载旧场景 step 1000 的 adapter 或 optimizer。父权重身份为
`a46122d40c572be8ac82b55ca7d421e7c64a4de690c7ae369091064ba1b9260c`。
历史场景权重、保护恢复包和正在执行的历史版本完整验证均保留。

## EXO 内参的证据与约定

`native30_export.py`、GRAIL 导出器和 TRUMANS 移动相机准备入口均把 EXO
内参存为 `[fx/W, fy/H, cx/W, cy/H]`。旧场景 loader 在这个数组上又 `/256`，
这是额外缩小，不是像素内参的首次归一化。

新入口复用旧场景几何构建，撤销额外缩放，并在每个 EXO 样本上断言：

1. 实际网络输入逐值等于导出的 EXO 数组。
2. 该数组与元数据声明的 `K`、`source_size` 计算出的宽高归一化值一致。

准备阶段还遍历全部合格 EXO 训练、验证序列逐一检查源数组，不以数值范围猜单位。
EGO 来自不同的像素值数组 `camera_intrinsics_128x96`，保留父模型的 `/256`
特征缩放，并在三个数据集的训练、验证真实样本上检查加载后的数值。
不修改标定、世界坐标、点云、人体、初始化扰动或 split。

## 实际优化参数

AdamW 分为两个互斥参数组，覆盖全部可训练参数，父模型参数均冻结：

| 参数组 | 参数 | 基础学习率 |
|---|---|---:|
| ego | EGO 场景 adapter | 1e-4 |
| exo | EXO condition、内部 projections、scene adapter | 3e-5 |

两个参数组分别乘同一 warmup / cosine 因子；每步记录 `lr_ego` 和 `lr_exo`，
不能只看配置文件。单元测试检查实际 AdamW 参数更新幅度及 EXO/EGO 比例 0.3。

每次更新累计 64 个 EGO 与 64 个 EXO 窗口。为安装实际 4 GiB allocator 上限，
使用每组 8 个窗口的同模式 microbatch，累计后只进行一次梯度裁剪与 optimizer step。
EXO hidden/missing 项使用整个模式 batch 的掩码分母，EGO 原生体形项使用整个
模式 batch 的 SMPL/SMPL-X 样本数；其他项按窗口数加权。保持原有 loss 项及系数，
不改变掩码样本的权重，也不增加目标。单元测试用原 `denoising_loss` 验证完整 batch
与分块累计的损失及梯度一致；浮点求和顺序和微批次噪声生成不宣称与旧训练逐位相同。

局部场景关系复用原函数，在 CPU 上计算当前预测查询的几何特征，网络在 GPU 上
前向与反向。几何查询原本在 `no_grad` 中；每步重新使用预测 anchors/rotations，
仅缓存同一调用中不变的观测数据和 KD 树。旧模型源码与评估依赖不热改。

## 验收与恢复

正式训练之前要求完整输入单位核查、CPU 和 CUDA 真实数据验收通过。
CUDA 验收执行完整有效 batch=128 的一次更新，再以真实样本检查保存/恢复后
下一次更新的 adapter、optimizer 和记录逐位一致；验收更新全部丢弃。
同时核验冻结父模型未改变、梯度有限、实际分组学习率和峰值显存。

新 checkpoint 保存 adapter、optimizer、数据 Generator、Torch CPU/CUDA、全局
NumPy 和 Python 随机状态，以及父权重和源码封印身份。单一 `last.pt` 每 25 步
原子替换并 fsync，不新增 best 或编号权重。恢复时隔离未提交日志尾部。
收到 SIGTERM/SIGINT 后在当前完整更新边界保存；SIGSEGV/SIGKILL 依赖最近已提交
checkpoint。RAM 12 GiB、四核 cgroup、4 GiB allocator 和 4 GiB 物理空闲预留
均需生效；不接纳未识别的外部 GPU 进程，不操作其他 WSL 任务。

完整输入核查覆盖 9,168 条合格 EXO 序列：训练集 EgoBody 130、TRUMANS 469、
GRAIL 7,406，验证集分别 34、33、1,096。核查来源归档在 `input_audit_sources`。
输入 loader 定义保持相同后，统一 `pipeline` 入口复用此核查、加载一次训练索引，
依次执行 CPU 验收、CUDA 完整 batch 验收并重置初始 adapter 和全部随机流，再启动训练。
核查来源与最终训练来源的衔接记录在 `INPUT_AUDIT_ACCEPTANCE.json`。

本轮训练完成不代表质量验证已通过。父模型的既有 AMASS EGO 保留门槛失败记录
继续保留，不能用场景重训成功覆盖；完整三数据集验证及运动稳定性仍须单独评估。
