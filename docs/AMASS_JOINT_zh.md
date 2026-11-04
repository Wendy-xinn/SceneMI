# AMASS：联合适配与从零交替训练

**2026-10-27 用户更正：当前主线是冻结共享主干的 EXO-only 追加 20k，
入口为 `scenemi.exo_continuation`，输出 `runs/amass_exo_continue128`。**
它从原 EGO 55k＋EXO 36k＋边界 6k 的选定权重开始，不加载误开的联合暖启动权重。
CPU/CUDA 验收均已通过；实际运行状态见该目录的 `STARTUP_STATUS.json` 与
`training/STATUS.json`。后台从零 `scratch` 110k 对照仍保留。

误开的 `warm` 联合训练在 1,350 更新停止，其权重按用户授权清理，
配置、日志、指标与 SHA256 留作历史记录，见
`runs/amass_joint128/warm/ARTIFACT_RETENTION.json`。
**下文 warm 配置是历史协议，不是当前主线。** `amass_joint.yaml` 和联合训练源码
仍被从零对照封存，保持不变。

两条现行路线的初始化、训练历史及更新参数不同，比较结果不能单独证明
冻结主干优于解冻。当前均不使用场景或独立测试集。

## 两条训练线

| 项目 | 主线 `warm` | 后台 `scratch` |
| --- | --- | --- |
| 初始化 | EGO-only 55k；EXO 修复 36k＋边界追加 6k | 所有可训练参数从零随机初始化；不加载预训练权重 |
| 新增更新 | 20,000 | 110,000 |
| 有效 batch | 8，EGO/EXO 各 4 | 同左 |
| 共享主干学习率 | 1e-5 | 1e-4 |
| EXO 条件/注入学习率 | 3e-5 | 1e-4 |
| 完整验证间隔 | 5,000 更新 | 10,000 更新，末次 110k 也验证 |
| 状态文件 | `runs/amass_joint128/warm/training/STATUS.json` | `runs/amass_joint128/scratch/training/STATUS.json` |

每次优化更新按 `EGO,EXO,EGO,EXO,EGO,EXO,EGO,EXO` 实际抽取样本，
随后分成同模式的两个 4 样本 microbatch，分别反传 0.5 倍模式损失，
合并梯度、裁剪后执行一次 AdamW 更新。不是 batch 1，也不是概率上接近一半。
采样仍按源动作 family、原始/反射版本、合法起点依次独立有放回抽取；
不限制同一 batch 的 family 重复，不改变为步长 1 的穷举训练。
EGO 相机轨迹连续；EXO 输入沿用声明的合成估计、局部遮挡和 8–32 帧缺失。

110k×4＝55k×8，故后台 EGO 曝光量与旧 EGO-only 55k 相同，EXO 也有同样多的样本。
它不具有相同总计算量：EXO 使用两次主干前向。它与历史部分关节 EXO 混训的
输入协议也不同，不能作为单变量证明“混训必然好/坏”。主线 20k 内每模式新增 80,000 个窗口样本。

## 一个可训练 U-Net，差分两路都反传

EGO 仍使用相机条件的绝对 x0 生成，并从允许观测预测 SMPL-H 16 维体形。
EXO 使用前置人体参数估计及体形；固定时间锚点、边界编码、内部残差注入、
置信度门控和 FK/SO(3) 修复规则保留。

旧版冻结主干时，未适配参考前向处于 `no_grad`。联合解冻版改为
`adapted(theta, adapter) - reference(theta)` 两路完整求导。
两路共享当前同一份主干参数，参考不是固定教师，也没有第二套扩散模型权重。
零注入时两路主干梯度应相消；CPU/CUDA 验收显式检查这一点，防止解冻后
产生来自单边停止梯度的伪修复更新。
场景模块、人体模型基底和未训练时间锚点保持冻结。

暖启动推理必须与当前 55k＋42k 权重完全一致，随后才允许更新共享主干。
因此原来的 EGO“冻结所以严格不变”保证已经不适用，必须实际做完整 EGO 验证。

## 验证、选模和恢复

- 固定全部 337 个验证 family、2,222 个不反射窗口。EGO 用 20 步 DDIM、batch 4；
  EXO 用 8 步 DDIM、batch 8，分别覆盖 identity/tracking/occlusion/gap。
  验证起点为历史 128 步长加对齐尾窗，保持同一清单和随机种子。
- 主线 step 0 只在 CPU/CUDA 验收证明初始化完全一致后复用既有完整指标。
  后续每个候选同步评测，不会因异步评测忙碌而漏掉某个规定候选。
- 综合分为 EGO MPJPE 与 EXO tracking、遮挡区域、缺失区域 MPJPE
  各自除以固定基线后等权平均，避免不同数量级直接相加。
- 主线还要求 EGO MPJPE 不高于 55k 基线；EXO 每种区域不比边界父权重恶化超过 1 mm，
  并要求正常观测/根轨迹相对输入不劣及 identity 保持。
  指标门槛是选模规则，不能代替独立测试或多种子证据。
- 原始 EXO 父模型仍未超过旧时序对照的遮挡指标；新实验不删除或重定义这个结果。
- 从零对照使用相同综合分和观测安全约束，但不把“必须已经超过训练好的 EGO 基线”
  设为早期权重保存条件。其是否可用要另行看最终完整指标。
- 每 500 更新原子保存 `last.pt`，同步完整验证后保存一个 `best.pt`。
  新检查点包含更新后的共享主干、EXO 适配器、优化器和所有随机状态，
  不再是依赖未变 55k 主干的纯适配器文件。断点恢复须匹配源码、数据、配置和设备。

配置：`configs/amass_joint.yaml`。模型：`scenemi/joint_model.py`。
训练/验证：`scenemi/joint_training.py`。准入：`scenemi/joint_runtime.py`。
新准入只识别同用户 SceneMI 作业，拒绝未识别 GPU 任务；每进程分配上限 5.5 GiB，
仍要求额外 1 GiB CUDA 及 2 GiB 空闲余量、10 GiB RAM 和两个 CPU 核的限额。

训练内记录两模式损失、实际样本身份和更新时间，并自动做上述验证。
不重新开启此前停用的外部定时检查器。主线完成后写 `HANDOFF.json`；
三数据集无场景训练仍需新的原生人体/观测适配和 CPU/CUDA 验收，
不能直接用历史 EXO-only、冻结主干的三数据集入口冒充联合训练。
三数据集限定 EgoBody、TRUMANS、GRAIL，遵循 `configs/scene_data_current.json`，
TRUMANS 使用已验收移动相机，排除 RICH。
