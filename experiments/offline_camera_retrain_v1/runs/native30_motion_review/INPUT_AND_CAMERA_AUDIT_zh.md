# EGO / EXO 输入与相机核查

核查时间：2026-10-18T17:42:02.202340+08:00。仅读取源码、既有训练审计、数据 manifest 和既有可视化预测；未修改科学源码、配置、权重，也未启动训练或扩散推理。关键文件身份和行号见同目录 `INPUT_AND_CAMERA_AUDIT.json`。

## 汇报可用结论

**41k 基座只用 EGO 训练，但直接接 EXO 已能产生合理输出；这说明共享表示、人体先验与观测约束具备迁移基础，还不能单独证明强遮挡补全或真实 tracker 泛化。** 当前 EXO 观测是由 GT 位置模拟的理想化前端，输入比 EGO 的单头部轨迹强得多，并有同一套观测锚定帮助。应在相同 EXO 观测、参考人体、场景与 seed 下对比 41k 和适配后，重点看缺失关节误差、可见部位保持以及旋转/时序稳定性。

## “GT 上加扰动”具体指什么

目前从 GT 身体与场景的 first-hit 渲染判断身体区域是否可见。只有满足区域命中至少 2 像素、可见比例至少 0.15 的关节，才拷贝其 GT 三维位置，并提供可见比例相关置信度。clean 条件保留这些准确位置；noisy 条件再添加平滑偏移、漂移和额外短时缺失。EXO 不提供关节旋转。缺失关节在原始观测张量中已 mask 并清零。

因此，可以称“基于 GT 的可见三维位置观测模拟”，但不能说“将整个人的 GT mesh 加噪后直接交给模型修正”。参考 mesh 由这些残缺观测和已知 rest 骨长通过运动学构造；遮挡部位保留的非零参考姿态是低置信度假设，并非未清掉的隐藏 GT 姿态，也不是实际视频 tracker 的完整预测 mesh。当前缺失/恢复时逐帧骨骼方向回退，能造成参考姿态突然翻转；EGO 的头部观测稳定、下游上肢缺少这些断续方向证据，因此两类输入表现不同。

证据：[可见性阈值](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_exo.py:121)、[可见 GT 位置写入](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_export.py:332)、[加噪与缺失](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_input_noise.py:30)、[mask 清零](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_observations.py:29)、[参考构造](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_refinement.py:66)。

另外需要区分扩散训练本身：GT-reference 是监督目标，训练 x_t 由该目标加扩散噪声形成；推理从随机高斯残差开始。不能把训练 x_t 与推理参考人体混为一谈。GT 还用于已知形状/尺度、虚拟头相机、模拟可见性与场景，以及监督损失。证据：[训练带噪状态](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/train_native30.py:127)、[推理采样](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_refinement.py:235)。

## 虚拟相机是否来自同一批动作

是。TRUMANS、RICH 和 EgoBody interactee 的伪 EGO 相机来自目标自身身体拟合的头部位置/旋转与固定安装偏移。EgoBody wearer 则使用实际 PV 相机轨迹；PV-to-head 的总体安装近似仅由官方 TRAIN 的 64 个 recording、1024 个样本帧拟合。

但 EXO 推理不会暗用目标自身的伪 EGO 头轨迹：在 `mode=exo` 时，数据加载明确用外部相机变换覆盖相机位置/旋转。EgoBody EXO 是发布的 Kinect master 标定；RICH EXO 从发布标定视角按序列 ID 的哈希选择；只有 TRUMANS 外部观察者方向和 K 明确采用合成设置。缓存中保留未使用的虚拟头相机字段，不能等同于实际模型 EXO 输入。

证据：[虚拟头相机生成](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_export.py:215)、[EXO 相机覆盖](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_data.py:160)、[外部相机来源](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_exo.py:27)、[TRAIN mount 拟合](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/fit_native30_camera_mount.py:22)。

## 为什么原 41k 直接吃 EXO 已经不错

已有训练审计逐行统计正式 trace 的前 41,000 次更新：82,000 个训练窗口全部为 EGO，EXO 为 0。其中真实 EGO 11,385 个、伪 EGO 70,615 个，动态身体观测只选 15 号头部。当前核查复用该统计，没有为同一结论重复读取约 800 MB 的 trace。[原训练输入审计](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_gait_v2_ego55k_oct15/TRAINING_INPUT_AUDIT_zh.md)

直接 EXO 输出由几个因素共同产生：它仍处于已学到的相同人体/动作语料分布；输入结构本来支持 22 关节；运动学参考已经包含大量可见全身位置；扩散每次预测中还有 gain=0.9 的置信度加权关节位置 IK 贴合与 EXO 躯干锚定。因此“原 41k 流程可兼容 EXO 输入”成立；“网络在完全没有 EXO 先验帮助时已自行解决遮挡”证据不足。这里的约束位于扩散预测流程内部，并不是后来适配层另加的一整个残差网络。

证据：[配置](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_gait_v2_ego55k_oct15/config.json:204)、[约束入口](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_refinement.py:207)、[关节位置 IK](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_observation_ik.py:14)。

抽搐例 `recording_20220215_S21_S22_01_interactee_1901-4901_master_full`，窗口起始 1984：clean/noisy 均平均每帧 19.182/22 个关节可用，12.81% 关节帧缺失，左肘仅 66/192 帧可用，无 EXO 旋转观测。整条 recording 的外部人/物/场景遮挡非常少，但自身遮挡及低像素区域仍能导致关节可用性切换。大量已知精确位置会拉低全身平均 MPJPE；少数手臂旋转翻转仍可能十分明显，不能被几毫米位置误差掩盖。

## 划分与可比性边界

当前 EgoBody manifest 的 train / validation / test recording 集合两两交集均为 0。上述抽搐 recording 中 wearer、interactee 的所有 EGO 与 EXO scopes 全部为官方 validation；没有同一录制经另一个角色混入 TRAIN 的证据。`native30_splits.py` 按 recording 的官方分配处理两类输入，而不是按 PV 可见片段把录制随机拆开。

该核查仅证明 recording 不交叉，不保证不同录制的人物身份或环境不重复。原生身体形状、rest 和世界尺度仍是已知条件；模拟 clean EXO 的准确位置不等于真实 tracker 表现。静态场景记忆是给定 recording 的可见表面离线并集，也不应描述为严格在线因果输入。证据：[官方 recording 分配](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/native30_splits.py:55)、[观测协议原审计](/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_gait_v2_ego55k_oct15/TRAINING_INPUT_AUDIT_zh.md)。

本审计保存当前所读源码 SHA；缓存身份中的原导出 SHA 另行保留。没有改写历史训练/评估的 source seal，也未把当前源码身份混称为每个历史缓存的导出源码身份。
