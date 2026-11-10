# Method 初稿：中文说明与实现核对

日期：2026-10-28。本文是论文工作稿的说明，不是实验结果报告。

**2026-10-29 状态更新：** 下文保留旧稿的组织思路与实现细节，其中“最值得突出共享差分”等写作建议不再视为架构结论。最新核对见 [架构复审与简化讨论](ARCHITECTURE_REVIEW_zh.md)：replay 完整验证已结束，AMASS EGO 门槛未通过；当前另有冻结主干、两套独立场景适配器的 EGO/EXO 实验，不能与下述无场景模型或早期独立 EGO 场景分支混写。

- **英文正文与公式**：[METHOD_DRAFT_CVPR2027.md](METHOD_DRAFT_CVPR2027.md)。
- **可放入 CVPR 模板的正文**：[paper/method.tex](paper/method.tex)，需要 amsmath、amssymb；参考文献在 [paper/method_references.bib](paper/method_references.bib)。
- **场景候选小节**：[paper/scene_extension.tex](paper/scene_extension.tex)。它对应独立 EGO 场景分支，尚未并入当前 joint128，因此没有混入已实现联合模型的正文。
- 当前环境没有 LaTeX 编译器；此版核对文本、公式、引用和实现对应关系，尚未完成 PDF 编译与双栏排版检查。

## 1. 这版论文怎么讲

主线可以概括为：

> 第一人称相机轨迹只提供很少的人体约束，第三人称人体估计则提供较完整但有噪声、遮挡和丢失的约束。我们用同一个动作扩散先验处理这两种观测：EGO 生成未观测的人体动作，EXO 利用可靠性控制对已有动作的修正。缺失区间通过显式的边界上下文获取远处有效观测，修复量由共享主干的条件差分产生，并受运动学和可靠性约束。

正文采用“任务和前置处理 → 共享先验 → 两个主要模块 → 训练 → 推理”的顺序。重点是每个模块解决什么问题、数学上如何作用，卷积宽度与训练参数放在复现说明中。

| 暂定名称 | 中文名称 | 要表达的机制 | 目前可以支持的陈述 |
|---|---|---|---|
| Shared Conditional Motion Prior | 共享条件动作先验 | EGO 和 EXO 使用同一套扩散主干参数 | 统一架构已实现；性能优势需要联合与独立模型对照 |
| Boundary Observation Conditioning（BOC） | 边界观测条件编码 | 显式输入缺失帧两侧最近有效估计的位移、速度、姿态和距离 | 可描述边界条件如何进入网络；改善长遮挡需消融 |
| Shared-Reference Residual Denoising（SRRD） | 共享参考残差去噪 | 同一主干在相同状态下计算“加适配条件”和“不加适配条件”的差，修正时间锚点 | 双通路同权重、双通路梯度已实现；有效性需匹配对照 |
| Reliability-Anchored Inference | 可靠性锚定推理 | 高可靠通道受锚点约束，低可靠通道接受更多去噪修正 | 是推理规则的概括；DDIM 与观测混合本身不单列为原创贡献 |
| Observed Geometry Adaptation（候选） | 观测几何适配 | 在预测人体表面查询实测场景，区分局部表面、测得自由空间和未知区域 | 独立 EGO 分支有实现；联合场景模型尚待集成与验证 |

这些名字是工作命名，不代表已经完成新颖性检索。主干 U-Net、连续 6D 旋转、DDPM/DDIM、零初始化注入和多任务训练都有既有来源。论文的贡献候选应落在具体组合机制及其验证上，不能把模块重命名当作新颖性证据。

最值得突出的公式是正文的共享差分：

$$
d_\tau=g\,\Pi_{135}\left[
F_\theta(X_\tau,\tau,c;\mathcal H_\phi)
-F_\theta(X_\tau,\tau,c;0)\right].
$$

它回答“一个生成式动作主干怎样修复已有动作”。两个输出共享同一组参数，EXO 训练对两条路径都求导；差分经可靠性门控和有界映射后作用在时间锚点上。不要将这里的 reference 写成冻结教师，也不要与独立场景分支的“先预测人体、再查询场景”混为一个过程。

## 2. 参考了哪些 CVPR 论文

参考的是组织方式与论述粒度，正文措辞和模块公式针对当前代码重新撰写。

- [RoHM，CVPR 2024](https://arxiv.org/html/2401.08570v2)：先明确噪声和遮挡下的重建问题，再命名主要网络/条件模块，随后解释推理和训练。我们的初稿借鉴了这种“问题—模块—机制”的结构。RoHM 的 TrajNet/PoseNet 是全局轨迹与局部姿态分解，不能拿来证明当前 EGO/EXO 权重共享的优势。
- [S²Fusion，CVPR 2024](https://arxiv.org/html/2404.04890v1)：按照先验初始化、条件获取、条件去噪和采样组织方法。它帮助厘清本项目的前置处理、扩散主干和场景条件边界。其输入包含头显和手柄等稀疏信号，不能直接当作当前 camera-only EGO 的同条件结果。
- [SceneMI，ICCV 2025](https://arxiv.org/abs/2503.16289)：是当前架构的来源，必须明确归属。当前 scratch 模型从头训练不消除架构来源；原 SceneMI 的全局/局部场景描述也不能被自动算作当前 joint128 已启用的组件。
- 旋转、扩散和采样分别引用 [6D rotation](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html)、[DDPM](https://arxiv.org/abs/2006.11239)、[DDIM](https://arxiv.org/abs/2010.02502)。

## 3. 本版对应的实验状态

这是 2026-10-28 的文件快照，后续状态以运行目录为准。

| 阶段 | 读取到的事实 | 写作含义 |
|---|---|---|
| AMASS joint scratch | 110,000 次更新完成，完整验证选中 100,000 步；不是最后 110,000 步 | 可写从头联合训练与验证选择 |
| 四数据集 replay | 2,000 次更新、256,000 抽样窗口完成；状态为 training_complete_validation_pending | 可写已执行的训练方式；不能提前写最终通过保留能力门槛 |
| 旧 native no-scene batch32 | 状态文件停在 1,500/6,000、full_validation、best=null；另有中断记录 | 不是本版最新父模型，也不能凭旧状态判断仍在训练 |
| 当前 joint128/replay | scene: false | 现阶段主文方法对应无场景联合模型 |
| 独立 EGO observed-geometry adapter | 有代码与独立实验路径 | 保留候选场景小节；不写成已完成 EGO/EXO 场景统一 |

父 checkpoint：runs/amass_joint128/scratch/selection/best.pt；SHA256 为 5145b2e65ac7792ddee225e90e97148c410578c81096c90f4fdb098018a2dafe。来源配置为 [amass_joint.yaml](../configs/amass_joint.yaml) 和 [scene_joint128_replay.yaml](../configs/scene_joint128_replay.yaml)。

## 4. 公式对应实现与关键细节

| 正文内容 | 主要实现 | 核对重点 |
|---|---|---|
| 201D 表示、FK | [supervision.py](../experiments/offline_camera_retrain_v1/supervision.py)、[native_joint_data.py](../scenemi/native_joint_data.py) | 3 根平移 + 132 旋转 + 66 位置；单位与共同 pelvis 参考 |
| 观测及体形 | [observations.py](../scenemi/observations.py)、[models.py](../scenemi/models.py)、[scene_model.py](../scenemi/scene_model.py) | EGO 从相机预测体形；不同人体格式使用原生参数空间 |
| 模拟 EXO 前置估计 | [exo_initialization.py](../scenemi/data/exo_initialization.py)、[native_repair.py](../scenemi/data/native_repair.py) | 当前是从标注合成的受控估计输入，不是已经接通真实 RGB 估计器的实验 |
| 时间锚点 | [exo_denoise.py](../scenemi/exo_denoise.py)、[exo_basis.py](../scenemi/exo_basis.py)、[temporal.py](../scenemi/geometry/temporal.py) | 固定混合权重；原生路径逐个处理缺失区间 |
| BOC | [exo_completion.py](../scenemi/exo_completion.py) | 383D 普通条件 + 150D 边界描述；边界必须是真实可用的估计帧 |
| SRRD | [joint_model.py](../scenemi/joint_model.py) | 联合模型覆盖了旧类的冻结主干逻辑；reference 与 adapted 都参与梯度 |
| EXO loss / 采样 | [exo_denoise.py](../scenemi/exo_denoise.py) | 坐标均方除以 3，旋转矩阵均方除以 9，速度乘 30 |
| 联合训练与回放 | [optimization.py](../scenemi/replay/optimization.py)、[stream.py](../scenemi/replay/stream.py) | 每次更新两模式各半；采样配额与 loss 权重不是同一组值 |
| 场景候选 | [scene_adapter.py](../scenemi/scene_adapter.py)、[scene_features.py](../scenemi/geometry/scene_features.py) | 冻结 EGO 先验、预测皮肤查询、14D 实测关系、两次前向 |

### 时间锚点的具体定义

六个候选依次为原始估计、宽度 3/5/9/17 的三角平滑及边界时间延拓，固定权重为 softmax([0, 0, 0, 1, 1, -1])。在根平移和每个关节上分别组合相对初值的候选增量：

$$
\delta p^A_t=\operatorname{clip}_{0.25\,m}
\left((1-\rho^p_t)\sum_k w_k(p^{(k)}_t-p^I_t)\right),
$$

$$
\delta\omega^A_{t,j}=\operatorname{clip}_{0.6\,rad}
\left((1-\rho^R_{t,j})\sum_k w_k
\operatorname{Log}(R^{(k)}_{t,j}(R^I_{t,j})^\top)\right).
$$

先做可靠性门控，再做径向裁剪，最后由 reconcile_motion 得到 A。完全可靠帧精确复制 I。这个固定基线需要与学习后的 SRRD 分开汇报；它本身不是学习到的混合专家。

原生多缺失区间的延拓分别处理每段。内部缺失段采用两端**已观测**边界 L、R，以 s=(t-L)/(R-L) 构造三次 Hermite 曲线：

$$
y(s)=(2s^3-3s^2+1)y_L+(s^3-2s^2+s)(R-L)v_L
+(-2s^3+3s^2)y_R+(s^3-s^2)(R-L)v_R.
$$

这里 v 是每帧增量，作用于根和 identity-centered 6D 旋转坐标，之后重建合法旋转。前缀/后缀缺失使用单侧延拓；它不读取缺失段真值。BOC 使用的根速度则按 30 Hz 转成 m/s，并按分量裁剪到 [-3,3]。

### 前置模型与输入的边界

初稿描述的前置模块是接口，不把某个尚未锁定的相机重建器或人体估计器写成已经使用的系统组件。目前训练的相机条件来自既定数据处理协议，EXO 初始化由标注原生人体加扰动生成。原生模板类别在模拟条件中也含有数据源先验；未来真实前置实验需要同时交代 pose、shape、template、world trajectory、visibility 与 reliability 的来源。

EGO 仅根据相机条件预测体形，不能表述成“精确恢复个人体形”。EXO 的估计 rest skeleton 与目标 rest skeleton 分开计算 FK；不能用目标体形偷偷构造预测位置。只有零噪声 identity 控制按定义使用干净初始化，需要在实验条件中明确。

## 5. 训练复现说明

联合 AMASS scratch 的两个模式各占一半，训练 110k updates，最终按完整验证选取 100k。四来源 replay 的单次更新包含 64 EGO + 64 EXO，两个模式顺序累积半权重梯度，再裁剪并执行一次 AdamW 更新。

| 来源 | 抽样比例 | 每个模式的 64 个窗口配额 | loss 权重 |
|---|---:|---:|---:|
| AMASS | 0.25 | 16 | 0.5 |
| EgoBody | 0.125 | 8 | 1/12 |
| TRUMANS | 0.28125 | 18 | 3/16 |
| GRAIL | 0.34375 | 22 | 11/48 |

四来源 loss 先各自归约，再按右列加权。共享主干学习率 2.5e-6，原生体形分支 2e-4，EXO 适配器 3e-5；warmup 按 2,400 个窗口计。这里只记录已执行配置，不把回放比例解释为已证明的最优值。

EXO 的四个合成条件（identity/tracking/occlusion/gap）概率为 0.2/0.35/0.25/0.2。形状扰动标准差 0.15；普通姿态噪声 0.015–0.04 rad，遮挡姿态噪声 0.12–0.25 rad；根噪声 0.01–0.04 m；姿态/根 AR 系数为 0.85/0.95；合成 gap 长度 8–32 帧。实际原生可见性还可能造成多段缺失，不能把所有遮挡均解释成单一合成 gap。

### EGO loss 的完整项

正文用求和写法保留可读性；完整实现为 [losses.py](../scenemi/losses.py)，系数来自 [amass_ego128.yaml](../configs/amass_ego128.yaml)。MSE 表示对所有相关元素取均值；G 是沿关节链累计后的全局旋转；速度和加速度按 30 Hz 计算。下表带“信号门控”的项先逐样本取均值，再乘 bar-alpha 后按 batch 取均值。

| 项 | 定义摘要 | 系数 |
|---|---|---:|
| x0 | 重建后 201D 与目标的加权 MSE，前三维权重 2 | 1 |
| raw_x0 | 未重建 raw 输出的同类加权 MSE | 0.5 |
| rotation | 全局旋转矩阵元素 MSE | 0.2 |
| position | FK 关节坐标 MSE | 2 |
| head_position | 头关节坐标 MSE | 1 |
| velocity | 关节速度 MSE，信号门控 | 0.025 |
| acceleration | 关节加速度 MSE，信号门控 | 1.25e-5 |
| root_orientation | 根旋转差的 Frobenius 范数平方乘 1/2 后取均值 | 0.25 |
| head_orientation | 同上，头关节 | 0.5 |
| neck_orientation | 同上，颈关节 | 0.25 |
| head_root_coordination | G_root^T G_head 与目标对应相对旋转的矩阵 MSE | 0.1 |
| turning | 对根/头的 1、6、12 帧相对旋转误差，乘 (30/lag)^2/3 后求和，信号门控 | 0.02 |
| foot_velocity | 双脚速度向量差平方和的均值，信号门控 | 0.075 |
| planting | 目标脚速 <0.2 m/s 时，对 max(预测脚速−目标脚速−0.02,0)^2 取掩码均值，信号门控 | 0.1 |
| raw_consistency | raw 位置通道还原为米后与 raw FK 坐标的 MSE | 0.5 |
| shape_betas | 预测与目标原生 AMASS 16D 形状系数 MSE | 0.05 |
| shape_rest | 预测与目标 rest skeleton 坐标 MSE | 0.5 |
| camera_position | 预测头位置加旋转后的 mount 与相机位置的 MSE，按相机 availability×confidence 归约 | 2 |
| camera_rotation | 预测头朝向与预处理后的相机朝向矩阵 MSE，同类相机掩码归约 | 0.5 |
| observed_position | (预测关节−观测关节)/0.02 的 smooth-L1，按观测位置可用性×置信度归约并乘信号门控 | 0.01 |

原生 SMPL/SMPL-X 分支将原 AMASS shape_betas 项系数设为零，另加 0.05×原生 10D shape MSE 和 0.01×模板分类交叉熵；shape_rest 仍保留。G1 使用公开固定模板。位置目标总是用 target_rest 计算，而非预测 rest。

EXO loss 的 8 项及系数已经全部写在正文中。速度项取所有坐标 MSE，preserve 项乘可靠度乘积的平方后取普通均值，不按可靠度重新归一化。正文的旋转误差是矩阵元素 MSE，而非 SO(3) 测地距离平方。

## 6. 主张需要哪些证据

先把消融集中在以下几项，不必继续增加模块名称：

1. **共享是否有价值**：共享 EGO/EXO 与对应独立训练控制，在相同数据、曝光和选模规则下比较；完整 replay 验证同时检查两种模式的 AMASS 保留能力。
2. **BOC 是否改善缺失段**：移除边界分支，按 gap 长度、位置和多个缺失区间汇报；同时看可见段是否退化。
3. **SRRD 是否优于固定锚点及简单修正**：至少保留 initial、时间平滑、固定 A、完整 SRRD；若比较直接预测/单通路修正，需要匹配参数量与训练预算，并报告每步两次主干前向的代价。
4. **能否落到实际输入及场景**：真实前置输出单列评估；scene 候选先与同父模型无场景、motion-only、points-only 控制比较，再决定是否并入联合主文。

在这些结果完成前，可以写统一模型的结构与推导，不能写“优于现有方法”“解决灾难性遗忘”“真实视频已泛化”“场景已同时改善 EGO/EXO”。正文也没有把局部平面距离写成全场景 SDF 或穿透约束。

现有 cvpr2027_pipeline_v1 图对应较早分支。后续正式图应与本稿统一：前置相机/人体接口 → 共享表示 → EGO 单次去噪 / EXO BOC+SRRD → 可靠性锚定采样与 FK。场景候选暂用独立虚线框，待集成路径明确后再并入；避免让图暗示当前 joint128 已启用场景。

