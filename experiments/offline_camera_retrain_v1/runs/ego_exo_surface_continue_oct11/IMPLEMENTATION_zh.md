# 离线 ego/exo 接入与55k续训记录（2026-10-11）

正式续训已从原始 native_dynamic_scene20_contact_55k_oct07/last.pt 启动，原权重和 optimizer 保留。新实验独立保存一个原子 last.pt，计划3000步，batch=2，固定128帧/20Hz。前500步回放三域ego，之后在EgoBody域内以50%概率改用exo，域采样仍按有效128帧覆盖时长的平方根；不会按相机数或重叠窗口数增权。exo启动前要求train的213片段和validation的99片段完整导出并通过逐文件哈希、缺失屏蔽和录制split审计。

适配器结构改变，所以从55k模型参数热启动，新增AdamW：旧参数lr上限2e-5，局部attention参数上限1e-4，50步warmup，clip=1。这不是原optimizer轨迹的逐位续跑。第100步保存后修正了外部control mask的导数掩码，检查通过后从同一checkpoint恢复model/optimizer/CPU、CUDA、数据和条件RNG。原始实现保存在git 6c3d608；修改和代码哈希迁移记录在protocol.json中。

## 点云、可见性与离线时间范围

不需要RGB，也不需要额外采集depth。原始完整场景网格、相机内外参与两个完整人体网格用于模拟观测：根据带镜头畸变的像素光线取近裁剪面之后的第一交点，自己、另一个人和静态物体都会遮挡。depth只是这个计算的中间量。最终模型得到的场景是该视角真正见过的点云，不是完整扫描。

只有点云也可以投影后做z-buffer/带点半径的可见性近似；稀疏点云的孔洞不能被当成真实无遮挡。本地有网格，因此导出采用三角形first-hit。人物网格只参与遮挡，不能写入静态场景记忆。exo另一个人当前只作为遮挡物，未作为可交互动态人体条件；ego回放中的TRUMANS动态物体保持当前帧可见性。

这是离线任务：允许使用当前可用观测流完整时间范围内的静态已观测表面并集，不宣称因果/在线。ego使用自己观测流的静态记忆，exo使用固定主Kinect的观测；两者不会互借缓存。动态物体仍只使用当前时刻可见几何。缺失邻域有有效性标记和null attention token，未观测区域不等于空闲空间，部分点云的最近距离不被称为signed SDF。

可见关节由实际第一交点中人体皮肤区域的像素支持确定，不向内部关节中心做射线：至少2像素，且可见像素占target-alone像素比例至少0.15。头/腕选用原生关节中心作为理想前端的3D轨迹；这是标签模拟输入的上界实验，不是RGB视频估计输入。当前身体模板使用已知原生gender/betas/scale，需单列为校准假设。

## 固定皮肤vertex与场景attention

每种原生模型/性别在中性模板上一次选定67个vertex ID，覆盖22个身体区域；编号不随样本、GT姿势或shape改变。锚点随生成身体的LBS/pose correctives移动。每个锚点查询自身及±8cm、±30cm的三个轴方向，共13个位置，注意力只看观测点及动态标记。局部特征由预测的接触门控送回去噪器，接触分支没有GT contact输入；contact BCE仅用于有标签的RICH。

每个去噪步先以现有输入生成一个proposal，再用它决定皮肤查询位置，最后送回去噪器；因此不依赖GT初始骨盆、GT姿势或GT全身前缀。ego旧camera-BPS作为热启动proposal条件保留，最终分支在第50到1000步逐步替换成body-BPS；exo没有camera-BPS，使用生成身体的查询。SMPL-X缺失手部参数遵循原生模型的平均手型，与解码网格一致，不宣称22关节表示能恢复手指。

[UniSH](https://arxiv.org/html/2601.01222v1)主要用人体点云与SMPL的几何对应改善人景对齐，不能把它直接引用为这个接触attention设计。[UniCon3R](https://arxiv.org/html/2604.19923v1)的局部人景上下文、cross-attention及接触反馈更接近这里的设计依据。当前67×13实现是本项目的适配选择，没有声称复现这两篇的完整结构。

## 轨迹监督与尺度

位置、旋转的availability/confidence独立。exo仅位置输入，旋转位没有填GT；隐藏输入清零。输入也增加线速度、加速度和角速度。导数按20Hz换成m/s、m/s²、rad/s，位置输送仍为m/2。有效导数要求完整差分stencil都可用，不能跨关节缺失、时间间隙或外部control mask。

相机轨迹不能直接当头关节。安装R/t只在TRAIN上拟合，并固定用于训练/验证监督。真实camera wearer的安装误差p90约6.5cm/14.9°，据此放宽尺度；其他虚拟ego的误差较小。anatomical输入直接监督对应native FK和global rotation。camera/meta type区别只在明确语义下转换，没有以相机朝向代替人体GT根朝向。

旧coordination_v1全身位置、速度、加速度、姿态、协调和足部目标保留。原速度/加速度以每帧差分写在基准代码中：物理单位等价系数分别10/20²、2/20⁴，不能简单乘20、400而保持系数不变。新监督使用Huber和每例有效mask归一，confidence及alpha²不进入分母，避免权重被归一化抵消。head/pelvis区域权重2、wrist/ankle/toe1.5、其他1。新系数在TRAIN样本共享输出梯度上初始化校准，未用validation调参；它不是全训练分布的梯度保证。

| 监督 | 归一尺度 | 系数 | 初始加权梯度/基础目标 |
|---|---:|---:|---:|
| position | 0.05 | 0.01036799 | 10.0% |
| rotation | 0.174533 | 0.006172422 | 4.0% |
| velocity | 0.15 | 0.0005389064 | 3.0% |
| acceleration | 1 | 8.051522e-06 | 0.5% |
| angular_velocity | 0.349066 | 0.0002830802 | 1.5% |
| body_turn_rate | 0.349066 | 0.0005136066 | 1.5% |

旋转监督之外保留独立的GT根角速度训练目标，以SO(3)相对旋转计算。线速度/加速度能约束轨迹变化，但后退、侧走、头身独立转向使平移方向不能唯一决定身体朝向。是否改善大转身以及是否伤害脚步，必须用固定的朝向、脚部和全局误差评估，不能从loss下降推出。

## 已运行的检查与评估范围

input_checks.json：固定皮肤点与原生SMPL/SMPL-X网格最大坐标差约2.4e-7m；畸变重投影最大差约3.2e-5像素；exo观测关节与native标签最大差约5.1e-7m。GT motion/joints/contact标签投毒、隐藏输入NaN投毒、外部屏蔽帧投毒都不改变输出；空场景、时间缺失掩码、近裁剪面、前景遮挡和完整反向传播通过。录制split为train64/val30/test29且无交集。

frozen_exo_inputs.py另提供不需要NativeBodyData、ego资产或身体标签的推理加载器，只读冻结的可见位置、可用性/置信度、可见场景点及标定；body profile单独提供已知校准。frozen_input_checks.json实际从随机噪声生成64帧，未加载motion/joints/contact或GT初始pose。

length_interface_checks.json：ego/exo分别在64、128、192帧执行随机噪声DDIM生成，shape和finite通过。当前依然是整个窗口联合去噪，不是逐帧自回归。4步仅验证接口，不能声称任意长度或不同长度质量已验证。此前混长参考是[EgoAllo](https://openaccess.thecvf.com/content/CVPR2025/html/Yi_Estimating_Body_and_Hand_Motion_in_an_Ego-sensed_World_CVPR_2025_paper.html)，本项目64/128/192的离散比例是工程选择；这次固定128。

32步integration为接入检查，已完成10-step DDIM、每域1个开发窗口的配对评分；这些小样本不代表正式性能或exo适配完成。TRUMANS/ego世界误差188.95→197.16mm，camera wearer/ego83.27→83.40mm，interactee/ego62.67→66.74mm，RICH/ego183.31→181.79mm。exo仍是零样本输入，误差很大，不把小幅变化当统一模型有效性的证据。一次真实TRAIN exo优化检查包含757个可见位置输入、0个旋转输入，各监督和梯度通过，模型随后丢弃。

正式训练完成后自动评估同人体/时间的EgoBody ego/exo模式及三域ego回放，对照原55k；从独立随机噪声生成完整128帧，没有GT身体前缀。未观测关节误差、观测关节一致性、净转向及native足部marker滑动会单列。当前exo采样继承原55k的有效native128窗口，扩大exo独立可用窗口、多个外相机及ego+exo同时融合尚未实现。

unified_metrics.py把世界坐标未对齐误差明确叫world_unaligned_mpjpe_mm；另实现100帧段前两帧Sim(3)对齐W和整段Sim(3)对齐WA，短尾不计入100帧指标。每帧PA、20Hz速度/加速度及native足部marker另列。[Human3R](https://fanegg.github.io/Human3R/Human3R.pdf)也使用前两帧/全段对齐，但官方关节回归器、数据集、采样率、输入信息及已知shape假设未对齐前，不能直接拿论文表格比。

## 转向复统计和剩余研究边界

原55k全部缓存验证为12,913窗口/139人体序列。净GT髋轴转向≥30°且水平投影可靠的6,133窗口中，明显反号756（12.33%），不足GT幅度一半3377（55.06%），两者并集3610（58.86%）；并集占所有窗口27.96%，涉及100/139序列。定义、窗口身份及分域计数见../turn_recount_oct11/REPORT_zh.md。重叠窗口不是独立转身事件；净转向漏掉转回来和局部时序错误。

真实Human3R/UniSH等前端仍需另外评估：人和场景的共同米制尺度、摄像机/场景与人体轨迹相关漂移、joint定义转换、人体身份切换、误检测和长时间遮挡不能用只加小幅独立噪声代表。静态已观测表面不能补出隐藏物体后侧；未知区域也不能强加接触/无碰撞。统一接口的当前受控实验和真实视频系统性能应分别报告。

## 文件与存储

正式日志/tmp/unified_formal_resume.log；trace.jsonl逐步保存身份、噪声t、模式、scene开关和loss，protocol.json记录配置/代码迁移，last.pt每100步原子覆盖并保留optimizer及RNG。模型与数组仅留本地。32步integration已完成评分，无续训计划，移除其optimizer并逐tensor校验model摘要一致，释放约963MB；原55k及正式last保持可续训。

完整exo导出及逐文件审计已通过：train {'segments': 213, 'recordings': 64, 'person_frames': 125940, 'any_head_wrist_frames': 125850, 'no_head_wrist_frames': 90, 'partial_head_wrist_frames': 22675}；validation {'segments': 99, 'recordings': 30, 'person_frames': 62400, 'any_head_wrist_frames': 62294, 'no_head_wrist_frames': 106, 'partial_head_wrist_frames': 9286}。全片段完全缺少场景或头/腕观测的条目数 0，保留为unknown而不补GT场景。

评分进一步区分相机没有可靠观测的region关节（含遮挡、画外、像素不足）和相机可见但未选作头/腕输入的关节；同报camera/anatomical轨迹位置、线速度/加速度、可用朝向/角速度一致性及根/头GT全局朝向。region可用性只用于评分，不能进入模型输入。度量修订记录独立evaluation_code_sha256，未改变进行中的训练目标。

最新状态：正式3000步训练与预登记开发面板评估已经完成，实际exo为349个优化step/698个样本。12个ego窗口平均世界误差138.38→149.95mm，6个exo窗口511.63→230.87mm；原55k的exo是零样本。新增attention输出屏蔽后ego149.46mm、exo230.87mm，当前没有稳定收益。完整18案例的动作播放器为http://127.0.0.1:8781，指标、54次同噪声场景消融及条件作用说明见[REVIEW_zh.md](REVIEW_zh.md)。last.pt保留完整optimizer和RNG；用户要求先看动作可视化，再决定续训或结构修改，当前不追加优化。
