# 可见轨迹扰动、GT 质量、窗口记忆与重训顺序

2026-10-12。当前仍是零训练步。本文记录代码核对和建议配置；**没有修改模型、正式训练配置、原始划分或源标注，也没有启动训练。** 新 native30 代码当前初始化方式仍为 from scratch，旧55k／3k权重保留。

## 1. 当前实现核对

|问题|当前 native30 的实际行为|
|---|---|
|训练窗口|30 Hz，固定192帧；从原始连续源段内随机选有效起点|
|重叠|候选起点逐帧移动，允许重叠；不是固定 stride=192 切块，也不是每步依次走下一帧|
|源有效性|窗口内192帧都须 observation_valid；关节被遮挡／缺失不等于源帧无效|
|划分顺序|先固定源 recording／序列及 split，再采样窗口；不把重叠窗口随机分到 train／validation|
|训练采样|按有效起点数分配段内权重；语料层有时长平方根平衡|
|当前比例|ego 70%／exo 30%；正式exo仍只支持EgoBody master，两套其他数据只是检查用exo|
|场景记忆|当前采样窗口192帧的可见静态点并集，2.5cm去重；没有窗口前prefix或跨窗口持久地图|
|动态表面|当前帧可见表面，不能跨帧积成静态地图|
|静态token复用|同一次DDIM采样内复用，不跨样本／跨窗口共享状态|
|已有扰动|每关节约1cm位置偏置＋窗口内约2cm线性漂移；30%概率让一个输入关节缺失18帧（0.6秒）|
|缺少的扰动|旋转、时序相关抖动、短时离群／跟踪重定位、成组部位丢失、场景重建不确定性|
|SMPL／SMPL-X|共享22个body joint的局部旋转和root输出；使用各自rest骨架、体型和native skinning资产|
|完整SMPL-X|没有生成手指／表情／眼睛／jaw自由度；额外关节使用固定默认旋转|
|评估循环|训练入口虽然建了validation对象，目前尚无正式采样／指标循环|
|GT质量mask|目前base x0／FK／全局旋转及v/a监督没有逐部位source-GT质量mask|

窗口例子 `[0,192)` 和 `[1,193)` 有191帧重叠；随机抽样也可能选到相距很远、互不重叠的起点。验证建议使用固定起点／seed／扰动，而不是每次随机改变难度。长视频推理可以先尝试stride=96的窗口重叠，输出还原同一世界坐标后混合root和SO(3)旋转再做FK；不能直接拼接或线性平均6D旋转。正式验证指标需按独立源帧归一，避免重复窗口反复计入同一帧。

当前静态记忆含窗口后半段看过的表面，因此前半段可用这些表面，属于离线协议；但没有用窗口外的过去或未来。若后续加入整段视频已观察地图或prefix记忆，应作为明确的新预算／消融，并对训练和所有前端对照统一。第一轮建议保留当前窗口预算，减少同时变化的变量。

## 2. 划分审计与官方测试集边界

原始源库存：TRAIN 744段，validation 160段。

|来源|TRAIN段数|validation段数|
|---|---:|---:|
|TRUMANS|469|33|
|EgoBody camera wearer|99|47|
|EgoBody interactee|114|52|
|RICH|62|28|

读完原始manifest及RICH源帧边界后确认：EgoBody的64／30个recording不重叠；TRUMANS和RICH的源序列也不重叠。EgoBody、TRUMANS内部scene family不交叉；RICH有4个scene共享，但序列不共享。不能把所有数据的validation都称为未见场景测试。

**当前EgoBody使用项目自定义场景划分，不是官方data_splits.csv。** 原始划分函数在 `diffusion-motion-inbetweening/experiments/offline_sequence_v1/index_sources.py` 中把 `cnb_dlab`／`seminar_g110` scene family设为validation；其他family进入TRAIN。对应官方recording来源为：

|项目当前划分|官方train recording|官方val recording|官方test recording|
|---|---:|---:|---:|
|TRAIN|43|9|12|
|validation|12|8|10|

这可以作为独立的内部场景泛化协议，但**不能在这套TRAIN上训练后，声称官方EgoBody test未参与训练，并与论文官方test数字直接比较**。旧模型结果保留其原内部协议身份。

若重训以正式exo比较为目标，建议新建版本化benchmark manifest：EgoBody官方train训练，官方val调参，官方test只评估，wearer／interactee及所有相机模式跟随同一recording划分。不要改写旧manifest或旧checkpoint依赖。需要从原始源索引重建，不能只移除12个test recording就声称覆盖官方完整训练集。RICH使用对应官方序列划分；TRUMANS固定公开可复现的场景／序列划分。安装标定拟合、confidence校准、噪声统计、loss梯度尺度也只用新TRAIN重做，原记录里的TRAIN指旧内部划分。

当前按长度计算的TRAIN候选起点上界为1,116,686，validation为149,507；未扣bad frames／对象有效性，**不是实际可训练窗口数**。新官方协议还未重建，因此这些数字只描述现有库存。

审计见 [WINDOW_AND_MEMORY_AUDIT.json](WINDOW_AND_MEMORY_AUDIT.json)，可用 [audit_native30_sampling.py](../../audit_native30_sampling.py) 重算。该脚本也直接验证两种native模型的前22个parents相同、各自67个anchor，以及窗口切片确实排除了prefix／suffix静态点。

## 3. GT拟合异常怎样处理

数据集的GT是拟合标注，不等于无误差的直接测量。已确认person crossing源SMPL序列的开头右臂有局部跳变；是否为拟合失败还需原图核对。当前这段属于项目validation、官方train，不能因为模型做不好就静默删除validation困难帧。

建议保留原始pkl和完整回放，另建source-quality sidecar，记录具体source frame、部位、判断证据和状态：`suspect / reviewed / refitted`。先做同一套TRAIN质量审计，指标结合局部SO(3)跳变、FK关节速度／加速度、骨架关系及可用多视角重投影；速度大本身不能判定错误。

对训练中已确认不可靠的部位，用独立GT质量权重处理位置、局部旋转和受父关节影响的global/FK标签；不能只把观测confidence降低。坏父关节会影响子链，质量mask也要按依赖传播。速度／加速度使用两帧／三帧端点的质量共同门控，不跨坏帧计算；相关contact和身体朝向辅助标签也应检查依赖。

少量不易判定且影响全身的TRAIN异常可以保守隔离相关窗口；局部异常优先保留可靠身体部位。validation保留原始报告，再附预先固定的质量分层指标。没有原图／多视角证据时不把简单低通或插值当成修复后的GT；有证据时可用重投影＋时序＋骨长／接触约束重拟合，另存版本并复核。不能逐关节平滑XYZ后直接当骨长正确的身体。

本次没有落实到训练loss中，也没有改变任何源标注。完整源拟合异常证据见 [FOLLOWUP_DESIGN_AND_AUDIT_zh.md](FOLLOWUP_DESIGN_AND_AUDIT_zh.md)。

## 4. 输入扰动建议

扰动只加到可用输入条件；经质量审核的完整动作目标保持干净。与扩散内部给motion加的噪声分开：前者模拟前端测量错误，后者是去噪训练过程。

建议保留20%–30%的干净／很轻噪声样本，其余混合下面几类；数值是初始实验范围，之后由TRAIN前端残差校准，不是已验证的最佳配置。

|输入误差|建议起点|约束|
|---|---|---|
|时序相关局部位置误差|约0.5–2cm，再逐步增加|每部位独立不确定性，避免只制造逐帧白噪声|
|共模位置偏置／慢漂移|保留1cm／2cm现有级别，后续按前端统计调整|区分root共模与局部关节误差|
|有旋转观测时的旋转误差|先约1–3°，再试3–8°|用SO(3)组合；无rotation观测的exo不凭GT补上|
|短时跟踪异常|少量5–15cm离群／短时重定位|附低可信标记，避免把随机噪声都宣称为真实误差|
|缺失／遮挡|真实几何mask为主，增加成组手臂／下肢短缺失|保留真实相关性；长时间全身消失仍排后|
|场景估计误差|轻度点稀疏、丢点、表面误差|unknown保持unknown，不补完整扫描；幅度单独校准|

ego要先扰动同一sensor pose，再反算head，安装误差与pose误差分开；不能让编码器绕过mask拿到干净camera轨迹。共同world旋转／平移是坐标增广，人体、场景、相机和目标都应一起变换；它不是让模型恢复绝对世界“真坐标”的测量噪声。身体／场景的相对重建误差可以单独模拟，但需要明确语义与不确定性。

**不能加白噪声后强迫模型逐帧匹配观测加速度。** 例如30Hz下每坐标1cm独立位置噪声，会产生约0.42m/s的速度标准差和22m/s²的加速度标准差。建议使用相关噪声；观测导数的不确定性、缺失stencil和离群权重需同步调整，必要时使用与推理一致的短时估计。导数仍来自输入，不可拿GT干净v/a偷换观测特征。改变噪声／GT质量规则后重新核定监督梯度尺度，避免v/a项压过姿态／场景学习。

递增噪声的课程有直接参考：[RoHM官方训练说明](https://github.com/sanweiliti/RoHM#training) 明确分阶段提高输入噪声，PoseNet配置的translation噪声从[第一阶段1cm](https://github.com/sanweiliti/RoHM/blob/main/cfg_files/train_cfg/posenet_train_stage1.yaml)增至[第二阶段3cm](https://github.com/sanweiliti/RoHM/blob/main/cfg_files/train_cfg/posenet_train_stage2.yaml)。这里借鉴递增难度原则，不照搬其AMASS语料、模型拆分或训练步数。

## 5. 兼容SMPL与SMPL-X的范围

共享body22的命名、顺序、parent树和201D输出，并不表示相同骨长或vertex ID。当前native rest／betas／scale分别进入FK和skinning，SMPL的6890顶点与SMPL-X的10475顶点各自选67个anchor。已用CPU资产缓存核查两者body22拓扑相同。

因此当前可以统一生成身体动作，不能声称统一生成完整SMPL-X手指／表情。体型目前仍属于数据集已知校准；接真实前端时需使用其估计体型或明确oracle体型实验，对照方法也要获得相同体型／尺度信息。

## 6. 建议训练顺序

|阶段|工作与退出条件|建议执行量|
|---|---|---|
|准备|新benchmark划分；GT质量sidecar接入loss；噪声配置接通；固定validation与原始前端基线；全量raw30准备和零步检查|不进行优化器更新|
|短诊断|从正式TRAIN选8–16个固定窗口，验证观测／朝向／骨长和场景梯度；验证另设独立recording及噪声seed|约500–1000步；只作工程验收，先看生成viser|
|主体训练|固定192帧、30Hz，共享输出；增加真实遮挡和噪声难度；场景从一开始可用，位置／旋转反馈逐步增强|先检查5k–10k里程碑；首轮总计划可设30k–50k，是否继续看验证趋势|
|前端适配／消融|接前端完整身体soft proposal及实际confidence；比较无场景、有场景、局部反馈与共享／独立模型|另列实验预算，不自动续训|

首轮先保留现有全局场景结构，避免同时重写所有模块。主体建议起点是ego/exo约50%/50%（当前仍是70%/30%），exo再按遮挡类型／部位／连续时长分层，而不是只按片段长度。必须有轻／无遮挡样本，检查是否破坏可靠动作。目标是exo提升，可根据固定validation分模式趋势调整比例，不默认牺牲ego。

全局／局部条件从开始训练；第一阶段主要学正确全身、root／heading及去噪。contact有可靠标签即可辅助训练，但实际root／rotation反馈增益可以先为0，再随主体阶段增加；同时实现此前的query可靠性门控。需要把这些phase规则写进同一个可复现配置，而不是途中临时修改配置冒充同一实验。当前训练循环没有课程或feedback ramp，以上尚待实现。

基础优化器可先沿用batch=2、AdamW lr=1e-4、weight decay=0.01、clip norm=1，是否合适由短诊断和固定validation决定。当前1000步cosine扩散、默认DDIM20步可保持作起点；没有足够结果证明只有2次去噪合适。

**当前不能直接按上述流程启动训练。** 尚缺正式validation循环、GT质量mask、实际噪声配置接口，以及JSON监督尺度与loss固定常量的统一。RICH／TRUMANS正式exo采样尚未接入；TRUMANS恢复R/t后还需K／光学轴约定，或明确用合成exo增广。也不能把六例诊断缓存当正式训练语料。

当前EgoBody raw导出会同时读取PV，即使之后采样master exo；整理新库存时还应检查是否因此丢弃了可用exo帧。模式可用性应分别记录：缺PV不必自动意味着master exo不可用。新benchmark需要正式test索引／导出与评估支持，当前native30入口只覆盖train／validation，不能只改一个split字符串。

另外，当前resume要求配置及planned steps完全一致：不能先把总预算设为5k，再把同一cosine schedule随意续成50k。应区分总计划和本次运行截止里程碑，并记录阶段／RNG；该接口也需要补齐。每个训练run保留一个原子可恢复last.pt，不复制大量编号权重。

固定报告：遮挡部位的世界坐标误差、可见部位退化、root／heading、全身误差、抖动／滑脚、可判定的接触／穿透，以及相同离线窗口与场景预算下的推理成本。比较前端原输出、动作修复无场景、加场景、加反馈，才能判断场景是否发挥作用。输入GT模拟／真实前端、已知／估计体型、内部／官方划分都须分开报告。
