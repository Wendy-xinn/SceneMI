# Method v2：实验配置与验收草案

2026-10-29，已按本轮对话修订 EGO 噪声、来源权重、B预算、EXO参考与时序目标。对应 [新方法正文](METHOD_V2_REVIEW_zh.md)，参数同步记录在 [审阅 YAML](../configs/method_v2_proposal.yaml)。**尚未实现或执行，不是训练启动指令。** 数值均为预声明建议，审阅后固定；新架构的显存、速度、完整验证耗时尚未测量。

## 1. 是否从 AMASS 重新训练

建议正式主模型从 AMASS 随机初始化开始，两模式共同训练，不继承旧多尺度混合、SRRD、EXO adapter 或其 optimizer。保留简化的单尺度参考 A。新结构改变了条件编码、输出头、粗动作头与预测目标；直接续训旧 joint 无法干净比较新方法。

不需要重建所有数据资产：复用核验后的 AMASS split、人体模型、坐标与原生来源清单。新模型不读取旧 run 的学习参数。既有可复现模型和结果继续保留。

执行顺序为：实现/输入验收 → 丢弃更新的 CPU/CUDA 检查 → 512-update 可行性 pilot → 固定本版协议后训练 A/B/C。pilot 不作为主表模型，不与正式预算混写，也不在 pilot 后悄悄延长预算追结果。

## 2. 训练阶段、预算与学习率

以下“窗口数”是 observation windows：同一 AMASS 动作分别构造 EGO、EXO 输入计为两个观测窗口。A 阶段每个 update 使用 64 段动作，各构造两种观测，因此有 128 个观测窗口、64 个底层动作窗口。原生来源两模式成员不完全相同，B/C 不强行假设一一配对，逐条记录实际 motion identity。

| 阶段 | update | 有效 batch | 每模式窗口 | 总观测窗口 | 初始/最高学习率 |
|---|---:|---:|---:|---:|---|
| A：AMASS joint scratch | 10,000 | 128=64+64 | 640,000 | 1,280,000 | 全部可训练参数 1e-4 |
| B：原生无场景适配 | 6,000 | 128=64+64 | 384,000 | 768,000 | 观测/边界 1e-4；任务粗/最终头 5e-5；EGO 体形头 1e-4 |
| C：共享场景条件 | 3,000 | 128=64+64 | 192,000 | 384,000 | 共享场景编码与投影 1e-4 |

单个正式主模型三阶段共 19,000 updates、2,432,000 观测窗口；不包含 pilot、对照、多 seed 和评估。旧 AMASS joint 110k×batch8 为 880,000 观测窗口；新 A 预算约为其 1.45 倍，不能只比较 update 数或暗示完全同预算。B的3k改作中途检查，6k是本轮建议的预声明上限，不是已证明足够收敛，也不在观察结果后自动延长。

所有阶段 AdamW，betas=(0.9,0.999)，weight_decay=0.01，bias/normalization/embedding 不衰减；warmup 为本阶段前 5% 窗口，之后 cosine 到峰值 lr 的 0.1。新阶段重置 optimizer 和 scheduler，不继承上一阶段动量。所有参数组按名称列出并断言无遗漏、无重复；禁止配置项未使用。

阶段 A/C 对实际可训练参数总梯度裁剪至 norm 1.0；阶段 B 按 EGO、EXO 各自参数集合独立裁剪至 1.0。A/C 共享梯度累计后只进行一次 optimizer step。首版 FP32、TF32 off，不启用 AMP、CFG、PCGrad 或动态 loss reweighting。

### 为什么先冻结 B 的主干

A 学相同 clean-motion 目标，B 主要处理真实相机、原生人体格式和前置估计分布。B 先冻结 trunk，允许两套观测路径和小型任务头适配，降低改写另一任务基础运动表示的机会；AMASS replay 继续约束每条任务路径。

这是有风险的容量选择，不保证固定主干足以完成跨域适配。若 B 失败，报告训练/验证曲线、coarse/final error 和各来源结果，再讨论有限解冻。有限解冻不是默认隐藏的 B2 阶段，也不预先开多个解冻分支。

## 3. batch、累积与资源

| 阶段 | 目标有效 batch | 初始实际 microbatch | 每模式累计次数 | 每次 update 的操作 |
|---|---:|---:|---:|---|
| A/B | EGO64+EXO64 | 每模式 16 | 每模式 4 次 | 所有 microbatch 的归一化梯度累计后更新一次 |
| C | EGO64+EXO64 | 每模式 8 | 每模式 8 次 | 两任务都对同一 scene 参数集合累计梯度，更新一次 |

显存验收后可将 microbatch 增至 32 或 64、相应减少累计次数；有效 batch、窗口流、loss 归约和 optimizer update 数保持不变。不得把 microbatch 16 的每次 backward 当成一次有效更新，也不在 OOM 后自动减半有效 batch 改变训练协议。

按 source/mode/sample 的预声明权重缩放每个 microbatch 的 loss，不能对各 microbatch 的均值简单平均，造成最后小块或来源配额权重错误。GroupNorm 不依赖跨样本统计，仍需在容差内验证不同累积方式的 loss/gradient 一致性。

已有无场景旧结构在真实数据上的 batch256 prefetch 约 47.6–50.0 windows/s、峰值分配约 4.4 GiB，只能用于说明数据预取值得保留，不能推断本版完整主干训练或场景查询的显存与速度。依据：[旧 benchmark](../runs/replay_benchmark_large/THROUGHPUT.json)。

拟定运行预算：12 GiB RAM、4 CPU cores；必要时在用户允许范围内将 RAM 提至 16 GiB。按实际空闲显存推导 allocator cap，保留 4 GiB 物理余量与额外 1 GiB CUDA allowance。新任务须重新经过现有资源准入，旧 run 的共享授权不自动转移。CPU 数据准备、H2D、主干前后向、几何构建和验证分别计时；不依据空闲显存直接加大 batch。

## 4. 数据与抽样

### A：AMASS

沿用 `configs/amass_splits.json` 的 train/val/test，模型只使用 train 更新；原有 family/反射增强身份记录保留。均匀 family → variant → 有效起点，128 帧连续时钟不压缩。每个 update 的 EGO/EXO 使用相同底层 motion crop/yaw，观测噪声分别生成；跨阶段不把相邻窗口跨 split 使用。

EGO 由虚拟相机/头部轨迹构造条件，不输入个体真实 shape。修订为50%原观测、50%轻量连续 SE(3) 测量扰动；pilot控制点标准差建议位置1–5 mm、旋转0.1–0.5度、时间间隔0.2–0.5 s。重新计算派生速度、坐标与置信度；保持目标不变。相机与点云若共用定位误差，需保持对应合同。主配置暂不加camera dropout/长时漂移；正式训练前需以训练侧设备误差校准，当前数值不是经验最优或已验证真实噪声。

EXO 合成分布优先沿用既有可比较设置：

| 项 | 建议值 |
|---|---|
| identity / tracking / occlusion / gap | 0.20 / 0.35 / 0.25 / 0.20 |
| beta 扰动标准差 | 0.15 |
| 普通姿态扰动标准差 | U[0.015,0.04] rad |
| 遮挡姿态扰动标准差 | U[0.12,0.25] rad |
| 根扰动标准差 | U[0.01,0.04] m |
| 姿态/根 AR 系数 | 0.85 / 0.95 |
| 总合成 gap 长度 | 8–32 帧 |
| gap 位置 | prefix / interior / suffix 等概率 |
| gap 样本中的多段比例 | 0.5；两段各 4–16 帧，总计不超过 32 |

本轮将EXO抖动提升为正式验收问题。第一轮可比pilot保留上述AR难度与原验证噪声，不通过降低验证噪声制造收益；随后校准前置估计的低频漂移、短时抖动、遮挡切换及根/局部运动误差。候选主噪声是低频漂移加小幅高频抖动，具体比例/幅度须由训练侧真实前置输出或单独输入审计固定，正式训练前完成；不能只把rho调大就声称模拟更真实。

多段分布是本版新声明的变化，旧 single-gap 成绩不能直接作为同分布架构对照。比较新结构与旧结构时，另固定一套双方都重训/兼容的共同分布；无法实现时只报告各自协议。

可靠性来自模拟前置协议，并向条件提供随机扰动后的分数以测试偏差：首版对有效非 identity confidence 加 U[-0.1,0.1]，clip 到 [0,1]；缺失项仍为零。它不是依据当前样本实际 GT 误差计算的 oracle certainty，也不是已标定的真实估计器概率。实际 estimator confidence 的标定/映射只用 train/validation 前置输出建立，测试前锁定。

### B：四来源无场景适配

AMASS replay先从1/2降至1/4；其余3/4按原生可用窗口容量的平方根分配，避免平均过采样小来源或完全按帧数压倒小来源。这里只是待验证初值，不是最佳采样分布的主张。

本轮只读train清单与有效性数组，复核得到以下容量；每条序列贪心取不重叠128帧窗口，见 [容量记录](../runs/data_inventory/TRAIN_WINDOW_CAPACITY.json)。它是容量代理，不是去重family后统计独立样本的数量。

| 模式/来源 | EgoBody | TRUMANS | GRAIL |
|---|---:|---:|---:|
| EGO不重叠窗口容量 | 1,499 | 7,828 | 9,834 |
| EXO不重叠窗口容量 | 1,756 | 7,828 | 9,834 |
| EGO原生内平方根权重 | 17.10% | 39.09% | 43.81% |
| EXO原生内平方根权重 | 18.26% | 38.54% | 43.20% |

每模式batch64按累计配额缺口取整，而非每次独立四舍五入。来源内仍按family→variant→有效起点抽样；每来源loss均值按上述同一目标权重组合，避免再次乘样本频率。验证仍报告来源等权macro和各来源明细，不把训练权重直接当选模权重。

旧3k×每模式64在AMASS占一半时，每个原生来源约32k观测窗口；相对上述容量，EGO约21.3/4.1/3.3倍。此比值不是真实epoch，也不保证随机采样覆盖。问题并非3k必然完全没见到大来源，而是小来源重复多、有效覆盖和收敛均未验证。新6k/25% replay保持每模式AMASS约96k曝光，同时把原生总曝光从96k提升到288k；按来源记录unique motion/family覆盖、重复率及验证曲线后才能判断是否训到位。

三原生来源只使用 `configs/scene_data_current.json`、checked loaders/source index 的有效范围。GRAIL 两模式 motion 范围一致；TRUMANS EXO 使用已完成的 moving-camera overlay；EgoBody 两模式实际目标人/成员差异保留清单，不伪装完全配对。B 不读取场景几何进行模型条件化。

当前数据合同仍写有 `exo_payload_adapter_ready=false`。本版 EXO pose/shape、独立位置/旋转有效性、boundary token、坐标与 observer metadata 的接口必须实际实现并通过验收，不能从 source-index 通过推断已 ready。

### C：原生共享场景

每模式沿用B的原生内部平方根容量权重；source配额与loss权重保持一致。AMASS没有场景，且B所有参数冻结，因此C不把训练batch浪费在scene分支严格无梯度的AMASS样本上。AMASS no-scene保持能力仍在验证中检查，并验证参数/输出恒等。若场景多样性审计显示动作容量不足以代表场景覆盖，应在正式训练前修订并固定采样，而非默认其必然适合几何学习。

20% 窗口关闭场景注入，scene/motion-only 控制使用相同 dropout 决定。history8192、current512/frame、K8、半径0.6 m、voxel0.05 m；关系法向有效性沿用已有观测几何规则，具体规则与源码版本封存。只读真实已观测静态 prefix 和逐帧点，不读取目标附近的隐藏完整网格。

## 5. loss 配置的完整口径

各项公式见新方法正文；以下是唯一的系数表，YAML 与此保持一致。

| 项 | 定义/量纲处理 | 系数 |
|---|---|---:|
| root | root translation Huber，尺度 0.1 m | 1.0 |
| pose | root-relative FK joints Huber，尺度 0.1 m，排除 root | 1.0 |
| rotation | 局部旋转矩阵差平方和 / 6 | 0.5 |
| velocity | 全局 joint velocity Huber，尺度 1 m/s | 0.1×alpha_bar |
| acceleration | EXO-only，相对GT的全局joint加速度误差Huber，尺度2 m/s² | 0.05×alpha_bar |
| observed | 按传感器测量方程比较有效位置/朝向，confidence 加权 | 0.1×alpha_bar |
| shape | EGO 的标准化 beta、rest，加 0.1×模板 CE | 0.1 |
| coarse | coarse root+pose+0.5rotation，不含 vel/obs/shape | 0.25 |

Huber threshold=1。alpha_bar 是 diffusion cumulative signal，使用平方信号，不与 sqrt(alpha_bar) 的场景注入 gate 混淆。C 阶段 coarse/shape 是常量，不加入可训练目标重复计数；报告时可单独保留其指标。A/B 计算完整目标。

EXO pose/rotation/velocity/acceleration 分别按可见/隐藏组等权归约，没有隐藏组时不凭空创建零组把loss减半；加速度只有三帧均可见才归入可见组。目标是导数误差而非零加速度，避免鼓励静止或抹去快速动作。零有效观测时L_obs=0，shape/template无监督时对应项mask，不制造NaN。所有最终loss先sample mean，再source mean，再mode mean。

## 6. 最小实验矩阵

### 基础任务与边界

| 对照 | 回答的问题 | 预算口径 |
|---|---|---|
| A-joint：本版 | 共享运动表示能否承担两类任务 | 10k×128，两模式各 640k 观测窗口 |
| A-ego-only | joint 是否损害/帮助 EGO | 10k×64 EGO，与 joint 的 EGO 曝光一致 |
| A-exo-only | joint 是否损害/帮助 EXO | 10k×64 EXO，与 joint 的 EXO 曝光一致 |
| A-joint-no-boundary | 新边界结构是否有效 | 同 joint 曝光、数据、noise、优化与选模预算 |

单任务对照的总计算比 joint 小，必须同时报告；不能声称这是总计算匹配。单任务保留其对应 coarse/final head、同一初始化 trunk、对应观测/shape 和相同每模式观测流；目标乘 0.5，以匹配 joint 中该任务梯度的显式系数。参数按模块名/seed 初始化，避免另一分支是否存在改变公共参数的初值。

单任务与joint的B也沿各自选定A正常训练：单任务有效batch64、6k updates，来源比例与joint对应模式一致。只有在相同原生适配后比较，才能声称统一模型在目标域是否保持单任务能力。无边界分支先看A；仅在必要性结论依赖目标域时继续其B。

若需要将“分任务小头”作为贡献，再增加共享读出头的同预算对照；首版不把该一般结构本身列为创新，也不预先扩大为多套专用 decoder。

### 场景

所有 C 变体从**同一个固定 B** 开始，不串行继承彼此。

| 对照 | 变化 | 公平性 |
|---|---|---|
| C-shared-scene | 一套共享 scene encoder，真实几何 | 主方案，3k×128 |
| C-motion-only | 相同 MLP/时间层/投影；关系12D置零，保留12D粗人体状态和region ID；evidence gate置一（场景 dropout 时仍置零） | 参数完全匹配；同曝光、同种子、同选模预算，具有动态人体输入 |
| C-separate-scene | 两任务各一套同结构 scene encoder | 直接检查共享是否损害某个任务；可训练场景参数约2倍，明确报告，不伪称等参数 |

另外用固定 C 权重诊断正确/同来源错配/空场景，但这些开关不能替代上述重新训练的控制。未知区域输出恒等由空 evidence 检查；错配保持点数/覆盖类型尽可能一致，禁止靠换成全零输入获得简单结论。

完整 v2 对旧结构的架构对照需要旧模型在共同数据/腐蚀/曝光协议下重训。现有旧 run 只作历史参考与部署结果对照，不在表格里冒充同预算消融。

上述首个seed的最小矩阵共有4条A、3条B和3条C分支，修订后合计67,000 optimizer updates、6,528,000观测窗口（A 3.84M、B 1.536M、C 1.152M）。它包含主模型路径，不能再把主模型2.432M重复相加；不含pilot、可选分支、旧架构重训和评估。锚点简化与acceleration先做短诊断对照；若声称其中某项是性能贡献，必须追加相应完整预算对照并另报成本，短pilot不能代替正式消融。不同阶段/分支每窗口计算量不同，窗口总量不能直接换算GPU小时。

## 7. 验证、选模与成功条件

训练只读 train；候选选择只读 validation；独立 test 在结构、模型、推理和指标锁定后执行。现有 val 序列与有效区间不重划，stride128、含 end-aligned tail；给所有变体相同窗口、前置腐蚀与噪声。

默认两模式都 20 DDIM，CFG off；固定 seed777，完整结果按 family 聚合，再对数据集等权。验证噪声按 case_id 与 seed 生成，不依赖 batch 分块，允许调整评估 batch 而不改变噪声。

| 阶段 | 预声明完整验证 checkpoint |
|---|---|
| A | 2000、4000、6000、8000、10000 |
| B | 初始化0；1500、3000、4500、6000 |
| C | 初始化0；750、1500、2250、3000 |

完整验证频率是预算的一部分。B/C 的 step0 只确认父模型和评估接口，不作为该阶段训练成功的候选；没有合格训练后候选时，记录该阶段失败并保留父模型。训练中可运行固定小清单作故障诊断，但不据其另留有利 checkpoint 或跳过不利的完整验证。

主选择分数为 0.5×EGO MPJPE/100 mm + 0.5×EXO分数；EXO 分数为 tracking MPJPE、occlusion区域MPJPE、gap区域MPJPE 各除100 mm后等权平均，来源仍等权。100 mm 是固定归一化常数，不从某个表现很差的初始化动态取分母。单任务仅使用其对应分数。

**资格先于综合分数：**

- A：EXO tracking/occlusion/gap 的实际可见区域 MPJPE 不超过输入 +1 mm；保存 identity 的真实平均/最大改动，取消旧版结构性精确复制主张。joint 的晋升还要求 EGO 不超过对应单任务 +5 mm，EXO 的 tracking/occlusion/gap 各主指标不超过对应单任务 +1 mm。门槛是草案，不是保证能通过。
- B：两模式 AMASS 各主指标相对各自 A 不超过 +5 mm；EXO 每原生来源可见 MPJPE 不超过该来源输入 +1 mm。选择分数分别计算 AMASS与原生部分并保留明细；晋升还检查对应完成 B 的单任务对照，EGO +5 mm / EXO +1 mm 上限不变。
- C：每个原生来源 EGO MPJPE、EXO 可见 MPJPE 不超过同 B 的 +1 mm；EXO occlusion/gap 的三来源平均区域 MPJPE 不退化。no-scene/empty-evidence 输出须与 B 一致。若没有合格候选，保留失败结论，不自动放宽门槛。

B选择分数按本节mode分数公式，AMASS权重1/4、三原生来源各1/4；C只对三原生来源各1/3。A只对AMASS family聚合。这里原生来源等权是报告和选模目标，不是训练抽样分布，不能混为一谈。所有候选同时保存每任务/来源分数，不只留一个总分。

场景收益需优于 C-motion-only，而不只优于冻结的 B；共享是否适当需与 C-separate-scene 一起判断。声称“同时促进”要求两任务各自的改善成立，不能用一个任务的收益抵消另一个任务退化。

主模型和关键结论的对照建议三 training seeds [270701,270702,270703]；先以第一 seed 做结构诊断，不能从三 seed 中挑最好一个汇报。报告跨 seed 均值/标准差及 family 配对差值范围；固定权重下 family bootstrap 不代表训练随机性。

### 指标

- EGO：MPJPE、标准 PA-MPJPE、root/head error、root-relative MPJPE、速度误差和 jerk。
- EXO：上述指标加 visible/occluded/gap-region error、相对输入改动、identity重建误差；按缺失长度8/16/32/64帧、单/多段、prefix/interior/suffix分组。64帧为超出合成训练范围的压力测试，明确标注。
- 场景：coarse查询误差/覆盖、实测几何区域的支撑/脚滑等可评指标，正确/错配场景差异；无签名几何不报告伪造的封闭体穿透真值。
- 效率：实际参数量、可训练参数、FLOPs（可测时）、主干调用数、端到端窗口时间、显存与数据/几何等待。20次新EXO调用不能与旧16次比较后宣称必然更快。

## 8. 短预算验收要回答什么

先做 CPU/小 batch 检查与真实数据 CUDA probe，更新全部丢弃：

1. 模式、shape/template、坐标、观测掩码和 first-hit 时间一致；GT motion/shape 混入 inference payload 时被拒绝或无效，缺失值 poison 不改变有效结果。
2. 旋转合法、FK一致、loss/grad有限；两任务粗/最终头与观测编码得到预期梯度；共享scene投影第一次更新有梯度，前级MLP在零投影打开后有梯度。
3. 两模式单独和梯度累积执行一致；按mode/source拆分的loss归约与一次完整batch一致（声明FP32容差）。
4. 关闭scene与无场景模型相同；真实scene变体无evidence时也须相同（motion-only控制有意以人体状态作为evidence，不适用该条）；C阶段只有scene参数改变，A/B参数digest不变；冻结decoder依然向scene注入反传。
5. 边界检索覆盖位置/旋转有效性不同、无左/无右、无两侧、多gap、完全无该关节观测；不把填充值当端点。
6. checkpoint恢复完整模型、optimizer、scheduler、CPU/CUDA/NumPy/Python RNG与数据流；验证恢复后下一更新与未中断参考一致。每100 updates原子保存last，拒绝污染/重复日志尾。

然后固定 512-update pilot（65,536观测窗口，A每模式32,768）。固定小验证清单只用于可行性：是否学得动、EXO可见误差是否快速恶化、coarse是否能定位、任务梯度量级、数据/几何耗时。短pilot不认证最终性能、正迁移或创新性。

C阶段需在B形成后再做其真实geometry probe；不能在只有随机A模型时就认证共享场景接口有效。完整预算、microbatch和完整验证耗时都需通过实际profile再确认。

## 9. 资产和运行独立性

每个阶段导出完整 inference model 状态，固定公共人体资产另列；last 另含该run的 optimizer/RNG 以恢复。B/C明确继承哪个父模型，只继承模型参数，不继承optimizer或不相关adapter。各对照引用同一不可变父模型，不读另一个对照的last。

命名使用稳定的 `scenemi/` 模块、`configs/` 配置和既有 module CLI；本版配置前缀为proposal，不提供当前不可运行的伪命令。运行记录绑定模型schema、显式源码闭包、数据合同、随机流和资产SHA；历史记录溯源与实际运行依赖分开。

last每100updates覆盖；只有预声明完整验证点临时保留模型候选，评估并完成依赖检查后留selected+last，其余按既有records-only规则清理。固定step评估的pilot只留一个last，不自动复制best；丢弃其更新后保留配置/指标/身份记录。

## 10. 审阅时最重要的取舍

本版保留简化的单尺度参考A，取消旧多尺度固定混合与双差分；两模式使用同一clean-motion目标，EXO采用无硬半径的参考相对读出。边界改成端点token，场景只有一套共享参数。为在一次前向内定位场景，新增轻量粗头；为表达两任务差异，保留小型任务头。它们都需要对照支持。

建议优先审阅：分任务读出是否足够、粗头查询是否可行、B冻结主干是否过于保守。通过这三个问题后再实现完整预算。下面的新诊断单独列出，不据少量预览临时增加本版模块。

## 11. 另一会话已落盘问题：证据与后续建议

本节只依据 [回放说明](REPLAY_VISUALIZATION_zh.md)、[PLAYBACK_AUDIT.json](../runs/scene_joint128_replay/visualization/PLAYBACK_AUDIT.json)（读取版本更新时间 2026-10-29 08:25 UTC）和 [现有训练诊断](SCENE_JOINT_TRAINING_DIAGNOSIS_zh.md)，不代表已读取另一会话的聊天内容。按当前要求，本节不改变前述 v2 结构或启动额外实验。

### 11.1 这些预览能说明什么

九例来自 validation 的特定误差分位，不能代表全体平均。显示 mesh 与实际 FK 的最大差小于 5×10^-7 m，支持“当前看见的轨迹问题存在于导出预测”，不能据此排除所有坐标/标定/模型问题。场景仅用于显示，不能把这些偏差解释成 scene encoder 导致。

CPU 预览与原 CUDA 验证即使同为 seed777，也没有相同噪声张量，因而不能逐例当作原验证结果重现。后续应保存原始初始噪声，或统一在一个确定设备生成后搬运；保存 case_id、噪声 SHA 与输入 SHA。不要通过反复采样选一个顺眼的预测。

### 11.2 EGO：先分清全局偏移、头部条件与体形

例02的根位置平均误差约382 mm，最高正向高度误差约603 mm；例07根误差约95→389 mm。audit 的 `root_height_min/max_mm` 是 **预测根高度减 GT 根高度**，不是人体离地高度，也不能直接当作脚穿地量。例02预测头部到输入相机的距离约0.105–0.476 m，而 GT 对应距离稳定约0.127 m，值得优先核查观测一致性。

建议后续固定同一预测，分别报告：world root error、root-relative pose error、由估计体形与设备标定得到的完整 head-camera SE(3) 残差，以及估计体形误差。相机与头关节存在正常安装偏移，不能以“距离必须为零”判错；仅有距离范围也不足以定位朝向错误。

先检查地面原点、相机外参、人体格式和 head mount 的合同，再判断是表示问题还是学习不足。可额外用 GT shape 作一次离线诊断来量化体形影响，但必须标为 oracle，不能进入部署输入或常规验证。若校准正确而观测残差仍大，再比较相机约束下恢复 root 的表示；目前不先加新的 neck/head/root 多项损失或硬投影。

### 11.3 EXO：输入已经抖动，输出也远未恢复真实时序

以下数值为有前置估计的连续帧中，**22关节的平均加速度模长**，不是 root 加速度，也不是相对 GT 的加速度误差：

| 预览案例 | 输入 / 预测 / GT（m/s²） | 速度误差，输入→预测（mm/s） |
|---|---:|---:|
| EgoBody 03 | 34.52 / 18.30 / 1.94 | 751→435 |
| TRUMANS 06 | 45.12 / 23.56 / 2.37 | 1015→587 |
| GRAIL 09 | 33.13 / 17.81 / 1.30 | 761→443 |

模型降低了这些输入的时序噪声，但残余明显；这既不支持“模型没做任何修复”，也不支持“抖动已经解决”。加速度越小不等于动作越正确，还要检查相对 GT 的速度/加速度误差、动作幅度与快速动作保持。

本轮对同三例缓存另做只读根轨迹分解，未运行新模型推理，见 [时序分解记录](../runs/scene_joint128_replay/visualization/TEMPORAL_COMPONENT_AUDIT.json)：旧多尺度A的根加速度误差约13.12/13.52/11.58 m/s²，最终预测约12.94/13.31/11.44；根部高频修复很小。仅用单5帧滤波并补缺的root参考约4.35/5.25/3.70，但TRUMANS/GRAIL全窗口root位置误差比旧A分别增加约3.8/5.4 mm。该诊断支持进一步比较简化参考与针对导数误差的监督，不证明完整单尺度运动学参考或新模型性能更好。此处重建的是replay的MultiGapTemporalBasis根分量，不把它冒充正在运行的scene分支最终结果。

现有 audit 按“该帧有前置估计”筛连续差分，已排除 outage 边界，但不是逐关节可见性筛选。后续细化为逐关节、逐模态有效连续段，单独报告遮挡、重新出现边界和无遮挡部分。也要核验合成 AR 噪声在30 Hz下的速度/加速度分布是否接近真实前置估计；不能仅调大 smoothness loss，把过强输入噪声的问题转成过度平滑输出。

例03首帧输入 root 几乎等于 GT，需结合现有合成器的首帧 root-error 归零约定解释，不能把这一帧当成真实 tracker 独立提供的精确约束，更不应据此给所有样本强制首帧复制。

### 11.4 前缀缺失和长序列需要另外看

例09是[0,24)的前缀缺失；首帧输入填充值 root 误差约831 mm，预测仍约721 mm，缺失区间 root 平均约354 mm。它没有左边界，不能与有双侧端点的 interior gap 混报。建议按 prefix/interior/suffix 分组，画误差随距有效观测的时间变化，并检查重新观测时的跳变。v2移除旧修正半径，允许纠正更大偏移，但并不保证能够正确外推。

每例只有128帧、约4.27秒，当前无法证明“长时间累计漂移”的来源或严重程度。后续按第7节的共同世界坐标长序列拼接协议，分别测单窗口误差、重叠一致性与整段轨迹误差；不通过逐帧根对齐、贴地或可视化居中掩盖。

处理顺序建议为：先固定预测/噪声身份和坐标标定，再校准输入噪声与逐关节时序指标，最后依据分解误差决定是否调整表示。上述检查结果出来之前，保留本版最小结构，不将每个预览症状对应成一个新模块。
