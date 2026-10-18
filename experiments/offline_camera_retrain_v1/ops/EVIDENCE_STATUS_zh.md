# 当前方法的证据与最短检验路径

## 10月18日：EXO内部适配初筛与checkpoint选择复核

从41k冻结主干、训练1,955,616个EXO门控内部参数的1000步试验已完成。实际更新累计34.5分钟；最终完整checkpoint独立CPU加载通过，基础权重摘要保持，原EGO六段DDIM预测逐值保留。固定开发验证的EXO隐藏误差三语料宏平均clean 114.53→86.99mm，noisy 135.57→129.03mm；TRUMANS/EgoBody改善，但RICH三窗clean 80.88→89.02mm、noisy109.94→141.59mm，双脚运动幅度误差也退化，故预注册验收未通过，不能宣称已得到整体可用版本。可见关节分组保护通过不等于每窗均改善。[完整结果](../runs/native30_exo_adapter_v1/RESULTS_zh.md)、[验收](../runs/native30_exo_adapter_v1/ACCEPTANCE.json)、[Viser前后对照](http://localhost:8802/)。权重和数组暂保留用于明确计划中的扩展复核。

主线best=41k的原选择口径只有一个固定RICH窗口，证据不足。扩展复核在通过20项CPU回归及首窗4次真实GPU预测后，已于10月18日11:40启动后台完整续跑：[运行状态](../runs/native30_checkpoint_review_v1/STATUS.json)、[预先固定的协议](../runs/native30_checkpoint_review_v1/PLAN.json)。现存41k/55k同条件比较RICH全部206窗/27录制，加TRUMANS50窗/33录制、EgoBody50窗/16录制，共1224次预测；按录制宏平均为主汇总，另报逐窗、有效计数加权及三数据集等权结果，完成后自动独立CPU复算。此前206窗全VAL是15,100步，不能给41k/55k复用指标。新复核完成前仅称41k为临时开发候选；本次比较也不代表遍历过全部历史权重，不从TEST回选，不自动覆盖原best/last。该评估是EGO输入，不能代替EXO遮挡泛化验证；EXO后续单变量实验建议另见[记录](../runs/native30_exo_adapter_v1/diagnostics/NEXT_EXPERIMENT_zh.md)，尚未启动下一轮训练。

冻结41k的独立RICH TEST现已670/670预测并经CPU复算通过：335窗、52录制、53人体流，clean/noisy等窗口World MPJPE155.67/169.48mm、PA113.94/121.10mm。该结果含观测场景和已知人体profile，不能当AMASS-only论文同协议排名，也不用于本轮调参或选checkpoint。[结果](../runs/rich_test_ego41k/SUMMARY.json)、[CPU核验](../runs/rich_test_ego41k/CPU_RECHECK.json)。

## 10月17日21:20最终复核：55k正常完成，验证平台与轻微回退，保留41k开发候选

主线已于 **2026-10-17 20:49:24** 正常完成 55,000 步，worker 返回码 0。真实 `last.pt` 为 55k，模型/Adam 有限、全部 RNG 与 sampler/counts 完整；`best.pt` 为 41k 模型副本，无 Adam/RNG。365 份源封印及配置核验一致。best 的保存规则仍只依赖一个固定 RICH VAL 窗口，不是全 RICH 最优。[最终统一报告](../runs/native30_gait_v2_ego55k_oct15/final_review/REPORT_zh.md)、[训练与固定验证曲线](../runs/native30_gait_v2_ego55k_oct15/final_review/TRAINING_CURVES.png)、[checkpoint 审计](../runs/native30_gait_v2_ego55k_oct15/final_review/CHECKPOINT_AUDIT.json)。

最后两个完整 5k 段总 loss 为 0.154302→0.150938（−2.18%），加权 stance 项为 0.00930831→0.00929848（−0.11%）。拟合仍缓慢改善，支撑相关监督接近平台；不能仅由总 loss 宣称脚部支撑已经学好。

已完成 41k/55k 同输入、同 seed 配对：14 条不同录制 VAL 窗口＝12 RICH＋1 TRUMANS＋1 EgoBody，主比较统一关闭推理物理修正，5 个核心窗补报物理开启；76/76 份预测与独立 CPU 重算均通过。RICH 12 窗的 clean/noisy World MPJPE 从 41k 的 152.890/178.002 增至 55k 的 156.934/182.672 mm；GT 低速期水平脚速为 0.23434/0.28637→0.22884/0.26239 m/s，同帧 GT 为 0.06385 m/s；脚高误差从 45.424/61.994 增至 48.358/69.799 mm。即滑步略改善，但姿态和高度误差回退，可靠支撑仍未解决。[配对报告](../runs/native30_gait_v2_ego55k_oct15/final_review/checkpoint_comparison/REPORT_zh.md)、[76 份 CPU 重算](../runs/native30_gait_v2_ego55k_oct15/final_review/checkpoint_comparison/CPU_RECHECK_all.json)。

**当前保留 41k 为主要开发候选，55k 保留完整续训状态；不延长训练、不自动接 EXO。** 结论限于开发验证：最新比较不是最终权重的全 RICH、正式 TEST、旧55k或外部论文同口径胜出证据。此前冻结 15,100 步的 206 窗/27 录制、412 次全 RICH VAL 预测已完成并通过独立 CPU 重算，不能把它标成 55k 全量结果。[旧全量报告](../runs/native30_live_monitor_oct16/rich_full_validation_oct16/REPORT_zh.md)、[412 份验收](../runs/native30_live_monitor_oct16/rich_full_validation_oct16/VERIFY_COMPLETE.json)。

以下保留 15:20 及更早的历史检查记录，其中“正在训练”“当前 best”等状态只适用于各自记录时间。

2026-10-17 15:20 复核，依据既有模型、日志与评估产物更新记录；不修改正在训练的源代码或配置。

**判断：网络在学习；关闭独立物理修正后，学习式场景通路仍有配对推理收益。新版在 RICH 上已达到可用质量、比旧版更好、或者优于公开方法，这三件事目前都没有证实。** Loss 下降是拟合证据，不能替代固定验证集的进步。用户要求的是场景条件扩散主线，当前保留的是从零训练的 head-only Native30 reference-residual 扩散，脚步监督已恢复，目标 55k；独立小 EXO 残差支线没有继续训练，tracker 也未接入。

## 10月17日15:20复核：yaw-only短训退化，不采用；主线约46.1k继续55k

yaw-only 分支已完成同父状态 24,500→25,500 步的 1000 次配对更新，batch、场景、噪声 RNG、全局学习率与最终 sampler/RNG 匹配审计通过。14 个不同录制 VAL 窗口的 76/76 份预测独立 CPU 重算通过；公共非 reference tensor 指纹一致，reference motion 按预期改变、confidence 不变。

RICH 12 窗、统一推理物理 OFF，control→yaw_only 的 clean/noisy World 为 175.707/196.494→295.901/303.861 mm，PA 为 138.447/147.531→155.822/168.102 mm；GT 低速期水平脚速 0.25174/0.33178→0.33362/0.34862 m/s、脚高误差 45.821/55.664→73.031/71.030 mm、连续低速段预测水平位移 104.342/144.274→150.831/163.651 mm，均退化。物理 ON 仅覆盖 3 个 RICH 核心窗，World 仍由 172.240/201.916 增至 256.987/269.082 mm，不能用该子集宣称候选获救。**本轮不提升为长训版本；1000 步适配失败不证明 yaw-only 从头训练必败。** [决策与口径](../runs/native30_live_monitor_oct16/reference_yaw_trial/DECISION_zh.md)、[76 份独立 CPU 记录](../runs/native30_live_monitor_oct16/reference_yaw_trial/evaluation/CPU_RECHECK_all.json)。

截至 15:18，主线原 worker/supervisor 持续到约 46.1k，checkpoint 46,100，IO 重启 0。完整 43k/44k/45k/46k 千步段总 loss 0.156982/0.158796/0.155279/0.161480，gait loss 0.066866/0.067152/0.067505/0.069803；近期接近平台、验证仍波动。固定 RICH 单窗 clean/noisy 均值：41k 113.69 mm（best）、44k 125.31、45k 144.51、46k 136.99。单窗 best 不等于全 RICH 最优，全 RICH 完整结果仍为 15,100 步权重。主线继续原配置至 55k，未自动接 EXO。

按用户“失败实验只留记录”的指令，已删除本轮失败权重的两个硬链接、76份新包装预测、6份播放缓存及14份本评价专属表面缓存，释放1,955,061,760 bytes（约1.96GB）；父模型/control及76份原始预测共80路径SHA复核不变，85份封存依赖未变。[逐文件清理记录](../runs/native30_live_monitor_oct16/reference_yaw_trial/CLEANUP.json)。8802已退役，主线8800与TensorBoard6006保持可用；配置、逐窗指标与历史CPU审计保留。下列启动与发布记录保留为历史，不代表对应失败权重、数组或视图仍可使用。

## 10月17日13:20：yaw-only参考训练已启动，效果待验证（历史启动记录）

用户授权后启动独立yaw-only reference分支：同24,500步完整父状态续训1000步，对照复用已匹配的原reference/物理ON 25,500步版本，不重跑对照。保持完整头位置/朝向、原reference confidence、场景、物理ON、贴头/IK和loss，只改ego缺少可靠torso时的参考root水平朝向；10项CPU不变量通过。未训练参考可能从倾斜抬脚转为下穿，故不能宣称已经改善物理效果。训练后自动做14窗两种输入的配对验证、独立CPU重算及预选Viser案例。主线55k未改。[实验与状态](../runs/native30_live_monitor_oct16/reference_yaw_trial/EXPERIMENT_zh.md)。

按用户“失败实验只留记录”的指令，另清理已拒绝的去贴头/弱直立/组合与投影数组、旧8801播放缓存，以及这些评估目录专属的可重建surface缓存，释放655,388,672 bytes；连同no_physics共2,794,397,696 bytes。保留原baseline、四份逐阶段数组和19k诊断权重、全部历史指标/源码/审计。旧8801/8802服务已退役，本文旧时间点的视图链接是历史发布记录；新yaw-only比较结束后再发布新8802。[精确清单](cleanup_oct17_rejected_support_PLAN.json)、[执行与保留核验](cleanup_oct17_rejected_support_EXECUTED.json)。

## 10月17日12:53复核：脚步短训完成，关物理候选不采用

同一24,500步完整状态各续训1000步的control/no_physics对照已完成；1000次实际输入、学习率与RNG匹配审计通过，114次预测独立CPU复算通过。推理统一关闭物理修正，RICH12个不同录制窗口clean/noisy：control→no_physics的World误差175.707/196.494→177.353/197.147mm、GT低速期水平脚速0.25174/0.33178→0.25228/0.33446m/s、脚高误差45.821/55.664→47.838/58.606mm；可见表面GT校准误差也退化。本轮没有解决滑步或悬空，不以该候选替换主线。[决策与完整口径](../runs/native30_live_monitor_oct16/paired_foot_trial/DECISION_zh.md)。按用户后续指令，失败候选权重/数组及旧8802缓存已删除，历史配置、逐窗指标和核验保留，释放约2.14GB；父状态/control仍用于新yaw-only对照。

主线原进程持续到约42.7k。完整39k/40k/41k/42k千步段总loss0.16041/0.16138/0.16068/0.16327，脚步loss0.06728/0.06858/0.06928/0.07101，近期接近平台，不能再称各项稳定下降。best.pt已保留41k，固定RICH窗clean/noisy109.095/118.290mm；42k反弹至135.063/183.479mm。单窗best不等于全RICH最优，尚无当前权重超过旧55k的同口径证据。主线继续原配置到55k，未自动接EXO或更改训练。

## 10月17日00:45复核：尚未整体收敛，脚步分阶段诊断已完成

主训练约24k仍为ego-only且正常运行。完整千步总loss20k0.243535→21k0.239917→22k0.235545→23k0.222777，训练尚未平台。固定RICH窗19k best130.557mm、22k153.481、23k168.546；前后三个验证点均值152.628→152.197，近期姿态近乎持平而波动，脚速有小幅改善，不能称全验证已经收敛。

冻结19k新完成4次DDIM20逐阶段观测及1次无监控控制：最后t=0，四例平均GT低速期水平足速原始FK0.423→学习反馈0.411→贴头0.328→IK0.328→物理0.328m/s。主要脚部误差在修正前已存在，本轮不支持直接去贴头。物理确有有效足部证据，收益依旧很小。区域接触标签偏正且不等于静止支撑相位，不应把预测接触接近全正直接判成实现错误。400组保存阶段数组独立CPU复算通过，0参数更新，主训练未变。[诊断报告](../runs/native30_live_monitor_oct16/foot_stage_diagnostics/REPORT_zh.md)、[训练趋势与脚步/EXO建议](../runs/native30_live_monitor_oct16/foot_stage_diagnostics/TRAINING_AND_NEXT_STAGE_zh.md)。

当时建议同父状态、每臂1000更新的control与仅关闭独立物理项候选，保留学习场景与既有脚步监督；两臂统一推理配置评估，再交叉报物理开启的系统效果。该训练分支随后完成，结果见本文12:53更新。EXO接口原生共享扩散，不必等脚步完美才准备，但需独立登记阶段并保留ego回放/对照；首轮可7:3、500–1000更新。当前55k不会自动切EXO，也没有开启旧独立残差网络。

## 10月16日23时复核：全RICH与最新训练

**全RICH已完成且复算通过**：冻结15100步，当前VAL索引206个合法窗口、27条录制，clean/noisy共412次预测。全窗计数加权World MPJPE185.582/196.940mm、PA-MPJPE139.941/140.660mm；GT低速期预测脚速0.311/0.379m/s，同帧GT约0.074m/s。不是当前21k权重的全量验证，也不是官方TEST协议。[全量报告](../runs/native30_live_monitor_oct16/rich_full_validation_oct16/REPORT_zh.md)、[412份CPU验收](../runs/native30_live_monitor_oct16/rich_full_validation_oct16/VERIFY_COMPLETE.json)。

27录制各取中位窗口的场景/物理配对也已完整结束，108次新增预测、216行指标复算通过。关物理仍保留学习场景的B，相对场景和物理均关的C，clean/noisy World误差降低28.862/46.957mm，21/27、19/27录制改善；低速脚速0.525/0.618→0.293/0.323m/s。该通路包含学习式contact/correspondence门控feedback，不能独归功于某个编码器。再开启物理修正的A仅使World变化+0.401/−1.378mm，脚速反升至0.322/0.384m/s。支持场景通路确有作用，不支持当前物理修正已解决滑步。[完整配对报告](../runs/native30_live_monitor_oct16/rich_full_validation_oct16/scene_causal_check/REPORT_zh.md)。

当前物理项是基于67个稀疏skin锚点的局部观测接触修正。推理中脚部平面项还依赖预测接触>0.5、可靠局部向上表面和浅层6cm带；每次x0最多2轮并保护头观测、限制root总修正2cm。它没有全身mesh三角形碰撞检测或人体自碰撞项，不能保证腿不穿墙或手不穿腿；哪一门控导致某个具体失败仍须逐例诊断。参数化旋转/FK只保证骨架与解码一致，不保证体积互斥。查看器用22身体关节解码mesh，手指/面部为中性姿态，并非生成的精细手部动作。

截至22:59，主worker106162持续训练到21k以上，已保存21k，IO重启0。完整千步段总loss：15–16k 0.28060、16–17k 0.27585、17–18k 0.26237、18–19k 0.25559、19–20k 0.24354、20–21k 0.23992。15–16k→20–21k身体/脚步/观测/场景加权贡献分别0.11446→0.10056、0.13268→0.10871、0.004071→0.003441、0.029390→0.027206；最近观测和场景项稍回升，不能称所有项单调下降。

固定RICH同一窗clean/noisy均值：19k130.557mm（119.149/141.965）、20k167.910、21k134.565（122.002/147.127）。best.pt当前保留19k，只是单固定窗临时最优。21k三数据集固定窗平均182.420/196.207mm，比20k170.643/174.841mm差，故训练拟合继续改善，验证仍有波动；尚无21k全RICH结果。依据为training/trace.jsonl与validation.jsonl，过滤scene_off，完整段比较，不用未完成尾段或单步loss推断趋势。

## 已有证据

| 证据 | 结果 | 能说明什么 / 限制 |
|---|---|---|
| 新版总训练 loss，完整千步段 | 12k 0.3387 → 15k 0.2959；身体/步态、观测、场景加权小计总体下降 | 确有拟合进展。固定 RICH 单窗 MPJPE 12k 133/152 → 15k 181/192 mm，泛化没有同步单调改善。 |
| 新版 16k，刚完成的固定 RICH 单窗 | clean/noisy 142.80/161.48 mm，较15k回落；同窗12k仍为133.49/151.74 mm | 近期反弹有所恢复，但单窗不能确认全量进步。当前已保存15100步该窗均值150.290 mm，16k均值152.141 mm，自动best正确保留15100步。 |
| 新版 15k，三个固定 VAL 窗同权重同 seed，场景 ON/OFF | MPJPE clean 186.73/210.49，noisy 189.82/221.14 mm；GT 低速期足速 clean 0.248/0.433，noisy 0.279/0.573 m/s | 场景总通路有正收益。只有三个窗口，且 OFF 同时去掉学习条件和物理修正，尚不能单独归因于学习式场景条件。 |
| 新版 15k，单个 RICH 固定窗同上 | MPJPE clean 181.07/205.39，noisy 191.50/227.19 mm | 该窗也有收益；不能代替全 RICH。 |
| 新版 15k 对固定观测骨架参考 | TRUMANS 129/141 对参考 498/495；EgoBody 250/236 对参考 291/297；RICH 181/192 对参考 464/463 mm | 学到的全身生成不是简单重复输入参考。但这个参考很弱，胜过它不等于胜过有竞争力方法。 |
| 新版 12.6k，30 个不同录制，三数据集各 10 窗 | RICH 236/246 mm；复杂姿态、身体局部腿姿态与滑步仍较差 | 明确短板。该集合单个 checkpoint 的横截面不能当训练趋势。 |
| **实际旧 55k**，12,913 窗 / 139 序列，场景配对消融 | World MPJPE ON 187.28 / OFF 208.81 mm；RICH 8,284 窗 / 28 序列 ON 215.09 / OFF 234.45 mm | 旧模型确实会用场景影响动作。20 Hz、窗口网格和 split 与新版不同，不能直接排名；旧模型含历史 EgoBody TEST 重叠。 |

旧 55k 的全量 GT 支撑期滑步 ON 0.6325 / OFF 0.5966 cm/帧，RICH 0.7162/0.6575，反而略坏；不能概括为“所有物理指标都改善”。其支撑高度也是代理，不是真实全身场景碰撞。请不要误引 `control55k_scene_ablation_oct03` 的 3,310 窗大收益：那是更早的另一个 55k 模型。

EXO 小残差加旧扩散先验与 scene 的开发组合曾有额外几毫米收益，但大部分提升来自独立残差网络，guard 是事后规则，旧 noisy 实为噪声混合。这不能作为当前“场景条件扩散”主张的主要证据。

## 不重训的下一项检验

### 16k 可视化后的脚部支撑复核（10月16日晚）

不能再用“已经迈步”代替可靠落脚。对冻结的两个16k窗口做了CPU复核，未修改模型、训练配置、封存源码或运行队列：

- TRUMANS第119帧，预测左右脚部最低网格点比GT高221/271 mm，比附近实际可见水平面高155.5/192.6 mm。附近实际输入有可靠向上表面，不能笼统归因看不到地面。GT本身相对扫描也有65–78 mm偏差，故保留相对GT与相对表面两套读数。
- 以GT脚关节3D速度<0.2 m/s作为低速代理，TRUMANS/RICH预测水平脚速为0.289/0.346 m/s，GT为0.019/0.066 m/s；只代表两个选定窗口，不是全验证统计。RICH第90帧还存在脚部俯仰与穿入，脚关节抬高不等于整个足底离地。
- 当前头部软平移锚定会移动整个身体，之后6次头部IK只调节脊柱祖先链，不会补回脚支撑。用GT注入单一spine旋转误差的受控CPU复现中：头均误差228.56 mm，经平移锚定+IK变为0.0318 mm，但双脚被抬高168.98 mm；仅IK时头误差0.4956 mm、脚位置保留。这个复现证明机制存在，不证明真实采样的全部浮空由锚定造成。
- 物理修正有contact/normal/距离等门控且每个x0根平移限2 cm；现有导出不足以确认本例是哪一门控或接受条件限制修正。不能直接断言25 cm查询门限已被触发。

最小候选应先在同一冻结权重、输入和seed上分离“头部整身平移”和“关节IK”的作用，再决定是否改训练；不把去掉整身平移已在真实采样中有效当作既成结论。持续训练及下面已排队的场景检验维持原配置。

原始读数：[脚部测量](../runs/native30_live_monitor_oct16/visuals/viser/FOOT_DIAGNOSTICS.json)、[受控锚定复现](../runs/native30_live_monitor_oct16/visuals/viser/ANCHOR_MECHANISM.json)、[实际场景支持与门控](../runs/native30_live_monitor_oct16/visuals/viser/SUPPORT_DIAGNOSTICS.json)。

**21:00后完成的真实配对实验更新：** 冻结15100步，5窗×clean/noisy×4组共40次DDIM20；独立CPU重算40/40通过，RICH baseline与原输出逐字节一致。去头部整身平移反而变差（10输入平均MPJPE160.70→174.03mm，低速期水平脚速0.314→0.391m/s）。弱root直立先验让MPJPE微降到159.07mm，但脚速升至0.332；另4次CPU可见支撑IK全部加剧垂直脚速。三种当前候选均不启用，主训练原样继续。不能再用受控机制测试替代真实采样结论，也不能把弱直立先验称为重力动力学。完整读数、反例覆盖限制及[决策记录](../runs/native30_live_monitor_oct16/support_ablation/DECISION_zh.md)、[同权重Viser](http://localhost:8801/)已经保存。

### 已完成的场景通路检验

用本次全 RICH 的同一份 **15,100 步冻结权重**，每条合格录制取一个中位合法窗口，27 窗，clean/noisy 分开，同输入、同 seed：

1. A：完整模型，直接复用已独立核验的全 RICH 输出。
2. B：只关闭 `guidance_enabled`，保留 DDIM 内的学习式全局/局部/关系场景条件和 learned feedback；观测 soft anchor、IK、参考和全部权重保持。
3. C：在 B 基础上 `use_scene=False`，关闭全部场景条件；观测、IK、参考、seed 保持。

**B 相对 C 的配对收益**回答“没有独立物理求解器，扩散中的场景条件是否仍然有用”；A 相对 B 回答物理修正的额外效果。再列静态观测参考的无训练结果，避免只相对某个坏初始化宣布胜利。各窗、各录制都保留，汇报全身/PA/方向/局部足误差与足速，不能只挑好看的例子。这个检验说明当前训练权重如何利用输入，不等于证明扩散优于 VAE，也不替代同预算、分别训练的架构消融。

该任务在全 RICH 结束并 `VERIFY_COMPLETE.json` 通过后启动 CUDA，已于21:15完整结束并通过CPU复算。结果放原全 RICH 目录的 `scene_causal_check/`，代码统一放 `ops/scene_causal_check.py`。最终数值见本文顶部更新与对应报告；这些配对窗口不替代206窗全量结果。

## 和 EgoGVAE、EgoEgo、EgoAllo 怎么比

**不必立刻把三种方法全部重训。** 先把问题分为训练数据相同的模型比较，和利用额外场景证据的系统比较；都能做，但不能混写成 AMASS-only 的胜负。

| 路线 | 做法 | 可支持的结论 / 成本 |
|---|---|---|
| 最快的实用系统对照 | 官方 AMASS 预训练权重，在统一 RICH TEST 清单、统一合成 head/CPF、同 evaluator 上重跑；明确我们的混合训练、观测场景、已知 body profile 等额外条件。补列 head-only scene-OFF 行。 | 回答现成系统在目标任务上的效果，训练数据不等，不能据此归因架构优越。需匹配 body shape、floor height、后处理、采样次数。 |
| 混合训练的受控对照 | 选一个有可复现训练入口的强 baseline，在与我们完全相同 TRAIN 序列、head 噪声、预算上重训/微调，然后同 RICH TEST 评估。AMASS 预训练如使用则两边一致，或明确单列。 | 最直接回答目标数据设置下的模型差异；不一定要先重训全部三种。baseline 保持原 head-only 输入时，场景是我们的额外模态，需单列。 |
| 严格沿用论文 AMASS-only 协议 | 我们另训练一个从未见 RICH 标注的 AMASS-only / 明确相同 pretrain 版本，官方 RICH TEST；地板高度、形状、head 定义与窗口一致。 | 才能直接对论文 AMASS→RICH 泛化表。AMASS 没有观测场景，场景主张需另外的场景数据或清楚标注的额外训练数据；不能靠把现有混训权重 scene-OFF 洗成 AMASS-only。 |

EgoGVAE Table 2/§4.2 的统一横比是 AMASS 训练、RICH TEST、128 帧 synthetic head/CPF；当前是混合训练后的 RICH VAL、192 帧。原 EgoEgo 指标实现含 pelvis-relative MPJPE，不能直接与 World MPJPE 混列。当前 RICH TRAIN 和 VAL 的序列 ID 无交集，但部分人物和场景重叠，不能描述为未见人物/场景泛化。选择 best 只用 VAL；最终 TEST 不参与挑权重。

## 可核验来源

- 新版：`experiments/offline_camera_retrain_v1/runs/native30_gait_v2_ego55k_oct15/training/validation.jsonl` 的 15k fixed，及 `native30_live_monitor_oct16/rich_full_validation_oct16/LOSS_TREND_zh.md`。
- 新版扩展验证：`native30_live_monitor_oct16/balanced_validation_oct16/RESULT_REVIEW_zh.md`，60 份预测 CPU 复核通过。
- 实际旧 55k：`native_dynamic_scene20_contact_55k_oct07/full_validation/{manifest,standard_metrics}.json`。
- EXO 与用户意图：`exo_occlusion_review_oct12/EXPERIMENT_AND_ARCHITECTURE_zh.md` §4.17–4.19、末尾 10 月 16 日续训说明。
- 跨方法协议逐文件核对：`native30_live_monitor_oct16/rich_full_validation_oct16/BASELINE_COMPARISON_PROTOCOL.json`、`PROTOCOL_REVIEW_zh.md`；[EgoGVAE §4.2 / Table 2](https://arxiv.org/html/2607.27755v1)、[EgoAllo 官方评估](https://github.com/brentyi/egoallo/blob/main/5_eval_body_metrics.py)、[EgoEgo 官方评估](https://github.com/lijiaman/egoego_release/blob/main/eval_stage2.py)。
