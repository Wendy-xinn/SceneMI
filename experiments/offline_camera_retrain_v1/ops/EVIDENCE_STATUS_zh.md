# 当前方法的证据与最短检验路径

2026-10-16，仅读既有模型、日志与评估产物；不修改正在训练的源代码或配置。

**判断：网络在学习，场景通路有可测的推理收益；新版在 RICH 上已达到可用质量、比旧版更好、或者优于公开方法，这三件事目前都没有证实。** Loss 下降是拟合证据，不能替代固定验证集的进步。用户要求的是场景条件扩散主线，当前保留的是从零训练的 head-only Native30 reference-residual 扩散，脚步监督已恢复，目标 55k；独立小 EXO 残差支线没有继续训练，tracker 也未接入。

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

### 已登记的场景通路检验

用本次全 RICH 的同一份 **15,100 步冻结权重**，每条合格录制取一个中位合法窗口，27 窗，clean/noisy 分开，同输入、同 seed：

1. A：完整模型，直接复用已独立核验的全 RICH 输出。
2. B：只关闭 `guidance_enabled`，保留 DDIM 内的学习式全局/局部/关系场景条件和 learned feedback；观测 soft anchor、IK、参考和全部权重保持。
3. C：在 B 基础上 `use_scene=False`，关闭全部场景条件；观测、IK、参考、seed 保持。

**B 相对 C 的配对收益**回答“没有独立物理求解器，扩散中的场景条件是否仍然有用”；A 相对 B 回答物理修正的额外效果。再列静态观测参考的无训练结果，避免只相对某个坏初始化宣布胜利。各窗、各录制都保留，汇报全身/PA/方向/局部足误差与足速，不能只挑好看的例子。这个检验说明当前训练权重如何利用输入，不等于证明扩散优于 VAE，也不替代同预算、分别训练的架构消融。

该任务只在全 RICH 结束并 `VERIFY_COMPLETE.json` 通过后启动 CUDA；不和全 RICH 争抢第二个评估槽。108 次新增 DDIM 大约 35–50 分钟，受主线共享 GPU 影响。结果放原全 RICH 目录的 `scene_causal_check/`，代码统一放 `ops/scene_causal_check.py`。

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
