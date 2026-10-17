# 冻结41k的EGO/EXO共存实验

目标是在一个完整模型中接受EGO与EXO输入，保留41k的EGO计算路径，同时改善EXO遮挡部位且保护可见部位。训练与验收进度以[STATUS.json](STATUS.json)为准；未完成验收前不宣称收益。

原41k约1.23亿参数全部冻结，保持eval；45个内部层增加rank16、alpha16的EXO门控低秩适配器，共1,955,616个可训练参数。适配器进入观测/参考/场景条件投影及U-Net内部卷积和时间条件，每一步扩散去噪仍通过共享主干。最后的201维动作输出层不加适配器。EGO按明确的输入模式绕过新增参数；EXO即使只剩头部可见也不会被误判成EGO。

这是共享主干加少量EXO专用参数，所有参数共同优化尚未验证。CPU单元测试覆盖零初始化等价、混合模式隔离、EXO学习有效而EGO不变、冻结与梯度、状态重载、异常和并发路由恢复；GPU实际父模型预检通过之前不标记可训练。

## 训练与保护

从VAL选定41k权重独立开始1000步EXO-only适配，B2、192帧、30Hz，新Adam及随机状态，LR最高1e-4，50步warmup、1000步cosine、最低0.2倍。只优化适配器；原损失、观察噪声、场景与物理/IK配置保持父协议。每100步覆盖一个完整可恢复last.pt；不产生编号或best副本，不自动延长、不自动推广。

GPU预检实际比较裸41k与零初始化适配器的EGO/EXO/混合前向，检查EXO反传只到适配器；检查隐藏GT污染不影响生成、无动作标签的EXO接口等价。训练后再次检查冻结基础权重摘要和EGO整段DDIM预测数组逐值一致。

前两次启动均在零更新预检中退出，见[ATTEMPTS.json](ATTEMPTS.json)。第二次发现仅改变基础参数的requires_grad标志就会改变CUDA混合批次数值，未安装适配器亦可复现；全局MHA fastpath开关不能消除差异。当前包装仅在串行no_grad推理期间恢复原始标志，退出时强制重新冻结；不修改参数值，梯度训练始终冻结主干。实际预检仍要求逐值一致，不放宽门槛。训练内部的no_grad proposal仍使用冻结标志，这个计算路径差别已在模型清单记录。

## 开发验证与验收

[固定面板](validation_cases.json)保留上轮6个固定窗口，新增三语料各2个移动且遮挡的EXO窗口，共12窗、每阶段24次clean/noisy预测。新增窗口按身份SHA排序，要求遮挡比例10%至85%、root水平位移至少0.4m、双脚相对运动坐标RMS至少0.025m，每个EXO语料内使用不同录制；选择不读取预测误差。9个EXO窗口、3个EGO窗口，同窗口同种子前后比较。这不是完整VAL或TEST。

训练前登记的工程门槛：

- EGO预测数组逐值一致，冻结基础权重摘要一致。
- 遮挡关节误差先在每语料/噪声条件内按关节数量加权，再按三语料等权汇总；clean/noisy宏平均分别改善至少5%。每语料/条件同时不得变差超过max(5mm,5%)。
- 可见关节误差在每语料/条件内不得变差超过max(1mm,2%)；逐窗变化另报，不能把池化通过解释成所有窗口均改善。
- 每语料/条件的GT低速脚部3D速度不得增加超过max(0.01m/s,5%)；双脚相对运动RMS误差和stride误差不得增加超过max(5mm,10%)。这些检查不等同于不存在漂浮或全身mesh碰撞。

最终由独立CPU进程重新打开数组、复算FK与指标，生成COMPARISON.json、CPU_RECHECK.json、ACCEPTANCE.json及尺度诊断。只使用TRAIN/VAL；冻结41k的RICH TEST继续独立运行，不用于本轮调参或验收。

## 尺度与对齐

当前观察、场景和人体骨架共用米制坐标，rest已含体型scale，归一化的positions/2不代表物理尺寸改变。[上一阶段的尺度诊断](PRIOR_STAGE_ALIGNMENT_DIAGNOSTICS.json)未发现明显全局尺度错误；它属于上一阶段证据，不是本轮结果。本轮固定rest与scale，分别报告原始世界误差、可见关节刚体/相似拟合、可见骨段与rest比值。GT拟合仅作诊断，不改输出、不进入通过门槛；每个窗口固定一个变换，不做逐帧GT对齐。

未来真实tracker若提供非米制坐标，应在输入端结合标定和可靠可见骨段估计尺度，再与场景统一。本轮输入仍为带自然遮挡和模拟噪声的关节轨迹，不能提前宣称真实tracker的尺度问题已解决。

## 入口

复用ops模块，运行配置与案例保存在本目录，不新建日期脚本：

```bash
python -m experiments.offline_camera_retrain_v1.ops.adapter_trial run --run experiments/offline_camera_retrain_v1/runs/native30_exo_adapter_v1
python -m experiments.offline_camera_retrain_v1.ops.adapter_trial status --run experiments/offline_camera_retrain_v1/runs/native30_exo_adapter_v1
python -m experiments.offline_camera_retrain_v1.ops.adapter_report --run experiments/offline_camera_retrain_v1/runs/native30_exo_adapter_v1 --watch
```

run先完成CPU源与数据复核，依次执行实际预检、1000步适配、固定验证与独立CPU复算。resume只恢复本阶段完整状态。当前TEST的封印源码保持不变。

报告进程只读取本轮文件，完成后生成RESULTS_zh.md；失败则明确记录错误，不把预检或首步结果当作最终结论。零更新排障若复用已核验CPU准备，会另存SOURCE_RESEAL.json，保留原完整数据复核时间；训练仍在第一次Adam更新前重新执行完整数据摘要核验。

独立推理使用`ops.adapter_model.load_inference_model(checkpoint_path, device="cuda", expected_sha256=...)`，返回模型与身份信息。它从一个完整checkpoint恢复基础模型和适配器，不打开原41k checkpoint或训练数据。调用方仍需通过现有推理接口提供观测、人体rest与场景；模式由输入中的`mode`明确指定。

新阶段登记有限并行策略：适配器进程先独占自己的准入锁，核对共享GPU锁确由正在采样的冻结41k TEST持有、模型身份及进程一致，并检查实时空闲显存至少覆盖本阶段25%显存预算加4GiB余量。条件均满足才允许与TEST并行，否则使用原共享GPU锁排队。实际准入证据写入GPU_ADMISSION.json；没有终止或暂停TEST，也不预先承诺并行加速。
