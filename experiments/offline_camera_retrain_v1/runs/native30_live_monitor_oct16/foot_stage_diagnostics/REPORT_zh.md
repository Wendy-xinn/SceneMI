# 冻结 19000 步权重：脚部误差逐阶段诊断

**已完成，包含无监控同 seed 控制。**

两段预选失败/走动窗口，clean/noisy 共4次真实DDIM20。这里比较同一次采样的局部阶段，不等价于删除对应模块的完整消融。

raw_fk 是原始输出按FK一致化后的结果；随后是学习反馈、整身贴头平移、头IK、原物理修正。

脚速是 GT 三维足速<0.2m/s 时预测的水平足速，不与训练验证中的三维足速混用。

## 最后一步 t=0

| 窗口/输入 | 阶段 | MPJPE mm | 头对输入 mm | 水平足速 m/s | 脚高绝对误差 mm |
|---|---|---:|---:|---:|---:|
| trumans / clean | raw_fk | 153.825 | 81.330 | 0.3594 | 44.893 |
| trumans / clean | learned_feedback | 153.859 | 83.092 | 0.3547 | 41.601 |
| trumans / clean | head_translation | 124.685 | 8.317 | 0.2531 | 41.928 |
| trumans / clean | head_ik | 124.457 | 0.096 | 0.2531 | 41.928 |
| trumans / clean | physical | 124.457 | 0.096 | 0.2531 | 41.928 |
| trumans / noisy | raw_fk | 165.398 | 95.485 | 0.4237 | 67.368 |
| trumans / noisy | learned_feedback | 165.005 | 97.093 | 0.4179 | 63.908 |
| trumans / noisy | head_translation | 152.615 | 61.098 | 0.3679 | 63.639 |
| trumans / noisy | head_ik | 146.469 | 5.642 | 0.3679 | 63.639 |
| trumans / noisy | physical | 146.123 | 0.984 | 0.3779 | 62.855 |
| rich / clean | raw_fk | 152.169 | 86.022 | 0.4679 | 38.458 |
| rich / clean | learned_feedback | 156.669 | 91.042 | 0.4541 | 47.973 |
| rich / clean | head_translation | 119.440 | 9.112 | 0.3291 | 25.036 |
| rich / clean | head_ik | 119.039 | 0.057 | 0.3291 | 25.036 |
| rich / clean | physical | 119.149 | 0.002 | 0.3249 | 24.499 |
| rich / noisy | raw_fk | 161.253 | 91.070 | 0.4402 | 28.861 |
| rich / noisy | learned_feedback | 164.009 | 93.491 | 0.4169 | 24.972 |
| rich / noisy | head_translation | 147.803 | 60.113 | 0.3624 | 27.670 |
| rich / noisy | head_ik | 142.726 | 11.032 | 0.3624 | 27.670 |
| rich / noisy | physical | 141.965 | 4.172 | 0.3576 | 27.991 |

## alpha>0.5 的逐步配对变化

以下是局部阶段均值变化（后减前）；多个DDIM步彼此相关，不是独立样本。

| 转换 | MPJPE mm | 头对输入 mm | 水平足速 m/s | 脚高绝对误差 mm |
|---|---:|---:|---:|---:|
| raw_fk → learned_feedback | +0.658 | +1.230 | -0.0131 | -0.815 |
| learned_feedback → head_translation | -17.415 | -46.468 | -0.0682 | -8.022 |
| head_translation → head_ik | -3.524 | -37.036 | -0.0000 | -0.000 |
| head_ik → physical | -0.542 | -5.044 | +0.0007 | -0.380 |

物理门控记录直接来自该次原始 prepare_evidence，不新增场景查询；没有记录到的距离/法向门控不能推测为失败原因。
GT仅在采样完成后的CPU评价使用；记录不代表新训练收益，也没有更改主训练。

## 判断与标签核查

这4个输入的原始FK输出已经有较高足速（0.359/0.424/0.468/0.440 m/s）。学习反馈小幅降低足速，随后贴头平移进一步降低；这些局部结果不支持“移除贴头就能解决滑步”。头IK不修改腿链，因此这一步脚速不变。该判断也与此前完整DDIM的ik_only消融变差一致。

物理修正实际运行：末步TRUMANS clean/noisy、RICH clean/noisy的正权重足锚点数为765/757/738/740，接受修正轮数为0/1/1/1。不能说物理门控全关闭。TRUMANS noisy这一步足速增加约0.010m/s，RICH两例下降约0.004–0.005m/s；效果有限且并不一致。未记录逐个候选被拒的原因，不能推测为某个距离或头保护门控造成。

另用CPU重建同窗，按已有training_support_targets离线核对标签（CONTACT_LABEL_AUDIT.json）。TRUMANS没有官方标签，4个脚/踝关节都是unknown；弱标签分别37/8/161/170帧positive，negative均0，其余unknown。RICH官方4关节分别179/184/183/187帧positive、13/8/9/5帧negative，均192帧valid。预测contact近全正不能直接证明分类实现错误，也不适合直接解释成左右脚交替、零速度的支撑相位。这里没有把unknown当negative。

## 核查和范围

4次受监控采样的20步×5阶段均保存数组，400组指标由保存数组CPU计算。末步物理输出经原sampler的reference残差算术和FK恢复后，与采样结果字节一致；直接物理阶段与采样终值的FK最大差为2.4e-7m。RICH clean另做一次完整无监控同seed采样，源代码直接np.array_equal检查通过；控制数组未单独保存，独立方可审查这条检查源码和完成记录，但无法重新对控制原始数组做哈希。

模型tensor前后digest一致，365份封存训练/执行源码未变，训练配置和优化器没有更新。首次工程验证遗漏sampler末端残差减加浮点往返，断言失败后修正并重新封存源码；旧尝试没有输出行混入新实验，失败源码/PLAN/STATUS保留在engineering_check_failed，ATTEMPTS.jsonl记录总共6次采样（1次失败验证、4次正式追踪、1次无监控控制）。

实验目录占用约549MiB，其中474MiB权重是现有best.pt的硬链，不新增权重数据块；新增数组/记录/可重建缓存约75MiB。实验前后文件系统可用空间均约68GiB。
