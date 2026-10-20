# EXO 10k 与过度修正处理结论

2026-10-20。10k训练、全部里程碑固定/轮换验证、独立CPU权重检查和保存数组时序审计均已完成。最后CPU启动曾因动态库加载错误退出127；仅进程环境调整后补跑成功，权重、best指针、配置与训练协议哈希保持不变，原失败保留在POSTRUN_RECOVERY.json。

固定100窗clean/noisy加权遮挡误差：8k 113.098mm，10k 112.870mm，改善0.20%。开发best为10k；这不是完整VAL/TEST或方法有效性验收。RICH noisy仍从输入参考60.058mm恶化到114.181mm，9/12窗口变差。

## 本轮已执行的修正对照

通用实现为ops/refinement_guard.py；只读取已有输出，未训练、未更改封存采样实现或默认模型。范围是相对参考的局部SO(3)改动，不是关节绝对弯曲角；投影函数不读取GT。分别比较30/60/90度。8k固定与轮换共200窗；10k最终固定与轮换也各100窗。仅EXO输出参加修正。

直接限制全部非root旋转会牵动可见关节。增加观测依赖规则后，只处理不会影响有效观测位置或全局旋转的关节；其几何性质已通过10项CPU测试，包括root与输入保持、FK骨长、非交换旋转与观测保持。

| 10k固定指标 | 原输出 | 仅无观测约束旋转，30度 |
|---|---:|---:|
| clean/noisy按VAL比例加权遮挡误差 | 112.870mm | 107.573mm |
| RICH noisy遮挡误差 | 114.181mm | 86.144mm |
| RICH noisy可见误差 | 30.198mm | 30.198mm |
| RICH noisy GT低速期预测足速 | 0.180m/s | 0.262m/s |
| RICH noisy加速度误差 | 2926.60mm/s² | 3881.42mm/s² |

因此该后处理有几何误差收益，但时序代价不合格，**不替换默认输出、不认定为最终修正模型**。逐帧观测依赖切换可能造成跳动；尚未隔离它与投影本身的各自贡献。更宽范围也不能充分救回严重错例。真实需要大幅修正的子集、P90、逐窗身份与输入哈希均保留在各guard_review目录的REVIEW.json。

## 下一步方案，尚未启动新训练

1. 在训练中加入相对reference的非退化监督：预测误差超过reference误差加容差时增加惩罚；FK位置和SO(3)旋转分别计算。GT只用于TRAIN loss，允许错误参考得到必要的大幅修正。
2. 加入少量高质量reference保持样本，检查模型能否在输入已好时保持；同时保留真实遮挡/带噪样本。单独identity目标不能保证隐藏关节恢复。
3. 若继续采用推理约束，先处理时间连续性，不能直接使用已出现抖动的逐帧硬开关。保留可见保持、足速、加速度与大动作恢复检查。
4. 使用同一预先确定窗口和种子，分开比较scene ON/OFF；先做受控短训，再决定长训。当前没有启动新的训练或自动提升修正候选。

GRAIL继续作为地形候选进行预处理，本轮不混入EXO训练。接入前需核对人体比例协议并筛查全序列足底支撑/穿模，随后单独验证地形数据增量收益，避免同时改变数据与EXO修正机制。

## 可复现入口

```bash
python -m experiments.offline_camera_retrain_v1.ops.refinement_guard --run experiments/offline_camera_retrain_v1/runs/native30_exo_natural10k_v1 --output /tmp/exo_guard_review --step 10000 --panels fixed rotating --caps 30 60 90 --scope unobserved
python -m unittest experiments.offline_camera_retrain_v1.ops.test_refinement_guard -v
```

旧all-scope诊断对应源码提交72bb149；新增观测依赖诊断的源码哈希保存在各REVIEW.json，全部记录仅用于开发验证，不改变best。
