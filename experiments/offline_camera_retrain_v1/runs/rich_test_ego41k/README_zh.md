# RICH TEST：冻结41k模型

模型在查看本轮TEST结果前已根据VAL选定：Native30 EGO step41000，checkpoint SHA256 `f8e2d22b155240c2fda8cb7d75aca451bb4152ede2441d592f3952f52068b4d0`。TEST不用于选择checkpoint、调参或选择物理修正开关。

使用已下载的官方RICH TEST body，覆盖52条录制、53个人物流。保持原始时间轴；缺失拟合不插值，窗口跨越目标或遮挡者缺失帧时排除。原生协议为30Hz、192帧、stride64，预计335个合法窗口；clean/noisy分别评价，共670次预测。最终数量及逐窗身份以准备完成后的PLAN.json为准。

固定20步DDIM、scene ON、原模型物理修正ON。输入包括头部观测、已知体型和当前协议的可见场景记忆；本轮不等同于其他方法的AMASS-only训练或128帧CPF评估协议，不能直接据此做论文排名。脚接触使用低速代理指标，不宣称HSC真实接触指标或完整mesh碰撞评价。

STATUS.json记录当前进度；PLAN.json冻结来源、窗口、随机种子和源码身份；rows.jsonl和SUMMARY.json保存逐窗与汇总指标。采样结束后必须通过独立CPU重算，才能报告完整结果。只完成少量预测时明确标记partial。

从SceneMI根目录使用同一个模块入口：

```bash
python -m experiments.offline_camera_retrain_v1.ops rich-test evaluate --out experiments/offline_camera_retrain_v1/runs/rich_test_ego41k --max-predictions 2
python -m experiments.offline_camera_retrain_v1.ops rich-test recheck --out experiments/offline_camera_retrain_v1/runs/rich_test_ego41k
python -m experiments.offline_camera_retrain_v1.ops rich-test evaluate --out experiments/offline_camera_retrain_v1/runs/rich_test_ego41k
```

GPU采样与EXO适配共用排他锁，按队列运行。已提交的预测和SHA通过检查后会被复用。

2026-10-17首次准备在22:03遇到Open3D段错误与动态加载器断言；随后WSL发生关闭/重新挂载，原因尚未确定，未发现OOM-kill证据。旧日志保留在attempts/。恢复时先复验已完成的5个包，再改用单个隔离导出进程继续；没有降低SHA检查要求。原41k/55k权重未因本轮恢复而改写。
