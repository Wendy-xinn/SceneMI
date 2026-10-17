# Native30 EGO/EXO 初版适配

2026-10-17已完成CPU准备、完整TRAIN/VAL源校验和实际父模型GPU预检（EXO/mixed反传、隐藏GT隔离、无标签EXO接口等价与12次固定VAL基线）。1000更新的新阶段已启动，实际更新记录在training/trace.jsonl；实时状态以STATUS.json及prepared/input_checks.json为准。父模型为已选定的41k EGO扩散权重，保留55k完整状态。

本阶段沿用同一个Native30共享扩散生成器，预测相对观测参考动作的扩散残差，通过20步DDIM生成动作；没有接回旧版独立EXO残差小网。EGO仅头部；EXO输入为22关节中自然可见位置、mask、置信度与观测场景，不补GT旋转，也不宣称原始RGB端到端。

以EXO80%、EGO replay20%的严格10样本配额适配1000更新，B2、L192、30Hz。该比例是时间预算选择，尚未证明优于其他比例。重置Adam、sampler和RNG，独立50步warmup、1000步cosine，LR最大2e-5、最低4e-6；父41k和本阶段步数分开记录。模型、损失、观测噪声、物理/IK设置均保持父协议。

准备阶段重新验证完整官方TRAIN/VAL输入、相机标定与依赖闭包，明确复用父模型已冻结的TRAIN observation loss系数，不冒称重新标定。CPU strict-load仅验证兼容性，不能据此标记training_ready。GPU预检须实际加载父模型，分别做EXO-only与mixed全目标反传，证明共享扩散主干收到EXO梯度，并验证隐藏GT不影响输出、无GT frozen EXO接口等价。预检不做optimizer更新。

基线和最终使用同一manifest的6个固定VAL窗口（三语料×两模式），clean/强制noisy分别报告，共12次预测/模型。首步不重复DDIM；1000更新最终验证与保存后停止。EXO先看未输入部位误差，相对于父模型和reference初始化器均报告；可见部位、root和脚部一起作保护指标，同时检查EGO遗忘。固定小面板用于首版开发判断，不替代正式TEST和全录制评估。

所有GPU阶段与TEST采样共用`/tmp/scenemi-native30-gpu.lock`，CPU准备/数据导出可并行。只保留一份原子可恢复last.pt，不自动生成best/编号副本、不自动延长、不自动推广。

从SceneMI根目录使用可复用入口；本轮已按显式依赖闭包封印，开始更新后不能原地重新准备：

```bash
python -m experiments.offline_camera_retrain_v1.ops.adapt prepare
python -m experiments.offline_camera_retrain_v1.ops.adapt preflight
python -m experiments.offline_camera_retrain_v1.ops.adapt train
python -m experiments.offline_camera_retrain_v1.ops.adapt status
```

`run`依次执行准备、预检、1000更新和最终配对验证；`train --resume`严格恢复本新阶段，仍以1000为预算。失败会落盘STATUS并返回非零。最终COMPARISON.json保存逐窗指标；CPU_RECHECK.json重新打开数组，在CPU独立用FK+NumPy公式复算关键指标并核对输入/GT逐元素一致。

首次准备后的WSL关闭和恢复记录保存在attempts/。恢复时发现一份E盘派生数组不符合原SHA，已隔离坏文件、从Linux侧原SHA正确的副本恢复，并重新核验全部1009份metadata和20,341个导出数组（28.92GB）；其余无持久不一致。没有改写metadata或放宽校验。
