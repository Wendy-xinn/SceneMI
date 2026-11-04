本轮只计算含噪输入，使用已训练的原10k EXO适配统一模型。所有失败的场景修复策略均未纳入。

范围：RICH、TRUMANS、EgoBody全部合法VAL窗口，共3541个mode-window；EGO1806、EXO1735；每窗同观测、同reference、同seed计算scene ON/OFF，共7082次预测。DDIM20、FP32、192帧、30Hz、stride64，解析物理修正两边都关闭。无TEST，不再计算clean。

入口与产物：
- STATUS.json：当前采样进程和精确进度。
- SUMMARY.json / REPORT_zh.md：每数据集、EGO/EXO、有无场景的指标和覆盖数量。
- OVERALL_SUMMARY.json / OVERALL_REPORT_zh.md：窗口加权总体、等数据集macro、严格成对ON−OFF，以及标注异质输入的EGO+EXO合并总体。未完成时明确partial。
- rows.jsonl / motions：每例指标、输入与预测SHA及可复核数组。
- BATCH_RUNNER.json / batch4_benchmark/RESULT.json：批量加速源码、逐窗seed、数值对照及耗时证据。12份VAL数值对照通过后才接续batch4，模型参数及窗口没有改变。
- JOB.json / ACCELERATION.json：实际命令、后台进程与自动退回逐窗的记录。
- OVERALL_WATCHER.json：CPU汇总后台。全部采样完成且采样进程退出后，自动运行已有CPU独立复核，再刷新总体报告。

断点恢复命令（勿与仍存活的JOB重复启动；健康运行库SHA为43731e24cd89e3749826304f304e8aa11fbecf1188715271b1f5018d6212b5e6）：

```bash
env TORCH_USE_RTLD_GLOBAL=1 LD_BIND_NOW=1 LD_PRELOAD=/tmp/scenemi_nvrtc_verified_runtime/libnvrtc.so.12 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR=/tmp/scenemi-mpl-final-metrics /home/wenxin/miniconda3/envs/scenemi/bin/python -u -m experiments.offline_camera_retrain_v1.ops.final_evaluation_batch evaluate --out /home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_final_metrics_v1/full_validation_noisy --batch-size 4 --device cuda --resume --memory-fraction .7 --status-every 8
```

原10k权重SHA为3247383297ca2d7129755c6cd96c82eb2009cb2758aa8e22b6e5d0acb04c66b5。原full_validation中28份已算含噪结果经文件SHA核验并硬链接复用；新的遍历只按dataset/mode轮换，case身份与seed不变。模型与输入协议均冻结，不根据本轮指标挑模型。全部文件仅保留本地。
