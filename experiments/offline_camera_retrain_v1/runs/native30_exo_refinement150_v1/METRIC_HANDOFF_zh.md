# 正式指标入口交接

2026-10-20：复用 `ops/final_evaluation.py`，无需再增加单次评估脚本。7项CPU测试通过；两份正式输入ENTRY与CPU模型冻结已完成。用户随后明确优先修复EXO、暂缓指标，因此当前暂停。**正式TEST预测0条、数组0份，full VAL也未采样。**

暂停记录：`runs/native30_final_metrics_v1/METRIC_PAUSE.json`。曾启动的GPU命令在默认sandbox的CUDA初始化处报“no NVIDIA driver”并退出，未进入TEST模型推理；host按完整命令检查无该评估进程存活。这不代表host GPU不可用。后续GPU评估应使用具备GPU可见性的host运行环境。

## 输入和数量

| 输入 | EGO窗口 | EXO窗口 | clean/noisy × scene ON/OFF预测数 |
|---|---:|---:|---:|
| RICH独立TEST | 335 | 不提供 | 1,340 |
| RICH完整VAL | 206 | 206 | 1,648 |
| TRUMANS完整VAL | 962 | 709 | 6,684 |
| EgoBody完整VAL | 638 | 820 | 5,832 |
| 三库完整VAL合计 | 1,806 | 1,735 | 14,164 |

RICH TEST共有52录制、53人体流，metadata与335窗源路径全部存在。使用已经准备好的 `runs/rich_test_ego41k/PLAN.json`、`test_ego_windows.npy` 和 `/mnt/d/Projects/SceneMI/cache/rich_test_native30_v1/manifest.jsonl` 的输入身份，**不复用旧模型或旧预测**。其53个RICH源均仅允许EGO，不得人为开放EXO。

VAL沿用 `data/native30_official_windows64_v1/dataset.json` 的持久合法索引。EGO和EXO窗口数不同，由各自可用观测与长空窗限制决定；不可声称两模式覆盖完全相同的窗口。

旧 `rich-test` 入口硬编码bare EGO模型、physics ON和scene ON；旧 `validation_review` 硬编码41k/55k、EGO306窗。它们不能直接接本次LoRA模型。新入口复用 `exo_longrun.load_inference_model` / `refinement_trial.load_model`、`Native30Data`、观测构造器、版本化EXO参考、scene-off守卫、既有native指标和独立NumPy FK，不修改旧封存源码。

## 冻结身份与恢复命令

`runs/native30_final_metrics_v1/SELECTION.json` 已于21:26:40在VAL上固定10k原始输出，checkpoint SHA256为 `3247383297ca2d7129755c6cd96c82eb2009cb2758aa8e22b6e5d0acb04c66b5`。manifest本身SHA256为 `b8b1d91bf2716df2d66c32b31d69e1750128b3e7cfce322a70d64d3204b23a11`。physics OFF，参考策略 `exo-local-so3-reference-gap-fill-v1`，DDIM20，不加入输出角度投影。

两份 `ENTRY.json` / `PLAN.json` 已准备，位置为 `runs/native30_final_metrics_v1/{rich_test,full_validation}`。后续EXO修复若决定采用另一模型，必须先只依据VAL重新决策并使用新的输出目录；不得修改这些已冻结计划、不得用TEST判断换模型。

以下命令**仅在用户恢复指标任务后执行**，工作目录 `/home/wenxin/projects/SceneMI`。GPU运行需host GPU可见；环境变量仅作用于此进程，不改全局库。

```bash
env TORCH_USE_RTLD_GLOBAL=1 LD_BIND_NOW=1 LD_PRELOAD=/tmp/scenemi_nvrtc_verified_runtime/libnvrtc.so.12 CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/scenemi-mpl-final /home/wenxin/miniconda3/envs/scenemi/bin/python -u -m experiments.offline_camera_retrain_v1.ops.final_evaluation evaluate --out experiments/offline_camera_retrain_v1/runs/native30_final_metrics_v1/rich_test --resume --status-every 10 --device cuda

env TORCH_USE_RTLD_GLOBAL=1 LD_BIND_NOW=1 CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/scenemi-mpl-final /home/wenxin/miniconda3/envs/scenemi/bin/python -m experiments.offline_camera_retrain_v1.ops.final_evaluation recheck --out experiments/offline_camera_retrain_v1/runs/native30_final_metrics_v1/rich_test
```

CPU保底为第一条命令改 `--device cpu` 并加 `CUDA_VISIBLE_DEVICES=''`；**同一out不能并发两个evaluate**。每条预测先原子提交数组，再fsync提交JSONL；`--resume`保留已提交记录，仅修复最后未完成的JSONL尾部。正式完整VAL用相同入口，将out换成 `full_validation`，建议在RICH TEST之后排队运行，避免争抢GPU。

检查当前状态：`python -m experiments.offline_camera_retrain_v1.ops.final_evaluation status --out ...`。有限首批可加 `--max-predictions 2`，完成后单独CPU recheck，再用 `--resume`继续。本轮尚无真实预测，因此不能声称正式推理链或CPU复算已实测通过；已经通过的是CPU严格加载和7项入口、源身份、断点、观测隔离测试。

## 时间和汇报边界

既有VAL日志中EGO采样均值约6.74秒、EXO约11.68秒，另有输入准备和写盘成本。按8–15秒/预测保守估计：RICH TEST两开关约3–5.6小时；三库完整VAL全部模式与开关约31.5–59小时。实际吞吐要以首批预测为准，CPU会更慢。这些是历史VAL速度推算，没有读取TEST结果来估时。

当前协议是30Hz、192帧（6.4秒）、stride64重叠窗口、22个通用SMPL/SMPL-X关节，已知人体profile；EGO主要为合成/伪头相机观测，EXO为根据可见性与噪声合成的3D观测，不能当作真实图像HMR输入的完整系统评测。场景是记录级可见静态并集，包含离线未来观测；场景OFF仅关闭模型场景路径，观测与相机、参考、已知body不变。

报告World MPJPE、逐帧相似变换PA-MPJPE、root-relative误差、可见/不可见误差、速度、加速度和足部低速代理，并同时给等窗口、等录制宏平均与可用分母加权。不可见关节不等同纯物理遮挡；GT低速脚不等同官方HSC接触；当前没有完整人体自穿模/场景mesh碰撞率指标。该协议尚未与EgoGVAE、EgoEgo、EgoAllo的AMASS-only训练、原生窗口/CPF、关节集合、输入以及对齐方式匹配，不能直接与论文表格横比。

2026-10-20 23:18补充：本次150+150短训和162次VAL预测已在局部健康NVRTC预加载下完成，均未推广为最终模型。当前scenemi环境原NVRTC文件存在128字节损坏；上面GPU命令补入已验证的局部副本。使用前按`RUNTIME_RECOVERY.json`核对SHA256；若/tmp已清空，需从记录中的同版本健康来源重建并核验，不能直接退回损坏库。`TORCH_USE_RTLD_GLOBAL=1`与局部`LD_PRELOAD`需同时存在。本轮VAL结果不等同正式TEST链已验证；正式指标继续暂停。
