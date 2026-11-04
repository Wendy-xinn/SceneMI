# AMASS 基线的检查点选择证据

截至本次核对，**没有足够证据证明这三篇论文发布的模型都是按独立验证集最优指标选择**。下面区分训练代码的保存规则、下载权重的实际元数据和论文未明确的部分。不从文件名 `best`、`latest` 或加载器变量名 `val` 推断选择依据。

| 方法 | 公开实现能确认的内容 | 是否能确认“验证集最优” |
|---|---|---|
| EgoAllo | 固定本地版本每 5,000 步保存，阶段性保留；发布目录为 `checkpoints_3000000`。下载权重附带 `run_config.yaml` 明确 `train_splits: [train, val]`、128 帧、batch 256 | 代码未实现独立 val 指标选优；本次发布配置将名为 val 的 AMASS 子库也用于训练。不能断言论文私有实验没有其他开发流程 |
| EgoGVAE | train-only；每个 epoch 覆盖 `_latest.pth`，每 10 个 epoch 另外保存。实际下载的 `_latest.pth` 记录零基 epoch 243，即已到第 244 个 epoch；仅包含 epoch/model/optimizer/loss，无 val/best 字段 | 训练脚本无验证循环或按 val 更新最佳权重的逻辑。论文指标对应权重的最终挑选过程未明确，不能由 latest 推断为验证最佳 |
| EgoEgo | stage 2 定期保存并用 EMA 模型生成少量预览；评测脚本固定加载 `stage2_diffusion_4.pt` | 没有完整验证指标排序逻辑。名为 `val_dl` 的 `train=False` 读取器实际指向 `test_amass_smplh_motion.p`；不能把该变量名当成独立验证集的证据，也不能据此推断论文进行了测试集选模 |
| 本模型 EGO-only | AMASS train-only；对全部 337 个 val 原始动作、2,222 个窗口固定评测，从已捕获的 10 个候选中选 50k | **是**：按家族等权全局 MPJPE 选择；test 前固定 SHA。55k 最终模型单列，未用 test 重选 |

EgoAllo 来源：[官方训练脚本](https://github.com/brentyi/egoallo/blob/0c20ae8c8d48be4ff682803ea523bcb00a6a708d/1_train_motion_prior.py)、[官方下载入口](https://github.com/brentyi/egoallo/blob/0c20ae8c8d48be4ff682803ea523bcb00a6a708d/download_checkpoint_and_data.sh)。下载配置和身份保存在本地 `runs/amass_ego128/evaluation/baselines/egoallo/`；通过安全的 YAML BaseLoader 读取标签数据，不执行 YAML Python 对象。

EgoGVAE 来源：[训练脚本](https://github.com/DCVL-3D/EgoGVAE_release/blob/1dca529bf1efd6b53bf81da9c769ea3483777ac3/train.py)、[保存函数](https://github.com/DCVL-3D/EgoGVAE_release/blob/1dca529bf1efd6b53bf81da9c769ea3483777ac3/model/utils_checkpoint.py)。下载权重 SHA 与实际 epoch 见本地 `baselines/egogvae/CHECKPOINT.json`；元数据通过 `torch.load(weights_only=True, map_location='cpu')` 读取。不是运行模型得到的评测结果。

EgoEgo 来源：[stage 2 训练器](https://github.com/lijiaman/egoego_release/blob/main/trainer_amass_cond_motion_diffusion.py)、[stage 2 评测器](https://github.com/lijiaman/egoego_release/blob/main/eval_stage2.py)、[AMASS 读取器](https://github.com/lijiaman/egoego_release/blob/main/egoego/data/amass_diffusion_dataset.py)。该源码原生示例使用 120 帧及与 EgoAllo 不同的测试子库；汇总表中 EgoEgo 数字引用 **EgoGVAE 论文的 128 帧比较表**，不能说是原仓库默认协议的复现。

本次比较仍保持 train-only、val 选模、test 最终报告；不因为基线默认合并了 train+val 就把本次 val 重新并入训练。若之后需要同训练子库的严格控制，应显式重训相应基线，并统一相机、指标与测试窗口。
