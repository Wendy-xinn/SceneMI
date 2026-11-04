# SceneMI: Motion In-betweening for Modeling Human-Scene Interactions

## 从这里开始

本仓库包含原始 SceneMI、已有的 EGO/EXO 场景动作模型，以及 **AMASS 128 帧重训接口**。全量 AMASS 已准备，场景关闭、预测体形的 EGO/EXO 统一先验已完成 55k 更新。当前测试评测与论文比较的口径见下方文档。

| 要做的事 | 入口 |
|---|---|
| 阅读代码结构与常用命令 | [代码导航](docs/CODE_MAP_zh.md) |
| 查看新 AMASS 协议、配置与准备进度 | [AMASS 重训说明](docs/AMASS_RETRAIN_zh.md)、[配置](configs/amass128.yaml) |
| 查看独立 EGO-only 从零训练对照 | [EGO-only 协议与命令](docs/AMASS_EGO_ONLY_zh.md)、[配置](configs/amass_ego128.yaml) |
| 查看部分解冻的三数据集 EGO/EXO 无场景先验适配 | [联合适配协议](docs/NATIVE_JOINT_PRIOR_zh.md)、[batch实测](docs/NATIVE_BATCH_THROUGHPUT_zh.md)、[配置](configs/scene_joint128_noscene_batch32.yaml) |
| 阅读 AMASS 测试与论文对比口径 | [AMASS 评测协议](docs/AMASS_EVALUATION_zh.md) |
| 区分旧场景版、无场景版、场景修复版与当前试验 | [场景分支关系与观测表面监督](docs/SCENE_RELATIONS_zh.md) |
| 检查 TRUMANS EXO 固定/移动相机、关节输入与 30k 收敛证据 | [EXO 数据预览与审计](docs/TRUMANS_EXO_REVIEW_zh.md) |
| 查看 AMASS EXO 去噪修复、夜间检查及三数据集条件接续 | [修复训练协议与进度](docs/EXO_DENOISING_zh.md) |
| 搞清相机、头部、CPF 和 SMPL/H/X | [坐标与人体接口](docs/COORDINATES_zh.md) |
| 查看已有模型、评测与 GRAIL 状态 | [当前实验索引](experiments/offline_camera_retrain_v1/runs/INDEX_zh.md) |
| 查看本次删除范围和保留依据 | [清理记录](docs/maintenance/README_zh.md) |

```text
scenemi/          新阶段公共接口：配置、数据、人体/相机、模型
configs/          当前与保留实验配置；机器路径放 paths.local.yaml
tests/            model/data/training/evaluation/runtime 分组；legacy/ 保留历史回归
docs/             协议、代码导航、清理与验证记录
model/、train/    原始 SceneMI 实现
experiments/      既有封存模型、可复用 ops 工具和本地实验资产
```

在 `scenemi` 环境、仓库根目录执行：

```bash
python -m scenemi config
python -m scenemi --help
python -m scenemi ops --help
```

四源 EGO-only 已完成 14k 更新。新训练配置是 `configs/ego_only_4datasets.yaml`；顶层模型从 `scenemi/pipeline.py` 阅读，旧 `scene_model.py` 保留兼容导入。训练、评估、数据装配、checkpoint 分别实现在 `train.py`、`evaluate.py`、`data/ego.py`、`checkpoint.py`；旧命名文件仅保留兼容导入。训练只由用户手动启动，提供终端进度和 TensorBoard。以新配置为例：

```bash
conda activate scenemi
python -m scenemi resources run \
  --device cuda --memory-gib 10 --cpu-count 4 -- \
  python -m scenemi train --config configs/ego_only_4datasets.yaml
# 独立验证：python -m scenemi evaluate --help
# 预测预览导出与播放：python -m scenemi view prepare --help / view serve --help
# 重任务仍需使用上面的资源入口；播放只需 CPU。
tensorboard --logdir runs/ego_only_4datasets/tensorboard
```

相机误差适配配置为 `configs/ego_only_camera_adaptation.yaml`：从固定 SHA256 的 14k 完整 checkpoint 恢复权重、优化器与 RNG，在新目录续训至 16k，学习率固定为 `1e-5`。训练混合 75% 干净 / 25% 扰动窗口，前 500 次更新逐步增加扰动。当前是未按真实前置模型校准的合成试验：每轴位置抖动标准差 5mm、旋转 0.5°，窗口末端平滑漂移标准差 15mm / 0.5°，时间相关系数 0.95。

相机扰动实现在 `scenemi/data/camera_noise.py`；监督相机与输入相机分离，置信度不使用合成误差真值。适配会先执行干净与扰动完整基线验证，之后在 15k/16k 做相同验证；选模约束干净宏平均回退不超过 2mm、各数据集不超过 3mm，再比较干净/扰动平均 MPJPE。该选择分数与原始干净验证分数含义不同。窗口噪声由案例身份固定，不随评估 batch 分组改变。已有 14k 的所有指标和逐窗口记录保留于 `runs/native_ego4_from_scratch_v2/selection/full_step_014000/`，完整表格为同目录 `SUMMARY.csv`。

干净训练对照使用 `configs/ego_only_clean_continuation.yaml`，输出到独立目录 `runs/ego_only_clean_continuation`。它与扰动适配使用同一 14k 父模型、优化器/RNG、学习率、2k 更新预算和完整验证清单，仅训练扰动概率由 0.25 改为 0；验证仍保留相同的干净/扰动输入和选模规则。该对照需要自己完成 CPU/CUDA preflight，并由用户手动启动。

两个分支可在不同终端并行运行：在 CUDA 资源入口的 `run` 后加 `--gpu-sharing`，例如 `python -m scenemi resources run --gpu-sharing --device cuda --memory-gib 10 --cpu-count 4 -- python -m scenemi train --config configs/ego_only_clean_continuation.yaml --device cuda`。这两个配置的 `runtime.allocator_gib: null` 表示 GPU 显存按需分配，不安装 PyTorch 进程显存上限，也不使用固定显存余量门槛。该模式识别同一用户、本仓库的 SceneMI 训练/验收进程，未知或跨 WSL 进程仍拒绝；RAM/CPU 仍按命令行预算运行。CPU/CUDA preflight 通过后再手动训练；恢复时在子命令末尾加 `--resume`。

适配首次启动前必须分别通过 CPU 和 CUDA preflight；配置或源码变化后重新验收。验收更新只保存在进程内并丢弃，不是正式续训。正式训练仍由用户执行：

```bash
conda activate scenemi
# 将 device 依次设为 cpu、cuda；CPU 验收已执行时可直接检查其记录。
python -m scenemi resources run --gpu-sharing --device cuda --memory-gib 10 --cpu-count 4 -- \
  python -m scenemi preflight --config configs/ego_only_camera_adaptation.yaml --device cuda
python -m scenemi resources run --gpu-sharing --device cuda --memory-gib 10 --cpu-count 4 -- \
  python -m scenemi train --config configs/ego_only_camera_adaptation.yaml --device cuda
tensorboard --logdir runs/ego_only_camera_adaptation/tensorboard
```

EGO 场景阶段使用 `configs/ego_scene_adaptation.yaml`，从原始四源 14k 最佳模型初始化，追加最多 8k 更新至累计 22k。沿用 `python -m scenemi preflight/train/evaluate`、完整 checkpoint、终端进度和 TensorBoard；场景组件集中在 `scene_adaptation.py`，数据适配在 `data/scene_ego.py`，训练标签缓存由 `data/contact_labels.py` 管理。基础运动、相机和体形参数全部冻结，仅训练零初始化的静态/当前帧场景条件；AMASS 没有场景时保持原输出。每 500 步快速验证，每 1k 在同一 5,693 窗口做有场景和无场景完整验证；选模同时检查各数据集回退和无场景结果不变，只保留 last/best。

场景地图复用 `recording_scene_observed.yaml`：同一录制实际观测到的静态 first-hit 点云，明确是离线非因果输入，动态表面仅使用当前帧。GRAIL 源人体和完整网格仅用于生成训练接触可靠性标签，按录制首次读取时计算并缓存；不得作为模型输入。沿用 2–5cm 间隙衰减、支撑覆盖率和脚速判据，仅替换 planting 权重，保留摆动及其余运动监督；本阶段不开启几何吸引或强制贴地，不修改源 GT 或验证成员。训练前通过本配置 CPU/CUDA 验收，再手动执行：

场景专用数据复查使用 `python -m scenemi grail scene-review --config configs/grail_scene_review.yaml --workers 2 --output runs/data_inventory/grail_scene_review/current_selection`。它只检查当前 GRAIL train/validation，生成独立源几何证据和场景训练隔离建议，不重写原数据。场景配置的 `scene_stage.quality_review` 固定结果文件及 SHA256；只有场景训练窗口应用此覆盖表，相机-only 数据及完整验证清单不变。当前全量 8,507 条复查没有新增符合严重规则的可用窗口；仅人工报告的源例105暂时隔离24个训练窗口。自动检查不是完整身体物理正确性的证明，新增候选应结合原始皮肤/场景和未标记对照复核。复查生成的接触标签证据会被训练复用，无需再次逐条计算相同源几何。审阅界面沿用 `python -m scenemi view grail-review --annotations ... --port 8097`。

```bash
python -m scenemi resources run --gpu-sharing --device cuda --memory-gib 12 --cpu-count 4 -- \
  python -m scenemi preflight --config configs/ego_scene_adaptation.yaml --device cuda
python -m scenemi resources run --gpu-sharing --device cuda --memory-gib 12 --cpu-count 4 -- \
  python -m scenemi train --config configs/ego_scene_adaptation.yaml --device cuda
tensorboard --logdir runs/ego_scene_adaptation/tensorboard
```

重任务统一经过资源入口，具体命令见代码导航。数据、人体模型、权重、完整清理账本及缓存保存在本机，不随源码发布。新功能按上述结构扩展，不再为每个 run 增加一组 Python 脚本。下方保留原作者的说明与引用。

<p align="left">
  <a href='https://arxiv.org/abs/2503.16289'>
    <img src='https://img.shields.io/badge/Arxiv-Pdf-A42C25?style=flat&logo=arXiv&logoColor=white'></a>
  <a href='https://inwoohwang.me/SceneMI/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=white'></a>
</p>

![teaser_image](https://inwoohwang.me/SceneMI/static/images/teaser.png)


## :hammer_and_wrench: Setup

```bash
# Create environment
conda create -n scenemi python=3.9
conda activate scenemi

# Install dependencies
pip install -r requirements.txt
```

## :file_folder: Dataset and Preparation

1. **Download the TRUMANS dataset** and place it under:
   ```
   dataset/TRUMANS/Data_release/
   ```

2. **Download SMPL-X body models** and place them under:
   ```
   body_models/smplx/
   ```

3. **Preprocess the dataset:**
   ```bash
   python preprocess_dataset.py
   ```

## :rocket: Training

To train the diffusion-based SceneMI model:
```bash
python -m train.train_diffusion_scenemib
```

## :star: Citation
```
@misc{hwang2025scenemimotioninbetweeningmodeling,
    title={SceneMI: Motion In-betweening for Modeling Human-Scene Interactions}, 
    author={Inwoo Hwang and Bing Zhou and Young Min Kim and Jian Wang and Chuan Guo},
    year={2025},
    eprint={2503.16289},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2503.16289}, 
}
```

## :handshake: Acknowledgements

We sincerely thank the open-source projects that our code builds upon and draws inspiration from:  
CondMDI, TRUMANS and MDM.
