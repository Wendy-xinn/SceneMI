# SceneMI: Motion In-betweening for Modeling Human-Scene Interactions

## 本仓库的本地实验扩展

本仓库基于下方原作者SceneMI代码，加入TRUMANS / EgoBody / RICH混合原生SMPL与SMPL-X监督、20Hz因果动态场景、RICH接触预测、标准动作指标，以及转身和不可靠轨迹条件诊断。

- [代码备份范围及本地依赖](docs/BACKUP_zh.md)
- [原生动态场景55k实验结果](experiments/offline_camera_retrain_v1/runs/native_dynamic_scene20_contact_55k_oct07/RESULTS_zh.md)
- [时间归一化和头部硬约束审核](experiments/offline_camera_retrain_v1/runs/turn_balance_oct09/ASSESSMENT_zh.md)
- [软轨迹与身体转向短训](experiments/offline_camera_retrain_v1/runs/soft_body_turn_robust_oct09/README_zh.md)

数据、人体模型、训练checkpoint和生成缓存仅保存在本机，不包含在代码备份中。下方为原作者项目说明及引用。

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