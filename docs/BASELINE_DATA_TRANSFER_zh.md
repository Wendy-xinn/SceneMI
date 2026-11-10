# EgoAllo 重训：跨机器数据传输清单

以 2026-10-28 当前 `scene_joint128_noscene_batch32/PROTOCOL.json` 封存的 EGO 数据选择为准。对应选择合同 `configs/scene_data_current.json` 的 SHA256 为 `dc2fbb5be2cef5a711609bde054fe6a048c1c4dd3ea8f9e59b58d5ff6acd9c42`。当前 AMASS scratch 和三来源训练仍在运行，这份包固定数据和验证片段，不代表最终模型已选定。

## 必须传的内容

| 内容 | 本机位置 | 用途 |
| --- | --- | --- |
| 本小包 | `baseline_ego_transfer_20261028_v1.zip`，在项目根目录 | 三来源精确选择、原始帧映射、固定验证片段和说明；不含大数据数组 |
| GRAIL 原始人体重建 | `/mnt/c/Datasets/GRAIL/data/` 中五类动作的 `recon/`、`meta/` | 源人体参数及元信息；不能拿 `robot/` 当人体训练动作 |
| TRUMANS 原始人体参数 | `/home/wenxin/projects/TRUMANS/smplx_result/` | 世界坐标 SMPL-X 人体，按 `processed/scene_expert_v1/clips.jsonl` 对应录制 |
| 精确 EGO 相机及动作导出 | 按 `transfer_lists/NATIVE_PREPARED.files.txt` 和 `GRAIL_PREPARED.files.txt` | 保持与 SceneMI 一样的相机输入、时钟、监督及米制世界坐标；只列所需数组，不搬场景点云 |
| 人体模型 | 本机 SMPL、SMPL-X、SMPL-H，以及包内 GRAIL G1 模板 | 各自保持正确人体拓扑与骨长；另一台已有且身份相同的模型不必重复传 |

若准备直接复制完整原始数据，GRAIL 可以传整个 `/mnt/c/Datasets/GRAIL/data/`，TRUMANS 可以传整个 `/home/wenxin/projects/TRUMANS/`。这是较省整理步骤的完整方案，但大小更大。纯 EGO 基线不要求 RGB、深度视频、EXO 数组或场景点云；若另一侧要重做完整场景预处理，则还需要 GRAIL `object_usd/`、`objects/` 和 TRUMANS 场景/物体文件。

**EgoBody 另一台机器已有原始数据，因此原始大文件不必重传。** 但仍要传本包的 EgoBody train/validation 清单、帧号映射，以及文件列表中列出的已标定 EGO 相机/动作导出。只凭 recording 名称或官方 train/val CSV 不足以复现这里的具体人物、裁剪范围和有效时间段。

GRAIL 原始根在 **C 盘** `/mnt/c/Datasets/GRAIL/data/`；**D 盘** `/mnt/d/Datasets/GRAIL/scenemi_recon30_v1/` 是预处理后的逐序列数组，二者不是同一份数据。NATIVE_PREPARED 通常在 `/mnt/e/Projects/SceneMI/cache/native30_official_ego_exo_v3/`。以包内 `ROOTS.local.json` 为准，另一台机器只需要重映射根目录，不能修改源数组和序列身份。

## 小包中的文件

- `manifests/ego_train.jsonl`、`ego_validation.jsonl`：所有三来源 EGO 序列。另有 `egobody_*`、`trumans_*`、`grail_*` 单独清单。
- 每条记录有数据集、录制、人物角色、原生人体、局部帧数、`eligible_start_ranges`、预处理相对路径、各数组 SHA256。EgoBody 另有 `raw_body_fits` 和原始 SMPL 帧文件模式，TRUMANS 另有原始 SMPL-X 文件名，GRAIL 另有原始 recon/meta 对应关系。
- `frame_maps/{dataset}_{split}.npz`：完整原始帧号映射。用记录的 `source_frame_ids_key` 取出数组，`source_id = ids[local_frame]`；保留原 dtype，GRAIL 重采样对应的源帧位置可能是小数。
- `evaluation/ego_validation_cases.jsonl`：当前固定 EGO 验证片段，包含序列名、局部起止帧及原始帧映射引用；不能把 `member` 数字当跨机器通用身份，应按 dataset/sequence_id 匹配。
- `transfer_lists/*.files.txt`：对应预处理根目录的相对文件清单；`*.sha256` 用于传输后逐文件校验。只包含相机、动作、rest、有效掩码、原始时钟和必要的 GRAIL 原生姿态，不含 EXO 观测或场景点云。
- `reference/`：筛选依据、当前选择合同、原始筛选清单、帧数统计、坐标说明、指标代码、AMASS 索引/划分及 G1 模板的快照。部分快照保留本机绝对路径，仅作追溯；远端加载用上述 portable manifests 和 ROOTS 映射。
- `STATUS.json`、`SHA256SUMS`：包内汇总、来源协议指纹及完整性校验。

这份包仅携带已明确的 train/validation。GRAIL test 筛选记录作为历史证据保留在 reference 中，不代表三来源最终独立 test 清单已经齐备，不得把它接入训练。

## 远端应对齐的数量

| 来源 | train 序列 | train 可用帧 | val 序列 | val 可用帧 |
| --- | ---: | ---: | ---: | ---: |
| EgoBody | 175 | 202,950 | 45 | 47,880 |
| TRUMANS | 469 | 1,031,365 | 33 | 66,664 |
| GRAIL | 7,411 | 1,563,865 | 1,096 | 222,429 |
| 合计 | 8,055 | 2,798,180 | 1,174 | 336,973 |

训练是 30 Hz、连续 128 帧，先等权选择数据集，再选家族、该家族内的序列和接受的起点，允许重复抽样。起点范围为 **0-based、两端包含**；取 `start:start+128`，末帧为 `start+127`。禁止将坏帧两侧拼接。GRAIL train 已包括 400 条坐姿动作，EgoAllo 不需要 SceneMI 的合成椅子场景条件，也不能因此把这些动作删掉。

## EgoAllo 格式适配需要另外完成

本包不是已经转换好的 EgoAllo HDF5/NPZ，不能直接交给官方 AMASS 预处理脚本后默认认为对齐完成。当地已核对的 EgoAllo loader 使用 SMPL-H、16D betas、CPF 相对变换及 Z-up 高度；这里原始标签包括 SMPL、SMPL-X 和 GRAIL 的特殊 G1 模板。

1. 摄像机轨迹应使用双方相同的已导出输入，再按明确的相机到 CPF 约定转换；不能给 EgoAllo 从 GT 人体重新生成另一条更干净的头部轨迹，不能在推理输入中使用个人 GT betas/rest。统一设置手部输入是否启用；与当前 SceneMI 的 camera-only EGO 对比时，不额外提供手部观测。
2. `motion_world.npy` 是 SceneMI 的 201D 编码，不是 SMPL 的 `transl` 或现成米制关节：位置通道除以 2，旋转 6D 为相对单位旋转的残差。原生导出的 pelvis 为 `motion[:,:3]*2 + rest[0]`，关节为末 66 维乘 2；该语义不同于模型训练时再次编码的 common-pelvis 表示。详见坐标说明和 `native_joint_data.py` 快照。
3. GRAIL 原始采用显式 24 Hz 规则，再以旋转 SLERP、位移线性插值到 30 Hz；不可根据 recon 内嵌 fps 重新猜测。G1 的固定模板/缩放不是普通零 beta SMPL-X，须保留包内 G1 模板定义。
4. EgoAllo 官方 AMASS 预处理有截取中间 80%、估计地面、重采样等行为；其数据构造还会为了相对 CPF 变换丢首帧。不能再对已选好的片段重复裁剪/平移并直接沿用旧帧号。应显式保持当前时钟，记录相对变换所需的前一帧上下文或首帧掩码，并按固定验证窗口回验。
5. SMPL/SMPL-X/G1 到 EgoAllo 的 SMPL-H 标签/解码兼容需要几何核验，不可直接复制 betas 或假设相同姿态角产生相同骨长。预先固定转换及统一 22 关节评测政策；在适配前不能称 EgoAllo 已可直接在三来源上训练。

评测先将各方法输出解码成同一米制世界坐标的 22 关节，使用同一窗口和 GT。`reference/SceneMI/scenemi/metrics.py` 定义全局 MPJPE、标准 PA-MPJPE、root-relative MPJPE 等，并区分标准 PA 与历史兼容指标。当前 EGO 汇总先求每个 family 内窗口的平均值，再将来源内 family 等权平均，最后将三个来源等权平均；没有额外的 GRAIL 动作类别加权。准确实现见 `native_joint_training.py:dataset_summary` 和 `exo_repair_training.py:aggregate` 快照；历史 `scene_selection.py` 的类别加权汇总不代表当前联合训练汇总。

若论文还要对齐 AMASS 预训练，另核对包内 AMASS train/val/test 索引、镜像和 30 Hz 规则，以及各方法预训练/微调预算。未在另一台机器的原始 AMASS 不包含在这个小包里。
