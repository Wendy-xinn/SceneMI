# GRAIL 人体重建数据准备审查

本目录保存独立的 GRAIL 数据准备记录，不注册或更改当前三数据集训练依赖。库存扫描在 CPU 上逐条读取，未启动新模型训练。

源数据入口 `/home/wenxin/projects/GRAIL` 是指向 `/mnt/c/Datasets/GRAIL` 的软链接。下载清单固定在 NVIDIA 数据集 revision `40e795761302e611c1e7e3a6caefdd010d56c199`，五类共 105,587 个文件、23,704,067,841 字节。下载器于北京时间 2026-10-20 02:16:32 报告全部下载和内容校验完成；本次库存另检查依赖存在、重建文件大小、人体参数形状和有限值，不重复全量内容哈希。

| 类别 | 人体重建序列 | 实际源帧数／序列 | 24→30 Hz 后帧数／序列 | 人体 scale | 物体 scale |
|---|---:|---:|---:|---|---|
| sitting | 1,748 | 121 | 151 | 1 | 0.8 |
| slope | 1,880 | 241 | 301 | 1 | 1 |
| curb | 1,769 | 241 | 301 | 1 | 1 |
| stair_p1 | 6,094 | 241 | 301 | 1 或 0.7633275 | 1 |
| stair_p2 | 6,094 | 241 | 301 | 1 或 0.7633275 | 1 |

首次全范围只读扫描共 17,585 条、4,028,225 源帧、7,177,162,168 字节重建文件，无读取或人体参数有限值错误。全部人体为 neutral SMPL-X，`poses[T,165]`、`trans[T,3]`、`betas[1,10]`，betas 全零；全部 `obj_R/obj_t` 为静态。新版完整库存及统计以 `inventory.jsonl`、`SUMMARY.json` 为准。

所有 pkl 的 `mocap_frame_rate` 都写 30，然而官方同一 revision 的数据卡给重建率 24 Hz，121/241 帧对应 5/10 秒，官方重定向 CLI 也显式指定 GeniHOI 默认 24 Hz。采用源时钟 24 Hz，保留原字段为 `source_raw_fps` 并记录冲突；输出 30 Hz 时旋转用 SLERP、平移线性插值，不外推。`robot/`、`objects/` 属于 25 Hz 的 G1 仿真层，不能将它们的物体轨迹与 `recon` 人体混用。

## 坐标与几何

代表 USD 均声明 Z up、metersPerUnit=1，读取 prim 变换后，再应用重建里的 `obj_scale`、`obj_R`、`obj_t`。不能把 USD 的 upAxis 当作对局部顶点额外旋转的指令。原始 stair 中 7,361 条写有 `scale≈0.7633275`，但不能据这个字段直接使用标准 SMPL-X 或再次缩放 G1 模板。

抽查 USD 只有交互物体／地形，没有完整房间和独立地板；相机输入必须从明确的场景几何实际投影／首交点产生。若补世界 z=0 地面，必须记录为新增的合成平面，不能称为原生扫描。虚拟头戴相机不能利用未来的椅子中心或目标动作确定朝向；地板和目标物体是否可见需要用实际投影统计验证。

模板现已核实并显式选择 `grail_g1_baked_v1`。官方实现替换 `v_template`、将 shape/expression directions 置零、将 pose directions 按参考比例缩放一次，并采用 flat hand mean；在这一 forward 下忽略原始 `human_data.scale`，保留该原字段供审查。`TEMPLATE_AUDIT.json` 记录了五类共 100 条独立首帧对照，没有逐动作拟合或高度修正：sitting/slope/curb 的 60 条 G1 模板均自然贴近零高度地面，标准模板对应约 17–18cm 的穿地偏差；楼梯可从高处起步，不强压到零高度。这是官方代码与几何的交叉证据，不冒称发行文件包含明确模板标签。

`grail_assets.GrailSkinAnchorCache` 单独加载相同模板，使 rest joints、FK 和用于场景查询的 skin anchors 一致；不能把 G1 模板冒充普通零 beta 的标准人体。原始重建不是完整动作物理无穿模的保证。缺少真实接触标注，contact validity 保持无效；低于零高度地板超过 2cm 的资产先隔离审查，避免补的地板遮住下降台阶。

## 划分与窗口

没有找到官方 TRAIN／VAL／TEST 划分或原生头戴相机标定。本准备使用独立、确定性的资产分组划分：先取 `source_object_path` 最后 `/mesh/` 后的 `namespace/asset/model.obj` 为 canonical family，再调用 `grail_data.family_split` 做 80/10/10 哈希划分，最后才切窗口。相同资产的上楼／下楼、不同生成序号及 `stair_p1/p2` 必须同组。不同命名空间（如 SynStairsV2 和 V3）保留区别。

不能直接按文件名去掉末尾序号来划分：`stairs_2064` 的部分序号来自 SynStairsV2，另一些来自 SynStairsV3。也不能按 USD 文件字节哈希判定独立场景：每个序列的纹理路径不同，会让相同资产产生不同文件哈希。

sitting 重采样后仅 151 帧。落盘和可视化保存整条真实序列，独立训练 loader 支持 64/128/192 帧，并在每段有效数据上保留首、尾窗口。151 帧坐下片段使用 128 帧时包含起点 0 和 23，最后坐稳阶段不会遗漏。301 帧片段使用 192 帧时保留尾起点 109。不要重复帧、拉慢动作或拼接不同录制来制造窗口。当前封存的三数据集训练不接入新数据。

## 相机与完整动作预览

默认保留原虚拟头戴相机的 HFOV 66.56°、VFOV 40.49°、4m 光学深度范围，不逐帧 look-at 目标。额外保存五帧宽视角下俯设置用于审查，不用它替换主输入。相机图标/frustum 示意长度仅影响显示，不能据图标大小判断真实 FOV。

当前三条完整 pilot 的首交点实测：坐下 151/151 帧看到零高度地面、前 26 帧看到椅子；坡道 301/301 帧看到地面、148 帧看到坡面；楼梯 `stairs_2064__0001` 为 277/301 帧看到零高度地面、301 帧看到楼梯。椅子离开当前视野后，完整序列已观测静态记忆仍能保留先前看到的椅子表面。完整场景仅供模拟与审查，输入使用真实 first-hit 可见表面，区分 EGO/EXO 的观测记忆和当前观测。

完整动作预览：http://localhost:8840 。复用现有 MotionViewer，EGO/EXO 分栏，显示原始重建（非模型预测），可以切换完整场景、观测并集、当前观测和相机。相机视锥显示长度 0.8m 仅为示意大小。

全量输出：`data/grail_recon30_v1` 软链接到 `/mnt/d/Datasets/GRAIL/scenemi_recon30_v1`；D 盘启动前约余 524GiB。`STATUS.json` 指向该持久化任务状态，`LAUNCH.json` 保存启动命令。先在 D 盘实跑 4 条并通过所有数组校验，再从同一不可变 JOB 继续全范围 CPU 处理。原三数据集模型训练保持独立。44 项 CPU 单测通过；预览三个完整序列的 mesh 与导出关节最大误差为零。

重启后继续同一处理任务（不要在任务运行中重复启动）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  nice -n 10 ionice -c 3 /home/wenxin/miniconda3/envs/scenemi/bin/python \
  -m experiments.offline_camera_retrain_v1.ops.grail_prepare \
  --inventory experiments/offline_camera_retrain_v1/runs/grail_data_v1/inventory.jsonl \
  --output /mnt/d/Datasets/GRAIL/scenemi_recon30_v1 --scope full
```

## 可检查的代表序列

- `sitting__acadian_rustic_wooden_dining_chair_distressed_white__000`：源根平移从 `[0.24890,1.50749,0.76240]` 到 `[0.01245,0.26088,0.51322]`，平面位移约 1.27 m、根高度下降约 0.249 m；膝部由接近伸直变为明显屈曲。椅子 world bbox 约 `[-0.2899,-0.4148,0]` 至 `[0.2885,0.4168,0.8160]`。可用于验证走近椅子坐下，但仍需先验证模板和接触。
- `terrain_slopes__slope_000__000`：源 241 帧，静态坡面；对应 source、USD、meta 均在 `data/slope` 的同名条目。
- `terrain_stairs__stairs_2064__0001`：源 241 帧、人体 scale≈0.7633275，根平面位移约 3.30 m、根高度上升约 0.646 m；适合检验坡度／楼梯高度保持。属于 SynStairsV2/stairs_072，与 p1 同源条目共用划分。

全 sitting 范围根平面位移 0.859–1.751 m，中位 1.397 m；末帧根高度全部比首帧下降，中位下降 0.258 m。这支持该类适合走近后坐下的候选筛查，但不替代逐帧几何审查，也不用于选择训练／验证划分。

复现库存：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 ionice -c 3 \
  /home/wenxin/miniconda3/envs/scenemi/bin/python \
  -m experiments.offline_camera_retrain_v1.ops.grail_inventory \
  --source-root /mnt/c/Datasets/GRAIL \
  --output experiments/offline_camera_retrain_v1/runs/grail_data_v1
```

来源：[官方数据卡](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL/blob/40e795761302e611c1e7e3a6caefdd010d56c199/README.md)、[官方人体加载与 source fps](https://github.com/NVlabs/GRAIL/blob/main/grail/retargeting/retarget.py)、[G1 比例人体模板](https://github.com/NVlabs/GRAIL/blob/main/grail/models/g1_smplx_model.py)。
