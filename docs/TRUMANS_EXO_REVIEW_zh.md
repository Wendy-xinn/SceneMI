# TRUMANS EXO 相机与输入检查

2026-10-26 更新：新增移动 EXO 观测版本，并完成四个原选窗口的逐帧重算。下文保留旧固定相机审计，便于对照；没有启动训练。

## 移动相机验收结果

配置为 [`configs/trumans_exo_moving.yaml`](../configs/trumans_exo_moving.yaml)，实现为 `scenemi/data/exo_camera.py`。使用已恢复的逐帧外参，一次固定的 `diag(1,-1,-1)` 光学坐标转换，以及明确声明的 192×108、66.56° 水平视场角、方形像素合成内参。没有新增根据真值人体逐帧重新对准目标的 look-at。原 RGB 内参与光学轴未核验，不能称为原视频完整标定。

四个窗口共 512 帧均已重新计算静态场景、动态物体、完整人体首命中、关节遮挡与置信度、缺失位置置零、每帧可见环境点。世界坐标与相机运动保持分离；模型输入不包含目标皮肤点。没有修改 EGO 相机定义。

| 原案例 | 旧固定平均可用关节 /22 | 移动平均可用关节 /22 | 旧固定全缺失帧 /128 | 移动全缺失帧 /128 | 移动窗口最大位移（m） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 · 坐下 | 9.60 | 17.36 | 0 | 0 | 1.905 |
| 2 · 坐下起身 | 0.00 | 17.77 | 128 | 0 | 2.859 |
| 3 · 双脚移动椅子 | 17.30 | 17.45 | 0 | 0 | 0.900 |
| 4 · 左手移动椅子 | 17.83 | 17.06 | 0 | 0 | 2.255 |

该变化是同一动作在不同相机下的观测变化，不是模型指标提升。特别是第二例，旧固定视角造成的完全遮挡已在新观测中消失；第四例也说明移动不保证每个关节都更容易看到。

核验包括：独立前向原相机坐标 SMPL-X，检查身体顶点及关节与恢复外参的一致性（最大误差低于 0.002 mm）；世界首命中点投影回当前相机像素（最大误差 0.000126 px）；全部关节的可见/静态遮挡/物体遮挡计数分解；无效帧与缺失关节遮罩。四项自动检查通过，其中几何回归测试验证相机侧移后墙后的目标从遮挡变为可见。以上检查验证内部几何一致性，不代替原 RGB 标定。

[本机可视化](http://localhost:8844/) 的案例菜单交替列出“移动”和“旧固定对照”。点击“从当前 EXO 相机看”进入对应视角；勾选“播放时跟随 EXO 相机视角”可随播放移动。紫色仍为原始人体重建，尚未经过生成模型。完整源网格和窗口可见点并集仅供展示，后者包含未来帧，不作为模型历史输入。

**范围：相机实现和四例预览验收，不是完整 TRUMANS 数据集重导出。** 新观测在 `runs/trumans_exo_moving_review/observations/`，主训练仍不能直接使用旧固定缓存冒充新版移动输入。此阶段也尚未加入估计 tracker 的深度漂移或噪声。实际输入仍是 first-hit 遮罩后的真值关节模拟。

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run --device cpu --memory-gib 6 --cpu-count 2 -- python -m scenemi view trumans move-camera --source-manifest runs/trumans_exo_review/PREPARED_CASES.json --camera-config configs/trumans_exo_moving.yaml --output runs/trumans_exo_moving_review
python -m experiments.offline_camera_retrain_v1.ops.resources run --device cpu --memory-gib 3 --cpu-count 2 -- python -m scenemi view trumans serve --manifest runs/trumans_exo_moving_review/PREPARED_CASES.json --port 8844
```

完成记录：`runs/trumans_exo_moving_review/STATUS.json`、`PROTOCOL.json`、`AUDIT.json`、`PREPARED_CASES.json`、`ACCEPTANCE.json`。`PROTOCOL.json` 记录代码、配置、人体资产及软件版本。浏览器已验证八个案例切换、全缺失警告、播放前进、进入相机视角及跟随播放，未发生 JavaScript 运行异常；详见 `BROWSER_CHECK.json`。截图为 `PREVIEW.png`、`MOVING_CAMERA_VIEW.png`、`MOVING_CAMERA_LATER.png`。原五组实验的 85 个封存源码校验仍全部匹配。

## 无场景 30k 的准确定位

30k 是目前完整验证候选中最好的 EGO-only 基座，尚不能确认已充分收敛。它从 AMASS EGO-only 50k 出发，在 EgoBody/TRUMANS/GRAIL 上不输入场景地适配 30k；不是 AMASS 的 30k，也不是 EGO/EXO 混训权重。

| 检查点 | 相同 1416 窗口 MPJPE（mm） |
| --- | ---: |
| 21k | 117.2903 |
| 28k | 116.7249 |
| 30k | 113.7933 |

28k→30k 仍改善约 2.93 mm，不能称为已验证的平台。另一次保持末端学习率的 3k 全主干续训退化，说明那一次配方失败，并不证明所有配方已无提升空间。新的冻结主干适配器仍改善，也不能用来直接判定原主干是否收敛。30k 可作为后续 EXO 适配的起点；不需要把“充分收敛”当成已经成立的前提。

## 当前 EXO 相机协议

当前 TRUMANS EXO 的 `exo_sensor_to_world` 是固定合成相机：取录制第一帧恢复的相机中心，再用合成 look-at 方向和约 66.56° 水平视场角生成首命中观测。方向按整段人体平移的中位数设定，之后不动。它不是原始视频的移动相机，也不能称为原视频逐帧校准的 EXO 基准。

导出另存了 `trumans_released_camera_to_world_reference`：由同一动作的世界坐标与相机坐标 SMPL-X 参数，考虑 native rest pelvis pivot 后恢复逐帧外参。位置轨迹明显移动；内参与光学轴尚未核验，所以页面只画移动位置轨迹，不画未经核验的原视频视锥，也不借用固定相机的可见性去冒充移动相机输入。

关节是 `ideal_GT_position_tracker_simulation_with_firsthit_confidence`：真值位置根据当前固定相机的 first-hit 可见性和有效性遮罩提供；不是实际从原视频估计的 3D tracker 输出。缺失位置为零，不提供真值关节旋转。后续接 VideoMimic 时需要对齐世界坐标、相机运动和估计误差分布。

## 可视化

[打开本机 TRUMANS EXO 页面](http://localhost:8844/)。四个案例均为验证源数据的 128 帧窗口，依据已有动作标注选择不同场景，未按模型误差挑例。紫色人体为原始 SMPL-X，橙色视锥为固定输入相机，青色为恢复的移动相机位置；网格和动态物体仅供查看，不是模型输入。

| 案例 | 原序列 | 帧区间 | 平均可用关节 /22 | 无关节帧 /128 | 移动相机窗口最大位移（m） |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | 2023-02-13@22-31-47 | 30–157 | 9.60 | 0 | 1.905 |
| 2 | 2023-02-17@10-53-45 | 480–607 | 0.00 | 128 | 2.859 |
| 3 | 2023-02-18@16-46-38 | 110–237 | 17.30 | 0 | 0.900 |
| 4 | 2023-02-18@22-16-27 | 2190–2317 | 17.83 | 0 | 2.255 |

第二例的 128 帧全部有效，但没有任何可用关节。first-hit 记录有 204,606 个人体投影像素，全部被静态场景（99,691）和刚体（104,915）挡住；不是视频帧丢失，也不是人体完全出画。该例明确标为问题对照并保留，没有删除或调整遮罩。

因此，当前固定相机 TRUMANS EXO 输入还不能直接认定已经准备好用于下一轮混训。后续需确定移动相机设定，重新一致生成相机轨迹、首命中场景、关节可用性和置信度；不能只换相机轨迹而保留旧遮罩。若采用训练覆盖筛选，应预先定义覆盖要求，并保留完整遮挡验证例单独报告。

## 下一阶段建议：保留 30k，先准备移动 EXO 观测

2026-10-26 本轮讨论的建议是：暂不再单独延长 EGO-only 全主干训练；保留原 30k 作为最强已评估无场景基座及后续适配起点。此决定基于当前实验优先级，不等于证明模型已经收敛。既有续训恢复父模型 AdamW 与随机数状态、保持末端学习率 `5e-6`，固定增加 3k 后，相同 1416 窗口 MPJPE 从 113.7933 上升至 124.2720 mm；这一分支未做 best checkpoint 搜索，因此只能判定该固定终点没有收益。证据见 `configs/scene_noscene_continue.yaml` 和 `runs/scene_ego128_noscene_low_lr/evaluation/REPORT_zh.md`。

用户希望 EXO 改为移动相机以模拟实际视频，后续准备按这个方向推进；本轮没有启动训练或将旧输入替换为新输入。具体要求如下：

1. 优先核验并利用 TRUMANS 已恢复的逐帧相机外参；先检查光学轴、内参、人体与物体重投影。官方提供世界与相机坐标的 SMPL-X 以及渲染视频，但这不等于当前已掌握逐帧完整标定。来源：[TRUMANS 官方数据说明](https://github.com/jnnan/trumans_utils#trumans-dataset)。若不能核实原内参，使用明确记录内参的合成移动相机，不能标成原视频标定相机。
2. 新版本必须逐帧同步生成相机位姿、first-hit 场景点、关节可见性、缺失标记、置信度和仅使用过去观测的历史。人体和场景须处于一致的世界坐标，不能将相机运动误当成人体根轨迹。旧固定相机导出与基线保持可复核。
3. 在几何观测正确后，再加入有时间相关性的 3D 跟踪误差，重点覆盖深度及根轨迹漂移，以及遮挡或出画造成的连续缺失。移动相机本身不能替代这些误差模拟；噪声参数待数据核验后确定，不在此处假定已经验收。
4. 首先用现有四个窗口检查新旧观测差异、完全遮挡情况及重投影，不按模型预测误差筛选。完全遮挡窗口不自动删除，训练覆盖策略与验证分组需先声明。
5. 此修改针对 EXO；EGO 保留“双眼中点前方 2 cm”的虚拟相机定义。跨数据集使用一致的坐标、遮挡和置信度语义，不要求每个数据集具有相同的相机路径。

移动 EXO 数据验收之后，再确定从 30k 进行 EXO 观测适配及 EGO 回放的训练配置，并分别检查两种模式。当前不预设混训比例，也不保证混训或移动相机会提升指标。

完整场景用于上下文检查，窗口可见点并集包含后续帧，因此页面明确标为仅展示。浏览器的“从当前固定 EXO 相机看”用于交互观察，相机位置、方向和竖直 FOV 来自导出；透明场景和浏览器视口比例使其不等同原始 RGB 或严格像素级重投影。

## 复现与核验

原始拟合保留手部姿态；四段重建的前 22 个关节均与导出逐帧完全匹配（最大误差 0）。所有显示的动态物体位置、旋转和有效性与导出一致，缺失关节未从真值补齐。

入口复用 `python -m scenemi view` 和共享 MotionViewer。选择写在运行目录清单，不新增每次预览一个脚本。CPU 重建及服务均通过现有 resources 包装器限额运行。

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run --device cpu --memory-gib 5 --cpu-count 2 -- python -m scenemi view trumans prepare --selection runs/trumans_exo_review/SELECTION.json --output runs/trumans_exo_review
python -m experiments.offline_camera_retrain_v1.ops.resources run --device cpu --memory-gib 3 --cpu-count 2 -- python -m scenemi view trumans serve --manifest runs/trumans_exo_review/PREPARED_CASES.json --port 8844
```

本地记录：`runs/trumans_exo_review/SELECTION.json`、`AUDIT.json`、`PREPARED_CASES.json`、`BROWSER_CHECK.json`；收敛复核：`runs/scene_ego128_noscene/CONVERGENCE_REVIEW.json`。
