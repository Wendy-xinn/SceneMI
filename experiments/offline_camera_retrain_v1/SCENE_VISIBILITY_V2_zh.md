# 动态场景与人体遮挡修复记录（2026-10-05）

## 实现与边界

本次是版本化的数据处理、训练输入适配和示例验收，**没有覆盖旧 55k 数据、权重，也没有启动新训练或声称生成质量已提升**。

- TRUMANS：人体、物体按相同源时间戳对齐，30 Hz → 20 Hz；旋转用 Slerp，平移线性插值。椅子使用 `object_flag/object_mat/object_list` 与对应 `Object_chairs` 网格；底座与座面分别运动，不能用不匹配的通用椅子替代。
- 可见性：共享 `scene_visibility_v2.py` 对完整扫描、当前动态物体和人体网格求每条射线最近交点。不是视锥内全部点，也不是稀疏点云近似遮挡。输出光轴深度及命中类别，可重新计算验证。
- RICH：保留自身颈部、躯干、四肢、双手；其他已标注人物使用完整网格。缺失人物帧或拓扑不匹配直接报错，不静默删人。
- 自身头部皮肤按 SMPL-X 蒙皮权重排除，避免虚拟相机位于头皮内部时封死视野。这是“外置头戴相机”的明确建模近似，不是完整真实光学遮挡；没有删掉整个自身人体。双人数据的另一个人不排除头部。
- 两个 RICH 示例统一使用修正后的坐标变换、固定眼部偏置和一次性脸部安装校准；不使用逐帧鼻尖方向。校准矩阵来自双眼线与鼻梁线构成的脸部平面，两个序列估计出的矩阵接近，说明它是共同的坐标安装修正。红色视锥仅是可视化线框，采用屏幕线宽。
- RICH 标定文件中的三位小数旋转矩阵投影回合法旋转后，扫描、人体、相机统一使用。拒绝反射或超过 0.002 的修正，不是重新调整相机朝向。
- 当前合成深度：128×96，水平/垂直视场 66.56°/40.49°，近裁剪 0.05 m，远裁剪为光轴深度 4 m。不能假装近裁剪以内也保留了遮挡；未标注的人物/物体也无法恢复。

## 实际输入模型的内容

绿点只来自当前帧首先命中的静态场景/物体表面。自身与他人的命中用于遮挡，但不作为静态场景点输入；不可见的物体真值不会补进观测。

动态物体不写入永久静态地图。时间局部 BPS 使用“截至当前时刻已观测的静态历史 + 当前可见动态物体”，避免椅子移动后的残影；当前不可见的动态物体不使用真值续接。无观测时 BPS 有有效性 mask，而不是把全零误当接触。

`OfflineSceneMIData.sample(..., scene_bundle=...)` 是显式选择的新协议，并校验序列、20 Hz 及每帧源时间戳。全局 occupancy 仍采用整个离线窗口的**静态可见并集**，不能将这个分支描述成完全在线因果。默认旧协议保留，方便复现 55k；尚未批量重建并切换全训练集。RICH 当前两个包是 5 Hz 遮挡诊断，不是已接入 20 Hz 训练的全量数据。

物理 tracking 的碰撞场景未改。动态感知几何不应自动变成固定碰撞障碍物。坐下后完全被人体遮住的座面可能不出现在当前 ego 输入，这是不可观测性，不应通过泄露完整 GT 椅子解决；后续若加入物体跟踪记忆，需要另行定义观测与失效规则。

## 已生成并检查的实例

| 数据/序列 | 采样 | 检查结果 | Viser |
|---|---|---|---|
| TRUMANS `2023-02-13@18-03-43` | 源帧 816–1006.5；128 帧，20 Hz，首尾 6.35 s | 确为坐椅滑动；座面位移 1.63575 m，底座 1.61831 m；人体关节与原始数据最大误差 0.816 mm | 8782 |
| TRUMANS `2023-02-18@22-17-40` | 源帧 132–322.5；128 帧，20 Hz，首尾 6.35 s | 推椅，不是坐椅；用于补充动态遮挡验证 | 8780 |
| RICH `Pavallion_006_sidebalancerun` | 126 帧，5 Hz，首尾 25 s | 修正坐标变换 + 固定脸部安装校准；界面默认第 63 帧展示遮挡 | 8783 |
| RICH `ParkingLot1_004_005_greetingchattingeating1` | 80 帧，5 Hz，首尾 15.8 s | 与 8783 相同的坐标变换和校准流程；另一人物参与最近交点遮挡；默认停在第 79 帧 | 8781 |

“命中人体”与“挡住原有背景”不是同一个计数，上表分别记录。此前旧 mesh 示例 `2023-02-13@22-24-57` 缺少 Object_pose 轨迹，manifest 为 null，object_flag 为 -1，无法真实补回其椅子。本次替代序列有标注；不能声称旧例已恢复。补齐旧例对应椅子网格及逐帧变换后才能恢复。

每个实例目录包含 `metadata.json`、逐帧深度 `depth.npy`、类别 `owner.npy`、源时间戳、当前可见点、人体/物体网格、`audit_contact_sheet.png` 和 `ego_occlusion.mp4`。视频为最近命中类别图，不是彩色照片。外部扫描显示仅供对齐检查，不能拿“外部视角看见背景”判断 ego 穿透；以右侧实际 ego 深度归属图为准。

## 验证

- `test_scene_visibility_v2.py` 的 10 项测试通过：自身、他人、移动物体最近遮挡，近裁剪、相机/世界坐标、缓存等价、整体刚体变换等价、旋转插值和越界拒绝、动态无残影/无未来输入、无观测 mask。
- `audit_visibility_bundles.py` 检查四个导出包所有帧：相机正交/行列式、时间间隔、深度有效性、实际输入点与命中点一一对应。采样点重建误差为 0；TRUMANS 静态历史不含动态点。
- 对两个 RICH 例子的遮挡最强帧重新读取扫描和人体进行射线计算，结果与保存的 owner/depth 一致；单独移除人体计算背景，再验证前景深度确实更近。
- `audit_temporal_training_input.py`：真实 128 帧 v2 输入通过模型前向/反向，输出 `[1,128,201]`，236 个梯度张量均有限。随机初始化网络，零 optimizer steps；不是训练收益实验。
- 已查看四个实例的外部 mesh 与 ego 类别检查图。Viser 已启动；交互播放仍需用户在浏览器验收，不能把离线图检查说成完整交互验收。

审计文件：`data/scene_visibility_v2_oct05/bundle_integrity_audit.json`，以及 `runs/training_contract_audit_oct07/temporal_v3_verification.json`。

## 重现命令

工作目录：`SceneMI/experiments/offline_camera_retrain_v1`，Python：`/home/wenxin/miniconda3/envs/scenemi/bin/python`。

```bash
python prepare_trumans_dynamic_v2.py --sequence 2023-02-13@18-03-43 --source-start 816 --frames 128
python prepare_rich_pseudo_ego.py --sequence Pavallion_006_sidebalancerun --face-calibration --subject-id 006 --stride 6 --max-frames 128 --output data/scene_visibility_v2_oct05/rich
python prepare_rich_pseudo_ego.py --sequence ParkingLot1_004_005_greetingchattingeating1 --face-calibration --subject-id 004 --stride 6 --max-frames 128 --output data/scene_visibility_v2_oct05/rich
python -m unittest -v test_scene_visibility_v2
python audit_visibility_bundles.py
python audit_temporal_training_input.py
```

后续批量化必须先统计缺失/间断物体轨迹及缺失人物标注，不能静默回退旧路径；间断椅子轨迹当前明确报错，尚不支持物体出现/消失 presence mask。
