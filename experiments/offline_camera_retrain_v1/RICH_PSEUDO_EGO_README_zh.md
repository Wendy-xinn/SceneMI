# RICH 伪 ego 处理说明

当前已经解压：

- `RICH/extracted/train_body`：63 个动作，约 3.77 万帧，包含 SMPL-X 参数和 10475 顶点人体网格；
- `RICH/extracted/train_hsc`：与人体帧对应的接触标注、最近三角形和距离；
- `RICH/extracted/scan_calibration`：6 个场景扫描和外部相机标定；
- `RICH/extracted/multicam2world`：场景扫描坐标到扫描 world 坐标的变换。

## 当前转换协议

`prepare_rich_pseudo_ego.py` 将 RICH 相机坐标先按官方 `c * vertices @ R + t` 变换到扫描 world，再变为 SceneMI 的 `x-right / y-up / z-forward` 坐标。由于 RICH 没有头戴相机轨迹，虚拟相机采用与 TRUMANS/EgoBody 一致的头部协议：以 SMPL-X 头关节 15 的全局旋转跟随头部，偏置由双眼中点/鼻尖在头部局部坐标中一次性标定，并在鼻尖外保留约 2 cm。这样不会把逐帧鼻眼拟合噪声误当成相机旋转。保存的逐帧可见点和并集地图均已转换回 SceneMI 世界坐标；不能把 `camera_cloud()` 返回的局部相机坐标直接当成场景世界坐标。

场景输入不是整张扫描直接喂入，而是从扫描表面采样后经过与 EgoBody 一致的 90°×70° 视锥（近 5 cm、远 4 m）投影；先做逐像素 z-buffer，再按像素均匀抽样。主体和 RICH 中其他已跟踪人物的 SMPL-X 网格会参与动态遮挡，保存逐帧可见点及整段伪 ego 轨迹中可见表面的并集。这与 SceneMI 的 ego-visible scene 训练约定一致，但它是由几何合成的，不是真实 RGB/depth 观测。

HSC 中的 `contact` 是 6890 点 SMPL 接触标签；10475 点 SMPL-X 接触标签编码在对应 `.obj` 的绿色顶点颜色中，转换脚本分别保存为 `contact_smpl.npy` 和 `contact_smplx.npy`。

## 当前模型资产和限制

SceneMI 仓库中已经有 `body_models/smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.{pkl,npz}`，转换脚本会用 RICH subject-gender 表和这些模型重建网格，并检查重建网格与 RICH `.ply` 的最大顶点误差。转换后还会保存精确的 22 关节、rest joints 和坐标变换后的姿态；样例的 FK 复现误差低于 0.001 cm。

仍然缺少的不是 SMPL-X 模型，而是真实的头戴相机轨迹：RICH train 只有外部多相机，因此当前结果必须标记为 pseudo-ego。多人物动作（例如 `ParkingLot1_004_005...`）的 `.ply/.pkl` 文件名是人物 ID，不是相机 ID；脚本通过 `--subject-id` 选择目标人物，其余已跟踪人物只作为动态遮挡体，不作为生成目标。
