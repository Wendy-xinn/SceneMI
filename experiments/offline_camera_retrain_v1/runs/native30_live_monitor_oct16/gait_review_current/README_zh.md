这次回放固定使用 **10000 步**真实验证输出；主线之后继续训练不会更改本次内容。冻结源文件 SHA256 为 `bf8a44058af5d1f4377ba7d628feca540be6011ae77af4f98f7f0a8ab7cedb19`，原始验证记录与身份见 `FROZEN_RECORD.json`、`PROVENANCE.json`。

打开 [连续回放](http://localhost:8793/REPLAY.html)，点击“播放”，或拖动时间条逐帧查看。支持0.5/1/2倍速、旋转、缩放及俯视/正视/侧视。左图GT，右图预测，统一相机；橙色左脚、绿色右脚，最近一秒轨迹保留在脚后面。黄色圈代表**对应GT脚关节速度低于0.2 m/s**；观察这些时刻预测脚是否仍移动，可辨认低速期滑动。低速不等于人工标注的物理接触。

下拉菜单包含已存的三个固定窗口及三个轮换窗口，clean/noisy各自切换；没有根据预测效果筛选。固定TRUMANS/RICH有移动、迈步与转身，并非匀速直行；固定EgoBody是近静止坐姿，不能作为交替步态证据。

默认是**绝对window坐标**：与训练相同的固定世界到窗口刚性变换，单位米，Y向上；没有逐帧对齐、平滑或贴地处理。切换“骨盆相对坐标”会分别减去各自骨盆平移，仅看腿相对运动；判断漂移请关闭。这里显示由已有动作与对应native rest shape计算的真实22关节FK骨架，脚为10/11关节，**不是完整mesh或mesh脚底**。静态场景和地板未画入此步态回放，灰色坐标轴不代表已知地板。

`FOOT_TRACES_case{0..5}_{clean|noisy}.png` 共12张，每张展示绝对脚XZ路径、绝对Y高度、脚速、骨盆相对Z、GT低速期预测脚速及骨盆位移。图中实线GT、虚线预测。已人工查看 `case0_noisy` 与 `case4_noisy`，预测在不少GT低速时段仍持续移动；定量结果与限制见 [GAIT_REVIEW_zh.md](GAIT_REVIEW_zh.md)、[GAIT_METRICS.json](GAIT_METRICS.json)。

`REPLAY.html` 实测4,798,867字节（约4.6 MiB），完全离线自包含，无外部JavaScript依赖；六窗、双输入共用同一冻结数据。只用单线程CPU重建已有FK，没有新增模型forward、调用GPU或修改训练。`VISUALIZATION_MANIFEST.json` 记录所有窗口和GT stored/FK一致性。

交互验证见 `INTERACTION_CHECK.json`：用IDE附带Node执行**生成HTML中的真实脚本**，在最小Canvas/DOM宿主上遍历6窗×clean/noisy、0/96/191帧、4视角及绝对/骨盆相对模式；244676次绘图调用均无非有限数，播放会推进帧，拖动会改变当前帧。此验证覆盖脚本行为，不是完整浏览器截图测试。另已通过HTTP读取页面和PNG（HTTP 200）。

构建代码是 `build_gait_visuals.py` 和 `replay_template.html`；前者只读取本目录冻结快照，不重新读取主线latest动作，避免跨步。交互检查脚本是 `verify_replay.cjs`。大NPZ、HTML、PNG留本地，源码、协议和小指标可备份。
