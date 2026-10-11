# VideoMimic数据与虚拟ego相机（2026-10-11）

**视频已解压，楼梯/虚拟ego预览已完成；当前不能直接续训SMPL55k，因为公开包里缺少人体重建参数。尚未启动VideoMimic人体重建或续训。**

用户ZIP：`/home/wenxin/projects/videomimic_release_video_data.zip`，14.50GiB；解压至`/home/wenxin/projects/videomimic_release_video_data`，123段、33436张JPEG，14.51GiB。没有SMPL、场景、标定或原始FPS文件。跳过Mac资源文件，保留源ZIP。部分片段来自同一原视频，123段不能等同123独立录制；后续必须按原视频/录制分组。

另下载官方simulation数据：`/home/wenxin/projects/VideoMimic/simulation/data/videomimic_captures`，123份`background_mesh.obj`与`retarget_poses_g1.h5`，解压约410MiB。H5是G1的23个机器人DOF、root/link姿态及左右脚接触，**不是人体SMPL局部旋转**。不能复制这些关节角继续当前人体模型，也不应把G1接触标签直接当作原人体接触真值。

## 官方issue检查结果

截至本次检查查看了26条issue及可读取评论：

- [#10作者回复](https://github.com/hongsukchoi/VideoMimic/issues/10#issuecomment-3300346288)：2025-09先发布重建动作，视频当时尚未发布。现在用户下载的JPEG视频包已发布。
- [#12作者回复](https://github.com/hongsukchoi/VideoMimic/issues/12#issuecomment-3403468810)：数据下载后存放simulation/data，指向simulation下载指南。实际检查这个包只有G1动作和背景网格。
- [#20作者回复](https://github.com/hongsukchoi/VideoMimic/issues/20#issuecomment-3561873937)：assets迁到Hugging Face。只用HTTP Range检查33.90GB assets ZIP的171个目录项，未下载整个包；里面未发现人体重建H5，主要是模型、配置和权重。
- 没有找到明确提供原视频SMPL/world人体重建结果的issue回复。**“需要自己跑人体重建”是根据公开资源检查作出的判断，而非作者明确说该人体包不会发布。**

官方[阶段说明](https://github.com/hongsukchoi/VideoMimic/blob/main/real2sim/docs/commands.md)给出了所需输出：`gravity_calibrated_megahunter.h5`中的人体参数、帧对应与同坐标的场景。获取当前缺失内容需运行阶段0–3；我们训练人体生成器，不需要先做阶段4机器人retarget。

## 可以查看的预览

Viser：`http://127.0.0.1:8781`，片段IMG_8044。它展示官方楼梯网格和蓝色G1参考运动，紫色虚拟相机及轨迹、黄色当前可见地形点；可开启ego跟随，并查看深度。**这不是当前SMPL生成模型在新地形上的结果。**

相机安装于真实head mesh的额头前方2cm，固定向下20°用于地形预览。没有把机器人head_link的原点直接当眼睛：此模型的头部几何在frame上方约0.325–0.531m。逐帧使用发布的世界link姿态，四元数xyzw。场景world+Z向上，此预览保留原坐标；以后人体训练将明确转到SceneMI的+Y向上。

采用现有精确mesh first-hit raycasting：128×96、66.56°×40.49°、0.05–4m，身体作为当前帧遮挡体，去掉头部；机器人命中不作为场景点。不积累身体或动态拖影。楼梯mesh高度跨度1.857m，采用向下安装后平均可见场景点填充约93.9%。完整背景mesh用于模拟传感器；模型以后应使用可见点/因果记忆，而非自动获得未观测完整mesh。公开视频包没有动态物体轨迹，不能因此声称所有动态物体已处理。

## 需要先解决的接入检查

1. **时间**：H5头写15/30Hz，导出代码固定写`30/subsample`；官方123条训练配置中122条写60Hz覆盖、1条写6Hz，而读取代码并未在H5已有fps时使用该覆盖值。无法仅凭这些字段认证真实时间。重建须核对原视频FPS与帧ID/抽帧stride，直接原生帧→20Hz；不先降到15Hz再补20Hz。若只剩低于20Hz的拟合，重跑更密的原始帧。
2. **人体**：原生SMPL即可接入现有SMPL/SMPL-X共享22关节接口，不需先转成SMPL-X。官方human group虽然名叫`our_pred_humans_smplx_params`，当前可视化代码实际创建SMPL、读取23个body rotation，必须核对真实拓扑。
3. **尺度/坐标**：VideoMimic的root_transl是绝对pelvis坐标，不是SMPL的transl；必须减去native rest root。人体和mesh须使用同一重力/尺度变换。避免`--use-g1-shape`把人体和世界重定标为机器人比例后又当作原人尺度。
4. **场景与脚步**：逐时刻可见mesh、身体遮挡、因果静态记忆、动态current-only规则保留。楼梯/坡面不能用一个最低水平地板平面做支撑；查询每只脚下的实际局部表面/法向并处理未知区域。重建轨迹是伪标签，需检查脚滑、穿透、台阶落脚和接触置信度。
5. **窗口**：按可靠时间计算可用长度，优先64帧（3.2s）；不足128帧的片段不能补帧/跨不同录制拼接。按原视频/场景分组后，再用原55k独立续训，加入旧数据回放防遗忘，另留未训练地形验收。

## 已实现的人体导入检查

`videomimic_human_contract.py`验证原生人体SO(3)、固定shape、帧ID/频率、重力坐标和绝对pelvis语义，并生成native22与刚性头部虚拟相机。用安装的完整SMPL forward交叉检查native FK；拒绝机器人H5、反射矩阵、低频后补帧和长追踪缺口。四项测试通过，含完整native SMPL坐标/相机fixture。**fixture是接口测试，不是本数据集重建结果。**

拿到人体结果后可执行（FPS/gender/up-axis必须用源资料确认，不能照抄假设）：

```bash
PYTHONPATH=. /home/wenxin/miniconda3/envs/scenemi/bin/python \
  -m experiments.offline_camera_retrain_v1.videomimic_human_contract \
  /真实重建目录/gravity_calibrated_megahunter.h5 \
  --person-id 1 --source-fps 60 --gender male --source-up-axis z \
  --output /核验输出目录
```

这只是人体接入预检；输出明确标记training_ready=false，仍需同帧场景渲染/完整缓存、划分及几何验收。尚未实现或启动新数据续训，不用G1轨迹冒充人体。下一项有效工作是先在一个楼梯片段完成原人体重建和mesh一致性验收，再准备续训。
