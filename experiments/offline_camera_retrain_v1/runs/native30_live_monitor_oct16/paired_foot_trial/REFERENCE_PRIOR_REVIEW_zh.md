# Head-only reference先验：纯CPU复核

**存在具体先验偏置，但不能据此认定最终输出由它导致。** 用本次24.5k父模型已保存的TRUMANS／RICH椅子clean/noisy输入，直接运行原`reference_from_observations`，无扩散推理、训练或CUDA。

`native30_refinement.py:73–89`在head-only时无法拟合躯干，直接令root旋转等于头完整旋转，保留pitch/roll，且root旋转confidence复制头confidence。平移为`p_root=p_head−R_head(rest_head−rest_root)`；其余无观测骨骼继承旋转，得到绕头倾斜的静止身体。

|输入|头前向仰角°¹|reference root倾角°|GT root倾角°|root旋转置信度|reference足关节Y−GT mm|父预测root倾角°|
|---|---:|---:|---:|---:|---:|---:|
|TRU/clean|-30.47|44.65|7.26|1.000|+318.4|9.00|
|TRU/noisy|-29.18|46.96|7.26|0.512|+364.1|8.74|
|RICH椅子/clean|-54.32|58.97|21.26|1.000|+424.1|9.43|
|RICH椅子/noisy|-51.00|58.15|21.26|0.552|+400.8|10.56|

¹负值表示头前向向下；倾角是root +Y相对全局+Y。脚高为足关节Y与GT的差，不能当作离地间隙或脚底高度。

**固定头位置，仅改变向下pitch：**

|向下pitch°|reference root倾角°|足关节平均Y m|相对0°足关节上升mm|root置信度|
|---:|---:|---:|---:|---:|
|0|0.00|-1.622|0.0|1.0|
|30|30.00|-1.446|176.1|1.0|
|60|60.00|-0.883|739.8|1.0|

**Y-up存在，动力学重力项没有显式输入。** `anchor_rotation`和yaw增广均只绕Y；头／关节位置旋转保留Y方向；relation特征直接包含法线Y、垂直差和水平距离（`native30_scene_relations.py:114–118`）。未提供单独gravity/up向量、9.8m/s²、质心／支撑多边形。初始相机位置被整体减去，绝对世界高度未保留，地面相对高度依赖可见场景。

**对本次实验的含义：** ego soft-anchor只平移，不硬转root（`native30_refinement.py:144–165`）；父预测已经能大幅抵消reference倾斜。ON/OFF两臂共用此先验，物理消融阴性不能证明架构无效。修改reference会改变残差目标、查询轨迹与条件，须另做同起点训练验证；直接换给冻结权重属于分布变化诊断。

**相位边界：** 当前GT低足速指标按GT每只脚／每时刻<0.2m/s取mask（`support_metrics.py:139–150`），头轨迹不能唯一确定左右脚相位，可行的错相步态也可能受罚；这不能解释所有静止片段持续漂移。不能仅凭该数下降判胜；结合双足幅度／stride、脚高／同表面支撑和播放检查，防止冻结脚步。统计、源码／输入SHA和完整复核脚本见同名JSON。
