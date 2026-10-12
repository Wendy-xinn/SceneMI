> 历史记录，保留审计证据。当前配置、状态和后续任务只看 [唯一总清单](EXPERIMENT_AND_ARCHITECTURE_zh.md)。下文不继续维护。

# 六个 exo 例子的遮挡检查结论

2026-10-12。训练0步。这是有意选择的遮挡检查面板，不是随机采样，也不代表全数据集遮挡比例。相机来源和置信度定义见 [实验设置与架构](EXPERIMENT_AND_ARCHITECTURE_zh.md)。

[打开 viser](http://127.0.0.1:8783)；[六例 first-hit 预览](FIRST_HIT_SIX_CASES.png)。

## 1. 每段汇总

下表平均数以192帧为分母；遮挡帧数指至少一个目标皮肤像素被该来源取代。静态遮挡包含椅子／围栏／桌面／地面等扫描表面，当前没有逐静态物体语义标签。

|例子|源帧范围|相机|当前有效/22 平均|高置信度/22 平均（最少…最多）|他人／静态／刚体遮挡帧数|
|---|---|---|---:|---:|---:|
|EgoBody_01_person_crossing|2481…2672|master 真实标定|13.79|4.33（0…10）|104／181／0|
|EgoBody_02_partner_occlusion|2291…2482|master 真实标定|9.25|4.04（0…10）|117／62／0|
|TRUMANS_01_moving_chair|816…1007|合成固定|17.52|14.81（12…18）|0／192／192|
|TRUMANS_02_sit_down|30…221|合成固定|21.48|20.98（14…22）|0／186／0|
|RICH_01_wiping_chairs|70…261|cam00 真实标定|19.58|16.89（13…21）|0／112／0|
|RICH_02_over_fence|55…246|cam00 真实标定|20.06|18.43（11…22）|0／192／0|

## 2. 每段重点

EgoBody 两例来自不同 recording 的不同目标角色；同一帧的另一人完整皮肤参与最近交点查询。TRUMANS 两例分别是移动椅子重遮挡和坐下轻遮挡对照。RICH 两例分别为擦椅子和翻越围栏。

### EgoBody_01_person_crossing

master 真实标定视角；另一人逐渐挡住 camera wearer。

外部遮挡像素最多的源帧 **2653**（片段帧172）：当前有效 **2/22**，高置信度 **0/22**；他人／静态／刚体挡住目标像素分别 **238／0／0**。整个窗口最长连续无高置信度输入为 **17帧（0.57s）**。

被外部来源遮住的像素占目标单独可见皮肤的总比例：他人36.3%、静态6.1%、刚体0.0%。这是皮肤像素比例，不是关节估计误差。

|部位|当前confidence|当前有效／高筛选|可见／目标单独|人／静态／刚体遮挡|
|---|---:|---|---:|---:|
|left_hip|0.000|False／False|0／29|29／0／0|
|right_hip|0.000|False／False|0／27|27／0／0|
|pelvis|0.000|False／False|0／26|26／0／0|
|spine3|0.000|False／False|0／24|24／0／0|
|spine2|0.000|False／False|0／20|20／0／0|
|spine1|0.000|False／False|0／17|17／0／0|
|head|0.000|False／False|0／16|16／0／0|

本地浏览器截图：[起始帧](EgoBody_01_person_crossing_start.png)、[外部视角](EgoBody_01_person_crossing_world.png)、[exo视角](EgoBody_01_person_crossing_camera.png)。

### EgoBody_02_partner_occlusion

master 真实标定视角；interactee 从可见转为被另一人遮挡。

外部遮挡像素最多的源帧 **2482**（片段帧191）：当前有效 **1/22**，高置信度 **0/22**；他人／静态／刚体挡住目标像素分别 **149／0／0**。整个窗口最长连续无高置信度输入为 **81帧（2.70s）**。

被外部来源遮住的像素占目标单独可见皮肤的总比例：他人38.1%、静态0.6%、刚体0.0%。这是皮肤像素比例，不是关节估计误差。

|部位|当前confidence|当前有效／高筛选|可见／目标单独|人／静态／刚体遮挡|
|---|---:|---|---:|---:|
|pelvis|0.000|False／False|0／26|26／0／0|
|spine3|0.000|False／False|0／21|21／0／0|
|right_hip|0.000|False／False|0／20|20／0／0|
|right_knee|0.000|False／False|1／15|14／0／0|
|head|0.000|False／False|0／13|13／0／0|
|spine1|0.000|False／False|0／12|12／0／0|
|right_shoulder|0.000|False／False|0／8|8／0／0|

本地浏览器截图：[起始帧](EgoBody_02_partner_occlusion_start.png)、[外部视角](EgoBody_02_partner_occlusion_world.png)、[exo视角](EgoBody_02_partner_occlusion_camera.png)。

### TRUMANS_01_moving_chair

合成固定 exo；坐姿移动椅子，刚体椅子及静态场景遮挡。

外部遮挡像素最多的源帧 **906**（片段帧90）：当前有效 **17/22**，高置信度 **15/22**；他人／静态／刚体挡住目标像素分别 **0／79／29**。整个窗口最长连续无高置信度输入为 **0帧（0.00s）**。

被外部来源遮住的像素占目标单独可见皮肤的总比例：他人0.0%、静态6.4%、刚体3.0%。这是皮肤像素比例，不是关节估计误差。

|部位|当前confidence|当前有效／高筛选|可见／目标单独|人／静态／刚体遮挡|
|---|---:|---|---:|---:|
|left_knee|0.541|True／False|40／74|0／34／0|
|pelvis|0.750|True／True|81／108|0／0／27|
|left_ankle|0.000|False／False|0／24|0／24／0|
|right_ankle|0.026|True／False|2／13|0／11／0|
|left_foot|0.000|False／False|0／6|0／6／0|
|right_knee|0.900|True／True|27／30|0／3／0|
|left_hip|0.980|True／True|96／98|0／0／2|

本地浏览器截图：[起始帧](TRUMANS_01_moving_chair_start.png)、[外部视角](TRUMANS_01_moving_chair_world.png)、[exo视角](TRUMANS_01_moving_chair_camera.png)。

### TRUMANS_02_sit_down

合成固定 exo；坐下与起身前动作，检查椅子及桌面遮挡。

外部遮挡像素最多的源帧 **57**（片段帧27）：当前有效 **22/22**，高置信度 **21/22**；他人／静态／刚体挡住目标像素分别 **0／17／0**。整个窗口最长连续无高置信度输入为 **0帧（0.00s）**。

被外部来源遮住的像素占目标单独可见皮肤的总比例：他人0.0%、静态0.6%、刚体0.0%。这是皮肤像素比例，不是关节估计误差。

|部位|当前confidence|当前有效／高筛选|可见／目标单独|人／静态／刚体遮挡|
|---|---:|---|---:|---:|
|right_foot|0.292|True／False|7／14|0／7／0|
|right_ankle|0.821|True／True|23／28|0／5／0|
|left_ankle|0.812|True／True|13／16|0／3／0|
|left_foot|0.867|True／True|13／15|0／2／0|
|right_elbow|1.000|True／True|47／47|0／0／0|
|left_elbow|1.000|True／True|47／47|0／0／0|
|right_shoulder|1.000|True／True|46／46|0／0／0|

本地浏览器截图：[起始帧](TRUMANS_02_sit_down_start.png)、[外部视角](TRUMANS_02_sit_down_world.png)、[exo视角](TRUMANS_02_sit_down_camera.png)。

### RICH_01_wiping_chairs

cam_00 真实标定视角；椅子遮挡擦拭动作。

外部遮挡像素最多的源帧 **113**（片段帧43）：当前有效 **15/22**，高置信度 **13/22**；他人／静态／刚体挡住目标像素分别 **0／177／0**。整个窗口最长连续无高置信度输入为 **0帧（0.00s）**。

被外部来源遮住的像素占目标单独可见皮肤的总比例：他人0.0%、静态2.8%、刚体0.0%。这是皮肤像素比例，不是关节估计误差。

|部位|当前confidence|当前有效／高筛选|可见／目标单独|人／静态／刚体遮挡|
|---|---:|---|---:|---:|
|right_knee|0.203|True／False|14／69|0／55／0|
|left_knee|0.613|True／False|68／111|0／43／0|
|left_ankle|0.000|False／False|0／36|0／36／0|
|right_ankle|0.000|False／False|0／23|0／23／0|
|right_foot|0.000|False／False|0／14|0／14／0|
|left_foot|0.000|False／False|0／6|0／6／0|
|right_elbow|1.000|True／True|45／45|0／0／0|

本地浏览器截图：[起始帧](RICH_01_wiping_chairs_start.png)、[外部视角](RICH_01_wiping_chairs_world.png)、[exo视角](RICH_01_wiping_chairs_camera.png)。

### RICH_02_over_fence

cam_00 真实标定视角；翻越围栏的下肢遮挡。

外部遮挡像素最多的源帧 **164**（片段帧109）：当前有效 **18/22**，高置信度 **14/22**；他人／静态／刚体挡住目标像素分别 **0／399／0**。整个窗口最长连续无高置信度输入为 **0帧（0.00s）**。

被外部来源遮住的像素占目标单独可见皮肤的总比例：他人0.0%、静态7.3%、刚体0.0%。这是皮肤像素比例，不是关节估计误差。

|部位|当前confidence|当前有效／高筛选|可见／目标单独|人／静态／刚体遮挡|
|---|---:|---|---:|---:|
|left_knee|0.491|True／False|113／230|0／117／0|
|right_knee|0.598|True／False|134／224|0／90／0|
|left_ankle|0.000|False／False|0／73|0／73／0|
|right_ankle|0.000|False／False|0／46|0／46／0|
|left_foot|0.000|False／False|0／24|0／24／0|
|right_hip|0.889|True／True|192／216|0／24／0|
|left_hip|0.924|True／True|281／304|0／23／0|

本地浏览器截图：[起始帧](RICH_02_over_fence_start.png)、[外部视角](RICH_02_over_fence_world.png)、[exo视角](RICH_02_over_fence_camera.png)。

## 3. 已确认和未解决的区别

- 已确认人／物体最近交点会使被完全挡住的区域失效；环境输入没有目标皮肤，未知位置不借GT填充，关节轨迹不跨缺失连接。
- 当前 availability 阈值是0.15可见比例，不是高confidence门限。两个重遮挡EgoBody窗口平均当前有效关节显著多于≥0.7筛选；若正式目标为“仅高可信轨迹”，需要显式配置硬筛选，而非把所有有效轨迹都称为高置信度。
- 关节位置是理想GT拟合轨迹，confidence是皮肤可见性代理；尚未运行真实前端，不能据此证明实际跟踪关节精度。
- 皮肤部分可见也可输入内部关节；自遮挡可见支撑、出画、目标身份混淆、真实遮挡物漏重建仍需更细校准。
- 两例EgoBody存在连续完全没有高置信度输入的区间。当前只核查mask；没有验证生成器能在这些区间补出正确或自然动作。
- 两例TRUMANS为合成相机。曾尝试同一序列后部窗口，按窗口pelvis中位数摆放相机后被墙完全遮住；已用坐下的早期片段替换。这个被拒绝的候选说明固定半径自动摆相机不保证在房间内；真实接入前要检查相机位置及视野。
- RICH只有几何／标定检查，尚无所选片段原始RGB重投影验收。所有静态扫描也可能有缺洞或拟合与场景的微小冲突；first-hit处理的是现有几何。
- 本轮六例不进入正式训练。当前正式exo分支仍为EgoBody master；新统一模型尚未训练，原55k/3k未覆盖。

## 4. 检查证据

单元检查见 [UNIT_TESTS.log](UNIT_TESTS.log)，实际浏览器交互见 [BROWSER_CHECKS.json](BROWSER_CHECKS.json)，完整身份／SHA／mask与相机检查见 [manifest.json](manifest.json)。
