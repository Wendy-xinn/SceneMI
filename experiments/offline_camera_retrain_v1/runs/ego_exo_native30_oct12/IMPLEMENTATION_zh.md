> 历史记录：这是旧内部划分、窗口内记忆版本的零步验收。当前官方划分、跨窗记忆与全部待办只看 [唯一总清单](../exo_occlusion_review_oct12/EXPERIMENT_AND_ARCHITECTURE_zh.md)。下文不继续维护。

# native30：统一关节观测、场景 token 与位置／旋转反馈

2026-10-12。用户同意30Hz方案并补充位置修正后，已完成独立新版本的代码、配置、输入回放和零优化器步验收。**训练继续暂停；当前结果验证接口与梯度，没有验证训练后的动作或接触质量。** 原20Hz数据、v2代码和55k/3k可恢复权重保留。

输入回放：[浏览器动画](http://127.0.0.1:8782/INPUT_REPLAY.html)，或直接打开本地 [INPUT_REPLAY.html](INPUT_REPLAY.html)。可切换12种来源/模式/划分，逐帧查看192帧。蓝色骨架为GT参考，橙色为可信观测，灰色为本窗口实际观察到的静态点云，青色为当前动态表面；绿色/紫色分别为参考头部/pelvis方向。**画面没有将未训练模型输出当成生成结果。** 静态预览为 [INPUT_ALIGNMENT.png](INPUT_ALIGNMENT.png)。

## 已落实的改动

|模块|实际实现|代码|
|---|---|---|
|30Hz数据与时间轴|直接读raw30拟合、真实PV映射和物体轨迹，重新逐帧first-hit射线查询；固定192帧；保留实际虚拟安装参数；RICH contact用原始对应帧；严格拒绝20Hz/不完整缓存作训练替代|[native30_export.py](../../native30_export.py)、[native30_data.py](../../native30_data.py)|
|人体观测|self相机反算anatomical head；ego头/可信手腕与exo全部可信可见22关节共用接口；position/rotation有效性及confidence分开；不使用GT补exo旋转|[native30_observations.py](../../native30_observations.py)|
|相机条件|self用安装关系/K表达观察几何，运动只由关节输入；other使用独立光学pose、K、相对可信身体位置与固定/移动关系；全部通路屏蔽零confidence|[native30_model.py](../../native30_model.py)|
|全局场景|保留原ViT的16个patch及米制中心，再做22部位query cross-attention；CLS继续提供AdaGN summary；缺点的patch视为unknown，null token防止NaN|同上|
|当前动态场景|逐帧可见owner的点集中心、可见范围、可用速度组成token，进入身体cross-attention；动态不进入静态union；同时参与局部查询；目标皮肤排除，另一人可见皮肤保留|[native30_geometry.py](../../native30_geometry.py)|
|局部几何|67个固定native vertex，3半径8/25/60cm，各8个覆盖邻点；保留22部位token；噪声早期结合confidence加权的可靠关节邻域，后期转向预测x0皮肤；法向/owner/速度附mask|[native30_assets.py](../../native30_assets.py)、模型与geometry|
|接触反馈|接触概率、对应可靠性、时序卷积共同生成root米制位移和22关节SO(3)残差；接回当前x0并重新FK，全局关节与骨架保持一致|模型|
|监督|完整身体/原始global-joint辅助分支、绝对root/head/neck方向、相对转向、头躯干协调、各输入轨迹位置/旋转/v/a/角速度，以及mask后的contact/表面对应/相对滑动|[native30_losses.py](../../native30_losses.py)|
|准备与未来训练|当前prepare只做forward/backward、输入回放和标定记录；独立train入口需要全量native30与相同零步审计；checkpoint为一个原子last.pt，保存optimizer/全部RNG/采样状态|[prepare_native30.py](../../prepare_native30.py)、[train_native30.py](../../train_native30.py)|

## 位置具体怎么改

位置修正已经进入生成器内部，既有root直接平移，也有关节旋转导致的局部位置改变。没有给每个vertex独立位移。

```text
当前xt、t和可信条件
  → 第一遍预测完整x0 proposal（不查询GT身体）
  → 根据proposal FK/LBS，做多尺度表面查询
  → 全局patch、当前动态token与22部位几何反馈
  → 第二遍预测x0
  → contact × 对应可靠性 × noise × 有效几何门控
  → 整段时序root位移 + SO(3)旋转残差
  → FK重新计算所有global joints
  → 这个修正后的x0进入下一次DDIM更新
```

root修正最大向量长度为8cm，关节旋转修正最大为0.12rad，之后还乘预测接触/几何可靠性和`cosine_alpha²`；都在配置中显式记录。这是初始限幅，并非实验选出的最优值。两层5帧卷积的直接感受野为9帧，主干仍看整段192帧；允许局部证据影响邻帧，但没有无条件把某帧误差复制到所有相邻帧。退化的预测6D旋转使用proper SO(3)兜底，避免归一化零列导致骨架坍缩；原始6D仍接受直接监督。

没有有效邻域时，位置和旋转反馈严格为零。contact低置信度也会屏蔽反馈，包含时序层/输出bias。最终201D的global-joint分支由修正后的FK重新生成；第二遍原始global-joint输出保留辅助一致性监督。接触loss和修正路径共享局部几何，不是生成之后另跑一个外部优化器。

## 相机、形态与观测边界

头部逆变换为`R_h=R_c R_mount^T`、`p_h=p_c-R_h t_mount`。安装平移以米计，只有最终运动传输使用m/2。相机旋转缺失且安装平移非零时，head位置也标为未知；退化或不有限的6D旋转会失效，不借GT兜底。

虚拟相机使用本次导出真正用于生成的安装参数，小面板反算误差约1e-7m。真实wearer继续使用冻结TRAIN近似，位置/旋转不确定性保留在监督尺度中：小面板TRAIN/validation平均head位置误差约7.12/2.93cm。这个误差并没有被虚拟相机的精确往返检查掩盖。没有新拟合validation安装参数。

人体形态目前仍是受控实验中的已知native SMPL/SMPL-X标定：rest、body type、scale进入身体条件，betas决定skinning资产。未知网络视频的形态/身高估计未实现。固定anchor按native拓扑和gender模板选择，跨不同betas保持ID；仅选择dominant body22区域，避免拿无法预测的手指/jaw/eye顶点承担身体接触监督，脚/踝优先覆盖较低支撑表面。67点仍是稀疏查询，不是全身密集接触。

`load_frozen_ego`仅打开camera/可见场景/已知身体标定，直接转换head；`load_frozen_exo`仅打开exo可见轨迹/场景/已知身体标定。两条加载器均禁止读取motion/contact标签，已经逐项对齐模拟器输入。只提供相机时不能凭空给出可信手腕。外部前端以后可提供同世界坐标中的可信关节，但任意网络视频仍后置。

目前exo数据为EgoBody master的实际视角；interactee虚拟头戴仍是数据增广。A的真实PV观察B、自然上半身视野与长出画再入画的数据扩展尚未导出，长缺失补全继续排最后。

## 配置与监督尺度

完整配置：[native30_config.json](../../native30_config.json)，实际审计配置与代码/数据/标定指纹：[effective_config.json](effective_config.json)。

新模型从头初始化，保留共享ego/exo的201D输出。共享输出卷积采用std=0.001的小随机初始化；如果沿用整个输出全零初始化，第一步的身体loss无法反传给观测/场景编码器。root/rotation反馈出口仍为零初始化，保证起始时没有未经学习的几何位移；其时序上游需要出口开始学习后才获得身体loss梯度，contact/对应辅助监督另有直接梯度路径。旧20Hz初始化未改。

速度为m/s，加速度为m/s²，旋转导数为rad/s，缺失处的导数不跨gap。旧20Hz base中的`10*dpos²`和`2*ddpos²`分别换算为`.025*v²`和`.0000125*a²`，没有把逐帧系数直接用于物理导数。慢脚阈值使用0.2m/s；长时转向间隔使用6/12帧对应0.2/0.4秒。绝对root SO(3)仍单独监督，避免仅靠v/a或转向速度遗漏恒定朝向误差。除了最终FK一致动作，另以0.5系数监督主干原始201D x0；若只对归一化后的旋转算弦长误差，恰好180°处可能成为驻点，原始Euclidean 6D监督保留了非零纠正梯度。专门单元检查验证此路径，仍不代表训练后的朝向问题已经解决。

观测误差尺度为位置5cm、旋转10°、速度0.15m/s、加速度1m/s²、角速度20°/s；真实PV安装p90更大时提高对应位置/旋转尺度。22部位importance来自既有显式配置，head/root较强，手腕/脚等分别加权。初始观测系数由6类TRAIN输入×3噪声档的共享输出梯度比例确定，见[TRAIN_gradient_scales.json](TRAIN_gradient_scales.json)。这是小面板初始标定，不是全数据分布保证；geometry系数目前只做有限梯度检查，仍为保守初值。

默认总体ego70%/exo30%；ego65%头、35%头/双腕，exo使用全部可信可见关节。语料间按可用时间的平方根平衡，EgoBody角色按时间比例；窗口按有效时长采样；world yaw在首相机锚定之后共同作用，避免被锚定抵消。当前小面板采样比例不代表全量904段的最终比例；全量prepare会重新记录实际计划。

几何监督只使用有证据的邻域：RICH接触标签缺失时mask，不把无标签当negative；已标接触部位才施加表面距离/相对滑动；没有可靠刚体运动时不算相对速度。法向只是面向观察相机的局部PCA平面，弱半空间代理有2cm容差，不能解释为完整场景SDF或保证无穿模。

当前动态刚体速度来自模拟器已知源轨迹，仅在实际可见表面且相邻pose有效时提供，明确登记为`source_known_simulation`；这不是从无对应点云估计出的真实视频速度。另一人的皮肤速度未知，保持velocity mask=0。刚体表面速度计算`v+omega×(p-centre)`，相对滑动使用这个速度；不会把随物体运动的接触当作世界坐标静止。

默认DDIM20步，每步两次core forward，生成整段192帧；2步只是接口smoke，不能当质量设置。静态及动态输入token在一次采样中复用，结束即清掉；训练不会跨optimizer更新缓存学习特征。推理接口支持64/128/192帧，当前只验收192帧的完整通路，其他时长的质量没有测试。

## 已完成的核验与仍待验证的内容

[input_checks.json](input_checks.json)记录零优化器步的完整证据：原始时间轴、native GT/FK一致性、虚拟head转换、真实head误差、无GT加载、隐藏标签poison、零confidence/NaN屏蔽、世界yaw、各条件梯度、位置/SO(3)反馈、全窗口DDIM及scene/patch/local输出连接。

9个物理/接口单元检查通过。batch=2、192帧完整objective反向传播有限，峰值显存约2.60GiB；观测、相机、patch、动态token、局部表面和位置/旋转出口都有非零梯度。这是零步检查，未分配Adam状态，不能将该显存数当成完整训练峰值。

受控位置检查在**未训练proposal自己的脚附近**放置观察平面，不使用GT顶点，临时设置反馈bias后修正root约1.55cm、关节最多约2.14cm；骨长误差最大约1.34e-7m，global-joint/FK误差为零。测试后恢复参数，没有保存成学习权重。它证明反馈能改变位置并保持骨架，不证明真实动作改进。相同输入/noise下场景、patch和局部条件都能改变输出，但“有影响”仍不等于“影响正确”。

新模型尚未学会root定位，两个exo检查窗口在低噪声预测proposal上的局部覆盖率为零，反馈按规则关闭。这需要未来训练后再评估，不通过扩大查询到任意远处或偷偷用GT身体查询伪造覆盖。早期可信关节宽查询、全局patch及动态token提供已有可见证据，细节反馈随身体估计变好再发挥作用。

当前仅导出了8段×192帧raw30小面板，约21MiB，共12种ego/exo输入回放。全量源目录共904段（TRAIN744、validation160），输入数组粗估约14.4GiB，不含额外记录和checkpoint；**全量导出未启动**。训练入口拒绝诊断小面板，正式训练还需要全量导出、全量零步检查和后续动作/场景可视化验收。本轮训练未启动，无新checkpoint。当前磁盘剩余约67GiB；大数组/HTML回放留本地，代码与紧凑审计记录备份到user-backup。

## 复现入口

在SceneMI仓库根目录，使用`/home/wenxin/miniconda3/envs/scenemi/bin/python`：

```bash
python -m experiments.offline_camera_retrain_v1.prepare_native30 --export-review
python -m unittest experiments.offline_camera_retrain_v1.test_native30 -v
```

前者只导出检查用输入、构建模型、做forward/backward并写预览，**没有optimizer.step**。独立全量导出器按序列续做并原子更新manifest，命令为：

```bash
python -m experiments.offline_camera_retrain_v1.native30_export --split train
python -m experiments.offline_camera_retrain_v1.native30_export --split validation
python -m experiments.offline_camera_retrain_v1.prepare_native30 --full-data
```

上述全量命令本轮未执行。`train_native30 --help`只检查CLI；训练函数只能通过显式steps调用，在当前review准备状态下会拒绝运行。未来如更改代码、配置、数据或标定，须重新零步核验；原55k/3k不得覆盖。
