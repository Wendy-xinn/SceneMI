# Method V3：共享动作先验、可靠性 EXO 控制与场景感知重建

> 版本：2026-11-01。本文是拟议的新架构，不等同于当前已经完成的旧场景联合实验。旧实验可作为基线和输入/场景管线验证，不能直接作为本方法的结果。

## 3. 方法

### 3.1 问题定义与坐标预处理

我们的目标是在相机观测稀疏、人体估计含噪或存在缺失的情况下，重建连续的全身人体动作。模型处理两类观测：

- **EGO**：相机位姿、相机可用性、置信度、内参和设备标定；
- **EXO**：人体估计的根轨迹、局部关节旋转、估计体形、关节可见性、可用性和可靠度。

部分输入还包含与人体坐标系对齐的局部场景观测。

每个窗口包含 \(T=128\) 帧，采样率为 30 Hz，人体包含 \(J=22\) 个关节。动作表示为

\[
x_t=
\left[
\frac{p_t-s_0}{\kappa},
\{\gamma(R_{t,j})\}_{j=0}^{J-1},
\left\{\frac{P_{t,j}}{\kappa}\right\}_{j=0}^{J-1}
\right]\in\mathbb R^{201},
\]

其中 \(p_t\) 是根关节位置，\(s_0\) 是统一的中性骨盆参考，\(P_{t,j}\) 是关节位置，\(\kappa=2\) m 是空间归一化尺度。旋转使用以单位旋转为中心的六维连续表示：

\[
\gamma(R)=[Re_1;Re_2]-e,
\qquad
e=(1,0,0,0,1,0)^\top .
\]

根旋转在全局坐标系表示，其余旋转相对于父关节表示。网络只需要预测根平移和关节旋转；位置通道通过正交化旋转和前向运动学重新计算。

#### 窗口坐标中的 yaw

数据加载时会把每个窗口变换到以相机为参考的局部坐标系。给定窗口起点的相机位置 \(c_0\)、相机朝向对齐矩阵 \(A_{\mathrm{cam}}\) 和一个绕竖直轴的角度 \(\psi\)，使用

\[
A=R_y(\psi)A_{\mathrm{cam}},
\qquad
p_t^{\mathrm{local}}=(p_t^{\mathrm{world}}-c_0)A^\top ,
\qquad
R_t^{\mathrm{local}}=A R_t^{\mathrm{world}} .
\]

这里的 \(\psi\) 是数据增强或坐标规范化参数，不是网络预测的相机角度。旧数据加载器在省略 yaw 时从 \([-\pi,\pi]\) 随机采样；当前 native_ego4_from_scratch_v2 显式使用 \(\psi=0\)。每个样本都保存 \(c_0\) 和 \(A\)，可视化时必须应用逆变换恢复到世界坐标。训练输入、动作标签和场景点必须使用同一个 \(c_0,A\)。

### 3.2 共享动作扩散主干

我们使用一个共享的时序 U-Net 动作先验 \(F_{\theta_0}\)。给定带噪动作 \(X_\tau\)、扩散时间步 \(\tau\) 和条件 \(c\)，主干输出中间运动特征 \(H\)：

\[
H=F_{\theta_0}^{\mathrm{hidden}}(X_\tau,\tau,c),
\qquad
\widehat X_0=G(H).
\]

扩散前向过程为

\[
X_\tau
=
\sqrt{\bar\alpha_\tau}X_0+
\sqrt{1-\bar\alpha_\tau}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
\]

EGO 和 EXO 使用不同的观测编码器，但共享人体运动主干：

\[
c=
\begin{cases}
E_{\mathrm{ego}}(o_{\mathrm{ego}}), & \text{EGO},\\
E_{\mathrm{exo}}(o_{\mathrm{exo}}), & \text{EXO}.
\end{cases}
\]

共享部分负责学习人体动作的时序结构、关节协调和运动先验；模式专用编码器负责表达相机观测与人体估计各自的物理含义。

我们不复制完整 U-Net。EXO 和场景条件通过零初始化的残差适配器注入共享主干：

\[
H_l'
=
H_l+
Z_l(h),
\]

其中 \(h\) 是条件编码结果，\(Z_l\) 是对应层的 \(1\times1\) 投影。训练开始时 \(Z_l=0\)，因此适配器不会破坏已有动作先验。

### 3.3 EGO 条件编码

EGO 编码器接收相机位置和旋转、相机速度与角速度、相机可用性和置信度、归一化内参、设备标定和人体格式。

对于相机到人体头部的关系，AMASS 使用虚拟头戴相机，原生数据使用训练集拟合并冻结的设备安装标定。模型不会直接读取真实人体形状，而是从相机条件预测估计体形和静止姿态骨架 \(\widehat S\)。

EGO 采用绝对动作读出：

\[
\widehat z_{\mathrm{ego}}
=
G_{\mathrm{ego}}(H),
\]

其中 \(\widehat z_{\mathrm{ego}}\) 包含根平移和局部旋转，随后通过 FK 得到完整动作。

### 3.4 EXO 可靠性控制

EXO 输入包含初始人体估计 \(I\)、估计静止姿态骨架 \(\widetilde S\)、帧可用性 \(a_t\)、关节可见性 \(v_{t,j}\)、根可靠度 \(\rho^p_t\) 和关节旋转可靠度 \(\rho^R_{t,j}\)。

#### 3.4.1 逐关节条件 token

对每个关节构造 token：

\[
u_{t,j}=
[
I^p_{t,j},
I^R_{t,j},
\rho^p_{t,j},
\rho^R_{t,j},
a_{t,j},
v_{t,j},
A^p_{t,j},
A^R_{t,j},
e_j
],
\]

其中 \(A\) 是由初始估计经过简单时间平滑和缺失填充获得的参考动作，\(e_j\) 是关节身份嵌入。位置和旋转的有效性分别处理；缺失值仅用于构造有限网络输入，真实可用性仍由掩码表示。

EXO 编码器对每个关节 token 使用共享 MLP，再通过时序卷积和关节聚合得到条件特征：

\[
h_{\mathrm{exo}}
=
E_{\mathrm{exo}}(\{u_{t,j}\}_{j=0}^{J-1}).
\]

#### 3.4.2 边界上下文

对于连续缺失区间，找到当前帧左右两侧最近的实际有效估计：

\[
\ell_t=\max\{u\leq t:a_u=1\},
\qquad
r_t=\min\{u\geq t:a_u=1\}.
\]

分别构造左右端点 token：

\[
b^{L/R}_{t,j}
=
[
\Delta p^{L/R}_{t,j},
\Delta R^{L/R}_{t,j},
v^{L/R}_{t,j},
\rho^{L/R}_{t,j},
d^{L/R}_t,
m^{L/R}_t,
e_j
].
\]

当前关节 token 对左右端点和一个 null token 做 attention：

\[
h^{\mathrm{boundary}}_{t,j}
=
\operatorname{Attn}
\left(
u_{t,j},
\{b^L_{t,j},b^R_{t,j},b_\varnothing\}
\right).
\]

边界上下文只加入缺失或不可见关节的条件特征：

\[
\widetilde h_{t,j}
=
h_{\mathrm{exo},t,j}
+
m^{\mathrm{missing}}_{t,j}
W_bh^{\mathrm{boundary}}_{t,j}.
\]

该模块向网络提供缺失区间的入口状态、出口状态和过渡距离，不直接生成动作。

#### 3.4.3 单次受控去噪与 EXO 读出

将 EXO 条件编码为多尺度控制特征，并注入共享主干：

\[
H_l^{\mathrm{exo}}
=
H_l+
Z_l^{\mathrm{exo}}
\left(
E_{\mathrm{exo}}(I,A,\rho,a,v,h^{\mathrm{boundary}})
\right).
\]

与 EGO 的绝对读出不同，EXO 读出相对于参考动作 \(A\) 的修正：

\[
(\Delta p_t,\Delta r_{t,j})
=
G_{\mathrm{exo}}(H^{\mathrm{exo}}).
\]

可靠度控制修正幅度：

\[
\Delta p_t^{\mathrm{used}}
=
(1-\rho^p_t)\Delta p_t,
\]

\[
\Delta\omega_{t,j}^{\mathrm{used}}
=
(1-\rho^R_{t,j})\Delta\omega_{t,j}.
\]

最终动作由参考动作和修正组成：

\[
\widehat p_t
=
p^A_t+\Delta p_t^{\mathrm{used}},
\]

\[
\widehat R_{t,j}
=
\operatorname{Exp}
\left(
\Delta\omega_{t,j}^{\mathrm{used}}
\right)R^A_{t,j}.
\]

随后用 \(\widetilde S\) 进行 FK。这样，EXO 模型的行为是：

- 高可靠度区域尽量保持；
- 低可靠度区域可以修正；
- 缺失区域由动作先验和边界上下文补全。

EXO 不需要一个无条件 reference pass，也不需要通过两个绝对六维旋转输出相减来构造相对旋转。

### 3.5 场景感知 proposal/refinement

场景模块使用同一套人体—几何关系编码器处理 EGO 和 EXO。由于场景查询位置依赖人体位置，我们采用两阶段流程。

#### 3.5.1 Proposal

EGO 使用共享动作主干的初始输出得到 proposal；EXO 使用参考动作 \(A\) 或初始人体估计得到 proposal。proposal 只用于确定人体表面查询位置，不作为最终动作的硬约束。

对人体表面采样 \(K=67\) 个锚点：

\[
q_{t,k},\qquad k=1,\ldots,K.
\]

#### 3.5.2 局部场景关系

在实测场景点中查询每个锚点附近的观测点 \(p^\star_{t,k}\)，构造：

\[
s_{t,k}
=
[
q_{t,k}-p^\star_{t,k},
d_{t,k},
n_{t,k},
\eta_{t,k},
r_{t,k},
f_{t,k},
u_{t,k}
].
\]

其中：

- \(d_{t,k}\)：人体表面到观测点的距离；
- \(n_{t,k}\)：局部表面法向；
- \(\eta_{t,k}\)：局部平面偏移；
- \(r_{t,k}\)：表面观测可靠度；
- \(f_{t,k}\)：相机到首个实测表面之间的自由空间证据；
- \(u_{t,k}\)：未知区域标记。

没有实测证据时，场景关系被屏蔽；未知空间不被当作自由空间。

#### 3.5.3 共享场景适配器

场景关系和人体表面运动特征经过同一个场景编码器：

\[
h_{\mathrm{scene}}
=
E_{\mathrm{scene}}
\left(
\{s_{t,k},q_{t,k},\dot q_{t,k}\}_{k=1}^{K}
\right).
\]

经过人体区域池化和时序卷积后，场景特征注入共享主干：

\[
H_l'
=
H_l+
g_t
Z_l^{\mathrm{scene}}(h_{\mathrm{scene}}),
\]

其中：

\[
g_t
=
\left(\max_k g_{t,k}\right)
\sqrt{\bar\alpha_\tau}.
\]

当局部场景没有有效表面或自由空间证据时，\(g_t=0\)，模型退回到动作和观测条件。

第二阶段的受控主干输出经过 EGO 或 EXO 对应的读出头，得到最终动作。场景分支因此提供局部几何条件，不改变 EGO/EXO 的输出语义。

### 3.6 训练目标与推理

两种模式都预测相同语义的 clean motion。EGO 直接监督绝对动作；EXO 将相对参考动作的输出转换为完整动作后进行监督：

\[
\mathcal L
=
\frac12\mathcal L_{\mathrm{ego}}
+
\frac12\mathcal L_{\mathrm{exo}}
+
\lambda_{\mathrm{scene}}\mathcal L_{\mathrm{scene}}.
\]

动作损失包括 FK 位置、局部旋转、速度和必要的观测一致性项。EXO 额外报告可见区域、隐藏区域和缺失区间的误差。训练来源比例、回放顺序、学习率和 checkpoint 选择属于实验设置，不属于方法定义。

推理时：

- EGO 从噪声开始，使用相机和场景条件进行扩散采样；
- EXO 从参考动作附近初始化，使用人体条件、边界条件和场景条件进行采样；
- 每个扩散步骤只执行一次受控主干前向；
- 场景任务在需要更新查询位置时执行 proposal/refinement 两阶段；
- 最终输出通过 FK 重建。

### 3.7 方法贡献的准确表述

本文的结构贡献应表述为：

1. 使用共享动作扩散主干统一 EGO 生成和 EXO 修复；
2. 使用可靠性动作控制适配器，将 EXO 人体估计转化为对共享主干的多尺度控制；
3. 使用逐关节边界上下文补充长时间缺失区间；
4. 使用共享人体—场景关系编码器，将局部实测几何引入两类观测任务。

完整 ControlNet 复制、reference/adapted 双前向差分以及多次可靠性回填不作为本文的核心贡献。它们可以作为实验对照或历史结构。
