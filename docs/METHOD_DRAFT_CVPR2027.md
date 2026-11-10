# CVPR 2027 Method 初稿

版本：2026-10-28，依据 AMASS joint scratch 与四来源 replay 的当前实现撰写。

**2026-10-29 复审说明：** 本文保留为上述实现的历史工作稿，架构尚未定稿，不再建议直接作为投稿正文。时间锚点、双前向差分和重复可靠性约束的必要性尚未得到充分对照支持；当前场景阶段也与本文的无场景联合模型不同。逐项核对、原论文及官方代码依据、参数共享表和 run 依赖说明见 [架构复审与简化讨论](ARCHITECTURE_REVIEW_zh.md)。四来源 replay 已完成完整验证，step 2000 未通过 AMASS EGO 保持门槛。

**新的可审阅方案：** [Method v2](METHOD_V2_REVIEW_zh.md)，配套 [实验配置](METHOD_V2_EXPERIMENTS_zh.md)。它是拟议的新架构，不是对下文旧实现的重新命名。

**配套文件：** 模块中文说明、参考论文、代码对应关系及待补实验见 [METHOD_NOTES_zh.md](METHOD_NOTES_zh.md)。本版 LaTeX 源文件为 [method.tex](paper/method.tex)，参考文献为 [method_references.bib](paper/method_references.bib)；它们具有相同的工作稿状态。

**范围：** 正文描述当前无场景的 EGO/EXO 联合模型。文末场景小节是独立 EGO 分支的候选扩展，尚未并入 joint128。本文未填入未经验证的实验优势。

---

## 3. Method

We study full-body motion reconstruction under two observation regimes: egocentric (EGO) camera trajectories and exocentric (EXO) body estimates with noise and missing observations. EGO reconstruction requires generating the unobserved body from camera motion, whereas EXO reconstruction requires correcting an existing estimate while preserving reliable evidence. We address both regimes with one shared motion denoiser. Our design combines a common kinematic representation, *Boundary Observation Conditioning* (BOC) for missing intervals, and *Shared-Reference Residual Denoising* (SRRD) for reliability-controlled repair. The pipeline consists of observation preparation, conditional denoising, and kinematic reconstruction.

### 3.1 Problem Formulation and Observation Preparation

Let $X\in\mathbb{R}^{T\times D}$ denote a motion sequence with $T=128$ frames at 30 Hz, $J=22$ joints, and $D=201$. In a common sequence coordinate system, each frame is represented as
$$
x_t=\left[
  \frac{p_t-s_0}{\kappa},\quad
  \{\gamma(R_{t,j})\}_{j=0}^{J-1},\quad
  \left\{\frac{P_{t,j}}{\kappa}\right\}_{j=0}^{J-1}
\right],
\tag{1}
$$
where $p_t$ is the pelvis position, $s_0$ is a common neutral pelvis reference, $\kappa=2\,\mathrm{m}$ is the spatial normalization scale, and $P_{t,j}$ are joint positions. The root rotation is expressed in sequence coordinates; the remaining rotations are local to their parents. We use the continuous 6D rotation representation [Zhou et al.](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html), centered at identity: $\gamma(R)=[Re_1;Re_2]-e$, with $e=(1,0,0,0,1,0)^\top$. Rest skeletons are translated to share $s_0$ while retaining their native bone offsets. Forward kinematics (FK) reconstructs joint positions from root translation, rotations, and a rest skeleton $S$. Every denoiser output is reconciled by orthonormalizing rotations and recomputing its redundant position channels through FK.

**EGO observations.**
A camera frontend supplies a metric camera trajectory, availability and confidence masks, and calibration metadata. An observation encoder predicts body shape from the camera sequence, which is decoded into a rest skeleton $\widehat S$. Native body families retain separate shape parameterizations and decoders. A head–camera transform converts the camera observations into head constraints; virtual cameras use a shape-dependent mount, while recorded headset cameras use the corresponding training-calibrated mount. Camera and body conditions, together with the observation-mode metadata, condition the shared denoiser. Ground-truth personal shape is used for supervision only.

**EXO observations.**
A body-estimation frontend supplies an initial motion $I$, an estimated rest skeleton $\widetilde S$, frame availability $a_t\in\{0,1\}$, joint visibility $v_{t,j}\in\{0,1\}$, root reliability $\rho^p_t\in[0,1]$, and joint reliability $\rho^R_{t,j}\in[0,1]$. These quantities distinguish an available but uncertain estimate from a missing estimate. Missing entries are filled from available estimates to construct finite network inputs, while their availability masks remain zero. The external camera retains its observer role; it is not interpreted as the target's head motion.

For controlled training, we synthesize EXO estimates by perturbing annotated body pose, root trajectory, and shape, and applying joint occlusions and temporal outages. Reliability is assigned by this corruption protocol; it is not a calibrated probability from an image estimator. The current training interface assumes full pose-and-shape estimates on available frames. The camera and body frontends provide conditioning inputs and are outside the jointly optimized motion model.

### 3.2 A Shared Conditional Motion Prior

We build on the temporal U-Net architecture of SceneMI [SceneMI](https://arxiv.org/abs/2503.16289) and train the shared motion prior from scratch. For EGO training, the standard diffusion process [DDPM](https://arxiv.org/abs/2006.11239) corrupts a clean motion $X_0$ as
$$
X_\tau=\sqrt{\bar\alpha_\tau}X_0+
       \sqrt{1-\bar\alpha_\tau}\epsilon,\qquad
\epsilon\sim\mathcal N(0,\mathrm{Id}),
\tag{2}
$$
where $\tau$ is the diffusion index and $\bar\alpha_\tau$ is the cumulative signal coefficient under a 1,000-step cosine schedule. We use $t$ exclusively for motion frames. The denoiser $F_\theta(X_\tau,\tau,c)$ predicts clean motion, conditioned on the prepared observations $c$. EGO prediction directly reconciles this output with $\widehat S$.

For EXO, we construct a deterministic temporal anchor $A$ from $I$. Its proposals comprise the initial estimate, triangular smoothing at four temporal scales, and boundary-based temporal continuation. A fixed mixture combines their root and rotation offsets; reliability gates and magnitude limits constrain the resulting displacement from $I$. This anchor provides a reproducible initialization for learned repair. Its mixture weights are fixed rather than inferred by an additional network. The next two components condition and read out the same $F_\theta$ for EXO reconstruction.

### 3.3 Boundary Observation Conditioning

A missing interval can separate a frame from informative observations beyond a local temporal neighborhood. BOC explicitly exposes the nearest available estimates on both sides. Define $\ell_t=\max\{u\leq t:a_u=1\}$ and $r_t=\min\{u\geq t:a_u=1\}$ whenever these sets are nonempty. The boundary descriptor is
$$
\begin{aligned}
b_t=\big[&
d^\ell_t,d^r_t,m^\ell_t,m^r_t,
m^\ell_t\rho^p_{\ell_t},m^r_t\rho^p_{r_t},\\
&\{m^u_t[
p^I_u-p^I_t;\ \dot p^I_u;\
\{\operatorname{Log}(R^I_{u,j}(R^I_{t,j})^\top)\}_{j}]
\}_{u\in\{\ell_t,r_t\}}\big],
\end{aligned}
\tag{3}
$$
where $m^u_t$ indicates boundary availability, $d^\ell_t=(t-\ell_t)/T$, and $d^r_t=(r_t-t)/T$. An absent boundary has distance one and zero-valued remaining features. Velocities are estimated from available frames near each boundary. The descriptor has 150 channels, including relative root displacement and joint rotations.

The ordinary EXO feature $z_t$ concatenates $I_t$, the first 135 channels of $X_{\tau,t}$, both reliability signals, availability, visibility, and $\sqrt{\bar\alpha_\tau}$, giving 383 channels. We combine temporal encoders for the ordinary and boundary features and inject their projected output into the U-Net:
$$
h=E_\phi(z)+B_\phi(b),\qquad
H_k\leftarrow H_k+
\operatorname{Resize}_k(W_k h).
\tag{4}
$$
Here $H_k$ denotes a selected encoder, bottleneck, or decoder feature map. The ordinary encoder uses dilated temporal convolutions; the boundary branch uses a short temporal convolution. The final boundary projection and all injection projections $W_k$ are initialized to zero. BOC uses both past and future observations within the window and therefore defines an offline reconstruction setting.

### 3.4 Shared-Reference Residual Denoising

An absolute motion prediction can overwrite an already useful EXO estimate. SRRD instead interprets the effect of EXO adaptation as a correction around $A$. At each diffusion step, we evaluate a reference pass and an adapted pass using the same weights, noisy state, and public observation conditions:
$$
d_\tau=g\,\Pi_{135}\!\left[
 F_\theta(X_\tau,\tau,c;\mathcal H_\phi)
 -F_\theta(X_\tau,\tau,c;0)
\right],
\tag{5}
$$
where $\mathcal H_\phi$ denotes the injected features, $\Pi_{135}$ selects root and rotation channels, and $g=5$. Both passes participate in backpropagation through the current shared parameters $\theta$. There is one denoiser parameter set, and the reference pass is recomputed at every denoising step.

Let $\operatorname{clip}_c(v)=v\min(1,c/\|v\|_2)$, with the identity convention at zero, and define $\mathcal B_c(v)=\operatorname{clip}_c(c\tanh(v/c))$. Splitting $d_\tau$ into translation and per-joint 6D rotation components gives bounded corrections
$$
\begin{aligned}
\Delta p_t &=(1-\rho^p_t)\,
            \mathcal B_{c_p}(\kappa d^p_{\tau,t}),\\
\Delta\omega_{t,j}&=(1-\rho^R_{t,j})\,
 \mathcal B_{c_R}\!\left(
 \operatorname{Log}(\operatorname{GS}(e+d^R_{\tau,t,j}))
 \right),
\end{aligned}
\tag{6}
$$
where $\operatorname{GS}$ maps two 3D vectors to an orthonormal rotation, $c_p=0.5\,\mathrm{m}$, and $c_R=1.2\,\mathrm{rad}$. The corrections act on the anchor as
$$
\widehat p_t=p^A_t+\Delta p_t,\qquad
\widehat R_{t,j}=
\operatorname{Exp}(\Delta\omega_{t,j})R^A_{t,j}.
\tag{7}
$$
We reconstruct $\widehat X_0$ through FK using $\widetilde S$. Frames with unit root and joint reliability are copied exactly from $I$. With zero injected features, the two deterministic passes coincide and the learned correction vanishes, recovering the temporal anchor. The bounds apply to each clean-motion prediction relative to $A$; they do not establish a bound on the final trajectory error.

### 3.5 Joint Training

EXO training accounts for which channels should follow the anchor. We expand reliability to the motion representation:
$$
C_t=\left[
(\rho^p_t)^{\times3},\
\{(\rho^R_{t,j})^{\times6}\}_{j},\
\{(\min(\rho^p_t,\rho^R_{t,j}))^{\times3}\}_{j}
\right],
\tag{8}
$$
where the superscript denotes repetition. With a shared Gaussian noise realization, the EXO training state is
$$
X_\tau=
\sqrt{\bar\alpha_\tau}\big[(1-C)\odot X_0+C\odot A\big]
+\sqrt{1-\bar\alpha_\tau}\epsilon .
\tag{9}
$$
EXO samples use $\tau\in\{0,\ldots,200\}$; EGO uses the full training schedule.

For EXO supervision, let $e^P_{t,j}=\|\widehat P_{t,j}-P^*_{t,j}\|_2^2/3$ and $e^R_{t,j}=\|\widehat R_{t,j}-R^*_{t,j}\|_F^2/9$. Predictions and targets use their respective estimated and ground-truth rest skeletons. Define the hidden-joint mask $H_{t,j}=1-a_tv_{t,j}$, missing-frame mask $G_t=1-a_t$, and masked mean $\langle e\rangle_M=\sum Me/\max(1,\sum M)$. Unsubscripted brackets average all relevant entries, including batch entries. Our repair objective is
$$
\begin{aligned}
\mathcal L_X={}&
10\langle e^P\rangle+10\langle e^P\rangle_H
+5\langle e^P_{\cdot,0}\rangle
+5\langle e^P_{\cdot,0}\rangle_G\\
&+2\langle e^R\rangle+4\langle e^R\rangle_H
+0.01\mathcal L_{\mathrm{vel}}+\mathcal L_{\mathrm{keep}},
\end{aligned}
\tag{10}
$$
where $\mathcal L_{\mathrm{vel}}$ is the elementwise mean squared error of joint velocities in meters per second, and
$\mathcal L_{\mathrm{keep}}=
\langle(\rho^p_t\rho^R_{t,j})^2
\|\widehat P_{t,j}-P^I_{t,j}\|_2^2/3\rangle$.
Hidden-region terms explicitly supervise uncertain and missing observations.

The EGO objective is $\mathcal L_E=\sum_{k\in\mathcal K_E}\lambda_k\mathcal L_k$. It combines clean-motion and raw-output reconstruction, FK position and orientation supervision, velocity and acceleration terms, head–root coordination, camera consistency, foot-motion regularization, and body-shape supervision. Temporal terms are weighted by $\bar\alpha_\tau$ to reduce their contribution at high noise. Native body families use their own shape heads and corresponding supervision. The full term definitions and coefficients are provided in the accompanying implementation notes.

We first jointly train EGO and EXO on AMASS, then adapt the selected prior using AMASS replay together with EgoBody, TRUMANS, and GRAIL. The source-weighted objective is
$$
\mathcal L=
\frac12\sum_s w_s\,\mathbb E_s[\mathcal L_E]
+\frac12\sum_s w_s\,\mathbb E_s[\mathcal L_X],
\qquad \sum_s w_s=1.
\tag{11}
$$
Each optimizer update accumulates both observation modes before a single parameter update. During adaptation, the bottleneck, decoder, selected condition encoders, native shape heads, and EXO adapters are trainable. Source sampling proportions and loss weights are specified separately. Model selection uses validation data.

### 3.6 Reliability-Anchored Inference

EGO inference starts from Gaussian noise and uses 20 DDIM steps [DDIM](https://arxiv.org/abs/2010.02502). EXO starts near the temporal anchor:
$$
X_{\tau_s}=\sqrt{\bar\alpha_{\tau_s}}A+
\sqrt{1-\bar\alpha_{\tau_s}}\epsilon_0,\qquad \tau_s=200.
\tag{12}
$$
We use eight denoising evaluations from $\tau_s$ to zero. At a transition $\tau\to\tau'<\tau$, the predicted clean motion defines a deterministic DDIM update:
$$
\begin{aligned}
\widehat\epsilon_\tau&=
\frac{X_\tau-\sqrt{\bar\alpha_\tau}\widehat X_0}
{\sqrt{1-\bar\alpha_\tau}},\\
U_{\tau'}&=\sqrt{\bar\alpha_{\tau'}}\widehat X_0+
\sqrt{1-\bar\alpha_{\tau'}}\widehat\epsilon_\tau .
\end{aligned}
\tag{13}
$$
Before the next denoising evaluation, we blend this state with the correspondingly noised anchor:
$$
X_{\tau'}=(1-C)\odot U_{\tau'}+
C\odot\left(
\sqrt{\bar\alpha_{\tau'}}A+
\sqrt{1-\bar\alpha_{\tau'}}\epsilon_0
\right).
\tag{14}
$$
The same $\epsilon_0$ is reused along the trajectory. High-reliability channels remain tied to the anchor, while low-reliability channels follow the denoising update. We return the final reconciled clean-motion prediction. An entirely reliable input is returned directly.

---

## 场景候选小节：与当前联合正文分开

以下英文对应已实现的独立 EGO observed-geometry adapter，可在集成与验证后改写并入主文。LaTeX 文件：[scene_extension.tex](paper/scene_extension.tex)。

### Observed Geometry Adaptation: Candidate Extension

**Scope of this draft.**
This subsection describes the implemented standalone EGO scene adapter. Its integration with the joint EGO/EXO model remains pending; it is not a description of the current joint128 experiment.

Camera reconstruction can additionally provide observed scene points in the motion coordinate system. We formulate scene conditioning through body-relative measurements around a motion proposal. A frozen EGO denoiser first predicts a kinematically reconciled motion $X^b$. Skinning the predicted body yields 67 surface anchors $q_{t,k}$, their body-region labels, and associated global rotations $Q_{t,k}$.

For each anchor, we query nearby measured points and form a 14-dimensional relation descriptor
$$
\begin{aligned}
r_{t,k}=\big[&
Q_{t,k}^{\top}(p^\star-q_{t,k})/r_q,\
d/r_q,\
Q_{t,k}^{\top}n,\
\eta/r_q,\\
&v_n,\ c_n,\ \rho_s,\ f,\ h,\ u
\big],
\end{aligned}
\tag{S1}
$$
where $p^\star$ is the nearest observed point, $d$ its distance, $n$ a locally fitted normal, and $\eta=(q_{t,k}-p^\star)^\top n$ an oriented plane offset. The remaining entries encode normal validity, neighborhood coverage, surface reliability, measured free-space evidence, a nearby-point indicator, and an unknown-space indicator. The query radius is $r_q=0.6\,\mathrm{m}$. Invalid geometric entries are masked.

Free-space evidence is restricted to measured camera-to-first-hit ray segments. A local plane offset does not establish the inside/outside sign of a full scene surface. The unknown indicator marks anchors with neither a nearby observation nor measured free-space evidence; missing geometry is not converted to a free-space label.

An MLP embeds each relation together with anchor motion features, an anchor identity embedding, and the diffusion signal level. Evidence-weighted tokens are pooled within 22 body regions and processed by temporal convolutions. Zero-initialized projections inject the resulting features into the bottleneck and two decoder resolutions of a second pass through the same frozen denoiser. Each injection is further gated by
$$
G_t=\left(\max_k g_{t,k}\right)\bar\alpha_\tau^{1/2},
\tag{S2}
$$
where $g_{t,k}$ combines local surface reliability, a weaker point-only confidence, and measured free-space evidence.

Let $Y^b$ and $Y^g$ denote the raw baseline and geometry-conditioned outputs. The output correction is bounded before kinematic reconciliation:
$$
\begin{aligned}
\widetilde Y^p &=Y^{b,p}
+\kappa^{-1}\mathcal T_{c_s}
\big(\kappa(Y^{g,p}-Y^{b,p})\big),\\
\widetilde Y^R&=Y^{b,R}
+c_6\tanh\big((Y^{g,R}-Y^{b,R})/c_6\big),
\end{aligned}
\tag{S3}
$$
where $\mathcal T_c(v)=v\,\tanh(\|v\|_2/c)/(\|v\|_2/c)$ with its continuous value at zero, $c_s=0.15\,\mathrm{m}$, and $c_6=0.5$. The rotation limit is imposed on 6D coordinates rather than rotation angles. The final positions are reconstructed through FK with the predicted rest skeleton.

This adapter currently uses a frozen EGO prior. Its proposal/query pass has a different role from the differentiable reference pass in SRRD. A joint scene model must specify their composition, gradient paths, and inference cost before these components can be presented as one evaluated method.

