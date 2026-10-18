# RICH TEST完整已有结果（冻结41k EGO）

这是已完成结果的CPU纯数据导出，不是新一轮测试。模型为训练到55k的同一训练过程内、在VAL预先选择的**41k检查点**，并非55k最终权重，也不是EXO模型。冻结模型SHA256：`f8e2d22b155240c2fda8cb7d75aca451bb4152ede2441d592f3952f52068b4d0`。

## 范围与核验

52个RICH TEST录制、53条目标人体序列；335个合法窗口，每窗clean/noisy各一次，共670/670次预测。30Hz、192帧（6.4秒）、stride64、DDIM20；同case/regime只有一个固定种子样本，未做best-of-N。原始source时钟保留，跨任一目标/遮挡者缺失拟合帧的窗口排除。
53人体流共29903个源帧位，其中10个无效拟合帧位；合法窗口并集覆盖28224个不同人体帧位（94.39%）。每regime的窗口帧计数335×192=64,320，含重叠，不能视为64,320个独立帧；53人体流也不等于53个独立录制。

输入包含合成头相机轨迹、该相机录制内实际first-hit可见几何及已知body profile；学习场景与物理修正开启。三数据集训练、场景输入、192帧和22关节协议尚未对齐AMASS-only/128帧CPF或论文官方关节regressor，不能直接声明优于EgoGVAE/EgoEgo/EgoAllo。

CPU_RECHECK为independent_cpu_passed（670/670，CUDA未初始化）；全部已存原函数指标重算最大差0，独立主要公式最大差5.684341886080802e-14。VERIFY确认optimizer_updates=0、冻结模型张量不变。本次仅从原JSON/JSONL导出和聚合，未重新采样或初始化CUDA。

## 主要指标

| 指标 | 单位 | clean：等窗口 | noisy：等窗口 | clean：等录制 | noisy：等录制 | 有效窗(clean/noisy) |
|---|---|---:|---:|---:|---:|---:|
| World MPJPE `w_mpjpe_mm` | mm | 155.671955 | 169.477573 | 155.305482 | 170.378385 | 335/335 |
| Root-relative MPJPE `root_relative_mpjpe_mm` | mm | 146.876315 | 155.043476 | 147.474215 | 156.595386 | 335/335 |
| PA-MPJPE `pa_mpjpe_mm` | mm | 113.943586 | 121.100591 | 115.376992 | 123.947433 | 335/335 |
| 骨盆世界位置误差 `root_position_error_mm` | mm | 113.21688 | 117.187683 | 111.666884 | 116.452963 | 335/335 |
| 全身全局旋转误差 `regular_rotation_deg` | deg | 34.3479279 | 36.7321557 | 34.2160489 | 36.6481772 | 335/335 |
| 骨盆全局朝向误差 `root_orientation_deg` | deg | 21.3693453 | 23.3007997 | 20.7056352 | 22.4025039 | 335/335 |
| 头部全局朝向误差 `head_orientation_deg` | deg | 4.1332004 | 6.80220773 | 4.09414271 | 6.80961321 | 335/335 |
| 关节速度误差 `mpjve_mm_s` | mm/s | 337.046231 | 365.158832 | 348.027126 | 376.501787 | 335/335 |
| 关节加速度误差 `acceleration_error_mm_s2` | mm/s² | 6998.8809 | 7878.94195 | 7040.75654 | 7870.74307 | 335/335 |
| GT低速期预测水平脚速 `foot_joint_horizontal_speed_gt_low_m_s` | m/s | 0.279076337 | 0.336810735 | 0.281397617 | 0.336379151 | 335/335 |
| GT低速期预测三维脚速 `foot_speed_during_GT_stance_m_s` | m/s | 0.315486401 | 0.374502111 | 0.319491352 | 0.37667069 | 335/335 |
| GT低速期双脚关节Y绝对误差 `foot_joint_y_abs_error_gt_low_mm` | mm | 49.5789983 | 57.4743685 | 50.3812013 | 61.7676152 | 335/335 |

## 全部61项已有汇总字段

61项包含重复别名、GT参考值和描述性诊断，不能称作61个独立性能指标。完整逐字段中文定义见[ALL_METRICS.csv](ALL_METRICS.csv)。下表保留原字段；CSV保留JSON原始浮点精度，表中显示9位有效数字。

| 指标 | 单位 | clean：等窗口 | noisy：等窗口 | clean：等录制 | noisy：等录制 | 有效窗(clean/noisy) |
|---|---|---:|---:|---:|---:|---:|
| 关节加速度误差 `acceleration_error_mm_s2` | mm/s² | 6998.8809 | 7878.94195 | 7040.75654 | 7870.74307 | 335/335 |
| 双脚相对运动RMS幅值误差 `bilateral_motion_rms_error_mm` | mm | 109.97168 | 112.77697 | 120.635904 | 121.308826 | 335/335 |
| 预测双脚相对运动RMS幅值 `bilateral_motion_rms_m` | m | 0.164549076 | 0.178539985 | 0.17306517 | 0.186643981 | 335/335 |
| 双脚相对运动RMS幅值比 `bilateral_motion_rms_ratio` | 无量纲 | 1.14347731 | 1.30050545 | 1.05060877 | 1.16364078 | 331/331 |
| 双脚相对向量变化长度误差 `bilateral_stride_change_abs_error_mm` | mm | 108.482086 | 115.457697 | 115.541561 | 123.10111 | 335/335 |
| 预测双脚相对向量变化长度 `bilateral_stride_change_mean_m` | m | 0.106829001 | 0.110518372 | 0.113596597 | 0.116821756 | 335/335 |
| 全身关节垂直跨度绝对误差 `body_joint_vertical_extent_abs_error_mm` | mm | 60.751202 | 67.4282054 | 58.5525365 | 68.6789443 | 335/335 |
| 输入可用关节对GT的MPJPE `conditioned_mpjpe_mm` | mm | 0.0590412815 | 25.7714921 | 0.0744406176 | 25.9840365 | 335/335 |
| GT低速期预测水平脚速 `foot_joint_horizontal_speed_gt_low_m_s` | m/s | 0.279076337 | 0.336810735 | 0.281397617 | 0.336379151 | 335/335 |
| GT低速期水平脚速度误差 `foot_joint_horizontal_velocity_error_gt_low_m_s` | m/s | 0.289613662 | 0.345036284 | 0.291769559 | 0.344403661 | 335/335 |
| GT低速期预测垂直脚速 `foot_joint_vertical_speed_gt_low_m_s` | m/s | 0.102817153 | 0.115094359 | 0.105096658 | 0.119212923 | 335/335 |
| GT低速期垂直脚速度误差 `foot_joint_vertical_velocity_error_gt_low_m_s` | m/s | 0.110708224 | 0.122517813 | 0.112403605 | 0.125770018 | 335/335 |
| GT低速期双脚关节Y绝对误差 `foot_joint_y_abs_error_gt_low_mm` | mm | 49.5789983 | 57.4743685 | 50.3812013 | 61.7676152 | 335/335 |
| 双脚关节Y绝对误差 `foot_joint_y_abs_error_mm` | mm | 66.2813883 | 73.2585687 | 67.3850949 | 77.5487463 | 335/335 |
| GT低速期双脚关节Y有符号误差 `foot_joint_y_signed_error_gt_low_mm` | mm | 4.95340703 | 12.3993906 | -1.23160909 | 9.84602408 | 335/335 |
| 双脚关节Y有符号误差 `foot_joint_y_signed_error_mm` | mm | -14.426035 | -5.88319029 | -19.9945827 | -7.57786928 | 335/335 |
| GT低速期预测三维脚速 `foot_speed_during_GT_stance_m_s` | m/s | 0.315486401 | 0.374502111 | 0.319491352 | 0.37667069 | 335/335 |
| GT双脚相对运动RMS幅值 `gt_bilateral_motion_rms_m` | m | 0.237435765 | 0.237435765 | 0.256038053 | 0.256038053 | 335/335 |
| GT双脚相对向量变化长度 `gt_bilateral_stride_change_mean_m` | m | 0.149153602 | 0.149153602 | 0.161426803 | 0.161426803 | 335/335 |
| GT全身关节垂直跨度 `gt_body_joint_vertical_extent_mean_m` | m | 1.17543603 | 1.17543603 | 1.16799054 | 1.16799054 | 335/335 |
| GT低速期GT水平脚速 `gt_foot_joint_horizontal_speed_gt_low_m_s` | m/s | 0.0716361085 | 0.0716361085 | 0.0716436295 | 0.0716436295 | 335/335 |
| GT低速期GT垂直脚速 `gt_foot_joint_vertical_speed_gt_low_m_s` | m/s | 0.0400899063 | 0.0400899063 | 0.0399361012 | 0.0399361012 | 335/335 |
| 预测头部对GT位置误差 `gt_head_position_error_mm` | mm | 0.0590412817 | 27.7756421 | 0.074440617 | 28.0964218 | 335/335 |
| GT头至双脚最低关节垂直跨度 `gt_head_to_foot_vertical_span_mean_m` | m | 1.13757349 | 1.13757349 | 1.13241482 | 1.13241482 | 335/335 |
| GT低速脚关节区间比例 `gt_low_speed_foot_joint_fraction` | 比例 | 0.659748379 | 0.659748379 | 0.640546577 | 0.640546577 | 335/335 |
| GT低速脚关节区间数 `gt_low_speed_foot_joint_samples` | 脚关节区间/窗 | 252.023881 | 252.023881 | 244.688792 | 244.688792 | 335/335 |
| GT 骨盆旋转up轴倾角 `gt_root_tilt_deg` | deg | 23.8001618 | 23.8001618 | 23.145399 | 23.145399 | 335/335 |
| GT骨盆Y均值 `gt_root_y_mean_m` | m | -0.478523792 | -0.478523792 | -0.537021617 | -0.537021617 | 335/335 |
| GT 骨盆至颈部向量倾角 `gt_torso_pelvis_neck_tilt_deg` | deg | 29.4411057 | 29.4411057 | 27.5770484 | 27.5770484 | 335/335 |
| 头部全局朝向误差 `head_orientation_deg` | deg | 4.1332004 | 6.80220773 | 4.09414271 | 6.80961321 | 335/335 |
| 头至脚垂直跨度绝对误差 `head_to_foot_vertical_span_abs_error_mm` | mm | 49.8078165 | 55.1375694 | 50.9993958 | 59.1985213 | 335/335 |
| GT低速期左脚水平速度 `left_foot_joint_horizontal_speed_gt_low_m_s` | m/s | 0.281152196 | 0.336485799 | 0.283790966 | 0.334490645 | 335/335 |
| 左脚关节Y绝对误差 `left_foot_joint_y_abs_error_mm` | mm | 63.5804188 | 69.1014287 | 64.6569291 | 72.7425965 | 335/335 |
| 左脚关节Y有符号误差 `left_foot_joint_y_signed_error_mm` | mm | -22.1518396 | -14.7810586 | -27.9979194 | -17.0171508 | 335/335 |
| World MPJPE（原始float32别名） `mpjpe_mm` | mm | 155.671954 | 169.477573 | 155.305481 | 170.378385 | 335/335 |
| 关节速度误差 `mpjve_mm_s` | mm/s | 337.046231 | 365.158832 | 348.027126 | 376.501787 | 335/335 |
| 预测头部对输入位置误差 `observed_head_position_error_mm` | mm | 0.0590014891 | 5.44052316 | 0.074404187 | 6.35129765 | 335/335 |
| 预测头部对输入位置误差窗内P95 `observed_head_position_error_p95_mm` | mm | 0.159705676 | 16.7411723 | 0.212626187 | 20.129845 | 335/335 |
| 有效输入头帧数 `observed_head_position_valid_frames` | 帧/窗 | 192 | 186.197015 | 192 | 186.290384 | 335/335 |
| PA-MPJPE `pa_mpjpe_mm` | mm | 113.943586 | 121.100591 | 115.376992 | 123.947433 | 335/335 |
| 预测全身关节垂直跨度 `predicted_body_joint_vertical_extent_mean_m` | m | 1.17777946 | 1.17550338 | 1.18037849 | 1.17580407 | 335/335 |
| 预测头至双脚最低关节垂直跨度 `predicted_head_to_foot_vertical_span_mean_m` | m | 1.14802449 | 1.14430123 | 1.14921285 | 1.14231437 | 335/335 |
| 预测骨盆Y均值 `predicted_root_y_mean_m` | m | -0.473825188 | -0.476747835 | -0.527181871 | -0.5287261 | 335/335 |
| Regular位置MPJPE（默认全关节） `regular_position_mpjpe_mm` | mm | 155.671954 | 169.477573 | 155.305481 | 170.378385 | 335/335 |
| 全身全局旋转误差 `regular_rotation_deg` | deg | 34.3479279 | 36.7321557 | 34.2160489 | 36.6481772 | 335/335 |
| GT低速期右脚水平速度 `right_foot_joint_horizontal_speed_gt_low_m_s` | m/s | 0.27595058 | 0.336682854 | 0.276975078 | 0.33694287 | 335/335 |
| 右脚关节Y绝对误差 `right_foot_joint_y_abs_error_mm` | mm | 68.9823579 | 77.4157088 | 70.1132606 | 82.354896 | 335/335 |
| 右脚关节Y有符号误差 `right_foot_joint_y_signed_error_mm` | mm | -6.70023043 | 3.01467802 | -11.991246 | 1.86141223 | 335/335 |
| 骨盆全局朝向误差 `root_orientation_deg` | deg | 21.3693453 | 23.3007997 | 20.7056352 | 22.4025039 | 335/335 |
| 骨盆世界位置误差 `root_position_error_mm` | mm | 113.21688 | 117.187683 | 111.666884 | 116.452963 | 335/335 |
| Root-relative MPJPE `root_relative_mpjpe_mm` | mm | 146.876315 | 155.043476 | 147.474215 | 156.595386 | 335/335 |
| 骨盆旋转up轴倾角绝对误差 `root_tilt_abs_error_deg` | deg | 9.97354513 | 10.6831849 | 9.50121844 | 10.4408113 | 335/335 |
| 骨盆旋转up轴倾角 `root_tilt_deg` | deg | 19.8926875 | 19.2072121 | 19.8322718 | 19.0432543 | 335/335 |
| 骨盆旋转up轴方向误差 `root_up_direction_error_deg` | deg | 13.3881369 | 14.3726021 | 12.6698499 | 13.773044 | 335/335 |
| 骨盆Y绝对误差 `root_y_abs_error_mm` | mm | 45.3172066 | 48.672812 | 44.270381 | 49.5309064 | 335/335 |
| 骨盆Y有符号误差 `root_y_signed_error_mm` | mm | 4.69860417 | 1.77595764 | 9.83974559 | 8.29551645 | 335/335 |
| 骨盆至颈部向量方向误差 `torso_pelvis_neck_direction_error_deg` | deg | 12.6862783 | 12.7667637 | 12.5105576 | 12.6847659 | 335/335 |
| 骨盆至颈部向量倾角绝对误差 `torso_pelvis_neck_tilt_abs_error_deg` | deg | 8.89683224 | 8.93546946 | 8.76349163 | 8.9310263 | 335/335 |
| 骨盆至颈部向量倾角 `torso_pelvis_neck_tilt_deg` | deg | 29.8745581 | 29.6175946 | 29.153917 | 29.0568517 | 335/335 |
| 输入不可用关节对GT的MPJPE `unconditioned_mpjpe_mm` | mm | 163.082094 | 176.103318 | 162.697436 | 177.039176 | 335/335 |
| World MPJPE `w_mpjpe_mm` | mm | 155.671955 | 169.477573 | 155.305482 | 170.378385 | 335/335 |

## 聚合与关键定义

等窗口：在每个regime内，对每窗该指标等权平均；不是把所有脚部区间或头部有效帧合并加权。等录制：先在每个recording内对其全部合法窗口（可含多个subject流）平均，再对52个recording等权平均；不是先按subject等权。335窗有时间重叠；这些均值没有置信区间或显著性结论。

`bilateral_motion_rms_ratio`仅在GT RMS>0.01m时计算，每regime为331窗、仍覆盖52录制；case95、106、140、161为空。其余60项每regime均335个有效窗口。NULL不填0；逐窗另有`candidate_position_mpjpe_mm`、`candidate_rotation_deg`两字段全670为NULL，因此原rows有63个指标键，SUMMARY仅61个。evaluator未传GT质量权重而默认全regular，regular不是另一个高质量子集。

World：22个共同native SMPL/SMPL-X身体关节，含骨盆，世界坐标无对齐；root-relative：每帧各减自身骨盆；PA：逐帧22关节proper相似变换R+t+scale，无反射。旋转误差是全局关节SO(3)角，不是local pose角。所有速度和加速度实际使用30Hz（虽共用模块历史文字有20Hz默认描述）；只在窗内差分，不跨窗边界。

头部`gt_head_position_error_mm`为预测对GT头误差；`observed_head_position_error*`为预测对有效输入头误差，P95字段是窗内P95的均值而非全TEST pooled P95。conditioned/unconditioned按输入availability划分，噪声缺帧会改变掩码。

脚部low掩码为GT双脚关节10/11三维速度<0.2m/s，是真实接触的运动学代理。水平脚速使用XZ，三维脚速另列，不能混报。脚高是预测关节Y与GT关节Y之差；不是足底、地板高度或穿透。骨盆Y来自窗口锚定坐标，不是距地高度。双脚RMS及lag6/12向量变化使用各自骨盆朝向的左脚减右脚向量，只量化动作幅值/变化，不能当步长、相位或接触准确率。

## 尚未计算或尚未保存的指标

以下均没有本次已完成TEST数值，不能填0或由现有代理冒充：官方HSC接触precision/recall/F1、人体mesh与场景碰撞/穿透率/深度、人体mesh自穿模、PVE/MPVPE、真实足底滑移与接触保持率、local关节旋转误差、best-of-N/多样性、推理FPS或峰值显存。可选skin-anchor脚高也未计算（此调用未传asset）。独立CPU辅助函数能返回低速混淆矩阵、低速段轨迹漂移/相关性，但本TEST入口没有持久化这些补充返回值，因此也不属于本包的已有结果。

官方HSC contact标签未进入本次模型输入、窗口选择或TEST指标；SCENE_ASSET_AUDIT曾只使用HSC closest-triangle坐标识别对应扫描资产，不能说完全没有读取任何HSC文件。

## 文件

- [ALL_METRICS.csv](ALL_METRICS.csv)：61行，中文名、定义、单位及两种聚合。
- [PER_RECORDING.csv](PER_RECORDING.csv)：104行=52录制×2 regime；每指标在该录制内等窗平均。
- [PER_SEQUENCE.csv](PER_SEQUENCE.csv)：106行=53人体流×2 regime；每指标在该流内等窗平均。
- [PER_WINDOW.csv](PER_WINDOW.csv)：670行，原始身份/种子/时间、artifact哈希及原63指标键；两个全NULL字段保留为空。
- [COVERAGE.csv](COVERAGE.csv)：53人体流的源帧、合法窗、覆盖帧和无效帧。
- [EXPORT_AUDIT.json](EXPORT_AUDIT.json)：来源和输出SHA、聚合复算、行数/完整性与存储记录。
- 原始证据：[PLAN](../PLAN.json)、[SUMMARY](../SUMMARY.json)、[rows](../rows.jsonl)、[CPU_RECHECK](../CPU_RECHECK.json)、[VERIFY](../VERIFY.json)、[场景资产核验](../SCENE_ASSET_AUDIT.json)。

本包不含模型权重或预测数组。逐窗CSV的artifact绝对路径仅指向本机原文件；原始JSON中的源码、数据及权重路径用于溯源，不是随包分发的文件。
