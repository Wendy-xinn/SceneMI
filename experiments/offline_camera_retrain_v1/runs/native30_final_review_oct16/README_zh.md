# 10月16日：修改候选、配对选择与最终长训配置

用户授权现在修改，并要求次日给出有实际比较依据的结论和最终长训版。按此前6–12小时预算安排首轮判断，全部日期为北京时间。任务于2026-10-16 02:25启动，主进程77899；实际运行状态以[STATUS.json](STATUS.json)为准。

原Native30 V2在2453步温和停止并保存完整Adam、sampler、所有RNG，原checkpoint和根源码保持完整。2000步固定VAL的clean/noisy全身误差235.85/241.61mm，身体坐标脚位323.09/324.58mm；较1000步已有恢复，不能把早期脚位偏差当作必然失败。

|分支|变化|
|---|---|
|control|原V2，保留物理及IK|
|lean|关闭显式物理；降低head推测root朝向的置信度|
|foot|lean加raw/final逐脚身体坐标位置监督，系数1、raw权重0.5|
|ik_off|foot再关闭关节IK，保留便宜的头位置／root观测锚定|

全局patch、局部场景条件、原脚步监督、观测置信度、30Hz/L192/B2及官方划分保留。候选不改网络参数维度，不从零、不清空优化器，不接tracker或自动转EXO。来源与逐项改动见[FORK_MANIFEST.json](FORK_MANIFEST.json)及各arm的effective_config。

第一阶段四臂各1000更新，从完全相同的2453步权重／Adam／sampler／RNG开始；第二阶段control与第一阶段最有希望且通过守卫的候选各延长2000更新。所有学习率仍使用绝对步数和原55000分母。逐步核对样本身份、t、scene dropout以及noisy／轨迹／观测meta张量SHA，任何不一致不得记为配对通过。

验证使用此前预选的9个官方VAL窗（3固定、3步行、3额外覆盖），clean/noisy各自DDIM20、相同case seed。不是未使用过的TEST。每个候选先实际检查t25/300/800的完整前反传、有限梯度、无参数更新和推理输入不含GT；数据／mount／support的工程准备继承原来源，训练入口仍重新核对完整数据指纹。观测损失系数统一继承，不单独为每臂重拟合。

选择规则在结果前固定：每个regime全身≤control×1.05+2mm、头观测≤control+5mm、root朝向≤control+3度；步行局部脚位≤control×1.05+5mm、脚速RMSE≤control×1.10+0.02、幅度误差≤control×1.10+0.05。按局部脚位／幅度／脚速／全身的0.4/0.3/0.15/0.15相对误差评分，分数相差0.03以内优先计算较快者；control也参与竞争。不合格不强行换版。这些是实用开发非劣规则，不是统计显著性或论文验收标准。

选定后生成`FINAL_LONG_TRAIN_CONFIG.json`和`REPORT_zh.md`，继续到10k，增加固定窗scene-OFF检查。若相对所选阶段2端点通过同一组守卫，且复合质量改善至少2%，自动继续55k；否则在10k保留可恢复状态和明确原因，不无限消耗剩余预算。后台任务失败时保留失败记录，并在原worker已经退出后回退到原基线最多10k复核；用户主动中断不触发回退。

每臂只有一个原子`last.pt`。结束且淘汰的候选去掉Adam并核验模型张量未变，保留模型、配置、指标、RNG和配对审计；原父模型、control及被选版本继续可恢复。大权重／预测数组只留本地。

旧恢复入口（11:00故障复核后已由下段取代）：**不要对已被fallback推进parent的现状使用此入口**。沿用LAUNCH.json中的参数，将脚本改为`resume_review.py`，例如`/home/wenxin/miniconda3/envs/scenemi/bin/python -u /home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_final_review_oct16/resume_review.py --resume-manager --phase1-updates 1000 --phase2-updates 2000 --decision-hours 8 --final-stop 10000`。独立恢复入口核对STATUS中的PID及进程命令，拒绝已有同review进程；对已去Adam的落选模型先核验cleanup记录和模型hash，再幂等跳过。缺失证据时保守拒绝。它不修改六个封存扩展或原checkpoint元数据，仍校验源文件哈希、跳过已完成且步数匹配的阶段；worker按已提交checkpoint清理未提交日志尾部，并可仅重跑中断的阶段末验证。修改运行中扩展代码会触发哈希拒绝。

所有结果以实际记录为准。脚滑降低、总loss下降或短程胜出，都不单独等于步态正确或55k必然收敛。

## 11:00 故障复核与显式续行

原定8小时流程未完成：ik_off在源指纹检查退出，原manager的fallback随后因一个surface NPZ CRC损坏而停止，原目录checkpoint已到5200。前三臂各1000更新结果与配对证据有效，阶段结论见[INTERIM_REPORT_zh.md](INTERIM_REPORT_zh.md)。新检查1032项源指纹均匹配，坏surface单文件已保留证据并由原方法重建，见REAL_CACHE_REPAIR.json。

本次使用独立continue_review.py及recovered_worker.py/io_recovery.py续行原2453起点候选，保留5200 parent，不再fallback修改它；原六个封存扩展与各臂配置不改，新的扩展哈希记录在RECOVERY_PROTOCOL.json。所有原守卫保持不变，先补ik_off、再阶段2及10k门控。lean即使未胜出也保留Adam供后续审阅。CRC验证与读取重试可能影响准备耗时，比较新旧worker总耗时时需要计入这一差异。

当前恢复命令：`/home/wenxin/miniconda3/envs/scenemi/bin/python -u /home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_final_review_oct16/continue_review.py --phase1-updates 1000 --phase2-updates 2000 --decision-hours 8 --final-stop 10000 --expected-parent-step 5200`。仅在无活跃manager/worker时使用；PID守卫和进程锁拒绝重复启动。本次恢复重新计时，不声称原8小时任务已成功。

## 12:03 用户要求原版主线持续训练：本审查暂停

用户明确要求保持改输入后的原版一直训练至55000。已温和停止本目录manager及worker，IK-off完整保存在2979，其余候选3453均保留。本目录不自动恢复、不再占用主线训练预算。当前主线见[continuous_training_oct16](../native30_gait_v2_ego55k_oct15/continuous_training_oct16/README_zh.md)，原父目录会从5200持续推进；上述旧恢复命令对后续父checkpoint不再成立，不要同时运行。暂停证据见PAUSE_FOR_BASELINE.json。
