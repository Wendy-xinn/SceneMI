# 原版主线持续训练至55k

2026-10-16用户明确要求保持“改输入后的原版”一直训练。本目录恢复native30_gait_v2_ego55k_oct15的5200步模型、Adam、sampler及所有随机状态，原配置/模型/损失/55k学习率分母不改。所有消融调度已暂停，候选checkpoint保留；不再让选版流程占掉主线训练。

运行入口为supervise.py，子进程worker.py通过原封存run_training.py --resume --review-step 55000执行。到10k照常验证后继续，不受之前候选的10k选版门槛控制。每100步原子覆写原目录training/last.pt，保留原周期验证。

共享io_recovery.py只在原源SHA被重新确认后接受暂态错读；派生场景缓存先完整CRC校验，损坏则保留证据、由原方法重建。监督器仅对已识别暂态IO最多连续重启3次，参数/来源不匹配、数值异常或未知错误停止并显式记录。只在真正55000步完成后标记完成，不自动启动消融/EXO/tracker。

进度看STATUS.json（实际trace步数、最近已保存步数、worker PID）；worker.log记录原训练器输出，IO_RECOVERY.jsonl记录完整性恢复。启动和来源协议见LAUNCH.json、PROTOCOL.json。15项针对性测试通过。

重启监督器前须确认旧监督器和训练子进程均退出；独占锁和精确进程路径守卫拒绝重复训练。命令：`/home/wenxin/miniconda3/envs/scenemi/bin/python -u /home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/native30_gait_v2_ego55k_oct15/continuous_training_oct16/supervise.py`。
