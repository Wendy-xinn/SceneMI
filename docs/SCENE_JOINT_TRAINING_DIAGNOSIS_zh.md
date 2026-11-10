# 当前 EGO / EXO 场景混训诊断

2026-10-29 北京时间检查。当前任务为 `configs/scene_joint128_conditioning.yaml`，
入口 `scenemi.scene_joint_replay`。父权重是四数据集无场景 replay 的 step 2000
（SHA256 `a46122d40c572be8ac82b55ca7d421e7c64a4de690c7ae369091064ba1b9260c`）。
当前场景阶段只采样 EgoBody、TRUMANS、GRAIL；每次优化累计 EGO 64、EXO 64，
两条路径都使用场景关系，共享人体先验冻结，总预算 3000 次更新。

## 当前能否训练

本次宿主只读检查确认进程 110927 在受限 cgroup 内运行，活动日志从 step 558
持续推进至 573，未再次启动训练。562 → 573 的单调时钟观测耗时 350.36 秒，
平均约 31.85 秒/步，按此短时速度估算剩余约 21.5 小时。该估计包含观测边界误差，
不应作为稳定 ETA 或完成承诺。近期 50 步 EGO loss 0.125–0.194、EXO loss
0.099–0.190、梯度范数 0.052–0.257，均有限；这不能代替验证集评估。

当前 `STATUS.json` 每 20 步更新，`updates.jsonl` 每步写入，二者存在正常滞后。
checkpoint 独立 CPU 加载通过，确为 step 500、32,377,093 字节：
`63fd75ae6e02aab19ba2298b315dc60faec5414b197cc408e3a209124e901fa7`。
checkpoint SHA 是本次读取时的身份，后续正常保存会改变它。

## 为什么一直出问题

| 问题 | 证据与影响 | 当前结论 |
| --- | --- | --- |
| 早期启动实现错误 | 本地会话记录了遗留函数名、loss 配置层级、EXO 场景坐标元数据、共享封装 argparse 对象误用，以及 gap 外推访问第 128 帧 | 后续训练已越过这些路径；它们是不同故障，不能统称为显存不足 |
| 原生进程崩溃 | 内核日志 10 月 29 日 12:39:30 记录 Python 3.9 SIGSEGV，WSL crash capture 对应驱动 PID 8272；历史日志到 627 | 可确认发生段错误；没有足够堆栈证明由哪个扩展、读取操作或内核问题造成 |
| 缓存身份校验失败 | `resume_stdout.log` 在读取 `motion_world` 时报告 `Mapped array identity changed`；相关镜像以 `901116b…` 命名 | 当前源码按 SHA256 验证，stat 字段只是验证缓存的键；不能把该异常解释为 inode 或 mtime 改变。本次重新读取 1,939,376 字节 SHA 与原合同完全相同，证明当前副本匹配，不能证明故障时的字节也正确 |
| checkpoint 间隔太大 | 只在每 500 步和最终步保存；主日志及两份 crash tail 记录了从 500 之后重跑的历史 | 795 和 627 都不是可恢复进度。每次崩溃回到 500 会丢掉几小时工作 |
| 状态和测速误导 | trainer 用绝对 `step / 本次恢复后的耗时` 计算速度，未减去恢复的 500 步；旧 ops status 只读取历史 `continuous_training_oct16/STATUS.json` | 旧速度被明显高估，先前“剩余 1.5–2 小时”不成立；本次增加独立实测入口 |

第一个 step 795 附近中断缺少明确异常栈，不能把后来缓存失败的原因直接套到它上面。
当前缓存校验已经使用已有多次 mmap / buffered 验证逻辑，仍严格要求原 SHA，
没有跳过校验或更改数据合同；该修复不能宣称已经解决所有段错误。

## 资源与性能

本次 cgroup 采样：RAM 硬上限 12 GiB，软阈值约 9.6 GiB，使用量约 9.6 GiB，
其中匿名内存约 4.1 GiB、文件缓存约 5.4 GiB；swap 1 GiB 已用满，
`memory.events high=57160`、`oom=0`、`oom_kill=0`。这说明触发过大量软阈值回收，
本次没有 cgroup OOM 证据。采样时 memory / I/O pressure 很低，不能据累计次数认定
内存回收就是当前慢速的主要原因。GPU 采样空闲约 16.5 GiB，也不能据一次 GPU
利用率为 0% 判断任务已死亡。

源码每步进行 128 个窗口的数据准备、历史点构建和逐窗口/逐帧 KD-tree 查询，
CPU 场景处理和数据访问是需要进一步分段计时的候选瓶颈，目前没有精确 profile。
不应依据显存尚有空闲直接把 batch 继续调大，也不应未经测量将所有耗时归因于 GPU。

## 另外核实的恢复与配置缺口

这些缺口尚未改入当前执行中的 trainer；保持当前进度，不热改运行依赖。

- checkpoint 只有 adapter、optimizer、数据采样 Generator 状态、step 和 parent SHA。
  没有 Torch CPU/CUDA、全局 NumPy、Python 随机状态。续训会恢复模型与优化器，
  但不能重现相同扩散 timestep / noise；之前称“精确续训”不准确。
- `optimizer_for` 把所有分支放在一个参数组，`update` 只使用 `lr_ego`。
  配置中的 `lr_exo=3e-5` 没有生效；EXO 实际也使用以 `1e-4` 为峰值的同一调度。
  这属于实验配置偏差，尚无证据说明它造成进程崩溃。
- trainer 没有调用 allocator cap 安装函数，配置 `allocator_gib=4` 本身没有形成
  PyTorch 进程内上限；RAM cgroup 的 12 GiB 限制则确实生效。
- 缺少本轮运行锁、完整源码/配置 preflight seal、异常状态写回、退出时安全保存和
  自动隔离未提交日志尾部。遇到 SIGSEGV 无法依赖 Python finally 保存，恢复能力
  首先取决于更频繁的原子 checkpoint。现有协议每次恢复都需人工处理日志尾部。

改变上述训练语义需要在安全边界后用独立恢复入口或新版本执行，并重新做资源准入、
CPU/CUDA 验收和来源封存。若改变 EXO 学习率，须明确记录阶段边界，不能把此前
500 步已经使用的实际学习率改写成期望配置。旧 checkpoint 缺失的随机状态无法补回。

## 本次落地的监控修复

复用既有 ops CLI，仅修改独立监控入口并新增 `scenemi/monitoring.py`，
当前 trainer、模型、数据 loader、权重和资源限制均未修改。

```bash
python -m scenemi ops status --run runs/scene_joint128_conditioning
python -m scenemi ops status --run runs/scene_joint128_conditioning --sample-seconds 60
```

`training` 保留 trainer 原始报告，`observed_progress` 显示已落盘活动日志，
`measured_progress` 只根据观测期间新产生的更新和单调时钟计算速度。
日志回退、重复或重写时不报 ETA；短时间没新日志只表示没有观测到更新，不宣称进程死亡。
安全 checkpoint 仍须与日志进度分开核验。该入口不导入 Torch，不读取训练数组或 test，
不启动 GPU 计算，也不覆写 trainer 的状态。

验证：`python -m unittest tests.test_training_monitoring -v` 五项通过，覆盖恢复步数
不计入测速、日志重写、重复步数、半行日志以及新旧状态路径。CLI 实际读取也通过。
