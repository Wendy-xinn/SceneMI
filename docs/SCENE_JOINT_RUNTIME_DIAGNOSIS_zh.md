# 场景重训的原生崩溃与执行恢复

2026-10-30补充：用户确认其他场景也有死机和蓝屏后，扩大只读事件查询，已经找到
真实Windows BugCheck记录。主机确实发生过系统级崩溃，不能再仅按NumPy读取问题
解释历次中断。具体故障部件以及单独那次Linux SIGSEGV是否与其同源，仍未确定。

本轮观察到修正版进程在已完成第39步后，于第40步数据读取过程中收到 SIGSEGV。
`pipeline.log` 的 Python 栈停在 NumPy `open_memmap` 读取 NPY 文件头、
`_filter_header` / `tokenize`，上层是 EXO `exo_frame_points_world` 数组读取。
内核记录时间为北京时间2026-10-30 02:09:15。最近已保存的是第25步，
完整来源、配置、优化器和随机状态已保存在 `performance/recovery_source`。

## 已确认与尚未确认

- SIGSEGV 是原生进程异常，不能等同于“数据文件坏了”。这次不是 Python
  `ValueError`、SHA不匹配或 CUDA allocator OOM 报错。
- 解释器、tokenize.py、ast.py 和私有 NumPy 核心二进制各进行三次独立缓冲读取，
  校验和一致；这不构成完整的内存或硬件测试。
- 只读查询近三天 Windows WHEA、disk、stornvme、MemoryDiagnostics 事件，
  均返回明确的 NoMatchingEventsFound；没有把查询失败误写成硬件正常。
  特别是没有内存诊断事件，并不表示运行过且通过了内存诊断。
- 后续System BugCheck和Application WER事件确认了Windows蓝屏及物理内存内容
  损坏记录。没有WHEA事件不能排除这类问题；仍未确定是RAM、驱动/DMA、固件或其他
  部件引起。Linux调用栈只能定位那一次崩溃发生的读取路径，不能独立证明最终根因。

事件原文与重复读取记录分别保存在 `performance/WINDOWS_HARDWARE_EVENTS.json`
及 `performance/RUNTIME_FILE_READS.json`。不修改 Windows/WSL全局设置，不重启或
干扰另一用户的发行版；没有擅自执行占满全机内存的压力测试。

Windows报告两条32GiB内存，ConfiguredClockSpeed均为6000MT/s；9950X的
[AMD官方双条内存规格](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x.html)
为DDR5-5600。该配置值得做稳定性对照，但不能据此判定硬件故障或推断EXPO是否由
用户手动开启。三块SSD当前均报告Healthy/OK；详细StorageReliabilityCounter查询
返回CIM访问错误，不能把这一项写成SMART检测通过。完整内存稳定性测试尚未执行。

## Windows蓝屏的新证据

下列时间为北京时间的重启后事件登记时间，不能当成精确故障发生时刻。
System1001、Kernel-Power41与Application WER1001相互核对，保留原始XML。

| 事件日期/登记时间 | BugCheck | 转储文件 | 已确认的范围 |
|---|---|---|---|
| 10月30日00:58:50 | 0xA | `C:\Windows\Minidump\103026-12312-01.dmp` | Windows主机蓝屏并重启；与此前整批WSL任务消失时间吻合 |
| 10月28日02:07:43 | 0x12B | `C:\Windows\Minidump\102826-12265-01.dmp` | Windows检测到物理内存内容损坏，WER桶为`StCtDecompressFailed` |
| 10月26日10:49、25日14:57、20日10:57 | 0xBE | 相应日期的minidump均有记录 | 多次内核态只读内存写入异常，尚未确认责任部件 |

10月30日Event6008另报告此前非正常关机为00:55:50，故00:58:50是恢复启动后的登记时间。
这支持此前评估中断涉及Windows主机崩溃；02:09:15的独立Linux进程SIGSEGV发生在
主机恢复之后，不能仅凭先后关系认定两者根因相同。

`0x12B`的参数3、4均非零：`0x243bfa8da10`、`0xffffd682bf61f000`。
按[微软0x12B说明](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x12b---faulty-hardware-corrupted-page)，
这是Compressed Store Manager因物理内存内容损坏而解压失败的分支，不能套用参数3、4
均为零的单比特错误解释。可能原因包括RAM、驱动/设备错误DMA或MDL、设备或固件；
代码名称本身不能确诊某根内存条损坏。

WER桶分散在`nt!KiStartSystemThread`、`nt!MiAgePteWorker`、`dxgmms2!VIDMM...`、
`Vid!Vsmm...`及`exfat!FppDeleteFcb`。这些记录说明故障显现在不同内核路径，
不能把桶内的模块直接当成肇因，也不能据此断定NVIDIA驱动、exFAT或WSL本身有缺陷。
[0xA说明](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0xa--irql-not-less-or-equal)
和[0xBE说明](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0xbe--attempted-write-to-readonly-memory)
均需要结合转储调用栈继续定位。

证据文件均位于`runs/scene_joint128_conditioning_corrected/performance`：
`WINDOWS_BSOD_EVENTS.json`、`BSOD_SUMMARY.json`、`WINDOWS_BSOD_WER_METADATA.json`。
当前账号读取实际minidump和Report.wer均Access denied；机器未找到已安装的WinDbg/cdb。
因此没有完成转储分析，没有改变文件权限、安装调试器或把转储上传到外部。

### 下一步维护排查顺序

1. 先协调同机其他用户、保存各任务，在维护窗口由有Windows管理权限的环境读取
   10月28日和30日转储并做`!analyze -v`，核对调用栈和驱动信息。
2. 记录当前BIOS设置。若开启EXPO/XMP，先仅将内存恢复JEDEC默认设置，做同硬件的
   稳定性对照；6000MT/s高于该CPU官方双条5600规格，是排查线索，不是确诊。
   若还开启CPU降压/Curve Optimizer或超频，再单独恢复CPU默认设置，避免同时改动
   多项后无法判断原因。
3. 在维护窗口运行微软建议的Windows内存诊断并读取MemoryDiagnostics-Results。
   有错误则按内存条/插槽逐一定位；无错误仍不能单次排除所有间歇故障，结合默认设置下
   原工作负载复现情况决定下一步。若默认设置下仍崩溃，按转储证据检查相应驱动/固件。
4. 保留当前训练和每步恢复点；当前已经恢复运行不能证明主机稳定性通过。
   整机稳定后，应复核关键权重和验证结果的可重复性，再形成论文最终记录。

以上是待执行的维护方案，未触发重启、内存诊断排程、BIOS变更或全机压力测试。
遵守`AGENTS.md`关于与另一WSL用户协调维护窗口的限制。

## 保持数据不变的隔离措施

新入口的只读 NPY 映射后端保留现有内容寻址 Linux 缓存和 SHA256 校验，仅用
受限的标准 NPY 数值头解析替代 NumPy 的旧头部 tokenizer。检查 magic/version、
header长度、数值 dtype、shape/order 和准确 payload字节长度，不支持 pickle对象。
文件内容、相机标定、世界坐标、选样和 loss 都保持原值。

后端只在新训练进程的上下文中生效。正在运行的历史全量验证及其所有封存依赖
不热改。每次打开的最后一个数组路径落盘，便于再次异常时定位实际文件。
读取后仍由原 checked loader校验声明的完整 SHA，不绕过失败或跳过样本。

从第25步的数据 Generator状态重放26至40步，逐步对比已记录的样本身份哈希，
并通过原校验器读取实际数组。若原始数据本身存在确定性损坏，重放必须报错，
不能靠重采样掩盖。进度与结果见 `performance/INPUT_REPLAY.json`。

新 launcher还会观察子进程退出码并回写中断状态，避免原生崩溃后 STATUS仍显示
training。训练每步原子保存完整 checkpoint，先fsync已提交日志，再发布权重与
随机状态，降低下一次非正常退出的进度损失。

## 显存与预取的实测选择

用户允许训练与验证合计约16GiB allocator：训练12GiB、验证4GiB，另保留CUDA
开销和4GiB物理空闲。训练CPU仍限四个逻辑CPU，放到四个不同物理核上。

有效 batch继续是EGO64+EXO64。测试物理microbatch 8/16/32/64，不改变优化器、
学习率、数据或loss。随机timestep和noise仍按原8窗口的CUDA生成器调用形状产生，
保留相同样本和噪声流。数据预取使用单独的推测Generator，只有消费完一个模式的
样本后才提交该模式的随机状态；未来预取不会提前改变checkpoint的随机状态。

候选需保持相同样本身份、相同完整随机状态，并检查loss、梯度、optimizer更新与
参数误差。microbatch=8的预取路径必须与原执行逐位一致。较大microbatch的浮点
核差异按预声明阈值核验；不放宽阈值强行通过。选最快且通过的候选后丢弃所有
基准更新，验证保存/恢复，再按adapter/optimizer/RNG三项哈希一致的方式迁移
第25步checkpoint，继续原3000步预算。结果见 `performance/optimization`。

## 当前取得的验证证据

26至40步共1920个样本重放完成，26至39步的身份哈希与原记录全部一致，原40步
边界也通过了文件校验和读取。新执行从25步恢复后，26至39步的loss、梯度范数、
实际学习率与样本身份记录也逐项精确一致，随后实际GPU训练已越过40步且每步保存。
这排除了本轮已读输入的确定性损坏证据，仍不能证明已彻底排除原生库/WSL或硬件问题。

预取microbatch=8与原执行的参数、optimizer、梯度及全部RNG逐位一致。16/32/64
的样本和RNG也相同，但梯度相对L2差异分别约0.000196、0.001902、0.001903，
未通过事先设定的1e-4阈值，因此没有采用。它们并非被当作模型质量失败，而是本轮
优先保留完全相同续训轨迹的执行方案。显存预算允许16GiB，不意味着要人为占满显存。

恢复初期稳定步耗时约24至30秒；不同窗口仍有较大波动。最新连续耗时及当前保存步数
见 `EXECUTION_RECOVERY.json` 和 `training/STATUS.json`，不把单步基准当长期吞吐保证。
