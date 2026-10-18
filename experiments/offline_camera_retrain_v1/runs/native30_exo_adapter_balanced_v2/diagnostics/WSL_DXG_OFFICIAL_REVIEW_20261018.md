# WSL dxg告警：官方证据复核

核对时间：2026-10-18（Asia/Shanghai）。只读调查，未更改系统、驱动、内核、训练配置或运行状态。

**结论：当前版本与公开报告吻合；微软已修复这一FORTIFY告警的数组声明，但没有找到“这个补丁已解决CUDA挂死或WSL异常重启”的官方确认。当前训练已自行恢复，先完成既定实验，不实施系统更新或内核替换。**

本机 `uname -r` 为 `6.18.33.2-microsoft-standard-WSL2`。只读解析 `C:/Program Files/WSL/wsl.exe` 与 `wslservice.exe` 的 `VS_FIXEDFILEINFO`，两者文件版本均为 **2.7.10.0**；`product_version` 为5.0.1.1，不能当作WSL发行版本。由于WSLInterop缺失，不能运行 `wsl --version`，故以上是安装文件版本证据，不是该命令输出。

| 来源 | 已确认内容 | 适用边界 |
|---|---|---|
| [WSL #40580，Feng Wang评论](https://github.com/microsoft/WSL/issues/40580#issuecomment-4484490146) | 维护参与者认为该调用点可能是FORTIFY误报。 | 评论有保留语气；并未确认训练挂死原因。 |
| [#40580，assignee Mitchell Levy评论](https://github.com/microsoft/WSL/issues/40580#issuecomment-4783531578) | 维护者承诺后续内核修复此issue；该issue后来由其关闭。 | 这是该告警问题的修复承诺。 |
| [微软内核正式提交1dda0bd](https://github.com/microsoft/WSL2-Linux-Kernel/commit/1dda0bd6b031fec40afb3c6c9d59b8b89fd9e8db) | 将 `fence_values[1]` 改成柔性数组 `fence_values[]`，并同步命令缓冲区长度计算；提交说明明确针对FORTIFY告警。 | 提交没有宣称修复CUDA训练停顿、9P阻塞或VM重启。 |
| [WSL #41017](https://github.com/microsoft/WSL/issues/41017) | 提报者使用同样WSL2.7.10.0和内核6.18.33.2，报告同调用点告警及CUDA挂死；后续用户还报告进程崩溃/VM重启。 | 这些是用户报告；该issue被policy bot关闭，不能把“closed”解读为维护者确认根因或修复已验证。 |

另直接核对微软发行标签源码：[6.18.33.2](https://raw.githubusercontent.com/microsoft/WSL2-Linux-Kernel/linux-msft-wsl-6.18.33.2/drivers/hv/dxgkrnl/dxgvmbus.h)仍有`[1]`，[6.18.40.1](https://raw.githubusercontent.com/microsoft/WSL2-Linux-Kernel/linux-msft-wsl-6.18.40.1/drivers/hv/dxgkrnl/dxgvmbus.h)已为`[]`。因此后续维护有正式补丁可核对，**只能保证针对该告警的源码变化，不能保证解决本次全部异常**。

现场14:07的同类dxg告警与随后 `p9_client_rpc` D态等待是两个不同的观察点，没有足够证据连成因果链。运行负责人已观察到进程未经信号干预在14:13恢复并继续训练；不应只因出现该告警就自动杀进程、反复重启或改模型。当前保留checkpoint及故障时间线，继续既定实验；确需系统维护时，再另行核对正式内核/驱动和Windows事件，并以相同工作负载验证。
