已确认本次中断的直接原因是 WSL Linux 内核 panic。2026-10-21 08:41:55 的 Windows 侧转储，与停止的虚拟机 GUID 以及最后一份评估结果时间精确吻合。原文末尾为 `Kernel panic - not syncing: Fatal exception`。

调用栈为 `__list_del_entry_valid_or_report ← ext4_es_scan ← shrink_slab ← try_to_free_pages ← __alloc_pages_slowpath`：致命异常出现在 ext4 缓存的内存回收路径，随后整个 WSL 虚拟机退出。该证据尚不足以区分具体内核缺陷或底层内存稳定性问题，也不能据此认定 RAM 耗尽或硬盘损坏。

10月17日22:05和20日14:44也有 panic 转储，但调用栈不同，不合并认定为同一根因。之前的 WSL_INTERRUPTION_AUDIT.json 只记录 Linux journal 未发现 panic 的当时结论；Windows 新转储补证已确认实际发生了内核崩溃。

原始转储见 [diagnostics/wsl_kernel_panic_oct21.txt](diagnostics/wsl_kernel_panic_oct21.txt)，精确文件 SHA、时间、GUID 和事件关联见 [WSL_CAUSE_WINDOWS_AUDIT.json](WSL_CAUSE_WINDOWS_AUDIT.json)。本轮只读排查，没有改系统配置、更新或重启 WSL。
