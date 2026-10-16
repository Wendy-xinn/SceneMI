# 实验根目录脚本清理审查

本次只读审计，未移动、删除或修改根目录任何文件。

根目录目前 **352份Python源码全部被当前训练的effective_config封存**，加model/外部依赖共359项，再加持续训练执行扩展6项。检查SHA仍匹配。根.py中没有逐值相同的重复文件，没有不受当前seal约束的.py，因此本轮可永久删除/移动的根Python文件清单为 **空**。

问题来自 `prepare_native30.code_hashes()` 用 `HERE.glob('*.py')` 直接封存整个实验根目录，包含大量历史launch/audit/test。这是工程边界太宽，并非352个算法都用于训练。当前主训、完整RICH验证、验证核对和scene因果队列尚活动，不能为了目录整洁修改源码、有效配置或来源守卫。把旧文件移走再加symlink也会被 `p.resolve()`改变封存字典键，不能透明绕过。

## 本轮可真实收拾的非Python文件

`ROOT_NONPY_CLEANUP_PLAN.json`给每个根非.py文件的SHA、大小、操作和理由：

- 14份已结束导出/测试console日志可删除，共839,024字节。没有当前openFD/参数/数据或源码seal引用；源manifest和小审计保留。
- 24份历史JSON/config/文档与旧 `.sh` 可原样归档到 `archive/retired_runs_oct16/root_records/`。旧README/EXPERIMENTS_INDEX先保留历史原文，再由root重写唯一当前README。
- `resume_native30_checks.sh` 是旧stride64恢复/8784查看器包装；其准备目录已经退役。当前续训使用独立supervisor和io_recovery，不用此shell脚本。
- 历史config是旧CLI默认入口；当前主线明确传 `runs/native30_gait_v2_ego55k_oct15/config.json`，监控/验证同样读run配置。归档后旧CLI默认复跑需按映射恢复原路径，不改历史实现。
- RICH_PSEUDO_EGO与SCENE_VISIBILITY数据说明可保留；root README应指向当前数据/资产职责。

未加入files.exclude，也不把隐藏当作实际清理。全部删除/移动由root按清单执行，执行后需记录MOVED/删除结果与重新核对365项seal。

## 后续瘦身应如何做

`ROOT_SCRIPT_AUDIT.json`用AST提供当前导入边界：

|范围|根目录Python数|
|---|---:|
|训练入口的模块级直接/递归导入|42|
|训练全函数/条件导入的保守闭包|61|
|合并当前viewer/监控/完整RICH/scene/guardian保守闭包|63|

这不是声称已通过运行证明的绝对最小依赖。训练foot_metrics/io_recovery动态装载已显式加入；活跃入口的同目录Python也保守保留，避免漏掉report等字符串动态加载。原Python路径和每文件SHA全在机器记录里。

`ROOT_DEFERRED_ARCHIVE_PLAN.json`首先列出106份有明确退役run字符串引用、且不在活动静态闭包中的旧launcher/evaluate/audit/test，建议未来归档独有历史实现而非永久删除。另111份入口/测试仅因未被运行导入，不足以证明无用，继续保留待审查；72份算法/其他模块也不因无导入就删除。test属于持续回归资产，不能当作无调用的运行代码全部清掉。

**拒绝在活跃训练/评估期间执行根.py归档。** 后续步骤：

1. 等主线达到登记终点并保存完整checkpoint；完整RICH、CPU核对、前后子集比较、scene因果队列全部完成，相关读代码进程退出。
2. 保存原359+6完整源码bundle、配置、checkpoint/Adam/RNG及逐文件原路径/SHA恢复映射。旧checkpoint的strict resume仍依赖原路径，重跑旧协议前必须恢复，不可声称随意删除旧.py后仍能直接原样resume。
3. 对下一次明确新版本的准备流程，将 `glob` 更改为显式模块白名单与导入闭包，补充模型/损失/场景开关/数据模式/可视化的动态加载路径。用真实干净重启、训练前后向和固定验证覆盖必要分支，再封存完整实际依赖。
4. 该改动是未来准备和恢复工具的版本变更，不回写当前实验来源、指标或协议。新训练可继续同模型/Adam/RNG，但若执行包装/来源机制发生变化，要登记明确的新阶段/迁移审计，不冒称当前源码完全未变的strict resume。
5. 最后按已审阅清单移动独有历史入口并保留紧凑结论；逐值重复文件或确定可重建输出方可永久删除。今后运行产物放runs、正式工具复用ops统一入口，避免再把一次性脚本堆进根目录。
