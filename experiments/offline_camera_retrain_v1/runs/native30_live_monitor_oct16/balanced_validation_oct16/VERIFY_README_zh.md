# 独立 CPU 验收

`verify_results.py` 只读 runner、训练源码/配置、冻结 checkpoint、计划、已提交 rows 和 motion 文件。它只在本目录写 `VERIFY*.json`，不调用模型、不初始化 CUDA、不创建/修复数据缓存、不更新优化器。

运行环境为 `/home/wenxin/miniconda3/envs/scenemi/bin/python`，CPU/BLAS 均为单线程，建议用 `nice -n 15`。

```bash
/home/wenxin/miniconda3/envs/scenemi/bin/python verify_results.py
/home/wenxin/miniconda3/envs/scenemi/bin/python verify_results.py --require-complete
```

普通模式检查读取时已完整提交的行，尾部尚未写完的一行只报告、不修改。所有已提交 artifact 都检查 SHA、身份、冻结步数、坐标约定、shape 和有限值；partial 模式另外在 CPU 重算最先提交的三个预测的 raw/extra/activity 全部指标。完成模式要求 30 个独立 recording × clean/noisy 共 60 条覆盖，重算全部 60 条指标。数值比较容差为相对 `1e-6`、绝对 `3e-5`；计数、None 和布尔值要求相等。

另核对 359 个原训练源码 + 6 个连续训练文件、原配置/effective config、plan/runner/checkpoint 哈希、官方 VAL 元数据/合法 median start/identity seed、三数据集各 10 个独立 recording、EgoBody camera wearer/interactee 各 5 个，以及主线 worker 存活、目标仍为 55k、步数相对评估开始已推进。大体积原始数组不重新哈希或重建；保存的 GT 数组作为重算指标的参考。

`VERIFY_PARTIAL.json` 保存局部验收；`VERIFY_COMPLETE.json` 只在完整验收通过时产生；`VERIFY.json` 是最新一次通过的结果，必须读取其 `status` 与检查时间。异常另外写 `VERIFY_FAILURE.json`，不覆盖已通过的历史记录。

后台等待记录在 `VERIFY_WATCH.json`，输出在 `verify_results.log`。等待进程每 15 秒只用标准库读取状态，待 STATUS/PROTOCOL 都 completed 后，以相同 PID exec 全量验收。它在等待阶段不导入 torch/numpy。

来源证据边界：冻结时记录了完整 checkpoint SHA，runner 使用原子发布文件的 hard link 固定 inode，再保留 model/config/step。原完整 pinned inode 已删除，且未事先单独记录其 model 张量 digest；因此事后只能独立验证 compact checkpoint、模型张量 digest、step/config 与记录的冻结流程，无法再直接逐张量对比原完整源。变化中的主线 `last.pt` 不能替代原冻结源。
