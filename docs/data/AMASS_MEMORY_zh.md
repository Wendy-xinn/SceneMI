# AMASS 长序列的低内存处理

2026-10-22 全量准备在提交 13,381 / 14,096 个源文件后停滞，下一条为 `KIT/969/Trial_02_poses.npz`。进程超过两小时没有新提交，内存约 15.3 GiB，超过该任务 12.8 GiB 的 MemoryHigh；内存压力约 82%。已中断该自有进程，保留逐序列已提交结果。没有修改其他任务或共享系统设置。

上游 0a 用 sklearn DBSCAN 对静止脚点高度进行一维聚类。该实现展开全部邻接关系；这条长序列有 34,416 个静止脚点，高度密集时出现很大的内存需求。新的 `HeightDBSCAN` 对高度排序、识别核心点连通分量，并按原始输入中最早核心点的顺序分配标签，边界点也遵循原版首次访问的规则。空间复杂度为 O(N)，不生成完整邻接图。

## 等价性证据

- `tests/test_amass_floor.py`：随机与重复高度、恰好位于 epsilon 边界的浮点数、多个核心分量共用边界点及不同输入顺序；逐项比较原版标签与核心点索引。另检查十万相同高度点。
- 从已完成的 19 个数据来源分别选最短有效源文件，原始及镜像共 38 份；重新执行原版和低内存版 0a，所有 NPZ 字段与彼此及已有提交结果逐位一致。
- 对原来卡住的长序列，原始及镜像均在 6 GiB / 4 CPU 限制内完成；全部 34,416 个高度点的核心点分类与独立 KD-tree 计数一致，每种反射另做 10 组真实高度子样本的原版标签比较。没有声称在原版完整邻接图上重跑并完成了该长序列。

核验命令可重复执行；输出为本地 `runs/amass128/preparation/memory_equivalence/RESULT.json`，含实现与验证器哈希：

本次已通过的精简证据见 [高度聚类核验](../validation/amass_height_clustering.json)。

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 6 --cpu-count 4 -- \
  /home/wenxin/projects/.venvs/amass-preprocess/bin/python -m scenemi \
  amass verify-height-clustering --paths configs/paths.local.yaml
```

## 恢复与封存

`amass_prepare.py`、数据划分、相机约定与上游代码保持原文件哈希。显式执行适配器只替换上游模块内的高度聚类调用；不改变数据集输出协议、接触阈值或标签定义。

恢复前完整验证已有清单和所有已提交输出的 SHA256，备份原 `JOB.json` 为 `JOB.original.json`。`BOUNDED_ADOPTION.json` 固定原任务、已有清单前缀的字节数和哈希、已核验源文件/变体数以及等价性证据哈希。新的任务封存增加执行适配器与 CLI 源文件哈希；完成索引也检查这些额外依赖。这样可以区分执行实现，并可证明原有结果没有被重新解释或覆盖。

```bash
python -m experiments.offline_camera_retrain_v1.ops.resources run \
  --device cpu --memory-gib 6 --cpu-count 4 -- \
  /home/wenxin/projects/.venvs/amass-preprocess/bin/python -u -m scenemi \
  amass prepare --paths configs/paths.local.yaml \
  --inventory runs/amass128/raw_inventory.jsonl --bounded-height-clustering
```

准备进度仍以 `runs/amass128/preparation/STATUS.json` 为准。准备结束后须通过完整索引、CPU 真实批次检查和 GPU 资源准入及批次检查；这次恢复不自动启动 GPU 训练。

2026-10-22 14:20（北京时间）已确认越过原卡点并提交至 13,607 / 14,096，零处理失败。恢复后任务观测峰值约 0.94 GiB，MemoryHigh 触发次数为零；原有清单前缀哈希与备份任务封存一致。该数字是恢复时点的记录，并非全量已完成，证据见 [恢复核验](../validation/amass_memory_recovery.json)。
