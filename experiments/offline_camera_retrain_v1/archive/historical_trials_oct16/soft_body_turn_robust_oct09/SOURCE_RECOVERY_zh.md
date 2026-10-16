# native数据依赖恢复与边界

原兄弟仓库`diffusion-motion-inbetweening/experiments`被删除；SceneMI主线仍依赖其中的源清单、窗口审计与运行缓存。此前第一组训练又在约880步中断，没有checkpoint；没有生成两组短训结果。

加载器和辅助代码从Git备份恢复；index_sources.py及audit_intervals.py从本机会话的原补丁恢复并重新执行。原始TRUMANS、EgoBody、RICH数据、SceneMI 20Hz姿态及动态场景、人体模型和正式55k权重仍存在。

## 恢复方法

recover_native_source_archive.py为native_v1恢复源索引与必要运行数组：

- TRUMANS pelvis从原始human_joints按原30→20Hz查询线性采样，用原辅助函数；其原生姿态继续读取已保留的20Hz导出。
- EgoBody原生动作继续读取已保留的SMPL姿态、平移和形状；归档关节由原生FK重建。
- 输入相机直接引用完整、已审计的20Hz场景缓存；没有降频后插值，也没有改动场景数组。
- 用pose清单保留原序列排序、角色和分段；重新执行原窗口资格审计。RICH仍为原官方train/val。
- 旧归档的静态并集引用已保留的可见图，仅供原适配器中间计算；native完整因果场景会覆盖模型最终occupancy/BPS。不得用于恢复旧版实验输入。

这没有恢复旧manifest的原字节，也没有恢复历史SMPL/旧相机档案或test运行缓存，因此不会伪装成原数据指纹。旧版/不完整场景消费者被明确拒绝。

## 独立核对

audit_recovered_native_source.py重新读取192个固定窗口，包括64/128/192帧、四个数据/角色组、48条开发验证人物序列；两组噪声DDIM20共384次预测。GT动作、rest、camera及此前v3生成值与删除前motions缓存的最大差异均为0。报告见[source_recovery_audit.json](source_recovery_audit.json)。这证明这些窗口的实际模型输入/目标回放一致，不是整个原归档逐字节恢复的证明。

source-recovery-audit是明确的数据恢复迁移入口：要求相同旧checkpoint源指纹、192窗/384次完整回放、新运行档案审计、恢复管线指纹与严格native完整场景协议。错源、旧协议、缺窗、改变张量或改变管线会被拒绝。保存当前新指纹，resume仍要求精确相同的新数据与目标指纹。

两组短训继续从原55k加载权重，后续会再次核对1000步sample_trace与旧v3一致。训练只加入软朝向损失；没有加入输入硬约束或GT接触条件。

## 防止再次丢失

运行依赖位于`diffusion-motion-inbetweening/experiments/offline_sequence_v1/data`，不能当作停用基线目录删除。源生成器已加入GitHub代码备份；运行数组属于数据资产，未上传。恢复脚本使用保留的SceneMI姿态和场景，请同时保留其data目录与原始数据。训练改为每250步持久保存，并在再次启动时自动从有效checkpoint恢复、截断checkpoint后的重复日志。
