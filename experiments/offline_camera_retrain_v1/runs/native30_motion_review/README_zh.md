# 统一动作查看器

使用同一个 `ops.motion_viewer` 服务，地址 http://localhost:8802/ 。`CASES.json` 是选例清单，不再为每次查看新增 Python 脚本。

案例名称只出现一次，下面分 EGO / EXO 图层。同一个窗口的适配前后共用时间轴、GT、观测和身体参数，准备时逐元素检查；没有保存另一模式结果时，该板块明确禁用，不以不同窗口冒充同一案例。

建议先看默认的 EgoBody interactee 行走（录制 `recording_20211002_S03_S18_04`）：6.4秒内骨盆水平路径约3.04米，clean World MPJPE约84.54毫米。其他选例、数据限制和原生成物理引导设置见 `SELECTION_zh.md`。

EXO 有两种快捷对照：

- “原41k / EXO适配”：橙色为原41k接收相同EXO观测的输出，紫色为共享41k加已有EXO适配后的输出；蓝色GT可独立开关。
- “输入参考 / 扩散结果”：绿色参考人体由实际输入关节点构造，紫色为扩散输出。参考人体优先读取原归档的 `reference_motion`，缺失时才复现原初始化；不使用GT姿态生成参考。

实际观测并不是一个真实tracker输出的完整mesh。置信骨架把未提供关节标灰、可用且置信度低于1标橙、置信度1标绿；这些颜色不改变原评估掩码。红点使用原始first-hit射线计数证明身体部位存在遮挡，部分遮挡仍可能有非零输入；噪声丢点和无投影支持不等同于遮挡。

完整静态扫描是GT参考，不喂给模型；静态输入是整段录制已见表面的并集，包含后续帧观测；当前帧场景只使用原导出的该帧512个环境first-hit采样。EGO和EXO各自显示自己的输入相机和场景，所有图层均能单独开关。

误差由已保存的关节计算，单位毫米，不追加对齐、贴地或物理修正。本次只解码CPU显示mesh，没有新训练或扩散推理。示例有意挑选，不能代替测试集总体结果；RICH退化对照保留。

复用命令（项目根目录，scenemi环境）：

```bash
python -m experiments.offline_camera_retrain_v1.ops.motion_viewer \
  --manifest experiments/offline_camera_retrain_v1/runs/native30_motion_review/CASES.json \
  --cache-root experiments/offline_camera_retrain_v1/runs/native30_motion_review/cache \
  --port 8802
```
