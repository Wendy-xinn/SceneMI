# 同权重支撑修正可视化

访问：http://localhost:8801/ 。原8800服务保持不变。

TRUMANS和RICH各有4组：原始贴头、去整身平移保留IK、原始贴头+弱重力、IK+弱重力。全部由同一个15100步checkpoint、同输入和随机种子的DDIM20采样产生；源选择清单中的16k只用于定位窗口。本查看器不会进行显示贴地、刚体配准或额外动作修正。蓝色GT，紫色预测，青色实际已见场景；可切clean/noisy、播放、帧滑块、场景开关。

真实Chromium浏览器检查通过：8组切换，帧119定位，播放从119推进至150，clean/noisy切换，完整静态扫描开关。全部16个源NPZ的truth/rest/observation/observation_meta在同窗口同regime四组间逐元素一致；native mesh与FK最大误差小于5e-7m，场景无加载错误。详见ACCEPTANCE.json。

截图均取clean第119帧；TRUMANS_PAIRED_119.png与RICH_PAIRED_119.png使用相同范围裁剪原始Viser截图。TRUMANS的去整身平移+IK未解决悬脚；弱重力有少量姿态变化，悬脚仍在。RICH同帧可观察到去平移组的头部偏离。单帧不能代表全段，更不能证明滑步修复；最后决策应依据40个真实采样结果及非直立反例。

重启命令（独立cache，不使用GPU）：

```bash
CUDA_VISIBLE_DEVICES='' python -m experiments.offline_camera_retrain_v1.ops view --manifest experiments/offline_camera_retrain_v1/runs/native30_live_monitor_oct16/support_ablation/visuals/PREPARED_CASES.json --cache-root experiments/offline_camera_retrain_v1/runs/native30_live_monitor_oct16/support_ablation/visuals/cache --port 8801
```

使用原有ops.motion_viewer，没有新增或修改仓库Python脚本。临时CPU准备和浏览器驱动仅保留在/tmp。约588MiB的播放缓存仅留本地，运行中的查看器依赖该缓存；代码和小型清单/验收/对照图可以备份，不备份生成数组。
