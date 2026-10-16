验证截图保存在 `visuals/step_XXXXXXXX/`。当前第一组可归属的完整验证是 **7000 步**，包含 TRUMANS、EgoBody、RICH 三个既定固定窗口，每个窗口 clean/noisy 各一张，共六张 PNG。

每张图的三行是同一窗口第 48、96、144 帧，三列分别是：

1. 蓝色 GT 原生人体 mesh + 灰色 GT 静态扫描；扫描仅用于显示。
2. 洋红色预测原生人体 mesh + 青色实际输入的观测静态记忆，以及橙色当前帧可见动态表面。
3. 同一相机、同一时刻的叠加图。

在 VS Code 文件树打开 `visuals/step_00007000/`，直接点击 PNG 即可；`visuals/INDEX.html` 是可在浏览器打开的本地图廊。TensorBoard 的 Images 页也会按 `visual/fixed/{dataset}/{clean|noisy}` 显示截图。

相机只由固定窗口 GT 关节范围确定，同一窗口跨训练步数保持一致，没有 GT 对齐、地面吸附或预测后处理。人体使用对应数据源真实 SMPL/SMPL-X 模型、性别、betas、缩放解码，neutral hands/face 是 22 关节表征的显示约定，不能将手脸细节视为完整原始 GT。每次记录 mesh 解码关节与训练 FK 的最大差值，超过 `1e-4 m` 不出图。

“GT 场景”准确指原始**静态扫描的局部裁剪**，不是含完整动态物体和所有人的完整时变 GT 世界。TRUMANS、EgoBody、RICH 使用与 `native30_cache_mesh_review.py` 相同的扫描路径及坐标变换。GT 动态物体和其他人的完整 GT mesh 暂不显示；输入动态表面仍单独显示。输入静态记忆按当前训练协议使用 supplied-recording observed-static union，不把未观测区域当作自由空间。GT 扫描从未交给模型。

`visual_capture.py` 作为 CPU 独立进程运行，主训练源码、配置、模型、Adam、随机数状态均不修改，也不调用训练模型或 CUDA。它读取训练原子发布的 `validation_latest.npz`，检查其中明确的 `step` 和 `identities_json`，与 `validation.jsonl` 的 step、序列、起点、source clock 和 clean/noisy 固定窗口一一核对后再生成图。npz 没有权威 step 时等待，不用 mtime 猜步数。记录中的源 NPZ SHA256 可审计，但不会额外保存大 NPZ。

已有 PNG+MANIFEST 的同一 source SHA256 不会重复渲染；进程重启仍适用。每 30 秒检查一次，所以后续每 1000 步验证发布后自动捕获。历史已被覆盖、没有对应预测数组的验证无法忠实补截图，因此不会用当前模型冒充 1000–6000 步结果。每次保留六张压缩 PNG 和小 JSON；不保存每步 mesh 数组或大验证副本。源数据、训练缓存只读复用。

后台启动信息和日志：`visuals/LAUNCH.json`、`visuals/watch.log`。PID 在 LAUNCH 中；`visuals/LATEST.json` 指向最近完整截图，只有全部六张输出完成才原子发布 MANIFEST。`visuals/LAST_ERROR.json` 若存在表示最近一次捕获异常，原训练不受影响。

手动一次性捕获：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
 /home/wenxin/miniconda3/envs/scenemi/bin/python -u \
 experiments/offline_camera_retrain_v1/runs/native30_live_monitor_oct16/visual_capture.py --once
```

后台等价命令是以上 `--once` 改为 `--watch --poll-seconds 30`，使用 `nohup` 和独立 session，stdin=/dev/null、stdout/stderr=`visuals/watch.log`。脚本自行设 `nice +10` 及 Torch CPU 线程数 1。

测试：

```bash
/home/wenxin/miniconda3/envs/scenemi/bin/python -m unittest discover \
 -s experiments/offline_camera_retrain_v1/runs/native30_live_monitor_oct16 \
 -p test_visual_capture.py -v
```

10 项测试覆盖：权威步数和源 SHA、较新日志不重标旧数组、缺 step 等待、步数不匹配、身份不匹配、clean/noisy 配对缺失、native body 缺失拒绝伪装骨架、半行 JSON、scene-OFF 不混入及正交相机。

已启动后台 PID **114842**（2026-10-16T13:40:12.375972+08:00）；实际启动命令及环境完整保存在 `visuals/LAUNCH.json`。`visuals/STATUS.json` 每轮更新心跳和最近截图步数。

8000 步场景核验后，未来 EgoBody 橙色图例明确为 `Other person's visible surface (partial)`；图库正文说明该颜色代表另一人的局部可见片段。原 7000/8000 共12张PNG均保持修改前mtime，重复捕获后SHA256与mtime保持不变，详见`visuals/DISPLAY_CLARIFICATION_8000.json`。当前截图守护PID **122610**。

8000步原始另一人GT几何核验：`visuals/step_00008000/egobody_owner101_audit/AUDIT.json`、俯视/正侧视`TOP_FRONT_SIDE.png`；可旋转网页 http://localhost:8792/ROTATE_3D.html，HTTP PID **122630**，仅绑定127.0.0.1且仅服务audit目录，实取验证HTTP 200。服务记录为该目录`HTTP_SERVER.json`。
