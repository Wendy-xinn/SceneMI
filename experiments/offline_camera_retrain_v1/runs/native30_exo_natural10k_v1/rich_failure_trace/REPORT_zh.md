本次只做 CPU 单窗口推理诊断，权重固定为 8k（SHA256 f8dcc1d63608895b1a8ade99fd4314d75f760014234de2de203dd6d6083d477a），无优化器更新，无 CUDA 初始化。4 线程，峰值 RSS 2.09 GiB，全过程 57.6 秒。

案例为固定 VAL Pavallion_006_yoga1，start=640，source_first=645，192 帧，seed=1742430047。clean/noisy 轨迹和 meta 与历史 GPU 文件完全相同，CPU 参考与历史参考最大特征差 5.01e-6。CPU 与 CUDA 相同数字种子不保证产生相同初始噪声，因此本轮 ON/OFF 是严格的 CPU 内部配对；与历史 GPU 的差异仅描述，不作逐值复现声明。

| 输入 | 场景 | 遮挡 MPJPE mm | 可见 MPJPE mm | 右膝局部角中位数 |
|---|---|---:|---:|---:|
| noisy | OFF | 68.25 | 20.38 | 7.68° |
| noisy | ON | 594.73 | 77.18 | 155.20° |
| clean | OFF | 74.30 | 2.17 | 4.60° |
| clean | ON | 86.17 | 3.35 | 5.83° |

noisy 的参考遮挡 MPJPE 95.26 mm；clean 参考 69.29 mm。本次 noisy ON 与历史 GPU ON 的全身关节平均距离 6.50 mm；两者遮挡 MPJPE 分别 594.73 / 594.56 mm，错误姿态模式一致，但未重现历史 CUDA 初始噪声。

错误在第一轮 t=999 的第一次核心预测就已经出现：右膝约 59.92°（OFF 5.44°），第二次核心预测 69.36°。此时学习反馈最大角增量约 4.0e-17°，观测 IK 前后右膝角不变。第一遍核心预测发生在本轮预测体表局部查询之前；builder 的初始 BPS/valid 为零。因此这条 CPU 配对证据表明，错误由启用场景条件的核心预测首先触发，随后 DDIM 演化扩大；不是最后物理修正或 IK 首先折坏膝盖。当前比较同时关闭/打开全局场景、patch/dynamic attention、局部查询和学习反馈，尚未拆出具体哪个编码/注入分支。

错误也涉及另一条腿：noisy ON 最后左右膝的局部旋转 GT 误差中位数为 133.0° /155.8°，左右踝位置误差均值 740.5 /941.8 mm；OFF 对应约 11.4° /5.8°、45.1 /51.9 mm。右膝点可见不等于膝关节旋转被约束；踝和脚隐藏时该旋转没有观测 IK 强度。

这仍是一个指定失败窗口的因果诊断，不能直接推断 RICH 全体或其他数据集都应关闭场景。精确历史 GPU 初始噪声和更多案例需另行核对。

完整逐步统计本地保存在四个 *_scene_*.json；TRACE_COMPACT.json 保留所有22关节的逐步旋转和位置均值/中位数、输入/权重/源代码身份。predictions.npz 为本次已授权比较保留的4个预测。INVOCATION.txt 是可复现操作记录，未新增 Python 脚本。为避开当前主机可选 nvrtc 预加载的 ELF relocation 错误，只有该 CPU 进程设置 TORCH_USE_RTLD_GLOBAL=1；没有修改库、环境文件或训练进程。

追加的非瑜伽窗口：LectureHall_020_wipingchairs1 start704 noisy，参考144.68毫米，CPU OFF280.09毫米、ON302.07毫米，二者都退化。可见错误不只发生于瑜伽，且场景不是所有失败的必要条件。具体身份与记录见 wipingchairs_rotating_case5/SUMMARY.json。

观察器不扰动验证已完成：原瑜伽noisy sceneON移除全部observer hooks，以相同CPU seed独立重复，输出逐值完全一致，最大运动特征差0、平均关节距离0。耗时22.15秒，CUDA未初始化。因此逐步记录未改变该CPU配对结论。历史GPU初始噪声不同的限制仍成立。
