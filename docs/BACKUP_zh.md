# 本地代码备份（2026-10-09）

此备份保留SceneMI修改、混合原生SMPL/SMPL-X训练与预处理代码、因果动态场景、RICH接触、评估及朝向/头部约束实验。数据、人体模型、权重、临时日志、wheel和第三方PerspectiveFields检出不上传。

当前长训55k已完成。时间归一化及头部硬约束六组消融已完成，均未通过质量验收；旧checkpoint损坏已通过重跑和持久化保存修复。后续soft_body_turn_robust_oct09实验保留软条件，分别测试高噪声转向权重和颈部父链朝向监督，并检查输入轨迹偏差及缺失。结果以对应实验报告为准。

用户备份仓库为https://github.com/Wendy-xinn/SceneMI；原作者origin保留用于追溯，不写入原作者仓库。

运行需要scenemi环境、原始数据及兄弟目录diffusion-motion-inbetweening。docs/local_dependencies下备份了该目录中被SceneMI直接使用的本地辅助代码及来源提交，恢复时复制到对应相对路径。PerspectiveFields通过requirements中的固定上游提交安装。训练权重仍保存在本机；此Git备份只能恢复代码和报告。
