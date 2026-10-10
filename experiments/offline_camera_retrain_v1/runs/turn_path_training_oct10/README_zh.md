# 无GT初始化的整体转向与脚步配对短训

正在进行两组各600步：continued_control只继续原coordination_v1，turn_path在相同初始化/采样/噪声上加入累计有符号转向、根部与腿部相对旋转、训练目标速率＋3°/帧的软超限惩罚，以及原55k脚部位置/速度保持。这里没有通用生理关节限位，不能称为完整关节安全阈值。

两组都只提供已知相机/场景/身体模板条件，不提供GT身体初始帧或历史。干净身体GT只用于扩散训练目标与监督；评估全部128帧从随机噪声生成。

`protocol.json`记录训练配置，`pre_registered_evaluation.json`在评分前登记固定16条序列×2种子及850/906门槛。保留三组模型配对评估，不能把继续训练本身的变化归因给新损失。

每组只保存一个evaluation-only last.pt，不保存optimizer、best或编号checkpoint。正式55k保持可续训。代码：train_turn_path_trial.py、turn_path_objective.py、evaluate_turn_path_trial.py。完成评估后更新ASSESSMENT_zh.md，未通过的motion展示候选不保留。
