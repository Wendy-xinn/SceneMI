第二遍保留patch的开发检验：NO。原先已冻结的NO保留不变，本轮未运行或读取confirmation。

本轮复用original30条，新生成唯一candidate30条；总计60条参加独立CPU NumPy FK复核，全部通过。6个candidate EGO输出与original逐值完全相同。

自然加权隐藏误差：clean 77.033→77.013mm；noisy 126.679→126.556mm；两条件平均 101.856→101.784mm，仅改善0.0703%。

未通过项：自然加权平均改善未达到3%；已知瑜伽noisy隐藏误差585.784→587.295mm，参考95.257mm，未修复。其他保护项通过；noisy中比参考差至少50mm的窗口仍为2→2。

该策略仅关闭EXO第一遍proposal的旧patch条件，第二遍refinement保留旧patch。结果几乎等同原版，不能作为修复版本；应继续定位第二遍旧patch条件。

所有阈值和自然数据集权重沿用前次预声明规则。完整逐组、逐窗和保护项见DEV_CHOICE.json。本开发检验不能作为完整验证集或论文最终指标。
