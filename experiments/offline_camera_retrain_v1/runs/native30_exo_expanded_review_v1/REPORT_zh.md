扩大EXO验证：裸41k、v1 adapter1000、balanced v2 adapter1000。
状态：partial_do_not_select，12/384次预测；不自动替换模型。

64个不同录制各1窗：RICH27、TRUMANS20、EgoBody17；新55与旧9回归分开。输入/种子逐窗匹配，30Hz/192帧/DDIM20，场景与物理修正开启。
主证据为RICH新24录制的隐藏关节及可见/脚部约束；旧9不作为新泛化证据。完整结果须独立CPU复算。

|范围|模型|数据/输入|窗/录制|World mm|隐藏 mm|可见 mm|低速3D脚速m/s|
|---|---|---|---:|---:|---:|---:|---:|
|new55|base41k|rich/clean|2/2|9.549|44.120|5.599|0.091|
|new55|base41k|rich/noisy|2/2|38.782|52.945|37.276|0.116|
|new55|v1adapter1000|rich/clean|2/2|5.757|33.610|2.936|0.092|
|new55|v1adapter1000|rich/noisy|2/2|38.456|70.848|34.721|0.132|
|new55|balancedv2adapter1000|rich/clean|2/2|5.171|33.723|2.296|0.091|
|new55|balancedv2adapter1000|rich/noisy|2/2|34.862|66.458|31.317|0.120|
|all64|base41k|rich/clean|2/2|9.549|44.120|5.599|0.091|
|all64|base41k|rich/noisy|2/2|38.782|52.945|37.276|0.116|
|all64|v1adapter1000|rich/clean|2/2|5.757|33.610|2.936|0.092|
|all64|v1adapter1000|rich/noisy|2/2|38.456|70.848|34.721|0.132|
|all64|balancedv2adapter1000|rich/clean|2/2|5.171|33.723|2.296|0.091|
|all64|balancedv2adapter1000|rich/noisy|2/2|34.862|66.458|31.317|0.120|

可见/隐藏按输入位置可用掩码；GT脚速<0.2m/s为运动学代理，不是真实接触或全mesh穿模。PA为逐帧相似变换；World未对齐。
同一录制的两角色未当成两个独立录制；clean/noisy不是独立样本；不同录制可能同主体/场景。配对逐录制变化、有效分母与工程guards详见SUMMARY.json。
