"""Render the reference-preserving followup using the shared review template."""
from pathlib import Path
import review

if __name__=='__main__':
    review.ROOT=Path(__file__).resolve().parent/'reference_anchor'
    review.main()
    review.main('predictions_protected.pt','PROTECTED_')
    for name in ('REPLAY.html','PROTECTED_REPLAY.html'):
        path=review.ROOT/name
        html=path.read_text().replace('Frozen55k initialization','Observation-only reference')
        html=html.replace('Residual trained without scene','55k prior + residual, scene OFF')
        html=html.replace('Residual trained with scene','55k prior + residual, scene ON')
        if name.startswith('PROTECTED'):
            html=html.replace('<h2>','<p>Posthoc guard: fully hidden branches only; available joint positions preserved. <a href="REPLAY.html">Raw residual replay</a></p><h2>',1)
            html=html.replace('D.status;',"'Exploratory guarded evaluation; not independent confirmation';")
        else:html=html.replace('<h2>','<p><a href="PROTECTED_REPLAY.html">Protected hidden-branch replay</a></p><h2>',1)
        html=html.replace('55k + EXO residual: development review','55k＋exo 遮挡修正对照')
        html=html.replace('Observation-only reference','可见点参考姿态')
        html=html.replace('55k prior + residual, scene OFF','55k 先验＋残差（无场景）')
        html=html.replace('55k prior + residual, scene ON','55k 先验＋残差（有场景）')
        html=html.replace('Blue = GT; magenta = prediction; orange = available observations. Gray/cyan = nearest observed scene patches (cyan: other person). Sparse patches are not a full scene scan. Display sampled at15Hz, metrics at30Hz. Known shape, ideal visible-joint frontend simulation; legacy55k overlap prevents clean official benchmark claims.',
            '蓝色＝GT；品红＝预测；橙色＝输入可见关节；灰色／青色＝局部观测场景点（青色为其他人）。三栏共用跟随GT根位置的观察视角，预测坐标未对齐或平移。可播放、旋转、缩放。当前使用理想可见点模拟前端与已知体型，尚非真实视频前端实验。')
        html=html.replace('Posthoc guard: fully hidden branches only; available joint positions preserved.','遮挡分支保护：只修改完全不可见分支，保持输入可见关节位置。')
        html=html.replace('Raw residual replay','查看普通残差').replace('Protected hidden-branch replay','查看遮挡保护版本')
        html=html.replace('Exploratory guarded evaluation; not independent confirmation','探索性对照，尚未独立验证')
        path.write_text(html)
