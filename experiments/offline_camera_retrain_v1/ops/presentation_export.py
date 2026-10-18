"""CPU-only paired exports from existing motion_viewer playback caches."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
FONT = '/mnt/c/Windows/Fonts/msyh.ttc'
BLUE, PURPLE, GREEN, ORANGE, TEAL = '#287bc8', '#b53397', '#15964b', '#e16b16', '#62b9bd'
W, H, TOP = 580, 570, 112


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def font(size):
    return ImageFont.truetype(FONT if Path(FONT).exists() else 'DejaVuSans.ttf', size)


def load(case, regime):
    meta = json.loads(Path(case['cache_manifest']).read_text())
    assert meta['mesh_fk_max_error_m'] < 1e-4 and not meta['scene_errors']
    with np.load(case['cache'], allow_pickle=False) as saved:
        keys = ('truth_joints', 'faces', 'visible_static', regime+'_vertices', regime+'_joints',
                regime+'_observations', regime+'_observation_meta')
        data = {key: saved[key] for key in keys}
    return data, meta


def projection(before, after, regime):
    normal = np.array([.8, .28, 1.]); normal /= np.linalg.norm(normal)
    right = np.cross([0., 1., 0.], normal); right /= np.linalg.norm(right)
    up = np.cross(normal, right)
    matrix = np.stack((right, up, normal), axis=1)
    points = np.concatenate([before['truth_joints'].reshape(-1, 3),
        before[regime+'_vertices'][:, ::20].reshape(-1, 3), after[regime+'_vertices'][:, ::20].reshape(-1, 3)]) @ matrix
    low, high = points[:, :2].min(0)-.17, points[:, :2].max(0)+.17
    centre = (low+high)/2
    scale = min((W-90)/(high[0]-low[0]), (H-95)/(high[1]-low[1]))
    def project(value):
        p = np.asarray(value) @ matrix
        return np.stack(((p[..., 0]-centre[0])*scale+W/2,
                         -(p[..., 1]-centre[1])*scale+(H-35)/2, p[..., 2]), axis=-1)
    return project, dict(camera_direction=normal.tolist(), pixels_per_metre=float(scale),
                        projection='orthographic; same camera, scale and coordinates before/after; +Y up')


def render(data, regime, frame, project, title, hidden_mm):
    result = Image.new('RGB', (W, H), '#fbfcfd'); draw = ImageDraw.Draw(result)
    g = data['truth_joints']; pred = data[regime+'_joints']
    # Only actual visible input surfaces, cropped to the human inspection area.
    points = data['visible_static']; lower = g.min((0, 1))-[.8,.2,.8]; upper = g.max((0, 1))+[.8,.3,.8]
    points = points[((points >= lower) & (points <= upper)).all(1)]
    for x, y, _ in project(points):
        if 0 < x < W and 36 < y < H-30:
            draw.ellipse((x-1,y-1,x+1,y+1), fill='#b6dedf')
    verts = data[regime+'_vertices'][frame]; faces = data['faces']; pv = project(verts)
    triangles = verts[faces]; normals = np.cross(triangles[:, 1]-triangles[:, 0],triangles[:, 2]-triangles[:, 0])
    light = np.abs(normals @ np.array([.25,.7,1.])) / np.maximum(np.linalg.norm(normals,axis=1),1e-12)
    colors = np.clip(np.array([181,51,151])[None]*(.64+.34*np.minimum(light,1.))[:,None],0,255).astype(np.uint8)
    order = np.argsort(pv[faces,2].mean(1))
    for index in order:
        draw.polygon([tuple(x) for x in pv[faces[index],:2]],fill=tuple(colors[index]))
    # The GT skeleton is deliberately drawn as an overlay for error inspection.
    truth, estimate = project(g[frame]), project(pred[frame])
    for joint, parent in enumerate(PARENTS):
        if parent >= 0:
            draw.line([tuple(truth[parent,:2]),tuple(truth[joint,:2])],fill=BLUE,width=3)
    for foot in (10,11):
        start = max(0,frame-30)
        for joints, color in ((g,BLUE),(pred,PURPLE)):
            trail = project(joints[start:frame+1,foot])
            if len(trail)>1: draw.line([tuple(p[:2]) for p in trail],fill=color,width=2)
    mask = data[regime+'_observation_meta'][frame,:,0]>0
    observations = project(data[regime+'_observations'][frame,:,:3]*2)
    for x,y,_ in observations[mask]: draw.ellipse((x-4,y-4,x+4,y+4),fill=GREEN,outline='white',width=1)
    for x,y,_ in estimate[~mask]: draw.ellipse((x-5,y-5,x+5,y+5),outline=ORANGE,width=3)
    draw.rectangle((0,0,W,52),fill='#ffffff')
    draw.text((16,8),title,font=font(22),fill='#17232b')
    draw.text((16,35),f'整段隐藏关节 MPJPE {hidden_mm:.1f} mm',font=font(14),fill='#455560')
    draw.rectangle((0,H-34,W,H),fill='#ffffff')
    draw.text((16,H-28),f'第 {frame:03d} 帧 · {frame/30:.2f} s   |   固定相机 / 米制坐标',font=font(15),fill='#455560')
    return result


def paired(before, after, regime, frame, project, caption, values):
    result = Image.new('RGB',(W*2,H+TOP),'white'); draw = ImageDraw.Draw(result)
    draw.text((22,12),caption,font=font(23),fill='#17232b')
    for x,label,color in ((22,'GT骨架',BLUE),(185,'预测mesh',PURPLE),(390,'可见输入',GREEN),(570,'隐藏预测关节',ORANGE),(835,'已见场景',TEAL)):
        draw.ellipse((x,57,x+12,69),fill=color); draw.text((x+19,50),label,font=font(17),fill='#364952')
    draw.text((22,82),'展示选例，不代表整体；无额外对齐、缩放或贴地。手脸中性；GT骨架为透视检查叠加。',font=font(15),fill='#56666f')
    for index,(data,title) in enumerate(((before,'适配前 · 父模型41k'),(after,'适配后 · 内部adapter 1000步'))):
        result.paste(render(data,regime,frame,project,title,values[index]),(index*W,TOP))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--cases',default='6:clean,9:clean,10:noisy')
    parser.add_argument('--viewer-url',default='http://localhost:8802/')
    args = parser.parse_args(); run=args.run.resolve();out=run/'presentation/visuals';out.mkdir(parents=True,exist_ok=True)
    prepared=run/'visuals/viser/PREPARED_CASES.json'; cases=json.loads(prepared.read_text())['cases']
    entries=json.loads((run/'prepared/validation_plan.json').read_text())['plan']
    comparisons=json.loads((run/'COMPARISON.json').read_text())['cases']
    report=dict(scope='VAL demonstration examples selected after known results; not overall or TEST evidence',
        GPU_used=False,inference_rerun=False,source_sha256={str(prepared):sha(prepared),str(run/'COMPARISON.json'):sha(run/'COMPARISON.json')},cases=[])
    acceptance_path=run/'ACCEPTANCE.json'
    acceptance=json.loads(acceptance_path.read_text())
    if acceptance.get('status') not in ('pass','failed'):
        raise ValueError('Presentation requires a completed development acceptance report')
    report['source_sha256'][str(acceptance_path)]=sha(acceptance_path)
    report['development_acceptance_status']=acceptance['status']
    report['viewer_url']=args.viewer_url
    covers=[]
    for specification in args.cases.split(','):
        index,text=specification.split(':');index=int(index);entry=entries[index];regime=text
        matched=sorted([c for c in cases if c['case_index']==index],key=lambda c:c['step']);assert len(matched)==2
        (before,bmeta),(after,ameta)=(load(c,regime) for c in matched)
        for key in ('truth_joints','visible_static',regime+'_observations',regime+'_observation_meta'):
            assert np.array_equal(before[key],after[key]),'Before/after reference or input differs: '+key
        comparison=next(r for r in comparisons if r['identity']['sequence_id']==entry['sequence_id'] and r['identity']['regime']==regime)
        values=[comparison['metrics']['unconditioned_mpjpe_mm'][k]['value'] for k in ('baseline','final')]
        for data,value in zip((before,after),values):
            error=np.linalg.norm(data[regime+'_joints']-data['truth_joints'],axis=-1)*1000
            assert abs(error[data[regime+'_observation_meta'][...,0]<=0].mean()-value)<.01
        caption=f"{entry['dataset'].upper()} EXO · {regime} · {'失败对照' if values[1]>values[0] else '改善选例'}  |  case {index}"
        project,camera=projection(before,after,regime);stem=f'case{index:02d}_{regime}'
        frames=list(range(0,192,3));gif=[]
        with imageio.get_writer(str(out/(stem+'.mp4')),fps=10,codec='libx264',quality=8,macro_block_size=2) as writer:
            for frame in frames:
                current=paired(before,after,regime,frame,project,caption,values)
                writer.append_data(np.asarray(current));gif.append(current.resize((928,546)))
        gif[0].save(out/(stem+'.gif'),save_all=True,append_images=gif[1:],duration=100,loop=0,optimize=False)
        sheet=Image.new('RGB',(W*2,(H+TOP)*4),'white')
        for row,frame in enumerate((0,64,128,191)):sheet.paste(paired(before,after,regime,frame,project,caption,values),(0,row*(H+TOP)))
        sheet.save(out/(stem+'_comparison.png'));covers.append(paired(before,after,regime,96,project,caption,values))
        item=dict(case_index=index,dataset=entry['dataset'],sequence_id=entry['sequence_id'],regime=regime,
            hidden_mpjpe_before_mm=values[0],hidden_mpjpe_after_mm=values[1],input_GT_scene_exact_before_after=True,
            mesh_fk_max_error_m=max(bmeta['mesh_fk_max_error_m'],ameta['mesh_fk_max_error_m']),camera=camera,
            source_fps=30,render_fps=10,render_source_frames=frames,static_source_frames=[0,64,128,191],
            artifacts=[stem+'.mp4',stem+'.gif',stem+'_comparison.png'],
            cache_sources={c['cache']:sha(c['cache']) for c in matched},viewer_menus=[c['label'] for c in matched])
        report['cases'].append(item);print(json.dumps(dict(completed=stem,hidden_mm=values),ensure_ascii=False),flush=True)
    cover=Image.new('RGB',(W*2,(H+TOP)*len(covers)),'white')
    for row,current in enumerate(covers):cover.paste(current,(0,row*(H+TOP)))
    cover.save(out/'EXO_examples.png')
    (out/'EXPORT_MANIFEST.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    body=''.join(f'<h2>{html.escape(c["dataset"])} · case {c["case_index"]} · {c["regime"]}</h2><p>隐藏关节MPJPE：{c["hidden_mpjpe_before_mm"]:.1f} → {c["hidden_mpjpe_after_mm"]:.1f} mm</p><video controls loop width="100%" src="{c["artifacts"][0]}"></video><p><a href="{c["artifacts"][1]}">GIF</a> · <a href="{c["artifacts"][2]}">四时刻对比PNG</a></p>' for c in report['cases'])
    screen='通过' if acceptance['status']=='pass' else '未通过'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>EXO前后对比</title><style>body{font-family:sans-serif;max-width:1160px;margin:32px auto;color:#24323d}video{border:1px solid #ddd}</style><h1>EXO内部适配：固定选例前后对照</h1>'
        f'<p>固定VAL选例，不能代替整体结论。本候选开发验收：{screen}；详见<a href="../../RESULTS_zh.md">结果报告</a>。无TEST挑例或重新推理。</p>'
        f'<p><a href="{html.escape(args.viewer_url,quote=True)}">Viser交互查看（需要本地服务或端口转发）</a> · <a href="EXO_examples.png">汇总PNG</a> · <a href="EXPORT_MANIFEST.json">来源与坐标核验</a></p>'+body)


if __name__=='__main__':
    main()
