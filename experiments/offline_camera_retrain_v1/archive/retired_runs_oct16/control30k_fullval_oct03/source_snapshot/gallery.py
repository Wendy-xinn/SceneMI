"""Standalone canvas animation of held-out joints; no network/CDN needed."""
import json
import html
from pathlib import Path


def write_run_index(run_dir):
    """Refresh a lightweight index after each successfully completed evaluation."""
    run_dir = Path(run_dir)
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    initial_steps = config.get('initial_total_steps', config.get('initial_checkpoint_step', 0))
    rows = []
    for path in sorted((run_dir / 'evaluations').glob('step_*/metrics.json')):
        report = json.loads(path.read_text())
        groups = list(report['groups'].values())
        variant = report['primary_variant']
        def mean(key):
            return sum(g['mean_variants'][variant][key] for g in groups) / len(groups)
        def optional(key, scale=1):
            if all(key in g['mean_variants'][variant] for g in groups):
                return f'{scale * mean(key):.3f}'
            return '—'
        gt_speed = sum(g['mean_ground_truth_proxy_metrics']['mean_joint_speed_m_per_frame']
                       for g in groups) / len(groups)
        link = html.escape(str(path.with_name('gallery.html').relative_to(run_dir)))
        rows.append(f'<tr><td><a href="{link}">{report["checkpoint_step"]}</a></td>'
                    f'<td>{mean("fk_mpjpe_cm"):.2f}</td><td>{mean("head_error_cm"):.2f}</td>'
                    f'<td>{mean("head_path_relative_error_cm"):.2f}</td>'
                    f'<td>{mean("mean_joint_speed_m_per_frame") / max(gt_speed, 1e-8):.2f}</td>'
                    f'<td>{mean("truth_stance_foot_slide_cm_per_frame"):.3f}</td>'
                    f'<td>{optional("slowest_foot_horizontal_cm_per_frame")}</td>'
                    f'<td>{optional("foot_relative_speed_ratio")}</td>'
                    f'<td>{optional("contact_coverage")}</td>'
                    f'<td>{100 * mean("support_floating_m"):.2f}</td>'
                    f'<td>{100 * mean("support_penetration_m"):.2f}</td></tr>')
    page = ('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60">'
            '<title>SceneMI training progress</title><style>body{font:16px system-ui;margin:30px}'
            'td,th{padding:12px;border-bottom:1px solid #ccc}a{color:#186ec6}</style>'
            f'<h2>SceneMI · training / gait diagnostics</h2><p>本阶段起点：累计 {initial_steps:,} 步；表内为本阶段步数。</p>'
            '<p>每 1,000 步更新；点击步数播放固定验证动作。'
            '速度比为生成/GT，接近 0 表示动作过静；脚滑须结合速度判断。'
            '较慢脚速度不要求与 GT 同相位；脚相对骨盆速度比反映是否真的迈步，不能单独作为质量分数。'
            '悬浮与穿透是 GT 脚高度代理，非真实场景碰撞。单位 cm（脚滑 cm/帧）。</p>'
            '<table><tr><th>Step / 播放</th><th>MPJPE</th><th>头部误差</th>'
            '<th>头路径相对误差</th><th>速度比</th><th>GT相位脚滑</th><th>较慢脚速度</th>'
            '<th>脚相对骨盆速度比</th><th>接近支撑高度比例</th><th>悬浮代理</th>'
            '<th>穿透代理</th></tr>' + ''.join(rows) + '</table>')
    temporary = run_dir / 'index.tmp'
    temporary.write_text(page)
    temporary.replace(run_dir / 'index.html')


def write_gallery(path, cases, step):
    payload = json.dumps(dict(step=step, cases=cases), ensure_ascii=True).replace('<', '\\u003c')
    template = r'''<!doctype html><meta charset="utf-8">
<title>SceneMI training samples</title>
<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;margin:24px}select,button,input{margin:8px;padding:6px}canvas{background:#172033;width:100%;max-width:1100px;border:1px solid #475569}pre{white-space:pre-wrap}label{display:inline-block}.muted{color:#aab8cc}</style>
<h2 id="title"></h2>
<p>GT 灰色 · 生成蓝色 · 输入相机轨迹橙色。固定世界坐标、20 fps，可暂停逐帧检查脚步。仅骨架诊断，未绘制场景网格；不能据此判定真实场景碰撞。</p>
<label>样本 <select id="case"></select></label><label>条件 <select id="variant"></select></label>
<label>视角 <select id="view"><option value="side">侧视 x/y</option><option value="front">正视 z/y</option><option value="top">俯视 x/z</option><option value="oblique" selected>斜视</option></select></label>
<button id="play">暂停</button><input id="frame" type="range" min="0" max="127" value="0"><span id="time"></span>
<label><input id="support" type="checkbox" checked>GT 脚关节支撑高度参考（非真实地面）</label>
<label><input id="trails" type="checkbox" checked>最近 1 秒脚迹：左青 / 右紫</label>
<canvas id="canvas" width="1100" height="660"></canvas><pre id="metrics"></pre>
<p class="muted">head_error 对照真值头关节；相机到头关节存在固定/姿态相关偏移。低脚滑需结合动作幅度和接触率判断。支撑高度指标为 GT 脚高度代理，非场景 SDF。</p>
<script>
const data=PAYLOAD;
const $=id=>document.getElementById(id), ctx=$('canvas').getContext('2d');
const parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19];
$('title').textContent=`SceneMI · step ${data.step} · ${data.cases.length} samples`;
data.cases.forEach((c,i)=>{const o=new Option(`${c.group} · ${c.identity.sequence_id} · ${c.identity.source_start_30fps}`,i);$('case').add(o)});
Object.keys(data.cases[0].tracks).filter(k=>k!=='GT').forEach(k=>$('variant').add(new Option(k,k)));
let playing=true,last=0,frame=0,bounds;
function project(p){switch($('view').value){case 'top':return [p[0],p[2]];case 'front':return[p[2],p[1]];case 'side':return[p[0],p[1]];default:return[.8*p[0]-.6*p[2],p[1]-.25*(p[0]+p[2])]}}
function updateBounds(){const c=data.cases[+$('case').value];let lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];
for(const track of [c.tracks.GT,c.tracks[$('variant').value]])for(const pose of track)for(const p of pose){const q=project(p);for(let d=0;d<2;d++){lo[d]=Math.min(lo[d],q[d]);hi[d]=Math.max(hi[d],q[d])}}
for(const p of c.camera){const q=project(p);for(let d=0;d<2;d++){lo[d]=Math.min(lo[d],q[d]);hi[d]=Math.max(hi[d],q[d])}}
bounds={center:lo.map((v,d)=>(v+hi[d])/2),scale:Math.min(990/Math.max(.5,hi[0]-lo[0]),550/Math.max(.5,hi[1]-lo[1]))};
const heights=c.tracks.GT.flatMap(p=>[p[10][1],p[11][1]]).sort((a,b)=>a-b);
bounds.support=heights[Math.floor(.05*(heights.length-1))];
const roots=c.tracks.GT.map(p=>p[0]);bounds.xz=[0,2].map(d=>[Math.min(...roots.map(p=>p[d]))-.5,Math.max(...roots.map(p=>p[d]))+.5]);
const m=c.metrics[$('variant').value];$('metrics').textContent=JSON.stringify(m,null,2);$('frame').max=c.tracks.GT.length-1;}
function xy(p){const q=project(p);return[550+(q[0]-bounds.center[0])*bounds.scale,330-(q[1]-bounds.center[1])*bounds.scale]}
function line(a,b,color,width){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...xy(a));ctx.lineTo(...xy(b));ctx.stroke()}
function draw(){ctx.clearRect(0,0,1100,660);const c=data.cases[+$('case').value];frame=Math.min(frame,c.tracks.GT.length-1);
if($('support').checked){const [[x0,x1],[z0,z1]]=bounds.xz,y=bounds.support;for(let i=0;i<=10;i++){let x=x0+(x1-x0)*i/10,z=z0+(z1-z0)*i/10;line([x,y,z0],[x,y,z1],'#334155',1);line([x0,y,z],[x1,y,z],'#334155',1)}}
if($('trails').checked){const track=c.tracks[$('variant').value];for(let t=Math.max(1,frame-19);t<=frame;t++){line(track[t-1][10],track[t][10],'#22d3ee',2);line(track[t-1][11],track[t][11],'#c084fc',2)}}
for(let t=1;t<c.camera.length;t++)line(c.camera[t-1],c.camera[t],'#e8a749',1);
for(const [name,color] of [['GT','#aab4c4'],[$('variant').value,'#54b9ff']]){const pose=c.tracks[name][frame];for(let j=1;j<22;j++)line(pose[parents[j]],pose[j],color,3);for(const p of pose){ctx.fillStyle=color;ctx.beginPath();ctx.arc(...xy(p),3,0,Math.PI*2);ctx.fill()}}
ctx.fillStyle='#e8a749';ctx.beginPath();ctx.arc(...xy(c.camera[frame]),6,0,Math.PI*2);ctx.fill();$('time').textContent=`frame ${frame} / ${(frame/20).toFixed(2)}s`;$('frame').value=frame;}
for(const id of ['case','variant','view'])$(id).onchange=()=>{updateBounds();draw()};
for(const id of ['support','trails'])$(id).onchange=draw;
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'暂停':'播放'};
$('frame').oninput=()=>{frame=+$('frame').value;playing=false;$('play').textContent='播放';draw()};
function tick(now){if(playing && now-last>=50){frame=(frame+1)%data.cases[+$('case').value].tracks.GT.length;last=now;draw()}requestAnimationFrame(tick)}
updateBounds();draw();requestAnimationFrame(tick);
</script>'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace('PAYLOAD', payload))
