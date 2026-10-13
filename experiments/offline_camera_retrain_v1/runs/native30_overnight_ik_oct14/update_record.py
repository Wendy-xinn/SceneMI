"""Update the user's single master record from durable overnight artifacts."""
import datetime as dt
import json
import os
from pathlib import Path
import time
import subprocess

BASE=Path(__file__).resolve().parent
OUT=BASE/'experiment'
MASTER=BASE.parent/'exo_occlusion_review_oct12/EXPERIMENT_AND_ARCHITECTURE_zh.md'
BEGIN='<!-- NATIVE30_OVERNIGHT_BEGIN -->'
END='<!-- NATIVE30_OVERNIGHT_END -->'


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def mean(rows,mode,regime,key,field='metrics'):
    selected=[r[field][key] for r in rows if r['mode']==mode and r['regime']==regime]
    count=sum(m['count'] for m in selected)
    return f"{sum(m['value']*m['count'] for m in selected if m['count'])/count:.2f}" if count else '—'


def update():
    protocol=read(OUT/'protocol.json');progress=read(OUT/'candidate/progress.json')
    report=read(OUT/'candidate/evaluation_latest.json');control=read(OUT/'control/evaluation_latest.json')
    audit=read(OUT/'bank_audit.json');failure=read(OUT/'failure.json')
    pid=read(OUT/'launcher.json').get('pid')
    if pid and not failure and protocol.get('status') not in ('completed','completed_gate_failed'):
        try:
            state=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
            alive=state!='Z'
        except OSError:alive=False
        if not alive:
            tail='\n'.join((OUT/'console.log').read_text() .splitlines()[-35:])
            failure=dict(traceback=tail or 'Detached pipeline exited before completion; inspect console.log',pid=pid,
                utc=dt.datetime.now(dt.timezone.utc).isoformat())
            temp=OUT/'failure.tmp';temp.write_text(json.dumps(failure,ensure_ascii=False,indent=2)+'\n');temp.replace(OUT/'failure.json')
    now=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S CST')
    lines=[BEGIN,'','## 4.3 10月14日晚间训练：独立VAL场景验收','',f"自动更新：{now}。阶段：`{protocol.get('status','sealing_zero_step_preparation')}`；候选已保存步数：**{progress.get('step',0)}**。"]
    lines+=['','本轮置信度约束放在扩散内部的预测proposal与最终x0：exo用可观测骨盆／双髋／脊柱约束根位姿，ego只用头部位置约束平移，不从相机朝向强制身体朝向。root gain=0.9；新增关节位置IK gain=0.9，6次阻尼迭代，lambda=0.0025m²，每次角度≤0.35rad，不修改root；都随alpha和置信度／不确定性减弱。raw x0仍受GT监督；场景可以通过主干、67vertex查询及位置／旋转反馈改变动作。',
        '', '控制流程：[代码](../../native30_overnight.py)、[配置](../../native30_overnight_ik_config.json)、[完整日志](../native30_overnight_ik_oct14/experiment/console.log)、[运行状态](../native30_overnight_ik_oct14/experiment/protocol.json)。每个模型只保留一份原子可续训last.pt；断开界面后独立进程继续，进程异常另记failure.json。',
        '', '|项目|本轮设置／实际状态|','|---|---|',
        f"|初始化与数据范围|{protocol.get('initialization','等待继续TRAIN pilot1000')}。候选／对照共有同一TRAIN预训练、Adam与RNG；没有使用旧55k。只有TRAIN优化，VAL监控；真实视频前端未接入。|",
        '|采样|batch2；每10个样本槽准确7 ego／3 exo；ego预期65%仅头、35%头＋双腕；50%clean，余下平滑轨迹误差／短gap；每次共同世界yaw增广。|',
        '|场景|同观察者整段实际观测静态并集，当前帧动态；不读取完整未观测扫描或目标人体皮肤。TRUMANS exo仍是合成观察者。|',
        '|候选预算|同一TRAIN pilot1000步后先到2200总步，失败可延到4000总步；全池训练最多15000总步，实际受早晨时间限制。|',
        '|中途监控|第500／1000／2000步在8个独立VAL条件窗用DDIM8观察趋势；最终验收固定16个VAL条件窗，clean/noisy分开，两个DDIM20种子。中途DDIM8与最终DDIM20分别记录。|',
        '|匹配对照|同初始化、窗口、扰动、timesteps、Gaussian扩散噪声、学习率和优化步数，关闭场景训练；逐步核对样本身份、t和输入轨迹校验和；并作同权重scene-off消融。|',
        '|扩大数据条件|ego整体、exo未观测关节误差均比初始参考低≥5%，且比同权重scene-off及匹配训练对照低≥2%；clean/noisy都通过、两个种子方向一致、每种条件至少两个语料有场景正收益。exo整体／可信部位允许初始参考＋5mm、root＋10mm以内。任何项失败均不扩全池。阈值是本轮预先指定的实验门槛，不是统计显著性或正式benchmark结论。|',
        '|物理诊断|报告GT静止期足速、可观察接触表面距离／查询覆盖率、局部朝相机平面背侧proxy；不把partial point cloud当完整SDF，缺失表面无碰撞测量。|']
    for split in ('train','validation'):
        row=audit.get(split,{})
        if row:lines.append(f"|{split}实际小集|{row['condition_windows']}个条件窗；{row['window_frame_slots']}累计窗帧；{row['unique_target_30hz_times']}个目标30Hz去重时间点；{row['distinct_target_streams']}个独立目标流。ego/exo可能覆盖同一时间，不重复宣称为独立动作。|")
    if report:
        full=[r for r in report['rows'] if r['scene']=='full'];off=[r for r in report['rows'] if r['scene']=='off']
        lines+=['',f"以下为候选第{report['step']}步的独立VAL固定窗监控，DDIM{report['ddim_steps']}，原始世界系、无GT对齐；不是整个VAL的正式benchmark。单位mm。",'',
            '|条件|初始整体|场景扩散整体|同权重无场景整体|匹配无场景训练整体|初始缺失部位|场景缺失部位|场景可信部位|场景root|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for mode in ('ego','exo'):
            for regime in ('clean','noisy'):
                values=[mean(full,mode,regime,'mpjpe_mm','reference_metrics'),mean(full,mode,regime,'mpjpe_mm'),
                    mean(off,mode,regime,'mpjpe_mm'),mean(control.get('rows',[]),mode,regime,'mpjpe_mm'),
                    mean(full,mode,regime,'unconditioned_mpjpe_mm','reference_metrics'),mean(full,mode,regime,'unconditioned_mpjpe_mm'),
                    mean(full,mode,regime,'conditioned_mpjpe_mm'),mean(full,mode,regime,'root_position_error_mm')]
                lines.append('|'+mode+' '+regime+'|'+'|'.join(values)+'|')
        step=report['step']
        for regime in ('CLEAN','NOISY'):
            path=OUT/f'candidate/VALIDATION_MESH_{regime}_STEP{step}.png'
            if path.exists():lines+=['',f"[{regime.lower()}原坐标native mesh对比](../native30_overnight_ik_oct14/experiment/candidate/{path.name})：蓝GT、灰初始参考、紫场景扩散，手／脸中性；每例第96帧，不能代替整窗指标。"]
    gate=protocol.get('last_gate')
    if gate:
        lines+=['',f"场景验收：**{'通过' if gate['passed'] else '未通过'}**。全训练池实际启动：**{bool(protocol.get('full_training_started') or protocol.get('full_training_authorized_by_gate'))}**。"]
        failed=[name for name,check in gate.get('checks',{}).items() if not check['passed']]
        if failed:lines+=['','未通过项：'+', '.join('`'+name+'`' for name in failed)+'.']
    if failure:lines+=['','**进程失败，已停止；此前last.pt／trace保留。** '+failure['traceback'].splitlines()[-1]]
    lines+=['',END]
    current=MASTER.read_text();block='\n'.join(lines)
    if BEGIN in current:current=current[:current.index(BEGIN)]+block+current[current.index(END)+len(END):]
    else:current+='\n\n'+block+'\n'
    temp=MASTER.with_suffix('.tmp');temp.write_text(current);temp.replace(MASTER)


def backup_results():
    """User-authorized repo: compact records only, no weights/generated arrays."""
    root=BASE.parents[3]
    if subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()!='backup/native-scene-contact-head-oct09':
        raise RuntimeError('Backup branch changed; result backup needs review')
    paths=[MASTER]+list(OUT.glob('scene_gate_step*.json'))
    for name in ['protocol.json','bank_audit.json','window_plan.json']:
        if (OUT/name).exists():paths.append(OUT/name)
    for folder in ['candidate','control']:
        for name in ['progress.json','evaluation_latest.json']:
            path=OUT/folder/name
            if path.exists():paths.append(path)
        if folder=='candidate':
            report=read(OUT/folder/'evaluation_latest.json')
            for regime in ['CLEAN','NOISY']:
                for extension in ['png','json']:
                    path=OUT/folder/f"VALIDATION_MESH_{regime}_STEP{report.get('step')}.{extension}"
                    if path.exists():paths.append(path)
    names=[str(p.relative_to(root)) for p in paths]
    subprocess.run(['git','add','-f','--',*names],cwd=root,check=True)
    changed=subprocess.run(['git','diff','--cached','--quiet','--',*names],cwd=root)
    if changed.returncode:
        subprocess.run(['git','commit','--only','-m','Record overnight native30 scene gate and held-out mesh results','--',*names],cwd=root,check=True)
    command=['git','-c','http.version=HTTP/1.1','-c','http.lowSpeedLimit=1024','-c','http.lowSpeedTime=30',
        'push','user-backup','HEAD:backup/native-scene-contact-head-oct09']
    subprocess.run(command,cwd=root,check=True,timeout=120)
    (BASE/'RESULT_BACKUP.json').write_text(json.dumps(dict(status='pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))+'\n')


if __name__=='__main__':
    limit=time.time()+8*3600
    while time.time()<limit:
        update();status=read(OUT/'protocol.json').get('status','')
        if status in ('completed','completed_gate_failed') or (OUT/'failure.json').exists():
            if status in ('completed','completed_gate_failed'):
                try:backup_results()
                except Exception as exc:(BASE/'RESULT_BACKUP.json').write_text(json.dumps(dict(status='failed',reason=repr(exc)))+'\n')
            break
        time.sleep(60)
