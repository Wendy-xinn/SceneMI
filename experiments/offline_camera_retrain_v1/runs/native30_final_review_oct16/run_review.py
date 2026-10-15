"""Sequential, time-bounded matched candidates followed by selected long training.

Only this review's directories are mutated. Original audited code and the
saved parent checkpoint remain intact. No TEST data or tracker is involved.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path.insert(0,str(REPO))
import torch
from experiments.offline_camera_retrain_v1.prepare_native30 import HERE,sha256,write_json,code_hashes
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from selection_policy import choose_candidate

PARENT = ROOT.parent/'native30_gait_v2_ego55k_oct15'
NAMES = ('control','lean','foot','ik_off')


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def equal_json(first,second):
    return json.dumps(first,sort_keys=True,separators=(',',':')) == json.dumps(second,sort_keys=True,separators=(',',':'))


def extensions():
    return {str(ROOT/name):sha256(ROOT/name) for name in
        ('candidate.py','worker.py','run_review.py','selection_policy.py','test_candidate.py','test_selection_policy.py')}


def fork_arms(phase1,phase2):
    checkpoint=PARENT/'training/last.pt'
    state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    parent_protocol=json.loads((PARENT/'training/protocol.json').read_text())
    assert parent_protocol['status']=='stopped'
    parent_step=state['step']
    assert parent_protocol['completed_steps']==parent_step==read_rows(PARENT/'training/trace.jsonl')[-1]['step']
    assert state['planned_steps']==55000 and 'optimizer' in state
    assert state['effective_config']['code_sha256']==code_hashes()
    parent_sha=sha256(checkpoint); extension_sha=extensions()
    manifest=dict(created_at=now(),parent_checkpoint=str(checkpoint),parent_sha256=parent_sha,
        parent_step=parent_step,original_initialization=state.get('initialization'),
        retained_fields=['model','optimizer','step','planned_steps','counts','data_rng','condition_rng','torch_rng','cuda_rng','numpy_rng'],
        phase1_updates=phase1,phase2_updates=phase2,optimizer_reset=False,
        lr_horizon=55000,parameter_shapes_unchanged=True,extension_code_sha256=extension_sha,
        free_disk_before={str(path):shutil.disk_usage(path).free for path in (ROOT,Path('/mnt/e'))})
    assert shutil.disk_usage(ROOT).free > 12*2**30, 'Need space for four atomic resumable candidates'
    write_json(ROOT/'FORK_MANIFEST.json',manifest)
    parent_checks=PARENT/'prepared/input_checks.json'
    assert json.loads(parent_checks.read_text())['status']=='passed'
    for name in NAMES:
        arm=ROOT/'arms'/name
        assert not arm.exists(), 'Review already initialized; do not overwrite candidates'
        (arm/'prepared').mkdir(parents=True);(arm/'training').mkdir()
        config=copy.deepcopy(state['effective_config']['config'])
        config['version']='native30-final-review-'+name+'-oct16'
        config['prepared_output']=str((arm/'prepared').relative_to(HERE))
        config['training_status']='Declared derived stage; same source model/Adam/RNG; sequential matched review then selected continuation.'
        config['candidate']=dict(root_reference_cap_enabled=name!='control',root_rotation_confidence_cap=.05,
            body_local_foot_enabled=name in ('foot','ik_off'),body_local_foot_coefficient=1.,raw_body_local_foot_weight=.5)
        if name!='control':config['physical_guidance']['enabled']=False
        if name=='ik_off':config['refinement']['joint_anchor_gain']=0.
        config['candidate_review']=dict(parent_step=parent_step,arm=name,parent_checkpoint_sha256=parent_sha,
            scope='Derived stage, not fresh training or unchanged resume; no automatic EXO or tracker.',
            phase1_updates=phase1,phase2_updates=phase2,absolute_update_cap=55000)
        config['run_control']['wrapper_sha256']=extension_sha[str(ROOT/'worker.py')]
        effective=copy.deepcopy(state['effective_config']);effective['config']=config
        effective['extension_code_sha256']=extension_sha
        effective['stage_lineage']=dict(parent_checkpoint=str(checkpoint),parent_checkpoint_sha256=parent_sha,
            parent_step=parent_step,arm=name,optimizer_reset=False,all_rng_preserved=True,
            observation_coefficients_source='Fixed parent TRAIN calibration; candidate ratios rechecked, not individually refitted.',
            changes=dict(candidate=config['candidate'],physics_enabled=config['physical_guidance']['enabled'],
                         joint_anchor_gain=config['refinement']['joint_anchor_gain']))
        write_json(arm/'config.json',config)
        write_json(arm/'prepared/effective_config.json',effective)
        write_json(arm/'prepared/input_checks.json',dict(status='pending_candidate_preflight',training_ready=False,
            optimizer_steps=0,data_mode='full',training_started=False,
            inheritance=dict(parent_preparation=str(parent_checks),sha256=sha256(parent_checks),
                scope='Official TRAIN/VAL manifests, source fingerprints, geometry caches, mounts and sampling plan inherited. Trainer revalidates full source digests before updates.'),
            candidate_preflight=dict(status='pending',required='Actual candidate B2 full F/B at t25/300/800, finite gradients, GT-isolated input, unchanged parameters.')))
        initialization=dict(type='forked_stage',original=state.get('initialization'),parent_checkpoint=str(checkpoint),
            parent_checkpoint_sha256=parent_sha,absolute_step_start=parent_step,optimizer_reset=False,
            all_model_parameters_trainable=True)
        derived=dict(state,effective_config=effective,initialization=initialization)
        save_checkpoint(derived,arm/'training/last.pt')
        write_json(arm/'training/protocol.json',dict(status='forked_not_started',completed_steps=parent_step,
            planned_steps=55000,initialization=initialization,effective_config=effective,
            checkpoint_policy='One atomic resumable last.pt; no numbered/best checkpoints.'))
        print('Forked',name,'at',parent_step,flush=True)
    del state
    return manifest


def paired_inputs(names,label,start,end):
    reference=None;reference_identities=None;counts={}
    for name in names:
        arm=ROOT/'arms'/name
        rows=[r for r in read_rows(arm/'input_trace.jsonl') if start<r['step']<=end]
        identities=[dict(step=r['step'],identities=r['identities'],timesteps=r['timesteps'],scene=r['scene'])
            for r in read_rows(arm/'training/trace.jsonl') if start<r['step']<=end]
        assert len(rows)==len(identities)==end-start,(name,len(rows),end-start)
        if reference is None:reference=rows;reference_identities=identities
        else:
            assert equal_json(reference,rows),'Noisy/observed inputs differ: '+name
            assert equal_json(reference_identities,identities),'Sample/timestep/scene differs: '+name
        counts[name]=len(rows)
    report=dict(status='passed',label=label,checked_at=now(),start_step=start,end_step=end,
        counts=counts,fields=['complete sample identities','timesteps','scene dropout','noisy tensor SHA256','trajectory SHA256','observation meta SHA256'])
    write_json(ROOT/(label+'_PAIRING.json'),report)


def report(selection,summaries,manifest,status):
    write_json(ROOT/'FINAL_SELECTION.json',dict(status=status,updated_at=now(),selection=selection,
        summaries=summaries,parent_step=manifest['parent_step'],
        limitations=['Short derived-stage development comparison, not independent TEST or fresh-from-scratch architecture proof.',
            'Practical noninferiority guardrails are not statistical significance. One fixed DDIM seed per window.',
            'Full long-training quality remains to be monitored; original data/Adam provenance preserved.']))
    selected=selection['selected']; lines=[
        '# Native30 修改与长训选择', '',
        f'更新时间（北京时间）：{now()}。状态：{status}。', '',
        f'选择：**{selected}**。所有分支从原模型第 {manifest["parent_step"]} 步的同一权重、Adam、sampler 和完整随机状态开始。',
        '原基线可恢复，原源码与 checkpoint 未覆盖。比较仅使用官方 TRAIN 与此前预选的 VAL，不接 tracker，不自动转 EXO。', '',
        '|分支|绝对步数|全身 clean/noisy mm|步行局部脚位 clean/noisy mm|步行幅度误差 clean/noisy|计算秒/更新|',
        '|---|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        a=s['aggregate']
        def pair(panel,key):return '/'.join(f'{a[panel+"/"+r][key]:.3f}' for r in ('clean','noisy'))
        lines.append(f'|{name}|{s["step"]}|{pair("all","mpjpe_mm")}|{pair("locomotion","body_local_foot_error_mm")}|{pair("locomotion","articulation_log_ratio_error")}|{s["timing"]["mean_compute_s"]:.3f}|')
    lines += ['', '选择规则在实验前冻结：头部、root、全身、脚位、脚速和幅度逐项守卫；质量接近时优先计算较快者。所有候选不合格则保留 control。',
        '这些是短程开发选择，不能保证55k收敛，也不把低脚速等同于正确步态。完整逐窗数据、输入配对证明和配置均在本目录。', '',
        'control：原设置；lean：关闭显式物理＋降低推测root置信度；foot：lean＋raw/final逐脚身体坐标位置项；ik_off：foot＋关闭关节IK，保留头位置锚定。', '',
        '最终配置见 FINAL_LONG_TRAIN_CONFIG.json；运行状态见 STATUS.json 和所选 arms/*/ACTIVE.json。']
    (ROOT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')


def compact_rejected(selected):
    records=[]
    for name in NAMES:
        if name in (selected,'control'):continue
        checkpoint=ROOT/'arms'/name/'training/last.pt'
        state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        before=checkpoint.stat().st_size
        h=hashlib.sha256()
        for key,value in state['model'].items():h.update(key.encode());h.update(value.contiguous().numpy().tobytes())
        digest=h.hexdigest()
        state.pop('optimizer')
        state['evaluation_only']=True;state['optimizer_removed_reason']='Finished rejected candidate; no resume planned. Model and all small RNG/sampler states retained.'
        save_checkpoint(state,checkpoint)
        del state
        kept=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False);h=hashlib.sha256()
        for key,value in kept['model'].items():h.update(key.encode());h.update(value.contiguous().numpy().tobytes())
        assert h.hexdigest()==digest
        records.append(dict(arm=name,model_sha256=digest,model_verified_unchanged=True,before_bytes=before,after_bytes=checkpoint.stat().st_size,evaluation_only=True))
        del kept
    write_json(ROOT/'STORAGE_CLEANUP.json',dict(completed_at=now(),records=records,
        preserved_resumable=['original parent','control',selected],free_disk_after=shutil.disk_usage(ROOT).free))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase1-updates',type=int,default=1000)
    parser.add_argument('--phase2-updates',type=int,default=2000)
    parser.add_argument('--decision-hours',type=float,default=8.)
    parser.add_argument('--final-stop',type=int,default=10000)
    parser.add_argument('--resume-manager',action='store_true')
    parser.add_argument('--initialize-only',action='store_true')
    args=parser.parse_args()
    started=time.monotonic(); deadline=started+args.decision_hours*3600
    assert 0<args.phase1_updates and 0<args.phase2_updates and args.final_stop<=55000
    if args.resume_manager:
        manifest=json.loads((ROOT/'FORK_MANIFEST.json').read_text())
        assert manifest['extension_code_sha256']==extensions(), 'Review code changed since initialization'
        assert manifest['phase1_updates']==args.phase1_updates and manifest['phase2_updates']==args.phase2_updates
    else:
        manifest=fork_arms(args.phase1_updates,args.phase2_updates)
    plan=json.loads((ROOT.parent/'native30_physics_schedule_probe_oct15/VAL_PLAN.json').read_text())
    write_json(ROOT/'VAL_PLAN.json',plan)
    state=dict(status='candidate_review',pid=os.getpid(),started_at=now(),parent_step=manifest['parent_step'],
        phase1_updates=args.phase1_updates,phase2_updates=args.phase2_updates,decision_budget_hours=args.decision_hours,
        automatic_exo=False,automatic_tracker=False,long_train_horizon=55000,
        long_train_review_stop=args.final_stop,original_parent_preserved=True)
    write_json(ROOT/'STATUS.json',state)
    if args.initialize_only:
        state['status']='initialized';write_json(ROOT/'STATUS.json',state)
        return
    child=None
    def run(name,stop,label,final=False):
        nonlocal child
        existing=ROOT/'arms'/name/(label+'_summary.json')
        if existing.exists():
            summary=json.loads(existing.read_text())
            assert summary['step']==stop
            return summary
        if not final and time.monotonic()>deadline:raise TimeoutError('Decision time budget exhausted before '+name)
        command=[sys.executable,'-u',str(ROOT/'worker.py'),'--arm',name,'--stop-step',str(stop),'--label',label]
        if final:command.append('--final')
        state.update(active_arm=name,active_label=label,active_stop=stop,updated_at=now())
        write_json(ROOT/'STATUS.json',state)
        with (ROOT/'arms'/name/(label+'.log')).open('a') as log:
            child=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            state['worker_pid']=child.pid;write_json(ROOT/'STATUS.json',state)
            while child.poll() is None:
                if not final and time.monotonic()>deadline:
                    child.send_signal(signal.SIGTERM)
                    raise TimeoutError('Decision time budget exhausted during '+name+'; graceful save requested')
                time.sleep(5)
            code=child.returncode;child=None
        if code:raise RuntimeError(f'{name}/{label} exited {code}; inspect log')
        return json.loads((ROOT/'arms'/name/(label+'_summary.json')).read_text())
    def stop(signum,frame):
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
        raise InterruptedError('Review manager interrupted; child checkpoint save requested')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        end1=manifest['parent_step']+args.phase1_updates
        summaries={name:run(name,end1,'phase1') for name in NAMES}
        paired_inputs(NAMES,'phase1',manifest['parent_step'],end1)
        choice=choose_candidate(summaries)
        write_json(ROOT/'PHASE1_SELECTION.json',choice)
        eligible=choice['eligible']
        candidate=None
        if eligible:
            best=min(choice['scores'][name] for name in eligible)
            close=[name for name in eligible if choice['scores'][name]<=best+.03]
            candidate=min(close,key=lambda name:summaries[name]['timing']['mean_compute_s'])
        end2=end1+args.phase2_updates
        summaries2={'control':run('control',end2,'phase2')}
        if candidate is not None:
            summaries2[candidate]=run(candidate,end2,'phase2')
            paired_inputs(('control',candidate),'phase2',end1,end2)
        choice2=choose_candidate(summaries2)
        selected=choice2['selected']
        assert sha256(PARENT/'training/last.pt')==manifest['parent_sha256']
        report(choice2,summaries2,manifest,'selected_for_extended_review')
        config=json.loads((ROOT/'arms'/selected/'config.json').read_text())
        write_json(ROOT/'FINAL_LONG_TRAIN_CONFIG.json',config)
        compact_rejected(selected)
        state.update(status='selected_long_training',selected=selected,decision_at=now(),
            decision_elapsed_hours=(time.monotonic()-started)/3600,
            decision='Declared development choice; continue to10k review, not unchecked55k.')
        write_json(ROOT/'STATUS.json',state)
        result=run(selected,args.final_stop,'long_review',final=True)
        write_json(ROOT/'LONG_REVIEW.json',result)
        extended=choose_candidate({'control':summaries2[selected],'extended':result})
        continue_long=('extended' in extended['eligible'] and extended['scores']['extended']<=.98*extended['scores']['control'])
        write_json(ROOT/'EXTENDED_REVIEW_DECISION.json',dict(checked_at=now(),continue_to55000=continue_long,
            rule='Pass the same noninferiority guards and improve composite development quality by at least2% versus the selected stage2 endpoint. Practical time-budget rule, not a statistical convergence test.',comparison=extended))
        with (ROOT/'REPORT_zh.md').open('a') as stream:
            stream.write(f'\n所选版本已到 {result["step"]} 步并完成18条件＋固定窗scene-OFF验证，可恢复checkpoint已保存。55k最终计划配置已备妥，尚未宣称质量通过。\n')
            stream.write('10k继续规则：相对所选阶段2端点通过相同非劣守卫，且复合质量改善至少2%；达标自动继续55k，否则保留10k等待质量复核。\n')
        if continue_long:
            state.update(status='long_training_to55000',review_ready_at=now(),completed_steps=result['step'])
            write_json(ROOT/'STATUS.json',state)
            completed=run(selected,55000,'final55k',final=True)
            state.update(status='completed55000',completed_at=now(),completed_steps=completed['step'])
        else:
            state.update(status='ready_for_user_review',completed_at=now(),completed_steps=result['step'],
                hold_reason='10k development continuation rule did not pass; full resumable state retained.')
        write_json(ROOT/'STATUS.json',state)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                state['child_save_pending']=True
            state['child_exit_code']=child.poll()
        state.update(status='failed',failed_at=now(),error=repr(exc));write_json(ROOT/'STATUS.json',state)
        (ROOT/'FAILURE_zh.md').write_text('候选审查未完成：'+repr(exc)+'\n原基线checkpoint仍完整可恢复；见 STATUS.json 和各分支日志。\n')
        if not isinstance(exc,InterruptedError) and (child is None or child.poll() is not None):
            # Preserve useful overnight progress even if an isolated candidate
            # fails. This uses the unchanged, audited original wrapper and is
            # capped at its10k review boundary rather than unattended55k.
            with (ROOT/'fallback_baseline.log').open('a') as log:
                fallback=subprocess.Popen([sys.executable,'-u',str(PARENT/'run_training.py'),
                    '--resume','--review-step','10000'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(fallback_baseline_pid=fallback.pid,fallback_review_step=10000)
            write_json(ROOT/'STATUS.json',state)
        raise


if __name__=='__main__':
    main()
