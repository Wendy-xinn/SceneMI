"""Explicitly bounded fresh V2 small-pool learnability/schedule experiment.

This is an overfit diagnostic, not the formal dataset sampler or 55k schedule.
The formal preflight-only config remains held. No trained checkpoint is loaded.
"""
import copy
import datetime
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.prepare_native30 import (
    HERE,read_config,make_model,code_hashes,write_json,window_options,sha256,
)
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_support_cache import cached_support
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs,training_motion,sample_motion
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d
from experiments.offline_camera_retrain_v1.train_native30 import model_digest,StopAtUpdateBoundary
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint

ROOT=Path(__file__).resolve().parent
UPDATES=128
MAX_SECONDS=3600
INITIAL='9d14dbbb56e6c092ca2c0f784e4f3935ad0a0cf6365cc35bf3894fe07dc90d69'
ARMS=dict(current=(True,.25),physics_off=(False,.25),late_four=(True,.9),late_two=(True,.98))
spec=importlib.util.spec_from_file_location('schedule_metrics',ROOT.parent/'native30_physics_schedule_probe_oct15/run_probe.py')
metric_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric_module)


def now():return datetime.datetime.now().astimezone().isoformat()


def dataset(c,split):
    return Native30Data(HERE/c['train_manifest'],split,c['seed'],frames=c['frames'],
        memory_protocol=c['scene']['memory_protocol'],quality_config=c['gt_reliability'],
        exo_max_empty_frames=c['exo_max_empty_frames'],**window_options(c))


def select_pool(data):
    selected=[];screened=[]
    for group in ('trumans','camera_wearer','rich'):
        candidates=sorted([i for i,(m,_) in enumerate(data.members[group]) if 'ego' in m.get('allowed_modes',[]) and len(data.starts_for(group,i,'ego'))],
            key=lambda i:data.members[group][i][0]['sequence_id'])
        eligible=[]
        for ix in np.linspace(0,len(candidates)-1,min(24,len(candidates)),dtype=int):
            member=candidates[ix];starts=data.starts_for(group,member,'ego');start=int(starts[len(starts)//2])
            sample,identity=data.sample(group,'ego',member=member,start=start,yaw=.6)
            m=torch.from_numpy(sample['motion'])[None];r=torch.from_numpy(sample['rest'])[None]
            with torch.no_grad():
                j=forward_kinematics(m,r)[0];rot=rotation_from_6d(m[0,:,3:9])
                local=torch.einsum('tji,tkj->tki',rot,j-j[:,:1])[:,(10,11)]
                amp=float((local-local.mean(0)).square().sum(-1).mean().sqrt())
                net=float((j[-1,0]-j[0,0]).norm())
                z=local[:,:,2].numpy();corr=float(np.corrcoef(z.T)[0,1]) if min(z.std(0))>1e-6 else 1.
                stance=float(((j[:,(10,11)].diff(dim=0)*30).norm(dim=-1)<.2).float().mean())
            row=dict(group=group,member=member,start=start,sequence_id=identity['sequence_id'],
                GT_root_net_m=net,GT_body_local_feet_RMS_m=amp,GT_foot_Z_correlation=corr,GT_stance_fraction=stance)
            row['eligible']=net>=.6 and amp>=.06 and corr<=-.1 and .1<=stance<=.9
            screened.append(row)
            if row['eligible']:eligible.append(row)
        eligible.sort(key=lambda r:(-r['GT_root_net_m'],r['sequence_id']))
        assert len(eligible)>=2, ('Insufficient locomotion TRAIN candidates',group,len(eligible))
        selected.extend(eligible[:2])
    write_json(ROOT/'TRAIN_POOL.json',dict(selection='GT-only 24 evenly spaced TRAIN sequences/group; middle legal window; root net>=.6m, feet RMS>=.06m, sagittal LR correlation<=-.1, stance fraction .1.. .9; top2 root displacement/group; no prediction-based selection.',
        selected=selected,screened=screened,small_pool_not_generalization=True))
    return selected


def main():
    start=time.monotonic();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    c=read_config(HERE/'native30_fresh_ego_gait_v2_config.json')
    sealed=json.loads((HERE/c['prepared_output']/'effective_config.json').read_text())
    checks=json.loads((HERE/c['prepared_output']/'input_checks.json').read_text())
    assert c==sealed['config'] and checks['engineering_ready'] and c['preflight_only']
    assert code_hashes()==sealed['code_sha256']
    assert sha256(HERE/c['mount_defaults'])==sealed['mount_defaults_sha256']
    assert shutil.disk_usage(ROOT).free>9*2**30
    protocol=dict(status='preparing',started_at=now(),pid=os.getpid(),optimizer_updates_per_arm=UPDATES,
        max_seconds=MAX_SECONDS,initialization='original seed, strictly from scratch; no trained weights or Adam loaded',
        expected_initial_sha256=INITIAL,formal_training_started=False,no_automatic_continuation=True,
        user_scope='Authorized bounded pilot; formal preflight-only/55k gate remains held.',
        objective='Same V2 for all four arms; only physical enabled/minimum_alpha changes.',
        sampling='Six GT-selected official TRAIN locomotion windows, clean+forced-noisy once; fixed yaw .6; paired shuffled cycles of 12 inputs; not formal corpus coverage.',
        optimizer='Fresh AdamW per arm; explicit overfit diagnostic LR1e-4, 8-step linear warmup then constant; not the formal 500-step/cosine schedule.',
        code_sha256=code_hashes(),runner_sha256=sha256(Path(__file__)),metric_runner_sha256=sha256(Path(metric_module.__file__)),
        data_audit='Reuses unchanged full TRAIN/VAL source hashes from V2 engineering preflight; no TEST.',
        disk_free_before={str(p):shutil.disk_usage(p).free for p in (ROOT,Path('/mnt/e'))},arms={})
    write_json(ROOT/'PROTOCOL.json',protocol)
    print('preparing TRAIN small locomotion pool',flush=True)
    data=dataset(c,'train');selected=select_pool(data)
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/c['mount_defaults']).read_text()),c['observation']['input_noise'])
    items=[];identities=[]
    for case,e in enumerate(selected):
        sample,identity=data.sample(e['group'],'ego',member=e['member'],start=e['start'],yaw=.6)
        for regime in ('clean','noisy'):
            b=builder if regime=='clean' else Native30ObservationBuilder(builder.skin_cache,builder.mounts,dict(builder.noise_config,clean_probability=0.))
            observed,scene,asset=b.build(data,sample,identity,'ego',(15,),noise_rng=np.random.default_rng(c['seed']+case) if regime=='noisy' else None)
            observed=cached_support(data,builder,observed,identity)
            items.append((observed,scene,asset,'ego'));identities.append(dict(case=case,regime=regime,**e))
    assert data.support_cache.misses==0, 'Unexpected source/support cache changes'
    rng=np.random.default_rng(c['seed']+311)
    order=[]
    while len(order)<UPDATES*2:order.extend(rng.permutation(len(items)).tolist())
    batches=[order[2*i:2*i+2] for i in range(UPDATES)]
    write_json(ROOT/'BATCH_PLAN.json',dict(batches=batches,items=identities,seed=c['seed']+311))
    # Reuse the same explicit input batches; all body-dependent queries stay online.
    cached_batches={}
    for ids in {tuple(b) for b in batches}:
        batch=collate_observations([items[i] for i in ids]);known=model_inputs(c,batch)
        cached_batches[ids]=(batch,known)
    train_eval=[collate_observations([items[4*i]]) for i in range(3)]
    val=dataset(c,'validation')
    fixed=json.loads((ROOT.parent/'native30_physics_schedule_probe_oct15/VAL_PLAN.json').read_text())['plan'][:3]
    val_eval=[]
    for e in fixed:
        sample,identity=val.sample(e['group'],'ego',member=e['member'],start=e['start'],yaw=0)
        for regime in ('clean','noisy'):
            b=builder if regime=='clean' else Native30ObservationBuilder(builder.skin_cache,builder.mounts,dict(builder.noise_config,clean_probability=0.))
            observed,scene,asset=b.build(val,sample,identity,'ego',(15,),noise_rng=np.random.default_rng(e['case_seed']) if regime=='noisy' else None)
            val_eval.append((e,regime,collate_observations([(observed,scene,asset,'ego')])))
    del items,data,val;gc.collect()
    protocol['status']='running';protocol['support_cache_misses']=0
    write_json(ROOT/'PROTOCOL.json',protocol)
    for name,(enabled,minimum) in ARMS.items():
        assert time.monotonic()-start<MAX_SECONDS
        folder=ROOT/name;folder.mkdir(exist_ok=False)
        cfg=copy.deepcopy(c);cfg['physical_guidance'].update(enabled=enabled,minimum_alpha=minimum)
        write_json(folder/'config.json',cfg)
        torch.manual_seed(c['seed']);np.random.seed(c['seed'])
        model=make_model(cfg).cuda()
        assert model_digest(model)==INITIAL
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=c['optimizer']['weight_decay'])
        assert not optimizer.state
        history=[];evaluations=[]
        def checkpoint(step):
            save_checkpoint(dict(step=step,planned_steps=UPDATES,model=model.state_dict(),optimizer=optimizer.state_dict(),
                initialization=dict(type='from_scratch',source_checkpoint=None,seed=c['seed'],initial_model_sha256=INITIAL,optimizer_reset=True),
                effective_config=dict(config=cfg,observation_coefficients=sealed['observation_coefficients'],code_sha256=sealed['code_sha256'],data_sha256=sealed['data_sha256']),
                pilot_protocol=dict(kind='paired_small_pool_overfit',batch_plan='BATCH_PLAN.json',planned_updates=UPDATES,lr='8-step warmup then1e-4'),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state()),folder/'last.pt')
        def evaluate(label):
            model.eval();arrays={}
            with torch.no_grad():
                for i,batch in enumerate(train_eval):
                    known=model_inputs(cfg,batch);seed=c['seed']+900+i
                    prediction=sample_motion(model,known,cfg,steps=20,seed=seed)
                    evaluations.append(dict(label=label,split='TRAIN_selected',case=i,regime='clean',metrics=metric_module.metrics(prediction,batch)))
                    arrays[f'TRAIN_{i}_prediction']=prediction[0].cpu().numpy();arrays[f'TRAIN_{i}_truth']=batch['motion'][0].cpu().numpy();arrays[f'TRAIN_{i}_rest']=batch['rest'][0].cpu().numpy()
                # VAL is assessed only at the end, never used for updates/pool choice.
                if label=='final':
                    for i,(entry,regime,batch) in enumerate(val_eval):
                        prediction=sample_motion(model,model_inputs(cfg,batch),cfg,steps=20,seed=entry['case_seed'])
                        evaluations.append(dict(label=label,split='VAL_fixed',case=i//2,regime=regime,metrics=metric_module.metrics(prediction,batch)))
                        arrays[f'VAL_{i}_prediction']=prediction[0].cpu().numpy();arrays[f'VAL_{i}_truth']=batch['motion'][0].cpu().numpy();arrays[f'VAL_{i}_rest']=batch['rest'][0].cpu().numpy()
            np.savez_compressed(folder/f'{label}_predictions.npz',**arrays)
            write_json(folder/'evaluation.json',dict(cases=evaluations,scope='TRAIN overfit learning plus small existing fixed VAL; not a final benchmark'))
        evaluate('initial')
        arm_start=time.monotonic()
        with StopAtUpdateBoundary() as stop,(folder/'trace.jsonl').open('w') as stream:
            for step,indices in enumerate(batches,1):
                assert time.monotonic()-start<MAX_SECONDS
                tick=time.monotonic();batch,known=cached_batches[tuple(indices)]
                # Seed each update independently: model/solver numerical paths cannot shift later draws.
                torch.manual_seed(c['seed']+10000+step)
                times=torch.randint(1000,(2,),device='cuda')
                low=torch.rand(2,device='cuda')<c['noise']['extra_low_noise_probability']
                times=torch.where(low,torch.randint(c['noise']['low_noise_steps'],times.shape,device='cuda'),times)
                signal=model.alphas[times];target=training_motion(cfg,batch,known)
                noisy=signal[:,None,None].sqrt()*target+(1-signal[:,None,None]).sqrt()*torch.randn_like(target)
                scene_on=bool(np.random.default_rng(c['seed']+step).random()>=c['scene']['dropout_probability'])
                rate=1e-4*min(step/8,1.)
                for group in optimizer.param_groups:group['lr']=rate
                model.train();optimizer.zero_grad(set_to_none=True)
                prediction,details=model(noisy,times,known,use_scene=scene_on,return_details=True)
                loss,terms=objective(prediction,batch['motion'],batch,signal,details,cfg,sealed['observation_coefficients'])
                if step==1:
                    gradient=torch.autograd.grad(terms['base']['gait_raw_total'],details['raw_motion'],retain_graph=True)[0]
                    legs=[3+6*j+k for j in (1,2,4,5,7,8,10,11) for k in range(6)]
                    audit=dict(raw_gait_leg_rotation_gradient_norm=float(gradient[...,legs].norm()),finite=bool(torch.isfinite(gradient).all()))
                    assert audit['finite'] and audit['raw_gait_leg_rotation_gradient_norm']>0
                    write_json(folder/'RAW_GAIT_GRADIENT.json',audit)
                assert torch.isfinite(loss)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['optimizer']['clip_norm'],error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize()
                row=dict(step=step,indices=indices,timesteps=times.tolist(),scene=scene_on,lr=rate,loss=float(loss.detach()),gradient_norm=float(norm),seconds=time.monotonic()-tick,
                    physical_accepted_steps=details.get('physical_accepted_steps',torch.zeros(2)).tolist(),
                    terms={group:{k:float(v.detach()) for k,v in values.items()} for group,values in terms.items()})
                history.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if step%16==0:print(name,step,'loss',round(row['loss'],4),'seconds',round(row['seconds'],3),flush=True)
                if step%64==0 or stop.requested:checkpoint(step)
                if stop.requested:raise InterruptedError('Stopped after completed update; resumable last.pt preserved')
        checkpoint(UPDATES)
        train_seconds=time.monotonic()-arm_start
        evaluate('final')
        protocol['arms'][name]=dict(status='completed',steps=UPDATES,initial_sha256=INITIAL,train_seconds=train_seconds,
            mean_update_seconds=float(np.mean([r['seconds'] for r in history])),checkpoint=str(folder/'last.pt'),optimizer_reset=True)
        write_json(ROOT/'PROTOCOL.json',protocol)
        del optimizer,model,prediction,details,loss,terms,batch,known,gradient;gc.collect();torch.cuda.empty_cache()
    traces={name:[json.loads(line) for line in (ROOT/name/'trace.jsonl').read_text().splitlines()] for name in ARMS}
    for name in ARMS:
        assert len(traces[name])==UPDATES
        for a,b in zip(traces['current'],traces[name]):
            assert all(a[k]==b[k] for k in ('step','indices','timesteps','scene','lr'))
    summary={}
    for name in ARMS:
        cases=json.loads((ROOT/name/'evaluation.json').read_text())['cases'];summary[name]={}
        for label,split,regime in (('initial','TRAIN_selected','clean'),('final','TRAIN_selected','clean'),('final','VAL_fixed','clean'),('final','VAL_fixed','noisy')):
            group=[r['metrics'] for r in cases if r['label']==label and r['split']==split and r['regime']==regime]
            values={k:float(np.mean([r[k] for r in group if r[k] is not None])) for k,v in group[0].items() if isinstance(v,(float,int))}
            summary[name]['/'.join((label,split,regime))]=values
    # Triage gate deliberately requires actual DDIM feet learning, not loss decline.
    gates={}
    for name in ARMS:
        initial=summary[name]['initial/TRAIN_selected/clean'];final=summary[name]['final/TRAIN_selected/clean']
        gates[name]=dict(body_local_foot_error_improves20pct=final['body_local_foot_error_mm']<initial['body_local_foot_error_mm']*.8,
            foot_velocity_error_improves10pct=final['foot_velocity_RMSE_m_s']<initial['foot_velocity_RMSE_m_s']*.9,
            amplitude_log_ratio_error_improves20pct=final['articulation_log_ratio_error']<initial['articulation_log_ratio_error']*.8)
    write_json(ROOT/'RESULTS.json',dict(status='completed_bounded_fresh_small_pool',updates_per_arm=UPDATES,
        all_sampling_noise_and_lr_draws_paired=True,initial_sha256=INITIAL,summary=summary,TRAIN_learnability_triage_gates=gates,
        limits='128 updates on six TRAIN windows is an overfit diagnostic. Fixed VAL is developmental. Neither pass nor failure establishes paper performance or rules out longer training.',
        long_training_started=False,no_automatic_continuation=True))
    protocol.update(status='completed',completed_at=now(),elapsed_s=time.monotonic()-start,all_draws_paired=True,
        disk_free_after={str(p):shutil.disk_usage(p).free for p in (ROOT,Path('/mnt/e'))})
    write_json(ROOT/'PROTOCOL.json',protocol);print('ALL FRESH PILOT ARMS COMPLETED',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        p=ROOT/'PROTOCOL.json';r=json.loads(p.read_text()) if p.exists() else {}
        r.update(status='failed_or_bounded_stop',error=repr(exc),failed_at=now());write_json(p,r)
        raise
