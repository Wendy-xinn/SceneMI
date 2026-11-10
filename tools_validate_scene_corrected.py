"""Balanced validation of the corrected scene branch against its replay parent."""
import copy, gc, json, os, time
from collections import OrderedDict, defaultdict
from pathlib import Path
import numpy as np, torch, yaml
from scenemi.config import ROOT
from scenemi.training import deterministic, now, tensor_digest, ensure_bounded
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.checked_scene_dataset import SceneSelection, CheckedLocalSceneDataset
from scenemi.data.local_scene import inference_payload as local_payload
from scenemi.data.exo_initialization import repair_payload
from scenemi.native_joint_data import ego_payload
from scenemi.observations import public_observations
from scenemi.exo_denoise import sample_denoising
from scenemi import scene_joint_training as corrected_training
from scenemi import scene_joint_replay as historical
from scenemi.replay.visualization import load_model as replay_model
from scenemi.metrics_batch import metric_rows, repair_rows
from scenemi.replay.scene_comparison import extra_metrics
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

OUT=ROOT/'runs/scene_joint128_conditioning_corrected/validation_step_003000_balanced'
CFG_PATH=ROOT/'configs/scene_joint128_conditioning_hybrid.yaml'
CHECKPOINT=ROOT/'runs/scene_joint128_conditioning_corrected/training/last.pt'
BASELINE_CHECKPOINT=ROOT/'runs/scene_joint128_replay/candidates/step_002000.pt'
BASELINE_CFG=ROOT/'configs/scene_joint128_replay.yaml'
MANIFEST=ROOT/'runs/scene_joint128_replay/VALIDATION_CASES.json'
SELECTION=ROOT/'configs/scene_data_current.json'
SEED=777
PER_GROUP=int(os.environ.get('SCENEMI_PER_GROUP','20'))

def make_jobs(manifest):
    expected={'ego_egobody':401,'ego_trumans':540,'ego_grail':2530,'exo_egobody':470,'exo_trumans':540,'exo_grail':2530}
    jobs=[]
    for mode in ('ego','exo'):
      for source in ('egobody','trumans','grail'):
        cases=[c for c in manifest['cases'] if c['mode']==mode and c.get('source')==source]
        if len(cases)!=expected[f'{mode}_{source}']: raise RuntimeError('validation coverage changed')
        take=np.linspace(0,len(cases)-1,min(PER_GROUP,len(cases)),dtype=int)
        chosen=[cases[i] for i in sorted(set(take))]
        for condition in (('ego',) if mode=='ego' else ('identity','tracking','occlusion','gap')):
          for c in chosen: jobs.append(dict(job_id=len(jobs),mode=mode,source=source,condition=condition,case=c))
    return jobs

def restricted(mode,cases,model,cfg,bundle):
    checked={r['path']:r for r in bundle.records(mode,'validation')}; records={}
    for c in cases:
      if c['path'] not in checked: raise RuntimeError('case absent from current selection')
      records[int(c['member'])]=checked[c['path']]
    common=model.base.shape.rest_template[:1].detach().cpu().numpy()
    if mode=='ego':
      d=object.__new__(CheckedLocalSceneDataset)
      d.__dict__.update(split='validation',config=cfg,records=records,common_pelvis=common,paths=bundle.paths,selection_bundle=bundle,bundle=bundle,metadata_cache=bundle.metadata_cache,array_cache=OrderedDict(),prefix_cache=OrderedDict())
    else:
      d=object.__new__(corrected_training.CorrectedSceneExoDataset)
      d.__dict__.update(split='validation',config=dict(cfg,selection=cfg['paths']['selection']),records=records,common_pelvis=common,paths=bundle.paths,selection_bundle=bundle,bundle=bundle,metadata_cache=bundle.metadata_cache,array_cache=OrderedDict(),prefix_cache=OrderedDict(),bank=copy.deepcopy(model.base.native_body.bank).cpu().eval())
    return d

def ego_points(base,motion,payload):
    shape=base.predicted_body(public_observations(payload),payload['body_format'])
    mask=payload['sensor_profile']==1
    if bool(mask.any()): shape['head_camera_translation'][mask]=base.pv_translation
    return forward_kinematics(motion,shape['rest'])

def predict(scene,base,batch,mode):
    if mode=='ego':
      sp=local_payload(batch); bp=ego_payload(batch)
      sm=sample_native30(scene.ego,sp,steps=20,seed=SEED,use_scene=True)
      bm=sample_native30(base.base,bp,steps=20,seed=SEED,use_scene=False)
      return ego_points(base.base,bm,bp),ego_points(scene.base,sm,sp),None
    bp=repair_payload(batch); sp=dict(bp); sp.update(local_payload(batch))
    sm=sample_denoising(scene.exo,sp,steps=8,start_timestep=200,seed=SEED)
    bm=sample_denoising(base,bp,steps=8,start_timestep=200,seed=SEED)
    rest=batch['initial_rest']
    return forward_kinematics(bm,rest),forward_kinematics(sm,rest),forward_kinematics(batch['initial_motion'],rest)

def get_metrics(points,truth,initial,batch,mode,condition):
    if mode=='exo':
      visible=(batch['joint_visible']>0)&(batch['estimate_available'][...,None]>0)
      rows=repair_rows(points,truth,initial,visible,condition); avail=batch['estimate_available'][0]>0
    else: rows=metric_rows(points,truth); avail=None
    rows[0].update(extra_metrics(points[0],truth[0],avail)); return rows[0]

def aggregate(rows):
    out={};
    for mode in ('ego','exo'):
      for source in ('egobody','trumans','grail'):
       for condition in (('ego',) if mode=='ego' else ('identity','tracking','occlusion','gap')):
        key=f'{mode}|{source}|{condition}'; subset=[r for r in rows if r['mode']==mode and r['source']==source and r['condition']==condition]
        for variant in ('baseline','scene'):
          vals=[r[variant] for r in subset]; fam=defaultdict(list)
          for r,v in zip(subset,vals): fam[r['family']].append(v)
          names=vals[0].keys()
          out[f'{key}|{variant}']={'windows':len(vals),'families':len(fam),'window_mean':{n:float(np.mean([v[n] for v in vals if v[n] is not None])) if any(v[n] is not None for v in vals) else None for n in names},'family_macro_mean':{n:float(np.mean([np.mean([v[n] for v in fs if v[n] is not None]) for fs in fam.values() if any(v[n] is not None for v in fs)])) if any(any(v[n] is not None for v in fs) for fs in fam.values()) else None for n in names}}
    diff={}
    for key in sorted({k.rsplit('|',1)[0] for k in out}):
      b=out[key+'|baseline']['family_macro_mean']; s=out[key+'|scene']['family_macro_mean']; diff[key]={n:(s[n]-b[n] if s[n] is not None and b[n] is not None else None) for n in b}
    return {'groups':out,'scene_minus_baseline':diff}

def main():
    ensure_bounded(); OUT.mkdir(parents=True,exist_ok=True); cfg=yaml.safe_load(CFG_PATH.read_text()); manifest=json.loads(MANIFEST.read_text()); jobs=make_jobs(manifest)
    protocol={'protocol':'corrected-scene-step3000-balanced-validation-v1','checkpoint':str(CHECKPOINT.relative_to(ROOT)),'checkpoint_sha256':sha256(CHECKPOINT),'step':3000,'parent_checkpoint':str(BASELINE_CHECKPOINT.relative_to(ROOT)),'parent_checkpoint_sha256':sha256(BASELINE_CHECKPOINT),'config':str(CFG_PATH.relative_to(ROOT)),'config_sha256':sha256(CFG_PATH),'manifest':str(MANIFEST.relative_to(ROOT)),'manifest_sha256':sha256(MANIFEST),'selection_sha256':sha256(SELECTION),'seed':SEED,'ego_ddim_steps':20,'exo_ddim_steps':8,'exo_start_timestep':200,'test_used':False,'full_validation':False,'balanced_per_dataset_mode_condition':PER_GROUP,'jobs':len(jobs),'paired_input':'same sampled current corrected batch; EXO export_normalized_no_extra_scale'}
    if (OUT/'PROTOCOL.json').exists() and json.loads((OUT/'PROTOCOL.json').read_text())!=protocol: raise RuntimeError('protocol changed')
    atomic_json(OUT/'PROTOCOL.json',protocol); rf=OUT/'rows.jsonl'; rows=[json.loads(x) for x in rf.read_text().splitlines() if x.strip()] if rf.exists() else []
    if [r['job_id'] for r in rows]!=list(range(len(rows))): raise RuntimeError('partial rows invalid')
    deterministic(SEED,'cuda'); print('loading current scene checkpoint',flush=True)
    scene=historical.make_model(cfg,'cuda'); state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False,mmap=True); assert state['step']==3000; historical.load_adapter_state(scene,state['adapter']); scene.requires_grad_(False).eval().cuda(); del state
    print('loading parent no-scene checkpoint',flush=True); base_cfg=yaml.safe_load(BASELINE_CFG.read_text()); base=replay_model(BASELINE_CHECKPOINT.parent.parent,base_cfg).cuda().eval(); base.requires_grad_(False)
    bundle=SceneSelection(SELECTION); valid=[c for c in manifest['cases'] if c.get('source') in ('egobody','trumans','grail')]; data={m:restricted(m,[c for c in valid if c['mode']==m],scene,cfg,bundle) for m in ('ego','exo')}; print('ready jobs',len(jobs),flush=True)
    started=time.monotonic()
    try:
      with rf.open('a') as stream:
       for job in jobs[len(rows):]:
        c=job['case']; tick=time.monotonic(); rng=np.random.default_rng(SEED+c['case_id'])
        pair=(data[job['mode']].sample(rng,member=c['member'],start=c['start'],yaw=0.) if job['mode']=='ego' else data['exo'].sample(rng,cfg['frontend'],condition=job['condition'],member=c['member'],start=c['start'],yaw=0.))
        batch=collate([pair[0]],'cuda'); truth=forward_kinematics(batch['motion'],batch['target_rest']); bp,sp,initial=predict(scene,base,batch,job['mode']); bm=get_metrics(bp,truth,initial,batch,job['mode'],job['condition']); sm=get_metrics(sp,truth,initial,batch,job['mode'],job['condition'])
        row={'job_id':job['job_id'],'mode':job['mode'],'source':job['source'],'condition':job['condition'],'family':c['family'],'case_id':c['case_id'],'member':c['member'],'start':c['start'],'identity':pair[1],'baseline':bm,'scene':sm,'seconds':time.monotonic()-tick,'input_sha256':tensor_digest(ego_payload(batch) if job['mode']=='ego' else repair_payload(batch))}; rows.append(row); stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush(); done=len(rows)
        if done%10==0 or done==1: atomic_json(OUT/'STATUS.json',{'state':'running','completed_pairs':done,'total_pairs':len(jobs),'updated_at_utc':now(),'pid':os.getpid()}); print(json.dumps({'completed':done,'total':len(jobs),'seconds':row['seconds'],'mode':job['mode'],'source':job['source'],'condition':job['condition'],'baseline_mpjpe':bm['mpjpe_mm'],'scene_mpjpe':sm['mpjpe_mm']}),flush=True)
        del batch,bp,sp,truth
      result={'state':'complete','step':3000,'checkpoint_sha256':protocol['checkpoint_sha256'],'parent_checkpoint_sha256':protocol['parent_checkpoint_sha256'],'completed_pairs':len(rows),'total_pairs':len(jobs),'full_validation':False,'test_used':False,'elapsed_seconds':time.monotonic()-started,'aggregate':aggregate(rows),'completed_at_utc':now()}; atomic_json(OUT/'RESULT.json',result); atomic_json(OUT/'STATUS.json',{'state':'complete','completed_pairs':len(rows),'total_pairs':len(jobs),'elapsed_seconds':result['elapsed_seconds'],'updated_at_utc':now()});
      lines=['# 修正版场景 step 3000 均衡快速验证','','固定 step 3000 checkpoint，与父 replay step 2000 无场景模型逐窗、同 seed、同输入对照；每个数据集×模式×EXO 条件抽取固定窗口。','', '| 分组 | 窗口 | baseline MPJPE | scene MPJPE | scene-baseline | baseline root | scene root |','|---|---:|---:|---:|---:|---:|---:|']
      for key,d in sorted(result['aggregate']['scene_minus_baseline'].items()):
       b=result['aggregate']['groups'][key+'|baseline']['family_macro_mean']; s=result['aggregate']['groups'][key+'|scene']['family_macro_mean']; lines.append(f"| {key} | {result['aggregate']['groups'][key+'|baseline']['windows']} | {b['mpjpe_mm']:.2f} | {s['mpjpe_mm']:.2f} | {d['mpjpe_mm']:.2f} | {b['root_error_mm']:.2f} | {s['root_error_mm']:.2f} |")
      lines += ['', '单位：mm；负的 scene-baseline 表示场景模型误差更低。该结果是均衡快速验证，不是完整 17,631 对全量验证，也不使用 test。']; (OUT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n'); print(json.dumps({'state':'complete','pairs':len(rows),'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    except BaseException as e: atomic_json(OUT/'STATUS.json',{'state':'failed','completed_pairs':len(rows),'total_pairs':len(jobs),'error':repr(e),'updated_at_utc':now()}); raise
    finally: del scene,base; gc.collect(); torch.cuda.empty_cache()
if __name__=='__main__': main()
