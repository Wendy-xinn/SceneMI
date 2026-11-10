"""Checked three-corpus EXO scene-free adaptation after AMASS repair passes.

Compose the accepted repair engine with native data and dataset-balanced
evaluation. The running AMASS files and prior sources remain immutable.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

from scenemi import exo_denoise_training as engine
from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.exo_initialization import CONDITIONS, repair_payload
from scenemi.data.native_repair import NativeRepairDataset
from scenemi.exo_denoise import DenoisingExoPrior, sample_denoising
from scenemi.exo_training import INDEX, make_model, settings
from scenemi.metrics import motion_metrics
from scenemi.parallel_resources import configure_cuda
from scenemi.selection import validation_manifest
from scenemi.training import contract, deterministic, ensure_bounded, now, rng_state, restore_rng
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

CONFIG = ROOT/'configs/scene_exo_repair.yaml'


class StageDataset(NativeRepairDataset):
    def __init__(self, config, split):
        super().__init__(split=split, selection=ROOT/config['selection'])
        canonical = self.bundle.summary(self.records)
        retained=[]; counts=defaultdict(lambda:dict(source_sequences=0, eligible_sequences=0,
            source_starts=0, eligible_starts=0))
        minimum = config['data']['minimum_tracked_frames_per_window']
        if minimum < config['frontend']['gap_frames'][1]+2:
            raise ValueError('A synthetic gap must leave two tracked frames')
        for record in self.records:
            meta=self.bundle.metadata(record,split)
            valid=np.asarray(self.bundle.array(meta,'exo_position_available')).any(-1)
            prefix=np.r_[0,np.cumsum(valid)]
            starts=record['starts']; active=starts[prefix[starts+128]-prefix[starts]>=minimum]
            group=counts[record['dataset']]; group['source_sequences']+=1; group['source_starts']+=len(starts)
            if len(active):
                retained.append(dict(record,starts=active)); group['eligible_sequences']+=1
                group['eligible_starts']+=len(active)
        self.records=retained; self.groups={key:defaultdict(list) for key in counts}
        for i,row in enumerate(retained):
            self.groups[row['dataset']][row['family']].append(i)
        self.families={key:sorted(value) for key,value in self.groups.items()}
        if len(counts)!=3 or any(not f for f in self.families.values()):
            raise ValueError('All three requested corpora need valid initializers')
        self.eligibility=dict(canonical_source_counts=canonical,repair_initializer_counts=dict(counts),
            minimum_tracked_frames_per_window=minimum,additional_filter='estimated initializer availability only',
            target_quality_ranges_unchanged=True,selection_sha256=sha256(ROOT/config['selection']))

    def validation_cases(self):
        cases=[]
        for member,row in enumerate(self.records):
            starts=row['starts']; breaks=np.r_[0,np.flatnonzero(np.diff(starts)>1)+1,len(starts)]
            selected=[]
            for lo,hi in zip(breaks[:-1],breaks[1:]):
                first,last=int(starts[lo]),int(starts[hi-1])
                selected.extend(range(first,last+1,128))
                if selected[-1]!=last:selected.append(last)
            for start in sorted(set(selected)):
                cases.append(dict(case_id=len(cases),member=member,start=start,dataset=row['dataset'],family=row['family']))
        return cases


def batch_for(data,rng,config,device,forced=False):
    pairs=[]
    for i in range(config['training']['batch_size']):
        kwargs=dict(condition=CONDITIONS[i%4]) if forced else {}
        if forced:
            dataset=('egobody','trumans','grail')[i%3]
            kwargs['member']=min(index for choices in data.groups[dataset].values() for index in choices)
        pairs.append(data.sample(rng,config['frontend'],**kwargs))
    return collate([sample for sample,_ in pairs],device),[identity for _,identity in pairs]


def aggregate(rows):
    by_dataset={dataset:engine.shared.aggregate_original([r for r in rows if r['identity']['dataset']==dataset])
                for dataset in sorted({r['identity']['dataset'] for r in rows})}
    values={field:float(np.mean([v['family_macro_mean'][field] for v in by_dataset.values()
                                if v['family_macro_mean'][field] is not None]))
            for field in rows[0]['metrics']}
    return dict(windows=len(rows),families=len({(r['identity']['dataset'],r['family']) for r in rows}),
        family_macro_mean=values,dataset_macro_mean=values,by_dataset=by_dataset,
        aggregation='equal_dataset_then_equal_family; family_macro_mean retained as engine compatibility alias')


@torch.no_grad()
def evaluate(model,data,cases,config,device,out,step,full=False):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULT.json').exists():raise ValueError('Do not overwrite completed evaluation')
    saved=rng_state(np.random.default_rng(),device);was_training=model.training;model.eval()
    rows=[];started=time.monotonic();size=config['validation']['batch_size']
    try:
        with (out/'metrics.jsonl').open('w') as stream:
            for condition in CONDITIONS:
                for offset in range(0,len(cases),size):
                    chunk=cases[offset:offset+size]
                    pairs=[data.sample(np.random.default_rng(777+c['case_id']),config['frontend'],
                        member=c['member'],start=c['start'],yaw=0.,condition=condition) for c in chunk]
                    batch=collate([s for s,_ in pairs],device);payload=repair_payload(batch)
                    prediction=sample_denoising(model,payload,steps=config['diffusion']['ddim_steps'],
                        start_timestep=config['diffusion']['start_timestep'],seed=config['diffusion']['seed'])
                    variants=dict(initial=batch['initial_motion'],temporal_smoothing=engine.shared.smooth_initialization(payload),
                        untrained_anchor=model.anchor(payload),repair=prediction)
                    truth=forward_kinematics(batch['motion'],batch['target_rest'])
                    initial=forward_kinematics(batch['initial_motion'],batch['initial_rest'])
                    for variant,motion in variants.items():
                        points=forward_kinematics(motion,batch['initial_rest'])
                        for i,case in enumerate(chunk):
                            visible=(batch['joint_visible'][i]>0)&(batch['estimate_available'][i,:,None]>0)
                            region=(~visible) if condition in ('occlusion','gap') else torch.ones_like(visible)
                            error=(points[i]-truth[i]).norm(dim=-1)*1000;metrics=motion_metrics(points[i],truth[i])
                            metrics.update(repair_region_mpjpe_mm=float(error[region].mean()),
                                tracked_visible_mpjpe_mm=float(error[visible].mean()),
                                max_change_from_initial_mm=float((points[i]-initial[i]).norm(dim=-1).max()*1000))
                            row=dict(case_id=case['case_id'],family=pairs[i][1]['family'],condition=condition,
                                variant=variant,identity=pairs[i][1],metrics=metrics)
                            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                    stream.flush();atomic_json(out/'STATUS.json',dict(state='evaluating',step=step,full_validation=full,
                        condition=condition,completed=min(offset+size,len(cases)),total=len(cases),
                        updated_at_utc=now(),elapsed_seconds=time.monotonic()-started))
    finally:
        model.train(was_training);restore_rng(saved,np.random.default_rng(),device)
    summaries={c:{v:aggregate([r for r in rows if r['condition']==c and r['variant']==v])
        for v in ('initial','temporal_smoothing','untrained_anchor','repair')} for c in CONDITIONS}
    result=dict(state='complete',step=step,conditions=summaries,full_validation=full,
        windows_per_condition=len(cases),families=len({(r['identity']['dataset'],r['family']) for r in rows}),
        test_used=False,elapsed_seconds=time.monotonic()-started,completed_at_utc=now())
    atomic_json(out/'RESULT.json',result);atomic_json(out/'STATUS.json',result)
    return result


def enrich(folder,config):
    result=json.loads((folder/'RESULT.json').read_text())
    if 'selection_score_mm' in result:return result
    rows=[r for r in map(json.loads,(folder/'metrics.jsonl').read_text().splitlines()) if r['variant']!='amass_transfer_control']
    predictions={(r['condition'],r['case_id']):r for r in rows if r['variant']=='repair'}
    baseline=folder.parent/'step_000000/metrics.jsonl'
    if not baseline.exists():raise ValueError('Complete AMASS-transfer baseline is required')
    controls=[]
    for line in baseline.read_text().splitlines():
        r=json.loads(line);key=r['condition'],r['case_id']
        if r['variant']!='repair':continue
        if r['identity']!=predictions[key]['identity']:raise ValueError('Transfer-control inputs differ')
        controls.append(dict(r,variant='amass_transfer_control'))
    if len(controls)!=len(predictions):raise ValueError('Missing transfer-control cases')
    temporary=folder/'.metrics.jsonl.tmp'
    with temporary.open('w') as stream:
        for r in rows+controls:stream.write(json.dumps(r,allow_nan=False)+'\n')
    os.replace(temporary,folder/'metrics.jsonl')
    for c in CONDITIONS:result['conditions'][c]['amass_transfer_control']=aggregate([r for r in controls if r['condition']==c])
    checks={}
    checks['identity_preserved']=max(r['metrics']['max_change_from_initial_mm'] for r in rows
        if r['condition']=='identity' and r['variant']=='repair')<=config['validation']['identity_max_change_mm']
    for c in ('tracking','occlusion','gap'):
        values=result['conditions'][c];repair=values['repair']['family_macro_mean']
        for field,name in [('mpjpe_mm','overall'),('tracked_visible_mpjpe_mm','visible'),('root_error_mm','root')]:
            checks[c+'_'+name+'_noninferiority']=repair[field]<=values['initial']['family_macro_mean'][field]+config['validation']['visible_noninferiority_mm']
        if c in ('occlusion','gap'):
            for v in ('initial','temporal_smoothing','untrained_anchor','amass_transfer_control'):
                checks[c+'_beats_'+v]=repair['repair_region_mpjpe_mm']<values[v]['family_macro_mean']['repair_region_mpjpe_mm']
    eligible=all(v for k,v in checks.items() if k=='identity_preserved' or 'noninferiority' in k)
    result.update(checks=checks,candidate_gate_passed=all(checks.values()),selection_eligible=eligible,
        ready_for_three_corpus_stage=False,scene_branch_gate_passed=all(checks.values()),
        selection_score_mm=float(np.mean([result['conditions'][c]['repair']['dataset_macro_mean']['repair_region_mpjpe_mm']
            for c in ('tracking','occlusion','gap')])))
    atomic_json(folder/'RESULT.json',result);atomic_json(folder/'STATUS.json',result)
    return result


def final_report(out,seal):
    best=json.loads((out/'selection/BEST.json').read_text());result=json.loads(Path(best['metrics']).read_text())
    if sha256(Path(best['checkpoint']))!=best['checkpoint_sha256']:
        raise ValueError('Selected native repair weights changed')
    rows=[json.loads(line) for line in Path(best['metrics']).with_name('metrics.jsonl').read_text().splitlines()]
    rng=np.random.default_rng(20261027)
    intervals={c:{v:paired_dataset_interval(rows,c,v,rng) for v in
        ('initial','temporal_smoothing','untrained_anchor','amass_transfer_control')}
        for c in ('tracking','occlusion','gap')}
    report=dict(best=best,conditions=result['conditions'],checks=result['checks'],
        paired_region_intervals=intervals,
        full_validation=True,test_used=False,ready_for_three_corpus_stage=False,
        scene_branch_gate_passed=result['scene_branch_gate_passed'],contract_sha256=seal['sha256'],
        validation_windows_per_condition=result['windows_per_condition'],validation_families=result['families'],
        aggregation='equal_dataset_then_equal_family',known_limitations=['Synthetic native frontend, not actual video estimator.',
            'Native-format initialized repair does not itself validate EGO native-format scene generation.'])
    atomic_json(out/'RESULT.json',report)
    text=['# 三数据集 EXO 无场景修复适配','',f"best={best['step']}；每条件 {result['windows_per_condition']} 窗口；三来源等权、各来源 family 等权；未使用 test。",'',
        '| 条件/区域 | 初始化 | 平滑 | 固定时间锚点 | AMASS迁移起点 | 适配后 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for c in CONDITIONS:
        nums=[result['conditions'][c][v]['dataset_macro_mean']['repair_region_mpjpe_mm'] for v in
              ('initial','temporal_smoothing','untrained_anchor','amass_transfer_control','repair')]
        text.append('| '+c+' | '+' | '.join(f'{v:.3f}' for v in nums)+' |')
    text+=['','场景分支门槛：'+str(result['scene_branch_gate_passed']),
        '遮挡/缺帧行仅统计对应区域，其余行全窗口；单位 mm。实际视频、场景作用及预训练先验语义贡献需独立实验。','']
    (out/'SUMMARY_zh.md').write_text('\n'.join(text));return report


def paired_dataset_interval(rows,condition,control,rng):
    relevant=[r for r in rows if r['condition']==condition and r['variant'] in ('repair',control)]
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for row in relevant:
        groups[row['identity']['dataset']][row['family']][row['variant']].append(
            row['metrics']['repair_region_mpjpe_mm'])
    means=[];draws=[]
    for families in groups.values():
        differences=np.array([np.mean(v['repair'])-np.mean(v[control]) for v in families.values()])
        means.append(float(differences.mean()))
        draws.append(differences[rng.integers(len(differences),size=(5000,len(differences)))].mean(1))
    bounds=np.percentile(np.mean(draws,axis=0),[2.5,97.5])
    return dict(delta_repair_minus_control_mm=float(np.mean(means)),percentile_95_interval_mm=bounds.tolist(),
                bootstrap_replicates=5000,aggregation='equal_dataset_then_equal_family')


def pin_parent(config,out):
    parent=ROOT/config['parent_run'];status=json.loads((parent/'training/STATUS.json').read_text())
    if config.get('diagnostic_only'):
        pinned=out/'initialization.pt'
        if pinned.exists():
            record=json.loads((out/'INITIALIZATION.json').read_text())
            if not record.get('diagnostic_only') or sha256(pinned)!=record['sha256']:
                raise ValueError('Diagnostic initializer changed')
            return record
        shutil.copyfile(parent/'training/last.pt',pinned)
        state=torch.load(pinned,map_location='cpu',weights_only=False)
        record=dict(parent_run=config['parent_run'],step=state['step'],sha256=sha256(pinned),
            checkpoint=str(pinned),parent_contract_sha256=state['contract']['sha256'],
            diagnostic_only=True,not_validation_selected=True)
        atomic_json(out/'INITIALIZATION.json',record);return record
    if status['state']!='complete' or status.get('ready_for_three_corpus_stage') is not True:
        raise ValueError('AMASS has not completed and passed its stronger repair gate')
    best=json.loads((parent/'selection/BEST.json').read_text());source=Path(best['checkpoint'])
    if sha256(source)!=best['checkpoint_sha256']:raise ValueError('Parent selected weights changed')
    pinned=out/'initialization.pt'
    if pinned.exists():
        if sha256(pinned)!=best['checkpoint_sha256']:raise ValueError('Parent initializer was already pinned differently')
    else:shutil.copyfile(source,pinned)
    record=dict(parent_run=config['parent_run'],step=best['step'],sha256=best['checkpoint_sha256'],
        checkpoint=str(pinned),parent_contract_sha256=best['contract_sha256'],parent_full_validation=best['metrics'])
    atomic_json(out/'INITIALIZATION.json',record);return record


def setup(config,path,device,out):
    initialization=pin_parent(config,out)
    deterministic(config['seed'],device)
    if device=='cuda':
        if float(os.environ['SCENEMI_ALLOCATOR_GIB'])!=config['runtime']['allocator_gib']:raise ValueError('Allocator mismatch')
        configure_cuda()
    paths=load_paths(ROOT/'configs/paths.local.yaml');old,reference,_=settings(ROOT/'configs/amass_exo128.yaml')
    original=make_model(old,reference,paths,device);base=original.base;del original
    model=DenoisingExoPrior(base,**config['model']).to(device)
    checkpoint=torch.load(out/'initialization.pt',map_location='cpu',weights_only=False)
    model.load_repair_state(checkpoint['repair']);del checkpoint
    data=StageDataset(config,'train');validation=StageDataset(config,'validation');cases=validation.validation_cases()
    amass_validation=AmassDataset(INDEX,paths,'val');amass_cases=validation_manifest(amass_validation.members)
    original_ego_check=engine.shared.ego_check
    engine.shared.ego_check=lambda m,d,c,dev:original_ego_check(m,amass_validation,amass_cases,dev)
    engine.shared.batch_for=batch_for;engine.shared.sample_initialized=sample_denoising;engine.shared.update=engine.update
    engine.shared.aggregate_original=engine.shared.aggregate;engine.shared.aggregate=aggregate
    engine.shared.evaluate=evaluate;engine.enrich_evaluation=enrich;engine.final_report=final_report
    seal=contract(config,paths,INDEX)
    parent=json.loads((ROOT/config['parent_run']/'PROTOCOL.json').read_text());seal['sources'].update(parent['sources'])
    seal['sources'][str(path.resolve().relative_to(ROOT))]=sha256(path)
    seal.update(protocol=config['version'],validation_cases=cases,initializer=initialization,
        assets={str((out/'initialization.pt').relative_to(ROOT)):initialization['sha256'],
            config['base']['checkpoint']:config['base']['sha256'],
            str(INDEX.relative_to(ROOT)):sha256(INDEX),
            config['selection']:sha256(ROOT/config['selection']),
            'runs/data_inventory/CURRENT_SELECTION_CHECK.json':sha256(ROOT/'runs/data_inventory/CURRENT_SELECTION_CHECK.json'),
            **{str(p):digest for p,digest in data.asset_hashes.items()}},
        source_manifest_assets={str(p):sha256(p) for p in data.bundle.sources.values()},
        eligibility=dict(train=data.eligibility,validation=validation.eligibility),
        test_used=False,scene_input=False,modes=['exo'])
    seal['assets'].update(seal['source_manifest_assets']);seal.pop('sha256')
    seal['sha256']=hashlib.sha256(json.dumps(seal,sort_keys=True).encode()).hexdigest()
    protocol=out/'PROTOCOL.json'
    if protocol.exists() and json.loads(protocol.read_text())!=seal:raise ValueError('Native stage protocol changed')
    atomic_json(protocol,seal);engine.shared.verify(seal)
    return model,data,validation,cases,seal


def pipeline(config_path):
    config=yaml.safe_load(config_path.read_text());out=ROOT/config['output'];out.mkdir(parents=True,exist_ok=True)
    pin_parent(config,out)
    python=sys.executable
    commands=[
        [python,'-m','experiments.offline_camera_retrain_v1.ops.resources','run','--device','cpu','--memory-gib','10','--cpu-count','2','--',
         python,'-m','scenemi.scene_repair_training','preflight','--device','cpu','--config',str(config_path)],
        [python,'-m','scenemi.scene_repair_training','run','--device','cuda','--memory-gib','10','--cpu-count','2','--allocator-gib','5.5','--',
         python,'-m','scenemi.scene_repair_training','preflight','--device','cuda','--config',str(config_path)],
        [python,'-m','scenemi.scene_repair_training','run','--device','cuda','--memory-gib','10','--cpu-count','2','--allocator-gib','5.5','--',
         python,'-m','scenemi.scene_repair_training','train','--device','cuda','--config',str(config_path)]]
    if (out/'training/last.pt').exists():commands[-1].append('--resume')
    for phase,command in zip(('cpu_preflight','cuda_preflight','training'),commands):
        atomic_json(out/'PIPELINE_STATUS.json',dict(state=phase,updated_at_utc=now()))
        try:subprocess.run(command,cwd=ROOT,check=True)
        except subprocess.CalledProcessError as error:
            atomic_json(out/'PIPELINE_STATUS.json',dict(state='failed',phase=phase,
                returncode=error.returncode,updated_at_utc=now()))
            raise
    atomic_json(out/'PIPELINE_STATUS.json',dict(state='complete',updated_at_utc=now()))


def main():
    if len(sys.argv)>1 and sys.argv[1]=='run':
        from scenemi import parallel_resources
        from scenemi.exo_runtime import main as guarded
        authorization=json.loads((ROOT/'runs/exo_preparation/three_corpus_repair/SHARING_AUTHORIZATION.json').read_text())
        if authorization.get('user_authorized') is not True or authorization.get('allocator_limit_gib')!=5.5:
            raise ValueError('Missing authorization to extend shared EXO repair to three corpora')
        parallel_resources.MODULES|={'scenemi.scene_repair_training'};guarded();return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['probe','preflight','train','pipeline'])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.action=='pipeline':pipeline(args.config);return
    ensure_bounded();config=yaml.safe_load(args.config.read_text());out=ROOT/config['output'];out.mkdir(parents=True,exist_ok=True)
    if args.action=='probe':
        config=dict(config,output='runs/exo_preparation/three_corpus_repair/model_probe',diagnostic_only=True)
        out=ROOT/config['output'];out.mkdir(parents=True,exist_ok=True)
    if config['version']!='native-exo-denoising-adaptation-v1' or config['validation']['test_used'] or config['data']['scene']:
        raise ValueError('Wrong native scene-free adaptation protocol')
    model,data,validation,cases,seal=setup(config,args.config,args.device,out)
    if args.action in ('preflight','probe'):
        engine.shared.preflight(model,data,validation,cases,config,args.device,out,seal)
        if args.action=='probe':
            path=out/'preflight'/f'{args.device}.json';record=json.loads(path.read_text())
            record.update(diagnostic_only=True,formal_training_started=False,eligibility=seal['eligibility'],
                          validation_windows_per_condition=len(cases))
            atomic_json(path,record)
    else:engine.train(model,data,validation,cases,config,args.device,out,seal,args.resume)


if __name__=='__main__':
    main()
