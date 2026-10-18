"""Persistent paired EXO VAL review: frozen41k, v1 and balanced-v2 adapters.

The identity-only 64-recording cohort is immutable. Future candidates are
registered by exact final checkpoint SHA on CPU before their first prediction.
Every prediction is durable; completing all three candidates automatically runs
a separate CPU verifier. This evaluator never trains or opens TEST results.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import fcntl
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from . import validation_review as common
from .dependencies import dependency_closure,sha256

BASE=common.BASE
ROOT=common.ROOT
DEFAULT_OUT=BASE/'runs/native30_exo_expanded_review_v1'
V1=BASE/'runs/native30_exo_adapter_v1'
V2=BASE/'runs/native30_exo_adapter_balanced_v2'
PROTOCOL='native30-expanded-exo-three-model-val-v1'
MODULE='experiments.offline_camera_retrain_v1.ops.adapter_evaluation'
LABELS=('base41k','v1adapter1000','balancedv2adapter1000')
BASE_HASH='4d789e1e63f1681885c000b1d1795ba444b5a3eee3fd6af225fc9b7b0da6008f'
BASE_SHA='f8e2d22b155240c2fda8cb7d75aca451bb4152ede2441d592f3952f52068b4d0'
read=common.read
write=common.write
require=common.require
now=common.now


def status(out,state,**fields):
    write(out/'STATUS.json',dict(protocol=PROTOCOL,status=state,checked_at=now(),pid=os.getpid(),
        optimizer_updates=0,TEST_used=False,automatic_checkpoint_promotion=False,**fields))


def closure():
    from vit_pytorch import vit
    return dependency_closure([Path(__file__),BASE/'ops/exo_cohort.py'],
        extra_files=[ROOT.parent/'diffusion-motion-inbetweening/sample/virtual_head_camera.py',Path(vit.__file__)],
        import_roots=[ROOT.parent/'diffusion-motion-inbetweening'])


def inference_signature(config):
    fields=('frames','fps','latent_dim','dim_mults','diffusion_target','refinement','feedback','scene',
            'scene_relations','physical_guidance','observation','sampling','anchor_protocol','scene_relations')
    result={key:config.get(key) for key in fields}
    result['query_backend']=config.get('runtime',{}).get('query_backend','reference')
    return result


def panel_of(entry):
    panel=entry['panel']
    require(panel in ('new55','old9'),'Unknown registered EXO panel')
    return panel


def validate_entries(data,entries,config):
    from .exo_cohort import validate_entries as validate_cohort
    coverage=validate_cohort(data,entries,config)
    require(data.split=='validation' and len(entries)==64,'Require exactly64 official EXO VAL recordings')
    require(Counter(r['dataset'] for r in entries)==dict(rich=27,trumans=20,egobody=17),'EXO corpus counts differ')
    require(Counter(panel_of(r) for r in entries)==dict(new55=55,old9=9),'New/regression coverage differs')
    require(len({(r['dataset'],r['recording']) for r in entries})==64,'EXO recordings repeated')
    require(all(r['dataset']=='rich' and panel_of(r)=='new55' for r in entries[:24]),'First24 must be new RICH')
    require(all(r['dataset']=='rich' and panel_of(r)=='old9' for r in entries[24:27]),'Next3 must be old RICH')
    for index,row in enumerate(entries):
        require(row['case']==index and row['mode']=='exo' and row['joints'] is None,'Noncanonical EXO case')
        meta,_=data.members[row['group']][row['member']]
        require(meta['split']=='validation' and meta['sequence_id']==row['sequence_id'],'VAL source identity differs')
        require(Path(meta['path']).resolve()==Path(row['path']).resolve(),'VAL source path differs')
        require(row['start'] in data.starts_for(row['group'],row['member'],'exo'),'Illegal EXO source start')
        require(row['source_first']==meta['source_first']+row['start'],'EXO source clock differs')
        require(row['recording']==(meta.get('recording') or meta['sequence_id']),'EXO recording identity differs')
    return coverage


def declarations():
    return {
        LABELS[0]:dict(kind='bare',checkpoint=str(common.MAIN/'training/best.pt'),expected_step=41000,checkpoint_sha256=BASE_SHA),
        LABELS[1]:dict(kind='adapter',run=str(V1),checkpoint=str(V1/'training/last.pt'),expected_step=1000,
                       declared_config_sha256=sha256(V1/'config.json')),
        LABELS[2]:dict(kind='adapter',run=str(V2),checkpoint=str(V2/'training/last.pt'),expected_step=1000,
                       declared_config_sha256=sha256(V2/'config.json'),future_candidate=True)}


def verify_adapter_record(declaration,config):
    """Require completed stage identity/source evidence, independent of score."""
    run=Path(declaration['run']);require(sha256(run/'config.json')==declaration['declared_config_sha256'],'Declared stage config changed')
    effective=read(run/'prepared/effective_config.json');check=read(run/'CPU_RECHECK.json')
    require(check['status']=='independent_CPU_primary_metrics_passed' and check['EGO_serialized_outputs_exact'] is True,
            'Adapter needs completed independent CPU and EGO retention verification')
    require(check['CUDA_initialized'] is False and check['frozen_base_sha256']==BASE_HASH,'Adapter CPU/base provenance differs')
    checksum=sha256(declaration['checkpoint'])
    require(check['final_checkpoint_sha256']==checksum,'CPU report does not describe this final checkpoint')
    require(effective['config']==read(run/'config.json'),'Adapter embedded effective config differs')
    require(inference_signature(effective['config'])==inference_signature(config),'Inference architecture/physics/input protocol differs')
    for path,digest in effective['code_sha256'].items():require(sha256(path)==digest,'Adapter stage source changed: '+path)
    return checksum,dict(CPU_recheck_sha256=sha256(run/'CPU_RECHECK.json'),effective_sha256=sha256(run/'prepared/effective_config.json'),
                        stage_source_sha256=effective['code_sha256'],config_sha256=declaration['declared_config_sha256'])


def load_registered(declaration,device='cpu'):
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model
    from experiments.offline_camera_retrain_v1.train_native30 import checkpoint_config,model_digest
    from .adapter_model import load_inference_model
    path=declaration['checkpoint'];require(sha256(path)==declaration['checkpoint_sha256'],'Registered checkpoint changed')
    if declaration['kind']=='adapter':
        model,metadata=load_inference_model(path,device=device,expected_sha256=declaration['checkpoint_sha256'])
        require(metadata['stage_step']==metadata['planned_stage_updates']==1000,'Only completed1000-update adapters may enter')
        require(metadata['frozen_base_sha256']==BASE_HASH,'Adapter has a different frozen41k base')
        return model,metadata
    state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    require(state['step']==41000,'Bare model is not retained41k')
    model=make_model(checkpoint_config(state));model.load_state_dict(state['model'],strict=True)
    require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()),'Nonfinite base checkpoint')
    digest=model_digest(model);require(digest==BASE_HASH,'Bare41k tensor identity differs')
    return model.to(device).eval(),dict(config=checkpoint_config(state),model_tensor_sha256=digest,stage_step=41000,
                                      strict_full_state_loaded=True,all_state_tensors_finite=True)


def register_models(out,labels):
    common.cpu_setup();plan=read(out/'PLAN.json');verify_plan(out,plan)
    registry=read(out/'MODELS.json') if (out/'MODELS.json').exists() else {}
    for label in labels:
        require(label in LABELS,'Unknown model label')
        if label in registry:
            verify_registered(registry[label]);continue
        declaration=dict(plan['model_declarations'][label]);source=Path(declaration['checkpoint'])
        if declaration['kind']=='adapter':checksum,evidence=verify_adapter_record(declaration,plan['config'])
        else:checksum=declaration['checkpoint_sha256'];evidence={}
        require(sha256(source)==checksum,'Declared source checkpoint SHA differs')
        target=out/'checkpoints'/(label+'.pt');target.parent.mkdir(exist_ok=True)
        if not target.exists():os.link(source,target)
        require(sha256(target)==checksum,'Pinned candidate differs')
        declaration.update(source_checkpoint=str(source),checkpoint=str(target.resolve()),checkpoint_sha256=checksum,stage_evidence=evidence)
        model,metadata=load_registered(declaration)
        require(inference_signature(metadata['config'])==inference_signature(plan['config']),'Candidate inference protocol differs')
        from experiments.offline_camera_retrain_v1.train_native30 import model_digest
        declaration.update(model_tensor_sha256=model_digest(model),loaded_metadata={k:v for k,v in metadata.items() if k!='config'},
                           registered_at=now(),CPU_strict_finite=True)
        del model;gc.collect();registry[label]=declaration;write(out/'MODELS.json',registry)
    return registry


def verify_registered(declaration):
    require(sha256(declaration['checkpoint'])==declaration['checkpoint_sha256'],'Pinned registered candidate changed')
    evidence=declaration.get('stage_evidence',{})
    for path,digest in evidence.get('stage_source_sha256',{}).items():require(sha256(path)==digest,'Registered candidate source changed: '+path)
    if evidence:
        run=Path(declaration['run'])
        require(sha256(run/'CPU_RECHECK.json')==evidence['CPU_recheck_sha256'],'Registered CPU evidence changed')
        require(sha256(run/'prepared/effective_config.json')==evidence['effective_sha256'],'Registered effective config changed')
        require(sha256(run/'config.json')==evidence['config_sha256'],'Registered stage config changed')


def prepare(out):
    common.cpu_setup();out.mkdir(parents=True,exist_ok=True)
    require(not (out/'PLAN.json').exists(),'Prepared immutable PLAN already exists')
    status(out,'preparing_cpu')
    config=common.configurations(out);cohort=read(out/'COHORT.json');data=common.validation_data(config)
    coverage=validate_entries(data,cohort['entries'],config)
    for path,digest in cohort['source_sha256'].items():require(sha256(path)==digest,'Cohort source changed')
    probe=probe_inputs(data,common.observation_builder(config),cohort['entries'][0],config)
    plan=dict(protocol=PROTOCOL,prepared_at=now(),config=config,config_sha256=sha256(out/'config.json'),
        cohort_sha256=sha256(out/'COHORT.json'),source_sha256=closure(),cohort_source_sha256=cohort['source_sha256'],
        metadata_sha256=common.metadata_identity(data,config),coverage=coverage,windows=64,planned_predictions=384,
        model_order=list(LABELS),model_declarations=declarations(),regimes=['clean','noisy'],frames=192,fps=30,ddim_steps=20,
        scene='on',physics='on',mode='exo',CPU_input_probe=probe,
        aggregation='New55/old9/all64 separately; one window per(dataset,recording). Report valid-count weighted and equal-recording within dataset separately; macro gives the three dataset recording means equal weight. clean/noisy separate.',
        primary='RICH new24 recordings hidden MPJPE with visible/foot guards; new55 equal-dataset macro secondary. Existing9 are regression only.',
        guard_rules=dict(hidden='final<=base+max(5mm,5%) per dataset/regime',visible='final<=base+max(1mm,2%)',
            foot_speed='GT-low-speed3D predicted foot speed<=base+max(.01m/s,5%)',
            bilateral='RMS/stride error<=base+max(5mm,10%)',macro_hidden='At least5% improvement clean/noisy separately; engineering screen, not significance'),
        automatic_promotion=False,TEST_used=False,optimizer_updates=0)
    write(out/'PLAN.json',plan)
    register_models(out,LABELS[:2])
    status(out,'prepared',completed_predictions=0,planned_predictions=384,registered_models=list(LABELS[:2]),future_model=LABELS[2])
    return plan


def verify_plan(out,plan,data=None):
    require(plan['protocol']==PROTOCOL and plan['planned_predictions']==384,'Unexpected EXO review protocol')
    require(plan['model_order']==list(LABELS),'Model comparison set changed')
    require(sha256(out/'COHORT.json')==plan['cohort_sha256'] and sha256(out/'config.json')==plan['config_sha256'],'Frozen cohort/config changed')
    for path,digest in {**plan['source_sha256'],**plan['metadata_sha256'],**plan['cohort_source_sha256']}.items():
        require(sha256(path)==digest,'Sealed source/metadata changed: '+path)
    if data is not None:
        require(common.metadata_identity(data,plan['config'])==plan['metadata_sha256'],'VAL metadata coverage changed')
        validate_entries(data,read(out/'COHORT.json')['entries'],plan['config'])


def source_observation(data,builder,entry,regime):
    import numpy as np
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    sample,identity=data.sample(entry['group'],'exo',member=entry['member'],start=entry['start'],yaw=0)
    require(identity['source_start_30fps']==entry['source_first'],'Actual source clock differs')
    sensors={k:v for k,v in sample.items() if k not in ('motion','joints','contact_target','contact_valid') and not k.startswith('gt_')}
    selected=builder if regime=='clean' else Native30ObservationBuilder(builder.skin_cache,builder.mounts,dict(builder.noise_config,clean_probability=0.))
    observed,scene,asset=selected.build(data,sensors,identity,'exo',None,
        noise_rng=None if regime=='clean' else np.random.default_rng(entry['case_seed']))
    return sample,observed,scene,asset


def collated_inputs(config,observed,scene,asset,device):
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    batch=collate_observations([(observed,scene,asset,'exo')],device=device)
    known=common.inference_inputs(config,batch);common.assert_input_device(known,device)
    require(known['mode']==['exo'],'Explicit EXO mode missing')
    return batch,known


def probe_inputs(data,builder,entry,config):
    import torch
    result={}
    for regime in ('clean','noisy'):
        sample,observed,scene,asset=source_observation(data,builder,entry,regime)
        batch,known=collated_inputs(config,observed,scene,asset,'cpu')
        require(all(bool(torch.isfinite(value).all()) for _,value in common.tensor_leaves(known)),'Nonfinite EXO input')
        result[regime]=dict(case=entry['case'],recursive_CPU_input_sha256=common.tensor_fingerprint(known),label_free=True,explicit_mode='exo')
    require(not torch.cuda.is_initialized(),'CPU EXO preflight initialized CUDA')
    return result


def schedule(entries,labels):
    return [(entry,regime,label) for entry in entries for regime in ('clean','noisy') for label in labels]


def committed_rows(out,plan,entries,registry):
    # Existing helper enforces unique(case,regime,model), seed/source identity,
    # canonical artifact path, exact checkpoint step/SHA and durable last line.
    generic=dict(model_order=list(LABELS),models=registry)
    if not registry and (out/'rows.jsonl').exists():raise ValueError('Rows exist without frozen model registration')
    rows=common.committed_rows(out,generic,entries)
    indexed={entry['case']:entry for entry in entries}
    for row in rows:require(row['panel']==panel_of(indexed[row['case']]),'Result new/regression panel changed')
    return rows


def aggregate(rows):
    groups={}
    for scope in ('new55','old9','all64'):
        selected=[r for r in rows if scope=='all64' or r['panel']==scope]
        for label in LABELS:
            for dataset in ('rich','trumans','egobody'):
                for regime in ('clean','noisy'):
                    items=[r for r in selected if (r['model'],r['dataset'],r['regime'])==(label,dataset,regime)]
                    if not items:continue
                    metrics={}
                    for key in sorted({k for r in items for k in r['metrics']}):
                        valid=[r for r in items if r['metrics'].get(key) is not None]
                        if not valid:continue
                        recs=defaultdict(list)
                        for row in valid:recs[row['recording']].append(row)
                        def count_mean(part):
                            counted=[r for r in part if r['counts'].get(key,0)>0]
                            total=sum(r['counts'][key] for r in counted)
                            return sum(r['metrics'][key]*r['counts'][key] for r in counted)/total if total else statistics.mean(r['metrics'][key] for r in part)
                        count=sum(r['counts'].get(key,0) for r in valid)
                        metrics[key]=dict(window_mean=statistics.mean(r['metrics'][key] for r in valid),recording_macro=statistics.mean(count_mean(p) for p in recs.values()),
                            count_weighted=count_mean(valid) if count else None,count=count or None,windows=len(valid),recordings=len(recs))
                    groups['/'.join((scope,label,dataset,regime))]=dict(windows=len(items),recordings=len({r['recording'] for r in items}),metrics=metrics)
    macro={}
    for scope in ('new55','old9','all64'):
        for label in LABELS:
            for regime in ('clean','noisy'):
                items=[groups.get('/'.join((scope,label,d,regime))) for d in ('rich','trumans','egobody')]
                if not all(items):continue
                keys=set.intersection(*(set(item['metrics']) for item in items))
                macro['/'.join((scope,label,regime))]={k:statistics.mean(item['metrics'][k]['recording_macro'] for item in items) for k in sorted(keys)}
    return groups,macro


def comparisons(rows,groups,macro):
    indexed={common.row_key(r):r for r in rows};paired=[]
    for row in rows:
        if row['model']!=LABELS[0]:continue
        for label in LABELS[1:]:
            other=indexed.get((row['case'],row['regime'],label))
            if other is None:continue
            require(row['input_sha256']==other['input_sha256'] and row['counts']==other['counts'],'Matched model inputs/counts differ')
            metrics={key:dict(base=row['metrics'][key],candidate=other['metrics'][key],delta=other['metrics'][key]-row['metrics'][key])
                     for key in row['metrics'] if row['metrics'][key] is not None and other['metrics'].get(key) is not None}
            paired.append(dict(case=row['case'],dataset=row['dataset'],recording=row['recording'],panel=row['panel'],regime=row['regime'],candidate=label,metrics=metrics))
    guards=[]
    for key,a in groups.items():
        scope,label,dataset,regime=key.split('/')
        if label!=LABELS[0]:continue
        for candidate in LABELS[1:]:
            b=groups.get('/'.join((scope,candidate,dataset,regime)))
            if not b or b['windows']!=a['windows']:continue
            for metric,absolute,relative in [('hidden_mpjpe_mm',5.,.05),('visible_mpjpe_mm',1.,.02),('foot_joint_speed_gt_low_m_s',.01,.05),('bilateral_motion_rms_error_mm',5.,.1),('bilateral_stride_change_abs_error_mm',5.,.1)]:
                if metric not in a['metrics'] or metric not in b['metrics']:continue
                field='recording_macro'
                x=a['metrics'][metric][field];y=b['metrics'][metric][field]
                guards.append(dict(scope=scope,dataset=dataset,regime=regime,candidate=candidate,metric=metric,base=x,candidate_value=y,limit=x+max(absolute,relative*x),passed=y<=x+max(absolute,relative*x)))
    return paired,guards


def summarize(out,plan,rows,verified=False):
    groups,macro=aggregate(rows);paired,guards=comparisons(rows,groups,macro)
    complete=len(rows)==384 and verified
    report=dict(protocol=PROTOCOL,status='complete_CPU_verified' if complete else 'partial_do_not_select',checked_at=now(),
        completed_predictions=len(rows),planned_predictions=384,groups=groups,equal_dataset_macro=macro,paired_recordings=paired,engineering_guards=guards,
        primary=plan['primary'],scope=plan['coverage'],TEST_used=False,automatic_promotion=False,
        warning='New55 and old9 separate; the3-model panel must finish before a complete comparison. No automatic model promotion or statistical significance claim.')
    write(out/'SUMMARY.json',report)
    lines=['扩大EXO验证：裸41k、v1 adapter1000、balanced v2 adapter1000。',f"状态：{report['status']}，{len(rows)}/384次预测；不自动替换模型。",'',
        '64个不同录制各1窗：RICH27、TRUMANS20、EgoBody17；新55与旧9回归分开。输入/种子逐窗匹配，30Hz/192帧/DDIM20，场景与物理修正开启。',
        '主证据为RICH新24录制的隐藏关节及可见/脚部约束；旧9不作为新泛化证据。完整结果须独立CPU复算。','',
        '|范围|模型|数据/输入|窗/录制|World mm|隐藏 mm|可见 mm|低速3D脚速m/s|','|---|---|---|---:|---:|---:|---:|---:|']
    for scope in ('new55','old9','all64'):
        for label in LABELS:
            for dataset in ('rich','trumans','egobody'):
                for regime in ('clean','noisy'):
                    group=groups.get('/'.join((scope,label,dataset,regime)))
                    if not group:continue
                    m=group['metrics'];fmt=lambda k:f"{m[k]['recording_macro']:.3f}" if k in m else 'NA'
                    lines.append(f"|{scope}|{label}|{dataset}/{regime}|{group['windows']}/{group['recordings']}|{fmt('w_mpjpe_mm')}|{fmt('hidden_mpjpe_mm')}|{fmt('visible_mpjpe_mm')}|{fmt('foot_joint_speed_gt_low_m_s')}|")
    lines+=['','可见/隐藏按输入位置可用掩码；GT脚速<0.2m/s为运动学代理，不是真实接触或全mesh穿模。PA为逐帧相似变换；World未对齐。',
        '同一录制的两角色未当成两个独立录制；clean/noisy不是独立样本；不同录制可能同主体/场景。配对逐录制变化、有效分母与工程guards详见SUMMARY.json。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    return report


def progress(out,rows,started):
    recent=rows[-min(12,len(rows)):];per=statistics.mean(r['sampling_s']+r['metric_s']+r['preparation_share_s'] for r in recent) if recent else None
    value=dict(checked_at=now(),completed_predictions=len(rows),planned_predictions=384,elapsed_this_process_s=time.monotonic()-started,
        recent_predictions=len(recent),recent_seconds_per_prediction=per,estimated_remaining_sampling_s=(384-len(rows))*per if per else None,
        estimate_excludes='Queue/startup, waiting for futurev2, and final CPU verification; scenes can change throughput.')
    write(out/'PROGRESS.json',value)
    if len(rows)==12:write(out/'FIRST_12_TIMING.json',value)


def evaluate(out,labels,resume=False,max_predictions=0,memory_fraction=.5):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.native30_refinement import sample_motion
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    plan=read(out/'PLAN.json');verify_plan(out,plan)
    registry=register_models(out,labels);entries=read(out/'COHORT.json')['entries']
    require(resume or not (out/'rows.jsonl').exists(),'Existing rows require explicit resume')
    rows=committed_rows(out,plan,entries,registry);done={common.row_key(r):r for r in rows}
    if len(rows)==384:return True
    if all((entry['case'],regime,label) in done for entry,regime,label in schedule(entries,labels)):
        summarize(out,plan,rows)
        status(out,'selected_models_complete_pending_other_models',completed_predictions=len(rows),planned_predictions=384)
        return False
    require(0<memory_fraction<=.5,'GPU fraction must be <=.5')
    status(out,'waiting_for_gpu',completed_predictions=len(rows),planned_predictions=384,selected_models=labels)
    with common.gpu_lock():
        verify_plan(out,plan);os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        config=plan['config'];data=common.validation_data(config);verify_plan(out,plan,data)
        builder=common.observation_builder(config,device='cuda');models={}
        for label in labels:
            verify_registered(registry[label]);model,metadata=load_registered(registry[label],device='cuda')
            require(model_digest(model)==registry[label]['model_tensor_sha256'],'Loaded model digest differs')
            models[label]=model
        started=time.monotonic();generated=0;folder=out/'motions';folder.mkdir(exist_ok=True)
        for entry in entries:
            for regime in ('clean','noisy'):
                if all((entry['case'],regime,label) in done for label in labels):continue
                before=time.monotonic();sample,observed,scene,asset=source_observation(data,builder,entry,regime)
                batch,known=collated_inputs(config,observed,scene,asset,'cuda');input_sha=common.tensor_fingerprint(known)
                prep_share=(time.monotonic()-before)/sum((entry['case'],regime,label) not in done for label in labels)
                existing=[r for key,r in done.items() if key[:2]==(entry['case'],regime)]
                require(all(r['input_sha256']==input_sha for r in existing),'Resumed paired full input differs')
                for label in labels:
                    key=entry['case'],regime,label
                    if key in done:continue
                    status(out,'sampling',completed_predictions=len(rows),planned_predictions=384,current_case=entry['case'],dataset=entry['dataset'],regime=regime,model=label)
                    before=time.monotonic()
                    with torch.no_grad():prediction=sample_motion(models[label],known,config,steps=20,seed=entry['case_seed'],use_scene=True)
                    torch.cuda.synchronize();sampling_s=time.monotonic()-before
                    require(bool(torch.isfinite(prediction).all()),'Nonfinite EXO prediction')
                    artifact=folder/f"case{entry['case']:04d}_{regime}_{label}.npz";temporary=artifact.with_suffix('.tmp')
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream,prediction=prediction[0].detach().cpu().numpy(),truth=sample['motion'],rest=sample['rest'],observations=observed['trajectory'],observation_meta=observed['observation_meta'])
                        stream.flush();os.fsync(stream.fileno())
                    temporary.replace(artifact);before=time.monotonic()
                    with np.load(artifact,allow_pickle=False) as saved:metrics,counts=common.metrics_for(saved)
                    row=dict(case=entry['case'],dataset=entry['dataset'],group=entry['group'],mode='exo',model=label,panel=panel_of(entry),
                        recording=entry['recording'],sequence_id=entry['sequence_id'],source_first=entry['source_first'],start=entry['start'],regime=regime,seed=entry['case_seed'],
                        checkpoint_step=registry[label]['expected_step'],checkpoint_sha256=registry[label]['checkpoint_sha256'],input_sha256=input_sha,
                        artifact=str(artifact.resolve()),artifact_sha256=sha256(artifact),metrics=metrics,counts=counts,sampling_s=sampling_s,metric_s=time.monotonic()-before,preparation_share_s=prep_share,checked_at=now())
                    with (out/'rows.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                    rows.append(row);done[key]=row;generated+=1;progress(out,rows,started)
                    if len(rows)%12==0:summarize(out,plan,rows)
                    if max_predictions and generated>=max_predictions:
                        summarize(out,plan,rows);status(out,'partial_requested',completed_predictions=len(rows),planned_predictions=384);return False
                del sample,observed,scene,asset,batch,known
        for label,model in models.items():require(model_digest(model)==registry[label]['model_tensor_sha256'],'Inference model tensors changed')
        verify_plan(out,plan)
        del models,model,prediction,data,builder;gc.collect();torch.cuda.empty_cache()
    complete=len(rows)==384;summarize(out,plan,rows)
    status(out,'sampling_completed_pending_cpu' if complete else 'selected_models_complete_pending_other_models',completed_predictions=len(rows),planned_predictions=384)
    return complete


def recheck(out):
    import numpy as np
    import torch
    from .foot_trial_recheck import independent_metrics
    common.cpu_setup();plan=read(out/'PLAN.json');verify_plan(out,plan)
    registry=read(out/'MODELS.json');require(set(registry)==set(LABELS),'All3 models must be registered')
    for declaration in registry.values():verify_registered(declaration)
    entries=read(out/'COHORT.json')['entries'];data=common.validation_data(plan['config']);verify_plan(out,plan,data)
    builder=common.observation_builder(plan['config']);rows=committed_rows(out,plan,entries,registry)
    require(len(rows)==384,'All384 predictions required for final CPU check')
    indexed={common.row_key(r):r for r in rows};maximum=direct_maximum=0.;checked=0
    for entry in entries:
        for regime in ('clean','noisy'):
            sample,observed,scene,asset=source_observation(data,builder,entry,regime);paired=None
            for label in LABELS:
                row=indexed[entry['case'],regime,label]
                with np.load(row['artifact'],allow_pickle=False) as saved:
                    arrays={k:saved[k].copy() for k in saved.files}
                for key,value in [('truth',sample['motion']),('rest',sample['rest']),('observations',observed['trajectory']),('observation_meta',observed['observation_meta'])]:
                    require(np.array_equal(arrays[key],value),'Independent source GT/observations differ: '+key)
                actual,counts=common.metrics_for(arrays);direct,_,denominators=independent_metrics(arrays['prediction'],arrays['truth'],arrays['rest'],arrays['observations'],arrays['observation_meta']);direct.update(common.independent_core(arrays))
                require(counts==row['counts'] and set(actual)==set(row['metrics']),'Metric schema/denominator differs')
                for key,value in actual.items():
                    require((value is None)==(row['metrics'][key] is None),'Metric missingness differs')
                    if value is not None:maximum=max(maximum,abs(value-row['metrics'][key]))
                for key,value in direct.items():
                    if key not in actual:continue
                    require((value is None)==(actual[key] is None),'Independent missingness differs')
                    if value is not None:direct_maximum=max(direct_maximum,abs(value-actual[key]))
                if paired is not None:
                    old,previous=paired
                    require(old['input_sha256']==row['input_sha256'] and old['counts']==counts,'Paired tensor input fingerprints/counts differ')
                    require(all(np.array_equal(arrays[k],previous[k]) for k in ('truth','rest','observations','observation_meta')),'Pair arrays differ')
                paired=row,arrays;checked+=1
            if checked%24==0:status(out,'cpu_rechecking',verified_predictions=checked,completed_predictions=384,planned_predictions=384)
    require(maximum<1e-6 and direct_maximum<1e-3,'Independent CPU formulas differ')
    require(not torch.cuda.is_initialized(),'CPU recheck initialized CUDA');verify_plan(out,plan)
    write(out/'CPU_RECHECK.json',dict(status='independent_cpu_passed',checked_at=now(),predictions=checked,CUDA_initialized=False,
        maximum_recompute_difference=maximum,maximum_independent_formula_difference=direct_maximum,source_GT_rest_observations_exact=True,
        paired_recursive_tensor_input_hashes_and_denominators_exact=True,rows_sha256=sha256(out/'rows.jsonl'),
        registered_checkpoint_sha256={k:v['checkpoint_sha256'] for k,v in registry.items()},TEST_used=False,optimizer_updates=0))
    summarize(out,plan,rows,verified=True);status(out,'completed',completed_predictions=384,planned_predictions=384)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','register','evaluate','resume','run','recheck','status'))
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--models',nargs='+',choices=LABELS,default=list(LABELS))
    parser.add_argument('--memory-fraction',type=float,default=.5)
    parser.add_argument('--max-predictions',type=int,default=0)
    args=parser.parse_args(argv);out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    labels=[label for label in LABELS if label in args.models];require(len(labels)==len(args.models),'Duplicate selected model')
    if args.command=='status':print(json.dumps(read(out/'STATUS.json') if (out/'STATUS.json').exists() else dict(status='not_prepared'),ensure_ascii=False,indent=2));return
    acquired=False;complete=False
    try:
        with (out/'job.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
            if args.command=='prepare':prepare(out)
            elif args.command=='register':register_models(out,labels)
            elif args.command=='recheck':recheck(out)
            else:
                if args.command=='run' and not (out/'PLAN.json').exists():prepare(out)
                complete=evaluate(out,labels,resume=args.command in ('resume','run'),max_predictions=args.max_predictions,memory_fraction=args.memory_fraction)
        if complete:subprocess.run([sys.executable,'-m',MODULE,'recheck','--out',str(out)],check=True)
    except BaseException as error:
        if acquired:status(out,'failed',operation=args.command,error=repr(error))
        raise


if __name__=='__main__':main()
