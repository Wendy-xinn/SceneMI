"""Frozen full-pool ego run: 55k LR horizon, explicit resumable review boundary.

Uses the unchanged audited trainer/model/objective. Scoped wrappers add ego-only
monitoring, foot diagnostics and SIGTERM at a completed validation boundary.
No automatic transition beyond the selected review boundary or to EXO.
"""
import argparse
import copy
import datetime
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1 import train_native30 as trainer
from experiments.offline_camera_retrain_v1 import native30_validation as validation
from experiments.offline_camera_retrain_v1.prepare_native30 import HERE,sha256,write_json

ROOT=Path(__file__).resolve().parent
INITIAL='9d14dbbb56e6c092ca2c0f784e4f3935ad0a0cf6365cc35bf3894fe07dc90d69'


def now():return datetime.datetime.now().astimezone().isoformat()


def at_review(step,review_step):
    return step>=review_step


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-step',type=int,default=1000)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    c=json.loads((ROOT/'config.json').read_text());control=c['run_control'];planned=c['phase_plan']['planned_updates']
    assert args.review_step in control['review_milestones'] and args.review_step<=planned
    assert sha256(Path(__file__))==control['wrapper_sha256']
    metrics_path=HERE/control['foot_metrics_source']
    assert sha256(metrics_path)==control['foot_metrics_sha256']
    spec=importlib.util.spec_from_file_location('frozen_foot_metrics',metrics_path)
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    output=ROOT/'training'
    if args.resume:
        state=torch.load(output/'last.pt',map_location='cpu',mmap=True,weights_only=False)
        assert state['step']<args.review_step and state['planned_steps']==planned
        del state
    launch=dict(status='starting',started_at=now(),pid=os.getpid(),planned_steps=planned,
        review_step=args.review_step,resume=args.resume,training_mode='ego',
        initialization='from_scratch' if not args.resume else 'strict same-plan resume',
        wrapper_sha256=control['wrapper_sha256'],automatic_exo=False,automatic_review_extension=False,
        memory_fraction=.35,threads=4,deterministic=True)
    write_json(ROOT/'invocation.json',launch)
    original_plan=validation.monitoring_plan
    original_validate=trainer.validate
    original_digest=trainer.dataset_digest
    original_model_digest=trainer.model_digest
    def checked_initial(model):
        result=original_model_digest(model)
        assert result==INITIAL, ('Fresh initialization changed',result)
        return result
    trainer.model_digest=checked_initial
    def monitoring(data,config,step):
        plan=[r for r in original_plan(data,config,step) if r['mode'] in config['validation']['monitor_modes']]
        if step==1 and config['validation']['initial_fixed_only']:
            plan=[r for r in plan if r['panel']=='fixed']
        assert len(plan)==(3 if step==1 else 6) and all(r['mode']=='ego' for r in plan)
        return plan
    validation.monitoring_plan=monitoring
    fingerprints={};digest_times={}
    def audited_digest(data):
        print('Revalidating full source fingerprints:',data.split,flush=True)
        tick=time.monotonic();result=original_digest(data)
        fingerprints.update(result);digest_times[data.split]=time.monotonic()-tick
        if len(digest_times)==2:
            effective=json.loads((ROOT/'prepared/effective_config.json').read_text())
            assert fingerprints==effective['data_sha256']
            write_json(ROOT/'DATA_REVALIDATION.json',dict(status='passed',checked_at=now(),
                splits=['train','validation'],TEST_used=False,durations_s=digest_times,
                unchanged_from=control['engineering_preflight'],files=len(fingerprints)))
            print('Full TRAIN/VAL fingerprints passed',digest_times,flush=True)
        return result
    trainer.dataset_digest=audited_digest
    def monitored_validate(model,data,builder,config,folder,step):
        print('Validation begins:',step,'ego only',flush=True)
        options=copy.deepcopy(config)
        if step==1 and not config['validation']['initial_scene_ablation']:
            options['validation']['scene_ablation_interval_steps']=0
        tick=time.monotonic();rows=original_validate(model,data,builder,options,folder,step)
        plan=monitoring(data,options,step);foot_rows=[]
        with torch.no_grad(),np.load(Path(folder)/'validation_latest.npz') as saved:
            for i,entry in enumerate(plan):
                truth=torch.from_numpy(saved[f'{i}_truth'])[None]
                rest=torch.from_numpy(saved[f'{i}_rest'])[None]
                for regime in ('clean','noisy'):
                    prediction=torch.from_numpy(saved[f'{i}_{regime}_prediction'])[None]
                    values=metrics.extra_metrics(prediction,dict(motion=truth,rest=rest))
                    foot_rows.append(dict(case=i,panel=entry['panel'],dataset=entry['dataset'],
                        sequence_id=entry['sequence_id'],regime=regime,metrics=values))
        summary={}
        for panel in ('fixed','rotating'):
            for regime in ('clean','noisy'):
                group=[r['metrics'] for r in foot_rows if r['panel']==panel and r['regime']==regime]
                if group:
                    summary[panel+'/'+regime]={k:float(np.mean([r[k] for r in group if r[k] is not None]))
                        for k,v in group[0].items() if isinstance(v,(float,int))}
        record=dict(step=step,checked_at=now(),validation_seconds=time.monotonic()-tick,
            scope='Same fixed windows for trends; rotating coverage separate; raw unweighted GT; original30Hz predictions.',
            cases=foot_rows,summary=summary,
            review_policy='128-update small-pool triage is not a formal convergence gate. Review phase/time-resolved feet plus pose, root, stance and scene on fixed/rotating official VAL.')
        with (Path(folder)/'foot_monitor.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        print('Validation complete:',step,'seconds',record['validation_seconds'],flush=True)
        if at_review(step,args.review_step):
            write_json(ROOT/'REVIEW_BOUNDARY.json',dict(status='validation_completed_checkpoint_pending',step=step,
                planned_steps=planned,observed_at=now(),reason='Explicit review boundary; preserve full55k LR schedule and Adam/RNG. No automatic extension.'))
            # Existing trainer handler finishes this validation, atomically saves
            # model+Adam+sampler+all RNG, records stopped and breaks before step+1.
            os.kill(os.getpid(),signal.SIGTERM)
        return rows
    trainer.validate=monitored_validate
    try:
        trainer.train(ROOT/'config.json',planned,output,args.resume)
        state=torch.load(output/'last.pt',map_location='cpu',mmap=True,weights_only=False)
        assert state['planned_steps']==planned and 'optimizer' in state
        if state['step']==args.review_step:
            write_json(ROOT/'REVIEW_BOUNDARY.json',dict(status='ready_for_review',step=state['step'],
                planned_steps=planned,checkpoint=str(output/'last.pt'),completed_at=now(),
                optimizer_and_sampler_RNG_preserved=True,automatic_extension=False))
        launch.update(status='review_hold' if state['step']==args.review_step else 'stopped',
            completed_steps=state['step'],finished_at=now())
        write_json(ROOT/'invocation.json',launch)
    except BaseException as exc:
        launch.update(status='failed',error=repr(exc),failed_at=now());write_json(ROOT/'invocation.json',launch)
        raise
    finally:
        validation.monitoring_plan=original_plan;trainer.validate=original_validate
        trainer.dataset_digest=original_digest;trainer.model_digest=original_model_digest


if __name__=='__main__':main()
