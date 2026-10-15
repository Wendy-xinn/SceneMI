"""Activate unchanged numerical configuration; retain prior audit provenance.

The original preflight-only engineering seal stays immutable. This is a clearly
recorded administrative activation, not a claim of repeating numeric checks.
The trainer independently rehashes all TRAIN/VAL data before the first update.
"""
import copy
import datetime
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from experiments.offline_camera_retrain_v1.prepare_native30 import HERE,code_hashes,sha256,read_config,write_json

ROOT=Path(__file__).resolve().parent


def main():
    source=HERE/'runs/native30_gait_v2_arch_review_oct15/prepared'
    effective=json.loads((source/'effective_config.json').read_text())
    checks=json.loads((source/'input_checks.json').read_text())
    assert checks['status']=='passed' and checks['engineering_ready']
    assert effective['code_sha256']==code_hashes()
    c=copy.deepcopy(effective['config'])
    c['preflight_only']=False
    c['prepared_output']=str((ROOT/'prepared').relative_to(HERE))
    c['training_status']='Frozen full-pool fresh ego;55k LR horizon; stop and review at1000; no automatic EXO.'
    c['phase_plan']['budget_status']='Global cap55000 from scratch; first review1000; later continuation requires validation review.'
    c['validation'].update(monitor_modes=['ego'],initial_fixed_only=True,initial_scene_ablation=False,
        scope='Ego phase only:3 fixed+3 rotating official VAL windows, clean/noisy separate; first update3 fixed only; scene-OFF at5000; full final evaluation protocol unchanged.')
    c['run_control']=dict(protocol='native30-fresh-ego55k-review-v1',
        wrapper_sha256=sha256(ROOT/'run_training.py'),
        foot_metrics_source='runs/native30_physics_schedule_probe_oct15/run_probe.py',
        foot_metrics_sha256=sha256(HERE/'runs/native30_physics_schedule_probe_oct15/run_probe.py'),
        engineering_preflight=str(source.relative_to(HERE)),
        review_milestones=[1000,2000,5000,10000,20000,35000,55000])
    # These actual learning/data/model settings must stay exactly unchanged.
    excluded={'preflight_only','prepared_output','training_status','phase_plan','validation','run_control'}
    for key in set(c)-excluded:assert c[key]==effective['config'][key],key
    assert {k:v for k,v in c['phase_plan'].items() if k!='budget_status'}=={k:v for k,v in effective['config']['phase_plan'].items() if k!='budget_status'}
    for key,value in effective['config']['validation'].items():
        if key!='scope':assert c['validation'][key]==value,key
    assert c['physical_guidance']['enabled'] and c['physical_guidance']['minimum_alpha']==.25
    assert sha256(HERE/c['mount_defaults'])==effective['mount_defaults_sha256']
    write_json(ROOT/'config.json',c);assert read_config(ROOT/'config.json')==c
    destination=ROOT/'prepared';destination.mkdir(exist_ok=False)
    promoted=copy.deepcopy(effective);promoted['config']=c
    promoted['activation']=dict(type='administrative_only',source=str(source),source_effective_sha256=sha256(source/'effective_config.json'),
        numeric_checks_reused=True,calibration_reused=True,root_training_source_unchanged=True,
        data_rehash_required_by_trainer=True,activated_at=datetime.datetime.now().astimezone().isoformat())
    write_json(destination/'effective_config.json',promoted)
    report=copy.deepcopy(checks)
    report.update(training_ready=True,preflight_only=False,training_started=False,optimizer_steps=0,
        activation=promoted['activation'],validation_monitoring_scope=c['validation']['scope'],
        active_monitoring=dict(ego_windows=6,first_update_fixed_windows=3,
            original_checks_cover_both_ego_exo=True))
    write_json(destination/'input_checks.json',report)
    plan=json.loads((source/'validation_plan.json').read_text())
    plan['plan']=[r for r in plan['plan'] if r['mode']=='ego'];plan['activation_note']='Ego phase monitor subset; first-step fixed subset controlled by sealed wrapper.'
    write_json(destination/'validation_plan.json',plan)
    write_json(ROOT/'ACTIVATION_AUDIT.json',dict(status='passed',
        reasoning='All unchanged engineering/data/finite-gradient checks passed;128-step small-pool triage was exploratory and is not proof of long-run failure. Convergence moves to recorded1000-step review, not a guarantee.',
        changed=['preflight-only administrative flag','output location/status metadata','ego-only cheaper monitoring','review wrapper registration'],
        unchanged=['all model parameters/architecture','from-zero seed','gaitV2 objective/coefficient calibration','B2FP32','official fullTRAIN data/sampler','LR1e-4/500warmup/55k horizon','physics enabled/13DDIM-eligible steps','IK6','query backend exactGPU'],
        numeric_checks_repeated=False,source_seal_sha256=sha256(source/'effective_config.json'),
        wrapper_sha256=c['run_control']['wrapper_sha256'],no_old_checkpoint_initialization=True,
        formal_training_ready=True,first_review=1000,max_steps=55000))
    print('Activated unchanged audited configuration; full data will be revalidated by trainer.',flush=True)


if __name__=='__main__':main()
