"""Compare reconstructed legacy train/VAL pools and explicit recovery gate."""
import datetime
import json
from pathlib import Path
import sys
import types
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.source_recovery_contract import validate_source_recovery

ROOT=Path(__file__).resolve().parent


def main():
    old=HERE/'runs/native_dynamic_scene20_contact_55k_oct07'
    config=json.loads((old/'config.json').read_text())
    reports={}
    for split,key in (('train','train_summary'),('validation','validation_summary')):
        data=NativeBodyData(split,seed=config['seed'],skeleton_profile=config['skeleton_profile'],
            rich_source=config['rich_source'],trumans_scene_manifest=config['trumans_scene_manifest'],
            trumans_window_protocol=config['trumans_window_protocol'],contact_root=config['rich_contact_root'],
            temporal_scene_manifest=config['temporal_scene_manifest'])
        summary=data.summary()
        reports[split]=dict(current=summary,historical=config[key],matches=json.loads(json.dumps(summary))==config[key])
        print(split,reports[split]['matches'],flush=True)
    try:
        validate_source_recovery(HERE/'runs/soft_body_turn_robust_oct09/source_recovery_audit.json',config['source_hash'],types.SimpleNamespace(**config))
        recovery=dict(status='passed',note='Explicit reconstructed-source transfer gate, not identical original source fingerprint.')
    except Exception as exc:
        recovery=dict(status='blocked',reason=repr(exc),note='Do not bypass or relabel old source hash.')
    report=dict(observed_at=datetime.datetime.now().astimezone().isoformat(),pools=reports,source_recovery_gate=recovery,
        original_source_hash=config['source_hash'],optimizer_steps=0,
        status='legacy_train_recipe_reconstructable' if all(x['matches'] for x in reports.values()) and recovery['status']=='passed' else 'partial_reproduction_only',
        limitation='Matching pool counts and four inference cases do not prove bitwise55000-step retraining. Legacy split remains development-only.')
    (ROOT/'OLD55K_TRAIN_RECIPE_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(report['status'],recovery,flush=True)


if __name__=='__main__':main()
