"""Declared posthoc inference guard, with zero further optimizer updates."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
import run_reference_probe as trial
import run_probe as common
from protected_residual import protected_residual
from experiments.offline_camera_retrain_v1.legacy55k_exo_residual import correction_payload
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.prepare_native30 import write_json,sha256
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint

ROOT=Path(__file__).resolve().parent/'reference_anchor'


@torch.no_grad()
def main():
    protocol=json.loads((ROOT/'GUARD_PROTOCOL.json').read_text())
    assert protocol['optimizer_updates']==0
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.28)
    bank=torch.load(ROOT/'replay_inputs.pt',map_location='cpu',weights_only=False)['validation']
    bank={k:v.cuda() for k,v in bank.items()}
    identities=json.loads((ROOT/'IDENTITIES.json').read_text())['validation']
    reports={};outputs={};maximum=0.
    for arm,(use_prior,use_scene) in trial.ARMS.items():
        state=torch.load(ROOT/arm/'last.pt',map_location='cpu',weights_only=False)
        model=trial.ReferenceResidual().cuda().eval();model.load_state_dict(state['model']);del state
        rows=[];poses=[]
        for i,row in enumerate(identities):
            batch=common.subset(bank,slice(i,i+1))
            _,details=model(correction_payload(batch),use_prior=use_prior,use_scene=use_scene)
            available=batch['observation_meta'][...,0]>0
            p=protected_residual(batch['initial'],details,available,batch['rest'])
            displacement=(forward_kinematics(p,batch['rest'])-forward_kinematics(batch['initial'],batch['rest'])).norm(dim=-1)
            moved=float(displacement[available].max()) if available.any() else 0.
            maximum=max(maximum,moved);assert moved<2e-6
            rows.append(dict(row,metrics=common.metrics(p,batch),available_joint_max_displacement_m=moved));poses.append(p.cpu())
        reports[arm]=dict(cases=rows,aggregate=common.aggregate(rows));outputs[arm]=torch.cat(poses)
        write_json(ROOT/arm/'GUARDED_VALIDATION.json',reports[arm])
        del model
    baseline=json.loads((ROOT/'BASELINE.json').read_text())
    verdict={}
    for regime in ('clean','noisy'):
        base=baseline['aggregate'][regime];on=reports['prior_scene']['aggregate'][regime]
        off=reports['prior_only']['aggregate'][regime];noprior=reports['scene_only']['aggregate'][regime]
        checks=dict(occluded_improvement=on['occluded_mm']<=base['occluded_mm']*.95,
            extra_scene_improvement=on['occluded_mm']<=off['occluded_mm']*.98,
            extra_prior_improvement=on['occluded_mm']<=noprior['occluded_mm']*.98,
            visible_preserved=on['visible_mm']<=base['visible_mm']+5,
            foot_velocity_preserved=on['foot_velocity_rmse_m_s']<=base['foot_velocity_rmse_m_s']*1.10)
        verdict[regime]=dict(checks=checks,passed=all(checks.values()),initial=base,
            **{k:v['aggregate'][regime] for k,v in reports.items()})
    outputs.update(no_scene=outputs['prior_only'],scene=outputs['prior_scene'])
    save_checkpoint(dict(outputs=outputs,truth=bank['motion'].cpu(),rest=bank['rest'].cpu(),
        initial=bank['initial'].cpu(),reference=bank['reference'].cpu(),missing=bank['natural_missing'].cpu()),ROOT/'predictions_protected.pt')
    write_json(ROOT/'GUARDED_RESULTS.json',dict(status='passed_exploratory_only' if all(v['passed'] for v in verdict.values()) else 'not_passed_do_not_extend',
        verdict=verdict,optimizer_updates=0,available_joint_max_displacement_m=maximum,
        initializer='observation reference',method='strict unobserved-descendant-branch guard',
        guard_source_sha256=sha256(Path(__file__).parent/'protected_residual.py'),
        limitation='Posthoc development guard chosen after first raw control validation; fixes only fully hidden branches, not joints between visible anchors; no independent confirmation'))
    print('GUARDED RESULTS',json.dumps(verdict))


if __name__=='__main__':main()
