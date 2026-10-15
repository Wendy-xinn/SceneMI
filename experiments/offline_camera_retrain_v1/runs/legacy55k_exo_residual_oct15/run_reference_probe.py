"""Reference-preserving residual fusion; frozen55k is auxiliary, not initializer overwrite.

Second declared development pilot, after the first initialization protocol failed.
Same source windows and draw indices; factorial controls isolate prior and scene.
"""
import copy
import datetime
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
PARENT=Path(__file__).resolve().parent
sys.path.insert(0,str(PARENT));sys.path.insert(0,str(PARENT.parents[3]))
import numpy as np
import torch
from torch import nn
import run_probe as common
from experiments.offline_camera_retrain_v1.legacy55k_exo_residual import ExoResidual,scene_features,residual_inputs,correction_payload
from experiments.offline_camera_retrain_v1.prepare_native30 import write_json,sha256,code_hashes
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

ROOT=PARENT/'reference_anchor'
ARMS={'observation_only':(False,False),'scene_only':(False,True),
      'prior_only':(True,False),'prior_scene':(True,True)}


class ReferenceResidual(ExoResidual):
    def __init__(self,width=192):
        super().__init__(width=width)
        self.input=nn.Conv1d(377+201+528,width,1)

    def forward(self,known,use_scene=True,use_prior=True):
        if not use_prior:
            known=dict(known,features=torch.cat((known['features'][...,:377],torch.zeros_like(known['features'][...,377:])), -1))
        return super().forward(known,use_scene=use_scene)


@torch.no_grad()
def build_banks(config):
    old=torch.load(PARENT/'replay_inputs.pt',map_location='cpu',weights_only=False)
    identities=json.loads((PARENT/'IDENTITIES.json').read_text())
    builder=common.Native30ObservationBuilder(common.Native30SkinAnchorCache(),
        json.loads((common.HERE/config['mount_defaults']).read_text()),config['observation']['input_noise'])
    banks={}
    for split in ('train','validation'):
        data=common.dataset(config,split);features=[]
        for i,row in enumerate(identities[split]):
            sample,identity=data.sample(row['group'],'exo',member=row['member'],start=row['start'],yaw=.6)
            prepared,scene,asset=builder.build(data,sample,identity,'exo',
                noise_rng=np.random.default_rng(row['seed']) if row['regime']=='noisy' else None)
            # Assert identical observation stream and coordinate/body protocol.
            for key in ('motion','rest','trajectory','observation_meta'):
                np.testing.assert_array_equal(prepared[key],old[split][key][i].numpy())
            j=forward_kinematics(old[split]['reference'][i:i+1],old[split]['rest'][i:i+1])[0].numpy()
            features.append(torch.from_numpy(scene_features(scene,j))[None])
            if (i+1)%16==0:print('reference scene prepared',split,i+1,flush=True)
        bank={k:v.cuda() for k,v in old[split].items()}
        original_prior=bank['initial']
        known=residual_inputs(bank['reference'],bank,torch.cat(features).cuda())
        known['features']=torch.cat((known['features'],original_prior-bank['reference']),-1)
        bank.update(known);bank['legacy_prior']=original_prior
        banks[split]=bank
        del data
    return banks,identities


@torch.no_grad()
def evaluate(model,bank,identities,use_prior,use_scene,tag):
    model.eval();rows=[];outputs=[]
    for i,row in enumerate(identities):
        batch=common.subset(bank,slice(i,i+1))
        p,_=model(correction_payload(batch),use_scene=use_scene,use_prior=use_prior)
        rows.append(dict(row,tag=tag,metrics=common.metrics(p,batch)));outputs.append(p.cpu())
    return dict(cases=rows,aggregate=common.aggregate(rows)),torch.cat(outputs)


def main():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'PROTOCOL.json').exists(),'Existing trial must not be overwritten'
    parent=json.loads((PARENT/'RESULTS.json').read_text());assert parent['status']=='not_passed_do_not_extend'
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.28);torch.manual_seed(common.SEED)
    config=common.read_config(common.HERE/'native30_fresh_ego_config.json')
    source_hash=sha256(common.OLD)
    protocol=dict(status='preparing',created_at=datetime.datetime.now().astimezone().isoformat(),
        purpose='Preserve observation reference; frozen55k absolute prediction is auxiliary prior delta. No overwriting visible reference pose.',
        development_followup_to=str(PARENT/'RESULTS.json'),same_validation_panel=True,
        seed=common.SEED,steps_per_arm=1000,batch_size=8,lr=2e-4,precision='FP32',fps=30,frames=192,
        arms=ARMS,paired='same weights, batch plan, observation streams, reference, frozen prior and geometry',
        legacy_checkpoint=str(common.OLD),legacy_checkpoint_sha256=source_hash,legacy_step=55000,
        legacy_official_TEST_overlap_recordings=12,official_TEST_examples_used=False,
        scope='Legacy-prior development; same VAL panel reused for declared protocol correction; not independent confirmation/paper benchmark',
        reference='visible-position-only reference_from_observations; not a real RGB frontend',
        prior='Existing frozen55k absolute-x0 projection, auxiliary delta only; same original seed and no GT control',
        input_noise='unchanged original clean/noisy streams; all128 TRAIN and32 VAL feature windows retained',
        geometry='Exact nearest observed static/dynamic patches queried once at reference, excluding target100, no completed floor',
        objectives='unchanged parent loss/base+30Hz gait+2 hiddenFK+2 observedFK+.001 residual prior',
        gate=dict(occluded_gain_min=.05,scene_extra_gain_min=.02,prior_extra_gain_min=.02,
            max_visible_regression_mm=5,max_foot_velocity_relative_regression=.10),
        budget='1000 updates each of four factorial arms; no automatic extension; max900s per arm',
        code_sha256=code_hashes(),runner_sha256=sha256(__file__),
        source_data_audit_sha256=sha256(PARENT/'DATA_AUDIT.json'),parent_replay_inputs_sha256=sha256(PARENT/'replay_inputs.pt'))
    write_json(ROOT/'PROTOCOL.json',protocol)
    banks,identities=build_banks(config)
    write_json(ROOT/'IDENTITIES.json',identities)
    save_checkpoint({s:{k:v.cpu() for k,v in b.items()} for s,b in banks.items()},ROOT/'replay_inputs.pt')
    rows=[dict(row,metrics=common.metrics(banks['validation']['initial'][i:i+1],common.subset(banks['validation'],slice(i,i+1))))
        for i,row in enumerate(identities['validation'])]
    baseline=dict(cases=rows,aggregate=common.aggregate(rows))
    write_json(ROOT/'BASELINE.json',baseline)
    torch.manual_seed(common.SEED);template=ReferenceResidual().cuda();initial=copy.deepcopy(template.state_dict());del template
    draws=json.loads((PARENT/'BATCH_PLAN.json').read_text())['indices']
    protocol.update(status='training',training_started_at=datetime.datetime.now().astimezone().isoformat())
    write_json(ROOT/'PROTOCOL.json',protocol)
    reports={};outputs={};step_counts={}
    for arm,(use_prior,use_scene) in ARMS.items():
        folder=ROOT/arm;folder.mkdir(exist_ok=True)
        model=ReferenceResidual().cuda();model.load_state_dict(initial)
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01);began=time.monotonic();times=[]
        for step,ids in enumerate(draws,1):
            torch.cuda.synchronize();tick=time.monotonic();model.train()
            batch=common.subset(banks['train'],ids);opt.zero_grad(set_to_none=True)
            p,details=model(correction_payload(batch),use_prior=use_prior,use_scene=use_scene)
            loss,terms=common.fit_loss(p,batch,details);assert torch.isfinite(loss)
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            torch.cuda.synchronize();elapsed=time.monotonic()-tick;times.append(elapsed)
            row=dict(step=step,loss=float(loss.detach()),grad=float(grad),seconds=elapsed,sample_indices=ids,
                **{k:float(v.detach()) for k,v in terms.items()})
            with (folder/'training_log.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            if step%100==0 or step==1:
                print(arm,step,row['loss'],elapsed,flush=True)
                save_checkpoint(dict(step=step,model=model.state_dict(),optimizer=opt.state_dict(),config=protocol,
                    torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),
                    numpy_rng_state=np.random.get_state(),batch_plan_next_index=step),folder/'last.pt')
            if time.monotonic()-began>900:break
        step_counts[arm]=step
        save_checkpoint(dict(step=step,model=model.state_dict(),optimizer=opt.state_dict(),config=protocol,
            torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),
            numpy_rng_state=np.random.get_state(),batch_plan_next_index=step),folder/'last.pt')
        reports[arm],outputs[arm]=evaluate(model,banks['validation'],identities['validation'],use_prior,use_scene,arm)
        reports[arm].update(step=step,mean_update_seconds=float(np.mean(times)),training_seconds=time.monotonic()-began,
            trainable_parameters=sum(p.numel() for p in model.parameters()))
        write_json(folder/'VALIDATION.json',reports[arm])
        if arm=='prior_scene':
            for tag,prior_on,scene_on in (('scene_input_OFF',True,False),('prior_input_OFF',False,True)):
                reports[tag],outputs[tag]=evaluate(model,banks['validation'],identities['validation'],prior_on,scene_on,tag)
                write_json(folder/(tag+'.json'),reports[tag])
        del model,opt;torch.cuda.empty_cache()
    verdict={}
    for regime in ('clean','noisy'):
        base=baseline['aggregate'][regime];on=reports['prior_scene']['aggregate'][regime]
        no_scene=reports['prior_only']['aggregate'][regime];no_prior=reports['scene_only']['aggregate'][regime]
        checks=dict(occluded_improvement=on['occluded_mm']<=base['occluded_mm']*.95,
            extra_scene_improvement=on['occluded_mm']<=no_scene['occluded_mm']*.98,
            extra_prior_improvement=on['occluded_mm']<=no_prior['occluded_mm']*.98,
            visible_preserved=on['visible_mm']<=base['visible_mm']+5,
            foot_velocity_preserved=on['foot_velocity_rmse_m_s']<=base['foot_velocity_rmse_m_s']*1.10)
        verdict[regime]=dict(passed=all(checks.values()),checks=checks,initial=base,
            **{k:v['aggregate'][regime] for k,v in reports.items()})
    assert code_hashes()==protocol['code_sha256']
    assert sha256(common.OLD)==source_hash
    paired=all(s==1000 for s in step_counts.values())
    status='passed_pilot_requires_expanded_validation' if paired and all(v['passed'] for v in verdict.values()) else 'not_passed_do_not_extend'
    write_json(ROOT/'RESULTS.json',dict(status=status,matched_full_budget=paired,step_counts=step_counts,verdict=verdict,
        old55k_unchanged=True,all_sources_unchanged=True,official_TEST_used=False,initialization_label='Observation reference; frozen55k auxiliary prior',
        limitation='Same small development VAL reused; legacy prior contains TEST overlap; ideal tracker; not publication acceptance'))
    # Common replay keys show initializer vs prior_only vs prior_scene, all on the
    # same body/world/frame. All four factorial predictions are retained too.
    outputs.update(no_scene=outputs['prior_only'],scene=outputs['prior_scene'])
    save_checkpoint(dict(outputs=outputs,truth=banks['validation']['motion'].cpu(),rest=banks['validation']['rest'].cpu(),
        initial=banks['validation']['initial'].cpu(),reference=banks['validation']['reference'].cpu(),
        missing=banks['validation']['natural_missing'].cpu()),ROOT/'predictions.pt')
    protocol.update(status='completed',result_status=status,completed_at=datetime.datetime.now().astimezone().isoformat())
    write_json(ROOT/'PROTOCOL.json',protocol);print('COMPLETE',status,flush=True)


if __name__=='__main__':main()
