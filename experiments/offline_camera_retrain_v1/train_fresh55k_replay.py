"""Matched trials initialized only from the official55k checkpoint."""
import argparse,json,time,hashlib,shutil,math
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,cosine_alphas
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.known_input_generation import camera_inputs_only,generate_without_body_initialization
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.supervision import supervised_losses
from experiments.offline_camera_retrain_v1.contact_supervision import contact_loss
from experiments.offline_camera_retrain_v1.training_contract_v2 import duration_schedule
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.sole_support_objective import SoleCache
from experiments.offline_camera_retrain_v1.fresh55k_replay_objective import replay_losses
H=Path(__file__).parent;O=H/'runs/fresh55k_replay_oct11';BASE=H/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt'
def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=600);args=p.parse_args();torch.set_num_threads(4);O.mkdir(exist_ok=True)
    if any(O.rglob('last.pt')):raise FileExistsError('Do not overwrite a scored trial')
    seed=2026101121;cp=torch.load(BASE,map_location='cpu',weights_only=False);c=cp['config']
    def build():
        model=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda();model.load_state_dict(cp['model']);return model
    teacher=build().eval().requires_grad_(False);sole=SoleCache();alpha_table=cosine_alphas().cuda()
    digest=hashlib.sha256()
    for name,t in sorted(cp['model'].items()):digest.update(name.encode());digest.update(t.numpy().tobytes())
    protocol=dict(status='training_running',base=str(BASE),base_model_sha256=digest.hexdigest(),variants=['denoise_control','rollout_replay'],steps=args.steps,batch_size=2,frames=128,seed=seed,lr=2e-5,initialization='BOTH original55k; no previous candidate weights',replay='50% batches: frozen original55k DDIM20 all-noise generation from TRAIN known inputs, then re-noise at t<100; control uses clean TRAIN target with same noise/t; remaining common standard diffusion',supervision='same turn+native sole support+head/neck+stance displacement; no frozen old foot position, no GT body input',storage='one final evaluation-only last.pt per arm; no optimizer/best/numbered weights',source_sha256={f:hashlib.sha256((H/f).read_bytes()).hexdigest() for f in ['train_fresh55k_replay.py','fresh55k_replay_objective.py']},free_disk_before=shutil.disk_usage(O).free)
    (O/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.monotonic()
    for variant in protocol['variants']:
        torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed+991)
        data=NativeBodyData('train',seed=seed,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'],window_sampling='duration')
        schedule,plan=duration_schedule(data,args.steps,seed,.5);model=build().train();opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01);folder=O/variant;folder.mkdir();arm=time.monotonic()
        with (folder/'trace.jsonl').open('w') as log:
            for step,(group,_) in enumerate(schedule,1):
                pairs=[data.sample(128,group) for _ in range(2)];batch={k:v.cuda() for k,v in collate([s for s,i in pairs]).items()};known=camera_inputs_only(batch);clean=batch['motion'];scene=bool(rng.random()>=.1);replay=bool(rng.random()<.5)
                t=torch.randint(100 if replay else 1000,(2,),device='cuda');alpha=alpha_table[t][:,None,None];noise=torch.randn_like(clean);basis=clean
                if replay and variant=='rollout_replay':
                    basis=generate_without_body_initialization(teacher,known,seed=seed+step*17,steps=20,use_scene=scene)
                noisy=alpha.sqrt()*basis+(1-alpha).sqrt()*noise;mask=fixed_control_mask(2,128,'head','cuda');opt.zero_grad(set_to_none=True)
                pred=model(noisy,t,known,control_mask=mask,use_scene=scene);losses=supervised_losses(pred,clean,batch,profile='coordination_v1',signal_weight=alpha.flatten())
                with torch.no_grad():old=teacher(noisy,t,known,control_mask=mask,use_scene=scene);target_soles=sole.soles(clean,[i for s,i in pairs])
                extra=replay_losses(pred,clean,batch['rest'],sole.soles(pred,[i for s,i in pairs]),target_soles,old,alpha.flatten());bce=contact_loss(model.contact_logits(pred,known,use_scene=scene),batch,alpha.flatten());total=losses['total']+extra['replay_total']+.1*bce
                assert torch.isfinite(total);total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);lr=2e-5*min(1,step/50)*max(.1,.5*(1+math.cos(math.pi*step/args.steps)))
                for g in opt.param_groups:g['lr']=lr
                opt.step();row=dict(step=step,group=group,length=128,identities=[i for s,i in pairs],t=t.tolist(),noise_checksum=float(noise.sum()),use_scene=scene,replay_schedule=replay,replay_applied=replay and variant=='rollout_replay',loss=float(total.detach()),gradient_norm=float(norm),lr=lr)
                if step==1 or step%25==0:row['terms']={k:float(v.detach()) for k,v in extra.items()};print(variant,step,round(row['loss'],4),'elapsed',round(time.monotonic()-arm,1),flush=True)
                log.write(json.dumps(row)+'\n');log.flush()
        config=dict(c,turn_trial=variant,trial_steps=args.steps,trial_seed=seed,known_input_protocol='camera_inputs_only; no body prefix',duration_sampling_plan=plan)
        save_checkpoint(dict(step=args.steps,model=model.state_dict(),config=config,evaluation_only=True,optimizer_removed=True,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampler_state=rng.bit_generator.state),folder/'last.pt');print('SAVED',variant,flush=True);del model,opt;torch.cuda.empty_cache()
    protocol.update(status='training_completed',elapsed_s=time.monotonic()-start,free_disk_after=shutil.disk_usage(O).free);(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
if __name__=='__main__':main()
