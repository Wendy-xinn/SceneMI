"""One bounded final trial: train the original55k on constrained rollout states.

TRAIN body truth is a target only. Runtime root path is inferred from known
head plus the original generated initial heading; it is not a pelvis sensor.
"""
import argparse, hashlib, json, math, shutil, time
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI, cosine_alphas
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.known_input_generation import camera_inputs_only, generate_without_body_initialization
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.supervision import supervised_losses, rotation_from_6d, IDENTITY_6D
from experiments.offline_camera_retrain_v1.typed_head_condition import camera_to_head
from experiments.offline_camera_retrain_v1.turn_manifold_sampling import project_turn_head, inferred_turn_path
from experiments.offline_camera_retrain_v1.fresh55k_replay_objective import replay_losses
from experiments.offline_camera_retrain_v1.sole_support_objective import SoleCache
from experiments.offline_camera_retrain_v1.contact_supervision import contact_loss
from experiments.offline_camera_retrain_v1.training_contract_v2 import duration_schedule
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint

H=Path(__file__).parent
OUT=H/'runs/projected_replay_final_oct11'
BASE=H/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt'

def observed_head(known):
    mount=json.loads((H/'runs/hierarchical_constraint_oct11/TRAIN_mount_calibration.json').read_text())
    head=camera_to_head(known['camera'], torch.eye(3,device=known['camera'].device), known['camera'].new_tensor(mount['translation_m']))
    return head[...,:3]*2, rotation_from_6d(head[...,3:]-head.new_tensor(IDENTITY_6D))

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1200);args=p.parse_args()
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    if (OUT/'last.pt').exists():raise FileExistsError('Final trial already saved; do not overwrite')
    seed=2026101151;torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed+1)
    cp=torch.load(BASE,map_location='cpu',weights_only=False);c=cp['config']
    def build():
        m=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda();m.load_state_dict(cp['model']);return m
    teacher=build().eval().requires_grad_(False);model=build().train()
    digest=hashlib.sha256()
    for k,v in sorted(cp['model'].items()):digest.update(k.encode());digest.update(v.numpy().tobytes())
    del cp
    data=NativeBodyData('train',seed=seed,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'],window_sampling='duration')
    schedule,plan=duration_schedule(data,args.steps,seed,.5);sole=SoleCache();alpha_table=cosine_alphas().cuda()
    protocol=dict(status='running',base=str(BASE),base_sha256=digest.hexdigest(),steps=args.steps,frames=128,batch_size=2,seed=seed,lr=2e-5,
        initialization='original55k only',training='50% ordinary clean-target diffusion; 50% original TRAIN all-noise generation projected to inferred root/head path then re-noised t<300. Student raw supervised to TRAIN target; projected student geometry also supervised for native stance/swing/landing and rotation coordination.',
        inference='all-noise DDIM20; constrained x0 last10 steps; original55k supplies generated initial root offset, never GT root. Raw sampling is diagnostic.',
        observed_mount='fixed TRAIN-only synthetic TRUMANS mount; not real PV calibration',
        limitations='inferred heading hypothesis assumes head/body turn coupling; not valid universally; planar foot geometry targets do not certify full body collision',
        acceptance='both850/906 two seeds: root mean error at least20% lower than original; GT stance and native foot stance slide <=original*1.02, ankle crossing <=original+.02; fixed broader development panel aggregate same non-regression gates. No further tuning if failed.',
        sampling='projected replay batches use TRAIN TRUMANS only: the TRAIN synthetic mount is not real EgoBody PV calibration. Remaining batches follow original duration sampler.',
        free_disk_before=shutil.disk_usage(OUT).free,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.monotonic();opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    with (OUT/'trace.jsonl').open('w') as log:
        for step,(group,_) in enumerate(schedule,1):
            replay=bool(rng.random()<.5);scene=bool(rng.random()>=.1)
            if replay:group='trumans'
            pairs=[data.sample(128,group) for _ in range(2)];ids=[i for s,i in pairs];batch={k:v.cuda() for k,v in collate([s for s,i in pairs]).items()};known=camera_inputs_only(batch);truth=batch['motion']
            t=torch.randint(300 if replay else 1000,(2,),device='cuda');a=alpha_table[t][:,None,None];noise=torch.randn_like(truth);basis=truth;eligible=False
            with torch.no_grad():
                hp,hr=observed_head(known)
                if replay:
                    generated=generate_without_body_initialization(teacher,known,seed=seed+step*17,use_scene=scene)
                    try:
                        path=inferred_turn_path(generated,hr);basis=project_turn_head(generated,batch['rest'],hp,hr,path);eligible=True
                    except ValueError:basis=generated
                target_soles=sole.soles(truth,ids)
            noisy=a.sqrt()*basis+(1-a).sqrt()*noise;mask=fixed_control_mask(2,128,'head','cuda');opt.zero_grad(set_to_none=True)
            pred=model(noisy,t,known,control_mask=mask,use_scene=scene)
            losses=supervised_losses(pred,truth,batch,profile='coordination_v1',signal_weight=a.flatten())
            with torch.no_grad():old=teacher(noisy,t,known,control_mask=mask,use_scene=scene)
            physical_pred=project_turn_head(pred,batch['rest'],hp,hr,path) if eligible else pred
            extra=replay_losses(physical_pred,truth,batch['rest'],sole.soles(physical_pred,ids),target_soles,old,a.flatten())
            bce=contact_loss(model.contact_logits(pred,known,use_scene=scene),batch,a.flatten())
            # Bound unusually bad generated poses; retain useful direction of
            # the gradient without one invalid rollout dominating the update.
            repair=5*torch.log1p(extra['replay_total']/5)
            total=losses['total']+repair+.1*bce
            if not torch.isfinite(total):raise FloatingPointError('Nonfinite final-trial loss')
            total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            lr=2e-5*min(1,step/50)*max(.1,.5*(1+math.cos(math.pi*step/args.steps)))
            for g in opt.param_groups:g['lr']=lr
            opt.step();row=dict(step=step,group=group,identities=ids,t=t.tolist(),replay=replay,projection_applied=eligible,use_scene=scene,loss=float(total.detach()),repair_loss=float(repair.detach()),gradient_norm=float(norm),lr=lr)
            if step==1 or step%25==0:
                row['terms']={k:float(v.detach()) for k,v in extra.items()};print(step,round(row['loss'],4),'elapsed_s',round(time.monotonic()-start,1),flush=True)
            log.write(json.dumps(row)+'\n');log.flush()
    config=dict(c,trial='projected_replay_final',trial_steps=args.steps,trial_seed=seed,duration_sampling_plan=plan,body_input_initialization='noise only')
    save_checkpoint(dict(step=args.steps,model=model.state_dict(),config=config,evaluation_only=True,optimizer_removed=True,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampler_state=rng.bit_generator.state),OUT/'last.pt')
    protocol.update(status='training_complete',elapsed_s=time.monotonic()-start,free_disk_after=shutil.disk_usage(OUT).free);(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print('TRAIN COMPLETE',flush=True)

if __name__=='__main__':main()
