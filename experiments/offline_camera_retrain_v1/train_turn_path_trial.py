"""Two matched short trials from official55k; head-only known inputs, no body prefix."""
import argparse,json,time,hashlib,shutil,math
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,cosine_alphas
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.known_input_generation import camera_inputs_only
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.supervision import supervised_losses
from experiments.offline_camera_retrain_v1.contact_supervision import contact_loss
from experiments.offline_camera_retrain_v1.training_contract_v2 import duration_schedule
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.turn_path_objective import turn_path_losses
H=Path(__file__).parent;OUT=H/'runs/turn_path_training_oct10'
def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=600);p.add_argument('--support-refinement',action='store_true');args=p.parse_args()
    global OUT
    if args.support_refinement:OUT=H/'runs/sole_support_oct11'
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);seed=2026101031
    cp=torch.load(H/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt',map_location='cpu',weights_only=False);c=cp['config']
    teacher=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();teacher.load_state_dict(cp['model']);teacher.requires_grad_(False)
    turn_teacher=None
    if args.support_refinement:
        del cp
        cp=torch.load(H/'runs/turn_path_training_oct10/turn_path/last.pt',map_location='cpu',weights_only=False);c=cp['config'];seed=2026101107
        turn_teacher=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();turn_teacher.load_state_dict(cp['model']);turn_teacher.requires_grad_(False)
        from experiments.offline_camera_retrain_v1.sole_support_objective import SoleCache,sole_support_losses
        sole_cache=SoleCache()
    protocol=dict(status='registered_before_training',steps=args.steps,batch_size=4,lr=2e-5,warmup_steps=50,seed=seed,variants=['continued_control','turn_path'],initialization='original55k; no body GT conditioning or prefix',condition='head-only raw camera and exact20 native scene/body template; no hard head projection',lengths=[64,128,192],duration_alpha=.5,low_noise_fraction=.5,optimizer='AdamW weight_decay .01; clip1',source_sha256=hashlib.sha256((H/'turn_path_objective.py').read_bytes()).hexdigest(),guard='training target local root/leg step + 3deg margin; not universal physical joint limits',acceptance='paired full128: turning and torso/leg spikes improve, feet/world/PA do not regress; fixed850/906 must show improvement; exposed panel not independent holdout',storage='one final evaluation-only last.pt per arm; no optimizer or numbered/best checkpoints',free_disk_before=shutil.disk_usage(H).free)
    if args.support_refinement:
        protocol.update(variants=['turn_control','sole_support'],initialization='600-step turn_path candidate; both arms same; original55k foot teacher',source_sha256=hashlib.sha256((H/'sole_support_objective.py').read_bytes()).hexdigest(),support='native sole centroid LBS/pose correctives; clean training support envelope and stance; no scene floor or inference GT projection',acceptance='fixed protocol in pre_registered_evaluation.json; preserve 906 turn direction and improve support/feet on exposed panel')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    started=time.monotonic();alpha_table=cosine_alphas().cuda()
    for variant in protocol['variants']:
        torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed+991)
        data=NativeBodyData('train',seed=seed,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'],window_sampling='duration')
        schedule,plan=duration_schedule(data,args.steps,seed,.5)
        model=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda();model.load_state_dict(cp['model']);model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
        folder=OUT/variant;folder.mkdir(exist_ok=True);arm_start=time.monotonic()
        with (folder/'trace.jsonl').open('w') as trace:
            for step,(group,length) in enumerate(schedule,1):
                pairs=[data.sample(length,group) for _ in range(4)];batch={k:v.cuda() for k,v in collate([x[0] for x in pairs]).items()};known=camera_inputs_only(batch)
                clean=batch['motion'];t=torch.randint(1000,(4,),device='cuda');t=torch.where(torch.rand(4,device='cuda')<.5,torch.randint(100,(4,),device='cuda'),t);alpha=alpha_table[t][:,None,None];noisy=alpha.sqrt()*clean+(1-alpha).sqrt()*torch.randn_like(clean)
                scene=bool(rng.random()>=.1);mask=fixed_control_mask(4,length,'head','cuda');optimizer.zero_grad(set_to_none=True)
                pred=model(noisy,t,known,control_mask=mask,use_scene=scene)
                losses=supervised_losses(pred,clean,batch,profile='coordination_v1',signal_weight=alpha.flatten())
                losses['contact_bce']=contact_loss(model.contact_logits(pred,known,use_scene=scene),batch,alpha.flatten());total=losses['total']+.1*losses['contact_bce']
                if variant!='continued_control':
                    with torch.no_grad():old=teacher(noisy,t,known,control_mask=mask,use_scene=scene)
                    extra=turn_path_losses(pred,clean,batch['rest'],old,alpha.flatten());total=total+extra['turn_total'];losses.update(extra)
                if variant=='sole_support':
                    with torch.no_grad():
                        turn_old=turn_teacher(noisy,t,known,control_mask=mask,use_scene=scene);true_soles=sole_cache.soles(clean,[x[1] for x in pairs])
                    predicted_soles=sole_cache.soles(pred,[x[1] for x in pairs]);support=sole_support_losses(pred,clean,predicted_soles,true_soles,turn_old,alpha.flatten());total=total+support['sole_total'];losses.update(support)
                assert torch.isfinite(total);total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
                lr=2e-5*min(1,step/50)*max(.1,.5*(1+math.cos(math.pi*step/args.steps)))
                for g in optimizer.param_groups:g['lr']=lr
                optimizer.step()
                row=dict(step=step,group=group,length=length,identities=[x[1] for x in pairs],t=t.tolist(),use_scene=scene,loss=float(total.detach()),gradient_norm=float(norm),lr=lr)
                if step%25==0 or step==1:row['terms']={k:float(v.detach()) for k,v in losses.items()};print(variant,step,round(row['loss'],5),'elapsed',round(time.monotonic()-arm_start,1),flush=True)
                trace.write(json.dumps(row)+'\n');trace.flush()
        config=dict(c,turn_trial=variant,trial_steps=args.steps,trial_seed=seed,loss_profile='coordination_v1',known_input_protocol='camera_inputs_only; no body prefix',duration_sampling_plan=plan)
        save_checkpoint(dict(step=args.steps,model=model.state_dict(),config=config,evaluation_only=True,optimizer_removed=True,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampler_state=rng.bit_generator.state),folder/'last.pt')
        print('SAVED',variant,flush=True);del model,optimizer;torch.cuda.empty_cache()
    protocol.update(status='training_completed',elapsed_s=time.monotonic()-started,free_disk_after=shutil.disk_usage(H).free);(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
if __name__=='__main__':main()
