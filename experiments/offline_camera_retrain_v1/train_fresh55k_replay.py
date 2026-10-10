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
    global O
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=600);p.add_argument('--guard-only',action='store_true');p.add_argument('--physics-only',action='store_true');p.add_argument('--coverage-only',action='store_true');p.add_argument('--scale-only',action='store_true');p.add_argument('--body-local-only',action='store_true');args=p.parse_args();torch.set_num_threads(4)
    if sum((args.guard_only,args.physics_only,args.coverage_only,args.scale_only,args.body_local_only))>1:p.error('Choose one trial')
    if args.scale_only or args.body_local_only:args.coverage_only=True
    if args.coverage_only:args.physics_only=True
    if args.guard_only:O=H/'runs/fresh55k_winding_oct11'
    if args.physics_only:O=H/'runs/fresh55k_scene_physics_oct11'
    if args.coverage_only:O=H/'runs/fresh55k_support_coverage_oct11'
    if args.scale_only or args.body_local_only:O=H/'runs/control_scale_body_scene_oct11'
    O.mkdir(exist_ok=True)
    expected_variant='body_local' if args.body_local_only else 'scale_calibrated'
    if args.scale_only or args.body_local_only:
        if (O/expected_variant/'last.pt').exists():raise FileExistsError('Do not overwrite scored arm')
    elif any(O.rglob('last.pt')):raise FileExistsError('Do not overwrite a scored trial')
    seed=2026101121;cp=torch.load(BASE,map_location='cpu',weights_only=False);c=cp['config']
    def build(local=False):
        cls=OfflineSceneMI
        if local:
            from experiments.offline_camera_retrain_v1.body_local_scene_model import BodyLocalSceneMI
            cls=BodyLocalSceneMI
        model=cls(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda();missing,unexpected=model.load_state_dict(cp['model'],strict=not local)
        if local:assert not unexpected and all(k.startswith('body_local_encoder.') for k in missing)
        return model
    teacher=build().eval().requires_grad_(False);sole=SoleCache();alpha_table=cosine_alphas().cuda()
    digest=hashlib.sha256()
    for name,t in sorted(cp['model'].items()):digest.update(name.encode());digest.update(t.numpy().tobytes())
    protocol=dict(status='training_running',base=str(BASE),base_model_sha256=digest.hexdigest(),variants=['denoise_control','rollout_replay'],steps=args.steps,batch_size=2,frames=128,seed=seed,lr=2e-5,initialization='BOTH original55k; no previous candidate weights',replay='50% batches: frozen original55k DDIM20 all-noise generation from TRAIN known inputs, then re-noise at t<100; control uses clean TRAIN target with same noise/t; remaining common standard diffusion',supervision='same turn+native sole support+head/neck+stance displacement; no frozen old foot position, no GT body input',storage='one final evaluation-only last.pt per arm; no optimizer/best/numbered weights',source_sha256={f:hashlib.sha256((H/f).read_bytes()).hexdigest() for f in ['train_fresh55k_replay.py','fresh55k_replay_objective.py']},free_disk_before=shutil.disk_usage(O).free)
    if args.guard_only or args.physics_only:
        protocol.update(variants=['winding_guard'],initialization='original55k ONLY; replay600 is comparison, NEVER initialization',guard='head/root winding at all noise with .25+.75alpha weight; top10% root/neck/head/legs relative-rotation and excess-rate supervision',source_guard_sha256=hashlib.sha256((H/'winding_guard_objective.py').read_bytes()).hexdigest())
    if args.physics_only:
        from experiments.offline_camera_retrain_v1.scene_floor_physics import FloorCache,FootCache,physics_losses
        floors=FloorCache();foot_vertices=FootCache();protocol.update(variants=['winding_scene'],physics='known camera ROI and start-causal static lowest horizontal patch; replace strong GT native height targets with scene penetration/ground height/persistent contact velocity; training GT used only for conservative stance tags near observed support',source_physics_sha256=hashlib.sha256((H/'scene_floor_physics.py').read_bytes()).hexdigest())
    if args.coverage_only:
        from experiments.offline_camera_retrain_v1.support_coverage_objective import FullFootCache,coverage_physics_losses
        foot_vertices=FullFootCache();physics_losses=coverage_physics_losses
        protocol.update(variants=['coverage_support'],physics='full native foot vertices; TRAIN reference support patch independent of predicted height; observed-plane patch height plus reference rolling velocity/4,8-frame displacement; same penetration and alpha squared; old region labels/BCE unchanged',source_coverage_sha256=hashlib.sha256((H/'support_coverage_objective.py').read_bytes()).hexdigest(),comparison='winding_scene600 only for evaluation, NEVER initialization; same schedule/noise/replay/scene dropout/guard')
    if args.scale_only or args.body_local_only:
        from experiments.offline_camera_retrain_v1.calibrated_control_objective import calibrated_control_losses
        protocol.update(variants=[expected_variant],control='opt-in tolerance-normalized robust TRAIN head position5cm/rotation5deg and head-body relative15deg; confidence1 synthetic; .1 multiplier on .20/.02/.10 from TRAIN encoder audit; same other objectives',source_control_sha256=hashlib.sha256((H/'calibrated_control_objective.py').read_bytes()).hexdigest())
    if args.body_local_only:
        from experiments.offline_camera_retrain_v1.body_local_scene import BodySceneCache
        body_scenes=BodySceneCache();protocol['source_body_local_sha256']={f:hashlib.sha256((H/f).read_bytes()).hexdigest() for f in ['body_local_scene.py','body_local_scene_model.py']}
        protocol['body_local']='two-pass preliminary generated x0 FK22 query; causal static/current dynamic only; zero-initialized32 scene residual; no GT body query; radius.75m unknown masked'
    protocol_path=O/(expected_variant+'_protocol.json') if args.scale_only or args.body_local_only else O/'protocol.json'
    protocol_path.write_text(json.dumps(protocol,indent=2));start=time.monotonic()
    for variant in protocol['variants']:
        torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed+991)
        data=NativeBodyData('train',seed=seed,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'],window_sampling='duration')
        schedule,plan=duration_schedule(data,args.steps,seed,.5);model=build(local=args.body_local_only).train();opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01);folder=O/variant;folder.mkdir();arm=time.monotonic()
        with (folder/'trace.jsonl').open('w') as log:
            for step,(group,_) in enumerate(schedule,1):
                pairs=[data.sample(128,group) for _ in range(2)];batch={k:v.cuda() for k,v in collate([s for s,i in pairs]).items()};known=camera_inputs_only(batch);clean=batch['motion'];scene=bool(rng.random()>=.1);replay=bool(rng.random()<.5)
                if args.body_local_only:known['body_scene_query']=[body_scenes.get(i) for s,i in pairs]
                t=torch.randint(100 if replay else 1000,(2,),device='cuda');alpha=alpha_table[t][:,None,None];noise=torch.randn_like(clean);basis=clean
                if replay and variant!='denoise_control':
                    basis=generate_without_body_initialization(teacher,known,seed=seed+step*17,steps=20,use_scene=scene)
                noisy=alpha.sqrt()*basis+(1-alpha).sqrt()*noise;mask=fixed_control_mask(2,128,'head','cuda');opt.zero_grad(set_to_none=True)
                pred=model(noisy,t,known,control_mask=mask,use_scene=scene);losses=supervised_losses(pred,clean,batch,profile='coordination_v1',signal_weight=alpha.flatten())
                with torch.no_grad():old=teacher(noisy,t,known,control_mask=mask,use_scene=scene);target_soles=sole.soles(clean,[i for s,i in pairs])
                extra=replay_losses(pred,clean,batch['rest'],sole.soles(pred,[i for s,i in pairs]),target_soles,old,alpha.flatten());bce=contact_loss(model.contact_logits(pred,known,use_scene=scene),batch,alpha.flatten());total=losses['total']+extra['replay_total']+.1*bce
                if args.physics_only:
                    total=total-.15*extra['native_envelope']-.1*extra['stance_height']
                    physics=physics_losses(pred,clean,batch['rest'],[i for s,i in pairs],alpha.flatten(),floors,foot_vertices);total=total+physics['scene_physics_total'];extra.update(physics)
                if args.guard_only or args.physics_only:
                    from experiments.offline_camera_retrain_v1.winding_guard_objective import winding_guard_losses
                    guard=winding_guard_losses(pred,clean,alpha.flatten());total=total+guard['winding_guard_total'];extra.update(guard)
                if args.scale_only or args.body_local_only:
                    calibrated=calibrated_control_losses(pred,clean,batch['rest'],alpha.flatten());total=total+calibrated['calibrated_control_total'];extra.update(calibrated)
                assert torch.isfinite(total);total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);lr=2e-5*min(1,step/50)*max(.1,.5*(1+math.cos(math.pi*step/args.steps)))
                for g in opt.param_groups:g['lr']=lr
                opt.step();row=dict(step=step,group=group,length=128,identities=[i for s,i in pairs],t=t.tolist(),noise_checksum=float(noise.sum()),use_scene=scene,replay_schedule=replay,replay_applied=replay and variant!='denoise_control',loss=float(total.detach()),gradient_norm=float(norm),lr=lr)
                if args.body_local_only:row['body_local_query']=model.last_body_query_stats
                if step==1 or step%25==0:row['terms']={k:float(v.detach()) for k,v in extra.items()};print(variant,step,round(row['loss'],4),'elapsed',round(time.monotonic()-arm,1),flush=True)
                log.write(json.dumps(row)+'\n');log.flush()
        config=dict(c,body_local_queries=args.body_local_only,turn_trial=variant,trial_steps=args.steps,trial_seed=seed,known_input_protocol='camera_inputs_only; no body prefix',duration_sampling_plan=plan)
        save_checkpoint(dict(step=args.steps,model=model.state_dict(),config=config,evaluation_only=True,optimizer_removed=True,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampler_state=rng.bit_generator.state),folder/'last.pt');print('SAVED',variant,flush=True);del model,opt;torch.cuda.empty_cache()
    protocol.update(status='training_completed',elapsed_s=time.monotonic()-start,free_disk_after=shutil.disk_usage(O).free);protocol_path.write_text(json.dumps(protocol,indent=2))
if __name__=='__main__':main()
