"""Audit actual native30 inputs and full model gradients; zero optimizer steps."""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder,inference_payload,load_frozen_exo,load_frozen_ego
from experiments.offline_camera_retrain_v1.native30_model import Native30SceneMI,reconcile_motion
from experiments.offline_camera_retrain_v1.native30_losses import base_losses,observation_losses,objective
from experiments.offline_camera_retrain_v1.native30_preview import preview
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,PARENTS
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas

CONFIG_PATH=HERE/'native30_config.json'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def code_hashes():
    # Record the actual imported repository dependencies, plus the trainer
    # which preparation must never import/execute as a side effect.
    paths=set(HERE.glob('*.py'))|set((HERE.parents[1]/'model').rglob('*.py'))
    from vit_pytorch import vit
    paths.add(Path(vit.__file__))
    paths.add(HERE.parents[2]/'diffusion-motion-inbetweening/sample/virtual_head_camera.py')
    return {str(p.resolve()):sha256(p) for p in sorted(paths)}


def read_config(path=CONFIG_PATH):
    config=json.loads(Path(path).read_text())
    from experiments.offline_camera_retrain_v1.unified_losses import SCALES
    if config['fps']!=30 or config['frames']!=192 or config['initialization']!='from_scratch':
        raise ValueError('Expected fresh native30 /192 frame configuration')
    if config['shared_output_initialization']!='normal_std_0.001_bias_zero' or config['anchor_protocol']!='native_body_dominant_only_67_low_foot_v1':
        raise ValueError('Shared output/anchor initialization differs from implementation')
    if config['observation']['scales']!=SCALES:
        raise ValueError('Observation scale definitions changed')
    if config['scene']['scales_m']!=[.08,.25,.6] or config['scene']['patch_tokens']!=16 or config['scene']['neighbors_per_scale']!=8:
        raise ValueError('Scene representation differs from current implementation')
    if config['noise']['steps']!=1000 or config['noise']['low_noise_steps']>1000:
        raise ValueError('Feedback schedule requires the registered cosine1000 clock')
    if not 0<config['ego_probability']<1 or not 0<=config['ego_head_only_probability']<=1:
        raise ValueError('Invalid sampling probabilities')
    if config['batch_size']<1 or config['optimizer']['lr']<=0:
        raise ValueError('Invalid batch size/learning rate')
    if not 2<=config['sampling']['ddim_steps']<=1000:
        raise ValueError('DDIM inference steps outside supported schedule')
    return config


def make_model(config):
    return Native30SceneMI(config['latent_dim'],tuple(config['dim_mults']),**{k:config['feedback'][k]
        for k in ('root_limit_m','rotation_limit_rad','feedback_gain')})


def dataset_digest(data):
    report={str(data.manifest.resolve()):sha256(data.manifest)}
    # Skinning depends on immutable native templates/weights too, rather than
    # only cached rest joints. Future training checks the same fingerprints.
    body_paths=list((HERE.parents[1]/'body_models/smplx').glob('SMPLX_*.*'))
    body_paths+=list((HERE.parents[2]/'ProtoMotions/data/smpl').glob('SMPL_*.pkl'))
    for path in body_paths:
        if path.suffix in ('.npz','.pkl'):report[str(path.resolve())]=sha256(path)
    for members in data.members.values():
        for meta,_ in members:
            folder=Path(meta['path']); report[str((folder/'metadata.json').resolve())]=sha256(folder/'metadata.json')
            for filename,digest in meta['output_sha256'].items():
                actual=sha256(folder/filename)
                if actual!=digest:raise ValueError(f'Corrupt raw30 bundle: {folder/filename}')
    return report


def sample_item(data,builder,config,plan,rng):
    ix=int(rng.choice(len(plan),p=[r['probability'] for r in plan]));row=plan[ix]
    sample,identity=data.sample(row['group'],row['mode'])
    joints=None if row['mode']=='exo' else ((15,) if rng.random()<config['ego_head_only_probability'] else (15,20,21))
    sample,scene,asset=builder.build(data,sample,identity,row['mode'],joints,noise_rng=rng)
    return (sample,scene,asset,row['mode']),identity


def export_review(config):
    from experiments.offline_camera_retrain_v1.native30_export import export,inventory
    folder=(HERE/config['review_manifest']).parent;folder.mkdir(parents=True,exist_ok=True)
    audit=json.loads((HERE/'runs/ego_exo_retrain_design_oct12/READONLY_HEAD_AUDIT.json').read_text())
    result=[]
    for entry in audit['rows']:
        row=next(r for r in inventory(entry['split']) if r['sequence_id']==entry['sequence_id'])
        result.append(export(row,folder,start=int(entry['source_start_30fps']),frames=config['frames']))
        temp=folder/'manifest.tmp';temp.write_text(''.join(json.dumps(r)+'\n' for r in result));temp.replace(folder/'manifest.jsonl')
    write_json(folder/'review_scope.json',dict(diagnostic_only=True,full_training_coverage=False,
               raw30_windows=len(result),frames=config['frames'],optimizer_steps=0))


def run(config_path=CONFIG_PATH,*,full_data=False,export_inputs=False):
    config=read_config(config_path)
    if export_inputs:export_review(config)
    folder=HERE/config['prepared_output'];folder.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',optimizer_steps=0,data_mode='full' if full_data else 'review',
                training_started=False,free_disk_before=shutil.disk_usage(HERE).free,checks={})
    write_json(folder/'input_checks.json',report);started=time.monotonic()
    torch.set_num_threads(4);torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    manifest=HERE/config['train_manifest' if full_data else 'review_manifest']
    data=Native30Data(manifest,'train',config['seed'],diagnostic=not full_data,frames=config['frames'])
    validation=Native30Data(manifest,'validation',config['seed']+1,diagnostic=not full_data,frames=config['frames'])
    mounts=json.loads((HERE/config['mount_defaults']).read_text());skins=Native30SkinAnchorCache()
    builder=Native30ObservationBuilder(skins,mounts);cases=[];train_cases=[];alignment=[]
    for source in (data,validation):
        for group in source.members:
            for mode in ('ego','exo') if group in ('camera_wearer','interactee') else ('ego',):
                sample,identity=source.sample(group,mode,member=0,start=int(source.members[group][0][1][0]),yaw=0)
                sample,scene,asset=builder.build(source,sample,identity,mode,joints=None)
                item=(sample,scene,asset,mode);label=f'{source.split}/{group}/{mode}'
                cases.append((label,item,identity))
                if source.split=='train':train_cases.append((label,item,identity))
                motion=torch.from_numpy(sample['motion'][None]);fk=forward_kinematics(motion,torch.from_numpy(sample['rest'][None]))[0].numpy()
                discrepancy=float(np.abs(fk-sample['joints']*2).max())
                if discrepancy>1e-4:raise AssertionError(f'Native GT/FK mismatch: {label}, {discrepancy}')
                available=sample['observation_meta'][...,0]>0
                errors=np.linalg.norm(sample['trajectory'][...,:3]*2-sample['joints']*2,axis=-1)
                head_error=float(errors[:,15].mean()) if mode=='ego' else None
                if mode=='ego' and group!='camera_wearer' and head_error>1e-4:
                    raise AssertionError('Actual virtual mount does not recover native head')
                if mode=='exo' and sample['observation_meta'][...,1].any():raise AssertionError('GT rotation leaked into exo')
                alignment.append(dict(label=label,source=identity['source_start_30fps'],frames=192,fps=30,
                    maximum_native_fk_m=discrepancy,head_inverse_mean_m=head_error,
                    reliable_position_mean=float(available.sum(-1).mean()),static_points=len(scene.points),
                    current_dynamic_points_mean=float(np.mean([len(c['points']) for c in scene.dynamic])),
                    contact_label_entries=int(sample['contact_valid'].sum())))
    preview(cases,folder);write_json(folder/'alignment_checks.json',alignment)
    model=make_model(config).cuda().eval();torch.cuda.reset_peak_memory_stats()
    checks=report['checks'];calibration=[];alphas=cosine_alphas().cuda()
    from scipy.spatial.transform import Rotation
    yaw_rotation=Rotation.from_rotvec([0,.6,0]).as_matrix();yaw_error=0.
    for label,item,identity in cases:
        source=data if identity['split']=='train' else validation
        altered,_=source.sample(identity['group'],identity['mode'],member=0,start=identity['start'],yaw=.6)
        expected=(item[0]['joints']*2)@yaw_rotation.T
        yaw_error=max(yaw_error,float(np.abs(altered['joints']*2-expected).max()))
    if yaw_error>1e-5:raise AssertionError('Post-anchor world yaw is inconsistent or cancelled')
    checks['post_anchor_common_world_yaw']=dict(angle_rad=.6,maximum_joint_error_m=yaw_error)
    for label,item,identity in train_cases:
        batch=collate_observations([item]);known=inference_payload(batch)
        state=torch.randn_like(batch['motion']);t=torch.tensor([25],device='cuda')
        with torch.no_grad():prediction,details=model(state,t,known,return_details=True)
        checks.setdefault('predicted_proposal_query_coverage',[]).append(dict(label=label,
            known_anchor_fraction=float(torch.cat([g['closest_valid'].flatten() for g in details['geometry']]).float().mean()),
            known_region_fraction=float(torch.cat([g['region_valid'].flatten() for g in details['geometry']]).float().mean()),
            query_source='untrained predicted x0, no GT skin queries'))
        for timestep in (25,300,800):
            signal=alphas[torch.tensor([timestep],device='cuda')]
            leaf=prediction.detach().requires_grad_(True)
            base=base_losses(leaf,batch['motion'],batch,signal,details['raw_motion'].detach())['total']
            terms=observation_losses(leaf,batch['motion'],batch,signal)
            reference=float(torch.autograd.grad(base,leaf,retain_graph=True)[0].norm())
            row=dict(label=label,timestep=timestep,reference_gradient_norm=reference,terms={})
            for key,value in terms.items():
                gradient=torch.autograd.grad(value,leaf,retain_graph=True)[0]
                if not torch.isfinite(gradient).all():raise AssertionError(f'Nonfinite observation gradient: {label}/{key}')
                row['terms'][key]=dict(loss=float(value.detach()),gradient_norm=float(gradient.norm()))
            calibration.append(row)
        print('audited TRAIN gradient bucket',label,flush=True)
    targets=config['observation']['target_gradient_ratios'];caps=config['observation']['maximum_coefficients']
    weights={}
    # Conservative across domains and noise buckets; unavailable rotations do
    # not dilute gradients of streams that actually provide rotation evidence.
    for key in targets:
        ratios=[r['terms'][key]['gradient_norm']/max(r['reference_gradient_norm'],1e-12)
                for r in calibration if r['terms'][key]['gradient_norm']>0]
        weights[key]=min(caps[key],targets[key]/max(ratios)) if ratios else 0.
    for row in calibration:
        for key,term in row['terms'].items():
            term['coefficient']=weights[key]
            term['weighted_gradient_ratio']=weights[key]*term['gradient_norm']/max(row['reference_gradient_norm'],1e-12)
    write_json(folder/'TRAIN_gradient_scales.json',dict(fitting_split='train',panel_windows=6,
        noise_buckets=[25,300,800],method='cap maximum weighted shared-output gradient ratio across available buckets',
        not_population_calibration=True,coefficients=weights,buckets=calibration,
        geometry_coefficients_status='conservative initial values; finite-gradient check, not efficacy calibration'))
    # Actual batch2 full-objective backward, no optimizer constructed or stepped.
    exo=next(item for label,item,_ in train_cases if label=='train/camera_wearer/exo')
    rich=next(item for label,item,_ in train_cases if label=='train/rich/ego')
    items=[exo,rich][:config['batch_size']]
    while len(items)<config['batch_size']:items.append(exo)
    batch=collate_observations(items);known=inference_payload(batch);t=torch.full((len(items),),25,device='cuda')
    signal=alphas[t];state=torch.randn_like(batch['motion'])
    model.train();prediction,details=model(state,t,known,return_details=True)
    loss,terms=objective(prediction,batch['motion'],batch,signal,details,config,weights)
    loss.backward()
    gradients={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise AssertionError('Nonfinite complete objective/parameter gradients')
    for name in ('root_feedback.weight','rotation_feedback.weight'):
        if gradients.get(name,0)<=0:raise AssertionError(f'Missing positional/SO3 feedback gradient: {name}')
    condition_names=('observation_encoder.0.weight','camera_encoder.0.weight','patch_projection.weight','dynamic_condition.weight',
                     'patch_condition.weight','surface_embedding.0.weight','region_condition.weight','contact_pose.0.weight')
    for name in condition_names:
        if gradients.get(name,0)<=0:raise AssertionError(f'Missing condition gradient: {name}')
    checks['full_objective_backward']=dict(batch_size=len(items),frames=192,finite=True,optimizer_steps=0,
        loss=float(loss.detach()),root_feedback_gradient_norm=gradients['root_feedback.weight'],
        rotation_feedback_gradient_norm=gradients['rotation_feedback.weight'],
        peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
        condition_gradient_norms={k:gradients[k] for k in condition_names},
        zero_initialized_feedback_upstream_gradients='temporal/contact path initially waits for zero residual heads to learn')
    model.zero_grad(set_to_none=True);model.eval()
    # Independent frozen loader cannot read GT motion/contact even accidentally.
    item=exo;label,_,identity=next(x for x in train_cases if x[0]=='train/camera_wearer/exo')
    baseline=collate_observations([item]);known=inference_payload(baseline)
    frozen=load_frozen_exo(identity['path'],skin_cache=skins)
    for key in ('trajectory','observation_meta','camera','occupancy'):
        if not torch.equal(known[key],frozen[key]):raise AssertionError(f'Frozen observations differ: {key}')
    checks['frozen_no_body_truth_loader']=True
    for label,item,identity in cases:
        if item[3]!='ego':continue
        expected=inference_payload(collate_observations([item]))
        actual=load_frozen_ego(identity['path'],start=identity['start'],skin_cache=skins,mounts=mounts)
        for key in ('trajectory','observation_meta','camera','occupancy'):
            if not torch.equal(expected[key],actual[key]):raise AssertionError(f'Frozen self input mismatch: {label}/{key}')
    checks['frozen_self_camera_head_without_body_truth']=True
    state=torch.randn_like(baseline['motion']);t=torch.tensor([25],device='cuda')
    with torch.no_grad():
        y,d=model(state,t,known,return_details=True)
        poison=dict(known,motion=torch.full_like(y,float('nan')),joints=torch.full_like(baseline['joints'],float('nan')))
        yp=model(state,t,poison)
        if not torch.equal(y,yp):raise AssertionError('Hidden body truth affects forward')
        checks['hidden_truth_poison']=True
        masked=dict(known,trajectory=torch.zeros_like(known['trajectory']),observation_meta=known['observation_meta'].clone(),
                    camera_confidence=torch.zeros_like(known['camera_confidence']))
        masked['observation_meta'][...,:4]=0
        pm=model(state,t,masked)
        masked_poison=dict(masked,trajectory=torch.full_like(masked['trajectory'],float('nan')),camera=torch.full_like(masked['camera'],float('nan')))
        mp=model(state,t,masked_poison)
        if not torch.equal(pm,mp):raise AssertionError('Zero-confidence observations leaked through a route')
        checks['zero_confidence_poison_all_paths']=True
        # Demonstrate the residual mechanism with temporary, explicit test
        # parameters; restore them and do not save as learned weights.
        # An untrained body can be too far from all real surfaces. Build a
        # controlled observed plane around its own predicted feet, without GT.
        from experiments.offline_camera_retrain_v1.native30_geometry import Native30SurfaceWindow
        from experiments.offline_camera_retrain_v1.unified_geometry import skin_points,occupancy_from_points
        vertices=skin_points(d['proposal'][0],known['skin_assets'][0])
        foot=vertices[:,torch.isin(known['skin_assets'][0]['labels'],vertices.new_tensor([10,11],dtype=torch.long))].mean((0,1)).cpu().numpy()
        grid=np.linspace(-.65,.65,23)
        plane_points=np.array([[foot[0]+x,foot[1]-.03,foot[2]+z] for x in grid for z in grid],np.float32)
        controlled=Native30SurfaceWindow(np.broadcast_to(plane_points,(192,*plane_points.shape)),
            np.zeros((192,len(plane_points)),np.int32),np.tile(foot+[0,.5,0],(192,1)))
        feedback_known=dict(known,surface_windows=[controlled],
            occupancy=torch.from_numpy(occupancy_from_points(plane_points))[None].cuda())
        saved={k:model.state_dict()[k].clone() for k in ('root_feedback.bias','rotation_feedback.bias')}
        model.root_feedback.bias.fill_(.35);model.rotation_feedback.bias.fill_(.2)
        corrected,cd=model(state,t,feedback_known,return_details=True)
        for k,v in saved.items():model.state_dict()[k].copy_(v)
        body=forward_kinematics(corrected,baseline['rest'])
        root_change=float(cd['root_delta_m'].norm(dim=-1).max())
        position_change=float((body-forward_kinematics(reconcile_motion(cd['raw_motion'],baseline['rest']),baseline['rest'])).norm(dim=-1).max())
        if root_change<=0 or position_change<=0:raise AssertionError('Contact feedback does not change position')
        bone_error=0.
        for j in range(1,22):
            length=(body[:,:,j]-body[:,:,PARENTS[j]]).norm(dim=-1)
            rest_length=(baseline['rest'][:,j]-baseline['rest'][:,PARENTS[j]]).norm(dim=-1)[:,None]
            bone_error=max(bone_error,float((length-rest_length).abs().max()))
        fk_error=float((corrected[...,135:].reshape(1,192,22,3)*2-body).abs().max())
        if bone_error>1e-5 or fk_error>1e-6:raise AssertionError('Position feedback breaks native skeleton consistency')
        checks['position_rotation_feedback']=dict(test_parameters_only=True,root_delta_max_m=root_change,
            joint_position_change_max_m=position_change,bone_length_error_max_m=bone_error,fk_branch_error_max_m=fk_error,
            geometry='controlled observed plane at untrained predicted feet; no GT body query',real_motion_quality_claim=False)
        # Suppressing predicted contact must suppress positional feedback,
        # including biases in temporal layers and output projections.
        contact_bias=model.contact_readout.bias.clone();root_bias=model.root_feedback.bias.clone()
        model.root_feedback.bias.fill_(.35);model.contact_readout.bias.fill_(-100.)
        _,suppressed=model(state,t,feedback_known,return_details=True)
        model.root_feedback.bias.copy_(root_bias);model.contact_readout.bias.copy_(contact_bias)
        if suppressed['root_delta_m'].abs().max()>1e-10:
            raise AssertionError('Temporal/output bias bypasses contact gating')
        checks['contact_gate_controls_position_feedback']=True
        no_scene=model(state,t,known,use_scene=False)
        scene_difference=float((y-no_scene).abs().max())
        if scene_difference<=0:raise AssertionError('Scene has no connection to output')
        checks['scene_condition_reaches_output']=dict(maximum_difference=scene_difference,quality_claim=False)
        model.patch_enabled=False;without_patch=model(state,t,known);model.patch_enabled=True
        checks['patch_condition_reaches_output']=dict(maximum_difference=float((y-without_patch).abs().max()),quality_claim=False)
        if checks['patch_condition_reaches_output']['maximum_difference']<=0:
            raise AssertionError('Patch tokens do not affect the actual output')
        controlled_output=model(state,t,feedback_known)
        model.local_enabled=False;without_local=model(state,t,feedback_known);model.local_enabled=True
        checks['local_condition_reaches_output']=dict(maximum_difference=float((controlled_output-without_local).abs().max()),
            geometry='controlled observed plane',quality_claim=False)
        if checks['local_condition_reaches_output']['maximum_difference']<=0:
            raise AssertionError('Local geometry does not affect the actual output')
        from experiments.offline_camera_retrain_v1.native30_model import sample_native30
        sample=sample_native30(model,known,steps=2,seed=config['seed'])
        if not torch.isfinite(sample).all():raise AssertionError('Nonfinite complete-window DDIM inference')
        checks['full_window_ddim_smoke']=dict(frames=192,steps=2,core_forwards=4,finite=True,
            quality_claim=False,normal_evaluation_steps=config['sampling']['ddim_steps'],static_tokens_reused=True)
        if model._sampling_scene is not None:raise AssertionError('Stale sampling tokens survived generation')
    report.update(status='passed',elapsed_s=time.monotonic()-started,training_ready=bool(full_data),
        data_scope='full corpus' if full_data else '8 raw30 review clips; 12 ego/exo input cases; full-corpus export pending',
        free_disk_after=shutil.disk_usage(HERE).free,model_parameters=sum(p.numel() for p in model.parameters()),
        alignment=alignment)
    fingerprints=dataset_digest(data)|dataset_digest(validation)
    effective=dict(config=config,data_mode=report['data_mode'],sampling_plan=data.sampling_plan(config['ego_probability']),
        observation_coefficients=weights,mount_defaults_sha256=sha256(HERE/config['mount_defaults']),
        data_sha256=fingerprints,code_sha256=code_hashes(),torch_version=torch.__version__,
        normalization='motion positions m/2; physical derivatives SI',optimizer_steps=0)
    write_json(folder/'effective_config.json',effective);write_json(folder/'input_checks.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG_PATH)
    parser.add_argument('--full-data',action='store_true')
    parser.add_argument('--export-review',action='store_true')
    args=parser.parse_args();run(args.config,full_data=args.full_data,export_inputs=args.export_review)
