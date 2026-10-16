"""Label-free, observation-protected articulated correction inside each x0.

This uses local observed patches, not an occupancy volume/closed signed SDF.
The small detached solver permits first-order denoiser training and inference
under torch.no_grad. FK after accepted corrections preserves all bone lengths.
"""
import torch
from pytorch3d.transforms import matrix_to_axis_angle

from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.native30_observations import sanitize_observations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
from experiments.offline_camera_retrain_v1.unified_geometry import skin_points
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas


def select_batch(batch, i):
    size = len(batch['mode'])
    return {k: v[i:i+1] if isinstance(v, torch.Tensor) and v.ndim and len(v) == size else
            [v[i]] if isinstance(v, list) and len(v) == size else v for k,v in batch.items()}


def norm_clip(value, limit):
    return value * (limit / value.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1)


def prepare_evidence(motion, batch, contact_probability, config):
    asset = batch['skin_assets'][0]; scene = batch['surface_windows'][0]
    vertices = skin_points(motion[0], asset)
    labels = asset['labels']; rotations = global_rotations(motion)[0, :, labels]
    raw = scene.query(vertices.detach().cpu().numpy(), rotations.detach().cpu().numpy())
    q = {k: torch.as_tensor(v, device=motion.device) for k,v in raw.items()}
    feet = torch.isin(labels,labels.new_tensor([7,8,10,11]))[None]
    distances=(vertices.detach()[:,:,None]-q['points'][:,:,2]).norm(dim=-1)
    compatible=q['valid'][:,:,2] & q['normal_reliable'][:,:,2]
    compatible &=~feet[...,None] | (q['normals'][:,:,2,:,1]>.7)
    chosen=torch.where(compatible,distances,torch.inf).argmin(-1)
    def gather(key):
        value=q[key][:,:,2]
        index=chosen[...,None]
        if value.ndim==4:index=index[...,None].expand(*index.shape,3)
        return value.gather(2,index).squeeze(2)
    points, normals = gather('points'), gather('normals')
    delta = vertices.detach() - points; signed = (delta * normals).sum(-1)
    tangent = (delta-signed[...,None]*normals).norm(dim=-1)
    known = compatible.any(-1)
    local = known & (delta.norm(dim=-1) < config['query_limit_m']) & (tangent < .09)
    probability = contact_probability.detach()[0]
    support = ((probability-config['contact_threshold'])/(1-config['contact_threshold'])).clamp(0,1)
    contact_weight = torch.zeros_like(signed)
    for joint in (0,1,2,7,8,10,11,20,21):
        ids = (labels==joint).nonzero().flatten()
        score = torch.where(local[:,ids], delta[:,ids].norm(dim=-1), torch.inf)
        closest = score.argmin(-1)
        mask = local[:,ids].gather(1,closest[:,None])[:,0]
        contact_weight.scatter_(1,ids[closest][:,None],(support[:,joint]*mask)[:,None])
    # A finite upward patch near a predicted contacting foot can constrain a
    # shallow support crossing. Walls/occluder back sides carry no such force.
    plane_weight = contact_weight * feet * (normals[...,1]>.7) * (signed.abs()<config['penetration_band_m'])
    trajectory, meta = sanitize_observations(batch['trajectory'], batch['observation_meta'])
    sigma = batch['observation_uncertainty'][...,0].clamp_min(.03)
    observation_weight = meta[...,0]*meta[...,2]*.05**2/(.05**2+sigma.square())
    trusted = (meta[...,0]>0) & (meta[...,2]>.7) & (sigma<.05)
    return dict(points=points, normals=normals, contact_weight=contact_weight,
        plane_weight=plane_weight, owners=gather('owners'),
        surface_velocity=gather('velocities'), velocity_valid=gather('velocity_valid'),
        observation=trajectory[...,:3]*2, observation_weight=observation_weight,
        trusted=trusted, base_positions=forward_kinematics(motion,batch['rest']).detach(),
        asset=asset, rest=batch['rest'])


def energy(motion, evidence, config):
    vertices = skin_points(motion[0],evidence['asset'])
    positions = forward_kinematics(motion,evidence['rest'])
    delta = vertices-evidence['points']; signed=(delta*evidence['normals']).sum(-1)
    def mean(value, weight):
        return (value*weight).sum()/weight.sum().clamp_min(1)
    cw=evidence['contact_weight']; pw=evidence['plane_weight']
    contact=mean((delta.norm(dim=-1)-config['contact_tolerance_m']).clamp_min(0).square()/.05**2,cw)
    plane=mean((-signed-config['plane_tolerance_m']).clamp_min(0).square()/.03**2,pw)
    velocity=vertices.diff(dim=0)*30-evidence['surface_velocity'][1:]
    slip_weight=torch.minimum(cw[1:],cw[:-1])*evidence['velocity_valid'][1:]*evidence['velocity_valid'][:-1]
    slip_weight=slip_weight*(evidence['owners'][1:]==evidence['owners'][:-1])
    slip=mean((velocity.norm(dim=-1)-.2).clamp_min(0).square()/.2**2,slip_weight)
    observation=mean((positions-evidence['observation']).square().sum(-1)/.05**2,evidence['observation_weight'])
    correction=positions-evidence['base_positions']
    prior=correction.square().mean()/.03**2
    smooth=correction.diff(dim=1).square().mean()/.01**2
    total=contact+config['plane_weight']*plane+config['slip_weight']*slip
    total=total+config['observation_weight']*observation+config['prior_weight']*prior+config['temporal_weight']*smooth
    return total


def guide_motion(motion,timestep,batch,details,config,constrain):
    output=[];before=[];after=[];accepted=[]
    alphas=cosine_alphas().to(motion.device)[timestep]
    with torch.enable_grad():
        for b in range(len(motion)):
            sub=select_batch(batch,b);base=motion[b:b+1].detach()
            if float(alphas[b])<config['minimum_alpha']:
                output.append(base);before.append(0.);after.append(0.);accepted.append(0.);continue
            evidence=prepare_evidence(base,sub,details['contact_logits'][b:b+1].sigmoid(),config)
            if not bool((evidence['contact_weight']>0).any()):
                output.append(base);before.append(0.);after.append(0.);accepted.append(0.);continue
            initial=float(energy(base,evidence,config).detach());current=base;value=initial;count=0
            obs=evidence['observation'];trusted=evidence['trusted']
            base_error=(evidence['base_positions']-obs).norm(dim=-1)
            for _ in range(config['iterations']):
                translation=torch.zeros_like(base[...,:3],requires_grad=True)
                rotation=torch.zeros(*base.shape[:2],22,3,device=base.device,requires_grad=True)
                proposal=reconcile_motion(current,sub['rest'],translation,rotation)
                score=energy(proposal,evidence,config)
                gt,gr=torch.autograd.grad(score,(translation,rotation))
                scale=base.shape[1]*float(alphas[b].square())
                dt=norm_clip(-gt*config['translation_lr']*scale,config['translation_step_m'])
                dr=norm_clip(-gr*config['rotation_lr']*scale,config['rotation_step_rad'])
                found=False
                for fraction in (1.,.5,.25):
                    candidate=reconcile_motion(current,sub['rest'],dt.detach()*fraction,dr.detach()*fraction)
                    candidate=constrain(candidate,timestep[b:b+1],sub).detach()
                    new_error=(forward_kinematics(candidate,sub['rest'])-obs).norm(dim=-1)
                    safe=not bool(((new_error-base_error>config['trusted_tolerance_m'])&trusted).any())
                    root_movement=((candidate[...,:3]-base[...,:3])*2).norm(dim=-1).max()
                    candidate_value=float(energy(candidate,evidence,config).detach())
                    if safe and float(root_movement)<=config['root_total_limit_m'] and candidate_value<value-1e-8:
                        current=candidate;value=candidate_value;count+=1;found=True;break
                if not found:break
            output.append(current);before.append(initial);after.append(value);accepted.append(float(count))
    fixed=torch.cat(output)
    # Treat the solver's small correction as fixed, avoiding second derivatives
    # through iterative optimization while keeping the x0/FK path trainable.
    old=rotation_from_6d(motion[...,3:135].detach().reshape(*motion.shape[:2],22,6))
    new=rotation_from_6d(fixed[...,3:135].reshape(*motion.shape[:2],22,6))
    angular=matrix_to_axis_angle(new@old.transpose(-1,-2)).detach()
    result=reconcile_motion(motion,batch['rest'],(fixed[...,:3]-motion[...,:3].detach())*2,angular)
    return result,dict(physical_energy_before=motion.new_tensor(before),physical_energy_after=motion.new_tensor(after),
                      physical_accepted_steps=motion.new_tensor(accepted),
                      physical_root_delta_m=(result[...,:3]-motion[...,:3])*2)
