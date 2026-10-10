"""Native sparse skin parity, gradients and per-time surface/GT invariants."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.native_surface_points import NativeSurfacePoints
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.observed_body_surfaces import ObservedPlanes,oriented_planes
from experiments.offline_camera_retrain_v1.observed_scene_refinement import refine_observed_scene
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
ROOT=Path(__file__).parent/'runs/observed_surface_refine_oct10'
def main():
    torch.set_num_threads(2);ROOT.mkdir(exist_ok=True);demo=ROOT.parent/'state_relative_spline_oct10/demo';manifest=json.loads((demo/'manifest.json').read_text());body=manifest['cases'][0]['identity']['native_body'];skin=NativeSurfacePoints(body);m=torch.tensor(np.load(demo/'850.npz')['refined_motion'][:4],device='cuda');vertices,_,_=decode_native_mesh(m.cpu().numpy(),body);actual=skin(m).detach().cpu().numpy();error=float(abs(actual-vertices[:,skin.indices]).max());assert error<2e-5,error
    variable=m.detach().clone().requires_grad_(True);skin(variable).square().mean().backward();assert torch.isfinite(variable.grad).all() and variable.grad[:,3:135].abs().max()>0
    panel=json.loads((ROOT/'panel_rows.json').read_text());smpl_body=next(row['identity']['native_body'] for row in panel if row['identity']['native_body']['model']=='smpl');smpl_skin=NativeSurfacePoints(smpl_body);smpl_vertices,_,_=decode_native_mesh(m.cpu().numpy(),smpl_body);smpl_error=float(abs(smpl_skin(m).detach().cpu().numpy()-smpl_vertices[:,smpl_skin.indices]).max());assert smpl_error<2e-5,smpl_error
    grid=np.array([[x,0,z] for x in np.linspace(-.3,.3,15) for z in np.linspace(-.3,.3,15)],np.float32);p,n,valid=oriented_planes(grid,np.broadcast_to([0,1,0],grid.shape));assert (n[valid,1]>.99).all()
    # Current dynamic surface absent at previous/future owner positions.
    dyn=grid+[2,0,0];frames=[(grid,n,np.zeros(len(grid),np.int32)),(dyn,n,np.ones(len(grid),np.int32))];scene=ObservedPlanes(frames);q=torch.tensor([[[2,-.02,0]],[[2,-.02,0]]],device='cuda');signed,normal,owner,known,_=scene.query(q);assert not known[0,0] and known[1,0] and owner[1,0]==1
    full=torch.tensor(np.load(demo/'850.npz')['refined_motion'][:24],device='cuda')[None];history=full[:,:16].clone();rest=skin.joints[:22][None]*body['scale'];gt=forward_kinematics(full,rest);r=global_rotations(full);track=torch.cat((gt/2,r[..., :,0],r[..., :,1]),-1);meta=torch.ones(1,24,22,5,device='cuda');planes=ObservedPlanes([(grid,n,np.zeros(len(grid),np.int32)) for _ in range(8)]);batch={'rest':rest,'trajectory':track,'observation_meta':meta,'motion':full.clone(),'contact_target':torch.zeros(1,24,22,device='cuda')}
    first,audit=refine_observed_scene(full,history,batch,planes,skin,iterations=2);poison=dict(batch,motion=torch.full_like(full,999),contact_target=torch.ones_like(batch['contact_target']));second,_=refine_observed_scene(full,history,poison,planes,skin,iterations=2);assert torch.equal(first,second);assert torch.equal(first[:,:16],full[:,:16]);assert torch.isfinite(first).all()
    result={'status':'passed','native_mesh_max_difference_m':error,'smpl_sparse_mesh_max_difference_m':smpl_error,'rotation_gradients_finite_nonzero':True,'observed_floor_normals_face_free_space':True,'dynamic_frames_not_accumulated':True,'future_body_contact_GT_poison_zero_effect':True,'trusted_prefix_unchanged':True};(ROOT/'tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
