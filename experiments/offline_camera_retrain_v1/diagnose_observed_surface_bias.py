"""Full watertight source mesh is an independent diagnostic, NEVER solver input."""
import json,time
from pathlib import Path
import numpy as np
import torch,trimesh,open3d as o3d
from experiments.offline_camera_retrain_v1.native_surface_points import NativeSurfacePoints
HERE=Path(__file__).parent;OUT=HERE/'runs/observed_surface_refine_oct10';DEMO=HERE/'runs/state_relative_spline_oct10/demo'
def main():
    torch.set_num_threads(2);clips={r['clip_name']:r for r in map(json.loads,Path('/home/wenxin/projects/TRUMANS/processed/scene_expert_v1/clips.jsonl').read_text().splitlines())};manifest=json.loads((DEMO/'manifest.json').read_text());results=[];scenes={}
    for case in manifest['cases']:
        name=case['identity']['sequence_id'];row=clips[name];path=Path('/home/wenxin/projects/TRUMANS')/row['scene']['mesh']
        if str(path) not in scenes:
            mesh=trimesh.load(path,force='mesh',process=False);assert mesh.is_watertight and mesh.is_winding_consistent
            ray=o3d.t.geometry.RaycastingScene();ray.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices,np.float32)),o3d.core.Tensor(np.asarray(mesh.faces,np.uint32)));scenes[str(path)]=ray
        ray=scenes[str(path)];old=dict(np.load(DEMO/f"{case['index']}.npz"));new=dict(np.load(OUT/f"{case['index']}_motions.npz"));skin=NativeSurfacePoints(case['identity']['native_body']);values={}
        cs=ray.compute_signed_distance(o3d.core.Tensor(old['camera'][16:48].astype(np.float32)),nsamples=3).numpy();free_sign=float(np.sign(np.median(cs)));assert (np.sign(cs)==free_sign).all() and abs(cs).min()>.02, 'No consistent known-free camera sign'
        motions={'gt':old['gt_motion'][:48],**new}
        for label,m in motions.items():
            p=skin(torch.tensor(m[16:48],device='cuda')).detach().cpu().numpy();world=p@old['anchor_rotation']+old['camera'][0];sd=ray.compute_signed_distance(o3d.core.Tensor(world.reshape(-1,3)),nsamples=3).numpy().reshape(p.shape[:2]);depth=np.maximum(-sd*free_sign,0)
            values[label]={'full_static_source_mesh_p95_inside_depth_cm':float(np.quantile(depth,.95)*100),'full_static_source_mesh_max_inside_depth_cm':float(depth.max()*100),'full_static_source_mesh_over1cm_fraction':float((depth>.01).mean()),'sign_calibrated_from_known_free_camera_component':True,'by_region':{label:{'p95_depth_cm':float(np.quantile(depth[:,np.isin(skin.labels.cpu().numpy(),ids)],.95)*100),'over1cm_fraction':float((depth[:,np.isin(skin.labels.cpu().numpy(),ids)]>.01).mean())} for label,ids in [('foot_ankle',[7,8,10,11]),('leg_hip',[0,1,2,4,5]),('upper_body',[3,6,9,12,15])]}}
        result={'index':case['index'],'source_mesh':str(path),'source_mesh_only_used_for_scoring':True,'camera_signed_distance_min_m':float(cs.min()),'known_free_raw_parity_sign':free_sign,'raw_negative_inside_is_INVALID_for_enclosing_room_shell':bool(free_sign<0),'topological_free_component_diagnostic_only':True,'metrics':values,'does_not_include_dynamic_objects':True};results.append(result);print(json.dumps(result),flush=True)
    (OUT/'full_source_mesh_diagnostic.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
