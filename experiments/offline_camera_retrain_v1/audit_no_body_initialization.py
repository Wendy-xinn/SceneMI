"""850/906 no-body-init reproduction, GT mutation and temporal diagnosis."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.known_input_generation import generate_without_body_initialization
from experiments.offline_camera_retrain_v1.evaluate_body_history import measure,crop
from experiments.offline_camera_retrain_v1.evaluate_state_spline_confirmation import leg_motion
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
HERE=Path(__file__).parent;OUT=HERE/'runs/no_body_initialization_oct10'
@torch.inference_mode()
def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    cp=torch.load(HERE/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt',map_location='cpu',weights_only=False);c=cp['config'];model=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();model.load_state_dict(cp['model']);del cp
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    demo=HERE/'runs/state_relative_spline_oct10/demo';manifest=json.loads((demo/'manifest.json').read_text());historic=[json.loads(s) for s in (HERE/'runs/native_dynamic_scene20_contact_55k_oct07/full_validation/standard_rows.jsonl').read_text().splitlines()];rows=[]
    for case in manifest['cases']:
        index=case['index'];w=historic[index]['window'];sample,identity=data.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);assert identity['sequence_id']==case['identity']['sequence_id']
        batch={k:v.cuda() for k,v in collate([sample]).items()};motion=generate_without_body_initialization(model,batch,seed=case['seed'])
        poisoned=dict(batch)
        for key in ('motion','joints','trajectory','contact_target','contact_valid'):
            if key in batch:poisoned[key]=torch.full_like(batch[key],True if batch[key].dtype==torch.bool else 1234)
        poisoned['executed_history']=torch.full((1,16,22,9),1234.,device='cuda');poisoned['initial_body_pose']=torch.full((1,201),1234.,device='cuda')
        second=generate_without_body_initialization(model,poisoned,seed=case['seed']);assert torch.equal(motion,second)
        with np.load(demo/f'{index}.npz') as arrays:
            error=float(np.abs(motion[0].cpu().numpy()-arrays['official55k_motion']).max());assert error<1e-5,error
        metrics=measure(motion,crop(batch,0,128),turn_threshold=30);metrics.update(leg_motion(motion))
        global_r=global_rotations(motion)[0].cpu().numpy();local_head=(global_r[:,0].transpose(0,2,1)@global_r[:,15]);angles=np.rad2deg(Rotation.from_matrix(local_head.copy()).magnitude())
        increments=[]
        for j in (0,9,12,15,1,2,4,5,7,8):
            inc=np.rad2deg(Rotation.from_matrix((global_r[1:,j]@global_r[:-1,j].transpose(0,2,1)).copy()).magnitude());peak=int(np.argmax(inc))+1
            increments.append({'joint':j,'max_step_deg':float(inc.max()),'peak_frame':peak,'peak_time_s':peak/20,'p95_step_deg':float(np.quantile(inc,.95))})
        rows.append({'index':index,'identity':identity,'seed':case['seed'],'initial_body_GT_read':False,'GT_mutation_max_difference':float((motion-second).abs().max()),'official55k_saved_motion_max_difference':error,'full128_metrics':metrics,'head_relative_root_angle_p95_deg':float(np.quantile(angles,.95)),'global_rotation_steps':increments})
    (OUT/'audit.json').write_text(json.dumps({'scope':'two exposed fixed examples; not new holdout or a repaired model','initialization':'all128 frames start from random diffusion noise; no GT16 body or initial pose alignment','condition':'known camera and scene; TRUMANS camera is synthetic head-mounted oracle input, not real video estimate','native_body':'configured template/rest/scale, held fixed in poison audit; not runtime initial pose','diagnostic_GT':'scoring only','source_sha256':{f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['known_input_generation.py','audit_no_body_initialization.py']},'rows':rows,'new_checkpoints':0,'new_motion_archives':0},indent=2));print(json.dumps([{'index':r['index'],'GT_poison':r['GT_mutation_max_difference'],'reproduction':r['official55k_saved_motion_max_difference']} for r in rows]))
if __name__=='__main__':main()
