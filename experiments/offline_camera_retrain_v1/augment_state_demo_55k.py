"""Add explicit untouched 55k-weight/no-prefix control to existing thin demos."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,ddim_sample
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.bounded_head_observation import prepare_bounded_observation
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.typed_head_condition import prepare_observation
from experiments.offline_camera_retrain_v1.evaluate_body_history import measure,crop
from experiments.offline_camera_retrain_v1.state_relative_planner import INPUT_KEYS
HERE=Path(__file__).parent;OUT=HERE/'runs/state_relative_spline_oct10/demo'
@torch.inference_mode()
def main():
    torch.set_num_threads(4);cp=torch.load(HERE/'runs/native_dynamic_scene20_contact_55k_oct07/last.pt',map_location='cpu',weights_only=False);c=cp['config'];model=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=True,contact_prediction=True).cuda().eval();model.load_state_dict(cp['model']);del cp
    current=json.loads((HERE/'runs/body_history_replan_oct10/delta_history/config.json').read_text())
    data=NativeBodyData('validation',seed=777,**{k:current[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=current['rich_contact_root'])
    historic=[json.loads(s) for s in (HERE/'runs/native_dynamic_scene20_contact_55k_oct07/full_validation/standard_rows.jsonl').read_text().splitlines()];manifest=json.loads((OUT/'manifest.json').read_text())
    for case in manifest['cases']:
        index=case['index'];w=historic[index]['window'];sample,identity=data.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);assert identity['sequence_id']==case['identity']['sequence_id'];batch={k:v.cuda() for k,v in collate([sample]).items()};obs=prepare_bounded_observation(batch,'joint');inputs={k:obs[k] for k in INPUT_KEYS if k in obs};motion=ddim_sample(model,inputs,128,steps=20,seed=case['seed'],control_mask=fixed_control_mask(1,128,'head','cuda'))
        path=OUT/f'{index}.npz';arrays=dict(np.load(path));assert np.array_equal(arrays['gt_motion'],sample['motion']);arrays['official55k_joint_motion']=motion[0].cpu().numpy();camera_inputs=prepare_observation(batch,'camera');camera_motion=ddim_sample(model,camera_inputs,128,steps=20,seed=case['seed'],control_mask=fixed_control_mask(1,128,'head','cuda'));arrays['official55k_motion']=camera_motion[0].cpu().numpy();temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,**arrays);temporary.replace(path)
        case['metrics']['official55k_joint']={label:measure(motion[:,start:end],crop(batch,start,end),turn_threshold=15 if label=='short32' else 30) for label,start,end in [('short32',16,48),('future112',16,None)]}
        case['metrics']['official55k']={label:measure(camera_motion[:,start:end],crop(batch,start,end),turn_threshold=15 if label=='short32' else 30) for label,start,end in [('short32',16,48),('future112',16,None)]}
    manifest['display_provenance']={'official55k':'orange; original55k weights + ORIGINAL CAMERA input, standard DDIM, same scene/seed, regenerated not historical sample replay','official55k_joint':'purple; original55k weights + anatomical head input, matched head/scene/seed, standard DDIM, no body prefix','original':'red; joint_all1000 -> delta_history500 weights + executed prefix inpainting; no spline IK','refined':'green; same red draw + state-relative spline IK','gt':'blue; future reference only; first16 history and head simulator are allowed oracle observations'};manifest['weights']='orange official55k; red/green same delta_history500; no new weights';manifest['body_history']='red/green first16 GT as simulator-state proxy; orange standard generation without body history';(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('55k demo controls added')
if __name__=='__main__':main()
