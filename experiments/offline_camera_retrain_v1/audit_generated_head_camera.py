"""Record camera-from-generated-mesh disagreement without modifying any motion."""
import json
from pathlib import Path
import numpy as np
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.generated_head_camera import head_world_track,apply_camera_mount,diagnostic_fixed_mount,camera_errors
H=Path(__file__).parent;D=H/'runs/state_relative_spline_oct10/demo';O=H/'runs/turn_path_training_oct10'
def main():
    cases=json.loads((D/'manifest.json').read_text())['cases'];rows=[]
    for case in cases:
        i=case['index'];body=case['identity']['native_body'];model=load_model(body['model'],body['gender'])
        with np.load(D/f'{i}.npz') as z:arrays=dict(z)
        with np.load(O/f'{i}.npz') as z:variants={'original55k':arrays['official55k_motion'],'continued_control':z['continued_control'],'turn_path':z['turn_path'],'gt':arrays['gt_motion']}
        heads={}
        for label,motion in variants.items():
            _,_,joints=decode_native_mesh(motion,body,model);heads[label]=head_world_track(motion,joints,arrays['anchor_rotation'],arrays['camera'][0])
        r,t=diagnostic_fixed_mount(*heads['gt'],arrays['camera'],arrays['rotation']);values={}
        for label,(hp,hr) in heads.items():
            cp,cr=apply_camera_mount(hp,hr,r,t);values[label]=camera_errors(cp,cr,arrays['camera'],arrays['rotation'])
        assert values['gt']['position_max_cm']<.03 and values['gt']['orientation_max_deg']<.01,values['gt']
        rows.append(dict(index=i,calibration='GT first-frame constant mount for DISPLAY ONLY; no generated pose/time/root changes',mount_translation_m=t.tolist(),mount_rotation=r.tolist(),metrics=values,frames=128));print(i,json.dumps(values),flush=True)
    (O/'generated_camera_diagnostic.json').write_text(json.dumps(dict(scope='850/906 development camera reconstruction; GT first-frame calibration is not a deployable estimator',rows=rows,motion_files_modified=False,new_training=False),indent=2))
if __name__=='__main__':main()
