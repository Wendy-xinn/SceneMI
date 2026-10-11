"""Record camera-from-generated-mesh disagreement without modifying any motion."""
import argparse,json
from pathlib import Path
import numpy as np
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.generated_head_camera import head_world_track,apply_camera_mount,diagnostic_fixed_mount,camera_errors
H=Path(__file__).parent;D=H/'runs/state_relative_spline_oct10/demo';O=H/'runs/turn_path_training_oct10'
def main():
    p=argparse.ArgumentParser();p.add_argument('--support-refinement',action='store_true');p.add_argument('--fresh55k-replay',action='store_true');p.add_argument('--fresh55k-coverage',action='store_true');p.add_argument('--scale-body-scene',action='store_true');p.add_argument('--angular-balance',action='store_true');args=p.parse_args()
    if sum((args.support_refinement,args.fresh55k_replay,args.fresh55k_coverage,args.scale_body_scene,args.angular_balance))>1:p.error('Choose one trial')
    global O
    if args.support_refinement:O=H/'runs/sole_support_oct11'
    if args.fresh55k_replay:O=H/'runs/fresh55k_replay_oct11'
    if args.fresh55k_coverage:O=H/'runs/fresh55k_support_coverage_oct11'
    if args.scale_body_scene:O=H/'runs/control_scale_body_scene_oct11'
    if args.angular_balance:O=H/'runs/control_scale_body_scene_oct11/angular_balance_eval'
    cases=json.loads((D/'manifest.json').read_text())['cases'];rows=[]
    for case in cases:
        i=case['index'];body=case['identity']['native_body'];model=load_model(body['model'],body['gender'])
        with np.load(D/f'{i}.npz') as z:arrays=dict(z)
        with np.load(O/f'{i}.npz') as z:
            variants={'original55k':arrays['official55k_motion'],'gt':arrays['gt_motion']}
            keys=['denoise_control','rollout_replay','denoise_terminal','replay_terminal'] if args.fresh55k_replay else (['turn_control','sole_support'] if args.support_refinement else ['continued_control','turn_path'])
            if args.fresh55k_coverage:keys=['scene_reference','coverage_support']
            if args.scale_body_scene:keys=['support_reference','scale_calibrated','body_local']
            if args.angular_balance:keys=['support_reference','scale_calibrated','angular_balanced']
            variants.update({k:z[k] for k in keys})
        if args.support_refinement:
            with np.load(H/'runs/turn_path_training_oct10'/f'{i}.npz') as z:variants['prior_turn600']=z['turn_path']
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
