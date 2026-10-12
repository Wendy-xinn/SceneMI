"""Read-only source-motion audit and paired-SMPL-X extrinsic recovery.

No training, source edits, smoothing or replacement of the six review cameras.
Recovered extrinsics do not determine intrinsics or the camera pixel convention.
"""
import csv
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from smplx.lbs import blend_shapes, vertices2joints

from experiments.offline_camera_retrain_v1.data import HERE, PROJECTS, SOURCE
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.native30_export import (
    forward_chunk, resample_fits)
from experiments.offline_camera_retrain_v1.exo_occlusion_review import RUN, sha


def recover_world_to_camera(world_orient, camera_orient, world_transl,
                            camera_transl, rest_pelvis):
    """Recover X_cam=R_cw X_world+t_cw, respecting the SMPL root pivot.

    The paired bodies must have identical local pose, shape and body model.
    A SMPL(-X) root joint is rest_pelvis+transl, not simply transl.
    """
    rw = Rotation.from_rotvec(np.asarray(world_orient)).as_matrix()
    rc = Rotation.from_rotvec(np.asarray(camera_orient)).as_matrix()
    r = rc @ rw.transpose(0, 2, 1)
    jw = np.asarray(world_transl) + rest_pelvis
    jc = np.asarray(camera_transl) + rest_pelvis
    t = jc - np.einsum('tij,tj->ti', r, jw)
    centre = -np.einsum('tji,tj->ti', r, t)
    return r, t, centre


def source_stats(positions, q, joint):
    v = np.linalg.norm(np.diff(positions[:, joint], axis=0), axis=-1)*30
    a = np.linalg.norm(np.diff(positions[:, joint], n=2, axis=0), axis=-1)*900
    top = np.argsort(v)[-5:][::-1]
    return dict(first_20_speed_m_s=v[:20].tolist(),
                first_20_peak_speed_m_s=float(v[:20].max()),
                first_20_peak_acceleration_m_s2=float(a[:19].max()),
                remaining_peak_speed_m_s=float(v[20:].max()),
                top_speed_frames=[dict(source_frame=int(q[i+1]), speed_m_s=float(v[i])) for i in top],
                loop_displacement_m=float(np.linalg.norm(positions[0,joint]-positions[-1,joint])))


def plot_arm_curves(curves, q):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for role,name in (('camera_wearer','Target'),('interactee','Other person')):
        v=np.linalg.norm(np.diff(curves[role+'_joints'],axis=0),axis=-1)*30
        axes[0].plot(q[1:],v[:,19],label=name+' right elbow')
        axes[1].plot(q[1:],v[:,21],label=name+' right wrist')
    axes[2].plot(q[1:],curves['camera_wearer_local_rotation_step_deg'][:,18],label='Target elbow local rotation')
    axes[2].plot(q[1:],curves['camera_wearer_local_rotation_step_deg'][:,20],label='Target wrist local rotation')
    for ax,y in zip(axes,('Speed (m/s)','Speed (m/s)','Rotation change per 1/30 s (deg)')):
        ax.axvspan(q[0],q[20],color='orange',alpha=.13,label='Audited opening 21 frames')
        ax.set_ylabel(y);ax.legend(loc='upper right');ax.grid(alpha=.2)
    axes[-1].set_xlabel('Native source frame at 30 Hz')
    fig.suptitle('Source SMPL arm jitter: native forward exactly reproduces exported meshes\nNo generated motion, no smoothing; playback boundary excluded')
    fig.tight_layout();fig.savefig(RUN/'SOURCE_ARM_JITTER.png',dpi=160);plt.close(fig)


def audit_egobody(manifest):
    case = manifest['cases'][0]
    folder = Path(case['path']); q = np.load(folder/'source_frame_ids.npy')
    sources = {r['sequence_id']: r for r in map(json.loads, (SOURCE/'sequences.jsonl').read_text().splitlines())}
    record = 'recording_20210907_S04_S03_01'
    info = next(r for r in csv.DictReader((PROJECTS/'egobody/data_info_release.csv').open())
                if r['recording_name']==record)
    # The scan's world calibration is recorded by the source sequence itself.
    row = next(r for r in map(json.loads,(SOURCE/'egobody_sequences/validation.jsonl').read_text().splitlines())
               if r['sequence_id']==case['sequence_id'])
    transform_path = PROJECTS/'egobody/calibrations'/record/'cal_trans/kinect12_to_world'/f"{row['scene_id']}.json"
    m = np.asarray(json.loads(transform_path.read_text())['trans'])
    people = []; curves = {}
    for index, role in enumerate(('camera_wearer','interactee')):
        fit_dir = Path(sources[f'{record}/{role}']['body_fits'])
        raw_paths = [fit_dir/f'results/frame_{int(f):05d}/000.pkl' for f in q]
        digest = hashlib.sha256()
        fits = []
        for path in raw_paths:
            raw = path.read_bytes(); digest.update(str(path).encode()); digest.update(raw)
            fits.append(pickle.loads(raw, encoding='latin1'))
        raw = {key:np.array([np.asarray(f[key]).reshape(-1) for f in fits],np.float32)
               for key in ('global_orient','body_pose','transl','betas')}
        params = {key:value.copy() for key,value in raw.items()}
        beta = np.array(case['native_body']['betas'],np.float32) if index==0 else np.median(raw['betas'],0)
        params['betas'] = np.broadcast_to(beta,raw['betas'].shape).copy()
        body_idx = int(fit_dir.name.rsplit('_',1)[1])
        gender = info[f'body_idx_{body_idx}'].split()[-1].lower()
        model = load_model('smpl',gender)
        vertices,joints = [],[]
        for lo in range(0,len(q),32):
            v,j = forward_chunk(model,params,lo,min(lo+32,len(q)),(1.,m[:3,:3],m[:3,3]))
            vertices.append(v);joints.append(j)
        vertices,joints = np.concatenate(vertices),np.concatenate(joints)
        cached_vertices=np.load(folder/f'person_{index}_vertices.npy',mmap_mode='r')
        if np.linalg.norm(vertices-cached_vertices,axis=-1).max()>1e-4:
            raise ValueError('Raw-fit forward differs from export; investigate before attributing jitter to source')
        resampled = resample_fits(fits,np.arange(len(q),dtype=float),'smpl')
        rotation_errors = {}
        for key in ('global_orient','body_pose'):
            source_r = Rotation.from_rotvec(raw[key].reshape(-1,3)).as_matrix()
            sampled_r = Rotation.from_rotvec(resampled[key].reshape(-1,3)).as_matrix()
            rotation_errors[key] = float(np.abs(source_r-sampled_r).max())
        local_r = Rotation.from_rotvec(raw['body_pose'].reshape(-1,3)).as_matrix().reshape(len(q),23,3,3)
        angular = np.degrees(Rotation.from_matrix((local_r[1:]@local_r[:-1].transpose(0,1,3,2)).reshape(-1,3,3)).magnitude()).reshape(-1,23)
        arm_stats={str(j):source_stats(joints,q,j) for j in (16,17,18,19,20,21)}
        elbow_plane=[]
        for shoulder,elbow,wrist in ((16,18,20),(17,19,21)):
            upper=joints[:,elbow]-joints[:,shoulder]; fore=joints[:,wrist]-joints[:,elbow]
            normal=np.cross(upper,fore); normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-12)
            elbow_plane.append(np.degrees(np.arccos(np.clip((normal[1:]*normal[:-1]).sum(-1),-1,1))))
        target_masks = {}
        if index==0:
            for j in (17,19,21):
                target_masks[str(j)]={key:np.load(folder/(key+'.npy'))[:21,j].tolist()
                                      for key in ('confidence','available','high')}
        people.append(dict(role=role,gender=gender,fit_directory=str(fit_dir),raw_fit_digest=digest.hexdigest(),
                           source_vs_export_max_vertex_error_m=float(np.linalg.norm(vertices-cached_vertices,axis=-1).max()),
                           source_vs_export_max_joint_error_m=float(np.linalg.norm(joints-np.load(folder/'gt_joints_reference.npy'),axis=-1).max()) if index==0 else None,
                           integer_slerp_max_rotation_matrix_difference=rotation_errors,
                           first_20_local_rotation_peak_step_deg={str(j):float(angular[:20,j-1].max()) for j in (16,17,18,19,20,21,22,23)},
                           first_20_elbow_plane_peak_step_deg=dict(left=float(elbow_plane[0][:20].max()),right=float(elbow_plane[1][:20].max())),
                           raw_beta_max_temporal_range=float(np.ptp(raw['betas'],axis=0).max()),
                           exported_betas_fixed=True,arm_stats=arm_stats,first_21_observations=target_masks,
                           first_20_peak_any_vertex_speed_m_s=float((np.linalg.norm(np.diff(vertices[:21],axis=0),axis=-1)*30).max()),
                           loop_peak_vertex_displacement_m=float(np.linalg.norm(vertices[0]-vertices[-1],axis=-1).max())))
        curves[role+'_joints']=joints
        curves[role+'_local_rotation_step_deg']=angular
        curves[role+'_elbow_plane_step_deg']=np.array(elbow_plane)
    np.savez_compressed(RUN/'EGO_ARM_AUDIT_CURVES.npz',source_frame_ids=q,**curves)
    plot_arm_curves(curves,q)
    return dict(case=case['spec']['id'],source_first=int(q[0]),source_last=int(q[-1]),people=people,
                raw_rgb_verified=False,model_loaded=False,geometry_smoothed=False,
                conclusion='Direct native forward reproduces the cached body; local arm jitter originates in released fits. Actor motion versus fit error requires source-image evidence.')


def smplx_params(fit,q,beta):
    p={key:np.asarray(fit[key],np.float32)[q] for key in
       ('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','expression','transl')}
    p['betas']=np.broadcast_to(beta,(len(q),len(beta))).copy()
    p['leye_pose']=p['reye_pose']=np.zeros((len(q),3),np.float32)
    return p


def audit_trumans(manifest):
    model=load_model('smplx','male');beta=np.zeros(10,np.float32)
    with torch.inference_mode():
        rest=vertices2joints(model.J_regressor,model.v_template[None]+blend_shapes(torch.from_numpy(beta[None]),model.shapedirs[:,:,:10]))[0,0].numpy()
    records=[]
    for case in manifest['cases']:
        if case['group']!='trumans':continue
        name=case['sequence_id'];paths=[PROJECTS/'TRUMANS'/s/(name+'_smplx_results.pkl') for s in ('smplx_result','smplx_result_in_cam')]
        world,camera=[pickle.loads(p.read_bytes()) for p in paths]
        local_fields=('body_pose','left_hand_pose','right_hand_pose','jaw_pose','expression','betas')
        diffs={k:float(np.max(np.abs(np.asarray(world[k])-np.asarray(camera[k])))) if np.asarray(world[k]).size else 0. for k in local_fields}
        if any(diffs.values()) or len(world['transl'])!=len(camera['transl']):raise ValueError('Paired bodies differ; extrinsics cannot be recovered by root transforms alone')
        if np.asarray(world['betas']).size:raise ValueError('This audit only accepts the verified zero-shape release')
        r,t,centres=recover_world_to_camera(world['global_orient'],camera['global_orient'],world['transl'],camera['transl'],rest)
        window=np.load(Path(case['path'])/'source_frame_ids.npy')
        q=np.unique(np.r_[np.linspace(0,len(r)-1,13,dtype=int),window[::24],window[-1]])
        pv=[];pc=[]
        for fit,store in ((world,pv),(camera,pc)):
            params=smplx_params(fit,q,beta)
            for lo in range(0,len(q),8):
                v,_=forward_chunk(model,params,lo,min(lo+8,len(q)),(1.,np.eye(3),np.zeros(3)));store.append(v)
        pv,pc=np.concatenate(pv),np.concatenate(pc)
        predicted=np.einsum('tij,tvj->tvi',r[q],pv)+t[q,None]
        error=np.linalg.norm(predicted-pc,axis=-1)
        naive_t=np.asarray(camera['transl'])-np.einsum('tij,tj->ti',r,np.asarray(world['transl']))
        naive_error=np.linalg.norm((np.einsum('tij,tvj->tvi',r[q],pv)+naive_t[q,None])-pc,axis=-1)
        output=RUN/(case['spec']['id']+'_RECOVERED_EXTRINSICS.npz')
        np.savez_compressed(output,source_frame_ids=np.arange(len(r)),R_world_to_camera=r,
                            t_world_to_camera=t,position_world=centres,rotation_camera_to_world=r.transpose(0,2,1),rest_pelvis=rest)
        if error.max()>1e-4:raise ValueError('Recovered camera fails independent full-mesh validation')
        records.append(dict(case=case['spec']['id'],sequence_id=name,frames=len(r),
                            world_source=str(paths[0]),camera_source=str(paths[1]),source_sha256=[sha(p) for p in paths],
                            local_parameter_max_difference=diffs,rest_pelvis_m=rest.tolist(),
                            verified_source_frames=q.tolist(),mesh_vertices_per_frame=int(pv.shape[1]),
                            mesh_transform_error_m=dict(mean=float(error.mean()),max=float(error.max())),
                            naive_transl_only_error_m=dict(mean=float(naive_error.mean()),max=float(naive_error.max())),
                            orthogonality_max_error=float(np.abs(r@r.transpose(0,2,1)-np.eye(3)).max()),
                            determinant_range=[float(np.linalg.det(r).min()),float(np.linalg.det(r).max())],
                            selected_window_camera_centre_range_m=np.ptp(centres[window],axis=0).tolist(),
                            selected_window_first_camera_position=centres[window[0]].tolist(),
                            selected_window_peak_camera_step_m=float(np.linalg.norm(np.diff(centres[window],axis=0),axis=-1).max()),
                            camera_is_static=False,intrinsics_recovered=False,pixel_axis_convention_verified=False,
                            camera_coordinates='released camera axes; not verified CV optical axes',
                            camera_pelvis_z_range_m=[float((np.asarray(camera['transl'])+rest)[:,2].min()),float((np.asarray(camera['transl'])+rest)[:,2].max())],
                            all_pelvis_camera_z_negative=bool(((np.asarray(camera['transl'])+rest)[:,2]<0).all()),
                            previous_review_camera_unchanged=True,extrinsics_output=str(output),output_sha256=sha(output)))
    return records


def main():
    torch.set_num_threads(1);before=shutil.disk_usage(HERE)
    manifest=json.loads((RUN/'manifest.json').read_text())
    result=dict(protocol='exo-followup-source-motion-paired-camera-v1',date='2026-10-12',training_steps=0,
                ego_arm=audit_egobody(manifest),trumans=audit_trumans(manifest),
                disk_free_bytes_before=before.free,disk_free_bytes_after=shutil.disk_usage(HERE).free,
                code_sha256=sha(Path(__file__)),scene_query_implementation_unchanged=True)
    (RUN/'FOLLOWUP_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
