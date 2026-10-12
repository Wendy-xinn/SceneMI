"""Independently forward selected raw-fit candidates with native SMPL/SMPL-X."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments.offline_camera_retrain_v1.audit_native_gt_quality import (
    HERE,PROJECTS,SOURCE,OUTPUT,PARENTS,load_source,rotation_distance)
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model

CASES=[
    ('original_arm_rotation','recording_20210907_S04_S03_01_camera_wearer_2001-2901',2495,19),
    ('original_arm_position','recording_20210907_S04_S03_01_camera_wearer_2001-2901',2424,19),
    ('other_ego_arm','recording_20210910_S06_S05_02_camera_wearer_3191-3731',3346,18),
    ('other_ego_position','recording_20211004_S19_S06_04_interactee_1071-2871',1363,19),
    ('other_ego_position2','recording_20210929_S05_S16_02_camera_wearer_1301-3281',2903,19),
    ('trumans_foot','2023-01-14@22-49-58',196,11),
    ('rich_wrist','BBQ_001_guitar',183,20),
    ('rich_position','ParkingLot1_007_overfence2',102,21),
]


def run():
    torch.set_num_threads(1)
    rows={r['sequence_id']:r for r in map(json.loads,(HERE/'data/native30_official_memory_v2/source_index.jsonl').read_text().splitlines())}
    sources={r['sequence_id']:r for r in map(json.loads,(SOURCE/'sequences.jsonl').read_text().splitlines())}
    profiles={(s,r['sequence_id']):r for s in ('train','validation','test')
              for r in map(json.loads,(HERE/f'data/poses/{s}.jsonl').read_text().splitlines())}
    info={r['recording_name']:r for r in csv.DictReader((PROJECTS/'egobody/data_info_release.csv').open())}
    models={};reports=[];curves=[]
    for name,seq,frame,joint in CASES:
        row=dict(rows[seq]);row['raw_first']=max(row['raw_first'],frame-15)
        row['raw_last_exclusive']=min(row['raw_last_exclusive'],frame+16)
        pose,transl,beta,kind,gender,scale,ids,missing,digest,path,raw_betas=load_source(row,sources,profiles,info)
        if missing:raise ValueError('Selected native verification has missing source frames')
        key=kind,gender
        if key not in models:models[key]=load_model(*key)
        model=models[key];pose_t=torch.from_numpy(pose.astype(np.float32));n=len(ids)
        body=pose_t[:,1:].reshape(n,63)
        if kind=='smpl':body=torch.cat((body,torch.zeros(n,6)),axis=1)
        def forward(b):
            extras={} if kind=='smpl' else dict(left_hand_pose=torch.zeros(n,45),
                right_hand_pose=torch.zeros(n,45),jaw_pose=torch.zeros(n,3),
                leye_pose=torch.zeros(n,3),reye_pose=torch.zeros(n,3),expression=torch.zeros(n,10))
            with torch.no_grad():
                output=model(global_orient=pose_t[:,0],body_pose=body,
                    transl=torch.from_numpy((transl/scale).astype(np.float32)),
                    betas=torch.from_numpy(b.astype(np.float32)),**extras)
            return output.joints[:,:22].numpy()*scale
        fixed_beta=np.broadcast_to(beta[:10],(n,10)).copy()
        fixed=forward(fixed_beta);original_shape=forward(raw_betas[:,:10])
        # Independent reference: native skinning/rigid transforms vs numpy FK.
        with torch.no_grad():
            from smplx.lbs import blend_shapes,vertices2joints
            rest=vertices2joints(model.J_regressor,model.v_template[None]+
                 blend_shapes(torch.tensor(beta[None,:10],dtype=torch.float32),model.shapedirs[:,:,:10]))[0,:22].numpy()*scale
        local=Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(n,22,3,3)
        glob=np.empty_like(local);glob[:,0]=local[:,0];reference=np.empty_like(fixed)
        reference[:,0]=rest[0]+transl
        for j in range(1,22):
            parent=PARENTS[j];glob[:,j]=glob[:,parent]@local[:,j]
            reference[:,j]=reference[:,parent]+np.einsum('tij,j->ti',glob[:,parent],rest[j]-rest[parent])
        error=float(np.linalg.norm(fixed-reference,axis=-1).max())
        if error>1e-4:raise ValueError(f'Native joint/FK mismatch: {seq}, {error}')
        center=int(np.flatnonzero(ids==frame)[0]);step=rotation_distance(local[1:],local[:-1])[:,joint]
        speed=np.linalg.norm(np.diff(fixed,axis=0),axis=-1)*30
        raw_speed=np.linalg.norm(np.diff(original_shape,axis=0),axis=-1)*30
        arm_bend={}
        for side,(shoulder,elbow,wrist) in {'left':(16,18,20),'right':(17,19,21)}.items():
            upper=fixed[:,elbow]-fixed[:,shoulder];fore=fixed[:,wrist]-fixed[:,elbow]
            cosine=(upper*fore).sum(-1)/np.maximum(np.linalg.norm(upper,axis=-1)*np.linalg.norm(fore,axis=-1),1e-12)
            arm_bend[side]=np.degrees(np.arccos(np.clip(cosine,-1,1)))
        reports.append(dict(case=name,dataset=row['dataset'],split=row['split'],sequence_id=seq,
            source_frames=ids.tolist(),source_sha256=digest,joint=joint,candidate_frame=frame,
            native_fk_max_error_m=error,local_rotation_before_after_deg=step[center-1:center+1].tolist(),
            fixed_shape_joint_speed_before_after_m_s=speed[center-1:center+1,joint].tolist(),
            original_shape_joint_speed_before_after_m_s=raw_speed[center-1:center+1,joint].tolist(),
            fixed_joint_excursion_from_neighbor_average_m=float(np.linalg.norm(fixed[center,joint]-(fixed[center-1,joint]+fixed[center+1,joint])/2)),
            original_shape_joint_excursion_from_neighbor_average_m=float(np.linalg.norm(original_shape[center,joint]-(original_shape[center-1,joint]+original_shape[center+1,joint])/2)),
            elbow_bend_before_at_after_deg={s:v[center-1:center+2].tolist() for s,v in arm_bend.items()},
            leaf_in_22_joint_tree=not np.any(PARENTS==joint),raw_rgb_reviewed=False,
            hands_face_fixed=True,GT_error_confirmed=False))
        curves.append((name,ids,frame,step,speed[:,joint],raw_speed[:,joint]))
    report=dict(protocol='native-source-GT-independent-forward-v1',optimizer_steps=0,cases=reports,
        interpretation='Native first22 joints verify source discontinuities. No RGB proof, no automatic bad-GT labels.')
    (OUTPUT/'GT_CANDIDATE_VERIFICATION.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(13,12))
    for ax,(name,ids,frame,step,speed,raw_speed) in zip(axes.flat,curves):
        ax.plot(ids[1:],speed,label='Joint speed, fixed shape');ax.plot(ids[1:],raw_speed,'--',label='Released frame shape')
        ax.axvline(frame,color='red',alpha=.4);ax.set_title(name);ax.set_ylabel('m/s');ax.set_xlabel('Native 30 Hz source frame')
        other=ax.twinx();other.plot(ids[1:],step,color='orange',alpha=.5,label='Local SO(3) step');other.set_ylabel('degrees / frame')
        ax.grid(alpha=.2)
    axes.flat[0].legend(loc='upper left',fontsize=8)
    fig.suptitle('Released pose candidates: native SMPL / SMPL-X, no smoothing or training\nRotation alerts do not establish incorrect GT; leaf rotations can preserve joint positions')
    fig.tight_layout();fig.savefig(OUTPUT/'GT_CANDIDATES.png',dpi=130);plt.close(fig)
    print(json.dumps({r['case']:{k:r[k] for k in ('native_fk_max_error_m','local_rotation_before_after_deg',
        'fixed_shape_joint_speed_before_after_m_s','original_shape_joint_speed_before_after_m_s',
        'fixed_joint_excursion_from_neighbor_average_m','leaf_in_22_joint_tree')} for r in reports}),flush=True)


if __name__=='__main__':run()
