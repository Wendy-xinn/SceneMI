"""Fit the real EgoBody PV-to-head approximation using official TRAIN only.

Does not export scene arrays or start training. Synthetic ego cameras use their
actual recorded mount; external observer cameras do not use a target head mount.
"""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.offline_camera_retrain_v1.audit_native_gt_quality import (
    HERE,PROJECTS,SOURCE,PARENTS,load_source,model_rest_bases)
from experiments.offline_camera_retrain_v1.native30_splits import (
    official_sources,official_recordings,SPLIT_PROTOCOL,OFFICIAL_CSV)
from experiments.offline_camera_retrain_v1.native30_export import raw_pv_camera


def fit():
    started=time.monotonic();assignments=official_recordings();all_sources=official_sources()
    selected={}
    for row in all_sources:
        if row['split']!='train' or row['role']!='camera_wearer' or 'ego' not in row['allowed_modes']:continue
        row=dict(row,group='camera_wearer',raw_first=row['source_first_frame'],raw_last_exclusive=row['source_last_frame_exclusive'])
        if row['recording'] not in selected or row['raw_last_exclusive']-row['raw_first']>selected[row['recording']]['raw_last_exclusive']-selected[row['recording']]['raw_first']:
            selected[row['recording']]=row
    sources={r['sequence_id']:r for r in map(json.loads,(SOURCE/'sequences.jsonl').read_text().splitlines())}
    profiles={(s,r['sequence_id']):r for s in ('train','validation','test')
        for r in map(json.loads,(HERE/f'data/poses/{s}.jsonl').read_text().splitlines())}
    info={r['recording_name']:r for r in csv.DictReader((PROJECTS/'egobody/data_info_release.csv').open())}
    bases=model_rest_bases();rotations=[];offsets=[];identities=[];digest=hashlib.sha256()
    for recording,row in sorted(selected.items()):
        if assignments[recording]!='train':raise ValueError('Non-TRAIN camera calibration source')
        count=min(128,row['raw_last_exclusive']-row['raw_first'])
        first=row['raw_first']+(row['raw_last_exclusive']-row['raw_first']-count)//2
        row=dict(row,raw_first=first,raw_last_exclusive=first+count)
        pose,transl,beta,kind,gender,scale,ids,missing,source_sha,path,_=load_source(row,sources,profiles,info)
        if missing:raise ValueError('Missing calibration body fit')
        camera,cam_r,_=raw_pv_camera(PROJECTS/'egobody',recording,row['scene_id'],ids,digest)
        tf=PROJECTS/'egobody/calibrations'/recording/'cal_trans/kinect12_to_world'/f"{row['scene_id']}.json"
        world=np.asarray(json.loads(tf.read_text())['trans']);digest.update(tf.read_bytes());digest.update(source_sha.encode())
        rest0,shape=bases[(kind,gender)];rest=rest0+np.einsum('jck,k->jc',shape,beta[:10])
        local=Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(count,22,3,3)
        glob=np.empty_like(local);joints=np.empty((count,22,3));glob[:,0]=local[:,0];joints[:,0]=rest[0]+transl
        for j in range(1,22):
            parent=PARENTS[j];glob[:,j]=glob[:,parent]@local[:,j]
            joints[:,j]=joints[:,parent]+np.einsum('tij,j->ti',glob[:,parent],rest[j]-rest[parent])
        head_r=world[None,:3,:3]@glob[:,15];head_p=joints[:,15]@world[:3,:3].T+world[:3,3]
        # Equal per-recording sample count; no long-recording dominance.
        take=np.linspace(0,count-1,min(16,count),dtype=int)
        rotations.append(head_r[take].transpose(0,2,1)@cam_r[take])
        offsets.append(np.einsum('tji,tj->ti',head_r[take],camera[take]-head_p[take]))
        identities.append(dict(recording=recording,sequence_id=row['sequence_id'],official_split='train',
            sampled_source_frames=ids[take].tolist(),raw_body_sha256=source_sha,body_profile_split=row['pose_profile_split']))
    relative=np.concatenate(rotations);translation=np.concatenate(offsets)
    rotation=Rotation.from_matrix(relative).mean().as_matrix();offset=np.median(translation,axis=0)
    position_error=np.linalg.norm(translation-offset,axis=-1);rotation_error=Rotation.from_matrix(relative@rotation.T).magnitude()
    mount=dict(rotation=rotation.tolist(),translation_m=offset.tolist(),position_p90_m=float(np.quantile(position_error,.9)),
        rotation_p90_rad=float(np.quantile(rotation_error,.9)),fitting_split='train',split_protocol=SPLIT_PROTOCOL,
        official_split_csv_sha256=hashlib.sha256(OFFICIAL_CSV.read_bytes()).hexdigest(),
        fitting_recordings=sorted(selected),identities=identities,fitting_frames=len(relative),source_sha256=digest.hexdigest(),
        method='raw30 fixed native shape; median head-local position; SO3 mean; equal16 samples/recording',
        limitations='Approximate population mount; sensor alignment/fit errors included in measured p90 uncertainty')
    destination=HERE/'data/native30_official_memory_v2/TRAIN_camera_mounts.json'
    if destination.exists():raise ValueError('Refuse to overwrite an existing official camera mount; review its provenance first')
    destination.write_text(json.dumps(dict(camera_wearer=mount),indent=2,allow_nan=False)+'\n')
    report=dict(protocol='native30-official-TRAIN-real-PV-head-mount-v1',optimizer_steps=0,
        fitting_recordings=len(selected),fitting_frames=len(relative),official_split_counts=dict(train=len(selected),validation=0,test=0),
        excluded_train_recordings=sorted(r for r,s in assignments.items() if s=='train' and r not in selected),
        mount_path=str(destination),mount_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_s=time.monotonic()-started,
        translation_m=offset.tolist(),position_p90_m=mount['position_p90_m'],rotation_p90_deg=float(np.rad2deg(mount['rotation_p90_rad'])))
    (HERE/'runs/exo_occlusion_review_oct12/OFFICIAL_TRAIN_MOUNT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':fit()
