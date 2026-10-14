"""Rebuild one ego first-hit array; replace only if published SHA is recovered."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.native30_export import (
    inventory,raw_scene,forward_chunk,virtual_head_camera,wearer_faces,render_scene,sampled_environment)
from experiments.offline_camera_retrain_v1.native30_cache_io import buffered_sha256,atomic_text

def repair(folder):
    torch.set_num_threads(1);folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text())
    row=next(r for r in inventory(meta['split']) if r['sequence_id']==meta['sequence_id'] and r['group']==meta['group'])
    q=meta['source_first']+np.arange(meta['frames'],dtype=np.float64);digest=hashlib.sha256()
    people,transform,static,camera,head,mount,intr,rest,profile,objects,valid=raw_scene(row,q,digest)
    self_faces,_=wearer_faces(people[0][0].faces,people[0][0].lbs_weights.numpy(),model_type=profile['model'])
    saved_owners=np.load(folder/'visible_frame_owner.npy')
    points=np.zeros((len(q),512,3),np.float32);owners=np.full((len(q),512),-1,np.int32)
    began=time.monotonic()
    for lo in range(0,len(q),32):
        hi=min(lo+32,len(q));bodies=[forward_chunk(m,p,lo,hi,transform) for m,p in people]
        cam,r=(virtual_head_camera(bodies[0][1][:,15],head[lo:hi],offset_head=np.asarray(mount['translation_m'])-[0,0,.02])
               if camera is None else (camera[0][lo:hi],camera[1][lo:hi]))
        for k,t in enumerate(range(lo,hi)):
            if not valid[t] or 'ego' not in row['allowed_modes']:continue
            meshes=[(bodies[0][0][k],self_faces,100)]+[(bodies[a][0][k],people[a][0].faces,100+a) for a in range(1,len(people))]
            hit=render_scene(meshes,cam[k],r[k],static_scene=static,intrinsics=intr[t],
                rigid_objects=[(sc,rr[t],pp[t],owner) for sc,rr,pp,owner,ov in objects if ov[t]])
            points[t],owners[t]=sampled_environment(hit['points'],hit['owner'])
        if lo%320==0:print('RENDER',lo,len(q),round(time.monotonic()-began,2),flush=True)
    staging=HERE/'runs/native30_runtime_oct14/repair_staging';staging.mkdir(parents=True,exist_ok=True)
    rebuilt=staging/(folder.name+'_visible.npy');np.save(rebuilt,points)
    expected=meta['output_sha256']['visible_frame_points_scenemi_yup.npy'];actual=buffered_sha256(rebuilt)
    report=dict(sequence=meta['sequence_id'],expected=expected,rebuilt_sha256=actual,
                owners_bitwise_equal=bool(np.array_equal(saved_owners,owners)),render_s=time.monotonic()-began)
    if actual!=expected or not report['owners_bitwise_equal']:
        atomic_text(staging/(folder.name+'_FAILED.json'),json.dumps(report,indent=2)+'\n')
        raise ValueError('Rebuild did not recover the published array; original retained')
    target=folder/'visible_frame_points_scenemi_yup.npy'
    old=staging/(folder.name+'_old.npy');shutil.copyfile(target,old)
    temporary=target.with_suffix('.verified_repair.tmp');shutil.copyfile(rebuilt,temporary)
    if buffered_sha256(temporary)!=expected:raise ValueError('DrvFS repair write checksum mismatch')
    temporary.replace(target)
    if buffered_sha256(target)!=expected:raise ValueError('Replaced file checksum mismatch')
    report.update(status='restored_original_published_bytes',old_sha256=buffered_sha256(old),source_metadata_edited=False)
    atomic_text(staging/(folder.name+'_REPAIRED.json'),json.dumps(report,indent=2)+'\n');print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);repair(p.parse_args().folder)
