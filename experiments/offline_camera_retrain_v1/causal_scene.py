"""Timestamp-gated static observations; no target-window scene union."""
import numpy as np
from experiments.offline_camera_retrain_v1.data import ANCHORS
from experiments.offline_camera_retrain_v1.scene_visibility_v2 import temporal_bps
from experiments.offline_sequence_v1.data_loader import anchor_rotation


def static_scene_inputs(frame_ids, points, mask, queries, camera, camera_rotation):
    frame_ids=np.asarray(frame_ids);queries=np.asarray(queries)
    if len(frame_ids)!=len(points) or mask.shape!=points.shape[:2] or np.any(np.diff(frame_ids)<=0):
        raise ValueError('Invalid static observation timestamps/shapes')
    if not np.isfinite(points[mask]).all():raise ValueError('Nonfinite static observations')
    # Each observation enters only once, at its first available target timestep.
    cut=np.searchsorted(frame_ids,queries,side='right')
    prior=np.asarray(points[:cut[0]][mask[:cut[0]]],dtype=np.float32)
    if len(prior):
        _,ix=np.unique(np.floor(prior/.025).astype(np.int64),axis=0,return_index=True);prior=prior[np.sort(ix)]
    frames=[np.empty((0,3),np.float32)]
    for lo,hi in zip(cut[:-1],cut[1:]):frames.append(np.asarray(points[lo:hi][mask[lo:hi]],dtype=np.float32))
    empty=[np.empty((0,3),np.float32) for _ in queries]
    bps,valid=temporal_bps(frames,empty,camera,camera_rotation,ANCHORS,return_valid=True,initial_static=prior)
    # Anchor/extent are fixed at the first frame, independent of future motion.
    anchor=anchor_rotation(camera_rotation);local=(prior-camera[0])@anchor.T
    ix=np.floor((local-[0,-.7,0]+[6.4,2.4,6.4])/[.26666667,.2,.26666667]).astype(np.int32)
    ix=ix[((ix>=0)&(ix<[48,24,48])).all(1)];occ=np.zeros((24,48,48),np.float32);occ[ix[:,1],ix[:,0],ix[:,2]]=1
    return occ,bps,valid,len(prior)


def temporal_scene_inputs(folder,queries,history_bundles=None):
    """Read a complete exact-rate bundle and expose only causal observations."""
    import json
    from pathlib import Path
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text());ids=np.load(folder/'source_frame_ids.npy');queries=np.asarray(queries)
    if meta.get('output_fps')!=20 or meta.get('occlusion_version')!=2:raise ValueError('Require exact20 first-hit scenes')
    ix=np.searchsorted(ids,queries)
    if np.any(ix>=len(ids)) or not np.allclose(ids[ix],queries,atol=1e-6):raise ValueError('Missing exact scene timestamps')
    if meta.get('memory_protocol')=='causal20-first-world-voxel25mm-current-dynamic-v1':
        if (folder/'observation_valid.npy').exists() and not np.load(folder/'observation_valid.npy')[ix].all():raise ValueError('Target overlaps invalid observation')
        memory=Path(meta['memory_bundle']);points=np.load(memory/'static_points.npy',mmap_mode='r');times=np.load(memory/'first_observed_source_frames.npy',mmap_mode='r');end=int(np.searchsorted(times,queries[0],side='right'));known=np.asarray(points[:end]);camera=np.load(folder/'camera_position_scenemi_yup.npy',mmap_mode='r')[ix];rotation=np.load(folder/'camera_rotation_scenemi_yup.npy',mmap_mode='r')[ix]
        local=(known-camera[0])@anchor_rotation(rotation).T;v=np.floor((local-[0,-.7,0]+[6.4,2.4,6.4])/[.26666667,.2,.26666667]).astype(np.int32);v=v[((v>=0)&(v<[48,24,48])).all(1)];occ=np.zeros((24,48,48),np.float32);occ[v[:,1],v[:,0],v[:,2]]=1
        bps=np.asarray(np.load(folder/'causal_bps_20.npy',mmap_mode='r')[ix]);valid=np.asarray(np.load(folder/'causal_bps_valid_20.npy',mmap_mode='r')[ix]);history=int(np.searchsorted(times,queries[0],side='left'))
        return dict(occupancy=occ,bps=bps,bps_valid=valid),camera,rotation,dict(scene_protocol='exact20 first-hit; immutable causal recording first-observation memory/current dynamic; window-independent BPS',scene_bundle=str(folder),scene_observation_fps=20,scene_history_points=history,memory_bundle=str(memory),memory_protocol=meta['memory_protocol'])
    p=np.load(folder/'visible_frame_points_scenemi_yup.npy',mmap_mode='r');o=np.load(folder/'visible_frame_owner.npy',mmap_mode='r')
    if (folder/'observation_valid.npy').exists() and not np.load(folder/'observation_valid.npy')[ix].all():raise ValueError('Target overlaps invalid observation')
    history=np.asarray(p[:ix[0]][o[:ix[0]]==0],np.float32)
    prior=[history];prior_times=[np.broadcast_to(ids[:ix[0],None],o[:ix[0]].shape)[o[:ix[0]]==0]]
    for other in (history_bundles or []):
        other=Path(other);past_ids=np.load(other/'source_frame_ids.npy');past_points=np.load(other/'visible_frame_points_scenemi_yup.npy',mmap_mode='r');past_owner=np.load(other/'visible_frame_owner.npy',mmap_mode='r')
        available=past_ids<queries[0];mask=past_owner[available]==0
        prior.append(np.asarray(past_points[available][mask],np.float32));prior_times.append(np.broadcast_to(past_ids[available,None],mask.shape)[mask])
    history=np.concatenate(prior);timestamps=np.concatenate(prior_times)
    if len(timestamps):history=history[np.argsort(timestamps,kind='stable')]
    if len(history):
        _,unique=np.unique(np.floor(history/.025).astype(np.int64),axis=0,return_index=True);history=history[np.sort(unique)]
    points=p[ix];owner=o[ix];static=[a[b==0] for a,b in zip(points,owner)];dynamic=[a[(b>0)&(b<100)] for a,b in zip(points,owner)]
    camera=np.load(folder/'camera_position_scenemi_yup.npy',mmap_mode='r')[ix];rotation=np.load(folder/'camera_rotation_scenemi_yup.npy',mmap_mode='r')[ix]
    bps,valid=temporal_bps(static,dynamic,camera,rotation,ANCHORS,return_valid=True,initial_static=history)
    known=np.concatenate((history,static[0]));local=(known-camera[0])@anchor_rotation(rotation).T
    v=np.floor((local-[0,-.7,0]+[6.4,2.4,6.4])/[.26666667,.2,.26666667]).astype(np.int32);v=v[((v>=0)&(v<[48,24,48])).all(1)];occ=np.zeros((24,48,48),np.float32);occ[v[:,1],v[:,0],v[:,2]]=1
    return dict(occupancy=occ,bps=bps,bps_valid=valid),camera,rotation,dict(scene_protocol='exact20 first-hit; causal static recording prefix/current dynamic; fixed start occupancy',scene_bundle=str(folder),scene_observation_fps=20,scene_history_points=len(history))
