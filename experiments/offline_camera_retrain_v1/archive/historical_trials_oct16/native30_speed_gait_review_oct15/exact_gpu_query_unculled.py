"""Experimental exact-radius membership on GPU; same ordered FPS and fields.

Not active in the trainer. Double precision computes KD-tree radius membership;
original probe precision computes nearest/FPS. No spatial approximation or cap.
"""
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.metric_surface_tokens import SCALES


def select(points, probes, pool, counts, neighbors):
    query=probes.to(dtype=probes.dtype)
    delta=points[pool].to(query.dtype)-query[:,None]
    available=torch.arange(pool.shape[1],device=points.device)[None]<counts[:,None]
    def squared(x):
        return (x[...,0]*x[...,0]+x[...,1]*x[...,1])+x[...,2]*x[...,2]
    first=torch.where(available,squared(delta),torch.inf).argmin(-1)
    row=torch.arange(len(pool),device=points.device);chosen=[first]
    spread=torch.full(pool.shape,torch.inf,dtype=delta.dtype,device=points.device)
    spread.masked_fill_(~available,-torch.inf)
    for _ in range(1,min(neighbors,pool.shape[1])):
        spread=torch.minimum(spread,squared(delta-delta[row,chosen[-1],None]))
        spread[row,chosen[-1]]=-1
        chosen.append(spread.argmax(-1))
    while len(chosen)<neighbors:chosen.append(torch.zeros_like(first))
    ids=pool[row[:,None],torch.stack(chosen,-1)]
    valid=torch.arange(neighbors,device=points.device)[None]<counts[:,None]
    return ids,valid


@torch.no_grad()
def query_surfaces_cuda(scene,vertices,rotations,device,neighbors=8,scale_indices=(0,1,2)):
    device=torch.device(device)
    if device.type!='cuda':
        from experiments.offline_camera_retrain_v1.native30_query_runtime import query_surfaces
        return query_surfaces(scene,vertices,rotations,neighbors,scale_indices)
    frames,anchors=vertices.shape[:2]
    if len(scene.dynamic)!=frames or rotations.shape!=(frames,anchors,3,3):
        raise ValueError('Predicted-body query does not match observed scene clock')
    if neighbors<1:raise ValueError('At least one radius sample is required')
    shape=(frames,anchors,3,neighbors)
    result={k:np.zeros((*shape,3),np.float32) for k in ('points','normals','velocities')}
    result.update({k:np.zeros(shape,bool) for k in ('valid','normal_reliable','velocity_valid')})
    result['owners']=np.zeros(shape,np.int64)
    if not anchors:return result
    static_count=len(scene.static['points'])
    counts=np.asarray([len(c['points']) for c in scene.dynamic],np.int64)
    offsets=static_count+np.r_[0,counts.cumsum()[:-1]]
    cloud={key:np.concatenate([scene.static[key]]+[c[key] for c in scene.dynamic]) for key in scene.static}
    if not len(cloud['points']):return result
    points=torch.as_tensor(cloud['points'],device=device)
    precise=points.to(torch.float64)
    max_dynamic=int(counts.max(initial=0))
    dynamic=np.zeros((frames,max_dynamic,3),np.float64)
    for t,c in enumerate(scene.dynamic):dynamic[t,:counts[t]]=c['points']
    dynamic=torch.as_tensor(dynamic,device=device)
    all_probes=torch.as_tensor(vertices.reshape(-1,3),device=device)
    dynamic_counts=torch.as_tensor(counts,device=device)
    dynamic_offsets=torch.as_tensor(offsets,device=device)
    # Bound temporary distances, independently of the model's scene budget.
    width=static_count+max_dynamic
    chunk=max(1,min(1024,192*1024**2//max(width*32,1)))
    for lo in range(0,frames*anchors,chunk):
        hi=min(frames*anchors,lo+chunk);probe=all_probes[lo:hi]
        frame_ids=torch.arange(lo,hi,device=device)//anchors
        p=probe.double()
        def distances(other):
            dx=p[:,None,0]-other[...,0];d=dx*dx
            dy=p[:,None,1]-other[...,1];d=d+dy*dy
            dz=p[:,None,2]-other[...,2];return d+dz*dz
        ds=distances(precise[None,:static_count])
        dd=distances(dynamic[frame_ids])
        distance=torch.cat((ds,dd),-1);del ds,dd
        dynamic_valid=torch.arange(max_dynamic,device=device)[None]<dynamic_counts[frame_ids,None]
        for si in scale_indices:
            valid=distance<=float(SCALES[si])**2
            valid[:,static_count:]&=dynamic_valid
            sizes=valid.sum(-1);maximum=int(sizes.max())
            if not maximum:continue
            rows,cols=valid.nonzero(as_tuple=True)
            ranks=torch.arange(len(rows),device=device)-(sizes.cumsum(0)-sizes)[rows]
            ids=torch.where(cols<static_count,cols,cols-static_count+dynamic_offsets[frame_ids[rows]])
            pool=torch.zeros((len(probe),maximum),device=device,dtype=torch.long)
            pool[rows,ranks]=ids
            chosen,available=select(points,probe,pool,sizes,neighbors)
            chosen=chosen.cpu().numpy();available=available.cpu().numpy()
            for key in ('points','normals','owners','velocities','velocity_valid'):
                values=cloud[key][chosen]
                dest=result[key].reshape(frames*anchors,3,neighbors,*result[key].shape[4:])
                dest[lo:hi,si]=np.where(available[...,None],values,0) if values.ndim==3 else np.where(available,values,0)
            result['normal_reliable'].reshape(-1,3,neighbors)[lo:hi,si]=cloud['reliable'][chosen]&available
            result['valid'].reshape(-1,3,neighbors)[lo:hi,si]=available
    return result
