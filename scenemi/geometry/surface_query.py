"""Detached nearest patch lookup on CUDA; continuous residuals stay differentiable."""
import numpy as np
from scipy.spatial import cKDTree
import torch


@torch.no_grad()
def nearest(query, points, valid):
    """Return nearest indices/distances with explicit empty masks.

    CUDA uses a tiled bounded-memory kernel. CPU acceptance uses an
    exact KD tree and never copies CUDA query points to the host.
    """
    shape = query.shape
    if points.ndim!=3 or query.shape[0]!=points.shape[0] or valid.shape!=points.shape[:2]:
        raise ValueError('Nearest patch lookup expects B,Q,3 and B,P,3')
    q = query.detach().reshape(len(query),-1,3)
    if query.device.type=='cuda':
        from scenemi.geometry.nearest_cuda import nearest_cuda
        index,distance = nearest_cuda(q,points,valid)
    else:
        index = torch.zeros(q.shape[:2],dtype=torch.long,device=query.device)
        distance = torch.full(q.shape[:2],torch.inf,device=query.device,dtype=query.dtype)
        for b in range(len(q)):
            ids = torch.where(valid[b])[0].numpy()
            if len(ids):
                d,j = cKDTree(points[b,ids].numpy()).query(q[b].numpy(),k=1,workers=1)
                index[b] = torch.from_numpy(ids[j])
                distance[b] = torch.as_tensor(d,dtype=query.dtype)
    return index.reshape(shape[:-1]),distance.reshape(shape[:-1])


def gather(points,index):
    flat = index.reshape(len(index),-1)
    if points.ndim==3:
        out = points.gather(1,flat[...,None].expand(-1,-1,points.shape[-1]))
        return out.reshape(*index.shape,points.shape[-1])
    return points.gather(1,flat).reshape(index.shape)


@torch.no_grad()
def query_surface(anchors,batch,policy,*,dynamic=True):
    index,distance = nearest(anchors,batch['static_points'],batch['static_valid'])
    centre = gather(batch['static_points'],index)
    normal = gather(batch['static_normals'],index)
    normal_valid = gather(batch['static_normal_valid'],index)
    owner = torch.zeros_like(index)
    if dynamic:
        b,t,a,_ = anchors.shape
        pts = batch['frame_scene_points']
        ids = batch['frame_scene_owner']
        mask = (ids>0)&(ids<100)
        di,dd = nearest(anchors.flatten(0,1),pts.flatten(0,1),mask.flatten(0,1))
        dc = gather(pts.flatten(0,1),di).reshape(b,t,a,3)
        dn = gather(batch['frame_normals'].flatten(0,1),di).reshape(b,t,a,3)
        dv = gather(batch['frame_normal_valid'].flatten(0,1),di).reshape(b,t,a)
        do = gather(ids.flatten(0,1),di).reshape(b,t,a)
        dd = dd.reshape(b,t,a)
        take = dd<distance
        centre = torch.where(take[...,None],dc,centre)
        normal = torch.where(take[...,None],dn,normal)
        normal_valid = torch.where(take,dv,normal_valid)
        owner = torch.where(take,do,owner)
        distance = torch.minimum(distance,dd)
    hit = distance<=policy['query_radius_m']
    valid = hit & normal_valid
    return dict(centre=centre,normal=normal,valid=valid,hit=hit,
                distance=torch.where(hit,distance,0.),owner=owner,index=index)


def plane_gap(anchors,patch):
    """Do not detach anchors: this residual must train predicted body motion."""
    return ((anchors-patch['centre'])*patch['normal']).sum(-1)
