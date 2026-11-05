"""Bounded-memory exact 3D nearest lookup for observed point clouds."""
import torch
import triton
import triton.language as tl


@triton.jit
def _nearest(query,points,valid,index,distance,Q:tl.constexpr,P:tl.constexpr,
             BQ:tl.constexpr,BP:tl.constexpr):
    batch=tl.program_id(0)
    qi=tl.program_id(1)*BQ+tl.arange(0,BQ)
    qx=tl.load(query+(batch*Q+qi)*3,qi<Q,0)
    qy=tl.load(query+(batch*Q+qi)*3+1,qi<Q,0)
    qz=tl.load(query+(batch*Q+qi)*3+2,qi<Q,0)
    best=tl.full((BQ,),float('inf'),tl.float32)
    chosen=tl.full((BQ,),0,tl.int32)
    for start in range(tl.cdiv(P,BP)):
        pi=start*BP+tl.arange(0,BP)
        ok=tl.load(valid+batch*P+pi,pi<P,False)
        px=tl.load(points+(batch*P+pi)*3,pi<P,0)
        py=tl.load(points+(batch*P+pi)*3+1,pi<P,0)
        pz=tl.load(points+(batch*P+pi)*3+2,pi<P,0)
        dx=qx[:,None]-px[None,:];dy=qy[:,None]-py[None,:];dz=qz[:,None]-pz[None,:]
        delta=dx*dx+dy*dy+dz*dz
        delta=tl.where(ok[None,:]&(qi[:,None]<Q),delta,float('inf'))
        current=tl.min(delta,1)
        candidate=tl.min(tl.where((delta==current[:,None])&ok[None,:],pi[None,:],2147483647),1)
        take=(current<best)|((current==best)&(candidate<chosen))
        best=tl.minimum(best,current)
        chosen=tl.where(take,candidate,chosen)
    tl.store(index+batch*Q+qi,chosen,qi<Q)
    tl.store(distance+batch*Q+qi,tl.sqrt(best),qi<Q)


def nearest_cuda(query,points,valid):
    if query.dtype!=torch.float32 or points.dtype!=torch.float32:
        raise ValueError('Observed surface CUDA lookup uses fp32')
    query,points,valid=query.contiguous(),points.contiguous(),valid.contiguous()
    b,q,_=query.shape;p=points.shape[1]
    index=torch.empty((b,q),device=query.device,dtype=torch.long)
    distance=torch.empty((b,q),device=query.device,dtype=query.dtype)
    _nearest[(b,triton.cdiv(q,16))](query,points,valid,index,distance,q,p,16,512,
                                 num_warps=4,enable_fp_fusion=False)
    return index,distance
