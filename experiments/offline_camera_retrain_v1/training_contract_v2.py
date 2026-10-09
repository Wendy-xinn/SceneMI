"""Versioned utilities for the next causal-scene experiment; not used by 55k.

Call observe() only after a genuine per-frame visibility export. Feeding the
old sequence union cannot reconstruct when a surface was first observed.
"""
import numpy as np
from scipy.spatial import cKDTree


def domain_probabilities(valid_frames, alpha=.5):
    """Tempered duration sampling. alpha=1 proportional; alpha=0 equal."""
    if not 0 <= alpha <= 1 or not valid_frames:
        raise ValueError('Require domains and alpha in [0,1]')
    names=list(valid_frames);counts=np.array([valid_frames[n] for n in names],float)
    if not np.isfinite(counts).all() or (counts<=0).any():
        raise ValueError('Domain sizes must be finite and positive')
    weights=counts**alpha;return dict(zip(names,(weights/weights.sum()).tolist()))


class CausalSceneMemory:
    """Recording-level static memory and current observed dynamic geometry.

    Carry this across target windows from the SAME recording. Reset between
    recordings/splits. Dynamic objects disappear from geometric input when
    unobserved; persistence/prediction needs a separate tracker with age and
    uncertainty. No invisible GT object completion is performed here.
    """
    def __init__(self,voxel_size=.025):
        if voxel_size<=0:raise ValueError('Positive voxel size required')
        self.voxel_size=voxel_size;self.reset()
    def reset(self):
        self.time=-np.inf;self.static={};self.dynamic=np.empty((0,3),np.float32)
    def observe(self,time,points,owners):
        points=np.asarray(points,np.float32);owners=np.asarray(owners)
        if not np.isfinite(time) or time<=self.time:raise ValueError('Strictly increasing observation times required')
        if points.ndim!=2 or points.shape[1]!=3 or owners.shape!=(len(points),):raise ValueError('Expected [N,3] points and [N] owner IDs')
        observed=owners>=0
        if not np.isfinite(points[observed]).all():raise ValueError('Nonfinite observed geometry')
        for p in points[owners==0]:
            key=tuple(np.floor(p/self.voxel_size).astype(np.int64))
            if key not in self.static:self.static[key]=(p.copy(),float(time))
        self.dynamic=points[(owners>0)&(owners<100)].copy();self.time=float(time)
    def surfaces(self):
        static=np.array([p for p,t in self.static.values()],np.float32).reshape(-1,3)
        return static,self.dynamic.copy()
    def bps(self,position,rotation,anchors):
        static,dynamic=self.surfaces();points=np.concatenate((static,dynamic))
        anchors=np.asarray(anchors,np.float32)
        if not len(points):return np.zeros_like(anchors),np.zeros(len(anchors),bool)
        probes=anchors@np.asarray(rotation).T+position
        nearest=points[cKDTree(points).query(probes)[1]]
        delta=(nearest-probes)@rotation
        return (delta/np.maximum(np.linalg.norm(delta,axis=-1,keepdims=True),1)/2).astype(np.float32),np.ones(len(anchors),bool)


def validate_contract(metadata):
    """Fail closed before a new experiment is allowed to claim causal input."""
    required={'body_conversion':'official_transfer_or_native','scene_time':'causal_recording_history','dynamic_time':'current_frame','visibility':'joint_first_hit','split_scope':'recording','geometry_provenance':'hashed_arrays_and_models'}
    problems=[f'{key}: expected {value}, got {metadata.get(key)!r}' for key,value in required.items() if metadata.get(key)!=value]
    if problems:raise ValueError('Training contract failed:\n'+'\n'.join(problems))


def covered_frames(valid_starts,length):
    """Union of eligible target time, so dense overlapping starts count once."""
    intervals=sorted((int(s),int(s)+length) for s in valid_starts)
    count=0;end=-1
    for start,stop in intervals:
        count+=max(0,stop-max(start,end));end=max(end,stop)
    return count


def duration_sampling_plan(dataset,alpha=.5):
    """Count union of all eligible target windows per role, independent of density."""
    frames={}
    for group in ('trumans','camera_wearer','interactee','rich'):
        source=dataset.rich_members if group=='rich' else dataset.base.groups[group]
        frames[group]=0
        for m,v in source:
            intervals=sorted((int(start if group=='rich' else start[0]),int(start if group=='rich' else start[0])+length) for length,starts in v.items() for start in starts)
            end=-1
            for start,stop in intervals:
                frames[group]+=max(0,stop-max(start,end));end=max(end,stop)
    domains={'trumans':frames['trumans'],'egobody':frames['camera_wearer']+frames['interactee'],'rich':frames['rich']}
    p=domain_probabilities(domains,alpha)
    roles=frames['camera_wearer']+frames['interactee']
    group_p={'trumans':p['trumans'],'rich':p['rich'],'camera_wearer':p['egobody']*frames['camera_wearer']/roles,'interactee':p['egobody']*frames['interactee']/roles}
    return {'unique_eligible_frames':frames,'domain_probabilities':p,'group_probabilities':group_p,'alpha':alpha}


def duration_schedule(dataset,steps,seed,alpha=.5):
    plan=duration_sampling_plan(dataset,alpha);groups=list(plan['group_probabilities']);probabilities=[plan['group_probabilities'][g] for g in groups]
    rng=np.random.default_rng(seed+7001)
    schedule=[(str(rng.choice(groups,p=probabilities)),int(rng.choice([64,128,192],p=[.30,.45,.25]))) for _ in range(steps)]
    return schedule,plan
