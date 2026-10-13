"""Strict raw30 bundles, scoped train/validation samplers and common world yaw."""
import copy
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.offline_camera_retrain_v1.native30_export import PROTOCOL, inventory
from experiments.offline_camera_retrain_v1.native30_exo import external_transforms
from experiments.offline_camera_retrain_v1.native30_splits import normalize_split, validate_official_bundle
from experiments.offline_camera_retrain_v1.native30_memory import ObservedStaticMemory, OFFLINE_MEMORY_PROTOCOL
from experiments.offline_camera_retrain_v1.data import anchor_rotation, rotation6d
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D
from experiments.offline_camera_retrain_v1.native30_gt_reliability import quality_weights,DEFAULT_GT_RELIABILITY,validate_gt_reliability


def exo_window_starts(observed,starts,frames,max_empty_frames=None):
    """Keep partial joint occlusion; optionally defer long all-joint gaps."""
    if max_empty_frames is not None and (isinstance(max_empty_frames,bool) or
            not isinstance(max_empty_frames,(int,np.integer)) or max_empty_frames<0):
        raise ValueError('exo_max_empty_frames must be a nonnegative integer or null')
    observed=np.asarray(observed,dtype=bool);starts=np.asarray(starts,dtype=np.int64)
    prefix=np.r_[0,np.cumsum(observed)]
    current=starts[prefix[starts+frames]-prefix[starts]>0]
    if max_empty_frames is None or max_empty_frames>=frames:return current
    limit=int(max_empty_frames)
    ends=np.zeros(len(observed),dtype=np.int64)
    ends[limit:]=(prefix[limit+1:]-prefix[:-(limit+1)])==0
    gaps=np.r_[0,np.cumsum(ends)]
    return current[gaps[current+frames]-gaps[current+limit]==0]


class Native30Data:
    def __init__(self, manifest, split, seed=20261012, *, diagnostic=False, frames=192,
                 memory_protocol=OFFLINE_MEMORY_PROTOCOL,quality_config=None,exo_max_empty_frames=None):
        split=normalize_split(split)
        self.manifest=Path(manifest); self.split=split; self.rng=np.random.default_rng(seed)
        self.frames=frames; self.members={g:[] for g in ('trumans','camera_wearer','interactee','rich')}
        self.quality_config=validate_gt_reliability(dict(DEFAULT_GT_RELIABILITY) if quality_config is None else quality_config)
        self.exo_max_empty_frames=exo_max_empty_frames
        self.arrays={};self.mode_starts={}
        expected_sources={(r['group'],r['sequence_id']):r for r in inventory(split)} if not diagnostic else {}
        memory_entries=[]
        for row in map(json.loads,self.manifest.read_text().splitlines()):
            if row['split'] != split:
                continue
            folder=Path(row['path']); meta=json.loads((folder/'metadata.json').read_text())
            if meta['protocol'] != PROTOCOL or meta['fps'] != 30 or meta['split'] != split:
                raise ValueError('Not a native30 scoped bundle')
            for field in ('sequence_id','bundle_id','split','group','frames','input_sha256','diagnostic_only'):
                if meta[field]!=row[field]:raise ValueError(f'Manifest/bundle identity mismatch: {field}')
            if meta['diagnostic_only'] and not diagnostic:
                raise ValueError('Restricted review bundle cannot be used for training')
            if not diagnostic:
                validate_official_bundle(meta, split)
            ids=np.load(folder/'source_frame_ids.npy',mmap_mode='r');valid=np.load(folder/'observation_valid.npy',mmap_mode='r')
            if len(ids) != meta['frames'] or not np.all(np.diff(ids)==1) or valid.shape != ids.shape:
                raise ValueError('Incomplete raw30 clock')
            if valid.dtype!=np.bool_ or ids[0]!=meta['source_first'] or ids[-1]!=meta['source_last']:
                raise ValueError('Invalid observation mask or source clock bounds')
            if not diagnostic:
                source=expected_sources.get((meta['group'],meta['sequence_id']))
                if source is None or ids[0]!=source['raw_first'] or ids[-1]!=source['raw_last_exclusive']-1:
                    raise ValueError('Full training bundle must cover its original scoped raw30 source')
                modes=source.get('allowed_modes',['ego','exo'] if source['group'] in ('camera_wearer','interactee') else ['ego'])
                if meta.get('allowed_modes')!=modes:
                    raise ValueError('Observer mode coverage changed from official source inventory')
                if meta['dataset']=='egobody' and any(meta.get(k)!=source.get(k) for k in
                    ('recording','scene_id','pose_profile_split')):
                    raise ValueError('Official recording/scene/body profile provenance changed')
            sums=np.r_[0,np.cumsum(~valid)]; starts=np.flatnonzero(sums[frames:]-sums[:-frames]==0)
            for mode in meta.get('allowed_modes',['ego','exo']):
                current=starts
                if mode=='exo':
                    observed=np.asarray(self.load(folder,'exo_position_available')).any(1)
                    current=exo_window_starts(observed,starts,frames,exo_max_empty_frames)
                self.mode_starts[(str(folder),mode)]=current
            memory_entries.append(dict(meta,path=str(folder)))
            if len(starts):
                self.members[meta['group']].append((dict(meta,path=str(folder)),starts))
        if not diagnostic:
            expected={(r['group'],r['sequence_id']) for r in inventory(split) if r['raw_last_exclusive']-r['raw_first']>=frames}
            # A full export with no valid windows may be retained but not sampled.
            all_rows=[json.loads(l) for l in self.manifest.read_text().splitlines()]
            covered={(r['group'],r['sequence_id']) for r in all_rows if r['split']==split and not r['diagnostic_only']}
            if expected-covered:
                raise ValueError(f'Full raw30 coverage missing {len(expected-covered)} sources; no native20 fallback')
        self.scene_memory=ObservedStaticMemory(self.load,memory_entries,protocol=memory_protocol)

    def load(self, folder, key):
        cache=self.arrays.setdefault(str(folder),{})
        if key not in cache:
            cache[key]=np.load(Path(folder)/f'{key}.npy',mmap_mode='r')
        return cache[key]

    def starts_for(self,group,index,mode):
        meta,starts=self.members[group][index]
        if hasattr(self,'mode_starts'):return self.mode_starts.get((meta['path'],mode),np.empty(0,dtype=int))
        return starts

    def sampling_plan(self, ego_probability=.7):
        durations={mode:{g:sum(len(self.starts_for(g,i,mode))/30 for i,(meta,starts) in enumerate(members) if
            mode in meta.get('allowed_modes',['ego','exo'] if g in ('camera_wearer','interactee') else ['ego']))
            for g,members in self.members.items()} for mode in ('ego','exo')}
        duration=durations['ego']
        corpus={'trumans':duration['trumans'],'egobody':duration['camera_wearer']+duration['interactee'],'rich':duration['rich']}
        normal=sum(np.sqrt(v) for v in corpus.values()); plan=[]
        for g in duration:
            if duration[g] == 0:
                continue
            domain='egobody' if g in ('camera_wearer','interactee') else g
            role=duration[g]/corpus[domain]
            plan.append(dict(group=g,mode='ego',probability=ego_probability*np.sqrt(corpus[domain])/normal*role))
        duration=durations['exo']
        corpus={'trumans':duration['trumans'],'egobody':duration['camera_wearer']+duration['interactee'],'rich':duration['rich']}
        normal=sum(np.sqrt(v) for v in corpus.values())
        if normal<=0:raise ValueError('Unified sampler needs external observer sources')
        for g in duration:
            if not duration[g]:continue
            domain='egobody' if g in ('camera_wearer','interactee') else g
            role=duration[g]/corpus[domain]
            plan.append(dict(group=g,mode='exo',probability=(1-ego_probability)*np.sqrt(corpus[domain])/normal*role))
        return plan

    def sample(self, group, mode='ego', *, member=None, start=None, yaw=None):
        members=self.members[group]
        if not members:
            raise ValueError(f'No valid native30 windows for {group}')
        # Duration/valid-window weighting rather than equal recording weighting.
        eligible=[i for i,(m,_) in enumerate(members) if mode in m.get('allowed_modes',
            ['ego','exo'] if group in ('camera_wearer','interactee') else ['ego']) and len(self.starts_for(group,i,mode))]
        if not eligible:raise ValueError(f'No {mode} observation coverage for {group}')
        mi=int(self.rng.choice(eligible,p=np.array([len(self.starts_for(group,i,mode)) for i in eligible])/sum(len(self.starts_for(group,i,mode)) for i in eligible))) if member is None else member
        if mi not in eligible:raise ValueError('Requested observer mode unavailable; do not substitute a virtual PV')
        meta,_=members[mi];starts=self.starts_for(group,mi,mode); first=int(self.rng.choice(starts)) if start is None else int(start)
        if first not in starts:
            raise ValueError('Window crosses an invalid native30 frame')
        folder=Path(meta['path']); sl=slice(first,first+self.frames)
        motion=np.asarray(self.load(folder,'motion_world')[sl]).copy();rest=np.asarray(self.load(folder,'rest')).copy()
        cam=np.asarray(self.load(folder,'camera_position_scenemi_yup')[sl]).copy()
        cr=np.asarray(self.load(folder,'camera_rotation_scenemi_yup')[sl]).copy()
        if mode=='exo':
            transform=external_transforms(self.load,folder,first,self.frames)
            cam=transform[:,:3,3].copy();cr=transform[:,:3,:3].copy()
        origin=cam[0].copy(); anchor=anchor_rotation(cr)
        # Apply after camera-first anchoring, so augmentation cannot cancel out.
        world_yaw=float(self.rng.uniform(-np.pi,np.pi)) if yaw is None else float(yaw)
        anchor=Rotation.from_rotvec([0,world_yaw,0]).as_matrix()@anchor
        # Raw export stores proper residual 6D columns.
        six=motion[:,3:135].reshape(-1,22,6)+np.asarray(IDENTITY_6D)
        r=np.stack((six[...,:3],six[...,3:],np.cross(six[...,:3],six[...,3:])), -1)
        r[:,0]=anchor[None]@r[:,0]
        root=(motion[:,:3]*2+rest[0]-origin)@anchor.T
        joints=(motion[:,135:].reshape(-1,22,3)*2-origin)@anchor.T
        motion=np.concatenate(((root-rest[0])/2,(rotation6d(r)-IDENTITY_6D).reshape(-1,132),joints.reshape(-1,66)/2),-1).astype(np.float32)
        optical=np.concatenate(((cam-origin)@anchor.T/2,rotation6d(anchor[None]@cr)), -1).astype(np.float32)
        sample=dict(motion=motion,joints=(joints/2).astype(np.float32),rest=rest.astype(np.float32),camera=optical,
                    body_type=np.eye(2,dtype=np.float32)[int(meta['native_body']['model']=='smplx')],
                    body_scale=np.array([meta['native_body']['scale']],np.float32),
                    contact_target=np.zeros((self.frames,22),np.float32),contact_valid=np.zeros((self.frames,22),bool))
        # Reliability is a loss-only label. Screen original world targets with
        # neighboring source frames, then slice; never expose it to inference.
        lo=max(0,first-1);hi=min(meta['frames'],first+self.frames+1)
        quality=quality_weights(np.asarray(self.load(folder,'motion_world')[lo:hi]),rest,self.quality_config)
        sample.update({key:value[first-lo:first-lo+self.frames].copy() for key,value in quality.items()})
        # Contact labels have their own raw30 clock and remain training-only.
        if (folder/'contact_target.npy').exists():
            sample['contact_target']=np.asarray(self.load(folder,'contact_target')[sl]).copy()
            sample['contact_valid']=np.asarray(self.load(folder,'contact_valid')[sl]).copy()
        identity=dict(meta,start=first,source_start_30fps=int(self.load(folder,'source_frame_ids')[first]),
                      origin=origin.tolist(),anchor=anchor.tolist(),yaw_augmentation_rad=world_yaw,mode=mode)
        return sample,identity

    def sampler_state(self):
        return copy.deepcopy(self.rng.bit_generator.state)

    def load_sampler_state(self,state):
        self.rng.bit_generator.state=copy.deepcopy(state)
