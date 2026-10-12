"""Strict raw30 bundles, scoped train/validation samplers and common world yaw."""
import copy
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.offline_camera_retrain_v1.native30_export import PROTOCOL, inventory
from experiments.offline_camera_retrain_v1.data import anchor_rotation, rotation6d
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D


class Native30Data:
    def __init__(self, manifest, split, seed=20261012, *, diagnostic=False, frames=192):
        self.manifest=Path(manifest); self.split=split; self.rng=np.random.default_rng(seed)
        self.frames=frames; self.members={g:[] for g in ('trumans','camera_wearer','interactee','rich')}
        self.arrays={}
        expected_sources={(r['group'],r['sequence_id']):r for r in inventory(split)} if not diagnostic else {}
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
            ids=np.load(folder/'source_frame_ids.npy');valid=np.load(folder/'observation_valid.npy')
            if len(ids) != meta['frames'] or not np.all(np.diff(ids)==1) or valid.shape != ids.shape:
                raise ValueError('Incomplete raw30 clock')
            if valid.dtype!=np.bool_ or ids[0]!=meta['source_first'] or ids[-1]!=meta['source_last']:
                raise ValueError('Invalid observation mask or source clock bounds')
            if not diagnostic:
                source=expected_sources.get((meta['group'],meta['sequence_id']))
                if source is None or ids[0]!=source['raw_first'] or ids[-1]!=source['raw_last_exclusive']-1:
                    raise ValueError('Full training bundle must cover its original scoped raw30 source')
            sums=np.r_[0,np.cumsum(~valid)]; starts=np.flatnonzero(sums[frames:]-sums[:-frames]==0)
            if len(starts):
                self.members[meta['group']].append((dict(meta,path=str(folder)),starts))
        if not diagnostic:
            expected={(r['group'],r['sequence_id']) for r in inventory(split) if r['raw_last_exclusive']-r['raw_first']>=frames}
            # A full export with no valid windows may be retained but not sampled.
            all_rows=[json.loads(l) for l in self.manifest.read_text().splitlines()]
            covered={(r['group'],r['sequence_id']) for r in all_rows if r['split']==split and not r['diagnostic_only']}
            if expected-covered:
                raise ValueError(f'Full raw30 coverage missing {len(expected-covered)} sources; no native20 fallback')

    def load(self, folder, key):
        cache=self.arrays.setdefault(str(folder),{})
        if key not in cache:
            cache[key]=np.load(Path(folder)/f'{key}.npy',mmap_mode='r')
        return cache[key]

    def sampling_plan(self, ego_probability=.7):
        duration={g:sum(len(starts)/30 for _,starts in members) for g,members in self.members.items()}
        corpus={'trumans':duration['trumans'],'egobody':duration['camera_wearer']+duration['interactee'],'rich':duration['rich']}
        normal=sum(np.sqrt(v) for v in corpus.values()); plan=[]
        for g in duration:
            if duration[g] == 0:
                continue
            domain='egobody' if g in ('camera_wearer','interactee') else g
            role=duration[g]/corpus[domain]
            plan.append(dict(group=g,mode='ego',probability=ego_probability*np.sqrt(corpus[domain])/normal*role))
        egobody=corpus['egobody']
        if egobody <= 0:
            raise ValueError('Unified sampler needs EgoBody exo sources')
        for g in ('camera_wearer','interactee'):
            if duration[g]:
                plan.append(dict(group=g,mode='exo',probability=(1-ego_probability)*duration[g]/egobody))
        return plan

    def sample(self, group, mode='ego', *, member=None, start=None, yaw=None):
        members=self.members[group]
        if not members:
            raise ValueError(f'No valid native30 windows for {group}')
        # Duration/valid-window weighting rather than equal recording weighting.
        mi=int(self.rng.choice(len(members),p=np.array([len(v) for _,v in members])/sum(len(v) for _,v in members))) if member is None else member
        meta,starts=members[mi]; first=int(self.rng.choice(starts)) if start is None else int(start)
        if first not in starts:
            raise ValueError('Window crosses an invalid native30 frame')
        folder=Path(meta['path']); sl=slice(first,first+self.frames)
        motion=np.asarray(self.load(folder,'motion_world')[sl]).copy();rest=np.asarray(self.load(folder,'rest')).copy()
        cam=np.asarray(self.load(folder,'camera_position_scenemi_yup')[sl]).copy()
        cr=np.asarray(self.load(folder,'camera_rotation_scenemi_yup')[sl]).copy()
        if mode=='exo':
            if group not in ('camera_wearer','interactee'):
                raise ValueError('Current real exo export is EgoBody master only')
            transform=np.asarray(self.load(folder,'exo_sensor_to_world'))
            cam=np.broadcast_to(transform[:3,3],cam.shape).copy(); cr=np.broadcast_to(transform[:3,:3],cr.shape).copy()
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
