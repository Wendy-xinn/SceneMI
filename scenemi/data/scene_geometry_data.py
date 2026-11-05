"""Prepared EGO geometry batches; public observations and private labels separate."""
import numpy as np
import torch

from scenemi.config import ROOT
from scenemi.data.ego import FourEgoData, ALL_KEYS, normalize_sample
from scenemi.data.scene_ego import apply_scene_review, SCENE_KEYS
from scenemi.data.contact_labels import GrailContactLabels
from scenemi.data.amass_dataset import AmassDataset
from scenemi.data.scene_geometry_cache import PreparedSceneDataset, PreparedAmass
from experiments.offline_camera_retrain_v1.unified_geometry import occupancy_from_points
from experiments.offline_camera_retrain_v1.data import anchor_rotation

GEOMETRY_KEYS = ('static_points','static_normals','static_valid','static_normal_valid',
                 'frame_normals','frame_normal_valid')
PRIVATE_KEYS = ('contact_weight',)


def payload(batch):
    from scenemi.data.ego import payload as camera_payload
    return dict(camera_payload(batch),**{k:batch[k] for k in SCENE_KEYS+GEOMETRY_KEYS})


def collate(samples, device, *, pin=False):
    result = {}
    for key in ALL_KEYS+SCENE_KEYS+GEOMETRY_KEYS+PRIVATE_KEYS:
        cpu = torch.from_numpy(np.stack([s[key] for s in samples]))
        if pin and device=='cuda':
            cpu = cpu.pin_memory()
        result[key] = cpu.to(device,non_blocking=pin and device=='cuda')
    return result


class GeometryEgoData(FourEgoData):
    def __init__(self,c,paths,split):
        if split not in ('train','validation'):
            raise ValueError('Geometry data never opens test')
        self.split = split
        root = ROOT/c['prepared_data']['root']
        self.amass = AmassDataset(ROOT/c['amass_index'],paths,'train' if split=='train' else 'val')
        # The sampler needs only the public canonical rest skeleton. Recover it
        # from the pinned parent rather than constructing another full model.
        parent = torch.load(ROOT/c['continuation']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        common = parent['model']['shape.rest_template'][:1].detach().numpy().copy()
        self.native = PreparedSceneDataset(ROOT/c['scene_paths'],common,split=split,config=c,
            selection=ROOT/c['selection'],cache_root=root/'scene',geometry_policy=c['geometry'])
        self.datasets = {'amass':self.amass,**{source:self.native for source in ('egobody','trumans','grail')}}
        evidence = apply_scene_review(self.native,c)
        self.native.index_paths()
        self.amass = PreparedAmass(self.amass,root/'amass')
        self.datasets['amass'] = self.amass
        self.labels = (GrailContactLabels(self.native.selection_bundle,root/'contact_labels',
            ROOT/c['scene_stage']['contact_policy'],prepared_evidence=evidence) if split=='train' else None)
        self.c = c

    def empty_geometry(self):
        n,k = self.c['geometry']['static_points'],self.c['geometry']['frame_points']
        return dict(occupancy=np.zeros((24,48,48),np.float32),
            frame_scene_points=np.zeros((128,k,3),np.float32),
            frame_scene_owner=np.full((128,k),-1,np.int64),
            static_points=np.zeros((n,3),np.float32),static_normals=np.zeros((n,3),np.float32),
            static_valid=np.zeros(n,bool),static_normal_valid=np.zeros(n,bool),
            frame_normals=np.zeros((128,k,3),np.float32),frame_normal_valid=np.zeros((128,k),bool),
            contact_weight=np.ones((128,2),np.float32))

    def sample(self,source,rng,*,member=None,start=None):
        sample, identity = super().sample(source,rng,member=member,start=start)
        geometry = self.empty_geometry()
        if source!='amass':
            row = self.native.records[identity['member']]
            saved = self.native.cached(row)
            span = slice(identity['start'],identity['start']+128)
            origin = np.asarray(saved['camera_position_scenemi_yup'][identity['start']],np.float64).copy()
            origin[1] = 0.
            anchor = anchor_rotation(np.asarray(saved['camera_rotation_scenemi_yup'][span],np.float64))
            local = ((saved['static_full']-origin)@anchor.T).astype(np.float32)
            geometry['occupancy'] = occupancy_from_points(local)
            n = len(saved['static_points'])
            geometry['static_points'][:n] = (saved['static_points']-origin)@anchor.T
            geometry['static_normals'][:n] = saved['static_normals']@anchor.T
            geometry['static_valid'][:n] = True
            geometry['static_normal_valid'][:n] = saved['static_normal_valid']
            owners = saved['frame_owner'][span].copy()
            frame = ((saved['frame_points'][span]-origin)@anchor.T).astype(np.float32)
            frame[owners<0] = 0.
            geometry.update(frame_scene_points=frame,frame_scene_owner=owners,
                frame_normals=(saved['frame_normals'][span]@anchor.T).astype(np.float32),
                frame_normal_valid=saved['frame_normal_valid'][span].copy(),
                contact_weight=saved['contact_weight'][span].copy())
            key,_ = self.native.cache_identity(row)
            identity.update(scene_input=True,scene_cache_key=key,
                scene_map_protocol='observed-recording-scene-v1',scene_map_causal=False,
                scene_map_observed_only=True,source_scene_mesh_is_input=False,
                scene_map_observed_frames=[0,len(saved['motion_world'])-1],
                scene_map_points_before_limit=len(local),scene_map_points_after_limit=n,
                synthetic_chair=bool(row.get('synthetic_chair')),
                scene_geometry='local oriented observed patches; not a complete SDF')
        return dict(sample,**geometry),identity

    def sample_camera_case(self,case):
        sample,identity = self.sample_case(case)
        sample.update(self.empty_geometry())
        identity = dict(identity,scene_input=False)
        return sample,identity

    def pairs(self,rng,*,forced=False):
        result = []
        for source in self.c['data']['datasets']:
            member = None
            if forced:
                member = (0 if source=='amass' else
                    min(i for ids in self.native.groups[source].values() for i in ids))
            for _ in range(self.c['data']['source_quota_per_update']):
                result.append(self.sample(source,rng,member=member))
        return result
