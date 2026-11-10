"""Four-corpus labels for the unchanged public SMPL-H EgoGVAE decoder.

Native rotations share the checked 22-joint hierarchy, but shape coefficients
do not share a basis. Native joints remain the supervision/evaluation truth;
only AMASS supplies SMPL-H beta and released contact labels.
"""
import json

import numpy as np
import torch

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset
from scenemi.data.ego import FourEgoData
from scenemi.data.grail_visibility import sha256
from scenemi.geometry.shape import SmplhShape
from scenemi.native_joint_data import SceneFreeEgoDataset
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d

SOURCES = ('amass', 'egobody', 'trumans', 'grail')


class EgoGVAEData(FourEgoData):
    """Reuse exact camera-only sampling without constructing a SceneMI network."""
    def __init__(self, reference, split, *, paths_file=ROOT/'configs/paths.local.yaml'):
        if split not in ('train', 'validation'):
            raise ValueError('EgoGVAE adaptation never opens test')
        proof = json.loads((ROOT/'runs/data_inventory/CURRENT_SELECTION_CHECK.json').read_text())
        if proof['state'] != 'passed' or proof['selection_sha256'] != sha256(ROOT/reference['selection']):
            raise ValueError('Current checked data selection requires a matching passed audit')
        paths = load_paths(paths_file)
        self.split, self.paths = split, paths
        self.amass = AmassDataset(ROOT/reference['amass_index'], paths,
                                  'train' if split == 'train' else 'val')
        pelvis = SmplhShape(paths['smplh_models']/'neutral/model.npz').rest_template[:1].numpy()
        self.native = SceneFreeEgoDataset(ROOT/reference['scene_paths'], pelvis, split=split,
                                         config=reference, selection=ROOT/reference['selection'])
        self.datasets = dict(amass=self.amass, **{s:self.native for s in SOURCES[1:]})

    def training_sample(self, source, rng, **kwargs):
        sample, identity = self.sample(source, rng, **kwargs)
        contact = np.zeros((128, 21), np.float32)
        valid = source == 'amass'
        if valid:
            row = self.amass.members[kwargs['member']] if 'member' in kwargs else None
            if row is None:
                # AMASS identities store a family rather than the sampled index.
                if not hasattr(self, '_amass_families'):
                    self._amass_families = {r['family']:r for r in self.amass.members}
                row = self._amass_families[identity['family']]
            variant = next(v for v in row['variants'] if v['reflected'] == identity['reflected'])
            with np.load(self.paths['prepared_amass']/variant['path'], allow_pickle=False) as raw:
                first = identity['start'] + 1  # Same single relative-pose drop as AmassDataset.
                contact = raw['contacts'][first:first+128, 1:22].astype(np.float32)
            if contact.shape != (128,21) or not np.isfinite(contact).all() or not ((contact>=0)&(contact<=1)).all():
                raise ValueError('Invalid released AMASS contact labels')
        sample.update(target_contacts=contact, beta_label_valid=np.bool_(valid),
                      contact_label_valid=np.bool_(valid))
        return sample, identity


def collate_labels(samples, public, device):
    """Separate observation-derived head inputs from training-only labels."""
    batch = {k:torch.as_tensor(np.stack([s[k] for s in samples]), device=device)
             for k in samples[0]}
    if not torch.all(batch['camera_available'] == 1):
        raise ValueError('This clean baseline requires all 128 camera observations')
    cpf = public.cpf(batch['camera'], batch['sensor_profile'])
    # SceneMI stores identity-centred COLUMNS; upstream uses absolute ROWS.
    local = rotation_from_6d(batch['motion'][...,3:135].reshape(-1,128,22,6))
    rotation_error = (local.transpose(-1,-2) @ local - torch.eye(3,device=device)).abs().amax()
    if rotation_error > 2e-4 or (torch.linalg.det(local)-1).abs().amax() > 2e-4:
        raise ValueError('Source rotations are not a shared SO(3) body pose')
    motion = public.gvae_rotation.matrix_to_rotation_6d(local).reshape(-1,128,132)
    truth = forward_kinematics(batch['motion'], batch['target_rest'])
    return dict(head=public.gvae_head(cpf), motion=motion, cpf=cpf,
                joints=truth, local_rotations=local,
                betas=batch['target_betas'], contacts=batch['target_contacts'],
                beta_valid=batch['beta_label_valid'], contact_valid=batch['contact_label_valid'],
                rest=batch['target_rest'], body_format=batch['body_format'])
