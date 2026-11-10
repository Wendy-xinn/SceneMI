"""Four-dataset camera-only sampling and fixed validation membership.

Only train/validation are exposed. Camera trajectories use their source
observations without artificial perturbations; coordinate normalization is
shared by inputs and targets.
"""
import numpy as np
import torch
from scenemi.config import ROOT, EGO_SOURCES as SOURCE_ORDER
from scenemi.data.amass_dataset import AmassDataset
from scenemi.native_joint_data import SceneFreeEgoDataset, validation_cases
from scenemi.observations import INPUT_KEYS
from scenemi.pipeline import build_ego_model
from scenemi.selection import validation_manifest

ALL_KEYS = INPUT_KEYS + ('motion', 'target_rest', 'target_betas',
                         'target_native_betas', 'target_template',
                         'body_format', 'sensor_profile')

def _zeros(shape, dtype=np.float32):
    return np.zeros(shape, dtype=dtype)


def normalize_sample(sample, identity, source):
    """Make AMASS and native camera-only records share one inference schema."""
    sample = dict(sample)
    sample['body_format'] = np.asarray(sample.get('body_format', 0), dtype=np.int64)
    sample['sensor_profile'] = np.asarray(sample.get('sensor_profile', 0), dtype=np.int64)
    sample['target_betas'] = np.asarray(sample.get('target_betas', _zeros((16,))), dtype=np.float32)
    sample['target_native_betas'] = np.asarray(sample.get('target_native_betas', _zeros((10,))), dtype=np.float32)
    sample['target_template'] = np.asarray(sample.get('target_template', 0), dtype=np.int64)
    for key in ALL_KEYS:
        if key not in sample:
            raise ValueError(f'Missing normalized EGO field {key} for {source}')
    identity = dict(identity, dataset=source, mode='ego')
    return {key: sample[key] for key in ALL_KEYS}, identity


class FourEgoData:
    def __init__(self, c, paths, split, *, native_factory=SceneFreeEgoDataset):
        if split not in ('train', 'validation'):
            raise ValueError('Four-corpus EGO data never opens test')
        self.split = split
        self.amass = AmassDataset(ROOT / c['amass_index'], paths,
                                  'train' if split == 'train' else 'val')
        common = build_ego_model(c, paths).shape.rest_template[:1].detach().cpu().numpy()
        self.native = native_factory(ROOT / c['scene_paths'], common,
                                          split=split, config=c,
                                          selection=ROOT / c['selection'])
        self.datasets = {'amass': self.amass, 'egobody': self.native,
                         'trumans': self.native, 'grail': self.native}

    def sample(self, source, rng, *, member=None, start=None):
        if source == 'amass':
            sample, identity = self.amass.sample(rng, 'ego', member=member, start=start,
                                                 temporal_dropout=None, yaw=0.)
        else:
            if member is None:
                families = self.native.families[source]
                family = families[int(rng.integers(len(families)))]
                member = int(rng.choice(self.native.groups[source][family]))
            sample, identity = self.native.sample(rng, member=member, start=start, yaw=0.)
        return normalize_sample(sample, identity, source)

    def sample_case(self, case):
        rng = np.random.default_rng(777 + int(case['case_id']))
        return self.sample(case['dataset'], rng, member=case['member'], start=case['start'])


def collate(samples, device):
    return {key: torch.as_tensor(np.stack([sample[key] for sample in samples]), device=device)
            for key in ALL_KEYS}


def payload(batch):
    return {key: batch[key] for key in INPUT_KEYS + ('body_format', 'sensor_profile')}


def source_cases(data, quick=False, families_per_dataset=8):
    cases = []
    amass_cases = validation_manifest(data.amass.members)
    if not quick:
        cases.extend(dict(x, dataset='amass', case_id=len(cases)) for x in amass_cases)
    else:
        chosen = sorted({x['family'] for x in amass_cases})[:families_per_dataset]
        cases.extend(dict(x, dataset='amass', case_id=len(cases))
                     for x in amass_cases if x['family'] in chosen) 
    native_cases = validation_cases(data.native.records)
    if quick:
        for source in ('egobody', 'trumans', 'grail'):
            selected = sorted({x['family'] for x in native_cases if x['dataset'] == source})[:families_per_dataset]
            chosen = [x for x in native_cases if x['dataset'] == source and x['family'] in selected]
            cases.extend(dict(x, case_id=len(cases)) for x in chosen)
    else:
        cases.extend(dict(x, case_id=len(cases)) for x in native_cases)
    return cases


