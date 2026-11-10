"""Train-only, source-pinned GRAIL contact labels, cached per recording."""
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import torch

from scenemi.config import ROOT
from scenemi.contact_loss import load_policy, contact_reliability
from scenemi.data.amass_prepare import sha256
from scenemi.data.grail_contact import ContactBody, load_source
from scenemi.data.grail_quality import contact_probabilities
from scenemi.data.grail_support import sole_measure
from experiments.offline_camera_retrain_v1.ops.grail_source import load_pickle
from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array


class GrailContactLabels:
    """Full meshes and estimated contact are private training labels only."""

    def __init__(self, bundle, output, policy, *, prepared_evidence=None):
        self.policy_path = Path(policy)
        self.policy = load_policy(policy)
        self.output = Path(output)
        self.rows = {r['path']: r for key in bundle.policy['grail_selection']['ego']['train']
                     for r in bundle.rows(key)}
        self.body = None
        self.prepared_evidence = prepared_evidence or {}
        self.identity = {str(p): sha256(p) for p in (Path(policy),
            ROOT / 'configs/grail_ego_contact.json', ROOT / 'configs/grail_ego_support_v2.json',
            Path(__file__), ROOT / 'scenemi/contact_loss.py', ROOT / 'scenemi/data/grail_contact.py',
            ROOT / 'scenemi/data/grail_support.py', ROOT / 'scenemi/data/grail_quality.py')}

    def window(self, path, start):
        row = self.rows[str(path)]
        if row['split'] != 'train':
            raise ValueError('Contact preparation is train-only')
        identity = dict(path=str(path), metadata_sha256=row['metadata_sha256'],
                        implementation=self.identity, split='train', scope='full_recording')
        prepared = getattr(self, 'prepared_evidence', {}).get(str(path))
        if prepared is not None:
            identity['prepared_evidence_sha256'] = prepared['sha256']
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        file = self.output / f'{key}.npz'
        if not file.exists():
            # Source-body construction must not consume the training CPU RNG.
            with torch.random.fork_rng(devices=[]):
                weight = self.compute(row)
            self.output.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(weight.tobytes()).hexdigest()
            with tempfile.NamedTemporaryFile(dir=self.output, prefix=key + '.', suffix='.npz', delete=False) as stream:
                temporary = Path(stream.name)
            try:
                np.savez_compressed(temporary, weight=weight, identity=json.dumps(identity, sort_keys=True),
                                    weight_sha256=digest)
                temporary.replace(file)
            finally:
                temporary.unlink(missing_ok=True)
        with np.load(file, allow_pickle=False) as saved:
            weight = saved['weight']
            if (json.loads(str(saved['identity'])) != identity
                    or hashlib.sha256(weight.tobytes()).hexdigest() != str(saved['weight_sha256'])
                    or weight.shape != (row['frames'], 2) or not np.isfinite(weight).all()
                    or ((weight < 0) | (weight > 1)).any()):
                raise ValueError('Contact label cache identity/content differs')
            result = weight[start:start + 128].copy()
        if result.shape != (128, 2):
            raise ValueError('Contact labels leave the source recording')
        return result, dict(contact_label_cache=str(file), contact_label_key=key,
                            contact_weight_mean=float(result.mean()),
                            contact_weight_zero_fraction=float((result == 0).mean()))

    def compute(self, row):
        prepared = self.prepared_evidence.get(row['path'])
        if prepared is not None:
            file = Path(prepared['file'])
            if sha256(file) != prepared['sha256']:
                raise ValueError('Prepared source-contact evidence changed')
            with np.load(file, allow_pickle=False) as saved:
                metadata = json.loads(str(saved['metadata_json']))
                if (metadata['source_metadata_sha256'] != row['metadata_sha256']
                        or metadata['split'] != 'train'
                        or metadata['contact_policy_sha256'] != sha256(getattr(self, 'policy_path', ROOT / 'configs/scene_contact_reliability.yaml'))
                        or metadata['contact_loss_sha256'] != sha256(ROOT / 'scenemi/contact_loss.py')):
                    raise ValueError('Prepared contact evidence source/policy differs')
                return saved['contact_weight'].copy()
        meta, params, rotation, offset = load_source(row)
        if meta['split'] != 'train':
            raise ValueError('No validation/test body in contact preparation')
        for key in ('recon', 'object_usd'):
            if sha256(meta['source_files'][key]) != meta['source_hashes'][key]:
                raise ValueError('Contact source identity changed')
        if self.body is None:
            self.body = ContactBody(json.loads((ROOT / 'configs/grail_ego_contact.json').read_text()))
        vertices, joints = self.body.forward(params, rotation, offset)
        target = readonly_array(meta, 'motion_world')[:, -66:].reshape(-1, 22, 3) * 2
        if np.abs(joints - target).max() > 2e-6:
            raise ValueError('Source contact body and motion targets disagree')
        ov, of = readonly_array(meta, 'scene_object_vertices'), readonly_array(meta, 'scene_object_faces')
        sv = np.concatenate((ov, readonly_array(meta, 'scene_ground_vertices')))
        sf = np.concatenate((of, readonly_array(meta, 'scene_ground_faces') + len(ov)))
        support = json.loads((ROOT / 'configs/grail_ego_support_v2.json').read_text())
        measured = sole_measure(vertices, self.body.feet, sv, sf, support)
        raw = load_pickle(meta['source_files']['recon'])
        probability = contact_probabilities(raw['human_data']['foot_contact_probs'],
            meta['frames'], meta['source_fps'], meta['fps'])
        return contact_reliability(probability, measured['sole_gap_q10_m'],
            measured['sole_supported_fraction'], measured['foot_speed_mps'],
            readonly_array(meta, 'observation_valid'), self.policy)['weight']
