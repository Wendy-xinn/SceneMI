"""Verified AMASS windows, observation-only conditioning and separate labels."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from scenemi.config import ROOT
from scenemi.data.amass import split_definition
from scenemi.data.amass_prepare import atomic_json, sha256, source_indices
from scenemi.geometry.body import AMASS_ZUP_TO_YUP, PARENTS, encode_body
from scenemi.geometry.shape import load_shape_basis, numpy_profile
from scenemi.observations import INPUT_KEYS
from .observation_gaps import sample_valid_frames, mask_observations

INDEX_PROTOCOL = 'amass128-verified-windows-v1'


def build_index(paths, *, diagnostic=False):
    """Freeze a complete manifest after checking identities and output hashes.

    A diagnostic index uses ONLY the completed 19-corpus pilot prefix. It can
    validate real batches while preparation runs, but cannot start full training.
    """
    root = paths['prepared_amass']
    state = json.loads((paths['output'] / 'preparation/STATUS.json').read_text())
    if not diagnostic and state['state'] != 'complete':
        raise ValueError('Full AMASS preparation is not complete; no formal training index')
    manifest = root / 'manifest.jsonl'
    lines = manifest.read_bytes().splitlines(keepends=True)
    if diagnostic:
        pilot = json.loads((paths['output'] / 'preparation/PILOT.json').read_text())
        lines = lines[:pilot['committed_sources']]
    elif len(lines) != state['total_raw_files'] or not lines[-1].endswith(b'\n'):
        raise ValueError('Incomplete final AMASS manifest')
    payload = b''.join(lines)
    rows = [json.loads(line) for line in lines]
    if len({row['source'] for row in rows}) != len(rows):
        raise ValueError('Duplicate source families in preparation manifest')
    definition = split_definition()['splits']
    job_path = root / 'JOB.json'
    job = json.loads(job_path.read_text())
    inventory_path = paths['output']/'raw_inventory.jsonl'
    if sha256(inventory_path) != job['inventory_sha256']:
        raise ValueError('Original inventory identity changed')
    inventory = {row['source']:row for row in map(json.loads, inventory_path.read_text().splitlines())}
    if not diagnostic and set(inventory) != {row['source'] for row in rows}:
        raise ValueError('Final manifest does not cover the complete original inventory')
    for name, digest in job['local_source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Sealed preprocessing source changed: {name}')
    model_path = paths['smplh_models'] / 'neutral/model.npz'
    if sha256(model_path) != job['assets_sha256'][str(model_path)]:
        raise ValueError('Native SMPL-H model changed since preparation')
    families, counts, variants_seen = [], Counter(), set()
    for row in rows:
        source = inventory.get(row['source'])
        if source is None or any(row[key] != source[key] for key in ('family', 'split', 'corpus')):
            raise ValueError('Prepared source membership differs from the original inventory')
        if any(row['inventory'][key] != source[key] for key in ('bytes', 'mtime_ns', 'split', 'family')):
            raise ValueError('Prepared source identity differs from inventory')
        split, corpus = row['split'], row['corpus']
        if corpus not in definition[split] or Path(row['source']).parts[0] != corpus:
            raise ValueError('Wrong corpus/split membership')
        raw = paths['amass_raw'] / row['source']
        raw.resolve().relative_to(paths['amass_raw'])
        identity = row['inventory']
        stat = raw.stat()
        if (stat.st_size, stat.st_mtime_ns) != (identity['bytes'], identity['mtime_ns']):
            raise ValueError(f'Raw source changed: {raw}')
        if row.get('raw_sha256') and sha256(raw) != row['raw_sha256']:
            raise ValueError(f'Raw source bytes changed: {raw}')
        if row['state'] != 'prepared':
            if row['state'] not in ('filename_excluded', 'too_short_upstream'):
                raise ValueError('Unknown or failed sequence in manifest')
            continue
        variants = row['variants']
        if len(variants) != 2 or {v['reflected'] for v in variants} != {False, True}:
            raise ValueError('Expected paired original/reflected variants')
        for variant in variants:
            output = root / variant['path']
            output.resolve().relative_to(root)
            if variant['path'] in variants_seen:
                raise ValueError('Duplicate output identity')
            variants_seen.add(variant['path'])
            if output.stat().st_size != variant['bytes'] or sha256(output) != variant['sha256']:
                raise ValueError(f'Prepared output changed: {output}')
            usable = variant['frames_0a'] - 1
            if usable != variant['frames_after_relative_pose_drop'] or (usable >= 128) != variant['eligible_128']:
                raise ValueError('128-frame eligibility/relative-pose clock changed')
            if variant['fk_max_error_m'] > 2e-4:
                raise ValueError('Unaudited FK in preparation manifest')
        if all(v['eligible_128'] for v in variants):
            families.append(row)
            counts[split] += 1
    if not all(counts[key] for key in ('train', 'val', 'test')):
        raise ValueError('No eligible data for one of the three disjoint splits')
    result = dict(protocol=INDEX_PROTOCOL, complete=not diagnostic,
                  diagnostic_only=diagnostic, frames=128, fps=30,
                  representation='201d-common-neutral-pelvis-v1',
                  manifest_sha256=hashlib.sha256(payload).hexdigest(),
                  job_sha256=sha256(job_path), body_model_sha256=sha256(model_path),
                  source_inventory_sha256=job['inventory_sha256'],
                  source_rows_verified=len(rows), eligible_families=dict(counts), families=families)
    folder = paths['output'] / 'index'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ('diagnostic.json' if diagnostic else 'complete.json')
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('A frozen data index already exists with a different identity')
    if not path.exists():
        atomic_json(path, result)
    print(json.dumps(dict(index=str(path), complete=result['complete'], eligible_families=dict(counts))), flush=True)
    return path


def six_columns(rotation):
    return np.concatenate((rotation[..., :, 0], rotation[..., :, 1]), -1)


def camera_anchor(rotation):
    forward = rotation[:, 2].copy()
    if np.linalg.norm(forward[[0, 2]]) < 1e-5:
        # A near-vertical optical axis has no usable yaw. Use measured right.
        right = rotation[:, 0]
        yaw = np.arctan2(-right[2], right[0])
    else:
        yaw = np.arctan2(forward[0], forward[2])
    return Rotation.from_rotvec([0., -yaw, 0.]).as_matrix()


def exo_mask(rng, frames=128):
    """Synthetic missingness only, explicitly unrelated to rendered occlusion."""
    patterns = (tuple(range(22)), (0, 1, 2, 3, 6, 9, 12, 15),
                (0, 1, 2, 4, 5, 7, 8, 10, 11), (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21),
                (0, 3, 6, 9, 12, 15, 16, 18, 20), (0, 3, 6, 9, 12, 15, 17, 19, 21))
    pattern = int(rng.integers(len(patterns)))
    mask = np.zeros((frames, 22), bool)
    mask[:, patterns[pattern]] = True
    if rng.random() < .5:
        start = int(rng.integers(1, frames-32))
        width = int(rng.integers(8, 33))
        joints = ((16, 18, 20), (17, 19, 21), (4, 7, 10), (5, 8, 11))[int(rng.integers(4))]
        mask[start:start+width, joints] = False
    return mask, pattern


class AmassDataset:
    def __init__(self, index, paths, split='train', *, diagnostic=False):
        self.index_path = Path(index)
        self.index = json.loads(self.index_path.read_text())
        if self.index['protocol'] != INDEX_PROTOCOL or self.index['frames'] != 128:
            raise ValueError('Wrong AMASS index protocol')
        if not self.index['complete'] and not diagnostic:
            raise ValueError('A partial diagnostic index cannot train the formal model')
        if split not in ('train', 'val', 'test'):
            raise ValueError('Explicit train, val or test split required')
        self.paths, self.split = paths, split
        self.members = [row for row in self.index['families'] if row['split'] == split]
        if not self.members:
            raise ValueError(f'No eligible {split} source families')
        model_path = paths['smplh_models'] / 'neutral/model.npz'
        if sha256(model_path) != self.index['body_model_sha256']:
            raise ValueError('Body model differs from frozen index')
        self.basis = load_shape_basis(model_path)

    @lru_cache(maxsize=8)
    def load(self, relative, digest):
        path = self.paths['prepared_amass'] / relative
        if sha256(path) != digest:
            raise ValueError(f'Changed training output: {path}')
        with np.load(path, allow_pickle=False) as data:
            # Match upstream 0b's single drop, without shifting twice.
            return dict(poses=np.concatenate((data['root_orient'][1:], data['pose_body'][1:]), -1),
                        joints=data['joints'][1:].copy(), betas=data['betas'][0].copy())

    def sample(self, rng, mode, *, member=None, start=None, reflected=None, yaw=None,
               temporal_dropout=None):
        if mode not in ('ego', 'exo'):
            raise ValueError('Unknown observer mode')
        mi = int(rng.integers(len(self.members))) if member is None else member
        row = self.members[mi]
        vi = int(rng.integers(2)) if reflected is None else int(reflected)
        variant = next(v for v in row['variants'] if v['reflected'] == bool(vi))
        data = self.load(variant['path'], variant['sha256'])
        n = len(data['poses'])
        first = int(rng.integers(n-128+1)) if start is None else start
        if first < 0 or first+128 > n:
            raise ValueError('Window crosses source boundaries')
        poses = data['poses'][first:first+128]
        local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(128, 22, 3, 3)
        local[:, 0] = AMASS_ZUP_TO_YUP @ local[:, 0]
        joints = data['joints'][first:first+128].astype(np.float64) @ AMASS_ZUP_TO_YUP.T
        rest, mount, native = numpy_profile(self.basis, data['betas'])
        rotations = [local[:, 0]]
        for joint in range(1, 22):
            rotations.append(rotations[PARENTS[joint]] @ local[:, joint])
        head_rotation = rotations[15]
        camera_position = joints[:, 15] + (head_rotation @ mount[:, None])[..., 0]
        mask, pattern = exo_mask(rng)
        frame_valid, outage = sample_valid_frames(rng, mode, 128, temporal_dropout)
        anchor_frame = int(np.flatnonzero(frame_valid)[0])
        if mode == 'ego':
            origin = camera_position[anchor_frame].copy()
            anchor = camera_anchor(head_rotation[anchor_frame])
        else:
            # AMASS has no real external camera. Do not invent its extrinsics
            # from hidden body truth. Anchor from the AVAILABLE joint tracks.
            origin = joints[anchor_frame, mask[anchor_frame]].mean(0)
            anchor = np.eye(3)
        origin[1] = 0.  # Preserve measured camera/joint height above 0a's floor.
        yaw_value = float(rng.uniform(-np.pi, np.pi)) if yaw is None else float(yaw)
        anchor = Rotation.from_rotvec([0., yaw_value, 0.]).as_matrix() @ anchor
        local[:, 0] = anchor @ local[:, 0]
        joints = (joints-origin) @ anchor.T
        motion = encode_body(local, joints[:, 0], rest)
        if np.linalg.norm(motion[:, 135:].reshape(128, 22, 3)*2-joints, axis=-1).max() > 2e-4:
            raise ValueError('Actual AMASS window disagrees with its shape/FK')
        camera = np.zeros((128, 9), np.float32)
        ca = np.zeros((128, 2), np.float32)
        trajectory = np.zeros((128, 22, 9), np.float32)
        meta = np.zeros((128, 22, 5), np.float32)
        meta[..., 4] = 1
        if mode == 'ego':
            camera[:, :3] = ((camera_position-origin) @ anchor.T)/2
            camera[:, 3:] = six_columns(anchor @ head_rotation)
            ca[:] = 1
        else:
            trajectory[..., :3] = np.where(mask[..., None], joints/2, 0.)
            meta[..., 0] = mask
            meta[..., 2] = mask
            # Positions only. Do not inject local/global GT rotations.
        sample = dict(camera=camera, camera_available=ca, camera_confidence=ca.copy(),
                      camera_context=np.array([int(mode=='ego'), int(mode=='exo'), 0, 1, 0, -1], np.float32),
                      camera_intrinsics=np.zeros((128, 4), np.float32),
                      camera_intrinsics_valid=np.zeros(128, np.float32),
                      trajectory=trajectory, observation_meta=meta,
                      observation_uncertainty=np.zeros((128, 22, 2), np.float32),
                      motion=motion, target_rest=rest.astype(np.float32),
                      target_betas=data['betas'].astype(np.float32))
        mask_observations(sample, frame_valid)
        ids = source_indices(row['raw_frames'], row['source_fps'])[1:][first:first+128]
        identity = dict(source=row['source'], family=row['family'], split=self.split,
                        corpus=row['corpus'], reflected=bool(vi), mode=mode, start=first,
                        source_first_frame=int(ids[0]), source_last_frame=int(ids[-1]),
                        variant_sha256=variant['sha256'], yaw=yaw_value,
                        origin=origin.tolist(), anchor=anchor.tolist(), synthetic_exo_pattern=pattern,
                        visible_fraction=float(meta[..., 0].mean()) if mode=='exo'
                            else float(frame_valid.mean()/22))
        identity.update(observation_gaps=outage, anchor_frame=anchor_frame)
        return sample, identity


def collate(samples, device='cpu'):
    return {key: torch.as_tensor(np.stack([s[key] for s in samples]), device=device)
            for key in samples[0]}


def inference_payload(batch):
    return {key: batch[key] for key in INPUT_KEYS}
