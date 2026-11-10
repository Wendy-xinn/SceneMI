"""Checked native-format EXO repair estimates for the three scene corpora.

Targets construct a declared synthetic frontend; targets are never inference
fields. Native shape spaces are decoded separately, never padded into SMPL-H.
This loader supplies no scene geometry to the scene-free repair experiment.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import six_columns
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.checked_scene_dataset import SceneSelection, DEFAULT_SELECTION
from scenemi.data.exo_corruption import correlated
from scenemi.data.exo_initialization import CONDITIONS, rotations_from_motion, fill_estimate_gaps
from scenemi.geometry.body import PARENTS, encode_body
from scenemi.geometry.native_shape import NativeShapeBank, GENDERS, source_assets
from scenemi.geometry.shape import load_shape_basis
from scenemi.training import ensure_bounded, now
from experiments.offline_camera_retrain_v1.data import anchor_rotation


class MissingInitialization(ValueError):
    """A repair window requires at least two actual tracked frames."""


def perturb_native(sample, identity, rng, policy, estimated_rest, actual_visible, *, condition):
    clean = condition == 'identity'; frames = 128
    local = rotations_from_motion(sample['motion'])
    root = sample['motion'][:, :3].astype(np.float64)*2+sample['target_rest'][0]
    pose_sigma = 0. if clean else float(rng.uniform(*policy['pose_sigma_rad']))
    root_sigma = 0. if clean else float(rng.uniform(*policy['root_sigma_m']))
    errors = correlated(rng, frames, (22, 3), pose_sigma, policy['pose_rho'])
    root_error = correlated(rng, frames, (3,), root_sigma, policy['root_rho'])
    root_error -= root_error[:1].copy()
    visible = np.ones((frames, 22), bool) if clean else actual_visible.copy()
    available = np.ones(frames, bool) if clean else visible.any(-1)
    reliability = np.full((frames, 22), 1-min(pose_sigma/.12, 1.))
    root_reliability = np.full(frames, 1-min(root_sigma/.06, 1.))
    interval = None
    if condition in ('occlusion', 'gap'):
        width = int(rng.integers(policy['gap_frames'][0], policy['gap_frames'][1]+1))
        placement = str(rng.choice(['prefix', 'interior', 'suffix']))
        start = 0 if placement == 'prefix' else frames-width if placement == 'suffix' else int(rng.integers(1, frames-width))
        interval = dict(start=start, end=start+width, placement=placement)
        if condition == 'occlusion':
            groups = ((16,18,20), (17,19,21), (1,4,7,10), (2,5,8,11))
            joints = groups[int(rng.integers(len(groups)))]; interval['local_rotation_joints'] = list(joints)
            visible[start:start+width, joints] = False
        else:
            available[start:start+width] = False; visible[start:start+width] = False
    if available.sum() < 2:
        raise MissingInitialization('Insufficient actual tracking for an estimated initializer')
    # Hidden joints retain a noisy frontend hypothesis when the person is
    # tracked; a fully missing person supplies no hidden motion payload.
    if not clean:
        sigma = float(rng.uniform(*policy['occluded_pose_sigma_rad']))
        errors += correlated(rng, frames, (22,3), sigma, policy['pose_rho'])*(~visible)[...,None]
    reliability[~visible] = 0.; reliability[~available] = 0.; root_reliability[~available] = 0.
    estimated = Rotation.from_rotvec(errors.reshape(-1,3)).as_matrix().reshape(frames,22,3,3)@local
    estimated_root = root+root_error
    if not available.all():
        estimated[~available] = np.nan; estimated_root[~available] = np.nan
        estimated,estimated_root = fill_estimate_gaps(estimated,estimated_root,available)
    initial = encode_body(estimated, estimated_root, estimated_rest)
    anchor_frame = int(np.flatnonzero(available)[0])
    origin = estimated_root[anchor_frame].copy(); origin[1] = 0.
    for name in ('motion',):
        sample[name][:,:3] -= origin/2; sample[name][:,135:] -= np.tile(origin/2,22)
    initial[:,:3] -= origin/2; initial[:,135:] -= np.tile(origin/2,22)
    sample['camera'][:,:3] -= origin/2
    global_rotations,global_reliability = [estimated[:,0]],[reliability[:,0]]
    for j in range(1,22):
        global_rotations.append(global_rotations[PARENTS[j]]@estimated[:,j])
        global_reliability.append(np.minimum(global_reliability[PARENTS[j]],reliability[:,j]))
    confidence = np.minimum(np.stack(global_reliability,1),root_reliability[:,None])
    confidence = np.clip(confidence,.05,1.)*available[:,None]
    sample['trajectory'][:] = 0.
    sample['trajectory'][...,:3] = initial[:,135:].reshape(frames,22,3)*available[:,None,None]
    sample['trajectory'][...,3:] = six_columns(np.stack(global_rotations,1))*available[:,None,None]
    sample['observation_meta'][:] = 0.
    sample['observation_meta'][...,0] = available[:,None]
    sample['observation_meta'][...,1] = available[:,None]
    sample['observation_meta'][...,2] = confidence; sample['observation_meta'][...,3] = confidence
    sample.update(initial_motion=initial, initial_rest=estimated_rest.astype(np.float32),
        initial_reliability=reliability.astype(np.float32), initial_root_reliability=root_reliability.astype(np.float32),
        estimate_available=available.astype(np.float32), joint_visible=visible.astype(np.float32))
    identity.update(frontend_condition=condition, frontend_estimate='synthetic_native_pose_and_native_shape',
        pose_sigma_rad=pose_sigma, root_sigma_m=root_sigma, estimate_gap_or_occlusion=interval,
        estimated_anchor_shift_m=origin.tolist(), anchor_frame=anchor_frame,
        origin=origin.tolist(), anchor=identity['anchor'],
        actual_visible_joint_fraction=float(actual_visible.mean()),
        reliability_from_realized_target_error=False, missing_estimates_interpolated_from_available_only=True)
    return sample,identity


class NativeRepairDataset:
    def __init__(self, *, split='train', selection=DEFAULT_SELECTION):
        if split not in ('train','validation'):
            raise ValueError('This adapter never opens test')
        check = json.loads((ROOT/'runs/data_inventory/CURRENT_SELECTION_CHECK.json').read_text())
        if check['state'] != 'passed' or check['selection_sha256'] != sha256(selection):
            raise ValueError('Current scene-data inventory check is stale')
        self.bundle = SceneSelection(selection); self.records = self.bundle.records('exo', split)
        self.split = split; self.groups = {key:defaultdict(list) for key in ('egobody','trumans','grail')}
        for i,row in enumerate(self.records):
            self.groups[row['dataset']][row['family']].append(i)
        self.families = {key:sorted(value) for key,value in self.groups.items()}
        paths = load_paths(ROOT/'configs/paths.local.yaml')
        self.common_pelvis = load_shape_basis(paths['smplh_models']/'neutral/model.npz')['rest_template'][0]
        self.bank = NativeShapeBank(self.common_pelvis).eval()
        self.asset_hashes = source_assets(paths)

    def sample(self, rng, policy, *, member=None, start=None, yaw=None, condition=None):
        if member is None:
            dataset = ('egobody','trumans','grail')[int(rng.integers(3))]
            family = self.families[dataset][int(rng.integers(len(self.families[dataset])))]
            choices = self.groups[dataset][family]; member = choices[int(rng.integers(len(choices)))]
        record = self.records[member]; meta = self.bundle.metadata(record,self.split)
        starts = record['starts']; first = int(rng.choice(starts)) if start is None else int(start)
        if first not in starts:
            raise ValueError('Native repair window leaves the current accepted motion ranges')
        span = slice(first,first+128)
        def array(name):
            return self.bundle.array(meta,name)
        native_rest = np.asarray(array('rest'),np.float64)
        source = np.asarray(array('motion_world')[span],np.float64)
        transforms = np.asarray(array('exo_sensor_to_world'),np.float64)
        if transforms.ndim == 2:
            transforms = np.broadcast_to(transforms,(128,4,4))
        else:
            transforms = transforms[span]
        if transforms.shape != (128,4,4):
            raise ValueError('EXO camera and native motion clocks differ')
        yaw = float(rng.uniform(-np.pi,np.pi)) if yaw is None else float(yaw)
        anchor = Rotation.from_rotvec([0,yaw,0]).as_matrix()@anchor_rotation(transforms[:,:3,:3])
        local = rotations_from_motion(source); local[:,0] = anchor@local[:,0]
        root = (source[:,:3]*2+native_rest[0])@anchor.T
        target_rest = native_rest-native_rest[:1]+self.common_pelvis
        target = encode_body(local,root,target_rest)
        expected = source[:,135:].reshape(128,22,3)*2@anchor.T
        fk_error = float(np.max(np.abs(target[:,135:].reshape(128,22,3)*2-expected)))
        if fk_error > 5e-5:
            raise ValueError('Native common-pelvis conversion changes target FK')
        condition = str(rng.choice(CONDITIONS,p=policy['probabilities'])) if condition is None else condition
        if condition not in CONDITIONS:
            raise ValueError('Unknown synthetic estimator condition')
        body = meta['native_body']; kind = 'grail_g1' if record['dataset']=='grail' else body['model']
        beta = np.asarray(body['betas'],np.float32)
        if beta.shape != (10,):
            raise ValueError('Use exact ten-dimensional native SMPL/SMPL-X parameters')
        sigma = 0. if condition == 'identity' or kind == 'grail_g1' else policy['beta_sigma']
        estimated_beta = beta+rng.normal(0,sigma,10)
        template = 0 if kind == 'grail_g1' else GENDERS.index(body['gender'])
        with torch.no_grad():
            decoded = self.bank(kind,torch.tensor(estimated_beta,dtype=torch.float32)[None],torch.tensor([template]))
            oracle = self.bank(kind,torch.tensor(beta)[None],torch.tensor([template]))
        if np.max(np.abs(oracle['rest'][0].numpy()-target_rest)) > 1e-5:
            raise ValueError('Configured native shape decoder differs from the source')
        intrinsics = np.asarray(array('exo_intrinsics'),np.float32)
        if intrinsics.ndim == 1:
            intrinsics = np.broadcast_to(intrinsics,(128,4)).copy()
        else:
            intrinsics = intrinsics[span].copy()
        optical = np.concatenate((transforms[:,:3,3]@anchor.T/2,
                                   six_columns(anchor@transforms[:,:3,:3])), -1).astype(np.float32)
        visible = np.asarray(array('exo_position_available')[span],bool)
        sample = dict(motion=target,target_rest=target_rest.astype(np.float32),target_betas=beta,
            camera=optical,camera_available=np.ones((128,2),np.float32),camera_confidence=np.ones((128,2),np.float32),
            camera_context=np.array([0,1,0,1,0,-1],np.float32),camera_intrinsics=intrinsics,
            camera_intrinsics_valid=np.ones(128,np.float32),trajectory=np.zeros((128,22,9),np.float32),
            observation_meta=np.zeros((128,22,5),np.float32),observation_uncertainty=np.zeros((128,22,2),np.float32))
        identity = dict(dataset=record['dataset'],corpus=record['dataset'],family=record['family'],
            member=member,path=record['path'],split=self.split,start=first,yaw=yaw,
            source_input_sha256=meta['input_sha256'],native_format=kind,synthetic_beta_sigma=sigma,
            synthetic_template_assumption='source_template_simulates_estimated_template',
            native_target_fk_error_m=fk_error,mode='exo',scene_input=False,
            # The perturbation stage records the final estimated origin; keep
            # the camera/world rotation here for scene-point conversion.
            anchor=anchor.tolist())
        return perturb_native(sample,identity,rng,policy,decoded['rest'][0].numpy(),visible,condition=condition)


def audit(config_path, out):
    ensure_bounded(); torch.set_num_threads(1)
    config = yaml.safe_load(config_path.read_text()); rows=[]; missing=[]
    for split in ('train','validation'):
        data = NativeRepairDataset(split=split)
        for dataset in data.groups:
            selected = sorted(i for choices in data.groups[dataset].values() for i in choices)[:2]
            for member in selected:
                for condition in CONDITIONS:
                    try:
                        sample,identity = data.sample(np.random.default_rng(777+member),config['frontend'],
                            member=member,start=int(data.records[member]['starts'][0]),yaw=0.,condition=condition)
                    except MissingInitialization as error:
                        missing.append(dict(split=split,dataset=dataset,member=member,condition=condition,error=str(error)))
                        continue
                    assert all(np.isfinite(value).all() for value in sample.values())
                    mask = sample['estimate_available']==0
                    assert not sample['trajectory'][mask].any() and not sample['observation_meta'][mask].any()
                    rows.append(identity)
    result = dict(state='source_examples_passed_model_preflight_pending',cpu_cuda_accepted=False,
        exo_payload_adapter_ready=False,test_arrays_opened=False,examples=len(rows),missing_initializer_examples=missing,
        selection_sha256=sha256(DEFAULT_SELECTION),examples_identity=rows,
        sources={str(Path(__file__).resolve()):sha256(Path(__file__)),str(config_path.resolve()):sha256(config_path)},
        native_assets_sha256=data.asset_hashes,completed_at_utc=now(),
        next='Full model CPU/CUDA and deterministic initializer-eligibility manifest required before training.')
    atomic_json(out,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/amass_exo_denoise.yaml')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/exo_preparation/three_corpus_repair/READINESS.json')
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    audit(args.config,args.output)


if __name__=='__main__':
    main()
