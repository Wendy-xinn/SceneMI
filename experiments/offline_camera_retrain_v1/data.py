"""Mixed TRUMANS/EgoBody adapter for SceneMI's occupancy, BPS and SMPL motion."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[2]
MOTION_ROOT = PROJECTS / 'diffusion-motion-inbetweening'
if str(MOTION_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTION_ROOT))
from experiments.offline_sequence_v1.data_loader import MixedSequences, anchor_rotation

SOURCE = MOTION_ROOT / 'experiments/offline_sequence_v1/data'
LENGTHS = (64, 128, 192)
GROUPS = ('trumans', 'camera_wearer', 'interactee', 'rich')
RICH_SOURCES = ('legacy5interp', 'native20_faceout_oct07')
VISIBLE_ROOTS = {
    'trumans': HERE / 'data/visible_trumans_5hz',
    'egobody': HERE / 'data/visible_egobody_5hz',
}
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
           16, 17, 18, 19)


def camera_body_anchors():
    # Inference-safe camera-centered volume: no true body joints or key poses.
    y_levels = (-1.55, -1.2, -.8, -.4, 0.)
    x_levels = (-.45, -.23, 0., .23, .45)
    z_levels = (-.25, 0., .3)
    grid = np.array([(x, y, z) for y in y_levels for x in x_levels for z in z_levels],
                    np.float32)
    ids = np.linspace(0, len(grid) - 1, 66).astype(int)
    return np.concatenate((np.zeros((1, 3), np.float32), grid[ids]), axis=0)


ANCHORS = camera_body_anchors()


def rotation6d(matrices):
    return np.concatenate((matrices[..., :, 0], matrices[..., :, 1]), axis=-1)


class OfflineSceneMIData:
    def __init__(self, split, seed=2026, skeleton_profile='archived',
                 rich_source='legacy5interp', trumans_scene_manifest=None, window_sampling='coverage', trumans_window_protocol='legacy_stable_v1', native_source_recovery=False):
        ready_path=SOURCE/'READY.json'
        if ready_path.is_file() and json.loads(ready_path.read_text()).get('reconstruction_protocol') and not native_source_recovery:
            raise ValueError('Recovered archive supports only native_v1 with complete exact20 temporal scenes; historical/legacy inputs unavailable')
        if skeleton_profile not in ('archived', 'trumans_male_v2', 'canonical_smpl'):
            raise ValueError(f'Unknown skeleton profile: {skeleton_profile}')
        if rich_source not in RICH_SOURCES:
            raise ValueError(f'Unknown RICH source: {rich_source}')
        if window_sampling not in ('coverage','duration'): raise ValueError('Unknown window sampler')
        self.window_sampling = window_sampling
        if trumans_window_protocol not in ('legacy_stable_v1','temporal_valid_v1'):raise ValueError('Unknown TRUMANS window protocol')
        if trumans_window_protocol=='temporal_valid_v1' and trumans_scene_manifest is None:raise ValueError('Dynamic-inclusive windows require full temporal scenes')
        self.trumans_window_protocol=trumans_window_protocol
        self.trumans_window_records=[]
        self.trumans_scene_bundles = {}
        self.trumans_exclusions = {}
        if trumans_scene_manifest is not None:
            manifest_path = Path(trumans_scene_manifest).resolve()
            rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
            for row in rows:
                if row.get('split') != split: continue
                name = row['sequence_id']
                if name in self.trumans_scene_bundles or name in self.trumans_exclusions: raise ValueError('Duplicate temporal scene sequence')
                if row.get('status')=='excluded':
                    if row.get('reason')!='missing_source_object_tracks':raise ValueError('Unsupported scene exclusion reason')
                    self.trumans_exclusions[name]=row
                    continue
                folder = Path(row['path'])
                if not folder.is_absolute(): folder = manifest_path.parent / folder
                self.trumans_scene_bundles[name] = folder
        self.skeleton_profile = skeleton_profile
        self.rich_source = rich_source
        self.trumans_rest = (np.load(HERE / 'data/body_templates/trumans_male_rest.npy')
                             if skeleton_profile == 'trumans_male_v2' else None)
        self.base = MixedSequences(split, trumans_dir=SOURCE / 'trumans_sequences',
                                   egobody_dir=SOURCE / 'egobody_sequences', seed=seed)
        if self.trumans_exclusions:
            original={r['clip_name']:r for r in map(json.loads,(PROJECTS/'TRUMANS/processed/scene_expert_v1/clips.jsonl').read_text().splitlines())}
            known={m['sequence_id'] for m,v in self.base.groups['trumans']}
            if set(self.trumans_exclusions)-known:raise ValueError('Scene exclusion references unknown sequence')
            for name in self.trumans_exclusions:
                if original[name]['object_tracks']['path']:raise ValueError('Cannot exclude a recording with source tracks as missing')
            self.base.groups['trumans']=[(m,v) for m,v in self.base.groups['trumans'] if m['sequence_id'] not in self.trumans_exclusions]
        if trumans_window_protocol=='temporal_valid_v1':
            from experiments.offline_camera_retrain_v1.temporal_windows import rebuild_trumans
            self.base.groups['trumans'],self.trumans_window_records=rebuild_trumans(split,self.trumans_scene_bundles,self.trumans_exclusions,SOURCE,HERE,PROJECTS,LENGTHS)
        self.prepared = {r['sequence_id']: r for group in self.base.groups.values()
                         for r, _ in group}
        for sequence_id, row in self.prepared.items():
            source_meta = json.loads((Path(row['folder']) / 'metadata.json').read_text())
            protocol = source_meta.get('observation_protocol', '').lower()
            if 'projected' not in protocol and 'rays' not in protocol:
                raise ValueError(f'{sequence_id}: scene source is not camera-visible geometry')
        pose_manifest = HERE / 'data/poses' / f'{split}.jsonl'
        self.pose_meta = {r['sequence_id']: r for r in
                          (json.loads(line) for line in pose_manifest.read_text().splitlines() if line)}
        if set(self.prepared) - set(self.pose_meta):
            raise ValueError(f'{split}: missing pose sequences')
        self.visible_meta = {}
        rich_split = 'train' if split == 'train' else 'val'
        rich_name = (f'rich_{rich_split}_smpl20' if rich_source == 'legacy5interp'
                     else f'rich_{rich_split}_smpl_native20_faceout_oct07')
        self.rich_root = HERE / 'data/scene_visibility_v2_oct05' / rich_name
        self.rich_members = []
        if self.rich_root.is_dir():
            for folder in sorted(p for p in self.rich_root.iterdir() if p.is_dir()):
                meta_path = folder / 'metadata.json'
                if not meta_path.is_file(): continue
                meta = json.loads(meta_path.read_text())
                frames = int(np.load(folder/'joints_scenemi_yup.npy', mmap_mode='r').shape[0])
                valid = {L: list(range(0, frames-L+1)) if frames >= L else [] for L in LENGTHS}
                if any(valid.values()): self.rich_members.append((meta, valid))
        for dataset, root in VISIBLE_ROOTS.items():
            manifest = root / f'{split}.jsonl'
            if not manifest.is_file():
                raise FileNotFoundError(f'Missing 5 Hz ego-visible manifest: {manifest}')
            for row in (json.loads(line) for line in manifest.read_text().splitlines() if line):
                protocol = row.get('protocol', '').lower()
                if (row.get('sample_stride') != 4 or row.get('candidate_count') != 65536
                        or row.get('query_radius_m') != 6.5 or 'project' not in protocol):
                    raise ValueError(f"{row.get('sequence_id')}: invalid 5 Hz ego-visible map metadata")
                self.visible_meta[row['sequence_id']] = row
        missing_visible = set(self.prepared) - set(self.visible_meta)
        if missing_visible:
            raise ValueError(f'{split}: missing 5 Hz ego-visible maps: {len(missing_visible)}')
        if trumans_scene_manifest is not None:
            required = {m['sequence_id'] for m,v in self.base.groups['trumans']}
            missing = required - set(self.trumans_scene_bundles)
            if missing: raise ValueError(f'{split}: temporal TRUMANS bundles missing {len(missing)} sequences; no frozen fallback')
            for m,v in self.base.groups['trumans']:
                folder = self.trumans_scene_bundles[m['sequence_id']]
                meta = json.loads((folder/'metadata.json').read_text())
                ids = np.load(folder/'source_frame_ids.npy')
                expected = m.get('source_first_frame',0) + np.arange(m['frames_20fps'])*1.5
                if (meta.get('occlusion_version') != 2 or meta.get('sequence_id') != m['sequence_id']
                        or meta.get('output_fps') != 20 or len(ids)!=len(expected)
                        or not np.allclose(ids,expected,atol=1e-5)):
                    raise ValueError(f'{m["sequence_id"]}: temporal bundle must cover full prepared recording at exact source times')
                valid_path=folder/'observation_valid.npy'
                observed=np.load(valid_path) if valid_path.exists() else np.ones(len(ids),bool)
                if observed.shape!=(len(ids),) or observed.dtype!=np.bool_:raise ValueError('Invalid observation mask')
                if int((~observed).sum())!=meta.get('invalid_observation_count',0):raise ValueError('Missing invalid observation mask')
                for length in LENGTHS:
                    v[length]=[(offset,source) for offset,source in v[length] if observed[offset:offset+length].all()]
        self.pose_cache = {}
        self.split = split
        self.window_lists = {}
        self.window_cursors = {}
        for group in ('trumans', 'camera_wearer', 'interactee'):
            for length in LENGTHS:
                members=[(mi,v) for mi,(_,v) in enumerate(self.base.groups[group]) if v[length]]
                self.window_lists[(group,length)] = [(fi,si) for fi,(_,valid) in enumerate(members) for si in range(len(valid[length]))]
                self.window_cursors[(group,length)] = 0
        for length in LENGTHS:
            members=[(mi,v) for mi,(_,v) in enumerate(self.rich_members) if v[length]]
            self.window_lists[('rich',length)] = [(fi,si) for fi,(_,valid) in enumerate(members) for si in range(len(valid[length]))]
            self.window_cursors[('rich',length)] = 0
        for windows in self.window_lists.values():
            self.base.rng.shuffle(windows)

    def _pose(self, sequence_id):
        if sequence_id not in self.pose_cache:
            meta = self.pose_meta[sequence_id]
            folder = Path(meta['folder'])
            self.pose_cache[sequence_id] = (np.load(folder / 'pose_axis_angle_world.npy', mmap_mode='r'),
                                             np.load(folder / 'translation_world.npy', mmap_mode='r'),
                                             np.load(folder / 'rest_joints.npy', mmap_mode='r'))
            if len(self.pose_cache) > 8:
                self.pose_cache.pop(next(iter(self.pose_cache)))
        return self.pose_cache[sequence_id]

    def sample(self, length, group, *, sequence_index=None, start_index=None, scene_bundle=None):
        if sequence_index is None and start_index is None and self.window_sampling == 'duration':
            from experiments.offline_camera_retrain_v1.training_contract_v2 import covered_frames
            source = self.rich_members if group == 'rich' else self.base.groups[group]
            members = [(m,v) for m,v in source if v[length]]
            weights = np.array([covered_frames(v[length] if group=='rich' else [s for s,source_id in v[length]],length) for m,v in members],float)
            sequence_index = int(self.base.rng.choice(len(members),p=weights/weights.sum()))
            start_index = int(self.base.rng.integers(len(members[sequence_index][1][length])))
        if sequence_index is None and start_index is None:
            windows=self.window_lists.get((group,length), [])
            if windows:
                cursor=self.window_cursors[(group,length)]
                sequence_index,start_index=windows[cursor]
                next_cursor=cursor+1
                if next_cursor==len(windows):
                    self.base.rng.shuffle(windows)
                    next_cursor=0
                self.window_cursors[(group,length)]=next_cursor
        if group == 'rich':
            return self._sample_rich(length, sequence_index=sequence_index,
                                     start_index=start_index)
        basic, identity = self.base.sample(length, group=group,
                                           sequence_index=sequence_index, start_index=start_index)
        name = identity['sequence_id']
        if scene_bundle is None and group == 'trumans' and self.trumans_scene_bundles:
            scene_bundle = self.trumans_scene_bundles[name]
        meta = self.prepared[name]
        pose, translation, rest = self._pose(name)
        corrected_trumans = self.trumans_rest is not None and self.pose_meta[name]['dataset'] == 'trumans'
        if corrected_trumans:
            rest = self.trumans_rest
        source_start = identity['source_start_30fps']
        first_source = meta.get('source_first_frame', 0)
        offset = (source_start - first_source) / 1.5
        if abs(offset - round(offset)) > 1e-5:
            raise ValueError(f'{name}: 30/20 Hz index mismatch')
        offset = int(round(offset))
        if offset + length > len(pose):
            raise ValueError(f'{name}: pose sequence too short')
        joints_full, camera_full, rotation_full, _ = self.base._load(meta)
        visible_path = Path(self.visible_meta[name]['path'])
        if not visible_path.is_file():
            raise FileNotFoundError(f'Missing 5 Hz ego-visible map: {visible_path}')
        static = np.load(visible_path, mmap_mode='r')
        camera_position = np.asarray(camera_full[offset:offset + length])
        camera_rotation = np.asarray(rotation_full[offset:offset + length])
        temporal = None
        cached_scene = None
        if scene_bundle is not None:
            # Explicit versioned opt-in: never silently alter historical 55k
            # inputs and never fall back to a frozen object for missing frames.
            folder = Path(scene_bundle)
            bundle_meta = json.loads((folder/'metadata.json').read_text())
            if (bundle_meta.get('occlusion_version') != 2 or
                    bundle_meta['sequence_id'] != name or bundle_meta.get('output_fps') != 20):
                raise ValueError('Wrong sequence, rate or version for temporal scene bundle')
            frame_ids = np.load(folder/'source_frame_ids.npy')
            queries = source_start + np.arange(length)*1.5
            ix = np.searchsorted(frame_ids, queries)
            if np.any(ix>=len(frame_ids)) or not np.allclose(frame_ids[ix], queries, atol=1e-5):
                raise ValueError('Temporal scene bundle does not cover requested source frames')
            if bundle_meta.get('memory_protocol')=='causal20-first-world-voxel25mm-current-dynamic-v1':
                from experiments.offline_camera_retrain_v1.causal_scene import temporal_scene_inputs
                cached_scene,camera_position,camera_rotation,scene_identity=temporal_scene_inputs(folder,queries)
                temporal='cached';scene=np.empty((0,3),np.float32);identity.update(scene_identity)
            else:
                all_points = np.load(folder/'visible_frame_points_scenemi_yup.npy',mmap_mode='r')
                all_owner = np.load(folder/'visible_frame_owner.npy',mmap_mode='r')
                points = all_points[ix]
                owner = all_owner[ix]
                # Preserve observations BEFORE this target window from the same recording.
                history_points = np.asarray(all_points[:ix[0]][all_owner[:ix[0]]==0])
                if len(history_points):
                    _,unique = np.unique(np.floor(history_points/.025).astype(np.int64),axis=0,return_index=True)
                    history_points = history_points[np.sort(unique)]
                camera_position = np.load(folder/'camera_position_scenemi_yup.npy',mmap_mode='r')[ix]
                camera_rotation = np.load(folder/'camera_rotation_scenemi_yup.npy',mmap_mode='r')[ix]
                static_frames = [p[o==0] for p,o in zip(points,owner)]
                dynamic_frames = [p[(o>0)&(o<100)] for p,o in zip(points,owner)]
                temporal = (static_frames,dynamic_frames,history_points)
                # Global branch contains ONLY static surfaces (offline window
                # union). Temporal dynamic surfaces belong exclusively to BPS.
                # Fixed global conditioning contains only surfaces available at target start.
                scene = np.concatenate((history_points,static_frames[0])).astype(np.float32)
                identity['scene_protocol'] = 'v3: target-start static occupancy; recording-history static/current dynamic BPS'
                identity['scene_bundle'] = str(folder)
                identity['dynamic_object_points'] = int(((owner>0)&(owner<100)).sum())
        else:
            dynamic = self.base._object_points(meta, source_start, camera_position, camera_rotation)
            scene = np.concatenate((np.asarray(static), dynamic)).astype(np.float32)
        if temporal is None and len(scene) < 32:
            raise ValueError(f'{name}: insufficient scene points')
        anchor = anchor_rotation(camera_rotation)
        origin = camera_position[0]
        if temporal is not None:
            camera_r = np.einsum('ij,tjk->tik',anchor,camera_rotation)
            basic['camera'] = np.concatenate(((camera_position-origin)@anchor.T/2.,
                                              camera_r[:,:,0],camera_r[:,:,1]),axis=-1).astype(np.float32)
        local_pose = np.asarray(pose[offset:offset + length]).copy()
        root_matrix = Rotation.from_rotvec(local_pose[:, 0]).as_matrix()
        root_matrix = anchor[None] @ root_matrix
        body_matrix = Rotation.from_rotvec(local_pose[:, 1:].reshape(-1, 3)).as_matrix()
        body_matrix = body_matrix.reshape(length, 21, 3, 3)
        all_rotations = np.concatenate((root_matrix[:, None], body_matrix), axis=1)
        global_rotations = [all_rotations[:, 0]]
        for joint in range(1, 22):
            global_rotations.append(global_rotations[PARENTS[joint]] @ all_rotations[:, joint])
        global_rotations = np.stack(global_rotations, axis=1)
        # Predict residual 6D rotation; SceneMI's zero-initialized output then
        # represents valid identity rotations instead of a degenerate matrix.
        identity6 = np.asarray([1., 0., 0., 0., 1., 0.], np.float32)
        pose6d = (rotation6d(all_rotations) - identity6).reshape(length, 132)
        # SMPL's transl is relative to its fixed template root joint. Rotating
        # the coordinate frame requires compensating that rest-root offset.
        transl = ((np.asarray(translation[offset:offset + length]) + rest[0] - origin)
                  @ anchor.T - rest[0]) / 2.
        joint_positions = (np.asarray(joints_full[offset:offset + length]) - origin) @ anchor.T / 2.
        if corrected_trumans:
            # The released joint archive is male SMPL-X, despite empty gender
            # in the fit files. Also remove one source sequence's rigid fit
            # translation bias. This is a clean TRAINING TARGET, not an input.
            transl = joint_positions[:, 0] - rest[0] / 2.
        motion = np.concatenate((transl, pose6d, joint_positions.reshape(length, 66)), axis=-1)
        # Generic per-joint control values: position plus global orientation.
        # The head entry uses the actually available camera trajectory, while
        # other entries are candidate supervised controls selected by a mask.
        trajectory = np.concatenate((joint_positions, rotation6d(global_rotations)), axis=-1)
        trajectory[:, 15] = basic['camera']

        # SceneMI global branch: coarse occupancy of the video-visible map.
        # The fixed extent gives consistent metric meaning across lengths.
        scene_anchor = (scene - origin) @ anchor.T
        camera_anchor = (camera_position - origin) @ anchor.T
        center = np.array([camera_anchor[:, 0].mean(), camera_anchor[:, 1].mean() - .7,
                           camera_anchor[:, 2].mean()])
        if temporal is not None:center=np.array([0.,-.7,0.])
        index = np.floor((scene_anchor - center + [6.4, 2.4, 6.4]) /
                         [.26666667, .2, .26666667]).astype(np.int32)
        valid = ((index >= 0) & (index < [48, 24, 48])).all(-1)
        # Only observed surfaces are asserted. Zero means unknown, not free;
        # the current point archive does not retain complete ray-carved space.
        voxel = np.zeros((24, 48, 48), np.float32)
        index = index[valid]
        voxel[index[:, 1], index[:, 0], index[:, 2]] = 1.

        # SceneMI local branch: ordered BPS offsets around a camera-centered
        # body volume, calculated solely from camera poses and observed points.
        tree = cKDTree(scene if len(scene) else np.zeros((1,3),np.float32))
        anchor_world = (ANCHORS[None] @ camera_rotation.transpose(0, 2, 1)
                        + camera_position[:, None])
        nearest = (scene if len(scene) else np.zeros((1,3),np.float32))[tree.query(anchor_world.reshape(-1, 3), k=1)[1]].reshape(length, 67, 3)
        bps_m = np.einsum('tpi,tij->tpj', nearest - anchor_world, camera_rotation)
        bps_norm = np.linalg.norm(bps_m, axis=-1, keepdims=True)
        # Match SceneMI's 1 m BPS truncation so distant scan gaps cannot
        # dominate the frame conditioner; orientation still signals direction.
        bps = bps_m / np.maximum(bps_norm, 1.) / 2.
        bps_valid = np.ones((length,67),bool)
        if cached_scene is not None:
            bps,bps_valid,voxel=cached_scene['bps'],cached_scene['bps_valid'],cached_scene['occupancy']
        elif temporal is not None:
            from experiments.offline_camera_retrain_v1.scene_visibility_v2 import temporal_bps
            bps,bps_valid = temporal_bps(temporal[0], temporal[1], camera_position, camera_rotation, ANCHORS, return_valid=True, initial_static=temporal[2])
        result = dict(motion=motion.astype(np.float32), joints=joint_positions.astype(np.float32),
                      trajectory=trajectory.astype(np.float32), camera=basic['camera'],
                      bps=bps.astype(np.float32),
                      occupancy=voxel, rest=np.asarray(rest, np.float32).copy())
        result['bps_valid'] = bps_valid
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError(f'{name}: nonfinite SceneMI sample')
        return result, identity

    def _sample_rich(self, length, *, sequence_index=None, start_index=None):
        if not self.rich_members: raise ValueError('RICH canonical bundles are missing')
        members = [(m,v) for m,v in self.rich_members if v[length]]
        if not members: raise ValueError(f'RICH has no {length}-frame windows')
        if sequence_index is None: sequence_index=int(self.base.rng.integers(len(members)))
        meta, valid = members[sequence_index]
        if start_index is None: start_index=int(self.base.rng.integers(len(valid[length])))
        start=valid[length][start_index]; folder=Path(meta['path']) if meta.get('path') else self.rich_root/meta['sequence_id']
        if not folder.is_dir(): folder=self.rich_root/meta['sequence_id']
        joints=np.asarray(np.load(folder/'joints_scenemi_yup.npy',mmap_mode='r')[start:start+length])
        camera_position=np.asarray(np.load(folder/'camera_position_scenemi_yup.npy',mmap_mode='r')[start:start+length])
        camera_rotation=np.asarray(np.load(folder/'camera_rotation_scenemi_yup.npy',mmap_mode='r')[start:start+length])
        pose=np.asarray(np.load(folder/'pose_axis_angle_scenemi_yup.npy',mmap_mode='r')[start:start+length])
        translation=np.asarray(np.load(folder/'translation_scenemi_yup.npy',mmap_mode='r')[start:start+length])
        rest=np.asarray(np.load(folder/'rest_joints_scenemi_yup.npy',mmap_mode='r'))
        points=np.asarray(np.load(folder/'visible_static_points_scenemi_yup.npy',mmap_mode='r'))
        if len(points)<32: raise ValueError(f'{meta["sequence_id"]}: insufficient RICH scene')
        anchor=anchor_rotation(camera_rotation); origin=camera_position[0]
        local_pose=pose.copy(); root=Rotation.from_rotvec(local_pose[:,0]).as_matrix(); root=anchor[None]@root
        body=Rotation.from_rotvec(local_pose[:,1:].reshape(-1,3)).as_matrix().reshape(length,21,3,3)
        allr=np.concatenate((root[:,None],body),axis=1); globals_=[allr[:,0]]
        for j in range(1,22): globals_.append(globals_[PARENTS[j]]@allr[:,j])
        globals_=np.stack(globals_,axis=1); identity6=np.asarray([1,0,0,0,1,0],np.float32)
        pose6=(rotation6d(allr)-identity6).reshape(length,132)
        transl=((translation+rest[0]-origin)@anchor.T-rest[0])/2.
        joint_pos=(joints-origin)@anchor.T/2.; motion=np.concatenate((transl,pose6,joint_pos.reshape(length,66)),axis=-1)
        traj=np.concatenate((joint_pos,rotation6d(globals_)),axis=-1); cam=np.concatenate(((camera_position-origin)@anchor.T/2.,np.einsum('ij,tjk->tik',anchor,camera_rotation)[:,:,0],np.einsum('ij,tjk->tik',anchor,camera_rotation)[:,:,1]),axis=-1);traj[:,15]=cam
        scene=(points-origin)@anchor.T; center=np.array([((camera_position-origin)@anchor.T)[:,0].mean(),((camera_position-origin)@anchor.T)[:,1].mean()-.7,((camera_position-origin)@anchor.T)[:,2].mean()]); idx=np.floor((scene-center+[6.4,2.4,6.4])/[.26666667,.2,.26666667]).astype(np.int32); validv=((idx>=0)&(idx<[48,24,48])).all(-1); voxel=np.zeros((24,48,48),np.float32); idx=idx[validv]; voxel[idx[:,1],idx[:,0],idx[:,2]]=1.
        tree=cKDTree(points); aw=ANCHORS[None]@camera_rotation.transpose(0,2,1)+camera_position[:,None]; near=points[tree.query(aw.reshape(-1,3),k=1)[1]].reshape(length,67,3); bps_m=np.einsum('tpi,tij->tpj',near-aw,camera_rotation); bps=bps_m/np.maximum(np.linalg.norm(bps_m,axis=-1,keepdims=True),1.)/2.
        result=dict(motion=motion.astype(np.float32),joints=joint_pos.astype(np.float32),trajectory=traj.astype(np.float32),camera=cam.astype(np.float32),bps=bps.astype(np.float32),occupancy=voxel,rest=rest.astype(np.float32),bps_valid=np.ones((length,67),bool))
        identity=dict(dataset='rich',group='rich',sequence_id=meta['sequence_id'],source_start_30fps=float(np.load(folder/'source_frame_ids.npy')[start]))
        return result, identity

    def summary(self):
        result=self.base.summary()
        result['rich']={str(length): {'sequences': sum(bool(v[length]) for _,v in self.rich_members), 'starts': len(self.window_lists.get(('rich',length), []))} for length in LENGTHS}
        return result

    def sampler_state(self):
        return dict(window_lists=self.window_lists, window_cursors=self.window_cursors,
                    rng_state=self.base.rng.bit_generator.state)

    def load_sampler_state(self, state):
        if set(state['window_lists']) != set(self.window_lists):
            raise ValueError('Sampler buckets changed since checkpoint')
        for key, windows in state['window_lists'].items():
            if len(windows) != len(self.window_lists[key]):
                raise ValueError(f'Sampler bucket size changed: {key}')
        self.window_lists = state['window_lists']
        self.window_cursors = state['window_cursors']
        self.base.rng.bit_generator.state = state['rng_state']


def collate(samples):
    return {key: torch.from_numpy(np.stack([sample[key] for sample in samples]))
            for key in samples[0]}
