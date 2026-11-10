"""Audit GRAIL source alignment and inspect the exact scene payload.

Read-only source diagnostics. Reuses MotionViewer; does not edit a running
training closure, source arrays, split membership, or quality thresholds.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.grail_augmentation import observed_prefix
from scenemi.data.grail_contact import ContactBody, load_source
from scenemi.data.grail_quality import quantiles, contact_probabilities
from scenemi.data.grail_visibility import index_ranges
from scenemi.data.grail_support import sole_measure
from scenemi.data.scene_dataset import bounded_frame_points
from experiments.offline_camera_retrain_v1.data import anchor_rotation
from experiments.offline_camera_retrain_v1.ops.grail_source import (
    WORLD_ROTATION, load_pickle, usd_mesh,
)
from experiments.offline_camera_retrain_v1.ops.motion_viewer import MotionViewer
from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def records(manifest, include_chairs=True):
    """Preserve the supplied review selection, append explicit train controls."""
    result = []
    quality = ROOT / 'runs/grail_heldout_quality_v1/full/validation'
    lookup = {r['path']: (r, status) for filename, status in
              [('selection.jsonl', 'retained'), ('excluded.jsonl', 'excluded')]
              for r in read_rows(quality / filename)}
    for case in json.loads(Path(manifest).read_text())['cases']:
        with np.load(case['models'][0]['case']['cache'], allow_pickle=False) as cache:
            identity = json.loads(str(cache['metadata_json']))['identity']
        row, status = lookup[identity['path']]
        result.append(dict(row=row, status=status, start=int(case['start']),
                           label=case.get('display_label', case['label'])))
    if include_chairs:
        selection = ROOT / 'runs/grail_ego_quality_v2/chair400/SELECTION.jsonl'
        chairs = sorted(read_rows(selection), key=lambda r: r['sequence_id'])
        for i in np.linspace(0, len(chairs)-1, 10, dtype=int):
            row = chairs[i]
            result.append(dict(row=row, status='train_synthetic_chair',
                start=int(row['eligible_start_ranges'][0][0]),
                label=f'T{len(result)-99:02d} · train 补椅对照 · {row["sequence_id"][:42]}'))
    return result


def prototypes():
    p = ROOT / 'runs/grail_ego_quality_v2/chair400/conditions/PROTOTYPES.json'
    return {r['path']: r for r in json.loads(p.read_text())['cases']}


def scene_payload(meta, start, cfg, artificial=None):
    """Reproduce LocalSceneDataset point limits and fixed camera anchor."""
    span = slice(start, start+128)
    cp = np.asarray(readonly_array(meta, 'camera_position_scenemi_yup')[span], np.float64)
    cr = np.asarray(readonly_array(meta, 'camera_rotation_scenemi_yup')[span], np.float64)
    origin = cp[0].copy(); origin[1] = 0.
    anchor = anchor_rotation(cr)
    raw = readonly_array(meta, 'visible_frame_points_scenemi_yup')
    owners = readonly_array(meta, 'visible_frame_owner')
    history = observed_prefix(raw, owners, start, cfg['data']['history_voxel_m'])
    actual_history = history.copy()
    if artificial is None:
        artificial = np.empty((0, 3), np.float32)
    history = np.concatenate((history, artificial))
    history = ((history-origin) @ anchor.T).astype(np.float32)
    if len(history) > cfg['local']['history_points']:
        history = history[np.linspace(0, len(history)-1, cfg['local']['history_points']).astype(int)]
    frame, owner = bounded_frame_points(raw[span], owners[span], cfg['local']['frame_points'])
    frame = ((frame-origin) @ anchor.T).astype(np.float32)
    frame[owner < 0] = 0
    return dict(history=history, actual_history=((actual_history-origin) @ anchor.T).astype(np.float32),
                synthetic=((artificial-origin) @ anchor.T).astype(np.float32),
                frame=frame, owner=owner, origin=origin, anchor=anchor,
                camera_positions=((cp-origin) @ anchor.T).astype(np.float32),
                camera_rotations=np.einsum('ij,tjk->tik', anchor, cr).astype(np.float32))


def artificial_for(meta, source):
    record = source.get(meta['path']) if meta['split'] == 'train' else None
    if record is None:
        return None
    if sha256(record['condition_file']) != record['condition_sha256']:
        raise ValueError('Synthetic chair identity changed')
    with np.load(record['condition_file'], allow_pickle=False) as data:
        return data['synthetic_chair_points_m'].copy()


def recording_payload(meta, start, cfg, artificial=None):
    """Explicit offline map, separate from the historical causal audit above."""
    from scenemi.data.recording_scene import load_policy, observed_recording, bounded_scene_points
    policy = load_policy()
    result = scene_payload(meta, start, cfg, artificial)
    world = observed_recording(readonly_array(meta, 'visible_frame_points_scenemi_yup'),
        readonly_array(meta, 'visible_frame_owner'), readonly_array(meta, 'observation_valid'),
        policy['voxel_m'])
    if artificial is None:
        artificial = np.empty((0, 3), np.float32)
    combined = np.concatenate((world, artificial))
    local = ((combined-result['origin']) @ result['anchor'].T).astype(np.float32)
    result.update(history=bounded_scene_points(local, policy['local_point_limit']),
        actual_history=((world-result['origin']) @ result['anchor'].T).astype(np.float32),
        recording_frames=int(meta['frames']), observed_static_points=len(world),
        points_before_limit=len(combined), map_protocol=policy['version'])
    return result


def raw_params(human, output_ids):
    """No interpolation: output indices divisible by 5 map to raw indices /5*4."""
    ids = np.asarray(output_ids)//5*4
    pose = np.asarray(human['poses'], np.float32)[ids]
    p = dict(global_orient=pose[:, :3], body_pose=pose[:, 3:66],
             transl=np.asarray(human['trans'], np.float32)[ids],
             left_hand_pose=np.asarray(human.get('left_hand_pose', human['poses'][:, 75:120]), np.float32)[ids],
             right_hand_pose=np.asarray(human.get('right_hand_pose', human['poses'][:, 120:165]), np.float32)[ids])
    p.update({k: np.zeros((len(ids), n), np.float32) for k,n in
              [('betas',10),('expression',10),('jaw_pose',3),('leye_pose',3),('reye_pose',3)]})
    return p


def audit(manifest, output, config):
    from scipy.spatial import cKDTree
    import open3d as o3d
    from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene
    cfg = yaml.safe_load(Path(config).read_text())
    contact = json.loads((ROOT/'configs/grail_ego_contact.json').read_text())
    support = dict(contact, **json.loads((ROOT/'configs/grail_ego_support_v2.json').read_text()))
    body = ContactBody(contact)
    proto = prototypes()
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(records(manifest)):
        row, start = spec['row'], spec['start']
        meta, params, rotation, offset = load_source(row)
        if meta['split'] not in ('train', 'validation'):
            raise ValueError('No independent test in this audit')
        for key in ('recon','object_usd'):
            if sha256(meta['source_files'][key]) != meta['source_hashes'][key]:
                raise ValueError('Original source identity changed: '+key)
        raw = load_pickle(meta['source_files']['recon'])
        uv, uf, _ = usd_mesh(meta['source_files']['object_usd'])
        obj = raw['obj_data']
        original = ((uv*np.asarray(obj['obj_scale']).reshape(-1)) @ np.asarray(obj['obj_R'])[0].T
                    +np.asarray(obj['obj_t'])[0]) @ WORLD_ROTATION.T
        ov, of = readonly_array(meta,'scene_object_vertices'), readonly_array(meta,'scene_object_faces')
        object_error = float(np.abs(original-ov).max())
        if not np.array_equal(uf, of) or object_error > 1e-5:
            raise ValueError('Original object USD/pose differs from export')
        span = slice(start,start+128)
        window = {k:v[span] for k,v in params.items()}
        vertices, joints = body.forward(window, rotation, offset)
        expected = readonly_array(meta,'motion_world')[span,-66:].reshape(128,22,3)*2
        joint_error = float(np.abs(joints-expected).max())
        # Check every exact 24/30 Hz shared timestamp in this window.
        shared = np.arange(((start+4)//5)*5, start+128, 5)
        direct, direct_joints = body.forward(raw_params(raw['human_data'],shared), WORLD_ROTATION, np.zeros(3))
        raw_error = float(np.abs(direct-vertices[shared-start]).max())
        raw_joint_error = float(np.abs(direct_joints-joints[shared-start]).max())
        payload = scene_payload(meta,start,cfg,artificial_for(meta,proto))
        scene_v = np.concatenate((ov, readonly_array(meta,'scene_ground_vertices')))
        scene_f = np.concatenate((of, readonly_array(meta,'scene_ground_faces')+len(ov)))
        scene = static_ray_scene(scene_v,scene_f)
        # First-hit point geometry must lie on the same saved scene triangles.
        cloud = readonly_array(meta,'visible_frame_points_scenemi_yup')[span]
        owner = readonly_array(meta,'visible_frame_owner')[span]
        actual = np.asarray(cloud[owner==0],np.float32)
        point_ids = np.linspace(0,len(actual)-1,min(1024,len(actual)),dtype=int)
        dist = scene.compute_distance(o3d.core.Tensor(actual[point_ids]),nthreads=1).numpy()
        measured = sole_measure(vertices,body.feet,scene_v,scene_f,support)
        probability = contact_probabilities(raw['human_data']['foot_contact_probs'],meta['frames'],meta['source_fps'],meta['fps'])[span]
        lowest = measured['sole_gap_q10_m'].min(-1)
        joint_height = np.minimum(joints[:,10,1],joints[:,11,1])
        skin_height = np.minimum(vertices[:,body.feet[0],1].min(-1),vertices[:,body.feet[1],1].min(-1))
        current_counts = (payload['owner']>=0).sum(-1)
        # Percentage of current geometry farther than 5cm from start history.
        # This is expected to grow later in the window, not evidence of lost input.
        history = payload['actual_history']
        current = payload['frame']
        unseen = []
        if len(history):
            tree=cKDTree(history)
            for t in (0,64,127):
                v=current[t,payload['owner'][t]>=0]
                unseen.append(float((tree.query(v)[0]>.05).mean()) if len(v) else 0.)
        report=dict(sequence_id=row['sequence_id'],category=meta['category'],split=meta['split'],
            status=spec['status'],start=start,source_metadata_sha256=row['metadata_sha256'],
            original_object_max_error_m=object_error,exported_joint_max_error_m=joint_error,
            direct_raw_skin_max_error_m=raw_error,direct_raw_joint_max_error_m=raw_joint_error,
            raw_shared_timepoints=len(shared),
            sampled_scene_point_triangle_distance_m=quantiles(dist),
            actual_prefix_points=len(payload['actual_history']),model_history_points=len(payload['history']),
            synthetic_chair_points=len(payload['synthetic']),current_input_counts=quantiles(current_counts),
            current_outside_5cm_of_start_history_fraction_at_0_64_127=unseen,
            joint_minus_lowest_foot_skin_height_m=quantiles(joint_height-skin_height),
            lower_of_two_soles_gap_q10_m=quantiles(lowest),
            both_soles_gap_over_3cm_frames=int((lowest>.03).sum()),
            both_soles_gap_over_5cm_frames=int((lowest>.05).sum()),
            both_soles_gap_over_3cm_ranges=index_ranges(lowest>.03),
            both_soles_gap_over_5cm_ranges=index_ranges(lowest>.05),
            exact_raw_both_soles_gap_over_3cm_frames=int((lowest[shared-start]>.03).sum()),
            source_estimated_contact_gap_q10_m=quantiles(measured['sole_gap_q10_m'][probability>.8]),
            support_fraction=quantiles(measured['sole_supported_fraction']),
            quality_bad_frame_ranges=row.get('bad_frame_ranges',[]),
            seated_support_record=row.get('contact',{}).get('seat_gap_during_stable_pose_m'))
        if max(joint_error,raw_error,raw_joint_error,float(dist.max()))>5e-5:
            raise ValueError('Numerical alignment audit failed: '+row['sequence_id'])
        rows.append(report)
        atomic_json(output/'RECORDS.json',rows)
        if (index+1)%10==0:
            print(json.dumps(dict(checked=index+1,raw_error=raw_error,object_error=object_error)),flush=True)
    summary=dict(state='passed',protocol='grail-source-input-alignment-audit-v1',
        test_arrays_opened=False,model_inference=False,data_modified=False,
        manifest=str(Path(manifest).resolve()),manifest_sha256=sha256(manifest),
        config=str(Path(config).resolve()),config_sha256=sha256(config),
        implementation_sha256=sha256(__file__),cases=len(rows),
        max_original_object_error_m=max(r['original_object_max_error_m'] for r in rows),
        max_raw_skin_error_m=max(r['direct_raw_skin_max_error_m'] for r in rows),
        max_exported_joint_error_m=max(r['exported_joint_max_error_m'] for r in rows),
        max_sampled_point_triangle_distance_m=max(r['sampled_scene_point_triangle_distance_m']['max'] for r in rows),
        joint_minus_skin_height_m=quantiles([r['joint_minus_lowest_foot_skin_height_m']['median'] for r in rows]),
        retained_terrain_with_any_both_soles_gap_over_3cm=[r['sequence_id'] for r in rows if r['status']=='retained' and r['category']!='sitting' and r['both_soles_gap_over_3cm_frames']],
        retained_terrain_with_any_both_soles_gap_over_5cm=[r['sequence_id'] for r in rows if r['status']=='retained' and r['category']!='sitting' and r['both_soles_gap_over_5cm_frames']],
        input_semantics='history prefix through window start, plus actual current-frame points; train-only synthetic chairs explicitly added',
        rows=rows)
    atomic_json(output/'RESULT.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False),flush=True)
    return summary


def verify_payloads(manifest, output, config):
    """Compare the diagnostic display against the unchanged training adapter."""
    from scenemi.data.checked_scene_dataset import CheckedLocalSceneDataset
    from scenemi.native_joint_data import SceneFreeEgoDataset
    from experiments.offline_camera_retrain_v1.native30_observations import inverse_head
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    cfg=yaml.safe_load(Path(config).read_text()); specs=records(manifest); proto=prototypes()
    evidence=[]
    for split in ('validation','train'):
        dataset=CheckedLocalSceneDataset(ROOT/cfg['paths']['scene_paths'],np.zeros(3),
            split=split,config=cfg,selection=ROOT/cfg['paths']['selection'])
        members={r['path']:i for i,r in enumerate(dataset.records)}
        candidates=[s for s in specs if s['row']['path'] in members]
        chosen=[candidates[0],candidates[-1]]
        nonzero=next((s for s in candidates if s['start']>0),None)
        if nonzero:chosen.append(nonzero)
        for spec in chosen:
            row,start=spec['row'],spec['start']
            sample,identity=dataset.sample(np.random.default_rng(7),member=members[row['path']],start=start,yaw=0.)
            meta=dataset.selection_bundle.metadata(dataset.records[members[row['path']]],split)
            actual=scene_payload(meta,start,cfg,artificial_for(meta,proto))
            ref=sample['local_history_points'][sample['local_history_valid']]
            errors=dict(history=float(np.max(np.abs(ref-actual['history']))) if len(ref) else 0.,
                current=float(np.max(np.abs(sample['local_frame_points']-actual['frame']))),
                camera_position=float(np.max(np.abs(sample['camera'][:,:3]*2-actual['camera_positions']))))
            if max(errors.values())>1e-6 or not np.array_equal(sample['local_frame_owner'],actual['owner']):
                raise ValueError('Viewer payload differs from training loader')
            # Use the unchanged scene-free sampler on this already checked
            # index. Its inherited constructor/index state is identical; this
            # avoids rebuilding all corpus indexes for the same records.
            free,free_identity=SceneFreeEgoDataset.sample(dataset,np.random.default_rng(7),
                member=members[row['path']],start=start,yaw=0.)
            scene_keys=('occupancy','local_history_points','local_frame_points','frame_scene_points')
            if any(key in free for key in scene_keys) or free_identity['scene_input']:
                raise ValueError('Scene geometry entered the scene-free training sampler')
            if not np.array_equal(free['motion'],sample['motion']) or not np.array_equal(free['camera'],sample['camera']):
                raise ValueError('Scene-free and scene-loaded samplers disagree on motion/camera')
            fk=forward_kinematics(torch.from_numpy(free['motion'][None]),torch.from_numpy(free['target_rest'][None]))[0].numpy()
            joints=free['motion'][:,135:].reshape(128,22,3)*2
            fk_error=float(np.abs(fk-joints).max())
            head,_,_=inverse_head(free['camera'],free['camera_available'],free['camera_confidence'],meta['actual_virtual_mount'])
            head_error=float(np.abs(head[:,:3]*2-joints[:,15]).max())
            if max(fk_error,head_error)>5e-6:
                raise ValueError('Scene-free FK/camera-head consistency check failed')
            evidence.append(dict(split=split,path=row['path'],start=start,errors_m=errors,
                history_points=len(ref),synthetic_chair=identity['synthetic_chair'],
                current_points_min=int((actual['owner']>=0).sum(-1).min()),
                scene_free_sampler_scene_fields_absent=True,scene_free_motion_camera_equal=True,
                scene_free_fk_max_error_m=fk_error,camera_to_head_max_error_m=head_error))
        del dataset
    value=dict(state='passed',unchanged_training_loader=True,cases=evidence,
        actual_loader='scenemi.data.checked_scene_dataset.CheckedLocalSceneDataset',
        diagnostic_implementation_sha256=sha256(__file__))
    atomic_json(output/'LOADER_PARITY.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)
    return value


def verify_recording_maps(manifest, output, config):
    """Real-loader acceptance of the new map and unchanged EGO targets."""
    from scenemi.data.recording_scene import RecordingSceneDataset, DEFAULT_POLICY
    from scenemi.native_joint_data import SceneFreeEgoDataset
    cfg=yaml.safe_load(Path(config).read_text()); specs=records(manifest); proto=prototypes()
    evidence=[]
    for split in ('validation','train'):
        dataset=RecordingSceneDataset(ROOT/cfg['paths']['scene_paths'],np.zeros(3),
            split=split,config=cfg,selection=ROOT/cfg['paths']['selection'])
        members={r['path']:i for i,r in enumerate(dataset.records)}
        candidates=[s for s in specs if s['row']['path'] in members]
        for spec in (candidates[0],candidates[-1]):
            member=members[spec['row']['path']]; record=dataset.records[member]
            meta=dataset.selection_bundle.metadata(record,split)
            starts=sorted({spec['start'],int(record['starts'][-1])})
            reference_world=None; checks=[]
            for start in starts:
                sample,identity=dataset.sample(np.random.default_rng(7),member=member,start=start,yaw=0.)
                actual=recording_payload(meta,start,cfg,artificial_for(meta,proto))
                points=sample['local_history_points'][sample['local_history_valid']]
                error=float(np.abs(points-actual['history']).max()) if len(points) else 0.
                if error>1e-6 or not np.array_equal(sample['local_frame_owner'],actual['owner']):
                    raise ValueError('Offline recording display and loader differ')
                if not np.allclose(sample['local_frame_points'],actual['frame'],atol=1e-6,rtol=0):
                    raise ValueError('Current-frame point adapter changed')
                free,free_identity=SceneFreeEgoDataset.sample(dataset,np.random.default_rng(7),member=member,start=start,yaw=0.)
                for key in ('motion','camera','target_rest'):
                    if not np.array_equal(sample[key],free[key]):
                        raise ValueError('Recording map changed EGO target or camera: '+key)
                if free_identity['scene_input'] or 'occupancy' in free:
                    raise ValueError('Scene geometry leaked into scene-free sampler')
                world=actual['actual_history']@actual['anchor']+actual['origin']
                if reference_world is None:reference_world=world
                elif not np.allclose(world,reference_world,atol=2e-6,rtol=0):
                    raise ValueError('Recording world map changed with window start')
                if identity['scene_map_observed_frames'] != [0,int(meta['frames'])-1]:
                    raise ValueError('Incomplete recording scope')
                checks.append(dict(start=start,display_loader_error_m=error,
                    observed_static_points=actual['observed_static_points'],input_points=len(points),
                    synthetic_chair=identity['synthetic_chair'],unchanged_motion_camera_rest=True))
            evidence.append(dict(path=record['path'],split=split,world_map_independent_of_window_start=True,checks=checks))
        del dataset
    value=dict(state='passed',protocol='observed-recording-scene-v1',causal=False,
        full_mesh_is_input=False,active_training_changed=False,test_arrays_opened=False,
        policy_sha256=sha256(DEFAULT_POLICY),adapter_sha256=sha256(ROOT/'scenemi/data/recording_scene.py'),
        viewer_sha256=sha256(__file__),cases=evidence)
    output.mkdir(parents=True,exist_ok=True);atomic_json(output/'RECORDING_MAP_ACCEPTANCE.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)
    return value


@contextmanager
def diagnostic_labels():
    import viser
    api=viser.GuiApi; old=api.add_markdown
    def markdown(self,content,*args,**kwargs):
        if content.startswith('## 已保存动作'):
            content='## GRAIL · 源人体与实际场景输入\n\n紫色：原始 SMPL-X 表面。无模型预测；保留源坐标。'
        elif content.startswith('输入参考人体是'):
            content=('青色为整条原序列实际观测到的静态点云，供后续离线场景阶段使用；'
                     '不是完整场景网格。当前帧点可单独切换。橙色补椅仅在指定训练样本启用。'
                     '当前 EGO 无场景训练不读取这些点；旧场景模型仍按原历史输入复现。')
        return old(self,content,*args,**kwargs)
    api.add_markdown=markdown
    try:yield
    finally:api.add_markdown=old


class AlignmentViewer(MotionViewer):
    def __init__(self,specs,cfg,port):
        from experiments.offline_camera_retrain_v1.ops.grail_source import body_model_factory
        self.specs,self.cfg,self.proto=specs,cfg,prototypes()
        self.source_body=body_model_factory('g1')
        self.extras={}; self.mesh_control=None
        cases=[dict(label=s['label'],display_label=s['label'],dataset='grail',regimes=['clean'],
                    selection_reason=s['status'],models=[dict(key='source',label='原始 SMPL-X 表面',mode='ego')]) for s in specs]
        with diagnostic_labels():super().__init__(cases,port)
        self.mesh_control=self.server.gui.add_checkbox('显示原始人体表面（脚底/座面判断用）',initial_value=True)
        self.skeleton_control=self.server.gui.add_checkbox('叠加关节骨架（关节点不等于皮肤表面）',initial_value=False)
        self.synthetic_control=self.server.gui.add_checkbox('训练补椅点（橙色）',initial_value=True)
        self.point_size_control=self.server.gui.add_slider('整段/当前点统一大小（米）',min=.003,max=.02,step=.001,initial_value=.006)
        for control in (self.mesh_control,self.skeleton_control,self.synthetic_control,self.point_size_control):
            control.on_update(lambda _:self.update())
        self.update()

    def read_case(self,index):
        if index in self.cache:return self.cache[index]
        spec=self.specs[index]; row=spec['row']; start=spec['start']
        meta,params,rotation,offset=load_source(row)
        payload=recording_payload(meta,start,self.cfg,artificial_for(meta,self.proto))
        values={k:torch.from_numpy(np.asarray(v[start:start+128]).copy()) for k,v in params.items()}
        vchunks,jchunks=[],[]
        with torch.inference_mode():
            for lo in range(0,128,16):
                result=self.source_body(**{k:v[lo:lo+16] for k,v in values.items()},return_verts=True)
                vchunks.append(result.vertices.numpy()@rotation.T+offset)
                jchunks.append(result.joints[:,:22].numpy()@rotation.T+offset)
        origin,anchor=payload['origin'],payload['anchor']
        verts=((np.concatenate(vchunks)-origin)@anchor.T).astype(np.float32)
        joints=((np.concatenate(jchunks)-origin)@anchor.T).astype(np.float32)
        world=readonly_array(meta,'motion_world')[start:start+128,-66:].reshape(128,22,3)*2
        error=float(np.abs(joints-(world-origin)@anchor.T).max())
        if error>1e-5:raise ValueError('Viewer source body changes source joints')
        ov,of=readonly_array(meta,'scene_object_vertices'),readonly_array(meta,'scene_object_faces')
        sv=np.concatenate((ov,readonly_array(meta,'scene_ground_vertices')))
        sf=np.concatenate((of,readonly_array(meta,'scene_ground_faces')+len(ov)))
        center=(joints.min((0,1))+joints.max((0,1)))/2
        camera=payload['camera_positions']; cr=payload['camera_rotations']
        ends=camera[:,None]+cr.transpose(0,2,1)*.2
        data=dict(truth_vertices=verts,truth_joints=joints,clean_vertices=verts,clean_joints=joints,
            clean_reference_vertices=verts,clean_reference_joints=joints,faces=np.asarray(self.source_body.faces,np.int32),
            clean_observations=np.zeros((128,22,9),np.float32),clean_observation_meta=np.zeros((128,22,5),np.float32),
            scene_vertices=((sv-origin)@anchor.T).astype(np.float32),scene_faces=sf.astype(np.int32),
            visible_static=payload['history'],source_occluded=np.zeros((128,22),bool),
            center=center,radius=np.asarray(2.,np.float32),camera_positions=camera,
            camera_axis_segments=np.stack((np.broadcast_to(camera[:,None],ends.shape),ends),2),
            camera_frustum_segments=np.empty((128,0,2,3),np.float32),
            camera_path=np.stack((camera[:-1],camera[1:]),1),diagnostic_payload=payload)
        for t in range(128):
            data['current_static_'+str(t)]=payload['frame'][t,payload['owner'][t]>=0]
        identity=dict(path=meta['path'],sequence_id=meta['sequence_id'],start=start,origin=origin.tolist(),anchor=anchor.tolist())
        item=dict(entry=self.cases[index]['models'][0],data=data,meta=dict(identity=identity,scene_errors=[]))
        self.cache[index]=[item]
        while len(self.cache)>1:self.cache.popitem(last=False)
        return [item]

    def load_case(self):
        self.extras={}
        super().load_case()
        rt=self.runtime['ego']; data=rt['data']; p=self.panels['ego']
        p['static'].label='整段已见场景输入（最多8192点）'
        p['current'].label='当前帧输入（最多512点，随帧更新）'
        p['scan'].label='完整场景网格（仅参考）'
        p['gt'].value=False
        for k in ('gt','reference','reference_source','observations','uncertainty','occluded'):
            p[k].visible=False
        p['models']['source'].visible=False
        for group in self.preset_buttons.values():
            for button in group.values():button.visible=False
        self.regime.visible=False
        self.extras['synthetic']=self.points('/ego/diagnostic/synthetic_chair',(245,150,25),.006)
        self.extras['synthetic'].points=data['diagnostic_payload']['synthetic']
        self.extras['skeleton']=self.lines('/ego/diagnostic/joints',np.empty((0,2,3)),(100,20,100),2.)
        rt['scan'].opacity=.3
        p['static'].value=True
        p['current'].value=False
        p['scan'].value=False
        self.update()

    def update(self):
        super().update()
        if self.loading or not self.runtime:return
        from scenemi.geometry.body import PARENTS
        rt=self.runtime['ego'];data=rt['data'];p=self.panels['ego'];t=int(self.frame.value)
        spec=self.specs[self.index]; pl=data['diagnostic_payload']
        size=.006 if self.mesh_control is None else float(self.point_size_control.value)
        for key in ('static','current'):rt[key].point_size=size
        if self.mesh_control is not None:
            rt['models'][0]['mesh'].visible=bool(p['enabled'].value and self.mesh_control.value)
        if self.extras:
            self.extras['synthetic'].visible=bool(p['enabled'].value and (self.mesh_control is None or self.synthetic_control.value))
            self.extras['synthetic'].point_size=size
            joints=data['truth_joints'][t]
            self.extras['skeleton'].points=np.asarray([[joints[parent],joints[child]] for child,parent in enumerate(PARENTS) if parent>=0],np.float32)
            self.extras['skeleton'].visible=bool(self.mesh_control is not None and self.skeleton_control.value and p['enabled'].value)
        p['status'].content='原始数据审阅 · 无模型推理 · 源坐标未调整'
        p['info'].content=(f"原序列：{spec['row']['sequence_id']}\n\n状态：**{spec['status']}**；窗口起点：{spec['start']}；当前原帧：**{spec['start']+t}**。\n\n"
            f"整段观测范围：**0–{pl['recording_frames']-1}**；静态已见点去重后：**{pl['observed_static_points']}**；实际输入预算后：**{len(pl['history'])}**。\n\n"
            f"当前帧另有：**{int((pl['owner'][t]>=0).sum())}** 点；补椅点：**{len(pl['synthetic'])}**。\n\n"
            '此为后续离线场景输入协议：整条原序列实际观测点，2.5cm 体素去重、最多8192点；动态物体仅用当前帧。'
            '\n\n当前无场景训练不读场景。旧场景3000步模型仍使用原历史/当前输入，不作追溯替换。'
            '\n\n脚底/座面应检查紫色皮肤网格；骨架关节点位于人体内部。')
        self.info.content=(f"窗口第 **{t}** 帧；原帧 **{spec['start']+t}**；{spec['row']['split']} / {spec['status']}。\n\n"
            '整段已见点云已默认显示，完整网格默认关闭、仅供参考。保留/排除仍是原质量清单。')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['audit','payload-check','recording-check','serve'])
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/scene_joint128_conditioning_hybrid.yaml')
    parser.add_argument('--port',type=int,default=8094)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if args.action=='audit':audit(args.manifest,args.output,args.config)
    elif args.action=='payload-check':verify_payloads(args.manifest,args.output,args.config)
    elif args.action=='recording-check':verify_recording_maps(args.manifest,args.output,args.config)
    else:
        cfg=yaml.safe_load(args.config.read_text());specs=records(args.manifest)
        viewer=AlignmentViewer(specs,cfg,args.port)
        args.output.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output/'SERVER.json',dict(pid=os.getpid(),url=f'http://localhost:{args.port}/',cases=len(specs),
            viewer='MotionViewer + source-surface/point-payload adapter',model_inference=False,
            map_protocol='observed-recording-scene-v1',map_scope='complete_source_recording',
            causal=False,current_ego_training_changed=False,historical_scene_training_changed=False,
            full_scene_is_input=False,test_arrays_opened=False))
        print(f'VISER_URL=http://localhost:{args.port}/',flush=True)
        viewer.run()


if __name__=='__main__':main()
