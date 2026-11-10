"""Export saved-query diagnostics and play them in the shared MotionViewer.

The exporter replays complete original CUDA batches before selecting windows.
All point stages use the SAME camera-derived origin and rotation as training;
there is no mesh fitting, floor snapping or GT-based scene alignment.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.offline_camera_retrain_v1.ops.motion_viewer import MotionViewer

PROTOCOL = 'scene-query-diagnostic-preview-v1'
GROUP = '场景查询诊断 · EgoBody 零命中窗口'
PARTS = {'双踝及双脚 · 12锚点': (7, 8, 10, 11), '骨盆 · 3锚点': (0,),
         '双膝 · 6锚点': (4, 5), '全身 · 67锚点': tuple(range(22))}


def select_misses(rows, count):
    groups = {}
    for row in rows:
        if row['identity']['dataset']=='egobody' and row['audit']['correct']['zero_reason']=='no_anchor_neighbour_hits':
            groups.setdefault(row['identity']['family'], []).append(row)
    if count < 1 or count > len(groups):
        raise ValueError(f'Request between 1 and {len(groups)} different recordings')
    names = sorted(groups)
    result = []
    for i in np.linspace(0, len(names)-1, count).astype(int):
        candidates = groups[names[i]]
        gt_hit = [r for r in candidates if r['audit']['correct']['gt_query_hit_fraction']>0]
        ordered = sorted(gt_hit or candidates, key=lambda r:(r['case']['start'],r['case']['case_id']))
        result.append(ordered[(len(ordered)-1)//2])
    return result


def sphere_segments(centres, radius, steps=32):
    """Three orthogonal great circles, at the true metric query radius."""
    angles = np.arange(steps+1)*2*np.pi/steps
    paths = []
    for first, second in ((0,1),(0,2),(1,2)):
        circle = np.zeros((steps+1,3),np.float32)
        circle[:,first], circle[:,second] = np.cos(angles)*radius, np.sin(angles)*radius
        points = np.asarray(centres,np.float32)[:,None]+circle
        paths.append(np.stack((points[:,:-1],points[:,1:]),2).reshape(-1,2,3))
    return np.concatenate(paths).astype(np.float32)


def nearest_static(anchors, points):
    if not len(points):
        return np.full(anchors.shape[:-1],np.inf), np.zeros_like(anchors)
    distance, index = cKDTree(points).query(anchors.reshape(-1,3),workers=1)
    return distance.reshape(anchors.shape[:-1]), points[index].reshape(anchors.shape)


def nearest_input(anchors, static, dynamic, owners):
    distance, closest = nearest_static(anchors, static)
    for t in range(len(anchors)):
        points = dynamic[t,(owners[t]>0)&(owners[t]<100)]
        if len(points):
            d,p = nearest_static(anchors[t],points)
            take = d < distance[t]
            distance[t,take], closest[t,take] = d[take], p[take]
    return distance, closest


def prepare_video_layers(manifest, policy_file, output):
    """Attach full-video map layers without changing saved motion or old inputs."""
    from scenemi.config import ROOT
    from scenemi.data.amass_prepare import atomic_json, sha256
    from scenemi.data.checked_scene_dataset import SceneSelection
    from scenemi.data.video_scene import load_policy, VideoMapStore
    from scenemi.geometry.spatial_scene import SceneView
    from scenemi.visualization import prepared_playback
    from experiments.offline_camera_retrain_v1.data import anchor_rotation

    manifest, output = manifest.resolve(), output.resolve()
    if not output.is_relative_to(ROOT/'runs'):
        raise ValueError('Preview exports must stay inside runs')
    prepared_playback(manifest)  # Verify all historical motion assets first.
    value = json.loads(manifest.read_text());policy = load_policy(policy_file)
    if value['protocol'] != PROTOCOL:
        raise ValueError('Choose an existing query-diagnostic preview manifest')
    store = VideoMapStore(ROOT/policy['audit']['video_maps']/'MANIFEST.json')
    bundle = SceneSelection()
    records = {r['path']: r for r in bundle.rows('native_manifest')}
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for case in value['cases']:
        if not case.get('scene_query_diagnostic'):
            continue
        source = Path(case['models'][0]['case']['cache'])
        source = source if source.is_absolute() else manifest.parent/source
        with np.load(source, allow_pickle=False) as data:
            identity = json.loads(str(data['metadata_json']))['identity']
            meta = bundle.metadata(records[identity['path']], 'validation')
            camera = bundle.array(meta, 'camera_position_scenemi_yup')
            first = identity['start'];origin = np.array(camera[first], np.float64);origin[1] = 0.
            rotation = anchor_rotation(bundle.array(meta, 'camera_rotation_scenemi_yup')[first:first+128])
            index, row = store.get(identity['path']);view = SceneView(index, origin, rotation)
            np.testing.assert_allclose(view.to_local(camera[first:first+128]), data['camera_positions'], atol=1e-5)
            points = view.to_local(index.points).astype(np.float32)
            distance = index.nearest(view.to_world(data['query_predicted_anchors'])).astype(np.float32)
            saved_audit = ROOT/policy['audit']['output']/f'distances/case_{case["case_id"]:06d}.npz'
            with np.load(saved_audit, allow_pickle=False) as audited:
                np.testing.assert_allclose(distance, audited['video_voxel'], atol=1e-5)
        metadata = dict(map_key=row['key'], points=len(points), coverage=row['coverage'],
            observer=row['identity'], map_sha256=row['sha256'], policy_sha256=sha256(policy_file),
            original_motion_unchanged=True, used_by_historical_refiner=False,
            purpose='original-video spatial-query input review; not a retrained-model prediction')
        file = output/f'video_map_case_{case["case_id"]:06d}.npz'
        with file.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, video_observed_static=points, query_video_distance_m=distance,
                video_metadata_json=np.array(json.dumps(metadata)))
        file.with_suffix('.tmp').replace(file)
        case['scene_query_video_asset'] = dict(path=str(file), sha256=sha256(file))
        count += 1
    value.update(video_map_manifest_sha256=sha256(ROOT/policy['audit']['video_maps']/'MANIFEST.json'),
        source_manifest_sha256=sha256(manifest), original_motion_assets_unchanged=True)
    file = output/'PREPARED_CASES.json';atomic_json(file, value)
    print(f'Added original-video map layers to {count} cases', flush=True)
    return file


def prepare(diagnostic, output, count, device):
    import torch
    from scenemi.config import ROOT, load_paths
    from scenemi.data.amass_prepare import atomic_json, sha256
    from scenemi.data.scene_geometry_data import GeometryEgoData, collate, payload
    from scenemi.refinement.training import load_refined_model, fixed_cases, verify_contract
    from scenemi.geometry.surface_query import query_surface
    from scenemi.geometry.native_shape import GENDERS
    from scenemi.replay.visualization import mesh
    from scenemi.training import deterministic
    from scenemi.execution import configure_runtime
    from experiments.offline_camera_retrain_v1.data import anchor_rotation

    diagnostic, output = diagnostic.resolve(), output.resolve()
    if not output.is_relative_to(ROOT/'runs'):
        raise ValueError('Preview exports must stay inside runs')
    contract = json.loads((diagnostic/'CONTRACT.json').read_text())
    if not contract['full_validation'] or not json.loads((diagnostic/'STATUS.json').read_text())['state']=='complete':
        raise ValueError('Choose completed full frozen-refiner diagnostics')
    if device != 'cuda':
        raise ValueError('Exact validation replay requires the original CUDA noise stream')
    rows = [r for p in sorted((diagnostic/'batches').glob('*.json')) for r in json.loads(p.read_text())]
    selected = select_misses(rows, count)
    checkpoint = Path(contract['checkpoint'])
    if sha256(checkpoint) != contract['checkpoint_sha256']:
        raise ValueError('Diagnosed refiner checkpoint changed')
    model, config, step = load_refined_model(checkpoint, load_paths(ROOT/'configs/paths.local.yaml'))
    verify_contract(torch.load(checkpoint, map_location='cpu', weights_only=False,mmap=True)['contract'])
    configure_runtime(config, device); deterministic(config['seed'], device)
    model = model.to(device)
    data = GeometryEgoData(config, load_paths(ROOT/'configs/paths.local.yaml'), 'validation')
    cases = fixed_cases(data, config)
    if cases != contract['cases']:
        raise ValueError('Diagnostic validation membership changed')
    output.mkdir(parents=True,exist_ok=True)
    selection = dict(protocol=PROTOCOL, checkpoint=str(checkpoint), checkpoint_step=step,
        checkpoint_sha256=contract['checkpoint_sha256'], diagnostic_contract_sha256=sha256(diagnostic/'CONTRACT.json'),
        selection='Different recordings, evenly spread through sorted recording IDs; temporal median no-hit window per recording, preferring GT-hit no-hit windows when available',
        cases=selected, inference='original complete CUDA validation batches, DDIM20 seed777; export selected members only',
        coordinate_policy='original camera-derived window origin/rotation for ALL body and point stages; no extra alignment',
        source_sha256=sha256(__file__), test_used=False)
    selection_file = output/'SELECTION.json'
    if selection_file.exists() and json.loads(selection_file.read_text()) != selection:
        raise ValueError('Existing preview selection changed')
    atomic_json(selection_file, selection)
    prepared, checks, meshes = [], [], {}
    size = contract['batch_size']
    by_batch = {}
    for index,row in enumerate(selected):
        position = next(i for i,c in enumerate(cases) if c['case_id']==row['case']['case_id'])
        by_batch.setdefault(position//size*size,[]).append((index,position,row))
    with torch.inference_mode():
        for first, members in sorted(by_batch.items()):
            chunk = cases[first:first+size]
            pairs = [data.sample_case(case) for case in chunk]
            batch = collate([p[0] for p in pairs], device)
            public = payload(batch)
            raw, details = model.coarse(public, steps=config['sampling']['ddim_steps'],seed=config['validation']['seed'])
            placed = model.locate(raw,details)
            anchors, _ = model.skin_points(placed,details,batch['body_format'])
            teacher = dict(rest=batch['target_rest'],predicted_native_betas=batch['target_native_betas'],predicted_template=batch['target_template'])
            gt_anchors, _ = model.skin_points(batch['motion'],teacher,batch['body_format'])
            hit = query_surface(anchors,public,config['geometry'])['hit']
            for index, position, selected_row in members:
                i = position-first; case = chunk[i]; sample, identity = pairs[i]
                if identity != selected_row['identity'] or hit[i].any():
                    raise ValueError('Selected window no longer reproduces the saved no-hit result')
                record = data.native.records[case['member']]
                meta = dict(data.native.selection_bundle.metadata(record,'validation'),path=record['path'])
                saved = data.native.cached(record)
                span = slice(case['start'],case['start']+128)
                origin = np.asarray(saved['camera_position_scenemi_yup'][case['start']],np.float64).copy();origin[1]=0
                rotation = anchor_rotation(np.asarray(saved['camera_rotation_scenemi_yup'][span],np.float64))
                local = lambda x: ((np.asarray(x)-origin)@rotation.T).astype(np.float32)
                source_points = np.asarray(data.native.selection_bundle.array(meta,'visible_frame_points_scenemi_yup'))
                source_owners = np.asarray(data.native.selection_bundle.array(meta,'visible_frame_owner'))
                source_valid = np.asarray(data.native.selection_bundle.array(meta,'observation_valid'),bool)
                valid = (source_owners==0)&np.isfinite(source_points).all(-1)&source_valid[:,None]
                raw_points = local(source_points[valid])
                voxel = local(saved['static_full'])
                actual = sample['static_points'][sample['static_valid']]
                np.testing.assert_array_equal(actual,local(saved['static_points']))
                motion,rest = placed[i].cpu().numpy(),details['rest'][i].cpu().numpy()
                info = dict(meta['native_body'],gender=GENDERS[int(details['predicted_template'][i])],
                    betas=details['predicted_native_betas'][i].cpu().numpy().tolist())
                pv,faces,pj,pe = mesh(motion,rest,info,meshes)
                tv,truth_faces,tj,te = mesh(sample['motion'],sample['target_rest'],meta['native_body'],meshes)
                if not np.array_equal(faces,truth_faces):
                    raise ValueError('Displayed body topologies differ')
                source_joints = np.asarray(saved['motion_world'][span,135:]).reshape(128,22,3)*2
                coordinate_error = float(np.abs(local(source_joints)-tj).max())
                camera = sample['camera'][:,:3]*2
                camera_error = float(np.abs(local(saved['camera_position_scenemi_yup'][span])-camera).max())
                pred = anchors[i].cpu().numpy(); gt = gt_anchors[i].cpu().numpy()
                distance_raw,_ = nearest_static(pred,raw_points)
                distance_voxel,_ = nearest_static(pred,voxel)
                distance_actual,closest = nearest_input(pred,actual,sample['frame_scene_points'],sample['frame_scene_owner'])
                gt_distance,_ = nearest_input(gt,actual,sample['frame_scene_points'],sample['frame_scene_owner'])
                if np.any(distance_actual<=config['geometry']['query_radius_m']):
                    raise ValueError('Exported input point cloud disagrees with the actual CUDA query')
                measured = float(np.linalg.norm(pj.astype(np.float64)-tj.astype(np.float64),axis=-1).mean()*1000)
                if abs(measured-selected_row['variants']['baseline']['mpjpe_mm'])>.002:
                    raise ValueError('Preview differs from the saved CUDA baseline')
                audit = dict(case_id=case['case_id'],identity=identity,body_source_coordinate_max_error_m=coordinate_error,
                    camera_source_coordinate_max_error_m=camera_error,mesh_fk_max_error_m=max(pe,te),
                    raw_static_points=len(raw_points),voxel_static_points=len(voxel),actual_static_points=len(actual),
                    predicted_hits_raw_fraction=float((distance_raw<=.2).mean()),
                    predicted_hits_voxel_fraction=float((distance_voxel<=.2).mean()),
                    predicted_hits_actual_fraction=float((distance_actual<=.2).mean()),
                    gt_hits_actual_fraction=float((gt_distance<=.2).mean()),
                    mpjpe_mm=measured,validation_mpjpe_mm=selected_row['variants']['baseline']['mpjpe_mm'])
                checks.append(audit)
                six = sample['camera'][:,3:]
                camera_rotation = np.stack((six[:,:3],six[:,3:],np.cross(six[:,:3],six[:,3:])), -1)
                ends = camera[:,None]+camera_rotation.swapaxes(-1,-2)*.3
                geometry = {f'current_dynamic_{t}':sample['frame_scene_points'][t,(sample['frame_scene_owner'][t]>0)&(sample['frame_scene_owner'][t]<100)] for t in range(128)}
                file = output/f'case_{case["case_id"]:06d}.npz'
                metadata = dict(identity=identity,metrics=dict(world_mpjpe_mm=measured),
                    checkpoint_step=22000,target_shape_used_for_prediction=False,scene_errors=[],query_audit=audit,
                    observation_policy='complete source first-hit static observations; not a dense sensor reconstruction',
                    point_source_sha256={k:meta['output_sha256'][k+'.npy'] for k in
                        ('visible_frame_points_scenemi_yup','visible_frame_owner','observation_valid')})
                with file.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream,truth_vertices=tv,truth_joints=tj,faces=faces,
                        preview_vertices=pv,preview_joints=pj,preview_reference_vertices=pv,preview_reference_joints=pj,
                        preview_observations=sample['trajectory'],preview_observation_meta=sample['observation_meta'],
                        scene_vertices=np.empty((0,3),np.float32),scene_faces=np.empty((0,3),np.int32),visible_static=actual,
                        raw_observed_static=raw_points,voxel_observed_static=voxel,
                        query_predicted_anchors=pred,query_gt_anchors=gt,query_anchor_joints=np.asarray([j for j in range(22) for _ in range(3)]+[15]),
                        query_raw_distance_m=distance_raw.astype(np.float32),query_voxel_distance_m=distance_voxel.astype(np.float32),
                        query_input_distance_m=distance_actual.astype(np.float32),query_input_closest=closest,
                        query_gt_input_distance_m=gt_distance.astype(np.float32),query_radius_m=np.float32(.2),
                        center=tj[:,0].mean(0)+np.array([0,.4,0]),radius=np.float32(2.6),
                        camera_positions=camera,camera_axis_segments=np.stack((np.broadcast_to(camera[:,None],ends.shape),ends),2),
                        camera_frustum_segments=np.empty((128,0,2,3),np.float32),camera_path=np.stack((camera[:-1],camera[1:]),1),
                        metadata_json=np.array(json.dumps(metadata)),**geometry)
                file.with_suffix('.tmp').replace(file)
                tag = 'GT 可命中' if audit['gt_hits_actual_fraction']>0 else 'GT 亦无命中'
                label = f'EgoBody · 零命中 · {identity["family"]} · 窗口 {case["start"]}–{case["start"]+127} · {tag}'
                prepared.append(dict(label=label,display_label=label,visualization_group=GROUP,scene_query_diagnostic=True,
                    regimes=['preview'],initial_frame=64,case_id=case['case_id'],
                    selection_reason='不同 recording 的零命中窗口；保留验证坐标和完整 CUDA 批次噪声。诊断选例不代表平均性能。',
                    models=[dict(key='ego22k_coarse_B',label='22k 无场景粗动作＋时间一致相机定位（紫）',color=[185,40,160],mode='ego',
                                 case=dict(cache=str(file),cache_sha256=sha256(file)))]))
                print(f'Exported {len(prepared)}/{len(selected)} · {identity["family"]} · raw/voxel/input {len(raw_points)}/{len(voxel)}/{len(actual)}',flush=True)
    prepared.sort(key=lambda c:c['label'])
    atomic_json(output/'PREPARED_CASES.json',dict(protocol=PROTOCOL,cases=prepared))
    atomic_json(output/'AUDIT.json',dict(state='passed',cases=checks,source_arrays_modified=False,
        extra_scene_alignment=False,exact_cuda_validation_noise=True,all_selected_queries_have_zero_hits=True,
        scene_full_scan_used_as_input=False,raw_display_decimation=False))
    return output/'PREPARED_CASES.json'


class QueryDiagnosticViewer(MotionViewer):
    """Extra diagnostic layers; all rendering/playback stays in MotionViewer."""
    def __init__(self,cases,port):
        self.query_controls = None
        super().__init__(cases,port)
        self.original_help = self.help.content
        self.original_labels = {k:v.label for k,v in self.panels['ego'].items()
                                if hasattr(v,'label')}
        with self.server.gui.add_folder('场景查询诊断 · 独立图层'):
            self.query_controls = {name:self.server.gui.add_checkbox(label,initial_value=value) for name,label,value in (
                ('raw','点云 · 原导出片段观测（体素化前，灰）',True),
                ('voxel','点云 · 原导出片段体素并集（截断前，青）',False),
                ('video','点云 · 原视频全局体素地图（不限点数，绿）',False),
                ('gt_anchors','锚点 · GT 皮肤（蓝）',False),
                ('pred_anchors','锚点 · 粗预测皮肤（紫）',True),
                ('spheres','查询球 · 旧版实际半径 20 cm（橙）',True),
                ('wide','查询球 · 40 cm 范围参考（仅显示，灰）',False),
                ('context','查询球 · 新版中尺度 80 cm（上下文参考，绿）',False),
                ('nearest','连线 · 粗预测锚点至最近实际输入点（橙）',False))}
            self.query_parts = self.server.gui.add_dropdown('显示哪些查询锚点',options=tuple(PARTS),initial_value=next(iter(PARTS)))
            self.query_centres = self.server.gui.add_dropdown('查询球中心',options=('粗预测 · 模型实际查询','GT · 仅诊断参考'),initial_value='粗预测 · 模型实际查询')
            self.query_note = self.server.gui.add_markdown('')
        for control in [*self.query_controls.values(),self.query_parts,self.query_centres]:
            control.on_update(lambda _:self.update())
        self.load_case()

    def read_case(self, index):
        from scenemi.data.amass_prepare import sha256
        items = super().read_case(index)
        asset = self.cases[index].get('scene_query_video_asset')
        if asset and 'video_observed_static' not in items[0]['data']:
            if sha256(asset['path']) != asset['sha256']:
                raise ValueError('Original-video preview map changed')
            with np.load(asset['path'], allow_pickle=False) as saved:
                items[0]['data'].update({k: np.array(saved[k]) for k in saved.files})
            items[0]['meta']['query_video_map'] = json.loads(str(items[0]['data']['video_metadata_json']))
        return items

    def load_case(self):
        super().load_case()
        if self.query_controls is None:
            return
        diagnostic = bool(self.cases[self.index].get('scene_query_diagnostic'))
        self.regime.visible = not diagnostic
        self.help.content = ('灰色原始点云包含所用录制片段全部帧的已有 first-hit 静态观测，未体素化或截断；'
            '它不是完整场景扫描，也不是传感器稠密点云。青色为 2.5 cm 体素并集，橙色为最多 8192 点的实际静态输入。'
            '绿色为同一观察者的原视频完整可用观测体素地图，不限点数；它供新版查询检查，未参与此处旧模型预测。'
            '动态输入只显示当前帧，最多 64 点。20 cm 为旧模型实际查询半径；40 cm 和新版 80 cm 仅供观察范围。'
            '所有图层使用原始训练坐标，显示不追加对齐或贴地。' if diagnostic else self.original_help)
        for control in [*self.query_controls.values(),self.query_parts,self.query_centres,self.query_note]:
            control.visible = diagnostic
        panel = self.panels['ego']
        for key,label in self.original_labels.items():
            panel[key].label = label
        if not diagnostic:
            return
        with self.lock,self.server.atomic():
            self.regime.visible = False
            for key in ('reference','observations','uncertainty','reference_source','occluded','scan','current'):
                panel[key].visible = False
                if hasattr(panel[key],'value') and isinstance(panel[key].value,bool):
                    panel[key].value = False
            for buttons in self.preset_buttons.values():
                for button in buttons.values():button.visible = False
            panel['static'].label = '点云 · Refiner 实际静态输入（橙，最多 8192 点）'
            panel['dynamic'].label = '点云 · Refiner 实际动态输入（当前帧，最多 64 点）'
            panel['gt'].label = 'GT 人体 · 数据集真实体型与动作（蓝）'
            panel['static'].value = True
            rt = self.runtime['ego']; data = rt['data']
            for key in ('video','context'):
                self.query_controls[key].visible = 'video_observed_static' in data
            extra = {}
            for key,color,size in [('raw',(155,165,175),.003),('voxel',(25,185,195),.005),
                                    ('video',(70,190,90),.006),
                                    ('gt_anchors',(45,120,220),.022),('pred_anchors',(185,40,160),.022)]:
                extra[key] = self.points('/ego/query_diagnostic/'+key,color,size)
            extra['raw'].points=data['raw_observed_static'];extra['voxel'].points=data['voxel_observed_static']
            extra['video'].points=data.get('video_observed_static',np.empty((0,3),np.float32))
            rt['static'].colors = np.broadcast_to(np.array([245,145,25],np.uint8),
                                                 (len(data['visible_static']),3)).copy()
            for key,color in [('spheres',(245,145,25)),('wide',(125,135,145)),
                              ('context',(70,190,90)),('nearest',(245,145,25))]:
                extra[key] = self.lines('/ego/query_diagnostic/'+key,np.zeros((0,2,3)),color,1.5)
            rt['query_diagnostic'] = extra
        self.update()

    def update(self):
        super().update()
        if self.loading or self.query_controls is None or 'ego' not in self.runtime:
            return
        rt=self.runtime['ego'];extra=rt.get('query_diagnostic')
        if extra is None:
            return
        with self.lock,self.server.atomic():
            data=rt['data'];t=int(self.frame.value)
            mask=np.isin(data['query_anchor_joints'],PARTS[self.query_parts.value])
            for name in ('gt_anchors','pred_anchors'):
                field='query_gt_anchors' if name=='gt_anchors' else 'query_predicted_anchors'
                extra[name].points=data[field][t,mask]
            centres=data['query_gt_anchors' if self.query_centres.value.startswith('GT') else 'query_predicted_anchors'][t,mask]
            extra['spheres'].points=sphere_segments(centres,float(data['query_radius_m']))
            extra['wide'].points=sphere_segments(centres,.4)
            extra['context'].points=sphere_segments(centres,.8)
            extra['nearest'].points=np.stack((data['query_predicted_anchors'][t,mask],data['query_input_closest'][t,mask]),1)
            for name,handle in extra.items():
                handle.visible=bool(self.panels['ego']['enabled'].value and self.query_controls[name].value)
            audit=rt['meta']['query_audit']
            lines=[]
            for title,key in [('体素化前','query_raw_distance_m'),('体素并集','query_voxel_distance_m'),('实际输入','query_input_distance_m')]:
                distance=data[key][t,mask]
                lines.append(f'| {title} | {distance.min()*100:.1f} cm | {int((distance<=.2).sum())}/{int(mask.sum())} |')
            if 'query_video_distance_m' in data:
                distance=data['query_video_distance_m'][t,mask]
                lines.append(f'| 原视频全局地图 | {distance.min()*100:.1f} cm | {int((distance<=.2).sum())}/{int(mask.sum())} |')
            self.query_note.content=('**当前帧 · 所选粗预测锚点的精确最近距离**\n\n'
                '| 点云阶段 | 最近距离 | 20 cm 内锚点 |\n|---|---:|---:|\n'+'\n'.join(lines)+
                f'\n\n点数：体素化前 **{audit["raw_static_points"]:,}** → 体素并集 **{audit["voxel_static_points"]:,}** → 实际静态输入 **{audit["actual_static_points"]:,}**。'
                '\n\n整窗全部 67 个粗预测锚点的 20 cm 命中率（依次为体素化前静态点、体素并集、实际静态＋当前帧动态输入）：'
                + ' → '.join(f'{(data[key]<=.2).mean()*100:.2f}%' for key in
                    ('query_raw_distance_m','query_voxel_distance_m','query_input_distance_m')) + '。')
            if 'query_video_map' in rt['meta']:
                video=rt['meta']['query_video_map'];coverage=video['coverage']
                self.query_note.content += (f'\n\n原视频全局地图：**{video["points"]:,} 点**，'
                    f'原帧范围 **{coverage["original_video_bounds"][0]}–{coverage["original_video_bounds"][1]}**，'
                    f'已处理可用观测 **{coverage["observed_frames"]}/{coverage["original_clock_frames"]} 帧**；'
                    '缺失观测不补造。整窗原视频地图 20 cm／80 cm 命中率：'
                    f'**{(data["query_video_distance_m"]<=.2).mean()*100:.2f}%／'
                    f'{(data["query_video_distance_m"]<=.8).mean()*100:.2f}%**。')
            self.panels['ego']['info'].content=(f'**{rt["meta"]["identity"]["family"]}**\n\n'
                f'验证 case ID：{audit["case_id"]}；窗口起始：{rt["meta"]["identity"]["start"]}。'
                f'\n\nGT 与源动作坐标最大偏差：{audit["body_source_coordinate_max_error_m"]*1000:.4f} mm；'
                f'相机坐标最大偏差：{audit["camera_source_coordinate_max_error_m"]*1000:.4f} mm。'
                f'\n\n粗预测 MPJPE：{audit["mpjpe_mm"]:.2f} mm，复现原 CUDA 验证；Refiner 在这些窗口因无命中输出零修正。')


def main(argv=None):
    from scenemi.config import ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','prepare-video'))
    parser.add_argument('--diagnostic',type=Path,default=ROOT/'runs/ego_scene_refinement/diagnostics/scene_causality')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--recordings',type=int,default=10)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/ego_scene_spatial_query.yaml')
    args=parser.parse_args(argv)
    if args.action=='prepare-video':
        if args.manifest is None:parser.error('prepare-video requires --manifest')
        print(prepare_video_layers(args.manifest,args.config,args.output),flush=True)
        return
    print(prepare(args.diagnostic,args.output,args.recordings,args.device),flush=True)
