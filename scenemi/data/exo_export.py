"""Stream full TRUMANS moving-camera observations; retain native EGO sources.

The output is a versioned EXO overlay with explicit borrowed-array identities.
No old dataset, source split, checkpoint, or training configuration is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

import numpy as np
import yaml

from scenemi.data.exo_camera import camera_trajectory, render_frame, projection_error
from scenemi.native_visualization import ROOT, sha256


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def candidates(manifest):
    rows = [r for r in map(json.loads, manifest.read_text().splitlines())
            if r['dataset']=='trumans' and r['split'] in ('train','validation')
            and 'exo' in r['allowed_modes'] and not r['diagnostic_only']]
    if not rows or len({(r['split'],r['sequence_id']) for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate TRUMANS source identities')
    return sorted(rows, key=lambda r:(r['split'],r['sequence_id']))


def plan(manifest, output):
    rows = candidates(manifest)
    frames = sum(r['frames'] for r in rows)
    # 512 xyz/owner samples + masked joints/confidence/counts + per-frame poses.
    estimated = frames*(512*16 + 22*(12+1+4+5*4) + 16*4)
    disk_root = next(p for p in (output,*output.parents) if p.exists())
    return dict(source_manifest=str(manifest.resolve()), source_manifest_sha256=sha256(manifest),
        datasets=['trumans'], splits=['train','validation'], test_used=False,
        sequences=len(rows), frames=frames,
        by_split={s:dict(sequences=sum(r['split']==s for r in rows),
                         frames=sum(r['frames'] for r in rows if r['split']==s)) for s in ('train','validation')},
        estimated_output_bytes=estimated, free_disk_bytes=shutil.disk_usage(disk_root).free,
        output=str(output.resolve()), training_started=False)


def array_path(meta, filename):
    """Resolve an explicit borrowed array, preserving its content hash."""
    return Path(meta.get('array_sources', {}).get(filename, str(Path(meta['path'])/filename)))


def verify_output(meta):
    for filename, digest in meta['output_sha256'].items():
        if sha256(array_path(meta, filename)) != digest:
            raise ValueError(f'Incomplete or changed EXO overlay: {filename}')


def source_dependencies(explicit):
    """Ignore third-party synthetic module filenames such as '_classes.py'."""
    files = {ROOT/name for name in explicit}
    for module in list(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if not isinstance(name,str) or not name.endswith('.py'):
            continue
        path = Path(name)
        if not path.is_absolute() or not path.is_file():
            continue
        path = path.resolve()
        if ((path.is_relative_to(ROOT) or path.is_relative_to(ROOT.parent/'diffusion-motion-inbetweening'))
                and not path.is_relative_to(ROOT/'tests')):
            files.add(path)
    return sorted(files)


def export_one(row, folder, policy, model, face_labels, clips, status_update, *, diagnostic):
    import torch
    import trimesh
    from experiments.offline_camera_retrain_v1.data import PROJECTS
    from experiments.offline_camera_retrain_v1.native30_export import sampled_environment
    from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array
    from experiments.offline_camera_retrain_v1.prepare_trumans_dynamic_v2 import load_objects
    from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene

    disk_metadata = Path(row['path'])/'metadata.json'
    if {k:v for k,v in row.items() if k!='path'} != json.loads(disk_metadata.read_text()):
        raise ValueError('Native manifest and source metadata disagree')
    n = row['frames']
    q = np.asarray(readonly_array(row, 'source_frame_ids')).astype(int)
    valid = np.asarray(readonly_array(row, 'observation_valid')).copy()
    released = np.asarray(readonly_array(row, 'trumans_released_camera_to_world_reference'))
    pose, camera = camera_trajectory(released, policy)
    if len(q) != n or len(valid) != n or len(pose) != n:
        raise ValueError('Native source/camera clocks disagree')
    target = readonly_array(row, 'motion_world')
    clip = clips[row['sequence_id']]
    source_fit = PROJECTS/'TRUMANS'/clip['smplx_global']
    with source_fit.open('rb') as stream:
        fit = pickle.load(stream)
    beta = np.asarray(row['native_body']['betas'], np.float32)
    fit_beta = np.asarray(fit['betas'], np.float32).reshape(-1)
    if np.any(fit_beta[10:] != 0) or not np.array_equal(beta, np.pad(fit_beta[:10], (0,max(0,10-len(fit_beta[:10]))))):
        raise ValueError('Native source shape changed')
    mesh_path = PROJECTS/'TRUMANS'/clip['scene']['mesh']
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    static = static_ray_scene(mesh.vertices, mesh.faces)
    objects, object_audit = [], []
    for i, (name,path,r,p,source,ov,_) in enumerate(load_objects(PROJECTS/'TRUMANS', clip, q)):
        if (not np.allclose(r, readonly_array(row,'object_rotations')[:,i], atol=1e-5)
            or not np.allclose(p, readonly_array(row,'object_positions')[:,i], atol=1e-5)
            or not np.array_equal(ov, readonly_array(row,'object_valid')[:,i])):
            raise ValueError('Dynamic object clock/transform changed')
        obj = trimesh.load(path, force='mesh', process=False)
        objects.append((static_ray_scene(obj.vertices, obj.faces), r, p, i+1, ov))
        object_audit.append(dict(name=name, mesh=str(path), sha256=sha256(path), protocol=source))
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {}
    shapes = dict(exo_trajectory_world=((n,22,3),'float32'),
        exo_position_available=((n,22),'bool'), exo_position_confidence=((n,22),'float32'),
        exo_frame_points_world=((n,512,3),'float32'), exo_frame_owner=((n,512),'int32'))
    shapes.update({f'exo_{k}_counts':((n,22),'int32') for k in ('reference','visible','person','static','rigid')})
    for name, (shape,dtype) in shapes.items():
        arrays[name] = np.lib.format.open_memmap(folder/(name+'.npy'), mode='w+', dtype=dtype, shape=shape)
    np.save(folder/'exo_sensor_to_world.npy', pose, allow_pickle=False)
    w,h = camera['size'];k = camera['K']
    np.save(folder/'exo_intrinsics.npy', np.asarray([k[0,0]/w,k[1,1]/h,.5,.5], np.float32), allow_pickle=False)
    max_joint_error, max_projection_error = 0., 0.
    source_keys = ('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','expression','transl')
    for lo in range(0,n,32):
        hi = min(lo+32,n)
        params = {name:torch.from_numpy(np.asarray(fit[name],np.float32)[q[lo:hi]].copy()) for name in source_keys}
        params.update(betas=torch.from_numpy(np.broadcast_to(beta,(hi-lo,len(beta))).copy()),
            leye_pose=torch.zeros(hi-lo,3),reye_pose=torch.zeros(hi-lo,3))
        with torch.inference_mode():
            body = model(**params)
        vertices, joints = body.vertices.numpy(), body.joints[:,:22].numpy()
        expected = np.asarray(target[lo:hi,135:]).reshape(hi-lo,22,3)*2
        max_joint_error = max(max_joint_error,float(np.abs(joints-expected).max()))
        if max_joint_error > 1e-5:
            raise ValueError(f'Native body/export joint mismatch: {max_joint_error}m')
        for j,t in enumerate(range(lo,hi)):
            hit,rays = render_frame(camera,pose[t],static,[(vertices[j],model.faces)],face_labels,objects,t,valid[t])
            points,owner,parts,depth,available,confidence,reference,visible,blocked = hit
            max_projection_error = max(max_projection_error,projection_error(points,rays,camera,pose[t],owner))
            arrays['exo_position_available'][t],arrays['exo_position_confidence'][t] = available,confidence
            arrays['exo_trajectory_world'][t] = np.where(available[:,None],joints[j],0)
            arrays['exo_reference_counts'][t],arrays['exo_visible_counts'][t] = reference,visible
            for name,counts in blocked.items():
                arrays[f'exo_{name}_counts'][t] = counts
            if valid[t]:
                points,owner = sampled_environment(points,owner)
            else:
                points,owner = np.zeros((512,3),np.float32),np.full(512,-1,np.int32)
            arrays['exo_frame_points_world'][t],arrays['exo_frame_owner'][t] = points,owner
        if hi%256==0 or hi==n:
            status_update(hi,n)
    available = arrays['exo_position_available']
    summary = dict(mean_available_joints=float(available.sum(-1).mean()),
        frames_with_no_joint=int((available.sum(-1)==0).sum()), valid_frames=int(valid.sum()),
        joint_max_error_m=max_joint_error, projection_roundtrip_max_error_px=max_projection_error)
    summary.update({k+'_pixel_count':int(arrays[f'exo_{k}_counts'].sum()) for k in ('reference','visible','person','static','rigid')})
    for value in arrays.values():
        value.flush()
    del arrays
    new_hashes = {p.name:sha256(p) for p in folder.glob('*.npy')}
    serial = {key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in camera.items()}
    old_files = set(row['output_sha256'])-set(new_hashes)
    borrowed = {filename:str((Path(row['path'])/filename).resolve()) for filename in sorted(old_files)}
    payload = dict(source_input_sha256=row['input_sha256'],policy=policy,
        source_metadata_sha256=sha256(disk_metadata),source_fit_sha256=sha256(source_fit),
        static_mesh_sha256=sha256(mesh_path),objects=object_audit)
    input_digest = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    metadata = dict(row,path=str(folder.resolve()),exo_protocol=policy['protocol'],exo_camera=serial,
        exo_visibility_statistics=summary,exo_mean_available_joints=summary['mean_available_joints'],
        source_native_path=row['path'],source_native_metadata_sha256=sha256(disk_metadata),
        array_sources=borrowed,output_sha256=dict(row['output_sha256'],**new_hashes),
        input_sha256=input_digest,moving_camera_sources=payload,diagnostic_only=diagnostic)
    atomic_json(folder/'metadata.json',metadata)
    return metadata


def run(manifest, config, output, limit=None):
    import torch
    from experiments.offline_camera_retrain_v1.data import PROJECTS
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    from experiments.offline_camera_retrain_v1.export_egobody_exo import regional_faces
    # Import the dynamic renderer dependencies before sealing the source closure.
    from experiments.offline_camera_retrain_v1.native30_export import sampled_environment
    from experiments.offline_camera_retrain_v1.prepare_trumans_dynamic_v2 import load_objects
    from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene
    from scenemi.training import ensure_bounded

    ensure_bounded();torch.set_num_threads(1)
    output.mkdir(parents=True,exist_ok=True)
    policy = yaml.safe_load(config.read_text())
    rows = candidates(manifest)
    inventory = plan(manifest,output)
    if limit is not None:
        if limit < 1:raise ValueError('Pilot limit must be positive')
        rows = rows[:limit]
    source_names = ['scenemi/data/exo_export.py','scenemi/data/exo_camera.py',
        'experiments/offline_camera_retrain_v1/native30_exo.py',
        'experiments/offline_camera_retrain_v1/export_egobody_exo.py',
        'experiments/offline_camera_retrain_v1/export_rest_joints.py',
        'experiments/offline_camera_retrain_v1/native30_export.py',
        'experiments/offline_camera_retrain_v1/prepare_trumans_dynamic_v2.py',
        'experiments/offline_camera_retrain_v1/scene_visibility_v2.py',
        'experiments/offline_camera_retrain_v1/ops/viewer_geometry.py']
    source_files = source_dependencies(source_names)
    contract = dict(policy=policy,config_sha256=sha256(config),source_manifest_sha256=sha256(manifest),
        sources={str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):sha256(path) for path in sorted(source_files)},limit=limit,
        native_body_sha256=sha256(ROOT/'body_models/smplx/SMPLX_MALE.npz'),test_used=False)
    contract_path = output/'PROTOCOL.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Export source/configuration changed; use a separate output')
    atomic_json(contract_path,contract);atomic_json(output/'PLAN.json',inventory)
    if inventory['free_disk_bytes'] < inventory['estimated_output_bytes']+5*2**30:
        raise RuntimeError('Insufficient disk for full output plus 5 GiB reserve')
    now = lambda:datetime.now(timezone.utc).isoformat()
    started = time.monotonic();committed=[]
    status = dict(state='preparing',started_at_utc=now(),sequences=len(rows),
        completed_sequences=0,completed_frames=0,total_frames=sum(r['frames'] for r in rows),
        rendered_frames_this_invocation=0,reused_frames=0,
        diagnostic_only=limit is not None,training_started=False,test_used=False,
        full_dataset_reexported=False,protocol=policy['protocol'])
    atomic_json(output/'STATUS.json',status)
    model = load_model('smplx','male');labels = regional_faces(model)
    clips = {r['clip_name']:r for r in map(json.loads,
        (PROJECTS/'TRUMANS/processed/scene_expert_v1/clips.jsonl').read_text().splitlines())}
    current = None
    try:
        for row in rows:
            current = row;folder = output/'data'/row['split']/row['sequence_id']
            def progress(done,total):
                elapsed = time.monotonic()-started
                completed = status['completed_frames']+done
                rendered = status['rendered_frames_this_invocation']+done
                status.update(sequence_id=row['sequence_id'],split=row['split'],
                    current_frames=done,current_sequence_frames=total,updated_at_utc=now(),
                    elapsed_seconds=elapsed,frames_per_second=rendered/max(elapsed,1e-6),
                    eta_seconds=(status['total_frames']-completed)*elapsed/max(rendered,1))
                atomic_json(output/'STATUS.json',status)
            if (folder/'metadata.json').exists():
                metadata = json.loads((folder/'metadata.json').read_text());verify_output(metadata)
                status['reused_frames'] += row['frames']
            else:
                metadata = export_one(row,folder,policy,model,labels,clips,progress,diagnostic=limit is not None)
                status['rendered_frames_this_invocation'] += row['frames']
            committed.append(metadata)
            status.update(completed_sequences=len(committed),completed_frames=sum(r['frames'] for r in committed),updated_at_utc=now())
            with (output/'manifest.jsonl.tmp').open('w') as stream:
                for item in committed:stream.write(json.dumps(item,ensure_ascii=False)+'\n')
            (output/'manifest.jsonl.tmp').replace(output/'manifest.jsonl')
            atomic_json(output/'STATUS.json',status)
            print(json.dumps({k:status.get(k) for k in ('completed_sequences','sequences','completed_frames','total_frames','eta_seconds')}),flush=True)
        if any(sha256(ROOT/name)!=digest for name,digest in contract['sources'].items()):
            raise ValueError('Dependency changed during export')
        status.update(state='complete',updated_at_utc=now(),elapsed_seconds=time.monotonic()-started,
                      full_dataset_reexported=limit is None,eta_seconds=0.)
        atomic_json(output/'STATUS.json',status)
    except BaseException as error:
        status.update(state='failed',updated_at_utc=now(),error_type=type(error).__name__,error=str(error),
                      sequence_id=None if current is None else current['sequence_id'])
        atomic_json(output/'STATUS.json',status)
        raise
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','run'))
    parser.add_argument('--native-manifest',type=Path,required=True)
    parser.add_argument('--camera-config',type=Path,default=ROOT/'configs/trumans_exo_moving.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,help='Diagnostic full-recording pilot, never formal training data')
    args = parser.parse_args(argv)
    result = plan(args.native_manifest,args.output) if args.action=='plan' else run(
        args.native_manifest,args.camera_config,args.output,args.limit)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
