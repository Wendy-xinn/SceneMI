"""Frozen-weight, resumable full RICH VAL grid; separate from the trainer."""
from __future__ import annotations
import argparse
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
OUT = Path(__file__).resolve().parent
HERE = OUT.parents[2]
REPO = HERE.parents[1]
RUN = OUT.parents[1] / 'native30_gait_v2_ego55k_oct15'
sys.path.insert(0, str(REPO))


def now():
    return dt.datetime.now().astimezone().isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def resources():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name in ('MemAvailable', 'MemTotal'):
            values[name+'_bytes'] = int(value.split()[0])*1024
    try:
        values['main_training'] = json.loads((RUN/'continuous_training_oct16/STATUS.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    values['disk_free_bytes'] = {str(p):shutil.disk_usage(p).free for p in (OUT,Path('/mnt/e'))}
    return values


def verify_sources():
    effective = json.loads((RUN/'prepared/effective_config.json').read_text())
    continuation = json.loads((RUN/'continuous_training_oct16/PROTOCOL.json').read_text())
    for filename, expected in {**effective['code_sha256'], **continuation['expected_source_sha256']}.items():
        if sha(filename) != expected:
            raise ValueError('Sealed execution source changed: '+filename)
    if sha(RUN/'config.json') != continuation['original_config_sha256']:
        raise ValueError('Baseline config changed')
    if sha(RUN/'prepared/effective_config.json') != continuation['original_effective_config_sha256']:
        raise ValueError('Baseline preparation changed')
    return effective


def model_tensor_digest(model):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.items()):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Nonfinite frozen model tensor: '+name)
        meta = dict(name=name, dtype=str(value.dtype), shape=list(value.shape))
        digest.update(json.dumps(meta, sort_keys=True, separators=(',', ':')).encode())
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw).cast('B'))
    return digest.hexdigest()


def freeze():
    import torch
    effective = verify_sources()
    target = OUT/'checkpoint_eval.pt'
    meta = OUT/'FROZEN_CHECKPOINT.json'
    if target.exists() or meta.exists():
        if not target.exists() or not meta.exists():
            raise ValueError('Incomplete existing frozen checkpoint')
        record = json.loads(meta.read_text())
        if sha(target) != record['evaluation_checkpoint_sha256']:
            raise ValueError('Frozen evaluation checkpoint changed')
        return record
    # Wait for the requested milestone to be atomically published; do not save,
    # stop, or change the trainer. Its 15k validation precedes publication.
    while True:
        main = json.loads((RUN/'continuous_training_oct16/STATUS.json').read_text())
        if main.get('latest_committed_step', 0) >= 15000:
            break
        os.kill(main['worker_pid'], 0)
        write_json(OUT/'STATUS.json', dict(status='waiting_checkpoint', pid=os.getpid(),
            requested_minimum_step=15000, checked_at=now(), completed_predictions=0,
            planned_predictions=412, resources=resources()))
        time.sleep(5)
    # Atomic trainer publication + hard link pins one inode across later saves.
    pinned = OUT/'checkpoint_full.pinned.pt'
    if pinned.exists():
        raise ValueError('Unresolved checkpoint freeze from a previous process')
    os.link(RUN/'training/last.pt', pinned)
    try:
        full_sha = sha(pinned)
        state = torch.load(pinned, map_location='cpu', mmap=True, weights_only=False)
        if state['effective_config'] != effective or state['planned_steps'] != 55000:
            raise ValueError('Frozen checkpoint differs from baseline protocol')
        source_model_digest = model_tensor_digest(state['model'])
        temporary = target.with_suffix('.tmp')
        torch.save(dict(model=state['model'], step=state['step'], effective_config=effective,
                        planned_steps=state['planned_steps'], evaluation_only=True), temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        temporary.replace(target)
        compact_state = torch.load(target, map_location='cpu', mmap=True, weights_only=False)
        compact_model_digest = model_tensor_digest(compact_state['model'])
        if source_model_digest != compact_model_digest:
            raise ValueError('Compact/source model tensor mismatch')
        del compact_state
        record = dict(source_model_tensor_sha256=source_model_digest,
            compact_model_tensor_sha256=compact_model_digest, direct_source_tensor_digest_comparison=True,
            status='frozen', frozen_at=now(), source_checkpoint=str(RUN/'training/last.pt'),
            source_checkpoint_sha256=full_sha, checkpoint_step=int(state['step']),
            evaluation_checkpoint=str(target), evaluation_checkpoint_sha256=sha(target),
            evaluation_checkpoint_bytes=target.stat().st_size, evaluation_only=True,
            original_optimizer_and_RNG_untouched=True,
            copy_policy='One atomic hard-link snapshot of trainer last.pt, then retain only model/config/step; pinned full copy unlinked after compact publication.')
        write_json(meta, record)
        del state
        return record
    finally:
        pinned.unlink(missing_ok=True)


def summarize():
    path = OUT/'report.py'
    if path.exists():
        try:
            module('expanded_validation_report', path).main(OUT)
        except Exception as exc:
            # Reporting does not change sampling or invalidate committed rows.
            print(json.dumps(dict(event='report_failed',error=repr(exc))),flush=True)


def run():
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, window_options
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, sample_motion
    from experiments.offline_camera_retrain_v1.native30_validation import raw_metrics
    from experiments.offline_camera_retrain_v1.standard_motion_metrics import standard_motion_metrics, METRIC_PROTOCOL
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
    from experiments.offline_camera_retrain_v1 import native30_surface_cache as surfaces, native30_memory as memory
    from experiments.offline_camera_retrain_v1.native30_surface_cache import CanonicalSurfaceCache
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    frozen = freeze()
    effective = verify_sources()
    plan = json.loads((OUT/'PLAN.json').read_text())
    for filename, expected in plan['source_sha256'].items():
        if sha(filename) != expected:
            raise ValueError('Evaluation-plan dependency changed: '+filename)
    cfg = effective['config']
    if plan['frames'] != cfg['frames'] or plan['ddim_steps'] != cfg['sampling']['ddim_steps']:
        raise ValueError('Validation settings differ from training monitoring')
    protocol = dict(status='running', started_at=now(), pid=os.getpid(),
        checkpoint_step=frozen['checkpoint_step'], checkpoint_sha256=frozen['evaluation_checkpoint_sha256'],
        plan_sha256=sha(OUT/'PLAN.json'), runner_sha256=sha(Path(__file__)),
        effective_config_sha256=sha(RUN/'prepared/effective_config.json'),
        planned_windows=len(plan['plan']), planned_predictions=len(plan['plan'])*2,
        regimes=plan['regimes'], frames=192, fps=30, ddim_steps=20, scene='on', mode='ego',
        resources_before=resources(), memory_fraction=.25, cpu_threads=2,
        training_unchanged=True, optimizer_updates=0, test_used=False,
        scope='Every RICH validation/ego index window: 206 windows from 27 eligible sequences; 28 source sequences include one short excluded source. Not TEST or an assumed external-paper protocol.',
        metric_protocol={**METRIC_PROTOCOL, 'fps':30,
            'mpjve_mm_s':'World joint first-difference velocity error at explicit 30Hz, mm/s; no alignment',
            'acceleration_error_mm_s2':'World joint second-difference acceleration error at explicit 30Hz, mm/s^2; no alignment'},
        timing_scope='Concurrent single GPU: evaluation and training throughput can differ from isolated measurements.')
    proto_path = OUT/'PROTOCOL.json'
    if proto_path.exists():
        old = json.loads(proto_path.read_text())
        for key in ('checkpoint_sha256','plan_sha256','runner_sha256','effective_config_sha256'):
            if old[key] != protocol[key]:
                raise ValueError('Resume protocol changed: '+key)
        protocol['first_started_at'] = old.get('first_started_at', old['started_at'])
    write_json(proto_path, protocol)
    # Read existing runtime caches through independent CRC-checked bytes. A
    # corrupt cache stops only this evaluator; never quarantine shared files.
    io_adapter = module('validation_crc_reads', RUN.parent/'native30_final_review_oct16/io_recovery.py')
    surfaces.np = io_adapter.NumpyProxy(surfaces.np, OUT/'IO_READS.jsonl')
    memory.np = io_adapter.NumpyProxy(memory.np, OUT/'IO_READS.jsonl')
    status = dict(status='initializing', pid=os.getpid(), checkpoint_step=frozen['checkpoint_step'],
        completed_predictions=0, planned_predictions=protocol['planned_predictions'], checked_at=now())
    write_json(OUT/'STATUS.json', status)
    data = Native30Data(HERE/cfg['train_manifest'], 'validation', cfg['seed']+1,
        frames=cfg['frames'], memory_protocol=cfg['scene']['memory_protocol'],
        quality_config=cfg['gt_reliability'], exo_max_empty_frames=cfg['exo_max_empty_frames'], **window_options(cfg))
    # Derived surface writes stay within this evaluation. Shared static maps
    # are read-only; a missing map is reconstructed in RAM by original code.
    data.runtime_cache_root = OUT/'native30_runtime_cache_v2'
    shared_cache_path = data.scene_memory._cache_path
    def read_only_map_path(*args):
        path = shared_cache_path(*args)
        return path if path is not None and path.exists() else None
    data.scene_memory._cache_path = read_only_map_path
    data.surface_cache = CanonicalSurfaceCache(data, limit=1)
    data.scene_memory.cached_streams = 1
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(),
        json.loads((HERE/cfg['mount_defaults']).read_text()), cfg['observation']['input_noise'])
    noisy_builder = Native30ObservationBuilder(builder.skin_cache, builder.mounts,
        dict(builder.noise_config, clean_probability=0.))
    footpath = HERE/cfg['run_control']['foot_metrics_source']
    if sha(footpath) != cfg['run_control']['foot_metrics_sha256']:
        raise ValueError('Foot-metric implementation changed')
    feet = module('expanded_frozen_foot_metrics', footpath)
    checkpoint = torch.load(OUT/'checkpoint_eval.pt', map_location='cpu', mmap=True, weights_only=False)
    model = make_model(cfg).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    done = {}
    rowpath = OUT/'rows.jsonl'
    if rowpath.exists():
        for line in rowpath.read_text().splitlines():
            row = json.loads(line)
            key=(row['case'],row['regime'])
            if key in done or row['checkpoint_step'] != frozen['checkpoint_step']:
                raise ValueError('Duplicate or mixed-checkpoint evaluation row')
            entry=plan['plan'][row['case']]
            if any(row[k] != entry[k] for k in ('sequence_id','dataset','recording','start','case_seed')):
                raise ValueError('Existing row differs from frozen evaluation plan')
            artifact=OUT/row['motion_file']
            if not artifact.exists() or sha(artifact) != row['motion_sha256']:
                raise ValueError('Existing motion file missing or changed')
            done[key]=row
    (OUT/'motions').mkdir(exist_ok=True)
    started=time.monotonic()
    try:
        for entry in plan['plan']:
            case=entry['case']
            if all((case,r) in done for r in plan['regimes']):
                continue
            matches=[i for i,(m,_) in enumerate(data.members[entry['group']]) if m['path']==entry['path']]
            if len(matches)!=1:
                raise ValueError('Selected source not uniquely available')
            sample,identity=data.sample(entry['group'],'ego',member=matches[0],start=entry['start'],yaw=0)
            if identity['sequence_id']!=entry['sequence_id'] or identity['source_start_30fps']!=entry['source_first']:
                raise ValueError('Sample identity differs from fixed plan')
            truth=torch.from_numpy(sample['motion'])[None]
            rest=torch.from_numpy(sample['rest'])[None]
            gtj=forward_kinematics(truth,rest)
            gtpath=float(gtj[:,:,0][:,:,(0,2)].diff(dim=1).norm(dim=-1).sum())
            gtspeed=float(gtj.diff(dim=1).norm(dim=-1).mean()*30)
            grot=rotation_from_6d(truth[...,3:9])
            glocal=torch.einsum('btji,btkj->btki',grot,gtj-gtj[:,:,:1])[:,:,(10,11)]
            gtamp=(glocal-glocal.mean(1,keepdim=True)).square().sum(-1).mean(1).sqrt()[0]
            true_speed=gtj.diff(dim=1)[:,:,(10,11)].norm(dim=-1)*30
            low=true_speed<.2
            for regime in plan['regimes']:
                if (case,regime) in done:
                    continue
                status.update(status='evaluating', current_case=case, current_dataset=entry['dataset'],
                    current_regime=regime, completed_predictions=len(done), checked_at=now(),resources=resources())
                write_json(OUT/'STATUS.json',status)
                tick=time.monotonic()
                local=builder if regime=='clean' else noisy_builder
                observed,scene,asset=local.build(data,sample,identity,'ego',(15,),
                    noise_rng=None if regime=='clean' else np.random.default_rng(entry['case_seed']))
                batch=collate_observations([(observed,scene,asset,'ego')])
                known=model_inputs(cfg,batch)
                if any(k in known for k in ('motion','joints','contact_target','gt_position_weight')):
                    raise ValueError('Targets entered inference payload')
                preparation=time.monotonic()-tick
                torch.cuda.synchronize();tick=time.monotonic()
                with torch.no_grad():
                    prediction=sample_motion(model,known,cfg,steps=20,seed=entry['case_seed'])
                torch.cuda.synchronize();ddim=time.monotonic()-tick
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError('Nonfinite expanded validation output')
                pred=prediction.detach().cpu()
                conditioned=torch.from_numpy(observed['observation_meta'][...,0]>0)[None]
                cpu_batch=dict(motion=truth,rest=rest)
                with torch.no_grad():
                    raw=raw_metrics(pred,cpu_batch,conditioned)
                    extra=feet.extra_metrics(pred,cpu_batch)
                    pj=forward_kinematics(pred,rest)
                    standard = standard_motion_metrics(pj[0].numpy(), gtj[0].numpy(), fps=30.)
                    frames, joints = pj.shape[1:3]
                    benchmark_counts = {k:(frames*joints) for k in standard}
                    benchmark_counts.update(root_position_error_mm=frames,
                        mpjve_mm_s=(frames-1)*joints,
                        acceleration_error_mm_s2=(frames-2)*joints)
                    benchmark = {k:dict(value=v,count=benchmark_counts[k]) for k,v in standard.items()}
                    ps=pj.diff(dim=1)[:,:,(10,11)].norm(dim=-1)*30
                    root_error=pj[:,:,0]-gtj[:,:,0]
                    extra.update(gt_low_speed_count=int(low.sum()),gt_foot_velocity_samples=int(low.numel()),
                        gt_foot_speed_on_gt_low_speed_m_s=float(true_speed[low].mean()) if low.any() else None,
                        pred_speed_over_point2_on_gt_low_speed_count=int(((ps>.2)&low).sum()),
                        root_terminal_horizontal_error_change_m=float((root_error[:,-1]-root_error[:,0])[:,(0,2)].norm(dim=-1).mean()))
                motion_file=f'motions/case{case:02d}_{regime}.npz'
                target=OUT/motion_file;temporary=target.with_suffix('.tmp')
                with temporary.open('wb') as f:
                    np.savez_compressed(f,prediction=pred[0].numpy(),truth=sample['motion'],rest=sample['rest'],
                        observations=observed['trajectory'],observation_meta=observed['observation_meta'],
                        reference=known['reference_motion'][0].detach().cpu().numpy(),
                        identity_json=np.array(json.dumps(identity)),step=np.array(frozen['checkpoint_step']))
                    f.flush();os.fsync(f.fileno())
                temporary.replace(target)
                row={k:entry[k] for k in ('case','dataset','recording','group','sequence_id','start','source_first','case_seed')}
                row.update(regime=regime,checkpoint_step=frozen['checkpoint_step'],checked_at=now(),
                    metrics=raw,extra_metrics=extra,benchmark_metrics=benchmark,preparation_seconds=preparation,ddim_seconds=ddim,
                    activity=dict(gt_root_horizontal_path_m=gtpath,gt_mean_joint_speed_m_s=gtspeed,
                        gt_local_foot_RMS_left_right_m=gtamp.tolist(),
                        moving_proxy=gtpath>=.5,near_static_proxy=gtpath<.25 and float(gtamp.max())<.08),
                    motion_file=motion_file,motion_sha256=sha(target),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
                with rowpath.open('a') as f:
                    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                done[(case,regime)]=row
                print(json.dumps(dict(event='evaluated',case=case,dataset=entry['dataset'],regime=regime,
                    mpjpe_mm=raw['mpjpe_mm']['value'],stance_speed=raw['foot_speed_during_GT_stance_m_s']['value'],
                    ddim_seconds=ddim,completed=len(done),step=frozen['checkpoint_step'])),flush=True)
                del batch,known,prediction,pred,observed,scene,asset
                gc.collect();torch.cuda.empty_cache()
                status.update(completed_predictions=len(done),checked_at=now(),elapsed_seconds=time.monotonic()-started)
                write_json(OUT/'STATUS.json',status)
                summarize()
                if resources()['MemAvailable_bytes']<3*1024**3:
                    raise MemoryError('Evaluator stopped to preserve 3GiB host memory')
                time.sleep(.5)
            # Drop mapped source references for completed recordings. They are
            # read-only mirrors and can be reopened without changing values.
            data.arrays.clear()
        verify_sources()
        status.update(status='completed',checked_at=now(),elapsed_seconds=time.monotonic()-started,
                      completed_predictions=len(done),resources_after=resources())
        protocol.update(status='completed',finished_at=now(),elapsed_seconds=time.monotonic()-started,
                        resources_after=resources())
        write_json(OUT/'STATUS.json',status);write_json(proto_path,protocol)
        summarize()
    except BaseException as exc:
        status.update(status='failed',checked_at=now(),error=repr(exc),completed_predictions=len(done),resources_after=resources())
        write_json(OUT/'STATUS.json',status)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    if args.freeze_only:
        print(json.dumps(freeze(),ensure_ascii=False,indent=2))
    else:
        try:
            run()
        except BaseException as exc:
            path=OUT/'STATUS.json'
            state=json.loads(path.read_text()) if path.exists() else {}
            state.update(status='failed',error=repr(exc),checked_at=now())
            write_json(path,state)
            raise
