"""Independent RICH TEST preparation, frozen evaluation and CPU verification.

The model is selected on VAL before this command. TEST never enters the
training manifest, sampling schedule, coefficient fitting or model selection.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
PROJECTS = BASE.parents[2]
DEFAULT_OUT = BASE / 'runs/rich_test_ego41k'
DEFAULT_DATA = Path('/mnt/d/Projects/SceneMI/cache/rich_test_native30_v1')
PROTOCOL = 'rich-test-all-subjects-native30-ego-v1'


def now():
    return dt.datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    from .dependencies import sha256
    return sha256(path)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_inventory(root=None):
    """Index released TEST PKLs, retaining every target and the original clock."""
    from experiments.offline_camera_retrain_v1.native30_splits import SPLIT_PROTOCOL
    root = Path(root) if root is not None else PROJECTS / 'RICH/extracted/test_body'
    if not root.is_dir():
        return []
    result = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        people = collections.defaultdict(set)
        for frame in folder.iterdir():
            if frame.is_dir() and frame.name.isdigit():
                for path in frame.glob('*.pkl'):
                    require(path.stem.isdigit(), 'Non-numeric RICH subject: ' + str(path))
                    people[path.stem].add(int(frame.name))
        if not people:
            continue
        all_ids = set.union(*people.values())
        complete = set.intersection(*people.values())
        first, last = min(all_ids), max(all_ids) + 1
        missing = sorted(set(range(first, last)) - complete)
        scene = folder.name.split('_')[0]
        scan = ('scan_chair_scene_camcoord.ply' if 'chair' in folder.name else 'scan_yoga_scene_camcoord.ply') if scene == 'LectureHall' else 'scan_camcoord.ply'
        if scene == 'Gym' and 'cooking' in folder.name:
            scan = 'scan_table_camcoord.ply'
        for subject in sorted(people):
            result.append(dict(dataset='rich', group='rich', split='test', split_protocol=SPLIT_PROTOCOL,
                sequence_id=folder.name + '__subject_' + subject, source_sequence_id=folder.name,
                subject_id=subject, subject_ids=sorted(people), scene=scene,
                raw_first=first, raw_last_exclusive=last,
                missing_any_subject_frame_ids=missing, scan_filename=scan,
                allowed_modes=['ego'], skip_contact_labels=True, rich_test_protocol=PROTOCOL))
    return result


def read_test_fits(paths, digest):
    """Use explicit masked sentinels for missing fits; never interpolate them."""
    import numpy as np
    fits, present = [], []
    for path in paths:
        path = Path(path)
        digest.update(str(path).encode())
        exists = path.is_file()
        present.append(exists)
        if exists:
            raw = path.read_bytes()
            digest.update(raw)
            fits.append(pickle.loads(raw, encoding='latin1'))
        else:
            digest.update(b'EXPLICIT_MISSING_FIT')
            fits.append(None)
    available = next((fit for fit in fits if fit is not None), None)
    require(available is not None, 'Target/occluder has no released fit in its original clock')
    sentinel = {key:np.zeros_like(value) for key, value in available.items()}
    sentinel['betas'] = np.array(available['betas'], copy=True)
    return [fit if fit is not None else sentinel for fit in fits], np.asarray(present, bool)


def dependency_hashes():
    from .dependencies import dependency_closure
    from vit_pytorch import vit
    entries = [Path(__file__), BASE/'ops/__main__.py', BASE/'export_native30_all.py', BASE/'native30_data.py',
        BASE/'prepare_native30.py', BASE/'native30_refinement.py', BASE/'native30_observations.py',
        BASE/'native30_assets.py', BASE/'ops/support_metrics.py', BASE/'ops/foot_trial_evaluation.py',
        BASE/'ops/foot_trial_recheck.py']
    extra = [PROJECTS/'diffusion-motion-inbetweening/sample/virtual_head_camera.py', Path(vit.__file__)]
    return dependency_closure(entries, extra_files=extra,
        import_roots=[PROJECTS/'diffusion-motion-inbetweening'])


def verify_seal(plan):
    for path, expected in plan['dependency_sha256'].items():
        require(sha(path) == expected, 'Sealed TEST dependency changed: ' + path)
    require(sha(plan['checkpoint']) == plan['checkpoint_sha256'], 'Frozen TEST model changed')
    require(sha(plan['training_config']) == plan['training_config_sha256'], 'Original training config changed')


def prepare(args):
    import numpy as np
    from experiments.offline_camera_retrain_v1.export_native30_all import isolated_worker, initialize_launcher
    from experiments.offline_camera_retrain_v1.native30_window_index import valid_window_grid
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    out, output = args.out.resolve(), args.data_output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    sources = source_inventory()
    require(len(sources) == 53 and len({s['source_sequence_id'] for s in sources}) == 52,
        'Expected all 52 downloaded TEST recordings and all 53 subject streams')
    require(args.workers >= 1, 'Workers must be positive')
    require(shutil.disk_usage(output).free > sum(s['raw_last_exclusive']-s['raw_first'] for s in sources)*10000+20*2**30,
        'TEST cache destination needs estimated bytes plus 20GiB reserve')
    require(sha(args.checkpoint) == args.checkpoint_sha256, 'Declared VAL-selected model digest differs')
    import torch
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    require(state['step'] == args.expected_step, 'VAL-selected model step differs')
    cfg = state['effective_config']['config']
    require(read(args.config)==cfg,'Declared original training config differs from frozen model')
    require(cfg['fps']==30 and cfg['frames']==192, 'Expected native192 model')
    require(cfg['physical_guidance']['enabled'], 'The frozen full system uses physical guidance ON')
    del state
    pinned=out/'checkpoint.pt'
    if not pinned.exists():
        os.link(args.checkpoint.resolve(),pinned)
    require(sha(pinned)==args.checkpoint_sha256,'Pinned TEST checkpoint changed')
    original_manifest=BASE/cfg['train_manifest']
    original_test=[json.loads(line) for line in original_manifest.read_text().splitlines()
        if json.loads(line)['split']=='test' and json.loads(line)['dataset']!='rich']
    static = dict(protocol=PROTOCOL, checkpoint=str(pinned),
        original_checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=args.checkpoint_sha256, checkpoint_step=args.expected_step,
        training_config=str(args.config.resolve()), training_config_sha256=sha(args.config),
        independent_test_manifest=str(output/'manifest.jsonl'), source_inventory=sources,
        original_manifest=str(original_manifest),original_manifest_sha256=sha(original_manifest),
        supporting_other_TEST_sources=len(original_test),
        scene_asset_audit_sha256=sha(out/'SCENE_ASSET_AUDIT.json'),
        frames=192, fps=30, stride=64, ddim_steps=20, physics='on', learned_scene='on',
        regimes=['clean','noisy'], model_selection='Fixed on VAL before any TEST result; no TEST-driven tuning',
        contact_labels='absent; official HSC is not required by these motion metrics',
        source_scope='All released subject streams. Windows crossing any missing target/occluder fit are excluded; original source clock retained.',
        scene_scope='Only this target camera recording first-hit observations; target skin excluded; supplied-recording static union.',
        paper_comparison='native192/S64 with observed scene and known body profile; not the AMASS-only native128 CPF paper protocol',
        dependency_sha256=dependency_hashes())
    plan_path = out/'PLAN.json'
    if plan_path.exists():
        old = read(plan_path)
        require(all(old.get(k)==v for k,v in static.items()), 'Immutable TEST protocol changed')
    else:
        write(plan_path, dict(static, created_at=now(), status='preparing'))
    started = time.monotonic()
    manifest = output/'manifest.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines()] if manifest.exists() else []
    done = {(r['split'],r['bundle_id']):r for r in rows}
    for row in original_test:
        done[(row['split'],row['bundle_id'])]=row
    cpus = sorted(os.sched_getaffinity(0))
    with ThreadPoolExecutor(max_workers=args.workers, initializer=initialize_launcher,
            initargs=(cpus,args.workers,iter(range(args.workers)))) as pool:
        futures = {pool.submit(isolated_worker, row, str(output)):row for row in sources}
        for future in as_completed(futures):
            row = future.result()
            done[(row['split'],row['bundle_id'])] = row
            selected = original_test+[done[('test',s['sequence_id'])] for s in sources if ('test',s['sequence_id']) in done]
            temp = manifest.with_suffix('.tmp')
            temp.write_text(''.join(json.dumps(r)+'\n' for r in selected))
            temp.replace(manifest)
            write(out/'STATUS.json', dict(status='exporting', checked_at=now(), pid=os.getpid(),
                exported_sources=len(selected)-len(original_test), planned_sources=53, elapsed_s=time.monotonic()-started))
    data = Native30Data(manifest, 'test', frames=192, window_stride=64,
        memory_protocol=cfg['scene']['memory_protocol'], quality_config=cfg['gt_reliability'])
    cases, coverage = [], []
    for index, (meta, starts) in enumerate(data.members['rich']):
        valid = np.asarray(data.load(meta['path'],'observation_valid'))
        require(np.array_equal(starts,valid_window_grid(valid,192,64)), 'TEST legal grid mismatch')
        covered = np.zeros(meta['frames'], bool)
        for first in starts:
            first = int(first)
            source_first = meta['source_first']+first
            key = json.dumps([PROTOCOL,meta['source_sequence_id'],meta['subject_id'],source_first], separators=(',',':'))
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')
            cases.append(dict(case=len(cases), path=meta['path'], sequence_id=meta['sequence_id'],
                recording=meta['source_sequence_id'],subject_id=meta['subject_id'], member=index,
                start=first, source_first=source_first, case_seed=seed,
                metadata_sha256=sha(Path(meta['path'])/'metadata.json')))
            covered[first:first+192] = True
        coverage.append(dict(sequence_id=meta['sequence_id'],recording=meta['source_sequence_id'],
            subject_id=meta['subject_id'],frames=meta['frames'],windows=len(starts),
            covered_frames=int(covered.sum()), invalid_frames=int((~valid).sum())))
    require(len(cases)==335, 'Expected 335 legal all-subject windows after explicit missing-person exclusion')
    plan = read(plan_path)
    plan.update(status='prepared', cases=cases, coverage=coverage, unique_windows=len(cases),
        planned_predictions=2*len(cases), independent_manifest_sha256=sha(manifest),
        prepared_at=now(), preparation_elapsed_s=time.monotonic()-started)
    np.save(out/'test_ego_windows.npy', np.asarray([[e['member'],e['start']] for e in cases],np.int64))
    plan['window_index_sha256']=sha(out/'test_ego_windows.npy')
    write(plan_path,plan)
    write(out/'STATUS.json',dict(status='prepared',checked_at=now(),planned_predictions=len(cases)*2,
        completed_predictions=0,source_streams=53,recordings=52,unique_windows=len(cases)))
    print(json.dumps(dict(status='prepared',windows=len(cases),predictions=2*len(cases))),flush=True)


def summarize(out, plan, rows):
    groups = {}
    for regime in plan['regimes']:
        selected = [r for r in rows if r['regime']==regime]
        metrics = {}
        for key in sorted({key for r in selected for key in r['metrics']}):
            valid = [r for r in selected if isinstance(r['metrics'].get(key),(int,float))]
            if not valid:
                continue
            by_recording = collections.defaultdict(list)
            for row in valid:
                by_recording[row['recording']].append(row['metrics'][key])
            values = [r['metrics'][key] for r in valid]
            metrics[key] = dict(window_mean=sum(values)/len(values), windows=len(values),
                recording_macro=sum(sum(v)/len(v) for v in by_recording.values())/len(by_recording))
        groups[regime] = dict(predictions=len(selected),recordings=len({r['recording'] for r in selected}),metrics=metrics)
    summary = dict(status='complete' if len(rows)==plan['planned_predictions'] else 'partial',
        checkpoint_step=plan['checkpoint_step'], checkpoint_sha256=plan['checkpoint_sha256'],
        completed_predictions=len(rows),planned_predictions=plan['planned_predictions'],groups=groups,
        foot_proxy='GT speed <0.2m/s is a kinematic proxy, not official HSC contact.',
        aggregation='Each window metric equally weighted; also equal-recording macro. All windows have 192 frames; foot low-speed denominators can differ.')
    write(out/'SUMMARY.json',summary)
    lines = ['# RICH TEST 冻结模型评估', '',
        f"状态：{summary['status']}，完成 {len(rows)}/{plan['planned_predictions']} 次预测。模型为 VAL 预先选定的 {plan['checkpoint_step']} 步；TEST 不用于选型或调参。", '',
        '52 个录制、53 个目标人体流；30Hz、192帧、stride64、DDIM20、可见场景和物理修正开启。335 个合法窗口，每窗 clean/noisy 各一次。', '',
        '原始源时钟保留；跨任一目标/遮挡者缺标帧的窗口剔除。没有读取 HSC 标签，脚部低速期只是运动学代理。', '',
        '| 指标（等窗口均值） | clean | noisy |','|---|---:|---:|']
    for key,label in [('w_mpjpe_mm','World MPJPE，mm'),('pa_mpjpe_mm','PA-MPJPE，mm'),
            ('foot_joint_horizontal_speed_gt_low_m_s','GT低速期水平脚速，m/s'),
            ('foot_joint_y_abs_error_gt_low_mm','GT低速期脚高绝对误差，mm')]:
        values = [groups[r]['metrics'].get(key,{}).get('window_mean') for r in plan['regimes']]
        lines.append('| '+label+' | '+' | '.join('—' if v is None else f'{v:.3f}' for v in values)+' |')
    lines += ['', '同时记录逐窗口指标及等录制宏平均。当前输入包含合成头相机轨迹、可见几何和已知人体 profile；它不是 AMASS-only / 128帧 CPF 的论文比较协议。',
        '', '全部数组保存后需 CPU_RECHECK.json 通过，方可作为最终核验结果。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')


def metrics_for(saved):
    from .support_metrics import support_metrics
    from .foot_trial_evaluation import extended_metrics
    result = support_metrics(saved['prediction'],saved['truth'],saved['rest'],saved['observations'],saved['observation_meta'])
    result.update(extended_metrics(saved['prediction'],saved['truth'],saved['rest']))
    return result


def evaluate(args):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, sample_motion
    from experiments.offline_camera_retrain_v1.native30_surface_cache import CanonicalSurfaceCache
    out = args.out.resolve()
    plan = read(out/'PLAN.json')
    require(plan['status']=='prepared', 'TEST preparation incomplete')
    verify_seal(plan)
    require(sha(plan['independent_test_manifest'])==plan['independent_manifest_sha256'],'TEST manifest changed')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    state = torch.load(plan['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    require(state['step']==plan['checkpoint_step'],'Checkpoint step differs')
    cfg = state['effective_config']['config']
    data = Native30Data(plan['independent_test_manifest'],'test',frames=192,window_stride=64,
        memory_protocol=cfg['scene']['memory_protocol'],quality_config=cfg['gt_reliability'])
    data.runtime_cache_root = out/'runtime_cache'
    data.scene_memory.cache_root = data.runtime_cache_root/'static_maps'
    data.scene_memory.cached_streams = 1
    data.canonical_surface_cache = True
    data.surface_cache = CanonicalSurfaceCache(data,limit=1)
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(),read(BASE/cfg['mount_defaults']),cfg['observation']['input_noise'])
    noisy = Native30ObservationBuilder(builder.skin_cache,builder.mounts,dict(builder.noise_config,clean_probability=0.))
    path = out/'rows.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(r['case'],r['regime']):r for r in rows}
    expected = {(e['case'],regime) for e in plan['cases'] for regime in plan['regimes']}
    require(len(done)==len(rows) and set(done).issubset(expected),'Duplicate or unexpected TEST prediction')
    for row in rows:
        require(row['checkpoint_sha256']==plan['checkpoint_sha256'] and sha(row['artifact'])==row['artifact_sha256'], 'Existing TEST row changed')
    write(out/'STATUS.json',dict(status='waiting_gpu',checked_at=now(),pid=os.getpid(),completed_predictions=len(rows),planned_predictions=len(expected)))
    with Path('/tmp/scenemi-native30-gpu.lock').open('a') as gpu_lock:
        fcntl.flock(gpu_lock,fcntl.LOCK_EX)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        model = make_model(cfg).cuda().eval()
        model.load_state_dict(state['model'],strict=True)
        from .checkpoint_guardian import tensor_digest
        before = tensor_digest(model.state_dict())
        model.guidance_enabled = True
        del state
        started = time.monotonic()
        count = 0
        for entry in plan['cases']:
            require(sha(Path(entry['path'])/'metadata.json')==entry['metadata_sha256'],'TEST source metadata changed')
            members = [i for i,(m,_) in enumerate(data.members['rich']) if m['path']==entry['path']]
            require(len(members)==1,'Nonunique TEST source')
            sample,identity=data.sample('rich','ego',member=members[0],start=entry['start'],yaw=0)
            # Supervision is retained only for the subsequent CPU metric call.
            sensors = {k:v for k,v in sample.items() if k not in ('motion','joints','contact_target','contact_valid') and not k.startswith('gt_')}
            for regime in plan['regimes']:
                if (entry['case'],regime) in done:
                    continue
                observed,scene,asset=(builder if regime=='clean' else noisy).build(data,sensors,identity,'ego',(15,),
                    noise_rng=None if regime=='clean' else np.random.default_rng(entry['case_seed']))
                known=model_inputs(cfg,collate_observations([(observed,scene,asset,'ego')]))
                require(not {'motion','joints','contact_target','contact_valid','support_points'} & set(known),'GT leaked into TEST inference')
                fingerprint=hashlib.sha256()
                for key,value in sorted(known.items()):
                    if isinstance(value,torch.Tensor):
                        array=value.detach().cpu().contiguous().numpy()
                        fingerprint.update(json.dumps([key,str(array.dtype),list(array.shape)]).encode())
                        fingerprint.update(array.tobytes())
                write(out/'STATUS.json',dict(status='sampling',pid=os.getpid(),checked_at=now(),current_case=entry['case'],regime=regime,
                    completed_predictions=len(rows),planned_predictions=len(expected),elapsed_s=time.monotonic()-started))
                with torch.no_grad():
                    result=sample_motion(model,known,cfg,steps=20,seed=entry['case_seed'],use_scene=True)
                artifact=out/'motions'/f"case{entry['case']:04d}_{regime}.npz"
                artifact.parent.mkdir(exist_ok=True)
                with artifact.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream,prediction=result[0].detach().cpu().numpy(),truth=sample['motion'],rest=sample['rest'],
                        observations=observed['trajectory'],observation_meta=observed['observation_meta'])
                    stream.flush();os.fsync(stream.fileno())
                artifact.with_suffix('.tmp').replace(artifact)
                with np.load(artifact,allow_pickle=False) as saved:
                    metrics=metrics_for(saved)
                row=dict(case=entry['case'],recording=entry['recording'],subject_id=entry['subject_id'],sequence_id=entry['sequence_id'],
                    regime=regime,seed=entry['case_seed'],source_first=entry['source_first'],checkpoint_sha256=plan['checkpoint_sha256'],
                    checkpoint_step=plan['checkpoint_step'],input_sha256=fingerprint.hexdigest(),
                    artifact=str(artifact),artifact_sha256=sha(artifact),metrics=metrics,checked_at=now())
                with path.open('a') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                rows.append(row);done[entry['case'],regime]=row;count+=1
                summarize(out,plan,rows)
                if args.max_predictions and count>=args.max_predictions:
                    require(before==tensor_digest(model.state_dict()),'TEST smoke changed frozen model tensors')
                    write(out/'VERIFY_PARTIAL.json',dict(status='passed_partial',optimizer_updates=0,
                        frozen_model_tensors=True,model_tensor_sha256=before,completed_predictions=len(rows)))
                    write(out/'STATUS.json',dict(status='partial_requested',completed_predictions=len(rows),planned_predictions=len(expected),checked_at=now()))
                    return
        require(before==tensor_digest(model.state_dict()),'TEST changed frozen model tensors')
        write(out/'VERIFY.json',dict(status='passed_complete',optimizer_updates=0,frozen_model_tensors=True,
            model_tensor_sha256=before,completed_predictions=len(rows),planned_predictions=len(expected),checked_at=now()))
    write(out/'STATUS.json',dict(status='sampling_completed_pending_cpu',completed_predictions=len(rows),planned_predictions=len(expected),checked_at=now()))


def recheck(args):
    import numpy as np
    import torch
    from .foot_trial_recheck import independent_metrics
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
    from experiments.offline_camera_retrain_v1.native30_surface_cache import CanonicalSurfaceCache
    out=args.out.resolve();plan=read(out/'PLAN.json')
    require(not torch.cuda.is_initialized(),'CPU recheck must run without CUDA initialization')
    verify_seal(plan)
    cfg=read(plan['training_config'])
    data=Native30Data(plan['independent_test_manifest'],'test',frames=192,window_stride=64,
        memory_protocol=cfg['scene']['memory_protocol'],quality_config=cfg['gt_reliability'])
    data.runtime_cache_root=out/'runtime_cache';data.scene_memory.cache_root=data.runtime_cache_root/'static_maps'
    data.scene_memory.cached_streams=1;data.canonical_surface_cache=True
    data.surface_cache=CanonicalSurfaceCache(data,limit=1)
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),read(BASE/cfg['mount_defaults']),cfg['observation']['input_noise'])
    noisy=Native30ObservationBuilder(builder.skin_cache,builder.mounts,dict(builder.noise_config,clean_probability=0.))
    rows=[json.loads(line) for line in (out/'rows.jsonl').read_text().splitlines()]
    identities={(e['case'],regime):e for e in plan['cases'] for regime in plan['regimes']}
    require(len({(r['case'],r['regime']) for r in rows})==len(rows),'Duplicate TEST result')
    maximum=independent_maximum=0.
    for row in rows:
        entry=identities[(row['case'],row['regime'])]
        require(all(row[k]==entry[k] for k in ('recording','subject_id','sequence_id','source_first')),'TEST result identity differs')
        require(sha(row['artifact'])==row['artifact_sha256'],'Changed TEST prediction artifact')
        sample,identity=data.sample('rich','ego',member=entry['member'],start=entry['start'],yaw=0)
        sensors={k:v for k,v in sample.items() if k not in ('motion','joints','contact_target','contact_valid') and not k.startswith('gt_')}
        observed,_,_=(builder if row['regime']=='clean' else noisy).build(data,sensors,identity,'ego',(15,),
            noise_rng=None if row['regime']=='clean' else np.random.default_rng(entry['case_seed']))
        with np.load(row['artifact'],allow_pickle=False) as saved:
            require(np.array_equal(saved['truth'],sample['motion']) and np.array_equal(saved['rest'],sample['rest']),
                'Saved TEST ground truth differs from original source reload')
            require(np.array_equal(saved['observations'],observed['trajectory']) and np.array_equal(saved['observation_meta'],observed['observation_meta']),
                'Saved TEST observation differs from independently regenerated input')
            actual=metrics_for(saved)
            independent,_,_=independent_metrics(saved['prediction'],saved['truth'],saved['rest'],saved['observations'],saved['observation_meta'])
        for key,value in row['metrics'].items():
            require(actual[key] is None if value is None else actual[key] is not None,'Metric missingness differs')
            if value is not None:
                maximum=max(maximum,abs(actual[key]-value))
        for key,value in independent.items():
            if key in actual and value is not None and actual[key] is not None:
                independent_maximum=max(independent_maximum,abs(actual[key]-value))
    require(maximum<1e-6 and independent_maximum<1e-3,'CPU metric recomputation differs')
    complete=len(rows)==plan['planned_predictions']
    write(out/'CPU_RECHECK.json',dict(status='independent_cpu_passed' if complete else 'partial_independent_cpu_passed',
        predictions=len(rows),planned_predictions=plan['planned_predictions'],CUDA_initialized=False,
        maximum_recompute_difference=maximum,maximum_independent_formula_difference=independent_maximum,checked_at=now()))
    summarize(out,plan,rows)
    write(out/'STATUS.json',dict(status='completed' if complete else 'partial_verified',completed_predictions=len(rows),planned_predictions=plan['planned_predictions'],checked_at=now()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--out',type=Path,default=DEFAULT_OUT)
    prep.add_argument('--data-output',type=Path,default=DEFAULT_DATA);prep.add_argument('--workers',type=int,default=4)
    prep.add_argument('--checkpoint',type=Path,required=True);prep.add_argument('--checkpoint-sha256',required=True)
    prep.add_argument('--expected-step',type=int,default=41000)
    prep.add_argument('--config',type=Path,default=BASE/'runs/native30_gait_v2_ego55k_oct15/config.json')
    for command in ('evaluate','recheck'):
        p=sub.add_parser(command);p.add_argument('--out',type=Path,default=DEFAULT_OUT)
        if command=='evaluate':
            p.add_argument('--memory-fraction',type=float,default=.5);p.add_argument('--max-predictions',type=int,default=0)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    try:
        with (args.out/'job.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            {'prepare':prepare,'evaluate':evaluate,'recheck':recheck}[args.command](args)
        if args.command=='evaluate' and read(args.out/'STATUS.json')['status']=='sampling_completed_pending_cpu':
            subprocess.run([sys.executable,'-m',__spec__.name,'recheck','--out',str(args.out.resolve())],check=True)
    except Exception as exc:
        write(args.out/'STATUS.json',dict(status='failed',command=args.command,error=repr(exc),checked_at=now()))
        raise


if __name__=='__main__':
    main()
