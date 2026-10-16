"""Queued, zero-update RICH scene/guidance ablation of one frozen checkpoint.

Run --prepare-only to seal the outcome-independent cohort; run --wait to wait
for the full-grid evaluator and its independent complete CPU verification.
All model/dataset code is imported unchanged. In this process guidance_enabled
and use_scene specify the intervention; model tensors are checked before/after.
"""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
FULL = BASE/'runs/native30_live_monitor_oct16/rich_full_validation_oct16'
OUT = FULL/'scene_causal_check'
sys.path.insert(0, str(REPO))


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def prepare():
    OUT.mkdir(exist_ok=True)
    plan = read(FULL/'PLAN.json')
    groups = collections.defaultdict(list)
    for entry in plan['plan']:
        require(entry['dataset'] == 'rich', 'Unexpected dataset')
        groups[entry['recording']].append(entry)
    cohort = [sorted(entries, key=lambda e: e['start'])[len(entries)//2]
              for _, entries in sorted(groups.items())]
    require(len(cohort) == 27, 'Expected 27 eligible recordings')
    dependencies = [FULL/'PLAN.json', FULL/'FROZEN_CHECKPOINT.json', FULL/'runner.py',
                    BASE/'runs/native30_final_review_oct16/io_recovery.py', Path(__file__)]
    result = dict(protocol='frozen15100-rich-scene-guidance-factorial-v1',
        selection='Every eligible recording; middle legal indexed start, sorted by start, upper middle for even counts; no prediction/action selection.',
        checkpoint_step=15100, frames=192, fps=30, ddim_steps=20,
        regimes=['clean','noisy'], cohort=cohort, independent_recordings=27,
        planned_new_predictions=108, planned_reused_A_predictions=54,
        A_reconstruction_control=dict(case=cohort[0]['case'],regime='clean',additional_sampling_calls=1,
            requirement='Rebuilt complete A trajectory reproduces saved A; atol/rtol=1e-6 in 201-D normalized motion, record exact equality and maximum absolute delta.'),
        arms=dict(A='Full scene + physical guidance; reuse independently verified full-grid prediction.',
                  B='Scene ON, model.guidance_enabled=False; all learned scene/feedback + head anchor/IK unchanged.',
                  C='Same as B, sample_motion(use_scene=False); head anchor/IK/reference unchanged.',
                  reference='Observation-only reference from the exact saved inference input; CPU evaluation only.'),
        comparisons=dict(learned_scene='C minus B; total learned scene pathway, including contact/correspondence-gated root/rotation feedback; positive error delta means lower error without the independent physical solver. Not encoder-only attribution.',
                         physical_guidance='B minus A; positive error delta means physical solver reduces error.'),
        boundary='Same trained weights, inference intervention, development VAL; not separately trained models, TEST ranking or proof diffusion beats another generative family.',
        dependencies_sha256={str(p):sha(p) for p in dependencies})
    path = OUT/'PLAN.json'
    if path.exists():
        require(read(path) == result, 'Prepared cohort/source dependencies changed')
    else:
        write(path, result)
    return result


def full_ready():
    paths = [FULL/'STATUS.json', FULL/'PROTOCOL.json', FULL/'VERIFY_COMPLETE.json']
    if not all(p.exists() for p in paths):
        return False
    status, protocol, verified = map(read, paths)
    if status.get('status') != 'completed' or protocol.get('status') != 'completed':
        return False
    require(verified['status'] == 'passed_complete' and verified['all_final_predictions_recomputed'],
            'Full RICH complete verification did not pass')
    require(verified['committed_rows_verified'] == 412 and not verified['missing_case_regime_pairs'],
            'Full RICH complete verification coverage differs')
    # A completed status can precede the runner's final report by a moment.
    pid = protocol['pid']
    process = Path('/proc')/str(pid)/'cmdline'
    if process.exists() and str(FULL/'runner.py').encode() in process.read_bytes():
        return False
    return True


def wait_for_full(wait):
    while not full_ready():
        require(wait, 'Full RICH or independent complete verification is pending; use --wait')
        source_status = read(FULL/'STATUS.json')
        require(source_status.get('status') != 'failed', 'Full RICH evaluator failed; queued job stays unstarted')
        write(OUT/'STATUS.json', dict(status='queued_waiting_full_RICH_and_CPU_verification',
            pid=os.getpid(), checked_at=now(), CUDA_initialized=False,
            full_completed_predictions=source_status.get('completed_predictions'),
            full_planned_predictions=412, planned_new_predictions=108, completed_new_predictions=0))
        time.sleep(30)


def array_digest(arrays):
    digest = hashlib.sha256()
    for name in ('truth','rest','observations','observation_meta','reference','identity_json'):
        value = arrays[name]
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def assert_metrics_equal(actual, expected, label):
    if isinstance(expected, dict):
        for key, value in expected.items():
            require(key in actual, label+' missing '+key)
            assert_metrics_equal(actual[key], value, label+'/'+key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), label+' length')
        for i, value in enumerate(expected):
            assert_metrics_equal(actual[i], value, label+'/'+str(i))
    elif expected is None:
        require(actual is None, label+' None mismatch')
    else:
        require(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=3e-5), label+' recomputation differs')


def load_rows(plan):
    rows = []
    path = OUT/'rows.jsonl'
    if not path.exists():
        return rows
    raw = path.read_bytes()
    require(not raw or raw.endswith(b'\n'), 'Partial row tail; preserve and resolve explicitly before resume')
    selected = {e['case']:e for e in plan['cohort']}
    seen = set()
    for line in raw.splitlines():
        row = json.loads(line)
        key = (row['case'],row['regime'],row['arm'])
        require(key not in seen and row['case'] in selected, 'Duplicate/unplanned row')
        require(row['checkpoint_step'] == 15100 and row['regime'] in plan['regimes'], 'Mixed step/regime')
        require(row['arm'] in plan['arms'], 'Unknown arm')
        for key_name in ('recording','sequence_id','case_seed','start','source_first'):
            require(row[key_name] == selected[row['case']][key_name], 'Row cohort mismatch '+key_name)
        require(sha(Path(row['artifact'])) == row['artifact_sha256'], 'Saved prediction changed')
        seen.add(key); rows.append(row)
    return rows


def report(rows, complete=False):
    import numpy as np
    by_key = {(r['case'],r['regime'],r['arm']):r for r in rows}
    keys = ['mpjpe_mm','pa_mpjpe_mm','root_orientation_deg','body_local_foot_error_mm',
            'foot_speed_during_GT_stance_m_s','foot_velocity_RMSE_m_s']
    result = dict(status='complete' if complete else 'partial', checked_at=now(),
        checkpoint_step=15100, rows=len(rows), regimes={}, interpretation=read(OUT/'PLAN.json')['boundary'])
    text = ['# 冻结 15100 步：场景条件与物理修正配对检查', '',
            '每条 RICH VAL 合格录制一个预先确定的中位合法窗口。A 完整；B 无物理修正、保留全部学习式场景路径（含 contact/correspondence 门控的 root/rotation feedback）；C 无物理修正和全部场景路径；reference 为输入参考。', '',
            '**'+('已完整结束，54组输入、108次新增预测，所有结果完成CPU独立重算。' if complete else '部分结果，不能作为最终结论。')+'**', '',
            '误差/足速采用录制等权。低GT足速为运动学代理，不是真实接触/穿透率；更低脚速需结合姿态误差，避免奖励冻结。', '']
    for regime in ('clean','noisy'):
        arms = {}
        for arm in ('reference','A','B','C'):
            subset = [r for r in rows if r['regime']==regime and r['arm']==arm]
            valid = {k:[r['metrics'][k] for r in subset if r['metrics'][k] is not None] for k in keys}
            arms[arm] = dict(recordings=len(subset), metric_recording_counts={k:len(v) for k,v in valid.items()},
                **{k:statistics.mean(v) if v else None for k,v in valid.items()})
        pairs = {}
        for label, worse, better in [('learned_scene','C','B'),('physical_guidance','B','A')]:
            case_ids = sorted({r['case'] for r in rows if r['regime']==regime})
            case_ids = [i for i in case_ids if (i,regime,worse) in by_key and (i,regime,better) in by_key]
            paired = {}
            for k in keys:
                metric_cases = [i for i in case_ids if by_key[(i,regime,worse)]['metrics'][k] is not None
                                and by_key[(i,regime,better)]['metrics'][k] is not None]
                delta = np.array([by_key[(i,regime,worse)]['metrics'][k]-by_key[(i,regime,better)]['metrics'][k]
                                  for i in metric_cases], dtype=np.float64)
                if not len(delta):
                    continue
                rng = np.random.default_rng(2026101642)
                draws = delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
                paired[k] = dict(mean_delta=float(delta.mean()), median_delta=float(np.median(delta)),
                    positive_delta_recordings=int((delta>0).sum()), recordings=len(delta),
                    descriptive_paired_bootstrap_95_percent=np.percentile(draws,[2.5,97.5]).tolist())
            pairs[label] = paired
        result['regimes'][regime] = dict(arms=arms, paired_comparisons=pairs)
        text += ['## '+regime, '', '| 方案 | 录制数 | World MPJPE mm | PA-MPJPE mm | root旋转 ° | 身体局部足误差 mm | GT低速期预测足速 m/s |', '|---|---:|---:|---:|---:|---:|---:|']
        for arm, values in arms.items():
            numbers = [values[k] for k in keys[:-1]]
            text.append('| '+arm+' | '+str(values['recordings'])+' | '+' | '.join('—' if v is None else f'{v:.3f}' for v in numbers)+' |')
        text += ['', '配对差值定义为 C−B（学习场景）和 B−A（物理修正）；误差差值为正才是改善。']
        for label, values in pairs.items():
            if 'mpjpe_mm' in values:
                v = values['mpjpe_mm']; lo,hi=v['descriptive_paired_bootstrap_95_percent']
                text.append(f"- {label}: MPJPE 平均差 {v['mean_delta']:.3f} mm，{v['positive_delta_recordings']}/{v['recordings']} 录制改善；录制配对 bootstrap 95% 描述区间 [{lo:.3f}, {hi:.3f}] mm。")
        text.append('')
    text += ['同权重推理消融不等于分别训练的架构优越性证据，也不提供跨方法排名；所有样本属于开发 VAL。bootstrap 仅描述这批录制的配对差异，不是独立 TEST 确认。', '',
             '完整封存、输入对应与模型张量不变检查见 `PROTOCOL.json`、`VERIFY.json`；逐窗数据见 `rows.jsonl`。']
    write(OUT/'SUMMARY.json',result)
    temporary = OUT/'REPORT_zh.md.tmp'; temporary.write_text('\n'.join(text)+'\n'); temporary.replace(OUT/'REPORT_zh.md')


def run(plan):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, window_options
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, sample_motion
    from experiments.offline_camera_retrain_v1.native30_validation import raw_metrics
    from experiments.offline_camera_retrain_v1.standard_motion_metrics import standard_motion_metrics
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    from experiments.offline_camera_retrain_v1 import native30_surface_cache as surfaces, native30_memory as memory
    from experiments.offline_camera_retrain_v1.native30_surface_cache import CanonicalSurfaceCache
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    helper = module('full_rich_readonly_helpers', FULL/'runner.py')
    effective = helper.verify_sources(); cfg = effective['config']
    for filename, expected in plan['dependencies_sha256'].items():
        require(sha(filename)==expected, 'Prepared dependency changed: '+filename)
    for filename, expected in read(FULL/'PLAN.json')['source_sha256'].items():
        require(sha(filename)==expected, 'Dataset-plan dependency changed: '+filename)
    frozen = read(FULL/'FROZEN_CHECKPOINT.json'); full_verified = read(FULL/'VERIFY_COMPLETE.json')
    require(sha(FULL/'rows.jsonl')==full_verified['rows_committed_prefix_sha256'], 'Full RICH rows changed after independent verification')
    require(sha(FULL/'PLAN.json')==full_verified['plan_sha256'], 'Full RICH plan changed after independent verification')
    require(sha(FULL/'checkpoint_eval.pt')==frozen['evaluation_checkpoint_sha256'], 'Checkpoint SHA differs')
    checkpoint = torch.load(FULL/'checkpoint_eval.pt',map_location='cpu',mmap=True,weights_only=False)
    require(checkpoint['step']==15100 and checkpoint['effective_config']==effective, 'Frozen checkpoint configuration/step differs')
    before = helper.model_tensor_digest(checkpoint['model'])
    require(before==frozen['compact_model_tensor_sha256']==full_verified['checkpoint']['model_tensor_sha256_sorted_names_dtypes_shapes_and_bytes'], 'Checkpoint tensor digest differs')
    protocol = dict(status='running', started_at=now(), pid=os.getpid(), plan_sha256=sha(OUT/'PLAN.json'),
        runner_sha256=sha(__file__), checkpoint_sha256=frozen['evaluation_checkpoint_sha256'], checkpoint_step=15100,
        model_tensor_digest_before=before, source_verification_sha256=sha(FULL/'VERIFY_COMPLETE.json'),
        optimizer_updates=0, training_modified=False, memory_fraction=.25, resources_before=helper.resources(),
        arms=plan['arms'], comparisons=plan['comparisons'])
    if (OUT/'PROTOCOL.json').exists():
        old = read(OUT/'PROTOCOL.json')
        for key in ('plan_sha256','runner_sha256','checkpoint_sha256','model_tensor_digest_before'):
            require(old[key]==protocol[key], 'Resume protocol differs: '+key)
    write(OUT/'PROTOCOL.json',protocol)
    require(helper.resources()['MemAvailable_bytes']>=3*1024**3, 'Insufficient host RAM')
    model=make_model(cfg).cuda().eval(); model.load_state_dict(checkpoint['model'],strict=True); del checkpoint
    require(model.guidance_enabled is True and cfg['physical_guidance']['enabled'], 'A source model must enable physical guidance')
    require(hasattr(model,'relation_attention') and model.patch_enabled and model.local_enabled,
            'Expected learned scene paths missing')
    model.guidance_enabled=False
    io=module('scene_check_crc_reads', BASE/'runs/native30_final_review_oct16/io_recovery.py')
    surfaces.np=io.NumpyProxy(surfaces.np,OUT/'IO_READS.jsonl'); memory.np=io.NumpyProxy(memory.np,OUT/'IO_READS.jsonl')
    data=Native30Data(BASE/cfg['train_manifest'],'validation',cfg['seed']+1,frames=cfg['frames'],
        memory_protocol=cfg['scene']['memory_protocol'],quality_config=cfg['gt_reliability'],
        exo_max_empty_frames=cfg['exo_max_empty_frames'],**window_options(cfg))
    data.runtime_cache_root=OUT/'native30_runtime_cache_v2'
    original_cache_path=data.scene_memory._cache_path
    def read_only_path(*args):
        path=original_cache_path(*args)
        return path if path is not None and path.exists() else None
    data.scene_memory._cache_path=read_only_path
    data.surface_cache=CanonicalSurfaceCache(data,limit=1); data.scene_memory.cached_streams=1
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),read(BASE/cfg['mount_defaults']),cfg['observation']['input_noise'])
    noisy_builder=Native30ObservationBuilder(builder.skin_cache,builder.mounts,dict(builder.noise_config,clean_probability=0.))
    footpath=BASE/cfg['run_control']['foot_metrics_source']
    require(sha(footpath)==cfg['run_control']['foot_metrics_sha256'],'Foot metrics changed')
    feet=module('scene_check_original_foot_metrics',footpath)
    full_rows={(r['case'],r['regime']):r for r in map(json.loads,(FULL/'rows.jsonl').read_text().splitlines())}
    rows=load_rows(plan); done={(r['case'],r['regime'],r['arm']):r for r in rows}
    (OUT/'motions').mkdir(exist_ok=True)
    def metrics(prediction, truth, rest, meta):
        p=torch.from_numpy(prediction)[None]; batch=dict(motion=torch.from_numpy(truth)[None],rest=torch.from_numpy(rest)[None])
        conditioned=torch.from_numpy(meta[...,0]>0)[None]
        with torch.no_grad():
            raw=raw_metrics(p,batch,conditioned); extra=feet.extra_metrics(p,batch)
            benchmark=standard_motion_metrics(forward_kinematics(p,batch['rest'])[0].numpy(),
                forward_kinematics(batch['motion'],batch['rest'])[0].numpy(),fps=30.)
        combined={k:v['value'] for k,v in raw.items()}; combined.update(benchmark)
        combined.update({k:v for k,v in extra.items() if isinstance(v,(int,float)) or v is None})
        return combined,raw,extra,benchmark
    try:
        for entry in plan['cohort']:
            case=entry['case']
            if all((case,regime,arm) in done for regime in plan['regimes'] for arm in plan['arms']):
                continue
            members=[i for i,(m,_) in enumerate(data.members[entry['group']]) if m['path']==entry['path']]
            require(len(members)==1,'Source identity not unique')
            sample,identity=data.sample(entry['group'],'ego',member=members[0],start=entry['start'],yaw=0)
            for regime in plan['regimes']:
                source_row=full_rows[(case,regime)]; source_path=FULL/source_row['motion_file']
                require(sha(source_path)==source_row['motion_sha256'],'A artifact changed')
                with np.load(source_path,allow_pickle=False) as archive:
                    saved={key:archive[key] for key in archive.files}
                require(json.loads(str(saved['identity_json']))==identity,'A identity differs from rebuilt source')
                require(int(saved['step'])==15100,'A checkpoint step mismatch')
                for key in ('motion','rest'):
                    require(np.array_equal(sample[key],saved['truth' if key=='motion' else key]),'A '+key+' mismatch')
                local=builder if regime=='clean' else noisy_builder
                observed,scene,asset=local.build(data,sample,identity,'ego',(15,),
                    noise_rng=None if regime=='clean' else np.random.default_rng(entry['case_seed']))
                for key,original in [('trajectory','observations'),('observation_meta','observation_meta')]:
                    require(np.array_equal(observed[key],saved[original]),'A '+key+' input mismatch')
                batch=collate_observations([(observed,scene,asset,'ego')]); known=model_inputs(cfg,batch)
                require(not any(k in known for k in ('motion','joints','contact_target','gt_position_weight')),'Target in inference payload')
                require('body_local_embedding' not in known,'Stale scene-derived body condition in shared input')
                require(np.array_equal(known['reference_motion'][0].detach().cpu().numpy(),saved['reference']),'A reference mismatch')
                input_digest=array_digest(saved)
                control=plan['A_reconstruction_control']
                if case==control['case'] and regime==control['regime'] and not (OUT/'A_RECONSTRUCTION_CONTROL.json').exists():
                    require(model._sampling_scene is None,'Stale sampling cache before A control')
                    model.guidance_enabled=True
                    try:
                        with torch.no_grad():
                            replay=sample_motion(model,known,cfg,steps=20,seed=entry['case_seed'],use_scene=True)
                        rebuilt=replay[0].detach().cpu().numpy(); del replay
                    finally:
                        model.guidance_enabled=False
                    difference=float(np.max(np.abs(rebuilt-saved['prediction'])))
                    require(np.allclose(rebuilt,saved['prediction'],atol=1e-6,rtol=1e-6),'Rebuilt A control does not reproduce saved full-scene trajectory')
                    write(OUT/'A_RECONSTRUCTION_CONTROL.json',dict(status='passed',case=case,regime=regime,
                        checked_at=now(),input_sha256=input_digest,source_A_sha256=source_row['motion_sha256'],
                        exact_prediction_array_equality=bool(np.array_equal(rebuilt,saved['prediction'])),
                        maximum_absolute_201D_motion_delta=difference,absolute_tolerance=1e-6,relative_tolerance=1e-6))
                for arm in ('A','reference','B','C'):
                    key=(case,regime,arm)
                    if key in done:
                        require(done[key]['input_sha256']==input_digest,'Existing row input mismatch'); continue
                    write(OUT/'STATUS.json',dict(status='evaluating',pid=os.getpid(),checked_at=now(),
                        current_case=case,current_recording=entry['recording'],current_regime=regime,current_arm=arm,
                        completed_new_predictions=sum(r['arm'] in ('B','C') for r in rows),planned_new_predictions=108,
                        committed_metric_rows=len(rows),CUDA_initialized=torch.cuda.is_initialized(),resources=helper.resources()))
                    tick=time.monotonic()
                    if arm in ('A','reference'):
                        prediction=saved['prediction' if arm=='A' else 'reference']; artifact=source_path
                    else:
                        require(model.guidance_enabled is False,'Physical guidance unexpectedly enabled')
                        require(model._sampling_scene is None,'Stale sampling cache before B/C')
                        with torch.no_grad():
                            result=sample_motion(model,known,cfg,steps=20,seed=entry['case_seed'],use_scene=arm=='B')
                        torch.cuda.synchronize(); prediction=result[0].detach().cpu().numpy(); del result
                        require(np.isfinite(prediction).all(),'Nonfinite prediction')
                        artifact=OUT/'motions'/f'case{case:03d}_{regime}_{arm}.npz'
                        temporary=artifact.with_suffix('.tmp')
                        with temporary.open('wb') as stream:
                            np.savez_compressed(stream,prediction=prediction,step=np.array(15100),input_sha256=np.array(input_digest))
                            stream.flush(); os.fsync(stream.fileno())
                        temporary.replace(artifact)
                    combined,raw,extra,benchmark=metrics(prediction,saved['truth'],saved['rest'],saved['observation_meta'])
                    if arm=='A':
                        assert_metrics_equal(raw,source_row['metrics'],'A/raw')
                        assert_metrics_equal(extra,{k:source_row['extra_metrics'][k] for k in extra},'A/extra')
                        assert_metrics_equal(benchmark,{k:v['value'] for k,v in source_row['benchmark_metrics'].items()},'A/benchmark')
                    row={k:entry[k] for k in ('case','recording','sequence_id','start','source_first','case_seed')}
                    row.update(regime=regime,arm=arm,checkpoint_step=15100,input_sha256=input_digest,
                        artifact=str(artifact),artifact_sha256=sha(artifact),source_A_artifact=str(source_path),
                        source_A_sha256=source_row['motion_sha256'],metrics=combined,raw_metric_counts={k:v['count'] for k,v in raw.items()},
                        elapsed_seconds=time.monotonic()-tick,checked_at=now())
                    with (OUT/'rows.jsonl').open('a') as stream:
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
                    rows.append(row); done[key]=row
                    print(json.dumps(dict(case=case,regime=regime,arm=arm,mpjpe_mm=combined['mpjpe_mm'],rows=len(rows))),flush=True)
                    report(rows)
                    require(helper.resources()['MemAvailable_bytes']>=3*1024**3,'Evaluator stopped to preserve host RAM')
                del batch,known,observed,scene,asset,saved
                gc.collect(); torch.cuda.empty_cache()
            data.arrays.clear()
        require(len(rows)==216,'Expected 27 recordings x 2 regimes x 4 metric rows')
        # Reopen every saved prediction and recompute on CPU independently of the
        # in-memory result used at sampling. This also verifies reused A/reference.
        for row in rows:
            require(sha(row['artifact'])==row['artifact_sha256'],'Artifact changed before final verification')
            with np.load(row['source_A_artifact'],allow_pickle=False) as archive:
                saved={key:archive[key] for key in archive.files}
            require(array_digest(saved)==row['input_sha256'],'Saved input changed before verification')
            if row['arm'] in ('A','reference'):
                pred=saved['prediction' if row['arm']=='A' else 'reference']
            else:
                with np.load(row['artifact'],allow_pickle=False) as archive:
                    pred=archive['prediction']; require(str(archive['input_sha256'])==row['input_sha256'],'Prediction/input mismatch')
            recomputed,_,_,_=metrics(pred,saved['truth'],saved['rest'],saved['observation_meta'])
            assert_metrics_equal(recomputed,row['metrics'],'final/'+str(row['case'])+'/'+row['regime']+'/'+row['arm'])
        after=helper.model_tensor_digest(model.state_dict())
        require(after==before,'Evaluation altered model tensors')
        helper.verify_sources()
        require(sha(FULL/'checkpoint_eval.pt')==frozen['evaluation_checkpoint_sha256'],'Frozen checkpoint changed during evaluation')
        protocol.update(status='completed',finished_at=now(),model_tensor_digest_after=after,resources_after=helper.resources())
        write(OUT/'PROTOCOL.json',protocol)
        write(OUT/'VERIFY.json',dict(status='passed_complete',checked_at=now(),CPU_metric_rows_recomputed=len(rows),
            unique_recordings=27,regimes=plan['regimes'],model_tensors_unchanged=True,model_tensor_digest=after,
            A_reused_outputs_recomputed=54,rebuilt_truth_rest_observations_meta_reference_exact=True,
            A_reconstruction_control=read(OUT/'A_RECONSTRUCTION_CONTROL.json'),
            all_artifacts_digests_verified=True,sealed_training_sources_verified=365,optimizer_updates=0,
            limitation='Scene geometry rebuilt by unchanged sealed source from fixed source identities and CRC-checked caches; original A artifacts did not persist a scene-geometry byte digest.'))
        report(rows,complete=True)
        write(OUT/'STATUS.json',dict(status='completed',checked_at=now(),pid=os.getpid(),
            completed_new_predictions=108,planned_new_predictions=108,committed_metric_rows=len(rows),resources_after=helper.resources()))
    finally:
        model.guidance_enabled=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    with (OUT/'job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            plan=prepare()
            if args.prepare_only:
                print(json.dumps(dict(status='prepared',recordings=len(plan['cohort']),new_predictions=108,CUDA_initialized=False)))
                return
            wait_for_full(args.wait)
            run(plan)
        except BaseException as exc:
            previous=read(OUT/'STATUS.json') if (OUT/'STATUS.json').exists() else {}
            previous.update(status='failed',error=repr(exc),checked_at=now(),pid=os.getpid())
            write(OUT/'STATUS.json',previous)
            raise


if __name__=='__main__':
    main()
