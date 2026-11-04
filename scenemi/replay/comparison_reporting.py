"""Run and report a complete pinned scene/no-scene comparison.

The subprocess inherits the existing RAM/CPU cgroup. This reporting layer does
not change model sources or partial evaluation seals. Bootstrap units are
independent source families, never overlapping frames or windows.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'configs/scene_joint128_comparison.yaml'
METRICS = ('mpjpe_mm', 'pa_mpjpe_mm', 'root_error_mm', 'head_error_mm',
           'root_relative_mpjpe_mm', 'mpjve_mm_s', 'jerk_m_s3',
           'tracked_acceleration_error_m_s2', 'missing_mpjpe_mm',
           'camera_position_mean_mm')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def timestamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def bounded():
    relative = Path('/proc/self/cgroup').read_text().strip().split(':', 2)[2]
    maximum = (Path('/sys/fs/cgroup')/relative.lstrip('/')/'memory.max').read_text().strip()
    if maximum == 'max' or int(maximum)>16*2**30 or len(os.sched_getaffinity(0))>4:
        raise RuntimeError('Use the existing resources run wrapper')


def family_delta(rows, key, *, baseline='no_scene', draws=2000):
    """Paired equal-family effect and deterministic family-bootstrap interval."""
    families = defaultdict(list)
    for row in rows:
        variants = {r['variant']:r['metrics'] for r in row['rows']}
        a, b = variants[baseline].get(key), variants['scene_1000'].get(key)
        if a is not None and b is not None:
            families[row['job']['case']['family']].append((a, b))
    if not families:
        return None
    pairs = np.array([np.mean(families[k], axis=0) for k in sorted(families)])
    delta = pairs[:, 1]-pairs[:, 0]
    rng = np.random.default_rng(2026103001)
    samples = delta[rng.integers(len(delta), size=(draws,len(delta)))].mean(1)
    return dict(baseline_family_mean=float(pairs[:, 0].mean()), scene_family_mean=float(pairs[:, 1].mean()),
        scene_minus_baseline=float(delta.mean()), ci95_family_bootstrap=np.quantile(samples,[.025,.975]).tolist(),
        fraction_families_improved=float((delta<0).mean()), independent_families=len(delta),
        contributing_windows=sum(map(len, families.values())), bootstrap_draws=draws,
        uncertainty='Conditional on fixed checkpoints, validation membership and one DDIM seed; excludes seed/training uncertainty')


def produce(out, require_complete=True):
    protocol = json.loads((out/'PROTOCOL.json').read_text())
    expected_hash = sha(out/'PROTOCOL.json')
    seal = json.loads((out/'EVALUATOR_SEAL.json').read_text())
    records = []
    for job in protocol['jobs']:
        file = out/'rows'/f'{job["job_id"]:04d}.json'
        if not file.exists():
            if require_complete:
                raise ValueError('Incomplete evaluation: missing pair '+str(job['job_id']))
            continue
        row = json.loads(file.read_text())
        if row['job'] != job or row['protocol_sha256'] != expected_hash or row['evaluator_sha256'] != seal['sha256']:
            raise ValueError('Per-window protocol or evaluator identity changed')
        if sha(out/row['joints_file']) != row['joints_sha256']:
            raise ValueError('Per-window prediction evidence changed')
        if not row['common_public_input_parity'] or row['input_sha256'][0] != row['input_sha256'][1]:
            raise ValueError('Main comparison did not use identical public inputs')
        expected = {'no_scene', 'scene_1000'} | ({'historical_no_scene'} if job['case']['mode']=='exo' else set())
        if {r['variant'] for r in row['rows']} != expected:
            raise ValueError('Missing or unexpected comparison variant')
        records.append(row)
    groups = defaultdict(list)
    diagnostics = []
    for row in records:
        job = row['job']; case = job['case']
        if job['scope']=='diagnostic':
            diagnostics.append(row)
        else:
            groups[case['mode']+'_'+job['condition']+'_'+case['source']].append(row)
    summaries = {}
    intrinsic = []
    for name, rows in groups.items():
        item = dict(windows=len(rows), independent_families=len({r['job']['case']['family'] for r in rows}),
                    matched_inputs={k:family_delta(rows,k) for k in METRICS})
        if rows[0]['job']['case']['mode']=='exo':
            item['historical_pipeline']={k:family_delta(rows,k,baseline='historical_no_scene') for k in METRICS}
            changes = []
            for row in rows:
                variants = {r['variant']:r['metrics'] for r in row['rows']}
                changes.append(variants['no_scene']['mpjpe_mm']-variants['historical_no_scene']['mpjpe_mm'])
            intrinsic.append(dict(group=name, mean_window_mpjpe_normalized_minus_pixel_mm=float(np.mean(changes)),
                                  max_abs_window_mpjpe_difference_mm=float(np.abs(changes).max())))
        summaries[name] = item
    macro = {}
    for condition in ('ego_ego', 'exo_identity', 'exo_tracking', 'exo_occlusion', 'exo_gap'):
        available = [summaries[condition+'_'+source] for source in ('egobody','trumans','grail')
                     if condition+'_'+source in summaries]
        if len(available)==3:
            macro[condition]={}
            for metric in METRICS:
                values = [x['matched_inputs'][metric] for x in available]
                if all(v is not None for v in values):
                    macro[condition][metric]={key:float(np.mean([v[key] for v in values]))
                        for key in ('baseline_family_mean','scene_family_mean','scene_minus_baseline')}
    complete = len(records)==len(protocol['jobs'])
    result = dict(state='complete' if complete else 'partial', full_native_validation=complete,
        test_used=False, independent_test=False, training_ablation=False,
        completed_pairs=len(records), total_pairs=len(protocol['jobs']),
        counts=protocol['counts'], aggregation='family macro within dataset, then equal three datasets; conditions separate',
        per_dataset=summaries, dataset_macro=macro, intrinsics_sensitivity=intrinsic,
        diagnostics=diagnostics, protocol_sha256=expected_hash, evaluator_sha256=seal['sha256'],
        reporter_sha256=sha(Path(__file__)), completed_at_utc=timestamp())
    target = out/('PAIRED_COMPARISON.json' if complete else 'PAIRED_COMPARISON.partial.json')
    write_json(target,result)
    lines = ['# 场景第 1000 步与无场景 replay 第 2000 步：完整原生验证对照', '',
        f'状态：{result["state"]}，完成 {len(records)}/{len(protocol["jobs"])} 组。', '',
        '完整清单为 EGO 3,471 个窗口，EXO 3,540 个窗口 × 4 条件，另有九个独立诊断案例。',
        '每个可接受起点区间按 128 帧步长覆盖并保留末尾对齐窗口，不只选择表现较好的片段。',
        '下面主对照对两个模型使用完全相同的相机、归一化内参和 frontend 扰动；均为同设备、batch=1、seed=777。',
        '历史无场景 EXO 使用像素内参的原输出也完整保留，用于区分内参改动与模型版本改动。', '',
        '| 分组 | 窗口/家族 | 无场景 MPJPE | 场景 MPJPE | 场景−无场景 | 配对家族 95% CI |',
        '|---|---:|---:|---:|---:|---|']
    for name,item in summaries.items():
        m=item['matched_inputs']['mpjpe_mm'];lo,hi=m['ci95_family_bootstrap']
        lines.append(f'| {name} | {item["windows"]}/{item["independent_families"]} | '
            f'{m["baseline_family_mean"]:.2f} | {m["scene_family_mean"]:.2f} | {m["scene_minus_baseline"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] |')
    lines += ['', 'MPJPE 单位 mm，负差值代表场景版更低。先按独立家族汇总，再给三个数据集等权；不把重叠窗口当独立样本计算置信区间。', '',
        '## 历史内参口径对照（EXO）', '',
        '| 分组 | 历史无场景 MPJPE | 同内参无场景 MPJPE | 场景 MPJPE |', '|---|---:|---:|---:|']
    for name,item in summaries.items():
        if 'historical_pipeline' not in item:continue
        old=item['historical_pipeline']['mpjpe_mm'];matched=item['matched_inputs']['mpjpe_mm']
        lines.append(f'| {name} | {old["baseline_family_mean"]:.2f} | {matched["baseline_family_mean"]:.2f} | {matched["scene_family_mean"]:.2f} |')
    lines += ['', '## 解释范围', '',
        '这是完整三原生数据集验证集上的固定 checkpoint 版本对照。EXO frontend 为合成姿态/体形扰动，不能写成真实 tracker 结果。',
        '场景版本还经过额外适配，EXO 提案结构也不同，因此不是纯场景输入消融。未修改 checkpoint，也没有修改训练代码。',
        '当前未读取独立测试数组。若验证结果用于决定后续改模，论文最终泛化结论仍需冻结方案后的独立测试。',
        'PA-MPJPE、全局 root、相机残差、缺失区域、最坏帧与跟踪段时序指标均保留在 JSON；不用位置误差改善替代时序质量判断。',
        '家族 bootstrap 区间只反映固定权重与单个采样 seed 下的家族差异，不涵盖重新训练或扩散多 seed 方差。',
        '原无场景父版本的四来源全验证没有通过 AMASS EGO 保留门槛；本三来源对照不能覆盖或撤销该失败记录。']
    if not complete:
        lines += ['', '**当前尚未完成全量评估，表内部分结果不得作为完整验证结论。**']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    return result


def pipeline(config, device):
    c=yaml.safe_load(config.read_text());out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    status=dict(state='evaluating',pid=os.getpid(),device=device,config_sha256=sha(config),
        free_disk_before_bytes=shutil.disk_usage(out).free,reporter_sha256=sha(Path(__file__)),started_at_utc=timestamp(),
        numpy_version=np.__version__,numpy_import=str(Path(np.__file__).resolve()))
    write_json(out/'PIPELINE_STATUS.json',status)
    binaries=sorted((Path(np.__file__).parent/'core').glob('_multiarray_umath*.so'))
    write_json(out/'RUNTIME_NUMPY.json',dict(version=np.__version__,module=str(Path(np.__file__).resolve()),
        binaries={str(p):sha(p) for p in binaries},original_environment_modified=False))
    command=[sys.executable,'-u','-m','scenemi.replay.scene_comparison','run','--device',device,'--config',str(config)]
    child=subprocess.run(command,cwd=ROOT)
    if child.returncode:
        status.update(state='failed',evaluation_returncode=child.returncode,updated_at_utc=timestamp())
        write_json(out/'PIPELINE_STATUS.json',status)
        raise SystemExit(child.returncode)
    status.update(state='summarizing',updated_at_utc=timestamp());write_json(out/'PIPELINE_STATUS.json',status)
    result=produce(out)
    status.update(state=result['state'],elapsed_seconds=time.monotonic()-started,
        free_disk_after_bytes=shutil.disk_usage(out).free,completed_at_utc=timestamp())
    write_json(out/'PIPELINE_STATUS.json',status)
    print(json.dumps(dict(state=result['state'],report=str(out/'REPORT_zh.md')),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('pipeline','report'))
    parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args();bounded()
    if args.action=='pipeline':return pipeline(args.config,args.device)
    c=yaml.safe_load(args.config.read_text())
    result=produce(ROOT/c['output'],require_complete=not args.partial)
    print(json.dumps(dict(state=result['state'],completed_pairs=result['completed_pairs'])),flush=True)


if __name__=='__main__':
    main()
