"""Paired scene-on/off diagnosis of an already selected native scene checkpoint.

Entry: scenemi.runtime run -- ... python -m scenemi.benchmark scene --policy ...
Reuse the sealed selection windows and scene-on results; first reproduce one
scene-on window per corpus. This is an inference intervention, not a matched
training ablation or an independent test evaluation.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.scene_dataset import SceneDataset, inference_payload
from scenemi.data.local_scene import LocalSceneDataset, inference_payload as local_payload
from scenemi.evaluation import summarize
from scenemi.metrics import motion_metrics
from scenemi.parallel_resources import configure_cuda
from scenemi.scene_training import make_model, settings
from scenemi.scene_selection import aggregate_scene_rows
from scenemi.training import deterministic, ensure_bounded, now
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

DATASETS = ('egobody', 'trumans', 'grail')
SCENE_KEYS = ('occupancy', 'frame_scene_points', 'frame_scene_owner',
              'local_history_points', 'local_history_valid', 'local_frame_points', 'local_frame_owner')


def without_scene(payload):
    return {key: value for key, value in payload.items() if key not in SCENE_KEYS}


def check_parity(actual, reference, tolerance=1e-4):
    if actual.keys() != reference.keys() or any(
            not np.isfinite(value) or abs(value-reference[key]) > tolerance
            for key, value in actual.items()):
        raise ValueError('Scene-on reproduction disagrees with the saved selection metrics')


def paired_summary(on_rows, off_rows, *, category_balanced=False):
    if not on_rows or len(on_rows) != len(off_rows) or any(
            a['identity'] != b['identity'] for a, b in zip(on_rows, off_rows)):
        raise ValueError('A paired comparison needs exactly the same ordered identities')
    result = {}
    for dataset in DATASETS:
        on = [r for r in on_rows if r['identity']['dataset'] == dataset]
        off = [r for r in off_rows if r['identity']['dataset'] == dataset]
        if not on:
            raise ValueError('All three validation corpora must be present')
        a, b = summarize(on), summarize(off)
        benefits = {key: b['family_macro_mean'][key]-a['family_macro_mean'][key]
                    for key in a['family_macro_mean']}
        result[dataset] = dict(scene_on=a, scene_off=b, off_minus_on=benefits)
    overall = {condition: {key: float(np.mean([
        result[name][condition]['family_macro_mean'][key] for name in DATASETS]))
        for key in result['egobody']['scene_on']['family_macro_mean']}
        for condition in ('scene_on', 'scene_off')}
    overall['off_minus_on'] = {key: overall['scene_off'][key]-overall['scene_on'][key]
                               for key in overall['scene_on']}
    if category_balanced:
        on, off = (aggregate_scene_rows(rows) for rows in (on_rows, off_rows))
        categories = {}
        for name in on['grail_per_category']:
            a, b = on['grail_per_category'][name], off['grail_per_category'][name]
            categories[name] = dict(scene_on=a, scene_off=b,
                off_minus_on={key:b['family_macro_mean'][key]-a['family_macro_mean'][key]
                              for key in a['family_macro_mean']})
        grail = {condition:{key:float(np.mean([
            group[condition]['family_macro_mean'][key] for group in categories.values()]))
            for key in result['grail']['scene_on']['family_macro_mean']}
            for condition in ('scene_on', 'scene_off')}
        for condition in ('scene_on', 'scene_off'):
            for key in overall[condition]:
                overall[condition][key] = float(np.mean([
                    result['egobody'][condition]['family_macro_mean'][key],
                    result['trumans'][condition]['family_macro_mean'][key], grail[condition][key]]))
        overall['off_minus_on'] = {key:overall['scene_off'][key]-overall['scene_on'][key]
                                   for key in overall['scene_on']}
        return dict(per_dataset=result, dataset_macro=overall, grail_per_category=categories,
                    grail_category_macro=grail, grail_category_balanced=True)
    return dict(per_dataset=result, dataset_macro=overall)


def verify_seal(seal, config, scene_paths):
    value = {key: item for key, item in seal.items() if key != 'sha256'}
    if hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest() != seal['sha256']:
        raise ValueError('Scene training seal is invalid')
    if seal['protocol'] not in ('native-scene-ego128-v2', 'native-scene-ego128-local-v3') or seal['config'] != config:
        raise ValueError('Expected a sealed global or local native scene branch')
    inputs = yaml.safe_load(Path(scene_paths).read_text())
    for name, digest in seal['sources'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Sealed training source changed: {name}')
    for name, digest in seal['inputs'].items():
        if sha256(inputs[name]) != digest:
            raise ValueError(f'Scene input manifest changed: {name}')
    for name, digest in seal['assets'].items():
        if sha256(name) != digest:
            raise ValueError(f'Body asset changed: {name}')
    for name, version in seal['environment'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Training package changed: {name}')


@torch.inference_mode()
def predict(model, batch, *, use_scene, seed, steps, device, local=False):
    payload = (local_payload if local else inference_payload)(batch)
    if not use_scene:
        payload = without_scene(payload)
    motion = sample_native30(model, payload, steps=steps, seed=seed, use_scene=use_scene)
    _, details = model(motion, torch.zeros(1, device=device, dtype=torch.long),
                       payload, use_scene=use_scene, return_details=True)
    return forward_kinematics(motion, details['rest'])[0]


def report(out, result):
    local = result.get('grail_category_balanced', False)
    lines = ['# '+('EGO-only 局部场景模型：固定权重开／关场景' if local else '旧场景分支：训练结果与场景输入诊断'), '',
        f"固定已选定的 {result['step']} 步权重、原验证窗口、相机和原生体形解码，仅在推理时切换场景路径。",
        '关闭场景同时移除 occupancy 和逐帧场景点；相机轨迹、内参和可用性保持一致。', '',
        '| 数据集 | 窗口/家族 | 有场景 MPJPE | 无场景 MPJPE | 有场景 PA-MPJPE | 无场景 PA-MPJPE | 有场景 root | 无场景 root |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, group in result['per_dataset'].items():
        a, b = (group[c]['family_macro_mean'] for c in ('scene_on', 'scene_off'))
        if local and name == 'grail':
            a, b = (result['grail_category_macro'][c] for c in ('scene_on', 'scene_off'))
        n = group['scene_on']
        lines.append(f"| {name} | {n['windows']}/{n['independent_source_families']} | " +
            ' | '.join(f'{m[k]:.2f}' for k in ('mpjpe_mm', 'pa_mpjpe_mm', 'root_error_mm') for m in (a, b))+' |')
    a, b = (result['dataset_macro'][c] for c in ('scene_on', 'scene_off'))
    windows = sum(g['scene_on']['windows'] for g in result['per_dataset'].values())
    families = sum(g['scene_on']['independent_source_families'] for g in result['per_dataset'].values())
    lines += [f'| 三数据集等权 | {windows}/{families} | '+
        ' | '.join(f'{m[k]:.2f}' for k in ('mpjpe_mm', 'pa_mpjpe_mm', 'root_error_mm') for m in (a, b))+' |', '',
        ('所有表内误差单位为 mm，越低越好。沿用各自选模协议，三数据集等权；GRAIL '+
         ('按家族、再对五类动作等权。' if local else '未做动作类别等权。')), '',
        '这是选模验证集上的固定权重推理诊断，不能充当独立测试结果或匹配训练消融。关闭训练时始终存在的场景路径会改变输入分布；',
        '本结果测量当前权重对场景路径的依赖，不能将整个收益归因于场景训练本身。',
        'AMASS 父权重与场景权重还差原生体形适配和场景域训练，不能直接作为纯场景消融。', '',
        '不使用 RICH，不使用合成补椅子验证，不选择新 checkpoint。逐窗结果、数组及哈希见 RESULT.json / metrics.jsonl / joints/。',
        '原始场景开启结果逐条复用选模记录；三个数据集各首个窗口重新推理核验，在固定绝对误差容限内一致。']
    if local:
        lines += ['', '关闭场景时同时移除全局、逐帧和局部几何字段，显式 use_scene=False；相机输入和当前检查点的预测体形适配保持不变。',
                  '此处“关闭场景”仍是有场景训练的同一个模型，不能标为独立训练的无场景模型。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')


def run(args):
    ensure_bounded()
    if args.device == 'cuda':
        configure_cuda()
    policy = json.loads(args.policy.read_text())
    local = policy['protocol'] == 'native-scene-input-paired-local-v1'
    if local:
        from scenemi.local_training import settings as local_settings, make_model as local_model
    config = (local_settings if local else settings)(ROOT/policy['config'])
    paths = load_paths(args.paths)
    seal_path = ROOT/policy['contract']
    seal = json.loads(seal_path.read_text())
    verify_seal(seal, config, args.scene_paths)
    checkpoint, reference_path = (ROOT/policy[key] for key in ('checkpoint', 'scene_on_reference'))
    if sha256(checkpoint) != policy['checkpoint_sha256'] or sha256(reference_path) != policy['scene_on_reference_sha256']:
        raise ValueError('Pinned checkpoint or selection metrics changed')
    reference = json.loads(reference_path.read_text())
    if (policy['protocol'] not in ('native-scene-input-paired-v1', 'native-scene-input-paired-local-v1') or
            seal['sha256'] != policy['contract_sha256'] or
            reference['contract_sha256'] != seal['sha256'] or reference['step'] != policy['step'] or
            policy['seed'] != config['validation']['seed'] or
            policy['ddim_steps'] != config['validation']['ddim_steps']):
        raise ValueError('Use exactly the predeclared selection checkpoint and sampler')
    if local:
        if (seal['protocol'] != 'native-scene-ego128-local-v3' or
                policy.get('validation_tier') != 'comprehensive' or
                reference.get('selection_tier') != 'comprehensive' or reference.get('provisional') is not False):
            raise ValueError('Local diagnosis requires the fixed comprehensive selection reference')
        cases = seal['validation_cases']['comprehensive']
    else:
        cases = seal['validation_cases']
    if len(reference['rows']) != len(cases):
        raise ValueError('Selection reference coverage changed')
    out = ROOT/policy['output']
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(policy=policy, validation_cases=cases, device=args.device,
        inference_intervention_only=True, training_ablation=False, test_used=False,
        evaluator_sources={name: sha256(ROOT/name) for name in (
            'scenemi/scene_evaluation.py', 'scenemi/benchmark.py', 'scenemi/runtime.py')})
    record = out/'PROTOCOL.json'
    if record.exists() and json.loads(record.read_text()) != protocol:
        raise ValueError('Cannot alter a started paired evaluation protocol')
    atomic_json(record, protocol)
    deterministic(config['seed'], args.device)
    model = (local_model if local else make_model)(config, paths, args.scene_paths)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if state['step'] != policy['step'] or state['contract'] != seal:
        raise ValueError('Selected checkpoint does not match its sealed contract')
    model.load_state_dict(state['model'], strict=True)
    del state
    model = model.to(args.device).eval()
    data = (LocalSceneDataset if local else SceneDataset)(args.scene_paths, model.shape.rest_template[:1].cpu().numpy(),
                        split='validation', config=config)
    arrays = out/'joints'
    arrays.mkdir(exist_ok=True)
    metrics_file = out/'metrics.jsonl'
    rows = [json.loads(line) for line in metrics_file.read_text().splitlines()] if metrics_file.exists() else []
    if len(rows) > len(cases) or [r['case_id'] for r in rows] != list(range(len(rows))):
        raise ValueError('Invalid partial evaluation order')
    for i, row in enumerate(rows):
        if row['identity'] != reference['rows'][i]['identity'] or sha256(arrays/f'{i:04d}.npz') != row['joints_sha256']:
            raise ValueError('Partial evaluation arrays or identities changed')
    started = time.monotonic()
    checked = set()
    parity_rows = []
    try:
        with metrics_file.open('a') as stream:
            for i, (case, original) in enumerate(zip(cases, reference['rows'])):
                # Reproduce one enabled-scene case per corpus, including on resume.
                parity = case['dataset'] not in checked
                if i < len(rows) and not parity:
                    continue
                sample, identity = data.sample(np.random.default_rng(policy['seed']),
                    member=case['member'], start=case['start'], yaw=0.)
                if local:
                    identity['category'] = case['category']
                if identity != original['identity'] or identity['synthetic_chair'] or identity['split'] != 'validation':
                    raise ValueError('Evaluation sample differs from the original selection identity')
                batch = collate([sample], args.device)
                target = forward_kinematics(batch['motion'], batch['target_rest'])[0]
                if parity:
                    on = predict(model, batch, use_scene=True, seed=policy['seed'], steps=policy['ddim_steps'], device=args.device, local=local)
                    metrics = motion_metrics(on, target)
                    check_parity(metrics, original['metrics'])
                    parity_rows.append(dict(case_id=i, dataset=case['dataset'], metrics=metrics))
                    checked.add(case['dataset'])
                if i < len(rows):
                    continue
                off = predict(model, batch, use_scene=False, seed=policy['seed'], steps=policy['ddim_steps'], device=args.device, local=local)
                file = arrays/f'{i:04d}.npz'
                np.savez_compressed(file, scene_off_prediction=off.cpu().numpy(), target=target.cpu().numpy())
                row = dict(case_id=i, identity=identity, metrics=motion_metrics(off, target), joints_sha256=sha256(file))
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                status = dict(state='running', completed=len(rows), total=len(cases), pid=os.getpid(),
                    elapsed_seconds=time.monotonic()-started, updated_at_utc=now())
                atomic_json(out/'STATUS.json', status)
                if len(rows)%20 == 0 or len(rows)==1:
                    print(json.dumps(status), flush=True)
        if sha256(checkpoint) != policy['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during evaluation')
        result = dict(state='complete', checkpoint_sha256=policy['checkpoint_sha256'], step=policy['step'],
            **paired_summary(reference['rows'], rows, category_balanced=local), parity_checks=parity_rows,
            inference_intervention_only=True, training_ablation=False, test_used=False,
            elapsed_seconds=time.monotonic()-started, completed_at_utc=now())
        atomic_json(out/'RESULT.json', result)
        report(out, result)
        atomic_json(out/'STATUS.json', dict(state='complete', completed=len(rows), total=len(cases), updated_at_utc=now()))
        return result
    except BaseException as error:
        atomic_json(out/'STATUS.json', dict(state='failed', completed=len(rows), total=len(cases),
            error=repr(error), updated_at_utc=now()))
        raise


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'components':
        from scenemi.scene_components import main as components_main
        return components_main(argv[1:])
    if argv and argv[0] == 'repair':
        from scenemi.scene_repair import main as repair_main
        return repair_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--scene-paths', type=Path, default=ROOT/'configs/scene_paths.local.yaml')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result['dataset_macro'], indent=2), flush=True)


if __name__ == '__main__':
    main()
