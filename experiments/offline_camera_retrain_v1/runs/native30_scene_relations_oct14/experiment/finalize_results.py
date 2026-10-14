"""Finish the registered trial's compact report after an interface disconnect.

Reads saved metrics only. No training, checkpoint editing, or automatic full run.
"""
import datetime as dt
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
MASTER = ROOT.parents[1] / 'exo_occlusion_review_oct12/EXPERIMENT_AND_ARCHITECTURE_zh.md'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def average(rows, mode, regime, metric='mpjpe_mm', field='metrics', seed=None):
    entries = [r[field][metric] for r in rows if r['mode'] == mode and r['regime'] == regime
               and (seed is None or r['seed'] == seed)]
    known = [v for v in entries if v['count'] and v['value'] is not None]
    count = sum(v['count'] for v in known)
    return sum(v['value'] * v['count'] for v in known) / count if count else None


def finalize(protocol):
    report = dict(status=protocol['status'], full_training_started=False, test_used=False,
                  finalized_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  initialization=protocol['initialization'], updates=protocol['updates'])
    gate_path = ROOT / 'scene_gate.json'
    if not gate_path.exists():
        report['failure'] = read(ROOT / 'failure.json') if (ROOT / 'failure.json').exists() else 'Process exited without completed gate'
        write(ROOT / 'RESULT_SUMMARY.json', report)
        return
    report['gate'] = read(gate_path)
    candidate = read(ROOT / 'candidate/evaluation_latest.json')
    control = read(ROOT / 'control/evaluation_latest.json')
    full = [r for r in candidate['rows'] if r['scene'] == 'full']
    off = [r for r in candidate['rows'] if r['scene'] == 'off']
    report['ddim_steps'] = candidate['ddim_steps']; report['seeds'] = candidate['seeds']
    comparisons = []
    for mode in ('ego', 'exo'):
        for regime in ('clean', 'noisy'):
            row = dict(mode=mode, regime=regime,
                       initial_mm=average(full, mode, regime, field='reference_metrics'),
                       scene_mm=average(full, mode, regime),
                       off_mm=average(off, mode, regime),
                       control_mm=average(control['rows'], mode, regime),
                       hidden_scene_mm=average(full, mode, regime, 'unconditioned_mpjpe_mm'),
                       hidden_off_mm=average(off, mode, regime, 'unconditioned_mpjpe_mm'),
                       trusted_scene_mm=average(full, mode, regime, 'conditioned_mpjpe_mm'),
                       root_scene_mm=average(full, mode, regime, 'root_position_error_mm'))
            row['foot_speed_stance_m_s'] = {name: average(rows, mode, regime, 'foot_speed_during_GT_stance_m_s')
                                              for name, rows in (('scene', full), ('off', off), ('control', control['rows']))}
            row['observed_plane_proxy_mm'] = {name: average(rows, mode, regime, 'observed_plane_behind_proxy_mm', 'scene_metrics')
                                              for name, rows in (('scene', full), ('off', off))}
            comparisons.append(row)
    report['comparison'] = comparisons
    report['diagnostic_ablations'] = []
    for name in ('without_guidance', 'mismatched_scene'):
        path = ROOT / name / 'evaluation_latest.json'
        if not path.exists(): continue
        diagnostic = read(path); seed = diagnostic['seeds'][0]
        for mode in ('ego', 'exo'):
            for regime in ('clean', 'noisy'):
                metric = 'mpjpe_mm' if mode == 'ego' else 'unconditioned_mpjpe_mm'
                report['diagnostic_ablations'].append(dict(ablation=name, mode=mode, regime=regime,
                    metric=metric, seed=seed, correct_scene=average(full, mode, regime, metric, seed=seed),
                    ablated=average(diagnostic['rows'], mode, regime, metric, seed=seed)))
    trace = [json.loads(line) for line in (ROOT / 'candidate/trace.jsonl').read_text().splitlines()]
    report['candidate_elapsed_training_s'] = trace[-1]['elapsed_s']
    report['accepted_training_physical_iterations'] = sum(r['physical_accepted'] for r in trace)
    report['not_a_formal_full_VAL_benchmark'] = True
    write(ROOT / 'RESULT_SUMMARY.json', report)
    passed = report['gate']['passed']; status = '通过' if passed else '未通过'
    table = ['\n<!-- V4_SCENE_RESULTS_BEGIN -->', '\n本轮200步匹配试验已完成，独立VAL场景门槛：**' + status + '**。全训练池未启动。单位mm；原始世界坐标、无GT对齐，DDIM20两个预先固定种子。',
             '\n|条件|初始整体|场景整体|同权重关闭场景|匹配关闭场景训练|场景缺失部位|关闭场景缺失部位|场景可信部位|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        keys = ('initial_mm','scene_mm','off_mm','control_mm','hidden_scene_mm','hidden_off_mm','trusted_scene_mm')
        table.append('|' + r['mode'] + ' ' + r['regime'] + '|' + '|'.join(f'{r[k]:.2f}' for k in keys) + '|')
    failed = [k for k, v in report['gate']['checks'].items() if not v['passed']]
    table += ['\n未通过项：' + ('、'.join(f'`{k}`' for k in failed) if failed else '无') + '。',
              '\n[完整比较及物理／错配消融](../native30_scene_relations_oct14/experiment/RESULT_SUMMARY.json)、[原始验收](../native30_scene_relations_oct14/experiment/scene_gate.json)。弱标签覆盖与求解器能量下降是链路证据，不能代替上述独立动作指标。',
              '\n[clean mesh](../native30_scene_relations_oct14/experiment/candidate/VALIDATION_MESH_CLEAN_STEP4200.png)、[noisy mesh](../native30_scene_relations_oct14/experiment/candidate/VALIDATION_MESH_NOISY_STEP4200.png)，GT／参考／生成均保持原世界位置。',
              '\n<!-- V4_SCENE_RESULTS_END -->\n']
    text = MASTER.read_text()
    marker = '<!-- V4_SCENE_RESULTS_BEGIN -->'
    if marker in text:
        start = text.index(marker); end = text.index('<!-- V4_SCENE_RESULTS_END -->', start) + len('<!-- V4_SCENE_RESULTS_END -->')
        text = text[:start] + '\n'.join(table) + text[end:]
    else: text += '\n'.join(table)
    text = text.replace('**三项已接入；63项测试及真实数据反传通过，200步匹配实验运行中**',
                        '**三项已接入；200步匹配实验完成，场景门槛' + status + '**')
    text = text.replace('**三项已接入；33项测试及真实数据反传通过，200步匹配实验运行中**',
                        '**三项已接入；200步匹配实验完成，场景门槛' + status + '**')
    text = text.replace('下列试验正在运行，尚未获得新模型的独立VAL结论。',
                        '下列试验已完成，新模型独立VAL门槛' + status + '，具体结果在本节末表中。')
    temp = MASTER.with_suffix('.md.tmp'); temp.write_text(text); temp.replace(MASTER)


if __name__ == '__main__':
    pid = read(ROOT / 'launcher.json')['pid']
    while True:
        protocol = read(ROOT / 'protocol.json')
        if protocol['status'].startswith('completed_') or (ROOT / 'failure.json').exists(): break
        try: os.kill(pid, 0)
        except ProcessLookupError: break
        time.sleep(10)
    finalize(protocol)
