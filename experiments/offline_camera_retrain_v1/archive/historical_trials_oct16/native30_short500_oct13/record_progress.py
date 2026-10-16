"""Summarize saved logs; no model execution, source mutation, or GPU access."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'training'


def json_lines(path):
    result = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return result


def snapshot():
    record_path = ROOT / 'SHORT500_PROGRESS.json'
    record = json.loads(record_path.read_text())
    trace = json_lines(RUN / 'trace.jsonl')
    protocol = json.loads((RUN / 'protocol.json').read_text())
    validation = json_lines(RUN / 'validation.jsonl')
    metrics = ['mpjpe_mm', 'conditioned_mpjpe_mm', 'unconditioned_mpjpe_mm',
               'root_position_error_mm', 'root_orientation_deg', 'head_orientation_deg',
               'foot_speed_during_GT_stance_m_s']
    aggregates = []
    for report in validation:
        groups = defaultdict(list)
        for case in report['cases']:
            groups[(case['regime'], case['mode'])].append(case)
            groups[(case['regime'], case['group'] + '/' + case['mode'])].append(case)
        for (regime, stream), cases in sorted(groups.items()):
            values = {}
            for key in metrics:
                selected = [c['metrics'][key] for c in cases
                            if c['metrics'][key]['count'] and c['metrics'][key]['value'] is not None]
                count = sum(x['count'] for x in selected)
                values[key] = dict(value=sum(x['value'] * x['count'] for x in selected) / count
                                   if count else None, count=count)
            aggregates.append(dict(step=report['step'], regime=regime, stream=stream,
                                   cases=len(cases), metrics=values))
    record.update(captured_at_utc=datetime.now(timezone.utc).isoformat(), status=protocol['status'],
                  planned_steps=protocol['planned_steps'], trace_completed_step=trace[-1]['step'],
                  checkpoint_completed_step=protocol['completed_steps'],
                  trace_updated_seconds_ago=time.time() - (RUN / 'trace.jsonl').stat().st_mtime,
                  all_recorded_losses_finite=all(math.isfinite(x['loss']) for x in trace),
                  all_recorded_gradient_norms_finite=all(math.isfinite(x['gradient_norm']) for x in trace),
                  sample_counts=dict(Counter(i['group'] + '/' + i['mode']
                                             for row in trace for i in row['identities'])),
                  validation_steps=[row['step'] for row in validation], validation_aggregates=aggregates,
                  checkpoint=dict(path='training/last.pt', bytes=(RUN / 'last.pt').stat().st_size,
                                  policy=protocol['checkpoint_policy']),
                  disk=dict(linux_free_bytes=shutil.disk_usage(ROOT).free,
                            e_drive_free_bytes=shutil.disk_usage('/mnt/e').free),
                  training_elapsed_s_at_checkpoint=protocol.get('elapsed_s'),
                  scene_condition_on_steps=sum(row['scene'] for row in trace),
                  geometry_diagnostics={key:dict(nonzero_steps=sum(abs(row['terms']['geometry'][key]) > 1e-12
                                                                 for row in trace),
                                                 max_abs=max(abs(row['terms']['geometry'][key]) for row in trace))
                                        for key in trace[0]['terms']['geometry']})
    if (RUN / 'CHECKPOINT_AUDIT.json').exists():
        record['checkpoint_audit'] = json.loads((RUN / 'CHECKPOINT_AUDIT.json').read_text())
    temporary = record_path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, record_path)
    print(json.dumps({key: record[key] for key in ['status', 'trace_completed_step',
                                                  'checkpoint_completed_step', 'validation_steps']}))
    for row in aggregates:
        if row['step'] == validation[-1]['step'] and row['stream'] in ('ego', 'exo'):
            print(row['step'], row['regime'], row['stream'],
                  json.dumps({key: round(value['value'], 3) if value['value'] is not None else None
                              for key, value in row['metrics'].items()}))


if __name__ == '__main__':
    snapshot()
