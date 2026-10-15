"""Read scheduled VAL artifacts and track feet without touching the trainer."""
import datetime
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from review_feet import feet_metrics

ROOT = Path(__file__).resolve().parent


def collect():
    review = json.loads((ROOT / 'foot_review_step600/FOOT_REVIEW.json').read_text())
    prior = {(r['sequence_id'], r['regime']): r for r in review['comparisons']}
    entries = [dict(step=r['baseline_step'], dataset=r['dataset'], sequence_id=r['sequence_id'],
        regime=r['regime'], source='initial scheduled VAL', metrics=r['baseline_feet']) for r in review['comparisons']]
    entries += [dict(step=r['current_step'], dataset=r['dataset'], sequence_id=r['sequence_id'],
        regime=r['regime'], source='independent same-seed checkpoint review', metrics=r['current_feet']) for r in review['comparisons']]
    previous = ROOT / 'FOOT_TREND.json'
    if previous.exists():
        persisted = json.loads(previous.read_text())['entries']
        by_key = {(r['step'], r['sequence_id'], r['regime']): r for r in entries}
        for row in persisted:
            if (row['sequence_id'], row['regime']) not in prior:
                raise ValueError('Persisted foot trend contains an unknown fixed case')
            if not all(np.isfinite(v) for v in row['metrics'].values() if v is not None):
                raise ValueError('Persisted foot trend has nonfinite metrics')
            key = row['step'], row['sequence_id'], row['regime']
            if key in by_key and by_key[key] != row:
                raise ValueError('Persisted baseline foot metrics changed')
            by_key[key] = row
        entries = sorted(by_key.values(), key=lambda r: (r['step'], r['sequence_id'], r['regime']))
    counts = {}
    for row in entries:
        counts[row['step']] = counts.get(row['step'], 0) + 1
    if any(n != len(prior) for n in counts.values()):
        raise ValueError('Persisted foot trend has an incomplete panel')
    seen = set(counts)
    torch.set_num_threads(1)
    while True:
        with np.load(ROOT / 'training/validation_latest.npz') as outputs:
            step = int(outputs['step'])
            if step not in seen:
                lines = (ROOT / 'training/validation.jsonl').read_text().splitlines()
                matched = [json.loads(line) for line in lines if json.loads(line)['step'] == step]
                if len(matched) != 1:
                    raise ValueError('VAL outputs and records disagree')
                for row in matched[0]['cases']:
                    if row['panel'] != 'fixed' or row['mode'] != 'ego' or row.get('scene_variant', 'scene_on') != 'scene_on':
                        continue
                    key = row['sequence_id'], row['regime']
                    reference = prior[key]
                    with np.load(ROOT / 'foot_review_step600/validation_latest.npz') as check:
                        original = [r for r in json.loads((ROOT / 'foot_review_step600/validation.jsonl').read_text())['cases'] if (r['sequence_id'], r['regime']) == key][0]
                        now_key = f"{row['case']}_{row['regime']}"
                        old_key = f"{original['case']}_{original['regime']}"
                        if row['case_seed'] != original['case_seed']:
                            raise ValueError('Fixed VAL seed changed')
                        for suffix in ('observations', 'observation_meta'):
                            if not np.array_equal(outputs[now_key + '_' + suffix], check[old_key + '_' + suffix]):
                                raise ValueError('Fixed VAL observations changed')
                        if not np.array_equal(outputs[f"{row['case']}_truth"], check[f"{original['case']}_truth"]):
                            raise ValueError('Fixed VAL ground truth changed')
                    metrics = feet_metrics(outputs[now_key + '_prediction'], outputs[f"{row['case']}_truth"], outputs[f"{row['case']}_rest"])
                    entries.append(dict(step=step, dataset=reference['dataset'], sequence_id=key[0], regime=key[1],
                        source='scheduled fixed official VAL, same seed/observations/GT', metrics=metrics))
                seen.add(step)
        protocol = json.loads((ROOT / 'training/protocol.json').read_text())
        stopped = protocol['status'] in ('completed', 'stopped', 'stopped_by_user')
        if stopped:
            with np.load(ROOT / 'training/validation_latest.npz') as final_outputs:
                if int(final_outputs['step']) not in seen:
                    continue
        report = dict(status='completed' if stopped else 'monitoring', pid=os.getpid(),
            observed_at=datetime.datetime.now().astimezone().isoformat(),
            scope='Three existing fixed official VAL ego windows, clean/noisy. Track position and stance speed separately; no convergence guarantee.',
            live_training_modified=False, latest_val_step=max(seen), entries=entries)
        temporary = ROOT / 'FOOT_TREND.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(ROOT / 'FOOT_TREND.json')
        if stopped:
            break
        time.sleep(20)


if __name__ == '__main__':
    collect()
