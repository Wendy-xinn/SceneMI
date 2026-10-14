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
    seen = {1, 600}
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
