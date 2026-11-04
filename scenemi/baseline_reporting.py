"""Assemble completed pretrained validation results into compact CSV and JSON."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

METRICS = ('mpjpe_mm', 'pa_mpjpe_mm', 'root_error_mm', 'head_error_mm',
           'mpjve_mm_s', 'jitter_100m_s3', 'foot_sliding_avatarjlm_cm_frame',
           'foot_sliding_horizontal_mm_contact_frame', 'egoallo_cumulative_foot_skate_m')


def report(root):
    models = ('ours', 'egoallo', 'egogvae', 'egoallo_raw')
    values, rows, sources = {}, [], {}
    for model in models:
        path = root / model / 'RESULT.json'
        result = json.loads(path.read_text())
        if result['state'] != 'complete' or result['windows'] != 5693:
            raise ValueError('All validation windows must complete before the comparison is final')
        values[model] = result
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for dataset in ('amass', 'egobody', 'trumans', 'grail', 'macro'):
            metrics = result['dataset_macro_mean'] if dataset == 'macro' else result['by_dataset'][dataset]
            rows.append(dict(model=model, dataset=dataset,
                windows=5693 if dataset == 'macro' else result['counts'][dataset],
                **{key: metrics[key] for key in METRICS}))
    if any(values[m]['counts'] != values['ours']['counts'] for m in models):
        raise ValueError('Models used different validation membership')
    path = root / 'COMPARISON.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('model', 'dataset', 'windows') + METRICS)
        writer.writeheader(); writer.writerows(rows)
    result = dict(state='complete', rows=rows, source_sha256=sources,
        evaluation_protocol_sha256=hashlib.sha256((root / 'EVALUATION_PROTOCOL.json').read_bytes()).hexdigest(),
        all_models_same_validation=True, test_used=False,
        baseline_fine_tuning=False,
        caveats=[
            'Released baselines were not fine-tuned on this four-corpus training set.',
            'Released EgoAllo was pretrained on AMASS train+val; AMASS validation is not unseen pretraining data.',
            'Baselines decode SMPL-H; the supplied table uses native SMPL/SMPL-X/GRAIL targets with shared 22-joint ordering.',
            'Pinned EgoGVAE evaluation imports missing jitter/skating helpers; explicit cited-source formulas and units are recorded in EVALUATION_PROTOCOL.json.',
            'Ground-truth contacts are used only in metric computation, never as inference inputs.',
            'Reported Jitter is jerk divided by 100; smaller jitter alone is not proof of better motion accuracy.',
        ])
    (root / 'COMPARISON.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/pretrained_ego_baselines'))
    report(parser.parse_args().root)
