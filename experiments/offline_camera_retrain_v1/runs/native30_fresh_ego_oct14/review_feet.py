"""Read-only checkpoint review on the existing fixed official VAL ego panel."""
import copy
import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config, make_model, window_options, write_json
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1 import native30_validation as validation
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

ROOT = Path(__file__).resolve().parent


def feet_metrics(prediction, truth, rest):
    with torch.no_grad():
        p = forward_kinematics(torch.from_numpy(prediction)[None], torch.from_numpy(rest)[None])[0]
        t = forward_kinematics(torch.from_numpy(truth)[None], torch.from_numpy(rest)[None])[0]
    ids = (7, 8, 10, 11)
    pv, tv = p.diff(dim=0) * 30, t.diff(dim=0) * 30
    stance = tv[:, (10, 11)].norm(dim=-1) < .2
    root_relative = (p[:, ids] - p[:, :1]) - (t[:, ids] - t[:, :1])
    return dict(ankle_and_foot_world_error_mm=float((p[:, ids] - t[:, ids]).norm(dim=-1).mean() * 1000),
        ankle_and_foot_root_relative_error_mm=float(root_relative.norm(dim=-1).mean() * 1000),
        foot_velocity_vector_rmse_m_s=float((pv[:, (10, 11)] - tv[:, (10, 11)]).square().sum(-1).mean() ** .5),
        GT_stance_count=int(stance.sum()),
        foot_speed_during_GT_stance_m_s=float(pv[:, (10, 11)].norm(dim=-1)[stance].mean()) if stance.any() else None,
        GT_foot_speed_during_GT_stance_m_s=float(tv[:, (10, 11)].norm(dim=-1)[stance].mean()) if stance.any() else None)


def main():
    torch.set_num_threads(2)
    config = read_config(HERE / 'native30_fresh_ego_config.json')
    state = torch.load(ROOT / 'training/last.pt', map_location='cpu', weights_only=False)
    assert state['effective_config']['config'] == config
    step = int(state['step'])
    folder = ROOT / f'foot_review_step{step}'
    folder.mkdir(exist_ok=False)
    model = make_model(config).cuda()
    model.load_state_dict(state['model'], strict=True)
    del state
    gc.collect()
    data = Native30Data(HERE / config['train_manifest'], 'validation', config['seed'] + 1,
        frames=config['frames'], memory_protocol=config['scene']['memory_protocol'],
        quality_config=config['gt_reliability'], exo_max_empty_frames=config['exo_max_empty_frames'],
        **window_options(config))
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(),
        json.loads((HERE / config['mount_defaults']).read_text()), config['observation']['input_noise'])
    original_plan = validation.monitoring_plan
    validation.monitoring_plan = lambda d, c, s: [x for x in original_plan(d, c, s)
        if x['panel'] == 'fixed' and x['mode'] == 'ego']
    review_config = copy.deepcopy(config)
    review_config['validation']['scene_ablation_interval_steps'] = 0
    began = time.monotonic()
    rows = validation.validate(model, data, builder, review_config, folder, step)
    summarize(folder, step, time.monotonic() - began)


def summarize(folder, step, sampling_s=None):
    rows = json.loads((folder / 'validation.jsonl').read_text())['cases']
    baseline = json.loads((ROOT / 'training/validation.jsonl').read_text().splitlines()[0])
    old = {(x['sequence_id'], x['regime']): x for x in baseline['cases']
        if x['panel'] == 'fixed' and x['mode'] == 'ego' and x.get('scene_variant', 'scene_on') == 'scene_on'}
    previous = np.load(ROOT / 'training/validation_latest.npz')
    assert int(previous['step']) == 1, 'Baseline archive was replaced; do not compare different steps'
    current = np.load(folder / 'validation_latest.npz')
    comparisons = []
    for row in rows:
        prior = old[(row['sequence_id'], row['regime'])]
        assert row['case_seed'] == prior['case_seed']
        now_key, old_key = f"{row['case']}_{row['regime']}", f"{prior['case']}_{prior['regime']}"
        truth, rest = current[f"{row['case']}_truth"], current[f"{row['case']}_rest"]
        assert np.array_equal(truth, previous[f"{prior['case']}_truth"])
        assert np.array_equal(current[now_key + '_observations'], previous[old_key + '_observations'])
        assert np.array_equal(current[now_key + '_observation_meta'], previous[old_key + '_observation_meta'])
        comparisons.append(dict(sequence_id=row['sequence_id'], dataset=row['dataset'], regime=row['regime'],
            baseline_step=1, current_step=step, baseline_metrics=prior['metrics'], current_metrics=row['metrics'],
            baseline_feet=feet_metrics(previous[old_key + '_prediction'], truth, rest),
            current_feet=feet_metrics(current[now_key + '_prediction'], truth, rest)))
    report = dict(observed_at=datetime.datetime.now().astimezone().isoformat(), checkpoint_step=step,
        optimizer_steps_in_review=0, live_training_modified=False,
        scope='Existing three fixed official VAL ego windows, clean/noisy, same case seeds and identical observations/GT, DDIM20 scene ON; early monitoring only, not full benchmark.',
        elapsed_sampling_s=sampling_s, comparisons=comparisons,
        summary=json.loads((folder / 'validation.jsonl').read_text())['summary'])
    write_json(folder / 'FOOT_REVIEW.json', report)
    print(json.dumps(dict(folder=str(folder), step=step, summary=report['summary'], elapsed_s=report['elapsed_sampling_s'])), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--summarize':
        directory = Path(sys.argv[2])
        summarize(directory, int(np.load(directory / 'validation_latest.npz')['step']))
    else:
        main()
