"""Zero-update full-model equivalence/timing for GPU radius membership."""
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config, make_model, window_options, write_json
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_support_cache import cached_support
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, training_motion
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.train_native30 import model_digest
from experiments.offline_camera_retrain_v1 import native30_query_runtime as runtime
from gpu_radius_probe import query_gpu

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.4)
    state = torch.load(ROOT / 'training/last.pt', map_location='cpu', weights_only=False)
    config = read_config(HERE / 'native30_fresh_ego_config.json')
    assert state['effective_config']['config'] == config
    model = make_model(config).cuda()
    model.load_state_dict(state['model'], strict=True)
    weights = state['effective_config']['observation_coefficients']
    report = dict(status='building', observed_at=datetime.datetime.now().astimezone().isoformat(),
        checkpoint_step=int(state['step']), optimizer_steps=0, live_training_modified=False,
        scope='Three predefined TRAIN B2 pairs covering three corpora and ego/exo; same checkpoint, noise and dropout seeds, FP32, full forward/loss/backward. Concurrent trainer; excludes prep/clip/Adam/validation.',
        deterministic_algorithms=True, gradient_tolerance=dict(atol=1e-7, rtol=1e-6),
        prototype_sha256=hashlib.sha256((ROOT / 'gpu_radius_probe.py').read_bytes()).hexdigest(),
        initial_model_sha256=model_digest(model), cases=[])
    write_json(ROOT / 'GPU_FULL_TRAINING_EQUIVALENCE.json', report)
    del state
    gc.collect()
    data = Native30Data(HERE / config['train_manifest'], 'train', config['seed'], frames=config['frames'],
        memory_protocol=config['scene']['memory_protocol'], quality_config=config['gt_reliability'],
        exo_max_empty_frames=config['exo_max_empty_frames'], **window_options(config))
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(),
        json.loads((HERE / config['mount_defaults']).read_text()), config['observation']['input_noise'])
    bank = {}
    for group, mode in (('trumans', 'ego'), ('camera_wearer', 'ego'), ('rich', 'ego'), ('camera_wearer', 'exo')):
        member = next(i for i in range(len(data.members[group])) if len(data.starts_for(group, i, mode)))
        start = int(data.starts_for(group, member, mode)[0])
        sample, identity = data.sample(group, mode, member=member, start=start, yaw=.6)
        sample, scene, asset = builder.build(data, sample, identity, mode, (15,) if mode == 'ego' else None)
        sample = cached_support(data, builder, sample, identity)
        bank[group, mode] = (sample, scene, asset, mode)
    original = runtime.query_surfaces_cuda
    for pair in [(('trumans', 'ego'), ('camera_wearer', 'ego')),
                 (('trumans', 'ego'), ('rich', 'ego')),
                 (('camera_wearer', 'exo'), ('rich', 'ego'))]:
        batch = collate_observations([bank[k] for k in pair])
        known = model_inputs(config, batch)
        times = torch.full((2,), 300, device='cuda', dtype=torch.long)
        signal = model.alphas[times]
        target = training_motion(config, batch, known)
        generator = torch.Generator(device='cuda').manual_seed(6621)
        noisy = signal[:, None, None].sqrt() * target + (1-signal[:, None, None]).sqrt() * torch.randn(target.shape, device='cuda', generator=generator)
        case = dict(pair=pair, rows=[])
        baseline = None
        try:
            for variant in ('original_warmup', 'original', 'gpu', 'gpu', 'original'):
                runtime.query_surfaces_cuda = query_gpu if variant == 'gpu' else original
                model.train()
                model.zero_grad(set_to_none=True)
                torch.manual_seed(10047)
                torch.cuda.manual_seed_all(10047)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                began = time.monotonic()
                prediction, details = model(noisy, times, known, use_scene=True, return_details=True)
                loss, terms = objective(prediction, batch['motion'], batch, signal, details, config, weights)
                loss.backward()
                torch.cuda.synchronize()
                seconds = time.monotonic() - began
                gradients = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
                values = {g: {k: float(v.detach()) for k, v in terms_.items()} for g, terms_ in terms.items()}
                row = dict(variant=variant, seconds=seconds, loss=float(loss.detach()),
                    peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,
                    finite=bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in gradients.values()))
                if baseline is None:
                    baseline = dict(prediction=prediction.detach().cpu().clone(), gradients=gradients, loss=row['loss'], terms=values)
                else:
                    row.update(prediction_bitwise_equal=torch.equal(prediction.detach().cpu(), baseline['prediction']),
                        loss_equal=row['loss'] == baseline['loss'], all_loss_terms_equal=values == baseline['terms'],
                        gradient_max_abs_difference=max(float((g-baseline['gradients'][n]).abs().max()) for n, g in gradients.items()),
                        gradients_close=all(torch.allclose(g, baseline['gradients'][n], atol=1e-7, rtol=1e-6) for n, g in gradients.items()))
                    if not all(row[k] for k in ('prediction_bitwise_equal', 'loss_equal', 'all_loss_terms_equal', 'gradients_close', 'finite')):
                        raise AssertionError(row)
                case['rows'].append(row)
                print(json.dumps(dict(pair=pair, **row)), flush=True)
                del prediction, details, loss, terms, gradients
        finally:
            runtime.query_surfaces_cuda = original
        del baseline
        report['cases'].append(case)
        write_json(ROOT / 'GPU_FULL_TRAINING_EQUIVALENCE.json', report)
    report.update(status='passed', model_parameters_unchanged=model_digest(model) == report['initial_model_sha256'])
    assert report['model_parameters_unchanged']
    write_json(ROOT / 'GPU_FULL_TRAINING_EQUIVALENCE.json', report)


if __name__ == '__main__':
    main()
