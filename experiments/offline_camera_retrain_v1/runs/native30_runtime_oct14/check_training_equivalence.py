"""Check real TRAIN labels, forward/loss/backward without updating weights."""
import contextlib
import datetime as dt
import gc
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import time
import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, write_json, code_hashes
from experiments.offline_camera_retrain_v1.native30_overnight import scoped_data, build_bank, augment
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_geometry import Native30SurfaceWindow
from experiments.offline_camera_retrain_v1.native30_scene_support import attach_support_supervision
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, training_motion
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations

ROOT = Path(__file__).resolve().parent
RUN = HERE / 'runs/native30_scene_relations_oct14/experiment'
report = dict(status='building', optimizer_steps=0, config_changed=False,
              started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), code_sha256=code_hashes(),
              scope='One real TRAIN B2: EgoBody master exo and RICH ego; same noise, yaw, t300; no parameter updates')


@contextlib.contextmanager
def original_queries():
    names = ('query_motion', 'query_wide_motion', 'query_nearest', 'query_wide')
    saved = {k: getattr(Native30SurfaceWindow, k) for k in names}
    def motion(self, v, r, device, neighbors=8):
        return self.query_reference(v, r, neighbors)
    def nearest(self, v, r):
        return self.query_reference(v, r)
    Native30SurfaceWindow.query_motion = motion
    Native30SurfaceWindow.query_wide_motion = motion
    Native30SurfaceWindow.query_nearest = nearest
    Native30SurfaceWindow.query_wide = Native30SurfaceWindow.query_reference
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(Native30SurfaceWindow, k, v)


def main():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.40)
    saved = torch.load(RUN / 'candidate/last.pt', map_location='cpu', weights_only=False)
    config = saved['config']
    plan = json.loads((HERE / 'runs/native30_overnight_ik_oct14/experiment/window_plan.json').read_text())['train']
    chosen = [next(v for v in plan if (v['group'], v['mode']) == pair)
              for pair in [('camera_wearer', 'exo'), ('rich', 'ego')]]
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(),
        json.loads((HERE / config['mount_defaults']).read_text()), config['observation']['input_noise'])
    bank, ids = build_bank(scoped_data(config, 'train'), builder, chosen)
    report['identities'] = ids
    labelled = []
    for sample, scene, asset, mode in bank:
        with original_queries():
            before = attach_support_supervision(sample, scene, asset)
        after = attach_support_supervision(sample, scene, asset)
        if set(before) != set(after) or any(not np.array_equal(before[k], after[k]) for k in before):
            raise AssertionError('TRAIN supervision changed')
        labelled.append((after, scene, asset, mode))
    report['training_supervision_bitwise_equal'] = True
    model = make_model(config).cuda()
    model.load_state_dict(saved['model'])
    prefixes = tuple(saved['meta']['trainable_common_prefixes']) + tuple(saved['meta']['trainable_scene_prefixes'])
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(prefixes))
    weights = json.loads((HERE / config['prepared_output'] / 'effective_config.json').read_text())['observation_coefficients']
    del saved, bank
    gc.collect()
    items = [augment(item, identity, builder, config, np.random.default_rng(1400 + i), noisy=True, yaw=True)
             for i, (item, identity) in enumerate(zip(labelled, ids))]
    batch = collate_observations(items)
    known = model_inputs(config, batch)
    times = torch.full((2,), 300, device='cuda', dtype=torch.long)
    alpha = model.alphas[times]
    target = training_motion(config, batch, known)
    generator = torch.Generator(device='cuda').manual_seed(1426)
    noise = torch.randn(target.shape, device='cuda', generator=generator)
    state = alpha[:, None, None].sqrt() * target + (1-alpha[:, None, None]).sqrt() * noise
    rows = {}
    report['gradient_comparison_tolerance'] = dict(atol=1e-7, rtol=1e-6)
    for name in ('reference', 'reference_repeat', 'optimized'):
        manager = original_queries() if name.startswith('reference') else contextlib.nullcontext()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(14003)
        torch.cuda.manual_seed_all(14003)
        model.train()
        model.core.scene_embedding.eval()
        with manager:
            torch.cuda.synchronize()
            began = time.monotonic()
            prediction, details = model(state, times, known, use_scene=True, return_details=True)
            loss, terms = objective(prediction, batch['motion'], batch, alpha, details, config, weights)
            loss.backward()
            torch.cuda.synchronize()
            gradients = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None}
            rows[name] = dict(seconds=time.monotonic()-began, loss=float(loss.detach()),
                              accepted_physical_iterations=float(details['physical_accepted_steps'].sum()),
                              finite=bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in gradients.values()))
            if name == 'reference':
                baseline = dict(prediction=prediction.detach().cpu(), gradients=gradients,
                                terms={g: {k: float(v.detach()) for k, v in values.items()} for g, values in terms.items()})
            else:
                comparison = dict(prediction_bitwise_equal=torch.equal(baseline['prediction'], prediction.detach().cpu()),
                    loss_equal=rows['reference']['loss'] == rows[name]['loss'],
                    objective_terms_equal=baseline['terms'] == {g: {k: float(v.detach()) for k, v in values.items()} for g, values in terms.items()},
                    gradient_tensor_count=len(gradients),
                    gradient_bitwise_equal=set(gradients) == set(baseline['gradients']) and all(torch.equal(v, baseline['gradients'][k]) for k, v in gradients.items()),
                    gradients_close=set(gradients) == set(baseline['gradients']) and all(torch.allclose(v, baseline['gradients'][k], atol=1e-7, rtol=1e-6) for k, v in gradients.items()),
                    maximum_absolute_gradient_difference=max(float((v-baseline['gradients'][k]).abs().max()) for k, v in gradients.items()))
                if name == 'reference_repeat':
                    report['reference_repeat_comparison'] = comparison
                else:
                    report.update(comparison)
            print(name, rows[name], flush=True)
            del prediction, details, loss, terms, gradients
    report['timings'] = rows
    if not all(report[k] for k in ('prediction_bitwise_equal', 'loss_equal', 'objective_terms_equal', 'gradients_close')):
        raise AssertionError('TRAIN forward/loss/gradients changed')
    report.update(status='passed', finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())


if __name__ == '__main__':
    write_json(ROOT / 'TRAIN_RUNTIME_EQUIVALENCE.json', report)
    try:
        main()
    except BaseException:
        import traceback
        report.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        write_json(ROOT / 'TRAIN_RUNTIME_EQUIVALENCE.json', report)
