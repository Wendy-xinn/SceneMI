"""Zero-update complete-main-backbone timing, including actual TRAIN loading."""
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
from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, write_json, code_hashes, sample_item
from experiments.offline_camera_retrain_v1.native30_overnight import scoped_data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_scene_support import attach_support_supervision
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, training_motion
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations

ROOT = Path(__file__).resolve().parent
RUN = HERE / 'runs/native30_scene_relations_oct14/experiment'
REPORT = dict(status='building', optimizer_steps=0, config_changed=False,
    started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), code_sha256=code_hashes(), rows=[],
    scope='Full-backbone zero-update B2/t300: three fixed TRAIN batches and two draws from the unchanged full-corpus sampler; not a complete-corpus mean',
    includes='actual raw30 sample/builder/labels/collation/reference, full model forward/loss/backward, and resident Adam-sized tensors',
    excludes='parameter updates, validation, checkpoint writes; cached-stream timing does not represent random full-corpus I/O')


def main():
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.80)
    state = torch.load(RUN / 'candidate/last.pt', map_location='cpu', weights_only=False)
    config = state['config']
    REPORT['config'] = config
    began = time.monotonic()
    data = scoped_data(config, 'train')
    REPORT['data_index_initialization_s'] = time.monotonic()-began
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(),
        json.loads((HERE/config['mount_defaults']).read_text()), config['observation']['input_noise'])
    plan = json.loads((HERE/'runs/native30_overnight_ik_oct14/experiment/window_plan.json').read_text())['train']
    chosen = [next(v for v in plan if (v['group'], v['mode']) == p)
              for p in [('camera_wearer', 'exo'), ('rich', 'ego')]]
    model = make_model(config).cuda()
    model.load_state_dict(state['model'])
    for p in model.parameters():
        p.requires_grad_(True)
    REPORT['trainable_parameters'] = sum(p.numel() for p in model.parameters())
    # Represent two FP32 Adam moments without constructing/updating optimizer.
    adam_sized = [torch.zeros_like(p) for p in model.parameters() for _ in range(2)]
    REPORT['resident_adam_sized_bytes'] = sum(p.numel()*p.element_size() for p in adam_sized)
    weights = json.loads((HERE/config['prepared_output']/'effective_config.json').read_text())['observation_coefficients']
    del state
    gc.collect()
    full_plan = data.sampling_plan(config['ego_probability'])
    for repeat in range(5):
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rng = np.random.default_rng(2026101450+repeat)
        torch.cuda.synchronize()
        begun = time.monotonic()
        items = []
        identities = []
        source_times = []
        for entry in chosen:
            a = time.monotonic()
            if repeat < 3:
                sample, identity = data.sample(entry['group'], entry['mode'], member=entry['member'],
                    start=entry['start'])
                joints = None if entry['mode']=='exo' else ((15,) if rng.random()<config['ego_head_only_probability'] else (15,20,21))
                sample, scene, asset = builder.build(data, sample, identity, entry['mode'], joints, noise_rng=rng)
                sample = attach_support_supervision(sample, scene, asset)
                items.append((sample, scene, asset, entry['mode']))
            else:
                item, identity = sample_item(data, builder, config, full_plan, rng)
                sample, scene, asset, mode = item
                items.append(item)
            identities.append(dict(group=identity['group'], mode=identity['mode'], sequence_id=identity['sequence_id'],
                                   start=identity['start'], yaw=identity['yaw_augmentation_rad']))
            source_times.append(time.monotonic()-a)
        batch = collate_observations(items)
        known = model_inputs(config, batch)
        times = torch.full((2,), 300, device='cuda', dtype=torch.long)
        alpha = model.alphas[times]
        target = training_motion(config, batch, known)
        generator = torch.Generator(device='cuda').manual_seed(2026101450+repeat)
        noise = torch.randn(target.shape, device='cuda', generator=generator)
        noisy = alpha[:,None,None].sqrt()*target+(1-alpha[:,None,None]).sqrt()*noise
        torch.cuda.synchronize()
        ready = time.monotonic()
        model.train()
        prediction, details = model(noisy, times, known, use_scene=True, return_details=True)
        loss, terms = objective(prediction, batch['motion'], batch, alpha, details, config, weights)
        loss.backward()
        torch.cuda.synchronize()
        ended = time.monotonic()
        finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        row = dict(repeat=repeat, sampling='fixed_members' if repeat<3 else 'unchanged_full_corpus_sampler',
                   identities=identities, source_build_s=source_times, preparation_s=ready-begun,
                   full_forward_loss_backward_s=ended-ready, total_s=ended-begun, loss=float(loss.detach()),
                   finite=finite, peak_allocated_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
                   accepted_physical_iterations=float(details['physical_accepted_steps'].sum()),
                   static_map_cache_entries=len(data.scene_memory.maps))
        REPORT['rows'].append(row)
        REPORT['status']='probing'
        write_json(ROOT/'FULL_BACKBONE_TIMING.json', REPORT)
        print('FULL_BACKBONE', row, flush=True)
        if not finite:
            raise FloatingPointError('Nonfinite complete-model probe')
        del items, batch, known, times, alpha, target, noise, noisy, prediction, details, loss, terms, sample, scene, asset
        gc.collect()
    REPORT.update(status='passed', finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())


if __name__=='__main__':
    write_json(ROOT/'FULL_BACKBONE_TIMING.json', REPORT)
    try:
        main()
    except BaseException:
        import traceback
        REPORT.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        write_json(ROOT/'FULL_BACKBONE_TIMING.json', REPORT)
