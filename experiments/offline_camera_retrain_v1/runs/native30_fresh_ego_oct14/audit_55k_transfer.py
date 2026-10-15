"""Audit saved objectives and tensor compatibility without loading a live model."""
import datetime
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
HERE = ROOT.parents[1]


def main():
    old_dir = ROOT.parent / 'native_dynamic_scene20_contact_55k_oct07'
    old = torch.load(old_dir / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
    new = torch.load(ROOT / 'training/last.pt', map_location='cpu', weights_only=False, mmap=True)
    old_weights, new_weights = old['model'], new['model']
    matched = [k for k, v in old_weights.items() if k in new_weights and v.shape == new_weights[k].shape]
    old_logged = json.loads((old_dir / 'training_log.jsonl').open().readline())['losses']
    required = ['gait_stance_velocity_m2_s2', 'gait_swing_velocity_m2_s2',
        'gait_support_height_m2', 'gait_swing_height_m2', 'gait_relative_position_m2',
        'gait_stride_displacement_m2', 'coordination_amplitude_m2', 'coordination_relation_mse']
    assert old['step'] == 55000 and old['config']['loss_profile'] == 'coordination_v1'
    assert all(k in old_logged for k in required)
    config = new['effective_config']['config']
    report = dict(observed_at=datetime.datetime.now().astimezone().isoformat(),
        optimizer_steps=0, live_training_modified=False, old_checkpoint_step=old['step'],
        current_checkpoint_step=new['step'], old_loss_profile=old['config']['loss_profile'],
        old_gait_terms_confirmed_in_actual_training_log={k: old_logged[k] for k in required},
        source_sha256={n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ('supervision.py', 'coordination.py', 'native30_losses.py')},
        comparison=[
            dict(term='Foot velocity', old='Separate GT-stance 0.10 and swing 0.05 masked means, alpha signal gate',
                current='All foot velocity vector MSE 0.075, alpha squared gate and derivative reliability mask', identical=False),
            dict(term='Foot positions relative to pelvis', old='Dedicated term weight 1',
                current='Only implicit in full-body pose/FK position, no dedicated term', identical=False),
            dict(term='Multi-frame foot stride', old='Relative displacement lags 4,8 at20Hz (0.2,0.4s), weight2',
                current='No corresponding foot stride term; root/head turning lags are distinct', identical=False),
            dict(term='GT support/swing foot height', old='Separate weighted height and support envelope/penetration proxies',
                current='Full pose plus observed-surface contact/plane objectives; unknown geometry unpenalized', identical=False),
            dict(term='Limb coordination/amplitude', old='Dedicated lagged relations weight0.05 and temporal amplitude weight5',
                current='No corresponding temporal limb-amplitude term; head-root orientation coordination is distinct', identical=False),
            dict(term='Current v6 to v7 foot objective', old='Native30 v6', current='Same native30 loss and geometry sources, checked by prior FOOT_SUPERVISION_AUDIT.json', identical=True)],
        correction='Prior no-change audit was scoped to Native30 v6->v7. It does not establish equivalence to the historical 55k objective. Missing dedicated terms are a plausible hypothesis for suppressed articulation, not a demonstrated sole cause.',
        transfer=dict(old_tensor_count=len(old_weights), current_tensor_count=len(new_weights),
            matching_name_and_shape_count=len(matched),
            old_unmatched=[k for k in old_weights if k not in matched],
            current_unmatched=[k for k in new_weights if k not in matched],
            strict_resume_compatible=False, old_fps=20, current_fps=30,
            old_target='absolute motion x0', current_target=config['diffusion_target'],
            semantic_limit='Matching tensor shapes do not establish matching input/output meanings. New residual conditioning, observation encoder and scene-relation/refinement modules need an explicit transfer experiment.',
            official_split_limit='Historical internal TRAIN included 12 official EgoBody TEST recordings, documented in master section2. Not a clean initialization for that official benchmark.',
            possible_use='Clearly labeled development-only control with explicit weight mapping; not started.'),
        proposed_next_stage=dict(status='proposal only; not activated',
            preserve_current_run='Complete registered10000 updates and retain checkpoint; no live loss/config/schedule mutation.',
            priority='Validate articulation-oriented objective against unchanged-objective control from the same clean current checkpoint, matched samples/seeds.',
            candidate='Restore dedicated root-relative foot position/stride and limb amplitude; balance stance/swing velocity with30Hz lags6,12 and reliability masks. Select coefficients on TRAIN diagnostics and fixed VAL, never TEST.',
            causal_checks='Separate reference prior, learned raw output, observation IK and physical projection on declared VAL locomotion windows before attributing all loss to optimization budget.',
            runtime='Benchmark existing exact GPU query candidate on actual mixed batches before a provenance-recorded runtime migration; old weights alone do not reduce per-step geometry cost.',
            exo='Move to exo adaptation with ego retention after articulation review; no automatic handoff.'))
    (ROOT / 'SUPERVISION_AND_55K_TRANSFER_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('old_checkpoint_step', 'current_checkpoint_step', 'correction')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
