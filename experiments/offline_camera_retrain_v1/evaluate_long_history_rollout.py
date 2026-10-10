"""Frozen-model 8s streaming ablation. GT body initializes once, then scores only.

288-frame native observations are loaded for lookahead; network calls remain
64/128/192 frames. Evaluation-only window indices never alter source manifests.
"""
import hashlib
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from experiments.offline_camera_retrain_v1.audit_state_plan_comparison import load_models, inputs_only
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.bounded_head_observation import prepare_bounded_observation
from experiments.offline_camera_retrain_v1.body_history_condition import attach_executed_history
from experiments.offline_camera_retrain_v1.executed_prefix_sampling import sample_with_executed_prefix
from experiments.offline_camera_retrain_v1.state_relative_spline_refinement import refine_from_state
from experiments.offline_camera_retrain_v1.evaluate_body_history import crop, measure
from experiments.offline_camera_retrain_v1.evaluate_state_spline_confirmation import leg_motion
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

HERE = Path(__file__).parent
OUT = HERE / 'runs/long_history_rollout_oct10'
TIME_KEYS = ('trajectory', 'observation_meta', 'bps', 'bps_valid', 'camera')
LABELS = ('single192_raw', 'single192_spline', 'roll64_raw', 'roll128_raw', 'roll64_spline', 'roll128_spline')


def long_samples(data):
    """Choose by exposed recording identity and exact coverage, before scoring."""
    previous = json.loads((HERE/'runs/observed_surface_refine_oct10/panel_rows.json').read_text())
    samples, identities, exclusions = [], [], []
    import experiments.offline_sequence_v1.data_loader as legacy
    legacy.LENGTHS = (*legacy.LENGTHS, 288)  # process-local extraction only
    for group in ('trumans', 'camera_wearer', 'interactee', 'rich'):
        pool = data.rich_members if group == 'rich' else data.base.groups[group]
        preferred = [r['identity']['sequence_id'] for r in previous if r['group'] == group]
        ordered = sorted(pool, key=lambda mv: (mv[0]['sequence_id'] not in preferred,
                                              preferred.index(mv[0]['sequence_id']) if mv[0]['sequence_id'] in preferred else 0,
                                              mv[0]['sequence_id']))
        chosen = 0
        for meta, valid in ordered:
            for _, v in pool:
                v[288] = []
            name = meta['sequence_id']
            folder = Path(data.temporal_scenes.get(name, data.trumans_scene_bundles.get(name, '')))
            if not (folder/'source_frame_ids.npy').exists():
                exclusions.append({'group': group, 'sequence_id': name, 'reason': 'no exact temporal bundle'})
                continue
            ids = np.load(folder/'source_frame_ids.npy')
            good = np.load(folder/'observation_valid.npy') if (folder/'observation_valid.npy').exists() else np.ones(len(ids), bool)
            starts = valid[192]
            for start in starts:
                source = float(np.load(data.rich_root/name/'source_frame_ids.npy')[start]) if group == 'rich' else start[1]
                first = int(np.searchsorted(ids, source))
                if first+288 > len(ids) or not np.allclose(ids[first:first+288], source+np.arange(288)*1.5) or not good[first:first+288].all():
                    continue
                valid[288] = [start]
                try:
                    sample, identity = data.sample(288, group, sequence_index=0, start_index=0)
                except (ValueError, IndexError) as exc:
                    exclusions.append({'group': group, 'sequence_id': name, 'reason': str(exc)})
                    valid[288] = []
                    continue
                assert identity['sequence_id'] == name and len(sample['motion']) == 288
                samples.append(sample); identities.append(identity); chosen += 1
                break
            if chosen == 2:
                break
        if chosen != 2:
            raise ValueError(f'{group}: only {chosen} long covered recordings')
    return samples, identities, exclusions


def summary(rows):
    keys = ['mpjpe_cm', 'pa_mpjpe_mm', 'pelvis_orientation_mean_deg', 'head_cm',
            'gt_stance_slide_cm_frame', 'boundary_root_step_cm', 'boundary_velocity_change_cm_frame']
    result = {}
    for group in ('all', 'trumans', 'camera_wearer', 'interactee', 'rich'):
        selected = rows if group == 'all' else [r for r in rows if r['group'] == group]
        result[group] = {}
        for label in LABELS:
            members = [r for r in selected if r['label'] == label]
            result[group][label] = {k:float(np.mean([r['full'][k] for r in members])) for k in keys}
    reference = result['all']['single192_raw']['gt_stance_slide_cm_frame']
    result['acceptance'] = {label: {'overall_slide_nonincrease': result['all'][label]['gt_stance_slide_cm_frame'] <= reference,
        'each_role_slide_nonincrease': all(result[g][label]['gt_stance_slide_cm_frame'] <= result[g]['single192_raw']['gt_stance_slide_cm_frame'] for g in ('trumans','camera_wearer','interactee','rich')),
        'scope': 'point-estimate screen only, two recordings/role; no generalization or physical acceptance'} for label in LABELS if label != 'single192_raw'}
    return result


def evaluate(combined, truth, executed_history, boundaries):
    metrics = measure(combined, truth, turn_threshold=30)
    metrics.update(leg_motion(combined))
    positions = forward_kinematics(torch.cat((executed_history, combined), 1), truth['rest'])
    velocity = torch.diff(positions[:, :, 0], dim=1)
    at = torch.tensor([16+b-1 for b in boundaries], device=combined.device)
    metrics['boundary_root_step_cm'] = float(velocity[:,at].norm(dim=-1).mean()*100)
    metrics['boundary_velocity_change_cm_frame'] = float((velocity[:,at]-velocity[:,at-1]).norm(dim=-1).mean()*100)
    return metrics


@torch.inference_mode()
def main():
    torch.set_num_threads(4); OUT.mkdir(exist_ok=True); began=time.monotonic(); free=shutil.disk_usage(HERE).free
    models, configs, hashes = load_models(); del models['official']; model=models['history']; c=configs['history']
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'])
    samples, identities, exclusions=long_samples(data)
    protocol={'status':'registered_before_generation','identities':identities,'coverage_exclusions':exclusions,
        'weights':hashes['history'],'labels':LABELS,'replicate_seeds':[2026101101,2026101102],
        'Hz':20,'initial_history_frames':16,'scored_execution_frames':160,'execute_chunk_frames':8,
        'scene':'same original frame and initial static occupancy, cumulative time-indexed static/current dynamic BPS; no resets or dynamic union',
        'GT':'body only initializes first16; GT-derived ideal head is allowed; future body/contact score only',
        'feedback':'exact kinematic execution, not simulator/tracker','scope':'identity/coverage-selected exposed development recordings; not holdout',
        'acceptance':'overall and each-role slide must not rise vs single192_raw; uncertainty and worst recordings reported, no automatic adoption',
        'pairing':'same initial history, observation stream and step seeds; different window lengths have different noise shapes, not identical tensors',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'new_checkpoints':0}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    batch={k:v.cuda() for k,v in collate(samples).items()}; allowed=inputs_only(prepare_bounded_observation(batch,'joint')); rows=[]
    def observations(start, length):
        return {k:v[:,start:start+length] if k in TIME_KEYS else v for k,v in allowed.items()}
    for rep, seed in enumerate(protocol['replicate_seeds']):
        outputs={}
        initial=batch['motion'][:,:16].clone()
        for label in LABELS:
            rolling=label.startswith('roll'); length=int(label.split('_')[0][4:]) if rolling else 192
            history=initial.clone(); chunks=[]; count=20 if rolling else 1
            for step in range(count):
                obs=observations(step*8 if rolling else 0,length)
                raw=sample_with_executed_prefix(model,attach_executed_history(obs,history),history,seed=seed+step*1009,steps=20)
                if label.endswith('spline'):
                    raw,_=refine_from_state(raw,history,obs,pose_frames=32,iterations=60,relative_root=True)
                future=raw[:,16:24] if rolling else raw[:,16:176]
                chunks.append(future.detach().clone())
                history=torch.cat((history[:,8:],future),1) if rolling else history
                if rolling:
                    assert history.shape[1]==16
                if rolling and (step+1)%5==0:
                    print(f'rep={rep} {label} {step+1}/20',flush=True)
            combined=torch.cat(chunks,1); outputs[label]=torch.cat((initial,combined),1).cpu().numpy()
            for i,identity in enumerate(identities):
                one={k:v[i:i+1] for k,v in batch.items()}; future=combined[i:i+1]
                metrics=evaluate(future,crop(one,16,176),initial[i:i+1],list(range(0,160,8)))
                tails=measure(future[:,120:],crop(one,136,176),turn_threshold=15)
                segments=[measure(future[:,j:j+32],crop(one,16+j,48+j),turn_threshold=15) for j in range(0,160,32)]
                rows.append({'replicate':rep,'seed':seed,'label':label,'group':identity['group'],'identity':identity,'full':metrics,'last2s':tails,'segments1p6s':segments})
            (OUT/'rows.json').write_text(json.dumps(rows,indent=2)); (OUT/'summary.json').write_text(json.dumps(summary(rows),indent=2))
            print(label,'slide',np.mean([r['full']['gt_stance_slide_cm_frame'] for r in rows if r['replicate']==rep and r['label']==label]),flush=True)
        # Native motions only; no repeated vertex/scene arrays or checkpoints.
        np.savez_compressed(OUT/f'motions_rep{rep}.npz',gt=batch['motion'][:,:176].cpu().numpy(),**outputs)
    protocol.update(status='completed',elapsed_s=time.monotonic()-began,free_disk_before=free,free_disk_after=shutil.disk_usage(HERE).free)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps(summary(rows)['all']),flush=True)

if __name__=='__main__': main()
