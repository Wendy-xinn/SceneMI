"""Real TRAIN cache/nearest/compact-IK checks without optimizer updates."""
import json
import time
from pathlib import Path
import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config, write_json, window_options
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_support_cache import TrainingSupportCache
from experiments.offline_camera_retrain_v1.native30_scene_support import training_support_targets
from experiments.offline_camera_retrain_v1.native30_observation_ik import position_proximal
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations

def main():
    torch.set_num_threads(4)
    config = read_config();config['runtime']=dict(canonical_surface_cache=True)
    data = Native30Data(HERE/config['train_manifest'], 'train', frames=192,
        memory_protocol=config['scene']['memory_protocol'], quality_config=config['gt_reliability'],
        exo_max_empty_frames=config['exo_max_empty_frames'], **window_options(config))
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(),
        json.loads((HERE/config['mount_defaults']).read_text()), config['observation']['input_noise'])
    cache = TrainingSupportCache(data, builder)
    report = dict(status='running', optimizer_steps=0, rows=[])
    output = HERE/'runs/native30_runtime_oct14/CACHE_ACCELERATION_AUDIT.json'
    for group in ('camera_wearer','trumans','rich'):
        for mode in ('ego','exo'):
            mi = next(i for i, _ in enumerate(data.members[group]) if len(data.starts_for(group,i,mode)))
            start = int(data.starts_for(group,mi,mode)[len(data.starts_for(group,mi,mode))//2])
            for yaw in (0., 1.2):
                sample, identity = data.sample(group, mode, member=mi, start=start, yaw=yaw)
                sample, scene, asset = builder.build(data, sample, identity, mode, (15,) if mode=='ego' else None)
                began=time.monotonic(); actual = training_support_targets(sample, scene, asset)
                live_s=time.monotonic()-began
                began=time.monotonic(); stored=cache.get(identity); cache_s=time.monotonic()-began
                before = scene.query_nearest
                try:
                    scene.query_nearest = lambda v,r: scene.query_wide(v,r,neighbors=1)
                    began=time.monotonic(); reference=training_support_targets(sample, scene, asset)
                    reference_s=time.monotonic()-began
                finally: scene.query_nearest=before
                row=dict(group=group, mode=mode, start=start, yaw=yaw, sequence=identity['sequence_id'],
                         live_seconds=live_s, cached_seconds=cache_s, reference_seconds=reference_s,
                         nearest_bitwise_equal=all(np.array_equal(actual[k],reference[k]) for k in stored),
                         labels_equal=all(np.array_equal(actual[k],stored[k]) for k in ('target','valid','weight')),
                         surface_valid_equal=np.array_equal(actual['surface_valid'],stored['surface_valid']),
                         vector_max_error={k:float(np.abs(actual[k]-stored[k]).max()) for k in ('points','normals','velocities','vertices')})
                report['rows'].append(row);write_json(output,report);print(json.dumps(row),flush=True)
                if not row['nearest_bitwise_equal']: raise AssertionError('Nearest query changed labels or correspondences')
                if yaw==0 and not row['labels_equal']: raise AssertionError('Canonical label cache changed targets')
                if mode=='ego' and yaw==0:
                    batch=collate_observations([(sample,scene,asset,mode)])
                    torch.manual_seed(618)
                    delta=torch.randn(1,192,22,3,device='cuda')*.03
                    motion=reconcile_motion(batch['motion'],batch['rest'],rotation_axis_angle=delta).detach()
                    results=[];grads=[]
                    for compact in (False,True):
                        leaf=motion.clone().requires_grad_(True);torch.cuda.synchronize();began=time.monotonic()
                        fixed=position_proximal(leaf,batch,torch.ones(1,device='cuda'),.9,compact=compact)
                        fixed.square().mean().backward();torch.cuda.synchronize()
                        results.append(fixed.detach());grads.append(leaf.grad.detach())
                        row['compact_ik_s' if compact else 'full_ik_s']=time.monotonic()-began
                    row['ik_output_max_error']=float((results[0]-results[1]).abs().max())
                    row['ik_gradient_max_error']=float((grads[0]-grads[1]).abs().max())
                    if not torch.allclose(results[0],results[1],atol=2e-6,rtol=1e-5) or not torch.allclose(grads[0],grads[1],atol=1e-7,rtol=1e-5):
                        raise AssertionError('Compact IK changed output/gradients')
                    write_json(output,report)
    report.update(status='passed', yaw_label_changes=sum(not r['labels_equal'] for r in report['rows']),
                  static_cache_hits=data.scene_memory.cache_hits, static_cache_misses=data.scene_memory.cache_misses)
    if report['yaw_label_changes']: raise AssertionError('Canonical cached targets disagree after yaw')
    write_json(output,report)

if __name__=='__main__': main()
