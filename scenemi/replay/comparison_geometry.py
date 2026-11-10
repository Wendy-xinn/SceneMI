"""Bounded hybrid evaluation: CPU observed geometry, packed CUDA networks.

Keep frozen model files untouched. The scoped override runs the existing
observed-geometry function on CPU and caches only input observations/KD trees.
Predicted anchors and rotations are refreshed at every denoising step.
"""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys

import torch

from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json
from scenemi.geometry.scene_features import ObservedNeighborhoods, relation_features
from scenemi import scene_adapter, scene_joint_replay
from . import comparison_acceleration as acceleration


GEOMETRY_INPUTS = ('local_history_points', 'local_history_valid',
                   'local_frame_points', 'local_frame_owner', 'camera',
                   'camera_available', 'camera_confidence')


@contextmanager
def cpu_observed_geometry():
    previous = (scene_adapter.relation_features, scene_joint_replay.relation_features)
    payload_cache = [None, None, None]

    @torch.no_grad()
    def compute(anchors, rotations, batch, policy, neighborhoods=None):
        if payload_cache[0] is not batch:
            cpu = {key:batch[key].detach().cpu() for key in GEOMETRY_INPUTS}
            cache = neighborhoods
            if cache is None:
                cache = [ObservedNeighborhoods(cpu['local_history_points'][i],
                    cpu['local_history_valid'][i], cpu['local_frame_points'][i],
                    cpu['local_frame_owner'][i]) for i in range(len(anchors))]
            payload_cache[:] = [batch, cpu, cache]
        result = relation_features(anchors.detach().cpu(), rotations.detach().cpu(),
                                   payload_cache[1], policy, payload_cache[2])
        return tuple(value.to(anchors.device) for value in result)

    scene_adapter.relation_features = compute
    scene_joint_replay.relation_features = compute
    try:
        yield
    finally:
        scene_adapter.relation_features, scene_joint_replay.relation_features = previous


def main():
    if len(sys.argv)>1 and sys.argv[1]=='launch':
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('action')
        parser.add_argument('--config',type=Path,required=True)
        parser.add_argument('--device',choices=('cuda',),default='cuda')
        parser.add_argument('--sizes',default='8')
        parser.add_argument('--launch-action',choices=('benchmark','run'),default='benchmark')
        args=parser.parse_args()
        c=acceleration.ref.read_config(args.config);out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
        record=acceleration.admission(c,args.config)
        record['gpu']['policy']='Explicit user-authorized sharing; 4GiB allocator + 1GiB CUDA allowance + 4GiB reserve'
        atomic_json(out/'ADMISSION_launch.json',record);print(record,flush=True)
        if not record['admitted']:raise RuntimeError('Live sharing admission refused')
        command=[sys.executable,'-u','-m','scenemi.replay.comparison_geometry',args.launch_action,
                 '--config',str(args.config),'--device','cuda','--sizes',args.sizes]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            env[key]=str(c['runtime']['cpu_count'])
        raise SystemExit(subprocess.call(acceleration.resources.scope_command(command,
            c['runtime']['memory_gib'],c['runtime']['cpu_count']),env=env))
    with cpu_observed_geometry():
        acceleration.main()


if __name__=='__main__':main()
