"""AMASS CLI compatibility with user-managed GPU allocation and bounded CPU/RAM."""
import argparse
import json
import math
import os
import subprocess
import sys

from scenemi.config import DEFAULT_CONFIG, load_config
from experiments.offline_camera_retrain_v1.ops import resources as legacy

GIB = 2**30


def validate_runtime(runtime):
    if (not isinstance(runtime, dict) or set(runtime) != {
            'gpu_mode', 'pytorch_memory_limit_gib', 'cuda_runtime_allowance_gib',
            'free_reserve_gib'} or runtime['gpu_mode'] != 'shared_user_authorized'):
        raise ValueError('Expected the explicitly authorized shared GPU budget')
    for key in ('pytorch_memory_limit_gib', 'cuda_runtime_allowance_gib', 'free_reserve_gib'):
        value = runtime[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid runtime budget: {key}')
    return runtime


def admission(device, memory_gib, *, config):
    from scenemi.gpu_policy import admission as current
    return current(device,memory_gib)


def configure_cuda(config):
    from scenemi.gpu_policy import configure_cuda as current
    return current()


def main(argv=None):
    from scenemi.gpu_policy import resource_main
    return resource_main(argv)


if __name__ == '__main__':
    main()
