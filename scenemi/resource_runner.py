"""Compatibility entry for user-managed GPU execution and CPU/RAM isolation."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from experiments.offline_camera_retrain_v1.ops import resources as legacy


def own_public_job(pid):
    """Recognize public EGO entries without changing historical launchers."""
    from scenemi.config import ROOT
    try:
        proc = Path('/proc') / str(pid)
        words = [part.decode() for part in proc.joinpath('cmdline').read_bytes().split(b'\0') if part]
        module_index = words.index('-m') + 1
        return (proc.stat().st_uid == os.getuid() and proc.joinpath('cwd').resolve() == ROOT
                and words[module_index] == 'scenemi'
                and words[module_index + 1] in ('train', 'preflight', 'evaluate'))
    except (OSError, ValueError, IndexError, UnicodeError):
        return False


def shared_admission(device, memory_gib, allocator_gib):
    from scenemi.gpu_policy import admission
    return admission(device,memory_gib)


def main(argv=None):
    from scenemi.gpu_policy import resource_main
    return resource_main(argv)


if __name__ == "__main__":
    main()
