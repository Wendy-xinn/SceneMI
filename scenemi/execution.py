"""Local resource admission, bounded execution and configured process budgets."""
import json
import os
from pathlib import Path
import sys


def ensure_bounded():
    from scenemi.training import ensure_bounded as check
    return check()


def resource_main(argv):
    """Connect local bounded jobs to this user's existing systemd manager.

    IDE/WSL terminals may inherit a missing or stale runtime directory. Scope
    creation uses the existing private manager socket, without starting a
    session bus or changing any service or cgroup policy.
    """
    from scenemi.resource_runner import main as resources_main
    runtime = Path('/run/user') / str(os.getuid())
    private = runtime / 'systemd/private'
    previous = os.environ.get('XDG_RUNTIME_DIR')
    local_manager = private.is_socket() and private.stat().st_uid == os.getuid()
    if local_manager:
        os.environ['XDG_RUNTIME_DIR'] = str(runtime)
        if previous != str(runtime):
            print(f'Using existing user systemd manager: {private}', file=sys.stderr)
    try:
        return resources_main(argv)
    finally:
        if previous is None:
            os.environ.pop('XDG_RUNTIME_DIR', None)
        else:
            os.environ['XDG_RUNTIME_DIR'] = previous


def configure_runtime(c, device):
    """Require bounded CPU/RAM execution; GPU allocation is user-managed."""
    import torch
    ensure_bounded()
    relative = Path('/proc/self/cgroup').read_text().strip().split(':', 2)[2]
    maximum = (Path('/sys/fs/cgroup') / relative.lstrip('/') / 'memory.max').read_text().strip()
    if int(maximum) > c['runtime']['memory_gib'] * 2**30 or len(os.sched_getaffinity(0)) > c['runtime']['cpu_count']:
        raise RuntimeError('Resource wrapper exceeds the RAM/CPU budget in the configuration')
    if device == 'cuda':
        from scenemi.gpu_policy import configure_cuda
        return configure_cuda()


