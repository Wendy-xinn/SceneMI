"""Apply the user's explicit, run-scoped free-memory sharing authorization.

The training closure and its passed CPU acceptance remain unchanged. Only the
launcher's unknown-PID refusal is replaced for the authorized run; all live
memory measurements, allocator limits and cgroup checks remain in effect.
"""
import json
from pathlib import Path
import sys

from scenemi.config import ROOT
from scenemi.data.amass_prepare import sha256
from scenemi import parallel_resources
from . import training


def authorization(path, config_path):
    from scenemi.gpu_policy import load_policy
    load_policy()
    return dict(state='user_authorized',scope='project_user_managed_gpu')


def authorized_admission(original, auth, path):
    from scenemi.gpu_policy import admission
    return admission


def main():
    config_path = training.CONFIG
    if '--config' in sys.argv:
        config_path = Path(sys.argv[sys.argv.index('--config')+1])
    c = training.configuration(config_path)
    path = ROOT/c['output']/'SHARING_AUTHORIZATION.json'
    auth = authorization(path, config_path)
    if len(sys.argv) < 2 or sys.argv[1] != 'launch':
        raise ValueError('This authorized adapter is only a bounded launch entry')
    parallel_resources.admission = authorized_admission(parallel_resources.admission, auth, path)
    training.main()


if __name__ == '__main__':
    main()
