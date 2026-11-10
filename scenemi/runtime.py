"""Run new commands through the unchanged, sealed shared-resource policy.

The original launcher is an active training dependency. Extend only its module
registry in this launcher process; admission, cgroups, CPU affinity, CUDA caps
and rejection of unknown GPU jobs remain the original implementation.
"""
from scenemi import parallel_resources

EXTRA_MODULES = {'scenemi.benchmark', 'scenemi.local_training'}


def main():
    parallel_resources.MODULES = parallel_resources.MODULES | EXTRA_MODULES
    parallel_resources.main()


if __name__ == '__main__':
    main()
