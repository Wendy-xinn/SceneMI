"""Resource admission for the explicitly requested concurrent joint AMASS jobs.

Unknown GPU jobs remain refused. Does not inherit the previous EXO stage's
exception for unknown/cross-distro processes.
"""
from scenemi import parallel_resources


def main():
    parallel_resources.MODULES |= {'scenemi.joint_training'}
    parallel_resources.main()


if __name__ == '__main__':
    main()
