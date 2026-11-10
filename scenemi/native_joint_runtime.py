"""Bounded admission for native EGO/EXO motion-prior adaptation."""
from scenemi import parallel_resources


def main():
    # Recognize the already authorized scratch trainer; all original checks stay.
    parallel_resources.MODULES |= {'scenemi.native_joint_training', 'scenemi.joint_training',
                                  'scenemi.exo_continuation'}
    parallel_resources.main()


if __name__ == '__main__':
    main()
