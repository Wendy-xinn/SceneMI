"""Resource admission for the authorized frozen-prior adapter experiments.

Only register the new allocator-aware entries; keep all existing admission,
cgroup, CPU and unknown-process checks intact.
"""
from scenemi import parallel_resources


def main():
    parallel_resources.MODULES |= {'scenemi.scene_conditioning', 'scenemi.data.scene_inventory'}
    parallel_resources.main()


if __name__ == '__main__':
    main()
