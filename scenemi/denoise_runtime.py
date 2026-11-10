"""Reuse the user's memory-based EXO sharing policy for overnight repair."""
from scenemi import parallel_resources
from scenemi.exo_runtime import main as guarded_main


def main():
    parallel_resources.MODULES |= {'scenemi.exo_denoise_training'}
    guarded_main()


if __name__ == '__main__':
    main()
