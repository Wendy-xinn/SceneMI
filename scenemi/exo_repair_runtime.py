"""Reuse authorized EXO-stage memory sharing for its independent controls."""
from scenemi import parallel_resources
from scenemi.exo_runtime import main as guarded_main


def main():
    parallel_resources.MODULES |= {'scenemi.exo_diagnostics'}
    guarded_main()


if __name__ == '__main__':
    main()
