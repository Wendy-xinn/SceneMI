"""Admit the authorized EXO initialization pilot without editing old launchers."""
from scenemi import parallel_resources
from scenemi.exo_runtime import main as guarded_main


def main():
    parallel_resources.MODULES |= {'scenemi.exo_repair_training'}
    guarded_main()


if __name__ == '__main__':
    main()
