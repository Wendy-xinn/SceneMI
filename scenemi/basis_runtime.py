"""Use existing authorized memory admission for the temporal-basis backend."""
from scenemi import parallel_resources
from scenemi.exo_runtime import main as guarded_main


def main():
    parallel_resources.MODULES|={'scenemi.exo_basis_training'}
    guarded_main()


if __name__=='__main__':
    main()
