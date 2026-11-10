"""Compose the sealed repair trainer with the explicit temporal-basis backend.

The original trainer is immutable during its concurrent final evaluation.
This entry changes only its model factory, before contract creation; the new
backend and entry sources are included in that actual dependency closure.
"""
from scenemi import exo_repair_training as pilot
from scenemi import basis_runtime as _runtime_dependency
from scenemi.exo_basis import TemporalBasisExoPrior


def main():
    pilot.InitializedExoPrior=TemporalBasisExoPrior
    pilot.CONFIG=pilot.ROOT/'configs/amass_exo_basis.yaml'
    pilot.main()


if __name__=='__main__':
    main()
