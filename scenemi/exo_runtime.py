"""Admit the EXO stage with the explicitly authorized current GPU sharing."""
import json
from scenemi import parallel_resources
from scenemi.config import ROOT


def main():
    from scenemi.gpu_policy import load_policy
    load_policy()
    parallel_resources.MODULES |= {'scenemi.exo_training'}
    parallel_resources.main()


if __name__ == '__main__':
    main()
