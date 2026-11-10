"""Compatibility imports for historical four-dataset EGO commands.

Current implementations live in train, evaluate, data.ego, and checkpoint.
Existing checkpoints store state dictionaries and remain strictly loadable.
"""
from scenemi.train import (
    CONFIG, load_config, build_model, contract, optimizer_for, make_batch,
    update, configure_runtime, restore_selection, trim_update_log, preflight, train,
)
from scenemi.data.ego import (
    SOURCE_ORDER, ALL_KEYS, FourEgoData, normalize_sample, collate, payload, source_cases,
)
from scenemi.evaluate import aggregate, evaluate
from scenemi.checkpoint import checkpoint, load_checkpoint, load_trained_model


def main(argv=None):
    import sys
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == 'evaluate':
        from scenemi.evaluate import main as evaluate_main
        return evaluate_main(raw[1:])
    from scenemi.train import main as train_main
    return train_main(raw)


if __name__ == '__main__':
    main()
