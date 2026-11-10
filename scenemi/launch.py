"""Public training launch command with explicit experiment configuration."""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path

import yaml

from scenemi.config import ROOT

RUN_DISPATCH = """    if raw and raw[0] == 'run':
        from .launch import main as launch_main
        return launch_main(raw[1:])
"""
RUN_PARSER = "    commands.add_parser('run', help='Run training from an explicit YAML configuration')\n"
CLI = 'scenemi/__main__.py'


def verify_cli_extension(spec, *, source=None):
    if (set(spec) != {'kind', 'source', 'parent_sha256', 'current_sha256'}
            or spec['kind'] != 'unified_run_cli_dispatch_only' or spec['source'] != CLI):
        raise ValueError('Expected the declared unified run CLI extension')
    source = (ROOT / CLI).read_bytes() if source is None else source
    if hashlib.sha256(source).hexdigest() != spec['current_sha256']:
        raise ValueError('The audited current CLI changed')
    text = source.decode('utf-8')
    if text.count(RUN_DISPATCH) != 1 or text.count(RUN_PARSER) != 1:
        raise ValueError('Expected exactly the audited run dispatch and parser')
    original = text.replace(RUN_DISPATCH, '').replace(RUN_PARSER, '').encode('utf-8')
    if hashlib.sha256(original).hexdigest() != spec['parent_sha256']:
        raise ValueError('Changes beyond the run CLI extension are forbidden')
    return dict(spec, reconstructed_parent_verified=True, training_implementation_changed=False)


@contextmanager
def parent_cli_contract(c, seal, branch):
    """Authenticate the added CLI dispatch while preserving all trainer checks."""
    spec = c['continuation'].get('source_migration')
    if spec is None:
        yield seal, None
        return
    migration = verify_cli_extension(spec)
    comparison = copy.deepcopy(seal)
    if comparison['sources'].get(CLI) != migration['current_sha256']:
        raise ValueError('Child contract must seal the audited current CLI')
    comparison['sources'][CLI] = migration['parent_sha256']
    original_verify = branch.verify_training_seal

    def verify(old, config, paths, index):
        payload = {k: v for k, v in old.items() if k != 'sha256'}
        if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != old['sha256']:
            raise ValueError('Parent training contract changed')
        if old['sources'].get(CLI) != migration['parent_sha256']:
            raise ValueError('Parent does not seal the reconstructed CLI')
        view = copy.deepcopy(old)
        view['sources'][CLI] = migration['current_sha256']
        payload = {k: v for k, v in view.items() if k != 'sha256'}
        view['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return original_verify(view, config, paths, index)

    branch.verify_training_seal = verify
    try:
        yield comparison, migration
    finally:
        branch.verify_training_seal = original_verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Experiment YAML; no implicit training default')
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/amass_paths.local.yaml',
                        help='Machine-local AMASS paths')
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument('--check', action='store_true', help='Show the resolved settings without training')
    checks.add_argument('--preflight-only', action='store_true',
                        help='Run bounded CPU/CUDA acceptance checks without starting formal training')
    args = parser.parse_args(argv)
    spec = yaml.safe_load(args.config.read_text())
    if spec.get('version') != 'amass-optimizer-branch-v1' or 'parent_selection' not in spec:
        parser.error('This launcher currently supports AMASS continuation configurations with parent_selection')
    from scenemi.amass_extension import main as training_main
    command = ['config' if args.check else 'run', '--config', str(args.config),
               '--paths', str(args.paths)]
    if args.preflight_only:
        command.append('--preflight-only')
    return training_main(command)


if __name__ == '__main__':
    main()
