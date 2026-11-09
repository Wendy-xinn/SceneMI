"""Continue sealed AMASS weights after an authenticated GIMO CLI extension.

The saved parent remains immutable. Only its verification view substitutes the
audited CLI digest; every other source, model, optimizer and data check remains
in the existing branch implementation. Child contracts seal the current files.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scenemi import amass_finetuning as branch
from scenemi.config import ROOT

CLI = 'scenemi/__main__.py'
GIMO_DISPATCH = """    if raw and raw[0] == 'gimo':
        if raw[1:2] == ['poses']:
            from .data.gimo_pose import main as gimo_pose_main
            return gimo_pose_main(raw[2:])
        from .data.gimo import main as gimo_main
        return gimo_main(raw[1:])
"""
GIMO_PARSER = "    commands.add_parser('gimo', help='GIMO source audit, EGO preprocessing and Viser review')\n"
_parent_state = branch.parent_state


def verify_cli_migration(spec, *, source=None):
    required = {'kind', 'source', 'parent_sha256', 'current_sha256'}
    if (set(spec) != required or spec['kind'] != 'gimo_cli_dispatch_only'
            or spec['source'] != CLI):
        raise ValueError('Only the declared GIMO CLI extension is supported')
    source = (ROOT / CLI).read_bytes() if source is None else source
    if hashlib.sha256(source).hexdigest() != spec['current_sha256']:
        raise ValueError('The audited current CLI changed')
    text = source.decode('utf-8')
    if text.count(GIMO_DISPATCH) != 1 or text.count(GIMO_PARSER) != 1:
        raise ValueError('Expected exactly the audited GIMO dispatch and parser')
    original = text.replace(GIMO_DISPATCH, '').replace(GIMO_PARSER, '').encode('utf-8')
    if hashlib.sha256(original).hexdigest() != spec['parent_sha256']:
        raise ValueError('Changes beyond the GIMO CLI extension are forbidden')
    return dict(spec, reconstructed_parent_verified=True,
                training_implementation_changed=False)


def parent_state(c, paths, seal):
    migration = verify_cli_migration(c['continuation']['source_migration'])
    # Original parent_state compares all predecessor sources with the child.
    # Use the authenticated predecessor CLI for this comparison only. The
    # actual child contract retains the current digest and this adapter source.
    comparison = copy.deepcopy(seal)
    if comparison['sources'].get(CLI) != migration['current_sha256']:
        raise ValueError('Child contract does not seal the audited current CLI')
    comparison['sources'][CLI] = migration['parent_sha256']
    original_verify = branch.verify_training_seal

    def verify(old, config, parent_paths, index):
        if old['sources'].get(CLI) != migration['parent_sha256']:
            raise ValueError('Parent does not seal the reconstructed CLI')
        # Verify the immutable parent contract digest before deriving a
        # verification-only view with the authenticated CLI substitution.
        payload = {k: v for k, v in old.items() if k != 'sha256'}
        if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != old['sha256']:
            raise ValueError('Parent training contract changed')
        view = copy.deepcopy(old)
        view['sources'][CLI] = migration['current_sha256']
        payload = {k: v for k, v in view.items() if k != 'sha256'}
        view['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return original_verify(view, config, parent_paths, index)

    branch.verify_training_seal = verify
    try:
        state, best, provenance = _parent_state(c, paths, comparison)
    finally:
        branch.verify_training_seal = original_verify
    provenance['source_migration'] = migration
    return state, best, provenance


def run(c, paths_file):
    verify_cli_migration(c['continuation']['source_migration'])
    runtime = c['runtime']
    print(f"AMASS continuation: {c['continuation']['parent_step']} -> "
          f"{c['training']['planned_updates']}; effective batch={c['training']['batch_size']} "
          f"(one full batch per update); frames={c['data']['frames']}; "
          f"lr={c['optimizer']['lr']:.2e}; output={c['output']}", flush=True)
    for action, device in (('preflight', 'cpu'), ('preflight', 'cuda'), ('train', 'cuda')):
        command = [sys.executable, '-m', 'scenemi', 'resources', 'run',
            '--device', device, '--memory-gib', str(runtime['memory_gib']),
            '--cpu-count', str(runtime['cpu_count']), '--',
            sys.executable, '-u', '-m', 'scenemi.amass_cli_compatibility', action,
            '--config', c['_config_path'], '--paths', str(Path(paths_file).resolve()), '--device', device]
        if action == 'train' and (ROOT / c['output'] / 'training/last.pt').exists():
            command.append('--resume')
        subprocess.run(command, cwd=ROOT, check=True)


def main(argv=None):
    branch.parent_state = parent_state
    branch.run = run
    return branch.main(argv)


if __name__ == '__main__':
    main()
