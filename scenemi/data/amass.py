"""Inventory raw AMASS against the pinned EgoAllo corpus split, without CUDA."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from scenemi.config import ROOT

SPLIT_FILE = ROOT / 'configs/amass_splits.json'


def split_definition():
    definition = json.loads(SPLIT_FILE.read_text())
    corpora = [name for names in definition['splits'].values() for name in names]
    if len(corpora) != len(set(corpora)):
        raise ValueError('An AMASS corpus occurs in more than one split')
    return definition


def verify_upstream(repository):
    repository = Path(repository)
    definition = split_definition()
    commit = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != definition['commit']:
        raise ValueError('EgoAllo checkout differs from the recorded protocol commit')
    paths = ['src/egoallo/data/amass.py', 'src/egoallo/fncsmpl_extensions.py',
             '0a_preprocess_training_data.py', '0b_preprocess_training_data.py',
             '1_train_motion_prior.py']
    hashes = {}
    for name in paths:
        content = (repository / name).read_bytes()
        original = subprocess.check_output(['git', '-C', str(repository), 'show', f'{commit}:{name}'])
        if content != original:
            raise ValueError(f'Upstream protocol source has local edits: {name}')
        hashes[name] = hashlib.sha256(content).hexdigest()
    tree = ast.parse((repository / paths[0]).read_text())
    upstream = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'AMASS_SPLITS' for t in node.targets))
    for split, corpora in definition['splits'].items():
        if set(upstream[split]) != set(corpora):
            raise ValueError(f'Corpus membership differs from upstream: {split}')
    return dict(repository=str(repository), commit=commit, source_sha256=hashes)


def exclusion_reason(relative_path):
    """Same filename rules as upstream 0a.check_skip; no motion is decoded here."""
    name = str(relative_path)
    if 'BioMotionLab_NTroje' in name and ('treadmill' in name or 'normal_' in name):
        return 'egoallo_treadmill_or_normal'
    if 'MPI_HDM05' in name and 'dg/HDM_dg_07-01' in name:
        return 'egoallo_hdm05_dg_07_01'
    return None


def inventory(paths, output):
    """Write file identities; temporal/window eligibility requires preprocessing.

    Reflected and original motions retain the same source family. Never split
    adjacent windows or reflected variants across train and validation.
    """
    provenance = verify_upstream(paths['egoallo_repo'])
    definition = split_definition()
    output = Path(output)
    if output.exists() or output.with_suffix('.summary.json').exists():
        raise ValueError('Use a new inventory path; refuse to overwrite an audit')
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, excluded, byte_counts = Counter(), Counter(), Counter()
    temporary = output.with_suffix(output.suffix + '.tmp')
    try:
        with temporary.open('w') as stream:
            for split, corpora in definition['splits'].items():
                for corpus in corpora:
                    directory = paths['amass_raw'] / corpus
                    if not directory.is_dir():
                        raise ValueError(f'Missing AMASS corpus: {corpus}')
                    for path in sorted(directory.rglob('*_poses.npz')):
                        relative = path.relative_to(paths['amass_raw']).as_posix()
                        reason = exclusion_reason(relative)
                        stat = path.stat()
                        row = dict(source=relative, family=relative.removesuffix('_poses.npz'),
                                   split=split, corpus=corpus, bytes=stat.st_size,
                                   mtime_ns=stat.st_mtime_ns, exclusion=reason,
                                   window_eligibility='not_decoded')
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        counts[split] += 1
                        byte_counts[split] += stat.st_size
                        if reason:
                            excluded[split] += 1
                    print(f'{split}: inventoried {corpus}', flush=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    summary = dict(state='raw_inventory_only', created_at=datetime.now(timezone.utc).isoformat(),
                   **provenance, amass_raw=str(paths['amass_raw']), files=dict(counts),
                   upstream_filename_exclusions=dict(excluded), source_bytes=dict(byte_counts),
                   inventory_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                   optimizer_steps=0, decoded_sequences=0,
                   note='Not a prepared training set. Crop, floor, resampling, neutral conversion, camera export and window validation remain required.')
    output.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def preprocessing_plan(paths, python):
    """Return commands for an isolated EgoAllo environment, without executing."""
    provenance = verify_upstream(paths['egoallo_repo'])
    return dict(**provenance, cwd=str(paths['egoallo_repo']),
                command_0a=[str(python), '0a_preprocess_training_data.py',
                            '--data-root', str(paths['amass_raw']),
                            '--smplh-root', str(paths['smplh_models']),
                            '--out-root', str(paths['prepared_amass']), '--devices', '0'],
                execution='Use the existing resources run wrapper. For CPU, set CUDA_VISIBLE_DEVICES="" for the new child only.',
                environment='Upstream requires Python >=3.12; do not install it into the existing SceneMI Python 3.9 environment.',
                next='0b needs admitted CUDA; SceneMI 201D export is a separate adapter, not the EgoAllo HDF5 format.')
