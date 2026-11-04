"""Explicit, audited execution adapter for upstream AMASS height clustering.

Original sources and output fields remain unchanged. An adoption record seals
the old manifest prefix and the equivalence evidence before resuming it.
"""
import json
from pathlib import Path

from scenemi.config import ROOT
from . import amass_prepare as base
from .amass_floor import HeightDBSCAN


def verify(paths):
    """Compare all fields on 19 real corpora, then audit the stalled long input."""
    import hashlib
    import tempfile
    import time
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import KDTree

    module = base.load_preprocessor(paths['egoallo_repo'])
    rows = [json.loads(line) for line in (paths['prepared_amass']/'manifest.jsonl').read_text().splitlines()]
    selected = {}
    for row in rows:
        if row['state'] == 'prepared' and (row['corpus'] not in selected
                or row['raw_frames'] < selected[row['corpus']]['raw_frames']):
            selected[row['corpus']] = row
    if len(selected) != 19:
        raise ValueError('Expected committed examples covering all 19 original corpora')
    output = paths['output']/'preparation/memory_equivalence'
    output.mkdir(parents=True, exist_ok=True)
    record = dict(state='running', implementation_sha256=base.sha256(ROOT/'scenemi/data/amass_floor.py'),
        verifier_sha256=base.sha256(__file__),
        original_upstream_sha256=base.sha256(paths['egoallo_repo']/'0a_preprocess_training_data.py'),
        cases=[], long_input_checks=[], whole_field_bitwise_equal=True,
        previous_committed_outputs_modified=False)
    base.atomic_json(output/'RESULT.json', record)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(dir=output) as temporary, base.working_directory(paths['egoallo_repo']):
        temporary = Path(temporary)
        for row in selected.values():
            for reflect in (False, True):
                a, b = temporary/'reference.npz', temporary/'bounded.npz'
                for target, implementation in ((a, DBSCAN), (b, HeightDBSCAN)):
                    target.unlink(missing_ok=True)
                    module.DBSCAN = implementation
                    module.process_seq(str(paths['amass_raw']/row['source']), str(target),
                        str(paths['smplh_models']), 0, beta_neutral=True, reflect=reflect, split_frame_limit=256)
                prior = paths['prepared_amass']/next(v['path'] for v in row['variants'] if v['reflected'] == reflect)
                with np.load(a) as x, np.load(b) as y, np.load(prior) as z:
                    if set(x.files) != set(y.files) or set(x.files) != set(z.files):
                        raise ValueError('Different processed field sets')
                    for key in x.files:
                        if not np.array_equal(x[key], y[key]) or not np.array_equal(x[key], z[key]):
                            raise ValueError(f'Processed field differs: {row["source"]}, {reflect}, {key}')
                    record['cases'].append(dict(source=row['source'], reflected=reflect,
                        fields_checked=sorted(x.files), frames=int(x['num_frames']), bitwise_equal=True))
                base.atomic_json(output/'RESULT.json', record)
                print(json.dumps(dict(corpus=row['corpus'], reflected=reflect, passed=len(record['cases']))), flush=True)

        class AuditedHeights(HeightDBSCAN):
            def fit(self, x):
                super().fit(x)
                values = np.asarray(x, dtype=np.float64)
                counts = KDTree(values).query_radius(values, r=self.eps, count_only=True)
                if not np.array_equal(np.flatnonzero(counts >= self.min_samples), self.core_sample_indices_):
                    raise ValueError('Height core classification differs from independent KDTree queries')
                rng = np.random.default_rng(20261022)
                for _ in range(10):
                    sample = x[rng.choice(len(x), min(len(x), 1024), replace=False)]
                    reference = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(sample)
                    actual = HeightDBSCAN(self.eps, self.min_samples).fit(sample)
                    if not np.array_equal(reference.labels_, actual.labels_):
                        raise ValueError('Long-input subset labels differ from original DBSCAN')
                record['long_input_checks'].append(dict(points=len(x), core_counts_match_kdtree=True,
                    sklearn_subsets_equal=10, height_bytes_sha256=hashlib.sha256(np.asarray(x).tobytes()).hexdigest()))
                return self

        module.DBSCAN = AuditedHeights
        raw = paths['amass_raw']/'KIT/969/Trial_02_poses.npz'
        for reflect in (False, True):
            target = temporary/'long.npz'
            target.unlink(missing_ok=True)
            module.process_seq(str(raw), str(target), str(paths['smplh_models']), 0,
                beta_neutral=True, reflect=reflect, split_frame_limit=256)
            with np.load(target) as data:
                if int(data['num_frames']) <= 128 or any(not np.isfinite(data[k]).all()
                        for k in data.files if data[k].dtype.kind in 'fci'):
                    raise ValueError('Long motion did not produce finite 128-frame targets')
    record.update(state='passed', elapsed_seconds=time.monotonic()-started, cases_checked=len(record['cases']))
    base.atomic_json(output/'RESULT.json', record)
    return record


def prepare(paths, inventory_path):
    import fcntl

    evidence = paths['output']/'preparation/memory_equivalence/RESULT.json'
    proof = json.loads(evidence.read_text())
    if (proof['state'] != 'passed' or proof['cases_checked'] != 38
            or not proof['whole_field_bitwise_equal'] or len(proof['long_input_checks']) != 2
            or any(not row['core_counts_match_kdtree'] or row['sklearn_subsets_equal'] != 10
                   for row in proof['long_input_checks'])
            or proof['implementation_sha256'] != base.sha256(ROOT/'scenemi/data/amass_floor.py')
            or proof.get('verifier_sha256') != base.sha256(__file__)
            or proof['original_upstream_sha256'] != base.sha256(paths['egoallo_repo']/'0a_preprocess_training_data.py')):
        raise ValueError('Missing or stale real-data clustering equivalence evidence')

    original_load, original_seal = base.load_preprocessor, base.seal_dependencies

    def load(repository):
        module = original_load(repository)
        module.DBSCAN = HeightDBSCAN
        return module

    def seal(paths, inventory_path, module):
        job = original_seal(paths, inventory_path, module)
        job['execution_adapter'] = dict(name='linear_memory_1d_height_dbscan_v1',
            equivalence_sha256=base.sha256(evidence),
            output_protocol_unchanged=True, original_upstream_files_modified=False)
        for name in ('scenemi/data/amass_floor.py', 'scenemi/data/amass_prepare_bounded.py', 'scenemi/__main__.py'):
            job['local_source_sha256'][name] = base.sha256(ROOT/name)
        return job

    output = paths['prepared_amass']
    with (output/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        module = load(paths['egoallo_repo'])
        legacy = original_seal(paths, Path(inventory_path).resolve(), module)
        bounded = seal(paths, Path(inventory_path).resolve(), module)
        stamp = output/'JOB.json'
        current = json.loads(stamp.read_text())
        if current == legacy:
            rows = [json.loads(line) for line in Path(inventory_path).read_text().splitlines()]
            ledger = output/'manifest.jsonl'
            records = base.verified_records(ledger, output, rows)
            backup = output/'JOB.original.json'
            if backup.exists() and backup.read_bytes() != stamp.read_bytes():
                raise ValueError('Original job backup differs from the existing seal')
            if not backup.exists():
                backup.write_bytes(stamp.read_bytes())
            adoption = dict(state='verified', original_job_sha256=base.sha256(backup),
                prior_manifest_sha256=base.sha256(ledger), prior_manifest_bytes=ledger.stat().st_size,
                committed_sources_verified=len(records),
                committed_variants_verified=sum(len(r['variants']) for r in records.values()),
                equivalence_sha256=base.sha256(evidence),
                output_protocol_unchanged=True, committed_outputs_modified=False,
                raw_data_modified=False, original_upstream_files_modified=False)
            base.atomic_json(paths['output']/'preparation/BOUNDED_ADOPTION.json', adoption)
            base.atomic_json(stamp, bounded)
        elif current != bounded:
            raise ValueError('Existing preparation job differs from both audited execution seals')
    base.load_preprocessor, base.seal_dependencies = load, seal
    try:
        return base.prepare(paths, inventory_path)
    finally:
        base.load_preprocessor, base.seal_dependencies = original_load, original_seal
