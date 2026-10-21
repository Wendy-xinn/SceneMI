"""Exact saved-input/reference reconstruction without loading a generator.

This is an independent device-specific audit. It neither replaces the original
CPU recheck nor reads saved predictions. All comparisons are exact; a mismatch
is a failed audit, not grounds to change the original numeric tolerance.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from . import final_evaluation as final
from . import final_evaluation_batch as batch
from .dependencies import dependency_closure, sha256

PROTOCOL = 'selected-native30-exact-input-recheck-v1'
REPORT = 'GPU_INPUTS_RECHECK.json'
require, read, write = final.require, final.read, final.write
IDENTITY_KEYS = ('mode', 'dataset', 'recording', 'sequence_id', 'start', 'source_first')


def admit_rows(plan, rows):
    """Require the complete declared cohort and both scene arms before auditing."""
    require(plan['protocol'] == final.PROTOCOL and plan['state'] == 'frozen', 'Frozen parent plan required')
    require(plan['split'] == 'full-validation', 'This input audit admits full VAL only')
    require(plan['scene_variants'] == list(final.SCENES), 'Require both original scene arms')
    require(len(rows) == plan['planned_predictions'], 'Sampling must be complete before input audit')
    entries = {entry['case']: entry for entry in plan['cases']}
    require(len(entries) == len(plan['cases']) == plan['unique_mode_windows'], 'Invalid parent case identities')
    groups = defaultdict(dict)
    for row in rows:
        key = row['case'], row['regime']
        require(row['case'] in entries and row['regime'] in plan['regimes'], 'Unexpected case/regime')
        require(row['scene_variant'] in final.SCENES and row['scene_variant'] not in groups[key], 'Duplicate/unexpected scene arm')
        entry = entries[row['case']]
        require(all(row[k] == entry[k] for k in IDENTITY_KEYS) and row['seed'] == entry['case_seed'], 'Saved identity/seed differs')
        require(row['checkpoint_sha256'] == plan['selection']['checkpoint_sha256']
                and row['selection_sha256'] == plan['selection_manifest_sha256'], 'Saved model/selection differs')
        groups[key][row['scene_variant']] = row
    ordered = []
    for entry in plan['cases']:
        for regime in plan['regimes']:
            pair = groups.get((entry['case'], regime), {})
            require(set(pair) == set(final.SCENES), 'Incomplete source scene pair')
            require(pair['scene_on']['input_sha256'] == pair['scene_off']['input_sha256'], 'Scene arms have different inputs')
            ordered.append((entry, regime, [pair[scene] for scene in final.SCENES]))
    require(len(groups) == len(ordered) and len(rows) == len(ordered) * 2, 'Incomplete declared coverage')
    return ordered


def array_digest(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
    digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def exact_array(actual, expected, name):
    import numpy as np
    require(actual.dtype == expected.dtype and actual.shape == expected.shape, name + ': shape/dtype differs')
    if not np.array_equal(actual, expected):
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        maximum = float(np.nanmax(difference)) if difference.size else 0.
        index = list(np.unravel_index(np.nanargmax(difference), difference.shape)) if difference.size and np.isfinite(difference).any() else None
        raise ValueError(name + ': exact mismatch; max_abs=' + str(maximum) + '; argmax=' + str(index))


def compare_saved(entry, regime, pair, sample, identity, observed, reference, input_sha256):
    """Read source fields only; the prediction member is deliberately untouched."""
    import numpy as np
    source = {'truth': sample['motion'], 'rest': sample['rest'],
              'observations': observed['trajectory'], 'observation_meta': observed['observation_meta']}
    for row in pair:
        require(sha256(row['artifact']) == row['artifact_sha256'], 'Saved artifact changed during audit')
        require(input_sha256 == row['input_sha256'], 'Complete input tensor fingerprint differs from saved row')
        with np.load(row['artifact'], allow_pickle=False) as saved:
            require(str(saved['input_sha256'].item()) == input_sha256, 'NPZ input hash differs')
            require(str(saved['checkpoint_sha256'].item()) == row['checkpoint_sha256'], 'NPZ model identity differs')
            require(int(saved['case_seed'].item()) == entry['case_seed'], 'NPZ seed differs')
            require(bool(saved['use_scene'].item()) == (row['scene_variant'] == 'scene_on'), 'NPZ scene arm differs')
            require(json.loads(str(saved['identity_json'].item())) == identity, 'NPZ source identity differs')
            for key, actual in source.items():
                exact_array(np.asarray(actual), saved[key], row['scene_variant'] + '/' + key)
            exact_array(reference, saved['reference_motion'], row['scene_variant'] + '/reference_motion')
    return dict(case=entry['case'], regime=regime, seed=entry['case_seed'],
        **{key: entry[key] for key in IDENTITY_KEYS}, status='exact_pass',
        artifacts=[dict(scene_variant=row['scene_variant'], artifact=row['artifact'],
                        artifact_sha256=row['artifact_sha256']) for row in pair],
        input_sha256=input_sha256, reference_sha256=array_digest(reference),
        source_array_sha256={key: array_digest(value) for key, value in source.items()},
        source_arrays_exact=True, input_fingerprint_exact=True, reference_arrays_exact=True,
        checked_scene_artifacts=2)


def resume_rows(path, manifest_sha256, expected, *, resume):
    """Recover only this audit's uncommitted tail; never alter sampling records."""
    path = Path(path)
    if not path.exists():
        return {}
    require(resume, 'Existing audit records require --resume')
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        cut = raw.rfind(b'\n') + 1
        write(path.parent / 'TAIL_RECOVERY.json', dict(checked_at=final.now(), discarded_bytes=len(raw)-cut,
            discarded_sha256=hashlib.sha256(raw[cut:]).hexdigest(), committed_bytes=cut))
        with path.open('r+b') as stream:
            stream.truncate(cut); stream.flush(); os.fsync(stream.fileno())
        raw = raw[:cut]
    records = {}
    for line in raw.splitlines():
        row = json.loads(line)
        key = row['case'], row['regime']
        require(key in expected and key not in records, 'Unexpected/duplicate audit record')
        entry, pair = expected[key]
        require(row['audit_plan_sha256'] == manifest_sha256 and row['status'] == 'exact_pass', 'Audit manifest/status differs')
        require(all(row[k] == entry[k] for k in IDENTITY_KEYS) and row['seed'] == entry['case_seed'], 'Audit source identity differs')
        require(row['input_sha256'] == pair[0]['input_sha256'] and row['artifacts'] == [dict(
            scene_variant=r['scene_variant'], artifact=r['artifact'], artifact_sha256=r['artifact_sha256']) for r in pair], 'Audit input/artifact identity differs')
        require(row['checked_scene_artifacts'] == 2 and all(row.get(k) is True for k in (
            'source_arrays_exact', 'input_fingerprint_exact', 'reference_arrays_exact')), 'Audit exact comparisons missing')
        records[key] = row
    return records


def append_record(path, row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def inputs(out, *, device='cuda', resume=False, memory_fraction=.5, every=10):
    import torch
    from . import validation_review as common
    out = Path(out).resolve()
    audit = out / ('input_recheck_' + device)
    audit.mkdir(parents=True, exist_ok=True)
    require(device in ('cpu', 'cuda') and every > 0, 'Invalid audit device/status frequency')
    with (audit / 'worker.lock').open('a') as worker:
        fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = final.verify(out)  # Hashes the selected checkpoint; does not load it.
        rows = final.load_rows(out, plan, repair=False)
        ordered = admit_rows(plan, rows)
        binding = dict(protocol=PROTOCOL, parent_out=str(out), device=device,
            PLAN_sha256=sha256(out/'PLAN.json'), rows_sha256=sha256(out/'rows.jsonl'),
            selection_manifest_sha256=plan['selection_manifest_sha256'],
            checkpoint_sha256=plan['selection']['checkpoint_sha256'], model_tensor_sha256=plan['model_tensor_sha256'],
            source_sha256=dependency_closure([Path(__file__)]), input_source_sha256=plan['input_source_sha256'],
            parent_source_sha256=plan['source_sha256'], planned_inputs=len(ordered),
            planned_scene_artifacts=len(rows), memory_fraction=memory_fraction,
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                         deterministic_algorithms=True, threads=2, cudnn_deterministic=True, cudnn_benchmark=False),
            predictions_generated=0, generator_loaded=False, checkpoint_weights_loaded=False,
            GT_used_for_input_or_selection=False, comparison='np.array_equal plus complete original tensor_fingerprint; no tolerance')
        manifest_path = audit/'PLAN.json'
        if manifest_path.exists():
            require(resume, 'Existing input audit requires --resume')
            previous = read(manifest_path)
            require(previous['binding'] == binding, 'Immutable input audit binding changed')
        else:
            write(manifest_path, dict(binding=binding, created_at=final.now()))
        manifest_sha256 = sha256(manifest_path)
        records_path = audit/'rows.jsonl'
        expected = {(entry['case'], regime): (entry, pair) for entry, regime, pair in ordered}
        records = resume_rows(records_path, manifest_sha256, expected, resume=resume)
        report_path = out/REPORT if device == 'cuda' else audit/'RESULT.json'

        def report(state, **extra):
            complete = state == 'completed_exact'
            value = dict(binding, checked_at=final.now(), pid=os.getpid(), status=state, complete=complete,
                completed_inputs=len(records), checked_scene_artifacts=sum(r['checked_scene_artifacts'] for r in records.values()),
                input_audit_plan_sha256=manifest_sha256,
                input_audit_records=str(records_path), input_audit_records_sha256=sha256(records_path) if records_path.exists() else None,
                all_source_arrays_exact=complete, all_input_fingerprints_exact=complete,
                all_reference_arrays_exact=complete, sources_verified_unchanged=complete, **extra)
            write(report_path, value)
            write(audit/'STATUS.json', value)
            return value

        report('running')
        current = None
        tick = time.monotonic()
        try:
            # The original sampler must release its shared GPU lock first.
            with common.gpu_lock() if device == 'cuda' else nullcontext():
                batch._configure(device, memory_fraction)
                data = final.make_data(plan, out)
                builder = common.observation_builder(plan['config'], device=device)
                for entry, regime, pair in ordered:
                    current = dict(case=entry['case'], regime=regime, **{k:entry[k] for k in IDENTITY_KEYS})
                    key = entry['case'], regime
                    if key in records:
                        continue
                    started = time.monotonic()
                    with torch.no_grad():
                        sample, identity, observed, known = final.build_inputs(data, builder, entry, regime, plan['config'], device)
                        fingerprint = common.tensor_fingerprint(known)
                        reference = known['reference_motion'][0].detach().cpu().numpy()
                        row = compare_saved(entry, regime, pair, sample, identity, observed, reference, fingerprint)
                    row.update(audit_plan_sha256=manifest_sha256, checked_at=final.now(), device=device,
                               elapsed_s=time.monotonic()-started)
                    append_record(records_path, row)
                    records[key] = row
                    del sample, observed, known, reference
                    if len(records) % every == 0:
                        report('running', elapsed_s=time.monotonic()-tick, current=current)
                        print(json.dumps(dict(completed_inputs=len(records), planned_inputs=len(ordered),
                                              elapsed_s=time.monotonic()-tick)), flush=True)
                require(len(records) == len(ordered), 'Incomplete exact input coverage')
                require(final.verify(out) == plan and sha256(out/'PLAN.json') == binding['PLAN_sha256']
                        and sha256(out/'rows.jsonl') == binding['rows_sha256'], 'Parent plan/source/rows changed during audit')
                require(dependency_closure([Path(__file__)]) == binding['source_sha256'], 'Audit source changed during execution')
            return report('completed_exact', elapsed_s=time.monotonic()-tick)
        except BaseException as error:
            report('failed', elapsed_s=time.monotonic()-tick, failure=dict(current or {}, error=repr(error)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest='command', required=True)
    sub = command.add_parser('inputs')
    sub.add_argument('--out', type=Path, required=True)
    sub.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    sub.add_argument('--resume', action='store_true')
    sub.add_argument('--memory-fraction', type=float, default=.5)
    sub.add_argument('--every', type=int, default=10)
    args = parser.parse_args()
    value = inputs(args.out, device=args.device, resume=args.resume, memory_fraction=args.memory_fraction, every=args.every)
    print(json.dumps(value, ensure_ascii=False, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
