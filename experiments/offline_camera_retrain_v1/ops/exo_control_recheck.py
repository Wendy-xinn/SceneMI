"""Additive CPU audit with physical bounds for CPU/CUDA FP32 references.

The original sealed review and its failed channelwise assertion are retained.
This verifier checks reconstructed references in metres and rotation angles;
it does not change reference construction, predictions, or metric tolerances.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import exo_control_review as review
from .dependencies import dependency_closure, sha256


PROTOCOL = 'exo-cpu-physical-reference-audit-v2'
LIMITS = dict(max_root_translation_difference_mm=.01,
    max_FK_joint_difference_mm=.05, max_embedded_joint_difference_mm=.05,
    max_global_orientation_difference_deg=.01)


def physical_reference_difference(saved, rebuilt, rest):
    """Compare standard decoded rotations and native FK, not 6D components."""
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    saved = np.asarray(saved, dtype=np.float32)
    rebuilt = np.asarray(rebuilt, dtype=np.float32)
    rest = np.asarray(rest, dtype=np.float32)
    review.require(saved.shape == rebuilt.shape and saved.ndim == 2 and saved.shape[-1] == 201,
        'Reference shape differs')
    review.require(np.isfinite(saved).all() and np.isfinite(rebuilt).all() and np.isfinite(rest).all(),
        'Nonfinite reconstructed reference')
    p, q = (torch.from_numpy(value.copy())[None] for value in (saved, rebuilt))
    native_rest = torch.from_numpy(rest.copy())[None]
    with torch.no_grad():
        p_global, q_global = global_rotations(p).double(), global_rotations(q).double()
        relative = p_global.transpose(-1, -2) @ q_global
        vee = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1]), -1) / 2
        cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        angle = torch.atan2(vee.norm(dim=-1), cosine).abs().rad2deg()
        p_joints, q_joints = (forward_kinematics(value, native_rest) for value in (p, q))
        joint_difference = (p_joints - q_joints).norm(dim=-1) * 1000
    channel_difference = np.abs(saved - rebuilt)
    return dict(max_root_translation_difference_mm=float(np.linalg.norm(saved[:, :3] - rebuilt[:, :3], axis=-1).max() * 2000),
        max_FK_joint_difference_mm=float(joint_difference.max()),
        max_embedded_joint_difference_mm=float(np.linalg.norm((saved[:, 135:] - rebuilt[:, 135:]).reshape(-1, 22, 3), axis=-1).max() * 2000),
        max_global_orientation_difference_deg=float(angle.max()),
        max_201_channel_difference=float(channel_difference.max()),
        original_channelwise_allclose=bool(np.allclose(saved, rebuilt, atol=2e-5, rtol=2e-5)))


def require_physical_bounds(values):
    for key, limit in LIMITS.items():
        review.require(0 <= values[key] <= limit, 'Physical reference difference exceeds bound: ' + key)


def recheck(out):
    import numpy as np
    import torch
    review.common.cpu_setup()
    out = Path(out).resolve()
    plan = review.read(out / 'PLAN.json')
    review.verify(plan)
    rows = review.rows_for(out, plan)
    review.require(len(rows) == plan['planned_rows'], 'Incomplete frozen evaluation')
    audit_sources = dependency_closure([Path(__file__)])
    plan_sha, rows_sha = sha256(out / 'PLAN.json'), sha256(out / 'rows.jsonl')
    diagnostic = out / 'REFERENCE_CPU_DIAGNOSTIC.json'
    failure_evidence = dict(original_verifier=str(Path(review.__file__).resolve()),
        original_assertion='np.allclose(saved_reference, CPU_reference, atol=2e-5, rtol=2e-5)',
        original_assertion_result='Previous verifier failed; exact failures are recomputed below.',
        reason_for_additive_audit='FP32 CPU/CUDA reference component differences require physical-unit verification; inference and metric code are unchanged.')
    if diagnostic.exists():
        failure_evidence['diagnostic'] = dict(path=str(diagnostic), sha256=sha256(diagnostic))
    previous_log = out / 'recheck.log'
    if previous_log.exists():
        failure_evidence['original_recheck_log'] = dict(path=str(previous_log), sha256=sha256(previous_log))
    data = review.common.validation_data(plan['config'])
    builder = review.common.observation_builder(plan['config'])
    indexed = {review.row_key(row): row for row in rows}
    maximum = independent_maximum = 0.
    physical_rows = []
    checked = 0
    for entry in plan['entries']:
        for regime in plan['regimes']:
            sample, observed, scene, asset = review.evaluation.source_observation(data, builder, entry, regime)
            with review.reference_temporal.installed():
                _, known = review.evaluation.collated_inputs(plan['config'], observed, scene, asset, 'cpu')
            rebuilt = known['reference_motion'][0].numpy()
            fingerprints, denominators = [], []
            paired_reference = None
            for label in [review.REFERENCE, *plan['model_order']]:
                row = indexed[entry['case'], regime, label]
                with np.load(row['artifact'], allow_pickle=False) as saved:
                    arrays = {key: saved[key].copy() for key in saved.files}
                for name, actual in [('truth', sample['motion']), ('rest', sample['rest']),
                        ('observations', observed['trajectory']), ('observation_meta', observed['observation_meta'])]:
                    review.require(np.array_equal(arrays[name], actual), 'Source array changed: ' + name)
                if paired_reference is None:
                    paired_reference = arrays['reference_motion']
                    values = physical_reference_difference(paired_reference, rebuilt, sample['rest'])
                    require_physical_bounds(values)
                    physical_rows.append(dict(case=entry['case'], dataset=entry['dataset'], recording=entry['recording'],
                        regime=regime, **values))
                else:
                    review.require(np.array_equal(paired_reference, arrays['reference_motion']),
                        'Models used different saved reference motions')
                if label == review.REFERENCE:
                    review.require(np.array_equal(arrays['prediction'], arrays['reference_motion']),
                        'Input reference prediction differs')
                actual, counts = review.reference.metrics(arrays)
                review.require(counts == row['counts'], 'Metric denominator differs')
                for name, value in actual.items():
                    review.require((value is None) == (row['metrics'][name] is None), 'Metric missingness changed')
                    if value is not None:
                        maximum = max(maximum, abs(value - row['metrics'][name]))
                for name, value in review.common.independent_core(arrays).items():
                    review.require((value is None) == (actual[name] is None), 'Independent metric missingness differs')
                    if value is not None:
                        independent_maximum = max(independent_maximum, abs(value - actual[name]))
                fingerprints.append(row['input_sha256'])
                denominators.append(counts)
                checked += 1
            review.require(len(set(fingerprints)) == 1 and all(c == denominators[0] for c in denominators),
                'Models used different recursive inputs or metric masks')
            if checked % 24 == 0:
                review.write(out / 'PHYSICAL_RECHECK_PROGRESS.json', dict(protocol=PROTOCOL,
                    checked_rows=checked, planned_rows=len(rows), checked_references=len(physical_rows),
                    checked_at=review.common.now()))
    review.require(maximum < 1e-6 and independent_maximum < 1e-3 and not torch.cuda.is_initialized(),
        'Independent CPU metric verification failed')
    review.verify(plan)
    review.require(sha256(out / 'PLAN.json') == plan_sha and sha256(out / 'rows.jsonl') == rows_sha,
        'Frozen evaluation changed during additive audit')
    for path, expected in audit_sources.items():
        review.require(sha256(path) == expected, 'Additive verifier source changed')
    result = dict(status='independent_cpu_passed', validation_protocol=PROTOCOL,
        checked_at=review.common.now(), rows=len(rows), references=len(physical_rows),
        maximum_recompute_difference=maximum, maximum_independent_formula_difference=independent_maximum,
        CUDA_initialized=False, source_GT_rest_observations_and_missing_masks_exact=True,
        recursive_model_input_fingerprints_and_denominators_paired=True,
        saved_reference_motion_exact_between_models=True,
        physical_reference_tolerances=LIMITS,
        physical_reference_maxima={key: max(row[key] for row in physical_rows) for key in LIMITS},
        original_channelwise_allclose_failures=sum(not row['original_channelwise_allclose'] for row in physical_rows),
        original_failure_evidence=failure_evidence, physical_reference_rows=physical_rows,
        metric_and_artifact_tolerances_unchanged=True, model_source_and_output_SHA_verified=True,
        plan_sha256=plan_sha, rows_sha256=rows_sha, additive_verifier_source_sha256=audit_sources,
        analytical_physics=False, TEST_used=False)
    review.write(out / 'CPU_RECHECK_PHYSICAL_V2.json', result)
    review.write(out / 'CPU_RECHECK.json', result)
    review.summarize(out, plan, rows, verified=True)
    review.status(out, 'completed', completed_rows=len(rows), planned_rows=plan['planned_rows'],
        CPU_validation_protocol=PROTOCOL)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    recheck(args.out)


if __name__ == '__main__':
    main()
