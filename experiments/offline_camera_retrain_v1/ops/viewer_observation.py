"""NumPy-only display labels and saved-prediction metrics.

Native30 validation splits joints using ``observation_meta[..., 0] > 0``;
confidence does not change that split.  Display confidence classes are separate
from the evaluator's conditioned/unconditioned groups.  Missing observations
alone cannot establish image occlusion, especially for head-only EGO inputs.
No model, dataset loader, alignment, or inference is imported here.
"""
from __future__ import annotations

import numpy as np


MISSING = 0
LOW_CONFIDENCE = 1
TRUSTED = 2
CLASS_COLORS = np.array(((155, 160, 170), (245, 165, 35), (35, 195, 100)), dtype=np.uint8)
MASK_PROTOCOL = "native30-position-available-strictly-positive-v1"
MASK_EVIDENCE = {
    "native30_validation.py": "validate: conditioned = observation_meta[..., 0] > 0",
    "ops/adapter_trial.py": "recheck: conditioned = observation_meta[..., 0] > 0",
    "ops/validation_review.py": "independent_core: visible = observation_meta[..., 0] > 0",
    "native30_observations.py": "sanitize_observations: available positions require confidence > 0",
}


def _metadata(meta: np.ndarray) -> np.ndarray:
    value = np.asarray(meta)
    if value.ndim < 2 or value.shape[-2] != 22 or value.shape[-1] < 3:
        raise ValueError("Expected observation metadata shaped (..., 22, >=3)")
    if not np.isfinite(value[..., (0, 2)]).all():
        raise ValueError("Position availability and confidence must be finite")
    confidence = value[..., 2]
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Position confidence must lie in [0, 1]")
    return value


def observation_classes(meta: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return joint classes, RGB colors, and the exact display semantics.

    IDs are 0=missing/not supplied, 1=available with confidence below 1,
    2=available with confidence equal to 1.  A confidence of 0.99 remains
    *conditioned* for metrics; no arbitrary 0.5 confidence cutoff is applied.
    ``meta`` should be the sanitized metadata saved beside the prediction.
    """
    if mode not in ("ego", "exo"):
        raise ValueError("Observation mode must be ego or exo")
    value = _metadata(meta)
    available, confidence = value[..., 0] > 0, value[..., 2]
    if np.any(available & (confidence <= 0)):
        raise ValueError("Saved sanitized metadata has an available position with zero confidence")
    classes = np.full(available.shape, MISSING, dtype=np.uint8)
    classes[available & (confidence < 1)] = LOW_CONFIDENCE
    classes[available & (confidence == 1)] = TRUSTED
    missing_label = "未提供 / 缺失（不代表图像遮挡）" if mode == "ego" else "缺失（仅凭输入掩码不能确定遮挡原因）"
    explanation = dict(
        mode=mode,
        labels={str(MISSING): missing_label, str(LOW_CONFIDENCE): "可用，置信度低于 1", str(TRUSTED): "可用，置信度为 1"},
        colors={str(index): color.tolist() for index, color in enumerate(CLASS_COLORS)},
        mask_protocol=MASK_PROTOCOL,
        metric_conditioned="observation_meta[..., 0] > 0; confidence has no additional cutoff",
        display_confidence="Available confidence == 1 versus 0 < confidence < 1; not an evaluator threshold",
        trusted_is_ground_truth_guarantee=False,
        missing_is_occlusion=False,
        occlusion_requires="Original exo_reference/visible/person/static/rigid_counts, with source validity; noisy dropout alone is not occlusion",
        source_evidence=MASK_EVIDENCE,
    )
    return classes, CLASS_COLORS[classes], explanation


def masked_joint_metrics(pred: np.ndarray, truth: np.ndarray, meta: np.ndarray) -> dict:
    """World-position errors in mm, using the original availability mask.

    ``pred`` and ``truth`` contain already reconstructed joints in metres and
    the same coordinates, shaped (..., 22, 3). No alignment or rescaling occurs.
    Empty subsets return ``value=None, count=0``, never an artificial zero.
    GT is used only to describe saved playback and never alters the motion.
    """
    metadata = _metadata(meta)
    prediction, target = (np.asarray(array, dtype=np.float64) for array in (pred, truth))
    expected = metadata.shape[:-1] + (3,)
    if prediction.shape != expected or target.shape != expected:
        raise ValueError(f"Predicted/GT joints must both have shape {expected}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Predicted/GT joints must be finite; errors cannot silently discard joints")
    errors = np.linalg.norm(prediction - target, axis=-1) * 1000
    conditioned = metadata[..., 0] > 0

    def metric(mask):
        count = int(np.count_nonzero(mask))
        return dict(value=float(errors[mask].mean()) if count else None, count=count)

    visible, hidden = metric(conditioned), metric(~conditioned)
    return dict(
        mpjpe_mm=metric(np.ones(conditioned.shape, dtype=bool)),
        conditioned_mpjpe_mm=visible,
        unconditioned_mpjpe_mm=hidden,
        visible_mpjpe_mm=dict(visible),
        hidden_mpjpe_mm=dict(hidden),
        mask_protocol=MASK_PROTOCOL,
        note="Visible/hidden are conditioned/unconditioned input groups, not inferred image visibility; no confidence cutoff",
    )


def exo_first_hit_evidence(reference: np.ndarray, visible: np.ndarray,
                           blockers: dict[str, np.ndarray], *, valid=None) -> dict:
    """Describe original exported first-hit evidence without inferring it from GT.

    The export provides per-body-part ray counts, not joint contact labels.
    Counts with zero reference support cannot prove occlusion. This reproduces
    the exporter availability rule for auditing, without replacing the actual
    (potentially noisy/dropped) input mask used by the metrics above.
    """
    reference, visible = np.asarray(reference), np.asarray(visible)
    if reference.shape != visible.shape or reference.ndim < 1 or reference.shape[-1] != 22:
        raise ValueError("First-hit counts must have matching (..., 22) shapes")
    if set(blockers) != {"person", "static", "rigid"}:
        raise ValueError("Require the original person, static, and rigid blocker partitions")
    arrays = [reference, visible] + [np.asarray(blockers[name]) for name in ("person", "static", "rigid")]
    if any(array.shape != reference.shape or not np.isfinite(array).all() or np.any(array < 0)
           or np.any(array != np.floor(array)) for array in arrays):
        raise ValueError("First-hit counts must be finite nonnegative integers of the same shape")
    blocked = sum(arrays[2:])
    if not np.array_equal(reference, visible + blocked):
        raise ValueError("Original first-hit count partition does not balance")
    validity = np.ones(reference.shape[:-1], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if validity.shape != reference.shape[:-1]:
        raise ValueError("Source validity must match the non-joint count dimensions")
    fraction = visible / np.maximum(reference, 1)
    available = (visible >= 2) & (fraction >= .15) & validity[..., None]
    confidence = np.where(available, fraction * np.minimum(visible / 12., 1), 0)
    return dict(
        reference_supported=reference > 0,
        has_occluding_surface=(blocked > 0) & validity[..., None],
        no_reference_support=reference == 0,
        visible_fraction=fraction,
        exported_position_available=available,
        exported_position_confidence=confidence,
        availability_rule="visible_count >= 2 and visible/reference >= 0.15 and source observation_valid",
        source="native30_exo.py:first_hit; original per-body-part ray-count partition",
        note="Partial occlusion may coexist with available input; no reference support is not proof of occlusion. Noisy input dropout remains distinct.",
    )
