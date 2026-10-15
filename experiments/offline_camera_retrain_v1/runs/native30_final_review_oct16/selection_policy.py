"""Conservative development selection for the bounded, paired Native30 review.

These tolerances are practical development guardrails, not confidence intervals,
statistical significance tests, or a final benchmark acceptance criterion.
"""
import math
from collections.abc import Mapping
from numbers import Real


REGIMES = ("clean", "noisy")
PANELS = ("all", "locomotion")
METRICS = (
    "mpjpe_mm", "conditioned_mpjpe_mm", "root_orientation_deg",
    "body_local_foot_error_mm", "foot_velocity_RMSE_m_s",
    "articulation_log_ratio_error",
)
GUARDS = (
    ("all", "mpjpe_mm", 1.05, 2.0),
    ("all", "conditioned_mpjpe_mm", 1.0, 5.0),
    ("all", "root_orientation_deg", 1.0, 3.0),
    ("locomotion", "body_local_foot_error_mm", 1.05, 5.0),
    ("locomotion", "foot_velocity_RMSE_m_s", 1.10, 0.02),
    ("locomotion", "articulation_log_ratio_error", 1.10, 0.05),
)
QUALITY = (
    ("locomotion", "body_local_foot_error_mm", 0.40),
    ("locomotion", "articulation_log_ratio_error", 0.30),
    ("locomotion", "foot_velocity_RMSE_m_s", 0.15),
    ("all", "mpjpe_mm", 0.15),
)
TIE_TOLERANCE = 0.03
NORMALIZATION_FLOOR = 1e-12


def _valid_number(value, *, positive=False):
    return (isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(float(value))
            and (value > 0 if positive else value >= 0))


def _validate(summary):
    errors = []
    if not isinstance(summary, Mapping):
        return ["summary must be a mapping"]
    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, Mapping):
        errors.append("aggregate must be a mapping")
    else:
        for panel in PANELS:
            for regime in REGIMES:
                key = panel + "/" + regime
                row = aggregate.get(key)
                if not isinstance(row, Mapping):
                    errors.append("missing or invalid metrics: " + key)
                    continue
                for metric in METRICS:
                    if not _valid_number(row.get(metric)):
                        errors.append(key + "/" + metric + " must be finite and nonnegative")
    timing = summary.get("timing")
    if not isinstance(timing, Mapping):
        errors.append("timing must be a mapping")
    else:
        for key in ("mean_compute_s", "mean_update_s"):
            if not _valid_number(timing.get(key), positive=True):
                errors.append("timing/" + key + " must be finite and positive")
    return errors


def _score(summary, baseline):
    total = 0.0
    for regime in REGIMES:
        for panel, metric, weight in QUALITY:
            key = panel + "/" + regime
            denominator = max(float(baseline["aggregate"][key][metric]), NORMALIZATION_FLOOR)
            total += weight * float(summary["aggregate"][key][metric]) / denominator
    return total / len(REGIMES)


def choose_candidate(summaries, control="control"):
    """Return a JSON-safe choice; incomplete candidates fail closed.

    ``eligible`` lists non-control arms passing every clean/noisy guardrail.
    ``scores`` contains normalized quality scores (None for invalid input).
    Control participates in the quality/speed comparison as the fallback.
    Among arms within 0.03 of the minimum score, prefer lower compute time;
    quality and arm name break any exact compute-time ties deterministically.
    """
    if not isinstance(summaries, Mapping) or control not in summaries:
        raise ValueError("A valid control summary is required")
    if any(not isinstance(name, str) for name in summaries):
        raise ValueError("Arm names must be strings")
    control_errors = _validate(summaries[control])
    if control_errors:
        raise ValueError("Invalid control: " + "; ".join(control_errors))
    baseline = summaries[control]
    reasons = {control: ["Valid control; retained as a fallback and selection option"]}
    scores = {control: _score(baseline, baseline)}
    eligible = []
    for name in sorted(summaries):
        if name == control:
            continue
        summary = summaries[name]
        errors = _validate(summary)
        if errors:
            reasons[name] = errors
            scores[name] = None
            continue
        scores[name] = _score(summary, baseline)
        if not math.isfinite(scores[name]):
            scores[name] = None
            reasons[name] = ["Normalized quality score is nonfinite"]
            continue
        for regime in REGIMES:
            for panel, metric, multiplier, allowance in GUARDS:
                key = panel + "/" + regime
                value = float(summary["aggregate"][key][metric])
                limit = multiplier * float(baseline["aggregate"][key][metric]) + allowance
                if value > limit:
                    errors.append(f"{key}/{metric}: {value:.9g} exceeds {limit:.9g}")
        reasons[name] = errors or ["Passes every clean/noisy development guardrail"]
        if not errors:
            eligible.append(name)
    selectable = [control] + eligible
    best_score = min(scores[name] for name in selectable)
    near_best = [name for name in selectable if scores[name] <= best_score + TIE_TOLERANCE]
    selected = min(near_best, key=lambda name: (
        float(summaries[name]["timing"]["mean_compute_s"]), scores[name], name))
    reasons[selected].append("Selected by quality score, then compute time within 0.03 of best")
    policy = {
        "scope": "Practical development noninferiority tolerances; not statistical significance or benchmark acceptance",
        "regimes": list(REGIMES),
        "guardrails": [dict(panel=p, metric=m, control_multiplier=f, additive_allowance=a)
                       for p, m, f, a in GUARDS],
        "quality_weights": [dict(panel=p, metric=m, weight=w) for p, m, w in QUALITY],
        "quality_aggregation": "Weighted sum of candidate/control metric ratios, averaged equally across clean/noisy",
        "normalization_denominator_floor": NORMALIZATION_FLOOR,
        "quality_tie_tolerance": TIE_TOLERANCE,
        "speed_metric": "mean_compute_s",
        "control_participates": True,
        "invalid_candidate": "ineligible; score null",
        "invalid_control": "raise ValueError",
    }
    return dict(selected=selected, eligible=eligible, reasons=reasons, scores=scores, policy=policy)
