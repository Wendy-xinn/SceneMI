"""Identity-only, natural-size fixed and rotating official VAL panels.

Checkpoint selection uses the fixed EXO identities only. Rotating examples and
head-only EGO retention probes are deliberately excluded from selection. No
model, motion target, visibility outcome, TRAIN or TEST sample is inspected.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
import hashlib
import json
import math
import statistics

DATASETS = ("rich", "trumans", "egobody")
GROUP_DATASET = {"rich": "rich", "trumans": "trumans",
                 "camera_wearer": "egobody", "interactee": "egobody"}
PROTOCOL = "natural-window-fixed100-rotating100-exo-val-v1"
WRIST_SEQUENCE = "recording_20220215_S21_S22_01_interactee_1901-4901_master_full"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def identity(row):
    return row["group"], row["mode"], row["sequence_id"], int(row["start"])


def largest_remainder(counts, total):
    """Allocate an exact integer budget without replacing natural proportions."""
    if type(total) is not int or total < 1 or not counts or any(
            type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("Positive integer budget and nonnegative integer counts required")
    size = sum(counts.values())
    if size < total:
        raise ValueError("Panel exceeds the distinct eligible population")
    base = {key: total * value // size for key, value in counts.items()}
    for key in sorted(counts, key=lambda key: (-(total * counts[key] % size), key))[:total-sum(base.values())]:
        base[key] += 1
    if any(counts[key] and base[key] == 0 for key in counts):
        raise ValueError("Panel budget cannot represent every nonempty dataset")
    return base


def eligible_rows(data, mode, seed):
    """Enumerate actual starts_for eligibility; never infer a different grid."""
    if data.split != "validation" or mode not in ("ego", "exo"):
        raise ValueError("Panels may enumerate official validation EGO/EXO only")
    result = []
    for group in sorted(data.members):
        if group not in GROUP_DATASET:
            raise ValueError("Unknown native30 source group")
        for member, (meta, _) in enumerate(data.members[group]):
            if meta.get("split") != "validation":
                raise ValueError("Panel source is not official VAL")
            default = ["ego", "exo"] if group in ("camera_wearer", "interactee") else ["ego"]
            if mode not in meta.get("allowed_modes", default):
                continue
            for first in data.starts_for(group, member, mode):
                first = int(first)
                row = dict(group=group, mode=mode, member=member, start=first,
                    joints=[15] if mode == "ego" else None,
                    sequence_id=meta["sequence_id"], source_first=int(meta["source_first"]) + first,
                    dataset=GROUP_DATASET[group], recording=meta.get("recording") or meta["sequence_id"],
                    path=str(meta["path"]))
                key = json.dumps([group, mode, meta["sequence_id"], row["source_first"]], separators=(",", ":"))
                row["case_seed"] = (seed + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")) % 2**32
                result.append(row)
    if len({identity(row) for row in result}) != len(result):
        raise ValueError("Duplicated native validation source identities")
    return sorted(result, key=identity)


def _overlap(row, history, frames):
    source = row["group"], row["sequence_id"]
    start = row["start"]
    return sum(max(0, min(start+frames, earlier+frames)-max(start, earlier))
        for earlier in history.get(source, ()))


def _draw(pool, count, history, seed, frames):
    """Spread recordings, natural within-corpus roles, then separate time spans."""
    if len(pool) < count:
        raise ValueError("Insufficient distinct windows in current sampling cycle")
    remaining = {identity(row): row for row in pool}
    record_count, group_count, source_count = Counter(), Counter(), Counter()
    group_population = Counter(row["group"] for row in pool)
    selected = []
    for _ in range(count):
        def rank(row):
            source = row["group"], row["sequence_id"]
            record = row["dataset"], row["recording"]
            return (record_count[record],
                group_count[row["group"]] / group_population[row["group"]],
                _overlap(row, history, frames), source_count[source],
                digest([seed, list(identity(row))]))
        chosen = min(remaining.values(), key=rank)
        selected.append(copy.deepcopy(chosen))
        remaining.pop(identity(chosen))
        record_count[(chosen["dataset"], chosen["recording"])] += 1
        group_count[chosen["group"]] += 1
        source = chosen["group"], chosen["sequence_id"]
        source_count[source] += 1
        history[source].append(chosen["start"])
    return selected


def _label(rows, label, round_index):
    result = []
    for i, row in enumerate(rows):
        entry = dict(row, case=i, panel=label, monitor_round=round_index)
        entry["identity_sha256"] = digest(list(identity(row)))
        result.append(entry)
    return result


def _coverage(rows):
    return {name: dict(windows=sum(r["dataset"] == name for r in rows),
        recordings=len({r["recording"] for r in rows if r["dataset"] == name}),
        groups=dict(Counter(r["group"] for r in rows if r["dataset"] == name))) for name in DATASETS}


def build_panels(data, *, seed, count=100, milestones=(2000, 4000, 6000, 8000, 10000), ego_per_corpus=3):
    if type(seed) is not int or type(ego_per_corpus) is not int or ego_per_corpus < 1:
        raise ValueError("Integer seed and positive EGO probe count required")
    if not milestones or any(type(v) is not int or v < 1 for v in milestones) or list(milestones) != sorted(set(milestones)):
        raise ValueError("Milestones must be distinct increasing positive integers")
    exo = eligible_rows(data, "exo", seed)
    counts = {name: sum(r["dataset"] == name for r in exo) for name in DATASETS}
    if any(not value for value in counts.values()):
        raise ValueError("Every dataset needs eligible official EXO validation windows")
    quotas = largest_remainder(counts, count)
    frames = int(data.frames)
    fixed, rotating, cycles, coverage = [], {}, {}, {}
    histories = {name: defaultdict(list) for name in DATASETS}
    pools = {name: [r for r in exo if r["dataset"] == name] for name in DATASETS}
    for name in DATASETS:
        fixed.extend(_draw(pools[name], quotas[name], histories[name], [seed, "fixed", name], frames))
    fixed = _label(fixed, "fixed", 0)
    fixed_ids = {identity(r) for r in fixed}
    available = {name: [r for r in pools[name] if identity(r) not in fixed_ids] for name in DATASETS}
    if any(len(available[name]) < quotas[name] for name in DATASETS):
        raise ValueError("Population cannot keep fixed and rotating windows disjoint")
    used = {name: set() for name in DATASETS}
    cycle = {name: 0 for name in DATASETS}
    for round_index, step in enumerate(milestones, 1):
        chosen, events = [], {}
        for name in DATASETS:
            quota = quotas[name]
            remaining = [r for r in available[name] if identity(r) not in used[name]]
            selected = []
            started_cycle = cycle[name]
            # Consume the tail before restarting; do not silently abandon rare windows.
            if len(remaining) < quota:
                selected.extend(_draw(remaining, len(remaining), histories[name], [seed, step, name, "tail"], frames))
                used[name].update(identity(r) for r in selected)
                cycle[name] += 1
                used[name] = set()
                taken = {identity(r) for r in selected}
                remaining = [r for r in available[name] if identity(r) not in taken]
            extra = _draw(remaining, quota-len(selected), histories[name], [seed, step, name], frames)
            selected.extend(extra)
            used[name].update(identity(r) for r in extra)
            chosen.extend(selected)
            events[name] = dict(cycle_start=started_cycle, cycle_end=cycle[name],
                crossed_cycle=started_cycle != cycle[name], used_in_current_cycle=len(used[name]),
                rotation_population=len(available[name]))
        rotating[str(step)] = _label(chosen, "rotating", round_index)
        cycles[str(step)] = events
        coverage[str(step)] = _coverage(chosen)
    ego_pool = eligible_rows(data, "ego", seed)
    ego = []
    for name in DATASETS:
        candidates = [r for r in ego_pool if r["dataset"] == name]
        ego.extend(_draw(candidates, ego_per_corpus, defaultdict(list), [seed, "ego", name], frames))
    ego = _label(ego, "ego_retention", 0)
    wrist = [row for row in exo if row["sequence_id"] == WRIST_SEQUENCE and row["start"] == 1984]
    if len(wrist) > 1:
        raise ValueError("Known wrist regression identity is ambiguous")
    regression = _label([dict(row, case_seed=1568365135) for row in wrist], "regression_probe", 0)
    manifest = dict(protocol=PROTOCOL, seed=seed, split="validation", TEST_used=False,
        frames=frames, fixed_exo_count=count, rotating_exo_count=count, milestones=list(milestones),
        eligible_exo_counts=counts, eligible_exo_total=sum(counts.values()),
        natural_dataset_weights={name: counts[name]/sum(counts.values()) for name in DATASETS},
        dataset_quotas=quotas, fixed_exo=fixed, rotating_exo=rotating, ego_retention=ego,
        regression_exo=regression,
        regression_registration=dict(sequence_id=WRIST_SEQUENCE, start=1984, case_seed=1568365135,
            status="registered_exact_original_window" if wrist else "unavailable",
            reason=None if wrist else "Exact known source/start is not an eligible official VAL EXO window; no replacement substituted.",
            excluded_from_selection=True),
        rotation_cycles=cycles, fixed_coverage=_coverage(fixed), rotating_coverage=coverage,
        population_sha256=digest([dict(r) for r in exo]),
        selection_policy="Identity-only: spread recordings; balance source roles by eligible counts; prefer nonoverlapping native time spans; deterministic hash tiebreak. No GT or prediction ranking.",
        rotation_policy="EXO fixed identities excluded permanently. Consume rotating windows without replacement across evaluations, restarting a corpus only after exhaustion; cycles are explicit.",
        best_policy="Only fixed EXO windows, equal window means within corpus, actual eligible VAL-window corpus weights; clean/noisy equal weight. Empty hidden masks excluded. Rotating and EGO never select best.",
        disclosure="A stratified identity panel is monitoring, not a full-validation estimate or independent TEST. Per-corpus recording spreading changes within-corpus sample composition; natural weights refer to corpus sizes.",
        panel_sha256=dict(fixed_exo=digest(fixed), ego_retention=digest(ego), regression_exo=digest(regression),
            rotating_exo={key: digest(rows) for key, rows in rotating.items()}))
    validate_manifest(data, manifest)
    return manifest


def select_panel(manifest, step=0, kind="fixed", *, include_ego=False, include_regression=False):
    if manifest.get("protocol") != PROTOCOL or kind not in ("fixed", "rotating", "ego", "regression"):
        raise ValueError("Unregistered long-run panel")
    if kind == "fixed":
        rows = manifest["fixed_exo"]
    elif kind == "ego":
        rows = manifest["ego_retention"]
    elif kind == "regression":
        rows = manifest["regression_exo"]
    else:
        if str(step) not in manifest["rotating_exo"]:
            raise ValueError("Unregistered rotating evaluation milestone")
        rows = manifest["rotating_exo"][str(step)]
    result = copy.deepcopy(rows)
    if include_ego and kind != "ego":
        result.extend(copy.deepcopy(manifest["ego_retention"]))
    if include_regression and kind != "regression":
        result.extend(copy.deepcopy(manifest["regression_exo"]))
    # Native validate creates case=i itself; engine strips these output indices.
    return [dict(row, case=i) for i, row in enumerate(result)]


def validate_manifest(data, manifest):
    if data.split != "validation" or manifest.get("split") != "validation" or manifest.get("protocol") != PROTOCOL:
        raise ValueError("Only the registered official VAL manifest is valid")
    pool = eligible_rows(data, "exo", manifest["seed"])
    if digest(pool) != manifest["population_sha256"]:
        raise ValueError("Eligible validation population changed")
    counts = {name: sum(row["dataset"] == name for row in pool) for name in DATASETS}
    if counts != manifest["eligible_exo_counts"] or manifest["dataset_quotas"] != largest_remainder(counts, manifest["fixed_exo_count"]):
        raise ValueError("Natural population counts or quotas changed")
    if manifest["natural_dataset_weights"] != {name: counts[name]/sum(counts.values()) for name in DATASETS}:
        raise ValueError("Natural dataset weights changed")
    hashes = manifest["panel_sha256"]
    if digest(manifest["fixed_exo"]) != hashes["fixed_exo"] or digest(manifest["ego_retention"]) != hashes["ego_retention"]:
        raise ValueError("Fixed/retention panel hash changed")
    if digest(manifest["regression_exo"]) != hashes["regression_exo"]:
        raise ValueError("Known regression probe hash changed")
    fixed_ids = {identity(r) for r in manifest["fixed_exo"]}
    if len(fixed_ids) != manifest["fixed_exo_count"]:
        raise ValueError("Fixed EXO identities repeat")
    for key, rows in manifest["rotating_exo"].items():
        if digest(rows) != hashes["rotating_exo"][key] or len(rows) != manifest["rotating_exo_count"]:
            raise ValueError("Rotating panel hash or count changed")
        keys = {identity(r) for r in rows}
        if len(keys) != len(rows) or fixed_ids & keys:
            raise ValueError("Duplicated or fixed-overlapping rotating identity")
    all_panels = [manifest["fixed_exo"], *manifest["rotating_exo"].values(), manifest["ego_retention"], manifest["regression_exo"]]
    for rows in all_panels:
        for row in rows:
            meta, _ = data.members[row["group"]][row["member"]]
            expected_dataset = GROUP_DATASET[row["group"]]
            if row["dataset"] != expected_dataset or meta["sequence_id"] != row["sequence_id"] or \
                    int(meta["source_first"])+row["start"] != row["source_first"] or \
                    row["recording"] != (meta.get("recording") or meta["sequence_id"]) or \
                    row["start"] not in data.starts_for(row["group"], row["member"], row["mode"]):
                raise ValueError("Panel identity is not an eligible official window")
            if row["mode"] == "ego" and row["joints"] != [15] or row["mode"] == "exo" and row["joints"] is not None:
                raise ValueError("Panel changes the observation protocol")
    return manifest


def _metric_summary(rows, metric, field):
    values, weighted_sum, count = [], 0., 0
    for row in rows:
        item = row.get(field, {}).get(metric)
        if item is None:
            continue
        n, value = item.get("count", 0), item.get("value")
        if not n:
            if value is not None:
                raise ValueError("A zero metric denominator must have null value")
            continue
        if type(n) is not int or n < 0 or value is None or not math.isfinite(value):
            raise ValueError("Nonfinite or invalid fixed-panel metric")
        values.append(float(value)); weighted_sum += float(value)*n; count += n
    return dict(window_mean=statistics.mean(values) if values else None,
        joint_pooled_mean=weighted_sum/count if count else None,
        valid_windows=len(values), valid_joint_samples=count)


def score_fixed_rows(rows, manifest):
    """Score fixed EXO only; require complete clean/noisy registered identities."""
    registered = {identity(row): row for row in manifest["fixed_exo"]}
    expected = set(registered)
    selected = [row for row in rows if row.get("mode") == "exo" and row.get("panel") == "fixed"
        and row.get("scene_variant", "scene_on") in ("scene_on", None)]
    seen = Counter((identity(row), row.get("regime")) for row in selected)
    if set(seen) != {(key, regime) for key in expected for regime in ("clean", "noisy")} or any(value != 1 for value in seen.values()):
        raise ValueError("Fixed EXO selection needs exactly one clean/noisy result per registered window")
    for row in selected:
        original = registered[identity(row)]
        if any(row.get(key) != original[key] for key in
                ("dataset", "recording", "member", "source_first", "case_seed")):
            raise ValueError("Fixed result source metadata or seed differs from registration")
    fields = ["metrics"]
    if any("unconditioned_mpjpe_mm" not in row.get("metrics", {}) or
            "conditioned_mpjpe_mm" not in row.get("metrics", {}) for row in selected):
        raise ValueError("Every fixed result needs explicit hidden/visible metric denominators")
    if any("reference_metrics" in row for row in selected):
        if not all("reference_metrics" in row for row in selected):
            raise ValueError("Reference comparator is only partially recorded")
        fields.append("reference_metrics")
    metric_names = sorted({name for row in selected for field in fields for name in row[field]})
    groups, totals, recording_diagnostics = {}, {}, {}
    for field in fields:
        totals[field] = {}
        for regime in ("clean", "noisy"):
            corpus = {}
            for name in DATASETS:
                subset = [row for row in selected if row["dataset"] == name and row["regime"] == regime]
                corpus[name] = {metric: _metric_summary(subset, metric, field) for metric in metric_names}
                groups["/".join((field, regime, name))] = corpus[name]
                record_values = {}
                for record in sorted({row["recording"] for row in subset}):
                    current = [row for row in subset if row["recording"] == record]
                    record_values[record] = {metric: _metric_summary(current, metric, field) for metric in metric_names}
                recording_diagnostics["/".join((field, regime, name))] = dict(recordings=record_values,
                    equal_recording_means={metric: statistics.mean(values[metric]["window_mean"]
                        for values in record_values.values() if values[metric]["window_mean"] is not None)
                        if any(values[metric]["window_mean"] is not None for values in record_values.values()) else None
                        for metric in metric_names})
            total = {}
            for metric in metric_names:
                valid = {name: corpus[name][metric] for name in DATASETS if corpus[name][metric]["window_mean"] is not None}
                weight = sum(manifest["natural_dataset_weights"][name] for name in valid)
                total[metric] = dict(natural_window_mean=sum(manifest["natural_dataset_weights"][name]*item["window_mean"] for name, item in valid.items())/weight if weight else None,
                    equal_dataset_mean=statistics.mean(item["window_mean"] for item in valid.values()) if valid else None,
                    included_datasets=list(valid), included_dataset_weight=weight,
                    valid_windows=sum(item["valid_windows"] for item in valid.values()),
                    valid_joint_samples=sum(item["valid_joint_samples"] for item in valid.values()))
            totals[field][regime] = total
    hidden = "unconditioned_mpjpe_mm"
    scores = [totals["metrics"][regime][hidden]["natural_window_mean"] for regime in ("clean", "noisy")]
    if any(value is None for value in scores) or any(len(totals["metrics"][regime][hidden]["included_datasets"]) != len(DATASETS) for regime in ("clean", "noisy")):
        raise ValueError("Selection requires nonempty hidden evidence in every dataset/regime")
    result = dict(protocol=PROTOCOL, fixed_panel_sha256=manifest["panel_sha256"]["fixed_exo"],
        score=statistics.mean(scores), score_metric=hidden, lower_is_better=True,
        regimes=["clean", "noisy"], natural_dataset_weights=manifest["natural_dataset_weights"],
        fixed_exo_windows=len(expected), scored_predictions=len(selected),
        selection_excludes=["rotating", "ego_retention", "scene_off"],
        aggregation="Equal valid-window metric means within dataset; actual eligible VAL-window corpus weights; equal clean/noisy mean. Joint-pooled and equal-dataset metrics are diagnostics.",
        groups=groups, totals=totals, recording_diagnostics=recording_diagnostics)
    if "reference_metrics" in totals:
        result["reference_score"] = statistics.mean(totals["reference_metrics"][regime][hidden]["natural_window_mean"] for regime in ("clean", "noisy"))
    return result
