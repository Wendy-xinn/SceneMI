"""Natural TRAIN-window epochs, without changing sealed legacy samplers.

The population is the existing data object's eligible EXO windows. Every window
appears once per shuffled epoch, so corpus probabilities follow population size
rather than recording lengths or manually chosen corpus quotas. Overlapping
time intervals are allowed; this policy makes no non-overlap/cooldown claim.
"""
import copy
from contextlib import contextmanager
import hashlib
import json
import math

import numpy as np


PROTOCOL = "natural-global-train-windows-v1"
_INSTALLED = False


def validate_sampling_options(options, ego_probability):
    if options.get("protocol") != PROTOCOL:
        raise ValueError("Unknown natural TRAIN-window sampling protocol")
    if options.get("training_mode", "exo") != "exo" or ego_probability != 0:
        raise ValueError("Natural window sampling is an EXO-only TRAIN stage")
    if options.get("recording_cooldown", 0) != 0:
        raise ValueError("Natural global shuffle does not apply a recording cooldown")


def _population(data):
    if data.split != "train":
        raise ValueError("Natural window sampler cannot open VAL/test")
    rows, sources = [], []
    for group in sorted(data.members):
        for member, (meta, _) in enumerate(data.members[group]):
            if "exo" not in meta.get("allowed_modes", ["ego", "exo"]):
                continue
            starts = np.asarray(data.starts_for(group, member, "exo"))
            if starts.ndim != 1 or starts.dtype.kind not in "iu":
                raise ValueError("Eligible window starts must be a one-dimensional integer array")
            if (starts < 0).any() or len(np.unique(starts)) != len(starts):
                raise ValueError("Eligible window starts must be distinct and nonnegative")
            if not len(starts):
                continue
            dataset = "egobody" if group in ("camera_wearer", "interactee") else group
            record = dataset + "/" + (meta.get("recording") or meta["sequence_id"])
            identity = {key: meta.get(key) for key in (
                "sequence_id", "recording", "path", "source_first", "source_last",
                "bundle_id", "input_sha256", "allowed_modes")}
            sources.append(dict(group=group, member=member, identity=identity))
            rows.extend(dict(group=group, mode="exo", member=member, start=int(start),
                             recording_key=record) for start in starts)
    if not rows:
        raise ValueError("Natural TRAIN-window sampler has no eligible EXO windows")
    return rows, sources


def _plan_for_rows(rows):
    counts = {}
    for row in rows:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    return [dict(group=group, mode="exo", probability=count / len(rows))
            for group, count in sorted(counts.items())]


def natural_training_plan(data, config=None):
    """Return the actual EXO population proportions for preparation and logging."""
    if config is not None and config.get("training_mode") != "exo":
        raise ValueError("Natural window sampling requires training_mode='exo'")
    rows, _ = _population(data)
    return _plan_for_rows(rows)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return int(value)


class NaturalWindowSampler:
    """One unbiased permutation of all eligible windows per complete epoch.

    Restoring requires the same immutable data population, but no additional
    state file: the checkpoint contains the current permutation, RNG, cursor,
    epoch index and presentation count for every window.
    """

    def __init__(self, data, plan, options, seed):
        validate_sampling_options(options, 0.)
        self.data = data
        self.options = copy.deepcopy(options)
        self._rows, sources = _population(data)
        expected = _plan_for_rows(self._rows)
        actual = {}
        for row in plan:
            key = (row["group"], row["mode"])
            value = row["probability"]
            if key in actual or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid or duplicate natural window sampling plan row")
            actual[key] = value
        natural = {(row["group"], row["mode"]): row["probability"] for row in expected}
        if set(actual) != set(natural) or any(
                not math.isclose(actual[key], value, abs_tol=1e-12, rel_tol=0.)
                for key, value in natural.items()):
            raise ValueError("Sampling plan must equal natural eligible TRAIN-window proportions")
        self.plan = copy.deepcopy(plan)
        self.total_windows = len(self._rows)
        signature = dict(protocol=PROTOCOL, frames=int(data.frames), options=self.options,
                         plan=self.plan, sources=sources, windows=self._rows)
        self.fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        self.rng = np.random.default_rng(seed)
        self._permutation = self.rng.permutation(self.total_windows)
        self.cursor = 0
        self.cycle = 0  # Zero-based current epoch; its cursor can equal N at a boundary.
        self.draws = 0
        self._window_draw_counts = np.zeros(self.total_windows, dtype=np.int64)

    def draw(self):
        if self.cursor == self.total_windows:
            self._permutation = self.rng.permutation(self.total_windows)
            self.cursor = 0
            self.cycle += 1
        index = int(self._permutation[self.cursor])
        self.cursor += 1
        self.draws += 1
        self._window_draw_counts[index] += 1
        # None means no such constraint is applied, rather than claiming that
        # no temporal overlap/recording repetition occurred.
        return dict(self._rows[index], cooldown_fallback=None, overlap_fallback=None)

    def coverage_summary(self):
        groups = {}
        records = {}
        current = np.zeros(self.total_windows, dtype=bool)
        current[self._permutation[:self.cursor]] = True
        for index, row in enumerate(self._rows):
            group = row["group"]
            stats = groups.setdefault(group, dict(eligible_windows=0, total_draws=0,
                unique_windows_seen=0, current_cycle_windows_seen=0,
                eligible_recordings=0, recordings_seen=0))
            count = int(self._window_draw_counts[index])
            stats["eligible_windows"] += 1
            stats["total_draws"] += count
            stats["unique_windows_seen"] += int(count > 0)
            stats["current_cycle_windows_seen"] += int(current[index])
            seen = records.setdefault(group, {})
            seen[row["recording_key"]] = seen.get(row["recording_key"], False) or count > 0
        for group, stats in groups.items():
            stats["eligible_recordings"] = len(records[group])
            stats["recordings_seen"] = sum(records[group].values())
            stats["natural_probability"] = stats["eligible_windows"] / self.total_windows
            stats["window_coverage_fraction"] = stats["unique_windows_seen"] / stats["eligible_windows"]
            stats["repeated_presentations"] = stats["total_draws"] - stats["unique_windows_seen"]
        unique = int(np.count_nonzero(self._window_draw_counts))
        return dict(protocol=PROTOCOL, fingerprint=self.fingerprint,
            population_windows=self.total_windows, draws=self.draws,
            cycle_index=self.cycle, cursor=self.cursor, completed_cycles=self.draws // self.total_windows,
            unique_windows_seen=unique, repeated_presentations=self.draws - unique,
            min_presentations_per_window=int(self._window_draw_counts.min()),
            max_presentations_per_window=int(self._window_draw_counts.max()),
            recording_cooldown_applied=False, temporal_overlap_avoidance_applied=False,
            physical_recordings=len({row["recording_key"] for row in self._rows}), groups=groups)

    def state_dict(self):
        return copy.deepcopy(dict(protocol=PROTOCOL, fingerprint=self.fingerprint,
            options=self.options, plan=self.plan, rng=self.rng.bit_generator.state,
            draws=self.draws, cycle=self.cycle, cursor=self.cursor,
            permutation=self._permutation.tolist(), window_draw_counts=self._window_draw_counts.tolist(),
            coverage=self.coverage_summary()))

    def load_state_dict(self, state):
        if state.get("protocol") != PROTOCOL or state.get("fingerprint") != self.fingerprint:
            raise ValueError("Natural sampler configuration/source windows changed at resume")
        if state.get("options") != self.options or state.get("plan") != self.plan:
            raise ValueError("Natural sampler plan/options changed at resume")
        draws = _integer(state["draws"], "draws")
        cycle = _integer(state["cycle"], "cycle")
        cursor = _integer(state["cursor"], "cursor")
        if cursor > self.total_windows or draws != cycle * self.total_windows + cursor:
            raise ValueError("Invalid natural sampler cycle/cursor/draw count")
        permutation = np.array([_integer(v, "permutation index") for v in state["permutation"]], dtype=np.int64)
        if not np.array_equal(np.sort(permutation), np.arange(self.total_windows)):
            raise ValueError("Natural sampler permutation is not the complete window population")
        counts = np.array([_integer(v, "window draw count") for v in state["window_draw_counts"]], dtype=np.int64)
        expected = np.full(self.total_windows, cycle, dtype=np.int64)
        expected[permutation[:cursor]] += 1
        if not np.array_equal(counts, expected):
            raise ValueError("Natural sampler presentation counts do not match epoch coverage")
        restored_rng = np.random.default_rng()
        restored_rng.bit_generator.state = copy.deepcopy(state["rng"])
        # Commit only after the entire state validates.
        self.rng = restored_rng
        self._permutation = permutation
        self._window_draw_counts = counts
        self.draws, self.cycle, self.cursor = draws, cycle, cursor


@contextmanager
def install_natural_window_sampler():
    """Use the new policy through existing sample_item/data-resume interfaces."""
    global _INSTALLED
    from experiments.offline_camera_retrain_v1 import native30_sampling as legacy
    if _INSTALLED:
        raise RuntimeError("Natural window sampler scope cannot nest")
    old_sampler, old_validator = legacy.StratifiedRecordingSampler, legacy.validate_sampling_options
    _INSTALLED = True
    legacy.StratifiedRecordingSampler = NaturalWindowSampler
    legacy.validate_sampling_options = validate_sampling_options
    try:
        yield
    finally:
        legacy.StratifiedRecordingSampler = old_sampler
        legacy.validate_sampling_options = old_validator
        _INSTALLED = False
