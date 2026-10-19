"""CPU-only posthoc audit of an already completed natural-window EXO stage.

Consumes saved fixed-panel predictions only: no inference, dataset access,
training, TEST or checkpoint promotion. This independent entry point is sealed
in its own report and does not alter the historical training source seal.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from .dependencies import dependency_closure, sha256
from .exo_control_report import motion_metrics
from .exo_longrun_panel import DATASETS, select_panel

PROTOCOL = "natural-exo-longrun-saved-fixed-CPU-report-v1"
REGIMES = ("clean", "noisy")
IDENTITY_FIELDS = ("dataset", "recording", "group", "mode", "member", "sequence_id",
                   "source_first", "start", "case_seed", "panel")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def latest_rows(path):
    return json.loads(Path(path).read_text().splitlines()[-1])["cases"]


def bind_rows(rows, panel, step):
    indexed = {}
    for row in rows:
        index, regime = row["case"], row["regime"]
        require(type(index) is int and 0 <= index < len(panel) and regime in REGIMES,
                "Unregistered fixed-panel case/regime")
        require((index, regime) not in indexed, "Duplicate fixed-panel result")
        require(row["step"] == step and all(row.get(key) == panel[index][key] for key in IDENTITY_FIELDS),
                "Fixed-panel source identity, seed, or stage differs")
        require(not row.get("scene_variant"), "Scene-OFF arrays cannot enter the fixed scene-ON audit")
        indexed[index, regime] = row
    require(set(indexed) == {(i, regime) for i in range(len(panel)) for regime in REGIMES},
            "Incomplete fixed-panel result coverage")
    return indexed


def verify_arrays(before, after, panel):
    import numpy as np
    require(str(before["identities_json"]) == str(after["identities_json"]), "Saved source identities differ")
    ego = 0
    for i, entry in enumerate(panel):
        for key in (f"{i}_truth", f"{i}_rest"):
            require(np.array_equal(before[key], after[key]), "Paired GT/rest differs: " + key)
        for regime in REGIMES:
            for field in ("observations", "observation_meta", "reference_motion"):
                key = f"{i}_{regime}_{field}"
                require(np.array_equal(before[key], after[key]), "Paired saved input/reference differs: " + key)
            if entry["mode"] == "ego":
                key = f"{i}_{regime}_prediction"
                require(np.array_equal(before[key], after[key]), "EGO prediction changed: " + key)
                ego += 1
    return dict(exact=True, checked_anchor_regimes=ego,
                compared_inputs=["observations", "observation_meta", "reference_motion", "GT", "rest", "identities"],
                limitation="Scene occupancy/camera tensors are not serialized in these prediction archives; source seals remain separate evidence.")


def bind_archive(arrays, panel, step):
    require(int(arrays["step"]) == step, "Saved array stage differs")
    identities = json.loads(str(arrays["identities_json"]))
    require(len(identities) == len(panel), "Saved identity count differs")
    for source, entry in zip(identities, panel):
        require(all(source.get(key) == entry[key] for key in ("sequence_id", "group", "mode", "start"))
            and source["source_first"] + source["start"] == entry["source_first"]
            and source["source_start_30fps"] == entry["source_first"]
            and (source.get("recording") or source["sequence_id"]) == entry["recording"],
            "Saved source identity does not match registered case")


def saved_metrics(arrays, index, regime, *, reference=False):
    return motion_metrics(dict(prediction=arrays[f"{index}_{regime}_" + ("reference_motion" if reference else "prediction")],
        truth=arrays[f"{index}_truth"], rest=arrays[f"{index}_rest"],
        observation_meta=arrays[f"{index}_{regime}_observation_meta"]))


def summarize(rows, weights):
    """Equal valid-window means, natural corpus weights; pooled values separate."""
    groups = {}
    for label in sorted({row["model"] for row in rows}):
        for regime in REGIMES:
            corpus = {}
            for dataset in DATASETS:
                chosen = [row for row in rows if row["model"] == label and row["regime"] == regime
                          and row["dataset"] == dataset and row["mode"] == "exo" and row["panel"] == "fixed"]
                require(chosen, "Missing dataset/regime/model fixed metrics")
                corpus[dataset] = {}
                for metric in sorted(chosen[0]["metrics"]):
                    valid = []
                    for row in chosen:
                        value, count = row["metrics"][metric], row["counts"][metric]
                        require(type(count) is int and count >= 0 and (value is None) == (count == 0),
                                "Invalid CPU metric denominator")
                        if count:
                            require(math.isfinite(value), "Nonfinite CPU metric")
                            valid.append((value, count))
                    count = sum(n for _, n in valid)
                    corpus[dataset][metric] = dict(window_mean=statistics.mean(v for v, _ in valid) if valid else None,
                        pooled=sum(v*n for v, n in valid)/count if count else None,
                        count=count, valid_windows=len(valid), total_windows=len(chosen))
                groups["/".join((label, regime, dataset))] = corpus[dataset]
            total = {}
            for metric in corpus[DATASETS[0]]:
                valid = {name: corpus[name][metric]["window_mean"] for name in DATASETS
                         if corpus[name][metric]["window_mean"] is not None}
                denominator = sum(weights[name] for name in valid)
                total[metric] = dict(natural_window_mean=sum(weights[name]*value for name, value in valid.items())/denominator if denominator else None,
                    equal_dataset_mean=statistics.mean(valid.values()) if valid else None,
                    valid_datasets=list(valid), included_dataset_weight=denominator)
            groups["/".join((label, regime, "natural_weighted"))] = total
    return groups


def comparisons(groups, best_step):
    result = []
    label = "adapter_" + str(best_step)
    for regime in REGIMES:
        for dataset in (*DATASETS, "natural_weighted"):
            reduction = "natural_window_mean" if dataset == "natural_weighted" else "window_mean"
            current = groups["/".join((label, regime, dataset))]["hidden_mpjpe_mm"][reduction]
            for other in ("control", "input_reference"):
                baseline = groups["/".join((other, regime, dataset))]["hidden_mpjpe_mm"][reduction]
                result.append(dict(dataset=dataset, regime=regime, comparator=other,
                    best_hidden_mm=current, baseline_hidden_mm=baseline,
                    delta_mm=current-baseline if current is not None and baseline is not None else None,
                    better=current < baseline if current is not None and baseline is not None else None))
    return result


def report(run):
    import numpy as np
    import torch
    from .adapt import write, now
    run = Path(run).resolve()
    require(not torch.cuda.is_initialized(), "Report must start CPU-only")
    torch.set_num_threads(4)
    protocol = read(run / "training/protocol.json")
    require(protocol["status"] == "completed" and protocol["completed_steps"] == 10000,
            "Only a completed 10000-update run may be audited")
    best = read(run / "training/best.json")
    require(sha256(run / "training/best.pt") == best["checkpoint_sha256"] == sha256(best["source_checkpoint"]),
            "Best weights/pointer/source SHA differ")
    cpu = read(run / "CPU_MODEL_VERIFY.json")
    require(cpu["status"] == "independent_CPU_model_passed" and cpu["CUDA_initialized"] is False
            and cpu["model"]["checkpoint_sha256"] == best["checkpoint_sha256"],
            "Independent best-checkpoint CPU reconstruction must pass first")
    effective = read(run / "prepared/effective_config.json")
    for path, expected in effective["code_sha256"].items():
        require(sha256(path) == expected, "Historical training source changed: " + path)
    require(sha256(run / "prepared/panels.json") == effective["panels_sha256"], "Panel manifest seal differs")
    manifest = read(run / "prepared/panels.json")
    panel = select_panel(manifest, kind="fixed", include_ego=True, include_regression=True)
    baseline = run / "baseline_validation"
    baseline_rows = bind_rows(latest_rows(baseline / "validation.jsonl"), panel, 0)
    inputs = [run / "training/protocol.json", run / "training/best.json", run / "CPU_MODEL_VERIFY.json",
              run / "prepared/effective_config.json", run / "prepared/panels.json",
              baseline / "validation.jsonl", baseline / "validation_latest.npz"]
    checks = read(run / "prepared/input_checks.json")
    require(sha256(baseline / "validation.jsonl") == checks["baseline_validation_sha256"]
            and sha256(baseline / "validation_latest.npz") == checks["baseline_arrays_sha256"],
            "Preflight baseline arrays/metrics changed")
    inputs.append(run / "prepared/input_checks.json")
    rows, retention = [], {}
    with np.load(baseline / "validation_latest.npz", allow_pickle=False) as before:
        bind_archive(before, panel, 0)
        for (i, regime), row in baseline_rows.items():
            for label, reference in (("control", False), ("input_reference", True)):
                values, counts = saved_metrics(before, i, regime, reference=reference)
                rows.append(dict(row, model=label, metrics=values, counts=counts))
        for step in manifest["milestones"]:
            folder = run / "fixed_validation" / f"step_{step:06d}"
            review_path = run / "training/reviews" / f"step_{step:06d}.json"
            review = read(review_path)
            require(review["step"] == step and sha256(review["checkpoint"]) == review["checkpoint_sha256"],
                    "Milestone checkpoint/source SHA differs")
            require(sha256(folder / "validation_latest.npz") == review["fixed_arrays_sha256"],
                    "Milestone prediction arrays changed")
            current_rows = bind_rows(latest_rows(folder / "validation.jsonl"), panel, step)
            inputs.extend([review_path, folder / "validation.jsonl", folder / "validation_latest.npz"])
            with np.load(folder / "validation_latest.npz", allow_pickle=False) as after:
                bind_archive(after, panel, step)
                retention[str(step)] = verify_arrays(before, after, panel)
                for (i, regime), row in current_rows.items():
                    values, counts = saved_metrics(after, i, regime)
                    rows.append(dict(row, model="adapter_" + str(step), metrics=values, counts=counts))
    source_sha = dependency_closure([Path(__file__)])
    input_sha = {str(path): sha256(path) for path in inputs}
    input_sha[str((run / "training/best.pt").resolve())] = best["checkpoint_sha256"]
    for step in manifest["milestones"]:
        record = read(run / "training/reviews" / f"step_{step:06d}.json")
        input_sha[str(Path(record["checkpoint"]).resolve())] = record["checkpoint_sha256"]
    groups = summarize(rows, manifest["natural_dataset_weights"])
    output = dict(protocol=PROTOCOL, status="CPU_saved_array_audit_passed", checked_at=now(),
        CUDA_initialized=torch.cuda.is_initialized(), TEST_used=False, inference_calls=0,
        best_stage_step=best["stage_step"], best_checkpoint_sha256=best["checkpoint_sha256"],
        input_sha256=input_sha, report_source_sha256=source_sha,
        natural_dataset_weights=manifest["natural_dataset_weights"], groups=groups,
        best_hidden_comparisons=comparisons(groups, best["stage_step"]), EGO_retention=retention,
        wrist_probe_rows=[row for row in rows if row["panel"] == "regression_probe"],
        rows=rows, checkpoint_selection_changed=False, scientific_acceptance="Not automatically determined.",
        aggregation="Fixed EXO only, equal valid-window means per corpus, natural eligible-VAL-window corpus weights; equal-corpus and pooled diagnostics separate. EGO and wrist probes excluded from best.",
        limitations=["Development VAL, not independent TEST or proof of real monocular EXO generalization.",
            "Wrist angular quantiles alone do not prove or disprove jitter.",
            "Scene tensors are not archived; report verifies saved observation/reference/GT equality and historical source seals."])
    require(not torch.cuda.is_initialized(), "CPU report unexpectedly initialized CUDA")
    write(run / "CPU_TEMPORAL_REVIEW.json", output)
    lines = ["# EXO 10k 保存结果 CPU 配对审计", "", "best 步数：" + str(best["stage_step"]),
             "固定窗口的输入、GT、reference 已核对；各里程碑 EGO 输出精确保持。", "",
             "| 数据集 | 条件 | 对照 | best hidden mm | 对照 hidden mm | 变化 mm |",
             "|---|---|---|---:|---:|---:|"]
    for row in output["best_hidden_comparisons"]:
        def number(value):
            return "—" if value is None else f"{value:.3f}"
        lines.append("| " + " | ".join([row["dataset"], row["regime"], row["comparator"],
            number(row["best_hidden_mm"]), number(row["baseline_hidden_mm"]), number(row["delta_mm"])]) + " |")
    lines.extend(["", "速度、加速度、脚部与腕部时序、各窗口分母及源文件 SHA 见 CPU_TEMPORAL_REVIEW.json。",
        "这是开发验证集的事后审计，未更改 best，不自动认定科学验收通过。", ""])
    (run / "CPU_TEMPORAL_REVIEW_zh.md").write_text("\n".join(lines))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args(argv)
    result = report(args.run)
    print(json.dumps(dict(status=result["status"], best_step=result["best_stage_step"],
                         rows=len(result["rows"]), CUDA_initialized=False), indent=2))


if __name__ == "__main__":
    main()
