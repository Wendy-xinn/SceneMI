"""CPU-only sidecar audits for a sealed frozen reference-policy review.

The runner and its immutable plan are never changed. This audit checks the
entire recursively collated non-reference input, durable result pairing, and
reproducibility against any previously committed original-policy outputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import reference_review as review
from . import reference_temporal


ALLOWED_REFERENCE_KEYS = frozenset(("reference_motion", "observation_anchor_reference"))
DEFAULT_OUT = review.DEFAULT_OUT


def compare_inputs(original, repaired):
    """Return changed tensor paths; reject every other recursive difference."""
    import numpy as np
    import torch

    changed, opaque_shared = [], []

    def compare(a, b, path=""):
        if isinstance(a, torch.Tensor):
            review.require(isinstance(b, torch.Tensor), "Tensor type differs: " + path)
            equal = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
            if not equal:
                review.require(path.lstrip(".") in ALLOWED_REFERENCE_KEYS,
                               "Non-reference tensor changed: " + path)
                changed.append(path)
            return
        if isinstance(a, np.ndarray):
            review.require(isinstance(b, np.ndarray) and a.dtype == b.dtype
                           and np.array_equal(a, b), "Array changed: " + path)
            return
        if isinstance(a, dict):
            review.require(isinstance(b, dict) and a.keys() == b.keys(), "Keys differ: " + path)
            for key in sorted(a):
                compare(a[key], b[key], path + "." + str(key))
            return
        if isinstance(a, (list, tuple)):
            review.require(type(a) is type(b) and len(a) == len(b), "Container differs: " + path)
            for i, (first, second) in enumerate(zip(a, b)):
                compare(first, second, path + f"[{i}]")
            return
        if a is None or isinstance(a, (str, int, float, bool)):
            review.require(type(a) is type(b) and a == b, "Scalar differs: " + path)
            return
        review.require(a is b, "Opaque input object differs: " + path)
        opaque_shared.append(path)

    compare(original, repaired)
    return dict(changed_tensor_paths=changed, shared_opaque_objects=opaque_shared)


def audit_inputs(out):
    import torch

    review.common.cpu_setup()
    plan = review.read(out / "PLAN.json")
    review.verify(plan)
    data = review.common.validation_data(plan["config"])
    builder = review.common.observation_builder(plan["config"])
    cases = []
    for entry in plan["entries"]:
        for regime in plan["regimes"]:
            _, observed, scene, asset = review.evaluation.source_observation(data, builder, entry, regime)
            _, original = review.evaluation.collated_inputs(plan["config"], observed, scene, asset, "cpu")
            before = review.common.tensor_fingerprint(original)
            with reference_temporal.installed():
                _, repaired = review.evaluation.collated_inputs(plan["config"], observed, scene, asset, "cpu")
            compared = compare_inputs(original, repaired)
            review.require(before == review.common.tensor_fingerprint(original),
                           "Building repaired inputs mutated original input")
            confidence_exact = torch.equal(original["reference_confidence"], repaired["reference_confidence"])
            review.require(confidence_exact, "Reference confidence changed")
            nonreference = {key: value for key, value in original.items() if key not in ALLOWED_REFERENCE_KEYS}
            cases.append(dict(case=entry["case"], dataset=entry["dataset"], regime=regime,
                review_panel=entry["review_panel"], nonreference_tensor_sha256=review.common.tensor_fingerprint(nonreference),
                original_CPU_tensor_sha256=before, repaired_CPU_tensor_sha256=review.common.tensor_fingerprint(repaired),
                reference_confidence_exact=True, **compared))
    review.require(not torch.cuda.is_initialized(), "CPU sidecar initialized CUDA")
    review.verify(plan)
    report = dict(status="passed", checked_at=review.common.now(), CPU_threads=torch.get_num_threads(),
        CUDA_initialized=False, cases=len(cases), plan_sha256=review.sha256(out / "PLAN.json"),
        audit_source_sha256=review.sha256(__file__), allowed_changed_keys=sorted(ALLOWED_REFERENCE_KEYS),
        all_other_recursive_inputs_exact=True, reference_confidence_exact=True, entries=cases,
        note="CPU collation verifies that only versioned reference tensors change. CPU full-input hashes are not compared directly with CUDA hashes because original reference SVD/FK can differ by floating-point device roundoff.")
    review.write(out / "INPUT_POLICY_AUDIT.json", report)
    return report


def audit_results(out):
    import numpy as np
    import torch

    review.common.cpu_setup()
    plan = review.read(out / "PLAN.json")
    review.verify(plan)
    rows = review.rows_for(out, plan)
    indexed = {review.key(row): row for row in rows}
    old_path = review.SOURCE / "rows.jsonl"
    old_rows = [json.loads(line) for line in old_path.read_text().splitlines() if line]
    old_index = {(r["case"], r["regime"], r["model"]): r for r in old_rows}
    pair_evidence, reproducibility = [], []
    for entry in plan["entries"]:
        for regime in plan["regimes"]:
            group = [indexed.get((entry["case"], regime, policy, model))
                     for policy in review.POLICIES for model in review.MODELS]
            present = [row for row in group if row is not None]
            arrays = []
            for row in present:
                with np.load(row["artifact"], allow_pickle=False) as saved:
                    arrays.append({key: saved[key].copy() for key in saved.files})
                current = arrays[-1]
                review.require(str(current["checkpoint_sha256"]) == row["checkpoint_sha256"]
                               and int(current["checkpoint_step"]) == row["checkpoint_step"]
                               and str(current["reference_policy"]) == row["policy"],
                               "Saved archive provenance differs from committed row")
            if arrays:
                for current in arrays[1:]:
                    review.require(all(np.array_equal(current[key], arrays[0][key])
                                   for key in ("truth", "rest", "observations", "observation_meta", "identity_json")),
                                   "Cross-policy source observation identity differs")
            for policy in review.POLICIES:
                pair = [r for r in present if r["policy"] == policy]
                if len(pair) == 2:
                    review.require(pair[0]["seed"] == pair[1]["seed"]
                                   and pair[0]["input_sha256"] == pair[1]["input_sha256"]
                                   and pair[0]["counts"] == pair[1]["counts"], "Model pair differs")
                    selected = [a for r, a in zip(present, arrays) if r["policy"] == policy]
                    review.require(np.array_equal(selected[0]["reference_motion"], selected[1]["reference_motion"]),
                                   "Saved model-pair references differ")
            pair_evidence.append(dict(case=entry["case"], regime=regime, predictions=len(present),
                source_arrays_exact=True if arrays else None, complete=len(present) == 4))
            for row, current in zip(present, arrays):
                if row["policy"] != "original":
                    continue
                previous = old_index.get((row["case"], regime, row["model"]))
                if previous is None:
                    continue
                review.require(review.sha256(previous["artifact"]) == previous["artifact_sha256"],
                               "Historical artifact changed")
                review.require(row["seed"] == previous["seed"]
                               and row["checkpoint_sha256"] == previous["checkpoint_sha256"],
                               "Historical seed/checkpoint differs")
                with np.load(previous["artifact"], allow_pickle=False) as saved:
                    for key in ("truth", "rest", "observations", "observation_meta"):
                        review.require(np.array_equal(current[key], saved[key]), "Historical source array differs")
                    difference = np.abs(current["prediction"].astype(np.float64) - saved["prediction"])
                    exact = bool(np.array_equal(current["prediction"], saved["prediction"]))
                reproducibility.append(dict(case=row["case"], regime=regime, model=row["model"],
                    recursive_input_sha256_exact=row["input_sha256"] == previous["input_sha256"],
                    source_arrays_exact=True, prediction_array_exact=exact,
                    prediction_maximum_absolute_feature_difference=float(difference.max()),
                    prediction_mean_absolute_feature_difference=float(difference.mean()),
                    current_artifact_sha256=row["artifact_sha256"], previous_artifact_sha256=previous["artifact_sha256"]))
    review.require(not torch.cuda.is_initialized(), "CPU sidecar initialized CUDA")
    report = dict(status="complete" if len(rows) == plan["planned_predictions"] else "partial",
        checked_at=review.common.now(), completed_predictions=len(rows), planned_predictions=plan["planned_predictions"],
        CPU_threads=torch.get_num_threads(), CUDA_initialized=False, rows_sha256=review.sha256(out / "rows.jsonl"),
        historical_rows_sha256=review.sha256(old_path), plan_sha256=review.sha256(out / "PLAN.json"),
        audit_source_sha256=review.sha256(__file__), source_pairs=pair_evidence,
        original_policy_reproducibility=reproducibility,
        mutation_check=dict(status="static_source_review_only", runtime_GPU_input_hash_check=False,
            inspected_functions=["native30_refinement.sample_refinement", "Native30Refinement._core_pass",
                "Native30Refinement.observation_condition", "Native30SceneMI.forward",
                "Native30SceneMI.observation_condition", "Native30SceneMI.local_geometry",
                "native30_model.reconcile_motion", "native30_observation_ik.position_proximal"],
            finding="No direct in-place write to supplied known/batch tensors found in reviewed call paths. Allocated/cloned local tensors and model-scoped sampling caches are mutated. This is static evidence; opaque scene query caches and dependency internals were not dynamically instrumented."))
    review.write(out / "PAIRED_RESULT_AUDIT.json", report)
    return report


def plot_regression(out, case=62, regime="clean"):
    """Plot reference/model wrist rotation steps with unavailable-elbow spans."""
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

    review.common.cpu_setup()
    plan = review.read(out / "PLAN.json")
    review.verify(plan)
    rows = review.rows_for(out, plan)
    indexed = {review.key(row): row for row in rows}
    selected = [indexed.get((case, regime, policy, model))
                for policy in review.POLICIES for model in review.MODELS]
    review.require(all(row is not None for row in selected), "Regression quartet incomplete")
    saved = {}
    for row in selected:
        with np.load(row["artifact"], allow_pickle=False) as archive:
            saved[row["policy"], row["model"]] = {key: archive[key].copy() for key in archive.files}
    first = saved["original", "base41k"]
    for arrays in saved.values():
        review.require(all(np.array_equal(arrays[key], first[key])
                           for key in ("truth", "rest", "observations", "observation_meta")),
                       "Plot sources do not share exact observations and GT")

    def step_degrees(motion):
        tensor = torch.as_tensor(np.array(motion, dtype=np.float32, copy=True))[None]
        with torch.no_grad():
            rotations = global_rotations(tensor)[0, :, 20].double().numpy()
        relative = np.swapaxes(rotations[:-1], -1, -2) @ rotations[1:]
        cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2
        return np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))

    fps = float(plan["fps"])
    gt = step_degrees(first["truth"])
    clock = np.arange(1, len(gt) + 1) / fps
    missing = ~((first["observation_meta"][:, 18, 0] > 0)
                & (first["observation_meta"][:, 18, 2] > 0))
    boundaries = np.diff(np.r_[False, missing, False].astype(np.int8))
    spans = list(zip(np.flatnonzero(boundaries == 1) / fps,
                     np.flatnonzero(boundaries == -1) / fps))
    definitions = [("Reference initializer", "reference_motion", "base41k"),
                   ("Base 41k", "prediction", "base41k"),
                   ("EXO adapter v2", "prediction", "balancedv2adapter1000")]
    figure, axes = plt.subplots(3, 1, figsize=(12, 8.8), sharex=True,
                                gridspec_kw={"hspace": .17})
    records = []
    for ax, (title, key, model) in zip(axes, definitions):
        old = step_degrees(saved["original", model][key])
        repaired = step_degrees(saved[reference_temporal.POLICY, model][key])
        for start, end in spans:
            ax.axvspan(start, end, color="#df7381", alpha=.13, linewidth=0, zorder=0)
        ax.plot(clock, gt, color="#707780", linewidth=1.2, alpha=.9, label="GT")
        ax.plot(clock, old, color="#e58b26", linewidth=1.3, label="Original reference policy")
        ax.plot(clock, repaired, color="#079b98", linewidth=1.5, label="Offline gap-fill policy")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=4)
        ax.set_ylabel("Left wrist step\n(deg / frame)", fontsize=10)
        ax.set_ylim(0, max(20, np.ceil(max(old.max(), repaired.max(), gt.max()) / 20) * 20))
        ax.grid(axis="y", color="#dadde0", linewidth=.6, alpha=.8)
        ax.spines[["top", "right"]].set_visible(False)
        records.append(dict(layer=title, original_step_p95_deg=float(np.percentile(old, 95)),
            repaired_step_p95_deg=float(np.percentile(repaired, 95)),
            GT_step_p95_deg=float(np.percentile(gt, 95)),
            original_step_over30_count=int((old > 30).sum()),
            repaired_step_over30_count=int((repaired > 30).sum())))
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#df7381", alpha=.2))
    labels.append("Left elbow 3D position unavailable")
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.51, .955),
                  ncol=2, frameon=False, fontsize=9)
    axes[-1].set_xlabel("Time (s)", fontsize=11)
    axes[-1].set_xlim(0, len(first["truth"]) / fps)
    axes[-1].set_xticks([0, 1, 2, 3, 4, 5, 6, 6.4])
    figure.suptitle(f"EXO wrist jitter regression | Case {case} | {regime}", y=.988,
                    fontsize=13, fontweight="bold")
    figure.subplots_adjust(top=.875, bottom=.115, left=.095, right=.98)
    figure.text(.095, .025,
        "Offline gap fill; frozen weights; identical EXO observations and sampling seed.\n"
        "Shading marks unavailable elbow positions, not exclusively physical occlusion. "
        "Rotation diagnostic only; see separate position-error results.", fontsize=9, color="#434952")
    folder = out / "presentation"
    folder.mkdir(exist_ok=True)
    stem = folder / f"case{case:04d}_{regime}_wrist_rotation_diagnostic"
    figure.savefig(stem.with_suffix(".png"), dpi=180, facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(figure)
    report = dict(status="saved", checked_at=review.common.now(), case=case, regime=regime,
        sequence_id=selected[0]["sequence_id"], left_elbow_joint=18, left_wrist_joint=20,
        missing_elbow_frames=int(missing.sum()), frames=len(missing), fps=fps,
        rotation_definition="Geodesic angle of consecutive global left-wrist joint rotation matrices; no position alignment or smoothing.",
        missing_definition="Position availability channel <=0 or confidence channel <=0; includes more than physical occlusion.",
        diagnostic_only=True, source_sha256={row["artifact"]: row["artifact_sha256"] for row in selected},
        plot_sha256={str(stem.with_suffix(suffix)): review.sha256(stem.with_suffix(suffix))
                     for suffix in (".png", ".svg")}, metrics=records)
    review.require(not torch.cuda.is_initialized(), "Plot initialized CUDA")
    review.write(stem.with_suffix(".json"), report)
    return report


def summarize_complete(out):
    """Compare the complete primary panel and deduplicated input references."""
    import collections
    import statistics
    import numpy as np
    import torch

    review.common.cpu_setup()
    plan = review.read(out / "PLAN.json")
    review.verify(plan)
    rows = review.rows_for(out, plan)
    review.require(len(rows) == plan["planned_predictions"], "Complete frozen panel required")
    source_rows = []
    for row in rows:
        item = {key: row[key] for key in ("case", "dataset", "regime", "policy", "model", "review_panel")}
        source_rows.append(dict(item, metrics=row["metrics"], counts=row["counts"]))
        if row["model"] != "base41k":
            continue
        with np.load(row["artifact"], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays["prediction"] = arrays["reference_motion"]
        reference_metrics = review.common.independent_core(arrays)
        reference_metrics.update(review.temporal_metrics(arrays))
        source_rows.append(dict(item, model="input_reference", metrics=reference_metrics, counts=row["counts"]))

    groups = []
    for panel in ("primary", "diagnostic_regression"):
        for dataset in ("trumans", "egobody", "rich", "macro"):
            for regime in plan["regimes"]:
                for policy in review.POLICIES:
                    for model in ("input_reference", *review.MODELS):
                        selected = [r for r in source_rows if (r["review_panel"], r["regime"], r["policy"], r["model"])
                                    == (panel, regime, policy, model)
                                    and (dataset == "macro" or r["dataset"] == dataset)]
                        if not selected:
                            continue
                        by_dataset = collections.defaultdict(list)
                        for row in selected:
                            by_dataset[row["dataset"]].append(row)
                        if panel == "primary" and dataset == "macro":
                            review.require(set(by_dataset) == {"trumans", "egobody", "rich"},
                                           "Balanced three-dataset macro requires all datasets")
                        values = {}
                        for metric in sorted(set.intersection(*(set(r["metrics"]) for r in selected))):
                            valid = [r for r in selected if r["metrics"][metric] is not None]
                            if not valid:
                                continue
                            dataset_means, dataset_pooled = [], []
                            for part in by_dataset.values():
                                part = [r for r in part if r["metrics"][metric] is not None]
                                if not part:
                                    continue
                                dataset_means.append(statistics.mean(r["metrics"][metric] for r in part))
                                count = sum(r["counts"].get(metric, 0) for r in part)
                                if count:
                                    dataset_pooled.append(sum(r["metrics"][metric] * r["counts"].get(metric, 0)
                                                              for r in part) / count)
                            total = sum(r["counts"].get(metric, 0) for r in valid)
                            values[metric] = dict(equal_recording_then_equal_dataset=statistics.mean(dataset_means),
                                within_dataset_count_weighted_then_equal_dataset=statistics.mean(dataset_pooled) if dataset_pooled else None,
                                all_selected_samples_pooled=sum(r["metrics"][metric] * r["counts"].get(metric, 0)
                                                               for r in valid) / total if total else None,
                                sample_count=total or None)
                        fraction = [r["counts"]["conditioned_mpjpe_mm"] / r["counts"]["mpjpe_mm"] for r in selected]
                        groups.append(dict(panel=panel, dataset=dataset, regime=regime, policy=policy,
                            model=model, windows=len(selected), datasets=sorted(by_dataset), metrics=values,
                            position_observation_available_fraction_recording_mean=statistics.mean(fraction)))

    summary = dict(status="complete_saved_arrays_summary", checked_at=review.common.now(),
        predictions=len(rows), primary_windows=plan["primary_windows"], diagnostic_windows=plan["diagnostic_windows"],
        primary_datasets={d: sum(e["review_panel"] == "primary" and e["dataset"] == d for e in plan["entries"])
                          for d in ("trumans", "egobody", "rich")},
        source_rows_sha256=review.sha256(out / "rows.jsonl"), source_plan_sha256=review.sha256(out / "PLAN.json"),
        audit_source_sha256=review.sha256(__file__), groups=groups,
        reference_deduplication="One input reference per case/regime/policy, taken from base41k archive. Adapter receives the identical reference; it is not counted twice.",
        aggregation="One window per recording. Headline is equal recording within dataset then equal dataset macro. Joint-sample pooled values are separate; different availability denominators can change the result.",
        scope="Primary 12 new validation recordings and diagnostic 1 known jitter recording are separate. No TEST or optimizer updates. Historical RICH 20 is a distinct panel, not merged with this 12-window panel.",
        interpretation="Input-reference to sampled-output changes include the learned generator, in-diffusion observation IK/anchors and scene/physics feedback. They are not a pure network ablation. Angle-step amplitude error does not measure the direction of rotational velocity.",
        CPU_recheck_status=review.read(out / "CPU_RECHECK.json")["status"] if (out / "CPU_RECHECK.json").exists() else "pending")
    review.write(out / "PAIRED_COMPARISON.json", summary)
    lookup = {(g["panel"], g["dataset"], g["regime"], g["policy"], g["model"]): g for g in groups}
    text = ["# 冻结 EXO：41k / 适配 v2 与参考修复配对比较", "",
        "主面板为三个数据集各 4 个不同录制，共 12 个新验证窗口；已知抽搐例子单列。每窗 192 帧、30 Hz，同观测/场景/种子，冻结权重，无训练、无 TEST。",
        "", "下表箭头表示原 41k → EXO v2，所有位置指标单位 mm；等录制均值。macro 先按数据集内等录制平均，再三数据集等权。缺失/不可用不等同于纯物理遮挡。", "",
        "| 数据 | 输入 | 参考策略 | 全身 | 可用位置关节 | 缺失/不可用关节 |",
        "|---|---|---|---:|---:|---:|"]
    statistic = "equal_recording_then_equal_dataset"
    def number(group, metric):
        return group["metrics"][metric][statistic]
    for dataset in ("trumans", "egobody", "rich", "macro"):
        for regime in plan["regimes"]:
            for policy in review.POLICIES:
                a = lookup["primary", dataset, regime, policy, "base41k"]
                b = lookup["primary", dataset, regime, policy, "balancedv2adapter1000"]
                values = [f"{number(a, metric):.2f} → {number(b, metric):.2f}" for metric in
                          ("w_mpjpe_mm", "visible_mpjpe_mm", "hidden_mpjpe_mm")]
                text.append("| " + " | ".join((dataset, regime, "原始" if policy == "original" else "缺失段插值", *values)) + " |")
    text += ["", "## 输入参考 / 41k / 适配后", "",
        "以下为三数据集等权宏平均，每格依次是输入参考 → 原 41k → 适配 v2。输入参考从保存数组读取，每窗口条件只计算一次。", "",
        "| 输入 | 参考策略 | 全身 mm | 可用位置关节 mm | 缺失/不可用关节 mm |",
        "|---|---|---:|---:|---:|"]
    for regime in plan["regimes"]:
        for policy in review.POLICIES:
            gg = [lookup["primary", "macro", regime, policy, model] for model in ("input_reference", *review.MODELS)]
            cells = [" → ".join(f"{number(g, metric):.2f}" for g in gg) for metric in
                     ("w_mpjpe_mm", "visible_mpjpe_mm", "hidden_mpjpe_mm")]
            text.append("| " + " | ".join((regime, "原始" if policy == "original" else "缺失段插值", *cells)) + " |")
    text += ["", "参考到输出之间还包含采样内观测 IK、锚点和场景反馈，不能把全部差值归因于纯网络。原始参考与修复参考的输入分布不同，旧权重不保证位置误差同步改善。",
        "", "## 抽搐回归单例", "",
        "case 62 单列，不计入主面板均值。左腕转角 P95 是旋转诊断，不能替代位置精度。", "",
        "| 输入 | 模型 | 左腕转角 P95 原始→修复 °/帧 | >30° 跳帧数 原始→修复 | 全身 mm 原始→修复 | 缺失关节 mm 原始→修复 |",
        "|---|---|---:|---:|---:|---:|"]
    for regime in plan["regimes"]:
        for model in review.MODELS:
            a = lookup["diagnostic_regression", "egobody", regime, "original", model]
            b = lookup["diagnostic_regression", "egobody", regime, reference_temporal.POLICY, model]
            cells = [f"{number(a, metric):.2f} → {number(b, metric):.2f}" for metric in
                     ("left_wrist_step_p95_deg", "left_wrist_step_over30_count", "w_mpjpe_mm", "hidden_mpjpe_mm")]
            text.append("| " + " | ".join((regime, model, *cells)) + " |")
    text += ["", "独立侧审及 CPU 核验见 INPUT_POLICY_AUDIT.json、PAIRED_RESULT_AUDIT.json 和 CPU_RECHECK.json。",
        "历史 RICH 20 个录制结果是另一份既有验证面板，不能与本面板相加当作独立 TEST。完整等录制/按关节样本加权定义、分母、位置观测密度及参考指标见同名 JSON。", ""]
    (out / "PAIRED_COMPARISON_zh.md").write_text("\n".join(text))
    review.require(not torch.cuda.is_initialized(), "CPU summary initialized CUDA")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inputs", "results", "all", "plot", "summarize"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case", type=int, default=62)
    parser.add_argument("--regime", choices=("clean", "noisy"), default="clean")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.command in ("inputs", "all"):
        result = audit_inputs(args.out)
        print("Input policy audit:", result["status"], result["cases"], flush=True)
    if args.command in ("results", "all"):
        result = audit_results(args.out)
        print("Paired results:", result["status"], result["completed_predictions"],
              "historical overlap", len(result["original_policy_reproducibility"]), flush=True)
    if args.command == "plot":
        result = plot_regression(args.out, case=args.case, regime=args.regime)
        print("Plot saved:", *result["plot_sha256"], flush=True)
    if args.command == "summarize":
        result = summarize_complete(args.out)
        print("Complete paired summary:", result["primary_windows"], "primary windows;",
              result["diagnostic_windows"], "diagnostic", flush=True)


if __name__ == "__main__":
    main()
