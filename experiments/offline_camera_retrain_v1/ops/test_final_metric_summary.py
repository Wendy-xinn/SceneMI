"""Numerical coverage and paired-estimator tests, using hand-computable data."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .final_metric_summary import build_summary, report, recheck_ready, _sampler_alive


class OverallTests(unittest.TestCase):
    def fixture(self):
        datasets = ["rich", "trumans", "trumans", "egobody"]
        cases = [dict(case=i, mode="exo", dataset=dataset, recording=str(i), sequence_id=str(i),
            start=0, source_first=0, case_seed=100 + i) for i, dataset in enumerate(datasets)]
        plan = dict(state="frozen", split="full-validation", cases=cases, regimes=["noisy"],
            scene_variants=["scene_on", "scene_off"], planned_predictions=8,
            selection=dict(checkpoint_sha256="a" * 64, expected_step=10000), selection_manifest_sha256="b" * 64)
        rows = []
        for entry, value in zip(cases, [10., 20., 20., 40.]):
            for scene, offset in [("scene_on", 0.), ("scene_off", 2.)]:
                rows.append(dict(entry, seed=entry["case_seed"], regime="noisy", scene_variant=scene,
                    input_sha256="c" * 64, checkpoint_sha256="a" * 64, selection_sha256="b" * 64,
                    metrics={"w_mpjpe_mm": value + offset}, counts={"w_mpjpe_mm": 1}))
        return plan, rows

    def test_window_and_equal_dataset_means_are_different_estimators(self):
        plan, rows = self.fixture()
        result = build_summary(plan, rows)
        metric = result["groups"]["exo/noisy/scene_on"]["metrics"]["w_mpjpe_mm"]
        self.assertEqual(metric["window_weighted_mean"], 22.5)
        self.assertAlmostEqual(metric["dataset_macro_mean"], 70. / 3.)
        pair = result["paired_scene_comparison"]["exo/noisy"]["delta_scene_on_minus_off"]
        self.assertEqual(pair["metrics"]["w_mpjpe_mm"]["window_weighted_mean"], -2.)
        self.assertEqual(pair["metrics"]["w_mpjpe_mm"]["on_less_than_off_windows"], 4)

    def test_missing_dataset_macro_is_null_and_partial_is_explicit(self):
        plan, rows = self.fixture()
        result = build_summary(plan, [row for row in rows if row["dataset"] != "egobody"])
        group = result["groups"]["exo/noisy/scene_on"]
        self.assertEqual(group["status"], "partial")
        self.assertEqual(group["coverage"]["egobody"], dict(completed_windows=0, planned_windows=1))
        self.assertIsNone(group["metrics"]["w_mpjpe_mm"]["dataset_macro_mean"])
        self.assertEqual(group["missing_datasets"], ["egobody"])

    def test_pairing_excludes_missing_and_different_inputs_without_changing_raw_groups(self):
        plan, rows = self.fixture()
        rows = [row for row in rows if not (row["case"] == 0 and row["scene_variant"] == "scene_off")]
        next(row for row in rows if row["case"] == 1 and row["scene_variant"] == "scene_off")["input_sha256"] = "d" * 64
        result = build_summary(plan, rows)
        pair = result["paired_scene_comparison"]["exo/noisy"]
        self.assertEqual(pair["unpaired_windows"], 1)
        self.assertEqual(pair["mismatched_input_windows"], 1)
        self.assertEqual(pair["delta_scene_on_minus_off"]["completed_windows"], 2)
        self.assertEqual(result["groups"]["exo/noisy/scene_on"]["completed_windows"], 4)

    def test_incomplete_coverage_stays_partial_when_every_dataset_is_present(self):
        plan, rows = self.fixture()
        result = build_summary(plan, [row for row in rows if row["case"] != 2])
        group = result["groups"]["exo/noisy/scene_on"]
        self.assertEqual(group["missing_datasets"], [])
        self.assertEqual(group["status"], "partial")
        self.assertAlmostEqual(group["metrics"]["w_mpjpe_mm"]["dataset_macro_mean"], 70. / 3.)

    def test_modes_remain_separate_and_optional_merge_is_count_weighted(self):
        plan, rows = self.fixture()
        for row in plan["cases"]:
            row["mode"] = "ego" if row["case"] == 0 else "exo"
        for row in rows:
            row["mode"] = "ego" if row["case"] == 0 else "exo"
        separate = build_summary(plan, rows)
        self.assertEqual(separate["groups"]["ego/noisy/scene_on"]["metrics"]["w_mpjpe_mm"]["window_weighted_mean"], 10.)
        self.assertNotIn("combined_modes/noisy/scene_on", separate["groups"])
        merged = build_summary(plan, rows, combined_modes=True)
        self.assertEqual(merged["groups"]["combined_modes/noisy/scene_on"]["metrics"]["w_mpjpe_mm"]["window_weighted_mean"], 22.5)

    def test_wrong_model_duplicate_or_nonfinite_rows_are_rejected(self):
        plan, rows = self.fixture()
        wrong_model, nonfinite = deepcopy(rows), deepcopy(rows)
        wrong_model[0]["checkpoint_sha256"] = "f" * 64
        nonfinite[0]["metrics"]["w_mpjpe_mm"] = float("nan")
        for bad in (rows + [rows[0]], wrong_model, nonfinite):
            with self.assertRaises(ValueError):
                build_summary(plan, bad)

    def test_snapshot_never_repairs_active_rows_and_stale_CPU_pass_does_not_verify_it(self):
        plan, rows = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "PLAN.json").write_text(json.dumps(plan))
            raw = (json.dumps(rows[0]) + "\n" + '{"case":').encode()
            (out / "rows.jsonl").write_bytes(raw)
            (out / "CPU_RECHECK.json").write_text(json.dumps(dict(status="independent_CPU_passed",
                complete=True, rows_sha256="0" * 64)))
            result = report(out)
            self.assertEqual(result["completed_predictions"], 1)
            self.assertEqual(result["ignored_uncommitted_tail_bytes"], 8)
            self.assertEqual((out / "rows.jsonl").read_bytes(), raw)
            self.assertEqual(result["CPU_recheck"]["status"], "not_completed_for_this_snapshot")
            self.assertTrue((out / "OVERALL_REPORT_zh.md").is_file())

    def test_CPU_recheck_waits_for_complete_durable_rows_final_status_and_sampler_exit(self):
        result = dict(planned_predictions=8, completed_predictions=8, ignored_uncommitted_tail_bytes=0,
            CPU_recheck=dict(status="not_completed_for_this_snapshot"))
        state = dict(status="sampling_completed_pending_CPU", completed_predictions=8)
        self.assertTrue(recheck_ready(result, state, sampler_alive=False))
        self.assertFalse(recheck_ready(result, state, sampler_alive=True))
        for partial in (dict(result, completed_predictions=7),
                dict(result, ignored_uncommitted_tail_bytes=1),
                dict(result, CPU_recheck=dict(status="completed_verified_for_this_snapshot"))):
            self.assertFalse(recheck_ready(partial, state, sampler_alive=False))
        for running in (dict(state, status="sampling"), dict(state, completed_predictions=7)):
            self.assertFalse(recheck_ready(result, running, sampler_alive=False))
        with patch.object(Path, "read_text", return_value="123 (python worker) Z " + "0 " * 18 + "555"):
            self.assertFalse(_sampler_alive(123, expected_start_ticks=555))
        with patch.object(Path, "read_text", return_value="123 (python worker) R " + "0 " * 18 + "556"):
            self.assertFalse(_sampler_alive(123, expected_start_ticks=555))
            self.assertTrue(_sampler_alive(123, expected_start_ticks=556))


if __name__ == "__main__":
    unittest.main()
