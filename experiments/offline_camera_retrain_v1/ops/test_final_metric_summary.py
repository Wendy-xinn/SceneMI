"""Numerical coverage and paired-estimator tests, using hand-computable data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .final_metric_summary import build_summary, report, recheck_ready, _sampler_alive, NUMPY_LIMITS


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

    def evidence_fixture(self, out):
        plan, rows = self.fixture()
        source = out / "model.py"
        source.write_bytes(b"# frozen model source\n")
        source_manifest = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
        plan.update(source_sha256=source_manifest, model_tensor_sha256="f" * 64)
        plan_bytes = json.dumps(plan).encode()
        row_bytes = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
        (out / "PLAN.json").write_bytes(plan_bytes)
        (out / "rows.jsonl").write_bytes(row_bytes)
        binding = dict(PLAN_sha256=hashlib.sha256(plan_bytes).hexdigest(),
            rows_sha256=hashlib.sha256(row_bytes).hexdigest(), selection_sha256="b" * 64,
            checkpoint_sha256="a" * 64, model_tensor_sha256="f" * 64,
            source_sha256=plan["source_sha256"])
        cpu = dict(binding, protocol="native30-saved-metrics-independent-CPU-audit-v1",
            status="saved_metrics_independent_CPU_passed", planned_predictions=8, committed_rows=8,
            recomputed_predictions=8, CUDA_initialized=False, maximum_metric_recompute_difference=0.,
            counts_match=True, artifact_sha256_checked_by_load_rows=True, paired_saved_inputs_exact=True,
            all_pairs_complete=True, complete_ON_OFF_pairs=4, scene_off_guard_zero=True,
            rows_PLAN_selection_unchanged=True, independent_tolerances=NUMPY_LIMITS,
            independent_numpy_max_difference=dict.fromkeys(NUMPY_LIMITS, 0.))
        for name in ("counts_mismatch_rows", "availability_mismatch_rows", "identity_mismatch_rows",
                "nonfinite_rows", "pair_mismatch_rows", "scene_off_guard_mismatch_rows"):
            cpu[name] = []
        gpu = dict(binding, protocol="selected-native30-exact-input-recheck-v1", status="completed_exact",
            complete=True, completed_inputs=4, planned_inputs=4, checked_scene_artifacts=8,
            all_source_arrays_exact=True, all_input_fingerprints_exact=True,
            all_reference_arrays_exact=True, device="cuda", selection_manifest_sha256="b" * 64)
        return cpu, gpu

    def test_missing_or_wrong_snapshot_evidence_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cpu, gpu = self.evidence_fixture(out)
            missing = report(out)["verification_stages"]
            self.assertEqual(missing["CPU_saved_metric_recomputation"]["status"], "not_available")
            self.assertFalse(missing["all_phased_checks_passed"])
            for key in ("PLAN_sha256", "rows_sha256", "selection_manifest_sha256",
                    "checkpoint_sha256", "model_tensor_sha256", "source_sha256"):
                bad = dict(gpu, **{key: {} if key == "source_sha256" else "0" * 64})
                (out / "GPU_INPUTS_RECHECK.json").write_text(json.dumps(bad))
                stage = report(out)["verification_stages"]["original_CUDA_inputs"]
                self.assertEqual(stage["status"], "binding_mismatch", key)
                self.assertFalse(stage["binding_matches"], key)

    def test_exact_CPU_metrics_and_strict_NumPy_failure_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cpu, gpu = self.evidence_fixture(out)
            cpu["status"] = "saved_metrics_independent_CPU_FAILED"
            cpu["independent_numpy_max_difference"]["foot_joint_speed_gt_low_m_s"] = .00027
            (out / "CPU_SAVED_METRICS_AUDIT.json").write_text(json.dumps(cpu))
            (out / "GPU_INPUTS_RECHECK.json").write_text(json.dumps(gpu))
            attempt_bytes = json.dumps(dict(status="failed", PLAN_sha256=cpu["PLAN_sha256"],
                rows_sha256=cpu["rows_sha256"], returncode=1)).encode()
            (out / "OVERALL_CPU_RECHECK.json").write_bytes(attempt_bytes)
            result = report(out)
            stages = result["verification_stages"]
            self.assertEqual(stages["CPU_saved_metric_recomputation"]["status"], "passed")
            self.assertEqual(stages["independent_NumPy_FK"]["status"], "failed_strict")
            self.assertEqual(stages["original_CUDA_inputs"]["status"], "passed")
            self.assertFalse(stages["all_phased_checks_passed"])
            self.assertEqual(result["CPU_recheck"]["status"], "not_completed_for_this_snapshot")
            self.assertEqual((out / "OVERALL_CPU_RECHECK.json").read_bytes(), attempt_bytes)
            self.assertFalse((out / "CPU_RECHECK.json").exists())
            self.assertIn("0.00027", (out / "OVERALL_REPORT_zh.md").read_text())

    def test_GPU_FAILED_or_running_never_passes_even_with_exact_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cpu, gpu = self.evidence_fixture(out)
            (out / "CPU_SAVED_METRICS_AUDIT.json").write_text(json.dumps(cpu))
            for status, expected in (("failed", "failed"), ("running", "pending")):
                (out / "GPU_INPUTS_RECHECK.json").write_text(json.dumps(dict(gpu, status=status)))
                stages = report(out)["verification_stages"]
                self.assertEqual(stages["original_CUDA_inputs"]["status"], expected)
                self.assertFalse(stages["all_phased_checks_passed"])
            (out / "GPU_INPUTS_RECHECK.json").write_text(json.dumps(dict(gpu, completed_inputs=3)))
            self.assertNotEqual(report(out)["verification_stages"]["original_CUDA_inputs"]["status"], "passed")

    def test_separate_audit_closure_requires_parent_sources_and_hashes_every_extra(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cpu, gpu = self.evidence_fixture(out)
            extra = out / "audit_helper.py"
            extra.write_bytes(b"# independently sealed audit helper\n")
            extra_manifest = {str(extra): hashlib.sha256(extra.read_bytes()).hexdigest()}
            gpu.update(parent_source_sha256=cpu["source_sha256"], source_sha256=extra_manifest, status="running")
            path = out / "GPU_INPUTS_RECHECK.json"
            path.write_text(json.dumps(gpu))
            stage = report(out)["verification_stages"]["original_CUDA_inputs"]
            self.assertEqual(stage["status"], "pending")
            self.assertEqual(stage["source_checks"]["additional_audit_sources"], 1)
            gpu["status"] = "completed_exact"
            path.write_text(json.dumps(gpu))
            self.assertEqual(report(out)["verification_stages"]["original_CUDA_inputs"]["status"], "passed")
            extra.write_bytes(b"# changed helper\n")
            self.assertEqual(report(out)["verification_stages"]["original_CUDA_inputs"]["status"], "binding_mismatch")
            extra.write_bytes(b"# independently sealed audit helper\n")
            (out / "model.py").write_bytes(b"# changed original source\n")
            self.assertEqual(report(out)["verification_stages"]["original_CUDA_inputs"]["status"], "binding_mismatch")

    def test_extra_sources_cannot_substitute_for_missing_parent_or_conflict_with_it(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cpu, gpu = self.evidence_fixture(out)
            extra = out / "audit_helper.py"
            extra.write_bytes(b"# helper\n")
            extras = {str(extra): hashlib.sha256(extra.read_bytes()).hexdigest()}
            path = out / "GPU_INPUTS_RECHECK.json"
            for bad in (dict(gpu, source_sha256=extras),
                    dict(gpu, parent_source_sha256={}, source_sha256=extras),
                    dict(gpu, parent_source_sha256=cpu["source_sha256"],
                        source_sha256=dict(extras, **{str(out / "model.py"): "0" * 64}))):
                path.write_text(json.dumps(bad))
                self.assertEqual(report(out)["verification_stages"]["original_CUDA_inputs"]["status"], "binding_mismatch")

    def test_changed_tolerance_is_not_accepted_as_independent_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cpu, gpu = self.evidence_fixture(out)
            cpu["independent_tolerances"] = dict(NUMPY_LIMITS, foot_joint_speed_gt_low_m_s=1.)
            (out / "CPU_SAVED_METRICS_AUDIT.json").write_text(json.dumps(cpu))
            (out / "GPU_INPUTS_RECHECK.json").write_text(json.dumps(gpu))
            stages = report(out)["verification_stages"]
            self.assertEqual(stages["CPU_saved_metric_recomputation"]["status"], "passed")
            self.assertEqual(stages["independent_NumPy_FK"]["status"], "invalid_or_incomplete_evidence")
            self.assertFalse(stages["all_phased_checks_passed"])

    def test_bound_foot_diagnostic_explains_but_does_not_waive_strict_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cpu, gpu = self.evidence_fixture(out)
            cpu["status"] = "saved_metrics_independent_CPU_FAILED"
            cpu["independent_numpy_max_difference"]["foot_joint_speed_gt_low_m_s"] = .00027
            raw = json.dumps(cpu).encode()
            (out / "CPU_SAVED_METRICS_AUDIT.json").write_bytes(raw)
            foot = {key: cpu[key] for key in ("PLAN_sha256", "rows_sha256", "selection_sha256",
                "checkpoint_sha256", "source_sha256")}
            foot.update(protocol="native30-saved-foot-threshold-boundary-CPU-audit-v1",
                status="boundary_diagnosis_not_a_tolerance_waiver", computed_predictions=8,
                CUDA_initialized=False, previous_saved_metric_audit_sha256=hashlib.sha256(raw).hexdigest(),
                original_tolerance_m_s=.0001, tolerance_changed=False, original_full_audits_not_overridden=True,
                over_tolerance_rows=2, over_tolerance_unique_cases=1, GT_speed_threshold_m_s=.2,
                summary={"NumPy_on_original_mask_max_foot_difference_m_s": .000003})
            path = out / "CPU_FOOT_BOUNDARY_AUDIT.json"
            path.write_text(json.dumps(foot))
            stages = report(out)["verification_stages"]
            self.assertEqual(stages["foot_threshold_boundary_diagnostic"]["status"], "documented_boundary_exception")
            self.assertEqual(stages["independent_NumPy_FK"]["status"], "failed_strict")
            self.assertFalse(stages["all_phased_checks_passed"])
            foot["previous_saved_metric_audit_sha256"] = "0" * 64
            path.write_text(json.dumps(foot))
            self.assertNotEqual(report(out)["verification_stages"]["foot_threshold_boundary_diagnostic"]["status"],
                "documented_boundary_exception")


if __name__ == "__main__":
    unittest.main()
