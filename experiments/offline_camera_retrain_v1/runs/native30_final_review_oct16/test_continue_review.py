"""CPU checks for conservative stage reuse and recovery provenance."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import continue_review as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.arm = self.root / "arms/control"
        self.arm.mkdir(parents=True)
        self.plan = [dict(sequence_id="seq" + str(i), case_seed=i, dataset="corpus",
                          panel=("fixed" if i < 3 else "GT_locomotion" if i < 6 else "additional_VAL"))
                     for i in range(9)]
        (self.root / "VAL_PLAN.json").write_text(json.dumps(dict(plan=self.plan)))
        metrics = {key: 1. for key in recovery.METRICS}
        self.summary = dict(status="completed", label="phase1", step=3453,
            plan_sha256=recovery.original.sha256(self.root / "VAL_PLAN.json"),
            cases=[dict(case=i, regime=regime, metrics=dict(metrics), **entry)
                   for i, entry in enumerate(self.plan) for regime in ("clean", "noisy")],
            aggregate={panel + "/" + regime: dict(metrics)
                       for panel in ("all", "locomotion") for regime in ("clean", "noisy")},
            timing=dict(mean_compute_s=1., mean_update_s=2.), scene_ablation=False)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        (self.arm / "phase1_summary.json").write_text(json.dumps(self.summary))

    def load(self, **kwargs):
        return recovery.complete_summary(self.root, "control", "phase1", 3453, **kwargs)

    def test_missing_stage_is_not_skipped(self):
        self.assertIsNone(self.load())

    def test_complete_stage_reused(self):
        self.save()
        self.assertEqual(self.load(), self.summary)

    def test_missing_condition_rejected(self):
        self.summary["cases"].pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "Missing or duplicated"):
            self.load()

    def test_duplicate_condition_rejected(self):
        self.summary["cases"][-1] = copy.deepcopy(self.summary["cases"][0])
        self.save()
        with self.assertRaisesRegex(ValueError, "Missing or duplicated"):
            self.load()

    def test_wrong_step_rejected(self):
        self.summary["step"] = 3452
        self.save()
        with self.assertRaisesRegex(ValueError, "wrong stage"):
            self.load()

    def test_changed_plan_rejected(self):
        self.save()
        (self.root / "VAL_PLAN.json").write_text(json.dumps(dict(plan=self.plan, changed=True)))
        with self.assertRaisesRegex(ValueError, "plan differs"):
            self.load()

    def test_changed_identity_rejected(self):
        self.summary["cases"][0]["sequence_id"] = "other"
        self.save()
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.load()

    def test_nonfinite_per_condition_rejected(self):
        self.summary["cases"][0]["metrics"]["mpjpe_mm"] = float("nan")
        self.save()
        with self.assertRaisesRegex(ValueError, "Invalid per-condition"):
            self.load()

    def test_nonfinite_aggregate_rejected(self):
        self.summary["aggregate"]["all/clean"]["mpjpe_mm"] = float("inf")
        self.save()
        with self.assertRaisesRegex(ValueError, "Invalid control"):
            self.load()

    def test_final_stage_requires_scene_off_cases(self):
        self.summary["scene_ablation"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "scene-OFF"):
            self.load(final=True)
        for row in self.summary["cases"]:
            if row["panel"] == "fixed":
                row["scene_off_metrics"] = dict(row["metrics"])
        self.save()
        self.assertIsNotNone(self.load(final=True))

    def test_ineligible_best_score_never_promoted(self):
        choice = dict(eligible=[], scores=dict(control=1., bad=.1))
        self.assertIsNone(recovery.eligible_second_stage(choice, {}))
        choice["eligible"] = ["okay"]
        choice["scores"]["okay"] = .99
        summaries = {"okay": dict(timing=dict(mean_compute_s=2.)),
                     "bad": dict(timing=dict(mean_compute_s=.1))}
        self.assertEqual(recovery.eligible_second_stage(choice, summaries), "okay")

    def test_second_stage_keeps_original_quality_speed_rule(self):
        choice = dict(eligible=["a", "b", "c"], scores=dict(a=.9, b=.92, c=.94))
        summaries = {name: dict(timing=dict(mean_compute_s=seconds))
                     for name, seconds in (("a", 2.), ("b", 1.), ("c", .1))}
        self.assertEqual(recovery.eligible_second_stage(choice, summaries), "b")

    def test_new_recovery_pid_is_guarded(self):
        (self.root / "STATUS.json").write_text(json.dumps(dict(pid=123)))
        proc = self.root / "proc"
        (proc / "123").mkdir(parents=True)
        (proc / "123/cmdline").write_bytes(b"python\0" + str(self.root / "continue_review.py").encode() + b"\0")
        with self.assertRaisesRegex(RuntimeError, "Recovery process still active"):
            recovery.guard_processes(self.root, proc)

    def test_unrelated_reused_pid_allowed(self):
        (self.root / "STATUS.json").write_text(json.dumps(dict(worker_pid=123)))
        proc = self.root / "proc"
        (proc / "123").mkdir(parents=True)
        (proc / "123/cmdline").write_bytes(b"unrelated\0")
        recovery.guard_processes(self.root, proc)

    def test_original_sealed_hash_change_rejected(self):
        with patch.object(recovery.original, "extensions", return_value={"code": "changed"}):
            with self.assertRaisesRegex(ValueError, "sealed"):
                recovery.verify_original_extensions(dict(extension_code_sha256={"code": "old"}))

    def test_report_discloses_parent_was_advanced(self):
        def report(*unused):
            (self.root / "REPORT_zh.md").write_text("原基线可恢复，原源码与 checkpoint 未覆盖。")
            (self.root / "FINAL_SELECTION.json").write_text("{}")
        protocol = dict(current_parent=dict(step=5200, sha256="current"))
        with patch.object(recovery, "ROOT", self.root), patch.object(recovery.original, "report", side_effect=report):
            recovery.write_recovered_report({}, {}, dict(parent_step=2453), protocol)
        text = (self.root / "REPORT_zh.md").read_text()
        self.assertNotIn("checkpoint 未覆盖", text)
        self.assertIn("5200", text)
        self.assertIn("2453", text)
        self.assertIn("原定 8 小时流程未完成", text)
        record = json.loads((self.root / "FINAL_SELECTION.json").read_text())
        self.assertTrue(record["original_parent_overwritten_by_prior_fallback"])
        self.assertFalse(record["original_decision_budget_succeeded"])

    def test_preflight_failure_records_failure_without_launching_parent(self):
        (self.root / "FORK_MANIFEST.json").write_text(json.dumps(dict(phase1_updates=1000, phase2_updates=2000)))
        args = SimpleNamespace(phase1_updates=1000, phase2_updates=2000)
        with patch.object(recovery, "ROOT", self.root), patch.object(recovery, "guard_processes"), \
                patch.object(recovery, "build_protocol", side_effect=ValueError("audit rejected")), \
                patch.object(recovery.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "audit rejected"):
                recovery._main(args)
        launch.assert_not_called()
        failure = json.loads((self.root / "RECOVERY_FAILURE.json").read_text())
        self.assertFalse(failure["fallback_on_failure"])
        self.assertFalse(failure["parent_training_started"])
        self.assertEqual(json.loads((self.root / "STATUS.json").read_text()), failure)

    def make_lineage_fixture(self):
        parent_dir = self.root / "parent"
        (parent_dir / "training").mkdir(parents=True)
        initialization = dict(type="from_scratch")
        parent = dict(step=5200, planned_steps=55000, model={}, optimizer={},
                      initialization=initialization,
                      effective_config=dict(code_sha256={"source": "frozen"}, data_sha256={"data": "same"}))
        recovery.torch.save(parent, parent_dir / "training/last.pt")
        manifest = dict(parent_step=2453, parent_sha256="original2453hash",
                        original_initialization=initialization, retained_fields=["model", "optimizer"],
                        extension_code_sha256={"old": "hash"})
        for name in recovery.original.NAMES:
            arm = self.root / "arms" / name
            (arm / "prepared").mkdir(parents=True, exist_ok=True)
            (arm / "training").mkdir(exist_ok=True)
            effective = dict(extension_code_sha256=manifest["extension_code_sha256"],
                code_sha256=parent["effective_config"]["code_sha256"],
                data_sha256=parent["effective_config"]["data_sha256"],
                stage_lineage=dict(parent_checkpoint_sha256="original2453hash", parent_step=2453,
                                   arm=name, optimizer_reset=False))
            (arm / "prepared/effective_config.json").write_text(json.dumps(effective))
            state = dict(step=2453 if name == "ik_off" else 3453, planned_steps=55000,
                optimizer={}, effective_config=effective,
                initialization=dict(parent_checkpoint_sha256="original2453hash", absolute_step_start=2453))
            recovery.torch.save(state, arm / "training/last.pt")
        return manifest, parent_dir

    def test_advanced_parent_kept_separate_from_original_fork(self):
        manifest, parent = self.make_lineage_fixture()
        args = SimpleNamespace(expected_parent_step=5200, phase1_updates=1000,
                               phase2_updates=2000, decision_hours=8., final_stop=10000)
        with patch.object(recovery, "ROOT", self.root), patch.object(recovery.original, "PARENT", parent), \
                patch.object(recovery.original, "extensions", return_value={"old": "hash"}), \
                patch.object(recovery.original, "code_hashes", return_value={"source": "frozen"}), \
                patch.object(recovery, "recovery_extensions", return_value={"new": "hash"}):
            protocol = recovery.build_protocol(manifest, args)
        self.assertEqual(protocol["current_parent"]["step"], 5200)
        self.assertNotEqual(protocol["current_parent"]["sha256"], manifest["parent_sha256"])
        self.assertEqual(protocol["original_fork"]["parent_step"], 2453)
        self.assertEqual(protocol["candidate_states_at_recovery"]["ik_off"]["step"], 2453)
        self.assertFalse(protocol["original_decision_budget_succeeded"])
        self.assertFalse(protocol["fallback_on_failure"])

    def test_candidate_wrong_original_parent_lineage_rejected(self):
        manifest, parent = self.make_lineage_fixture()
        path = self.root / "arms/ik_off/prepared/effective_config.json"
        effective = json.loads(path.read_text())
        effective["stage_lineage"]["parent_checkpoint_sha256"] = "different"
        path.write_text(json.dumps(effective))
        args = SimpleNamespace(expected_parent_step=5200)
        with patch.object(recovery, "ROOT", self.root), patch.object(recovery.original, "PARENT", parent), \
                patch.object(recovery.original, "extensions", return_value={"old": "hash"}), \
                patch.object(recovery.original, "code_hashes", return_value={"source": "frozen"}):
            with self.assertRaisesRegex(ValueError, "fork lineage changed"):
                recovery.build_protocol(manifest, args)

    def fake_stage(self, name, label, stop, value=1., final=False):
        arm = self.root / "arms" / name
        arm.mkdir(parents=True, exist_ok=True)
        summary = copy.deepcopy(self.summary)
        summary.update(label=label, step=stop, scene_ablation=final)
        summary["timing"]["mean_compute_s"] = 1. if name == "lean" else 2.
        for metrics in summary["aggregate"].values():
            metrics.update({key: value for key in metrics})
        for row in summary["cases"]:
            row["metrics"].update({key: value for key in row["metrics"]})
            if final and row["panel"] == "fixed":
                row["scene_off_metrics"] = dict(row["metrics"])
        (arm / (label + "_summary.json")).write_text(json.dumps(summary))
        (arm / "config.json").write_text(json.dumps(dict(arm=name)))
        return summary

    def manager_fixture(self):
        manifest = dict(parent_step=2453, phase1_updates=1000, phase2_updates=2000)
        (self.root / "FORK_MANIFEST.json").write_text(json.dumps(manifest))
        for name, value in (("control", 1.), ("lean", .99), ("foot", 1.5)):
            self.fake_stage(name, "phase1", 3453, value)
        (self.root / "arms/ik_off").mkdir(parents=True)
        args = SimpleNamespace(phase1_updates=1000, phase2_updates=2000,
                               decision_hours=8., final_stop=10000, expected_parent_step=5200)
        protocol = dict(created_at=recovery.original.now(),
                        current_parent=dict(step=5200, sha256="current", path="parent"))
        return args, protocol

    def test_full_manager_resumes_missing_stage_selects_and_holds_at10k(self):
        args, protocol = self.manager_fixture()
        calls = []

        def fake_worker(command, **unused):
            name = command[command.index("--arm") + 1]
            label = command[command.index("--label") + 1]
            stop = int(command[command.index("--stop-step") + 1])
            calls.append((name, label, stop))
            value = 1.5 if label == "phase1" else .97 if name == "lean" and label == "phase2" else 1.
            self.fake_stage(name, label, stop, value, final="--final" in command)
            return SimpleNamespace(pid=123, poll=lambda: 0, returncode=0)

        with patch.object(recovery, "ROOT", self.root), patch.object(recovery.original, "ROOT", self.root), \
                patch.object(recovery, "guard_processes"), \
                patch.object(recovery, "build_protocol", return_value=protocol), \
                patch.object(recovery, "verify_preserved_parent") as parent_check, \
                patch.object(recovery, "cleanup_with_lean_retained") as cleanup, \
                patch.object(recovery.original, "paired_inputs") as pairing, \
                patch.object(recovery.subprocess, "Popen", side_effect=fake_worker), \
                patch.object(recovery.signal, "signal"):
            recovery._main(args)
        self.assertEqual(calls, [("ik_off", "phase1", 3453), ("control", "phase2", 5453),
                                 ("lean", "phase2", 5453), ("lean", "long_review", 10000)])
        self.assertEqual(pairing.call_count, 2)
        cleanup.assert_called_once_with("lean")
        self.assertEqual(parent_check.call_count, 2)
        state = json.loads((self.root / "STATUS.json").read_text())
        self.assertEqual(state["status"], "ready_for_user_review")
        self.assertEqual(state["selected"], "lean")
        self.assertEqual(state["completed_steps"], 10000)
        self.assertGreaterEqual(state["recovery_decision_elapsed_hours"], 0.)
        self.assertFalse(state["fallback_on_failure"])
        selection = json.loads((self.root / "FINAL_SELECTION.json").read_text())
        self.assertTrue(selection["original_parent_overwritten_by_prior_fallback"])
        self.assertFalse(json.loads((self.root / "EXTENDED_REVIEW_DECISION.json").read_text())["continue_to55000"])

    def test_worker_failure_never_launches_a_fallback(self):
        args, protocol = self.manager_fixture()
        with patch.object(recovery, "ROOT", self.root), patch.object(recovery.original, "ROOT", self.root), \
                patch.object(recovery, "guard_processes"), \
                patch.object(recovery, "build_protocol", return_value=protocol), \
                patch.object(recovery.subprocess, "Popen", return_value=SimpleNamespace(pid=123, poll=lambda: 1, returncode=1)) as launch, \
                patch.object(recovery.signal, "signal"):
            with self.assertRaisesRegex(RuntimeError, "exited 1"):
                recovery._main(args)
        self.assertEqual(launch.call_count, 1)
        self.assertIn("recovered_worker.py", launch.call_args.args[0][2])
        failure = json.loads((self.root / "RECOVERY_FAILURE.json").read_text())
        self.assertEqual(failure["status"], "recovery_failed")
        self.assertFalse(failure["fallback_on_failure"])

    def test_rejected_lean_optimizer_is_retained_without_relaxing_selection(self):
        folder = self.root / "arms/lean/training"
        folder.mkdir(parents=True)
        recovery.torch.save(dict(step=3453, optimizer={}), folder / "last.pt")
        with patch.object(recovery, "ROOT", self.root), patch.object(recovery.original, "ROOT", self.root), \
                patch.object(recovery, "compact_rejected_safely") as cleanup:
            recovery.cleanup_with_lean_retained("control")
        proxy, selected = cleanup.call_args.args
        self.assertEqual(selected, "control")
        self.assertEqual(proxy.NAMES, ("control", "foot", "ik_off"))
        kept = json.loads((self.root / "RECOVERY_RETAINED_ALTERNATIVE.json").read_text())
        self.assertTrue(kept["optimizer_preserved"])
        self.assertEqual(kept["step"], 3453)

    def test_selected_lean_remains_valid_cleanup_manager_member(self):
        folder = self.root / "arms/lean/training"
        folder.mkdir(parents=True)
        recovery.torch.save(dict(step=5453, optimizer={}), folder / "last.pt")
        with patch.object(recovery, "ROOT", self.root), patch.object(recovery.original, "ROOT", self.root), \
                patch.object(recovery, "compact_rejected_safely") as cleanup:
            recovery.cleanup_with_lean_retained("lean")
        proxy, selected = cleanup.call_args.args
        self.assertIn(selected, proxy.NAMES)


if __name__ == "__main__":
    unittest.main()
