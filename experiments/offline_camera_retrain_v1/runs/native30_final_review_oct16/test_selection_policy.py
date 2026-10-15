"""Decision counterexamples for the bounded development selector; CPU only."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("review_selection", Path(__file__).with_name("selection_policy.py"))
SELECTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTION)
choose_candidate = SELECTION.choose_candidate


def summary(*, compute=2.0):
    row = dict(mpjpe_mm=100.0, conditioned_mpjpe_mm=10.0, root_orientation_deg=20.0,
               body_local_foot_error_mm=200.0, foot_velocity_RMSE_m_s=1.0,
               articulation_log_ratio_error=1.0)
    return dict(aggregate={p + "/" + r: dict(row)
                           for p in ("all", "locomotion") for r in ("clean", "noisy")},
                timing=dict(mean_compute_s=compute, mean_update_s=compute + 0.3))


class SelectorTests(unittest.TestCase):
    def test_each_guard_includes_boundary_but_excludes_just_above(self):
        for regime in SELECTION.REGIMES:
            for panel, metric, multiplier, allowance in SELECTION.GUARDS:
                with self.subTest(regime=regime, panel=panel, metric=metric):
                    control = summary()
                    candidate = summary(compute=1.0)
                    key = panel + "/" + regime
                    boundary = multiplier * control["aggregate"][key][metric] + allowance
                    candidate["aggregate"][key][metric] = boundary
                    result = choose_candidate(dict(control=control, lean=candidate))
                    self.assertIn("lean", result["eligible"])
                    candidate["aggregate"][key][metric] = boundary + 1e-7
                    result = choose_candidate(dict(control=control, lean=candidate))
                    self.assertNotIn("lean", result["eligible"])
                    self.assertEqual(result["selected"], "control")

    def test_missing_and_nonfinite_candidates_fail_closed_json_safe(self):
        for value in (None, float("nan"), float("inf"), -1.0, True, "100"):
            with self.subTest(value=value):
                bad = summary(compute=0.1)
                bad["aggregate"]["all/noisy"]["mpjpe_mm"] = value
                result = choose_candidate(dict(control=summary(), lean=bad))
                self.assertEqual(result["selected"], "control")
                self.assertEqual(result["eligible"], [])
                self.assertIsNone(result["scores"]["lean"])
                json.dumps(result, allow_nan=False)
        bad = summary()
        del bad["aggregate"]["locomotion/clean"]["conditioned_mpjpe_mm"]
        self.assertEqual(choose_candidate(dict(control=summary(), foot=bad))["eligible"], [])

    def test_missing_or_invalid_control_raises(self):
        for arms in ({}, dict(lean=summary()), dict(control={}), dict(control=None)):
            with self.subTest(arms=arms):
                with self.assertRaises(ValueError):
                    choose_candidate(arms)
        bad = summary()
        bad["timing"]["mean_compute_s"] = float("nan")
        with self.assertRaises(ValueError):
            choose_candidate(dict(control=bad))

    def test_clear_quality_improvement_beats_faster_weak_arm(self):
        foot = summary(compute=2.5)
        for row in foot["aggregate"].values():
            for metric in ("mpjpe_mm", "body_local_foot_error_mm",
                           "foot_velocity_RMSE_m_s", "articulation_log_ratio_error"):
                row[metric] *= 0.8
        result = choose_candidate(dict(control=summary(), lean=summary(compute=0.1), foot=foot))
        self.assertEqual(result["selected"], "foot")
        self.assertAlmostEqual(result["scores"]["foot"], 0.8)

    def test_near_equal_quality_prefers_faster_compute(self):
        fast = summary(compute=0.8)
        for row in fast["aggregate"].values():
            for metric in ("mpjpe_mm", "body_local_foot_error_mm",
                           "foot_velocity_RMSE_m_s", "articulation_log_ratio_error"):
                row[metric] *= 1.02
        fast["timing"]["mean_update_s"] = 50.0
        result = choose_candidate(dict(control=summary(), lean=fast))
        self.assertEqual(result["selected"], "lean")
        self.assertAlmostEqual(result["scores"]["lean"], 1.02)

    def test_three_percent_tie_is_measured_from_best_not_transitive(self):
        arms = dict(control=summary(compute=5), lean=summary(compute=2), foot=summary(compute=1))
        for row in arms["lean"]["aggregate"].values():
            row["body_local_foot_error_mm"] *= 1.05  # score 1.02
        for row in arms["foot"]["aggregate"].values():
            row["body_local_foot_error_mm"] *= 1.10  # score 1.04; fails foot guard too
        # Use amplitude to keep the second near-match inside all guardrails.
        for row in arms["foot"]["aggregate"].values():
            row["body_local_foot_error_mm"] = 200.0
            row["articulation_log_ratio_error"] *= 1.14  # score 1.042
        result = choose_candidate(arms)
        self.assertEqual(result["selected"], "lean")

    def test_faster_arm_failing_head_or_root_is_not_selected(self):
        for metric, value in (("conditioned_mpjpe_mm", 15.01), ("root_orientation_deg", 23.01)):
            bad = summary(compute=0.01)
            bad["aggregate"]["all/noisy"][metric] = value
            result = choose_candidate(dict(control=summary(), ik_off=bad))
            self.assertEqual(result["selected"], "control")
            self.assertNotIn("ik_off", result["eligible"])

    def test_control_fallback_and_inputs_unchanged(self):
        arms = dict(control=summary(), lean=summary(compute=3.0))
        before = copy.deepcopy(arms)
        result = choose_candidate(arms)
        self.assertEqual(result["selected"], "control")
        self.assertEqual(arms, before)
        self.assertEqual(choose_candidate(dict(control=summary()))["selected"], "control")

    def test_invalid_timings_are_ineligible(self):
        for metric in ("mean_compute_s", "mean_update_s"):
            for value in (None, 0, -1, float("inf"), float("nan")):
                bad = summary()
                bad["timing"][metric] = value
                self.assertEqual(choose_candidate(dict(control=summary(), lean=bad))["eligible"], [])

    def test_zero_control_metrics_remain_json_safe(self):
        control = summary()
        for row in control["aggregate"].values():
            row["articulation_log_ratio_error"] = 0.0
        result = choose_candidate(dict(control=control, lean=copy.deepcopy(control)))
        json.dumps(result, allow_nan=False)
        self.assertTrue(all(value is None or value >= 0 for value in result["scores"].values()))


if __name__ == "__main__":
    unittest.main()
