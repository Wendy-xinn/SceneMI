"""CPU checks for comparable selection and outcome-independent VAL rotation."""
import copy
from collections import Counter
from types import SimpleNamespace
import unittest

import numpy as np

from . import exo_longrun_panel as panel


def fake_data(per_record=16):
    members = {}
    for group in ("rich", "trumans", "camera_wearer", "interactee"):
        members[group] = []
        for i in range(4):
            record = ("egobody" if group in ("camera_wearer", "interactee") else group) + str(i)
            members[group].append((dict(split="validation", path="/unused/"+group+str(i),
                sequence_id=group+str(i), recording=record, source_first=100,
                allowed_modes=["ego", "exo"]), np.arange(per_record)*64))
    result = SimpleNamespace(split="validation", frames=192, members=members)
    result.starts_for = lambda group, member, mode: members[group][member][1]
    return result


class PanelTests(unittest.TestCase):
    def test_real_size_allocation_and_invalid_small_budget(self):
        self.assertEqual(panel.largest_remainder(dict(rich=206, trumans=709, egobody=820), 100),
            dict(rich=12, trumans=41, egobody=47))
        with self.assertRaises(ValueError):
            panel.largest_remainder(dict(rich=1, trumans=1, egobody=1), 2)

    def test_deterministic_disjoint_rotation_recording_spread_and_ego_separation(self):
        data = fake_data()
        manifest = panel.build_panels(data, seed=71, count=24, milestones=(2000, 4000, 6000))
        self.assertEqual(manifest, panel.build_panels(data, seed=71, count=24, milestones=(2000, 4000, 6000)))
        self.assertEqual(manifest["dataset_quotas"], dict(rich=6, trumans=6, egobody=12))
        fixed = {panel.identity(row) for row in manifest["fixed_exo"]}
        seen = set()
        for rows in manifest["rotating_exo"].values():
            ids = {panel.identity(row) for row in rows}
            self.assertFalse(fixed & ids)
            self.assertFalse(seen & ids)
            self.assertEqual(len(ids), 24)
            seen.update(ids)
            for corpus in panel.DATASETS:
                subset = [r for r in rows if r["dataset"] == corpus]
                records = Counter(r["recording"] for r in subset)
                self.assertEqual(len(records), 4)
                self.assertLessEqual(max(records.values())-min(records.values()), 1)
            roles = Counter(row["group"] for row in rows if row["dataset"] == "egobody")
            self.assertEqual(roles, dict(camera_wearer=6, interactee=6))
        self.assertEqual(len(manifest["ego_retention"]), 9)
        result = panel.select_panel(manifest, 2000, "fixed", include_ego=True)
        self.assertEqual(len(result), 33)
        result[0]["start"] = -1
        self.assertNotEqual(result[0]["start"], manifest["fixed_exo"][0]["start"])
        with self.assertRaises(ValueError):
            panel.select_panel(manifest, 3000, "rotating")

    def test_rotation_exhaustion_consumes_tail_and_logs_cycle(self):
        manifest = panel.build_panels(fake_data(per_record=3), seed=11, count=12,
            milestones=(1, 2, 3, 4, 5, 6), ego_per_corpus=1)
        self.assertTrue(any(event["crossed_cycle"] for events in manifest["rotation_cycles"].values() for event in events.values()))
        for corpus in panel.DATASETS:
            population = manifest["eligible_exo_counts"][corpus]-manifest["dataset_quotas"][corpus]
            sequence = [panel.identity(row) for rows in manifest["rotating_exo"].values()
                for row in rows if row["dataset"] == corpus]
            self.assertEqual(len(set(sequence[:population])), population)

    def test_source_and_panel_tamper_and_wrong_split_rejected(self):
        data = fake_data()
        manifest = panel.build_panels(data, seed=3, count=12, milestones=(1,))
        altered = copy.deepcopy(manifest)
        altered["fixed_exo"][0]["start"] += 1
        with self.assertRaisesRegex(ValueError, "hash"):
            panel.validate_manifest(data, altered)
        altered = copy.deepcopy(manifest)
        altered["natural_dataset_weights"]["rich"] = .5
        with self.assertRaisesRegex(ValueError, "weights"):
            panel.validate_manifest(data, altered)
        data.split = "test"
        with self.assertRaisesRegex(ValueError, "validation"):
            panel.build_panels(data, seed=3, count=12, milestones=(1,))

    def test_known_regression_is_exact_or_explicitly_unavailable(self):
        data = fake_data()
        manifest = panel.build_panels(data, seed=3, count=12, milestones=(1,))
        self.assertEqual(manifest["regression_registration"]["status"], "unavailable")
        meta, _ = data.members["interactee"][0]
        meta["sequence_id"] = panel.WRIST_SEQUENCE
        data.members["interactee"][0] = (meta, np.array([0, 1984, 2048]))
        manifest = panel.build_panels(data, seed=3, count=12, milestones=(1,))
        probes = panel.select_panel(manifest, kind="regression")
        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0]["start"], 1984)
        self.assertEqual(probes[0]["case_seed"], 1568365135)
        self.assertEqual(probes[0]["panel"], "regression_probe")


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.manifest = panel.build_panels(fake_data(), seed=13, count=12, milestones=(1,))
        self.rows = []
        for row in self.manifest["fixed_exo"]:
            for regime in ("clean", "noisy"):
                value = dict(rich=10., trumans=30., egobody=60.)[row["dataset"]]
                metrics = dict(unconditioned_mpjpe_mm=dict(value=value, count=100),
                    conditioned_mpjpe_mm=dict(value=5., count=200))
                self.rows.append(dict(row, regime=regime, metrics=metrics, reference_metrics=copy.deepcopy(metrics)))

    def test_score_natural_weights_fixed_only_and_separate_reference(self):
        # Natural weights .25/.25/.5, rather than equal-corpus mean 33.333.
        result = panel.score_fixed_rows(self.rows, self.manifest)
        self.assertEqual(result["score"], 40.)
        self.assertEqual(result["reference_score"], 40.)
        self.assertAlmostEqual(result["totals"]["metrics"]["clean"]["unconditioned_mpjpe_mm"]["equal_dataset_mean"], 100/3)
        excluded = [dict(self.rows[0], panel="rotating", metrics={}),
            dict(self.rows[0], mode="ego", panel="ego_retention", metrics={}),
            dict(self.rows[0], panel="regression_probe", metrics={}),
            dict(self.rows[0], scene_variant="scene_off", metrics={})]
        self.assertEqual(result, panel.score_fixed_rows(self.rows+excluded, self.manifest))

    def test_window_mean_differs_from_joint_pooled_and_empty_masks_excluded(self):
        rich_clean = [r for r in self.rows if r["dataset"] == "rich" and r["regime"] == "clean"]
        rich_clean[0]["metrics"]["unconditioned_mpjpe_mm"] = dict(value=20., count=10)
        rich_clean[1]["metrics"]["unconditioned_mpjpe_mm"] = dict(value=None, count=0)
        result = panel.score_fixed_rows(self.rows, self.manifest)
        detail = result["groups"]["metrics/clean/rich"]["unconditioned_mpjpe_mm"]
        self.assertEqual(detail["valid_windows"], 2)
        self.assertEqual(detail["window_mean"], 15.)
        self.assertAlmostEqual(detail["joint_pooled_mean"], 1200/110)

    def test_missing_duplicate_nonfinite_or_incomplete_result_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            panel.score_fixed_rows(self.rows[:-1], self.manifest)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            panel.score_fixed_rows(self.rows+[self.rows[0]], self.manifest)
        broken = copy.deepcopy(self.rows)
        broken[0]["metrics"]["unconditioned_mpjpe_mm"]["value"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            panel.score_fixed_rows(broken, self.manifest)
        broken = copy.deepcopy(self.rows)
        del broken[0]["reference_metrics"]
        with self.assertRaisesRegex(ValueError, "partially"):
            panel.score_fixed_rows(broken, self.manifest)
        broken = copy.deepcopy(self.rows)
        broken[0]["case_seed"] += 1
        with self.assertRaisesRegex(ValueError, "metadata or seed"):
            panel.score_fixed_rows(broken, self.manifest)


if __name__ == "__main__":
    unittest.main()
