"""CPU-only guards for posthoc saved-array review; no inference runs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import exo_longrun_report as review


def entry():
    return dict(case=0, dataset="rich", recording="r", group="rich", mode="exo", member=0,
        sequence_id="r", source_first=64, start=64, case_seed=123, panel="fixed")


class PairingTests(unittest.TestCase):
    def test_array_identities_bound_to_registered_native_source_clock(self):
        source = dict(sequence_id="r", group="rich", mode="exo", start=64,
            source_first=0, source_start_30fps=64)
        arrays = dict(step=np.array(2000), identities_json=np.array(json.dumps([source])))
        review.bind_archive(arrays, [entry()], 2000)
        source["source_start_30fps"] = 65
        arrays["identities_json"] = np.array(json.dumps([source]))
        with self.assertRaisesRegex(ValueError, "identity"):
            review.bind_archive(arrays, [entry()], 2000)

    def test_complete_registered_seeded_rows_and_missing_duplicate_rejection(self):
        panel = [entry()]
        rows = [dict(panel[0], regime=regime, step=2000) for regime in review.REGIMES]
        self.assertEqual(len(review.bind_rows(rows, panel, 2000)), 2)
        for bad in (rows[:-1], rows+[rows[0]], [dict(rows[0], case_seed=999), rows[1]],
                    [dict(rows[0], step=4000), rows[1]]):
            with self.assertRaises(ValueError):
                review.bind_rows(bad, panel, 2000)

    def test_exact_saved_input_gt_reference_and_ego_retention(self):
        panel = [dict(entry(), mode="ego", panel="ego_retention")]
        arrays = dict(identities_json=np.array('[{"sequence_id":"r"}]'),
            **{"0_truth": np.zeros((3, 201)), "0_rest": np.zeros((22, 3))})
        for regime in review.REGIMES:
            for field in ("observations", "observation_meta", "reference_motion", "prediction"):
                arrays[f"0_{regime}_{field}"] = np.zeros((3, 2))
        self.assertEqual(review.verify_arrays(arrays, arrays, panel)["checked_anchor_regimes"], 2)
        for field in ("0_truth", "0_rest", "0_clean_observations", "0_noisy_observation_meta",
                      "0_clean_reference_motion", "0_clean_prediction"):
            other = copy.deepcopy(arrays)
            other[field].flat[0] = 1.
            with self.assertRaises(ValueError):
                review.verify_arrays(arrays, other, panel)


class AggregationTests(unittest.TestCase):
    def rows(self):
        result = []
        for dataset, value in (("rich", 10.), ("trumans", 30.), ("egobody", 60.)):
            for regime in review.REGIMES:
                for label in ("control", "input_reference", "adapter_2000"):
                    result.append(dict(dataset=dataset, regime=regime, model=label, mode="exo", panel="fixed",
                        metrics=dict(hidden_mpjpe_mm=value), counts=dict(hidden_mpjpe_mm=12)))
        return result

    def test_natural_weighted_fixed_only_comparisons(self):
        rows = self.rows()
        rows.append(dict(rows[0], panel="regression_probe", metrics=dict(hidden_mpjpe_mm=1e6)))
        groups = review.summarize(rows, dict(rich=.25, trumans=.25, egobody=.5))
        self.assertEqual(groups["control/clean/natural_weighted"]["hidden_mpjpe_mm"]["natural_window_mean"], 40.)
        comparisons = review.comparisons(groups, 2000)
        self.assertEqual(len(comparisons), 16)
        self.assertTrue(all(row["better"] is False and row["delta_mm"] == 0 for row in comparisons))

    def test_empty_masks_excluded_and_invalid_denominators_rejected(self):
        rows = self.rows()
        rows[0]["metrics"]["hidden_mpjpe_mm"] = None
        rows[0]["counts"]["hidden_mpjpe_mm"] = 0
        groups = review.summarize(rows, dict(rich=.25, trumans=.25, egobody=.5))
        self.assertIsNone(groups["control/clean/rich"]["hidden_mpjpe_mm"]["window_mean"])
        rows[0]["counts"]["hidden_mpjpe_mm"] = 1
        with self.assertRaisesRegex(ValueError, "denominator"):
            review.summarize(rows, dict(rich=.25, trumans=.25, egobody=.5))

    def test_incomplete_training_cannot_start_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "training").mkdir()
            (folder / "training/protocol.json").write_text(json.dumps(dict(status="training", completed_steps=20)))
            with self.assertRaisesRegex(ValueError, "completed"):
                review.report(folder)


if __name__ == "__main__":
    unittest.main()
