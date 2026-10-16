"""Safety checks for exact-step matching and independent model-only retention."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location("guardian", Path(__file__).with_name("checkpoint_guardian.py"))
guardian = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guardian)


def record(step=16000):
    common = dict(step=step, dataset="rich", panel="fixed", mode="ego", recording="record1",
                  group="rich", sequence_id="seq1", source_first=128, start=128, case_seed=5, joints=[15])
    rows = [dict(common, regime=regime, metrics={"mpjpe_mm": {"value": value, "count": 4224}})
            for regime, value in (("clean", 100.0), ("noisy", 120.0))]
    rows.append(dict(rows[0], panel="rotating", metrics={"mpjpe_mm": {"value": 1.0, "count": 4224}}))
    rows.append(dict(rows[0], scene_variant="scene_off", metrics={"mpjpe_mm": {"value": 1.0, "count": 4224}}))
    return dict(step=step, cases=rows)


class GuardianTests(unittest.TestCase):
    def test_only_fixed_scene_on_pair(self):
        self.assertEqual(guardian.fixed_rich_score(record())["score_mm"], 110.0)
        ambiguous = record()
        ambiguous["cases"].append(copy.deepcopy(ambiguous["cases"][0]))
        with self.assertRaises(ValueError):
            guardian.fixed_rich_score(ambiguous)
        mismatch = record()
        mismatch["cases"][1]["source_first"] = 999
        with self.assertRaises(ValueError):
            guardian.fixed_rich_score(mismatch)

    def test_atomic_exact_step_and_tensor_equality(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            training = run / "training"
            training.mkdir()
            checkpoint = dict(step=15900, planned_steps=55000, effective_config={"config": "sealed"},
                              model={"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
                              optimizer={"state": {"dummy": torch.ones(10)}})
            source = training / "last.pt"
            torch.save(checkpoint, source)
            score = guardian.fixed_rich_score(record())
            audit = ({"config": "sealed"}, {"sealed_sources": 365})
            with patch.object(guardian, "verify_sources", return_value=audit):
                self.assertEqual(guardian.capture_if_step(run, score, score)["status"], "waiting")
                self.assertFalse((training / "best.pt").exists())
                checkpoint["step"] = 16100
                torch.save(checkpoint, source)
                self.assertEqual(guardian.capture_if_step(run, score, score)["status"], "weights_unavailable")
                checkpoint["step"] = 16000
                torch.save(checkpoint, source)
                original = guardian.sha(source)
                result = guardian.capture_if_step(run, score, score)
                self.assertEqual(result["status"], "saved")
                self.assertEqual(guardian.sha(source), original)
                retained = torch.load(training / "best.pt", weights_only=False)
                self.assertNotIn("optimizer", retained)
                self.assertTrue(torch.equal(retained["model"]["weight"], checkpoint["model"]["weight"]))
                recovered = guardian.recover_best(training)
                self.assertEqual(recovered["checkpoint_step"], 16000)
                self.assertTrue(recovered["tensor_equality_verified"])
                self.assertEqual(recovered["checkpoint_sha256"], result["best"]["checkpoint_sha256"])
            self.assertEqual(list(training.glob(".guardian-*")), [])

    def test_unfinished_line_and_identity_change(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "validation.jsonl"
            path.write_text(json.dumps(record()) + "\n" + '{"step":17000')
            self.assertEqual(len(guardian.validation_scores(path)), 1)
            changed = record(17000)
            for row in changed["cases"]:
                row["source_first"] = 1000
            path.write_text(json.dumps(record()) + "\n" + json.dumps(changed) + "\n")
            with self.assertRaises(ValueError):
                guardian.validation_scores(path)


if __name__ == "__main__":
    unittest.main()
