"""Attribution tests: never label one validation's meshes with another step."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("visual_capture", Path(__file__).with_name("visual_capture.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ValidationAttribution(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.training = self.root / "training"
        self.training.mkdir()
        self.identities, self.cases, self.arrays = [], [], {}
        for i, dataset in enumerate(("trumans", "egobody", "rich")):
            self.identities.append(dict(sequence_id=f"seq{i}", dataset=dataset, start=i * 64,
                source_start_30fps=100 + i * 64, native_body=dict(model="smpl", gender="female", betas=[0.] * 10)))
            self.arrays[f"{i}_truth"] = np.zeros((192, 201), np.float32)
            self.arrays[f"{i}_rest"] = np.zeros((22, 3), np.float32)
            for regime in ("clean", "noisy"):
                self.cases.append(dict(case=i, step=7000, dataset=dataset, sequence_id=f"seq{i}",
                    start=i * 64, source_first=100 + i * 64, panel="fixed", mode="ego", regime=regime))
                self.arrays[f"{i}_{regime}_prediction"] = np.zeros((192, 201), np.float32)
        self.record = dict(step=7000, cases=self.cases)
        self.arrays.update(step=np.array(7000), identities_json=np.array(json.dumps(self.identities)))

    def publish(self):
        np.savez_compressed(self.training / "validation_latest.npz", **self.arrays)
        (self.training / "validation.jsonl").write_text(json.dumps(self.record) + "\n")

    def test_exact_published_pair_has_authoritative_step_and_digest(self):
        self.publish()
        saved, record, provenance = module.load_completed_snapshot(self.root)
        self.assertEqual(int(saved["step"]), 7000)
        self.assertEqual(record["step"], 7000)
        self.assertEqual(len(provenance["source_archive_sha256"]), 64)

    def test_newer_log_does_not_relabel_older_archive(self):
        self.publish()
        newer = dict(step=8000, cases=[])
        with (self.training / "validation.jsonl").open("a") as out:
            out.write(json.dumps(newer) + "\n")
        self.assertEqual(module.load_completed_snapshot(self.root)[1]["step"], 7000)

    def test_archive_without_step_is_never_guessed(self):
        del self.arrays["step"]
        self.publish()
        with self.assertRaisesRegex(module.CapturePending, "authoritative step"):
            module.load_completed_snapshot(self.root)

    def test_unmatched_step_waits(self):
        self.arrays["step"] = np.array(8000)
        self.publish()
        with self.assertRaisesRegex(module.CapturePending, "matches archive step 8000"):
            module.load_completed_snapshot(self.root)

    def test_identity_mismatch_fails_closed(self):
        self.cases[0]["start"] += 64
        self.publish()
        with self.assertRaisesRegex(module.CapturePending, "identity differs"):
            module.load_completed_snapshot(self.root)

    def test_clean_noisy_missing_fails_closed(self):
        self.cases.pop()
        self.publish()
        with self.assertRaisesRegex(module.CapturePending, "pair is incomplete"):
            module.load_completed_snapshot(self.root)

    def test_native_profile_missing_never_substitutes_skeleton(self):
        del self.identities[0]["native_body"]
        self.arrays["identities_json"] = np.array(json.dumps(self.identities))
        self.publish()
        with self.assertRaisesRegex(module.CapturePending, "cannot display native meshes"):
            module.load_completed_snapshot(self.root)

    def test_partial_json_tail_does_not_invalidate_completed_record(self):
        self.publish()
        with (self.training / "validation.jsonl").open("a") as out:
            out.write('{"step":8000,')
        self.assertEqual(module.load_completed_snapshot(self.root)[1]["step"], 7000)

    def test_scene_off_is_not_included_as_another_fixed_pair(self):
        off = dict(self.cases[0], scene_variant="scene_off")
        self.cases.append(off)
        self.publish()
        self.assertEqual(module.load_completed_snapshot(self.root)[1]["step"], 7000)

    def test_camera_is_orthonormal(self):
        basis = module.camera_basis()
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
