"""Real tiny checkpoints exercise repeated/interrupted cleanup on CPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch


SPEC = importlib.util.spec_from_file_location("review_recovery", Path(__file__).with_name("resume_review.py"))
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


class RecoveryCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        def write_json(path, value):
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(value, allow_nan=False))
            temporary.replace(path)
        def save_checkpoint(state, path):
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(state, temporary)
            temporary.replace(path)
        self.manager = SimpleNamespace(ROOT=self.root, NAMES=("control", "lean", "foot", "ik_off"),
            now=lambda: "test-time", write_json=write_json, save_checkpoint=save_checkpoint)
        for name in self.manager.NAMES:
            path = self.path(name)
            path.parent.mkdir(parents=True)
            torch.save(dict(model={"weight": torch.tensor([1., 2.])}, optimizer={"step": torch.tensor(9)},
                            data_rng={"state": 123}, torch_rng=torch.tensor([3, 4])), path)

    def path(self, name):
        return self.root / "arms" / name / "training" / "last.pt"

    def state(self, name):
        return torch.load(self.path(name), map_location="cpu", weights_only=False)

    def test_real_optimizer_cleanup_is_repeatable_and_protected_models_unchanged(self):
        protected = {name: self.path(name).read_bytes() for name in ("control", "lean")}
        before = {name: RECOVERY.model_hash(self.state(name)) for name in ("foot", "ik_off")}
        RECOVERY.compact_rejected_safely(self.manager, "lean")
        compact_bytes = {name: self.path(name).read_bytes() for name in ("foot", "ik_off")}
        RECOVERY.compact_rejected_safely(self.manager, "lean")
        for name in ("foot", "ik_off"):
            state = self.state(name)
            self.assertNotIn("optimizer", state)
            self.assertIs(state["evaluation_only"], True)
            self.assertEqual(RECOVERY.model_hash(state), before[name])
            self.assertEqual(state["data_rng"], {"state": 123})
            torch.testing.assert_close(state["torch_rng"], torch.tensor([3, 4]))
            self.assertEqual(self.path(name).read_bytes(), compact_bytes[name])
        for name, original in protected.items():
            self.assertEqual(self.path(name).read_bytes(), original)

    def test_original_cleanup_evidence_allows_already_compacted_model(self):
        state = self.state("foot")
        state.pop("optimizer"); state["evaluation_only"] = True
        torch.save(state, self.path("foot"))
        self.manager.write_json(self.root / "STORAGE_CLEANUP.json", dict(records=[dict(
            arm="foot", model_sha256=RECOVERY.model_hash(state), model_verified_unchanged=True)]))
        RECOVERY.compact_rejected_safely(self.manager, "lean")
        self.assertNotIn("optimizer", self.state("foot"))

    def test_missing_or_mismatched_evidence_fails_closed(self):
        state = self.state("foot")
        state.pop("optimizer"); state["evaluation_only"] = True
        torch.save(state, self.path("foot"))
        before = self.path("foot").read_bytes()
        with self.assertRaisesRegex(ValueError, "lacks matching cleanup hash"):
            RECOVERY.compact_rejected_safely(self.manager, "lean")
        self.manager.write_json(self.root / "STORAGE_CLEANUP.json", dict(records=[dict(
            arm="foot", model_sha256="wrong", model_verified_unchanged=True)]))
        with self.assertRaisesRegex(ValueError, "lacks matching cleanup hash"):
            RECOVERY.compact_rejected_safely(self.manager, "lean")
        self.assertEqual(self.path("foot").read_bytes(), before)
        self.assertIn("optimizer", self.state("ik_off"))

    def test_write_ahead_evidence_recovers_post_save_interruption(self):
        original_save = self.manager.save_checkpoint
        def interrupted_save(state, path):
            original_save(state, path)
            raise InterruptedError("after atomic checkpoint replacement")
        self.manager.save_checkpoint = interrupted_save
        with self.assertRaises(InterruptedError):
            RECOVERY.compact_rejected_safely(self.manager, "lean")
        self.assertNotIn("optimizer", self.state("foot"))
        journal = json.loads((self.root / RECOVERY.JOURNAL).read_text())
        self.assertEqual(journal["records"][0]["status"], "prepared")
        self.manager.save_checkpoint = original_save
        RECOVERY.compact_rejected_safely(self.manager, "lean")
        self.assertNotIn("optimizer", self.state("ik_off"))
        self.assertEqual(json.loads((self.root / RECOVERY.JOURNAL).read_text())["status"], "completed")

    def test_process_guard_rejects_live_review_and_ignores_reused_pid(self):
        proc = self.root / "fake_proc"
        owner = proc / "123"
        owner.mkdir(parents=True)
        self.manager.write_json(self.root / "STATUS.json", dict(pid=123, worker_pid=124))
        owner.joinpath("cmdline").write_bytes(b"python\0" + str(self.root / "run_review.py").encode() + b"\0")
        with self.assertRaisesRegex(RuntimeError, "still active"):
            RECOVERY.assert_no_live_review(self.root, proc_root=proc)
        owner.joinpath("cmdline").write_bytes(b"python\0unrelated.py\0")
        RECOVERY.assert_no_live_review(self.root, proc_root=proc)
        worker = proc / "124"
        worker.mkdir()
        worker.joinpath("cmdline").write_bytes(b"python\0" + str(self.root / "worker.py").encode() + b"\0")
        with self.assertRaisesRegex(RuntimeError, "worker_pid=124"):
            RECOVERY.assert_no_live_review(self.root, proc_root=proc)


if __name__ == "__main__":
    unittest.main()
