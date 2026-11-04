"""Cleanup must fail before deletion if dependencies or identities change."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.offline_camera_retrain_v1.ops import maintenance as m


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base = self.root / "experiments/offline_camera_retrain_v1"
        self.output = self.base / "runs/retired"
        self.output.mkdir(parents=True)
        self.source = self.base / "retired.py"
        self.source.write_text("x = 1\n")
        self.keep = self.base / "current.py"
        self.keep.write_text("x = 2\n")
        self.array = self.output / "prediction.npy"
        self.array.write_bytes(b"rebuildable")
        self.metric = self.output / "metrics.json"
        self.metric.write_text('{"error": 1}\n')
        self.patches = [patch.object(m, "ROOT", self.root), patch.object(m, "BASE", self.base)]
        for p in self.patches:
            p.start()
        bundle = m.snapshot_sources(self.base / "archive/sources.tar.gz")
        self.policy = self.root / "policy.json"
        m.write_json(self.policy, dict(protected={str(self.keep): m.digest(self.keep)},
            retired_sources=[str(self.source)], retired_output_roots=[str(self.output)], source_bundle=bundle))
        self.plan = self.root / "plan.json"
        m.plan(self.policy, self.plan)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temporary.cleanup()

    def test_only_enumerated_artifacts_removed_and_metrics_retained(self):
        report = self.root / "report.json"
        m.apply(self.plan, report)
        self.assertFalse(self.source.exists())
        self.assertFalse(self.array.exists())
        self.assertTrue(self.keep.exists())
        self.assertTrue(self.metric.exists())
        self.assertEqual(json.loads(report.read_text())["deleted_count"], 2)

    def test_changed_dependency_prevents_every_deletion(self):
        self.keep.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "Protected dependency changed"):
            m.apply(self.plan, self.root / "report.json")
        self.assertTrue(self.array.exists())
        self.assertTrue(self.source.exists())

    def test_changed_artifact_prevents_every_deletion(self):
        self.array.write_bytes(b"new prediction")
        with self.assertRaisesRegex(ValueError, "candidate changed"):
            m.apply(self.plan, self.root / "report.json")
        self.assertTrue(self.array.exists())
        self.assertTrue(self.source.exists())

    def test_symlink_to_external_data_is_never_a_target(self):
        outside = self.root / "source.npy"
        outside.write_bytes(b"source data")
        link = self.output / "linked.npy"
        link.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Symlink or external"):
            m.local_file(link)

    def test_root_run_retirement_is_exact_and_preserves_neighbours(self):
        folder = self.root / 'runs/finished/selection'
        folder.mkdir(parents=True)
        rejected = folder / 'unused.pt'
        selected = folder / 'best.pt'
        rejected.write_bytes(b'unselected')
        selected.write_bytes(b'selected')
        policy = json.loads(self.policy.read_text())
        policy.update(retired_sources=[], retired_output_roots=[],
                      retired_artifacts=[str(rejected)])
        policy['protected'][str(selected)] = m.digest(selected)
        m.write_json(self.policy, policy)
        m.plan(self.policy, self.plan)
        m.apply(self.plan, self.root / 'report.json')
        self.assertFalse(rejected.exists())
        self.assertTrue(selected.exists())
        self.assertTrue(self.metric.exists())

    def test_explicit_artifacts_reject_data_and_source_paths(self):
        for path in (self.root / 'dataset.npy', self.root / 'runs/retired.py'):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'protected input')
            policy = json.loads(self.policy.read_text())
            policy['retired_artifacts'] = [str(path)]
            m.write_json(self.policy, policy)
            with self.assertRaisesRegex(ValueError, 'generated file in root runs'):
                m.plan(self.policy, self.plan)


if __name__ == "__main__":
    unittest.main()
