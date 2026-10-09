import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from experiments.offline_camera_retrain_v1.source_recovery_contract import validate_source_recovery


class RecoveryContractTests(unittest.TestCase):
    def exercise(self, changes=None, protocol='native_v1', source_hash='old'):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);audit=root/'audit.json'
            report=dict(status='passed',protocol='native-source-recovery-v1',windows=192,paired_predictions=384,
                        maximum_difference={'motion':0.,'prediction':0.},reference_source_hash='old',
                        pipeline_sha256='pipeline',audited_archive_fingerprint='archive')
            report.update(changes or {});audit.write_text(json.dumps(report))
            (root/'READY.json').write_text(json.dumps(dict(reconstruction_protocol='native-source-recovery-v1',semantic_audit_path=str(audit))))
            args=SimpleNamespace(body_protocol=protocol,trumans_window_protocol='temporal_valid_v1',
                                 temporal_scene_manifest='exact20',trumans_scene_manifest='dynamic20')
            with patch('experiments.offline_camera_retrain_v1.source_recovery_contract.SOURCE',root), \
                 patch('experiments.offline_camera_retrain_v1.train.audited_data_fingerprint',return_value='archive'), \
                 patch('experiments.offline_camera_retrain_v1.train.fingerprint',return_value='pipeline'):
                return validate_source_recovery(audit,source_hash,args)

    def test_valid_frozen_replay_gate(self):
        self.assertEqual(self.exercise()['status'],'passed')

    def test_rejects_other_source_and_legacy_protocol(self):
        for keywords in [dict(source_hash='unrelated'),dict(protocol='legacy')]:
            with self.assertRaises(ValueError):self.exercise(**keywords)

    def test_rejects_missing_windows_and_changed_tensors(self):
        for changes in [dict(windows=191),dict(paired_predictions=383),dict(maximum_difference={'motion':.01})]:
            with self.assertRaises(ValueError):self.exercise(changes)

    def test_rejects_modified_pipeline_and_archive(self):
        for changes in [dict(pipeline_sha256='changed'),dict(audited_archive_fingerprint='changed')]:
            with self.assertRaises(ValueError):self.exercise(changes)

if __name__=='__main__':unittest.main()
