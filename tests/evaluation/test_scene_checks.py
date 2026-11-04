import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scenemi import scene_checks


class FixedStepTests(unittest.TestCase):
    def test_refused_admission_stops_watcher_without_fallback(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);config=root/'policy.json';config.write_text('{}')
            candidate=root/'control/selection/candidates/step_003000.json'
            candidate.parent.mkdir(parents=True);candidate.write_text('{}')
            c=dict(control_run='control',output='evaluation',step=3000,max_wait_seconds=60,
                   runtime=dict(memory_gib=8,cpu_count=2,allocator_gib=6))
            with patch.object(scene_checks,'ROOT',root),patch.object(scene_checks,'policy',return_value=(c,{})), \
                    patch('scenemi.parallel_resources.main',side_effect=SystemExit('Resource admission refused')) as launch, \
                    patch.dict('os.environ',{}),patch('sys.argv',[]):
                with self.assertRaises(SystemExit):scene_checks.watch(config)
                launch.assert_called_once()
            result=json.loads((root/'evaluation/WATCH_STATUS.json').read_text())
            self.assertEqual(result['state'],'admission_or_evaluation_failed')

    def test_test_membership_cannot_be_enabled_for_fixed_step_validation(self):
        with TemporaryDirectory() as folder:
            p=Path(folder)/'policy.json'
            p.write_text(json.dumps(dict(protocol='native-scene-fixed-step-v1',step=3000,
                                         test_used=True,validation_tier='comprehensive')))
            with self.assertRaises(ValueError):scene_checks.policy(p)


if __name__=='__main__':unittest.main()
