import copy
import json
from pathlib import Path
import tempfile
import unittest

from scenemi.data.amass_prepare import sha256
from scenemi.replay.sharing import authorization, authorized_admission
from scenemi.replay.training import CONFIG


class SharingTests(unittest.TestCase):
    def auth(self):
        return dict(state='user_authorized', scope='formal_four_corpus_replay_training_and_resume',
            output='runs/scene_joint128_replay', config_sha256=sha256(CONFIG),
            parent_sha256='5145b2e65ac7792ddee225e90e97148c410578c81096c90f4fdb098018a2dafe',
            sharing='actual_free_memory_without_pid_binding', physical_reserve_gib=4)

    def test_exact_configuration_and_scope_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'authorization.json'
            auth = self.auth()
            path.write_text(json.dumps(auth))
            self.assertEqual(authorization(path, CONFIG), auth)
            for field in ('state', 'output', 'parent_sha256', 'sharing', 'config_sha256'):
                changed = dict(auth, **{field: 'different'})
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    authorization(path, CONFIG)

    def test_authorization_does_not_forge_a_pass_or_change_memory_checks(self):
        report = dict(admitted=False, memory_admitted=False, memory_max_bytes=12*2**30,
            gpu=dict(unknown_compute_pids=[3405], memory_free_mib=2000, admitted=False, policy='original'))
        original = lambda *args: copy.deepcopy(report)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'authorization.json'
            path.write_text(json.dumps(self.auth()))
            wrapped = authorized_admission(original, self.auth(), path)
            result = wrapped('cuda', 12, 4)
            self.assertFalse(result['admitted'])
            self.assertFalse(result['memory_admitted'])
            self.assertEqual(result['gpu']['memory_free_mib'], 2000)
            self.assertEqual(result['gpu']['user_authorized_external_pids'], [3405])
            self.assertEqual(result['gpu']['unknown_compute_pids'], [])
            self.assertEqual(wrapped('cpu', 12, 0), report)
            changed = copy.deepcopy(report)
            changed['gpu']['unknown_compute_pids'] = [9999]
            result = authorized_admission(lambda *args: changed, self.auth(), path)('cuda', 12, 4)
            self.assertEqual(result['gpu']['user_authorized_external_pids'], [9999])


if __name__ == '__main__':
    unittest.main()
