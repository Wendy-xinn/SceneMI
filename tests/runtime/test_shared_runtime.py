import unittest
from unittest.mock import patch

from scenemi.shared_runtime import authorized_job


class SharedRuntimeTests(unittest.TestCase):
    def test_pinned_same_user_identity_is_required(self):
        record=dict(pid=17,uid=4,start_ticks=10,cwd='/a',executable='/b',command_sha256='x')
        with patch('scenemi.shared_runtime.os.getuid',return_value=4),patch('scenemi.shared_runtime.identity',return_value=record):
            self.assertTrue(authorized_job(17,[record]))
            self.assertFalse(authorized_job(17,[dict(record,start_ticks=11)]))
            self.assertFalse(authorized_job(17,[dict(record,command_sha256='y')]))
        with patch('scenemi.shared_runtime.os.getuid',return_value=5),patch('scenemi.shared_runtime.identity',return_value=record):
            self.assertFalse(authorized_job(17,[record]))

    def test_missing_or_reused_pid_is_not_admitted(self):
        with patch('scenemi.shared_runtime.identity',side_effect=FileNotFoundError()):
            self.assertFalse(authorized_job(17,[]))


if __name__=='__main__':unittest.main()
