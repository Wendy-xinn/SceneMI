import subprocess
import unittest
from unittest.mock import patch

from . import resources


class ResourceAdmissionTests(unittest.TestCase):
    def gpu(self, measurement, pids):
        results = iter([measurement, pids])
        return resources.inspect_gpu(lambda *a, **k: next(results))

    def test_foreign_pid_refused_even_when_not_visible_in_proc(self):
        self.assertFalse(self.gpu("500, 0\n", "5635\n")["admitted"])

    def test_busy_without_compute_pid_refused(self):
        self.assertFalse(self.gpu("14000, 70\n", "")["admitted"])

    def test_idle_gpu_admitted(self):
        self.assertTrue(self.gpu("500, 0\n", "")["admitted"])

    def test_unknown_driver_values_cannot_be_admitted(self):
        with self.assertRaises(ValueError):
            self.gpu("[N/A], [N/A]\n", "")

    def test_cpu_does_not_query_gpu(self):
        with patch.object(resources, "available_memory", return_value=20 * resources.GIB), \
             patch.object(resources, "inspect_gpu", side_effect=AssertionError):
            self.assertTrue(resources.admission("cpu", 12)["admitted"])

    def test_memory_reserve_required(self):
        with patch.object(resources, "available_memory", return_value=15 * resources.GIB):
            self.assertFalse(resources.admission("cpu", 12)["admitted"])

    def test_denial_never_starts_program(self):
        with patch.object(resources, "admission", return_value={"admitted": False}), \
             patch.object(resources.subprocess, "call") as call:
            with self.assertRaises(SystemExit) as error:
                resources.main(["run", "--", "python", "train.py"])
            self.assertEqual(error.exception.code, 2)
            call.assert_not_called()

    def test_scope_limits_only_the_new_unit(self):
        command = resources.scope_command(["python", "train.py"], 12, 4)
        self.assertIn("--scope", command)
        self.assertIn("--property=MemoryMax=" + str(12 * resources.GIB), command)
        self.assertIn("_exec", command)
        self.assertEqual(command[command.index("--cpu-count") + 1], "4")
        self.assertEqual(command[-3:], ["--", "python", "train.py"])


if __name__ == "__main__":
    unittest.main()
