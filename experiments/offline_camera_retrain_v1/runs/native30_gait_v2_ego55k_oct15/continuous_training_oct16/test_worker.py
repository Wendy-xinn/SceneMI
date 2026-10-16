import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('baseline_continuous_worker',
    Path(__file__).with_name('worker.py'))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class OriginalBaselineContinuationTests(unittest.TestCase):
    def test_original_wrapper_receives55000_and_adapter_before_main(self):
        calls = []
        trainer = object()
        def install(target, effective, log):
            self.assertIs(target.trainer, trainer)
            calls.append('install')
        def main():
            self.assertEqual(calls, ['install'])
            self.assertEqual(sys.argv[1:], ['--resume', '--review-step', '55000'])
            calls.append('main')
        previous = sys.argv
        worker.run_original(SimpleNamespace(trainer=trainer, main=main),
                            SimpleNamespace(install=install), {}, '/tmp/test-log')
        self.assertIs(sys.argv, previous)
        self.assertEqual(calls, ['install', 'main'])

    def test_failed_wrapper_restores_arguments_without_swallowing(self):
        previous = sys.argv
        def fail():
            raise RuntimeError('actual training failed')
        with self.assertRaisesRegex(RuntimeError, 'actual training failed'):
            worker.run_original(SimpleNamespace(trainer=None, main=fail),
                SimpleNamespace(install=lambda *args: None), {}, '/tmp/test-log')
        self.assertIs(sys.argv, previous)

    def test_uncommitted_rows_and_broken_tail_retained_before_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trace.jsonl'
            good = b'{"step":5200,"loss":1}\n'
            tail = b'{"step":5201,"loss":2}\n{"step":5257}\n{"step":'
            path.write_bytes(good + tail)
            record = worker.trim_uncommitted(path, 5200)
            self.assertEqual(path.read_bytes(), good)
            self.assertEqual(Path(record['retained_evidence']).read_bytes(), tail)
            self.assertEqual(record['discarded_rows'], 3)
            self.assertIsNone(worker.trim_uncommitted(path, 5200))

    def test_sealed_sources_must_include_worker_supervisor_wrapper_and_io(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'baseline/continuous'
            root.mkdir(parents=True)
            paths = [root / 'worker.py', root / 'supervise.py',
                     root.parent / 'run_training.py', root / 'io.py']
            for path in paths:
                path.write_text('original')
            record = dict(io_adapter=str(paths[-1]), expected_source_sha256={
                str(path): worker.sha256(path) for path in paths})
            (root / 'PROTOCOL.json').write_text(json.dumps(record))
            worker.verify_protocol(root)
            paths[-1].write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                worker.verify_protocol(root)


if __name__ == '__main__':
    unittest.main()
