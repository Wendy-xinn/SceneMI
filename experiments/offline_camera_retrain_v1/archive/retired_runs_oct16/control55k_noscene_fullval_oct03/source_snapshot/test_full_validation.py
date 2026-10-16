import unittest
from types import SimpleNamespace

from experiments.offline_camera_retrain_v1.evaluate_full_validation import enumerate_windows, summarize


class FullValidationTest(unittest.TestCase):
    def dataset(self):
        groups = {
            'trumans': [({'sequence_id': 'too_short'}, {128: []}),
                        ({'sequence_id': 'a'}, {128: [(0, 0), (20, 30)]})],
            'camera_wearer': [({'sequence_id': 'b'}, {128: [(40, 60)]})],
            'interactee': [({'sequence_id': 'c'}, {128: [(0, 150)]})],
        }
        return SimpleNamespace(base=SimpleNamespace(groups=groups),
                               summary=lambda: {g: {'128': {'starts': sum(len(v[128]) for _, v in rows)}}
                                                for g, rows in groups.items()})

    def test_exhaustive_filtered_sequence_indices(self):
        windows = enumerate_windows(self.dataset(), 128)
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows[0]['sequence_index'], 0)
        self.assertEqual(windows[1]['start_index'], 1)
        self.assertEqual(windows[-1]['source_start_30fps'], 150)
        self.assertEqual([r['index'] for r in windows], list(range(4)))

    def test_duplicate_start_rejected(self):
        dataset = self.dataset()
        dataset.base.groups['trumans'][1][1][128].append((20, 30))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            enumerate_windows(dataset, 128)

    def test_window_and_sequence_weights_differ(self):
        rows = [dict(group='trumans', identity={'sequence_id': name}, metrics={'error': v})
                for name, v in [('a', 1.), ('a', 3.), ('b', 8.)]]
        report = summarize(rows)
        self.assertEqual(report['windows'], 3)
        self.assertEqual(report['sequences'], 2)
        self.assertEqual(report['window_mean']['error'], 4.)
        self.assertEqual(report['sequence_mean']['error'], 5.)
        self.assertEqual(summarize([]), {'windows': 0, 'sequences': 0})


if __name__ == '__main__':
    unittest.main()
