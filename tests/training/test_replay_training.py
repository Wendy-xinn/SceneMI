import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from scenemi.replay.data import CORPORA
from scenemi.replay.screen import compare
from scenemi.replay.stream import add_noise, stream_requests
from scenemi.replay.training import OnlineStream, configuration
from scenemi.replay.validation import family_mean, select, EXPECTED


class Sampler:
    def sample(self, request):
        rng = np.random.default_rng(request['seed'])
        return dict(motion=rng.normal(size=(128, 201)).astype(np.float32)), request


class FormalReplayTests(unittest.TestCase):
    def test_online_stream_matches_screened_examples_and_restart(self):
        outputs = []
        for workers in (1, 4):
            loader = OnlineStream(Sampler(), 72, 128, workers)
            try:
                loader.get(0, next_start=128)
                outputs.append(loader.get(128))
            finally:
                loader.close()
        for a, b in zip(outputs[0][:2], outputs[1][:2]):
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]))
        self.assertEqual(outputs[0][2], outputs[1][2])
        pairs = [add_noise(Sampler().sample(r), r, 72) for r in stream_requests(72, 128, 128)]
        for offset, actual in enumerate(outputs[0][:2]):
            for key in actual:
                expected = torch.from_numpy(np.stack([p[0][key] for p in pairs[offset::2]]))
                self.assertTrue(torch.equal(actual[key], expected))
        loader = OnlineStream(Sampler(), 72, 128, 2)
        try:
            restarted = loader.get(128)
        finally:
            loader.close()
        self.assertEqual(restarted[2], outputs[0][2])
        self.assertTrue(torch.equal(restarted[0]['motion'], outputs[0][0]['motion']))

    def test_family_average_does_not_overweight_long_recordings(self):
        rows = [dict(identity=dict(family='long'), metrics=dict(error=10.)) for _ in range(20)]
        rows.append(dict(identity=dict(family='short'), metrics=dict(error=30.)))
        self.assertEqual(family_mean(rows)['error'], 20.)

    def test_complete_selection_keeps_original_retention_gates(self):
        c = yaml.safe_load(Path('configs/scene_joint128_replay.yaml').read_text())
        result = dict(ego={s: dict(repair=dict(mpjpe_mm=100.)) for s in CORPORA}, exo={}, identity_max_change_mm=0.)
        for condition in ('tracking', 'occlusion', 'gap'):
            result['exo'][condition] = {s: {v: dict(mpjpe_mm=100., root_error_mm=50.,
                tracked_visible_mpjpe_mm=75., repair_region_mpjpe_mm=120.) for v in ('repair', 'input')} for s in CORPORA}
        self.assertTrue(select(result, result, c)['selection_eligible'])
        changed = copy.deepcopy(result)
        changed['exo']['gap']['amass']['repair']['repair_region_mpjpe_mm'] += 2
        self.assertFalse(select(changed, result, c)['selection_eligible'])
        self.assertEqual(select(changed, result, c)['checks'], compare(changed, result)['checks'])
        changed = copy.deepcopy(result)
        changed['ego']['amass']['repair']['mpjpe_mm'] += 6
        self.assertFalse(select(changed, result, c)['selection_eligible'])

    def test_fixed_protocol_budget_and_all_modes(self):
        c = configuration(Path('configs/scene_joint128_replay.yaml'))
        self.assertEqual(c['training']['updates']*128, 256000)
        self.assertEqual(sum(EXPECTED['ego'].values())+4*sum(EXPECTED['exo'].values()), 28741)
        c['training']['updates'] -= 1
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.yaml'
            path.write_text(yaml.safe_dump(c))
            with self.assertRaises(ValueError):
                configuration(path)


if __name__ == '__main__':
    unittest.main()
