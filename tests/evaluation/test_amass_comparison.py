import copy
import numpy as np
import pytest
import torch

from scenemi.amass_comparison import check_identity, independent_metrics
from scenemi.baseline_evaluation import metric_rows


def test_pins_actual_reflection_and_window():
    case = dict(case_id=0, source='a', family='b', corpus='SSM_synced',
                reflected=False, variant_sha256='abc', start_after_0b=4)
    identity = dict(source='a', family='b', corpus='SSM_synced', reflected=False,
                    variant_sha256='abc', start=4, split='test', mode='ego', yaw=0.)
    check_identity(case, identity)
    for key, bad in [('reflected', True), ('variant_sha256', 'def'), ('start', 5), ('split', 'val')]:
        changed = copy.deepcopy(identity)
        changed[key] = bad
        with pytest.raises(ValueError):
            check_identity(case, changed)


def test_independent_temporal_reductions():
    rng = np.random.default_rng(42)
    gt = rng.normal(0, .002, (2, 128, 22, 3)).astype(np.float32)
    pred = gt + rng.normal(0, .004, gt.shape).astype(np.float32)
    reference = independent_metrics(pred, gt)
    shared = metric_rows(torch.from_numpy(pred), torch.from_numpy(gt))
    for key, values in reference.items():
        np.testing.assert_allclose(values, [r[key] for r in shared], atol=1e-9, rtol=1e-10)
