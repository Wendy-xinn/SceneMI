import copy
import numpy as np
import pytest

from scenemi.prefetch import OrderedPrefetch


def test_speculative_loading_preserves_next_batch_after_resume():
    rng=np.random.default_rng(31);reference=np.random.default_rng(31)
    build=lambda r:r.integers(0,1000000,size=16).tolist()
    with OrderedPrefetch(build,rng,2) as queue:
        for _ in range(3):assert queue.next()[0]==build(reference)
        state=copy.deepcopy(rng.bit_generator.state)
        expected=queue.next()[0]
    restored=np.random.default_rng();restored.bit_generator.state=state
    with OrderedPrefetch(build,restored,2) as queue:assert queue.next()[0]==expected


def test_worker_error_reaches_consumer():
    def fail(_):raise ValueError('bad source identity')
    with OrderedPrefetch(fail,np.random.default_rng(),1) as queue:
        with pytest.raises(ValueError,match='source identity'):queue.next()
