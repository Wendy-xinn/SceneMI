"""Bounded, ordered CPU prefetch with a checkpointable *consumed* RNG state."""
import copy
import queue
import threading
import time

import numpy as np


class OrderedPrefetch:
    """One producer owns its RNG; speculative batches never advance the consumer.

    Dataset loading must not touch torch/global RNGs. On resume, discard queued
    work and restart from ``consumer_rng``: the next batch is exactly replayed.
    """
    def __init__(self, build, consumer_rng, depth=2):
        if type(depth) is not int or depth < 0:
            raise ValueError('Prefetch depth must be a nonnegative integer')
        self.build, self.consumer_rng, self.depth = build, consumer_rng, depth
        self.stop = threading.Event()
        self.items = queue.Queue(maxsize=max(depth, 1))
        self.worker = None
        self.producer_rng = np.random.default_rng()
        self.producer_rng.bit_generator.state = copy.deepcopy(consumer_rng.bit_generator.state)
        if depth:
            self.worker = threading.Thread(target=self._produce, name='scene-data-prefetch', daemon=True)
            self.worker.start()

    def _put(self, value):
        while not self.stop.is_set():
            try:
                self.items.put(value, timeout=.1)
                return
            except queue.Full:
                pass

    def _produce(self):
        try:
            while not self.stop.is_set():
                start = time.perf_counter()
                value = self.build(self.producer_rng)
                self._put((value, copy.deepcopy(self.producer_rng.bit_generator.state),
                           time.perf_counter()-start, None))
        except BaseException as error:
            self._put((None, None, 0., error))

    def next(self):
        start = time.perf_counter()
        if not self.depth:
            value = self.build(self.consumer_rng)
            elapsed = time.perf_counter()-start
            return value, dict(data_wait_seconds=elapsed, data_prepare_seconds=elapsed)
        value, state, prepared, error = self.items.get()
        if error is not None:
            raise error
        self.consumer_rng.bit_generator.state = state
        return value, dict(data_wait_seconds=time.perf_counter()-start,
                           data_prepare_seconds=prepared)

    def close(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
