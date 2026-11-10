"""Linear-memory DBSCAN for the upstream one-dimensional foot-height query.

The sorted core components and original-input seed order reproduce DBSCAN's
border assignment. No pairwise neighborhood graph is materialized.
"""
import numpy as np


class HeightDBSCAN:
    """The finite, unweighted, one-dimensional Euclidean case used by 0a."""
    def __init__(self, eps=.5, min_samples=5):
        if not np.isfinite(eps) or eps <= 0 or int(min_samples) != min_samples or min_samples < 1:
            raise ValueError('Invalid height clustering parameters')
        self.eps, self.min_samples = float(eps), int(min_samples)

    def fit(self, points):
        x = np.asarray(points, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 1 or not len(x) or not np.isfinite(x).all():
            raise ValueError('HeightDBSCAN requires nonempty finite (N, 1) heights')
        order = np.argsort(x[:, 0], kind='stable')
        values = x[order, 0]
        # Use the distance comparison itself at both boundaries. x +/- eps
        # can round differently from the KDTree's |x_i-x_j| <= eps query.
        left = np.empty(len(x), np.int64)
        right = np.empty(len(x), np.int64)
        lo = hi = 0
        for i, value in enumerate(values):
            while value-values[lo] > self.eps:
                lo += 1
            while hi < len(x) and values[hi]-value <= self.eps:
                hi += 1
            left[i], right[i] = lo, hi
        core = right-left >= self.min_samples
        positions = np.flatnonzero(core)
        labels = np.full(len(x), -1, np.intp)
        if len(positions):
            component = np.r_[0, np.cumsum(np.diff(values[positions]) > self.eps)]
            seeds = np.full(int(component[-1])+1, len(x), np.int64)
            np.minimum.at(seeds, component, order[positions])
            seed_order = np.argsort(seeds)
            numbering = np.empty(len(seeds), np.intp)
            numbering[seed_order] = np.arange(len(seeds))
            labels[positions] = numbering[component]
            # A noncore point can touch two components. Upstream visits the
            # component with the earliest original core seed first.
            noncore = np.flatnonzero(~core)
            insertion = np.searchsorted(positions, noncore)
            for i, at in zip(noncore, insertion):
                neighbors = []
                if at and values[i]-values[positions[at-1]] <= self.eps:
                    neighbors.append(labels[positions[at-1]])
                if at < len(positions) and values[positions[at]]-values[i] <= self.eps:
                    neighbors.append(labels[positions[at]])
                if neighbors:
                    labels[i] = min(neighbors)
        self.labels_ = np.empty(len(x), np.intp)
        self.labels_[order] = labels
        self.core_sample_indices_ = np.sort(order[positions])
        self.components_ = x[self.core_sample_indices_].copy()
        return self
