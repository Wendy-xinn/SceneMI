"""Observed surface samples for sparse skin queries, with causal history."""
import numpy as np

from scenemi.data.scene_dataset import SceneDataset, SCENE_INPUT_KEYS, bounded_frame_points

LOCAL_INPUT_KEYS = SCENE_INPUT_KEYS+('local_history_points', 'local_history_valid',
                                  'local_frame_points', 'local_frame_owner')


def inference_payload(batch):
    return {key:batch[key] for key in LOCAL_INPUT_KEYS}


class LocalSceneDataset(SceneDataset):
    def sample(self, rng, **kwargs):
        sample, identity = super().sample(rng, **kwargs)
        record = self.records[identity['member']]
        first = identity['start']
        history = self.prefix_cache[(record['path'], first)]
        synthetic = record.get('synthetic_chair')
        if synthetic:
            with np.load(synthetic['condition_file'], allow_pickle=False) as materialized:
                history = np.concatenate((history, materialized['synthetic_chair_points_m']))
        origin, anchor = np.asarray(identity['origin']), np.asarray(identity['anchor'])
        history = ((history-origin)@anchor.T).astype(np.float32)
        count = self.config['local']['history_points']
        if len(history)>count:
            history = history[np.linspace(0, len(history)-1, count).astype(int)]
        hp = np.zeros((count, 3), np.float32)
        hv = np.zeros(count, bool)
        hp[:len(history)], hv[:len(history)] = history, True
        meta = dict(self.metadata(record), path=record['path'])
        span = slice(first, first+128)
        points, owners = bounded_frame_points(self.array(meta, 'visible_frame_points_scenemi_yup')[span],
            self.array(meta, 'visible_frame_owner')[span], self.config['local']['frame_points'])
        points = ((points-origin)@anchor.T).astype(np.float32)
        points[owners<0] = 0
        sample.update(local_history_points=hp, local_history_valid=hv,
                      local_frame_points=points, local_frame_owner=owners)
        identity.update(local_surface_history_cutoff=first, local_future_history=False,
                        local_surface_samples='observed_points; normals/velocities unavailable')
        return sample, identity
