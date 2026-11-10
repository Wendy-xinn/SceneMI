"""Trainable observation encoder used by the public motion pipeline.

This module contains feature extraction only.  It never receives target
motion, target body shape, or scene geometry as an input.
"""
import torch
from torch import nn

from scenemi.models import ObservationShapeEncoder


class NativeObservationEncoder(ObservationShapeEncoder):
    """Encode the camera/joint observation stream into one window feature."""

    def features(self, observations):
        trajectory, meta = observations['trajectory'], observations['observation_meta']
        values = torch.cat((trajectory[..., :3] * meta[..., 2:3],
                            trajectory[..., 3:] * meta[..., 3:4], meta[..., :4],
                            observations['observation_uncertainty']), -1)
        active = meta[..., :2].amax(-1) > 0
        encoded = self.joints(values) * active[..., None]
        ca, cc = observations['camera_available'], observations['camera_confidence']
        camera = torch.cat((observations['camera'], ca, cc), -1)
        frame_valid = active.any(-1) | (ca > 0).any(-1)
        x = self.body(encoded.flatten(2)) + self.camera(camera) * (ca > 0).any(-1)[..., None]
        x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        weights = frame_valid[..., None].to(x)
        mean = (x * weights).sum(1) / weights.sum(1).clamp_min(1)
        maximum = x.masked_fill(~frame_valid[..., None], -1e9).amax(1)
        maximum = torch.where(frame_valid.any(1)[:, None], maximum, 0.)
        return torch.cat((mean, maximum), -1)

    def forward(self, observations):
        return self.readout(self.features(observations))
