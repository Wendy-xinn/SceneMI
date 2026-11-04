"""Backward-compatible import path for the former scene-model module.

The implementation now lives in :mod:`scenemi.pipeline`.  Keeping these
exports avoids breaking old scripts, saved run source closures, and checkpoint
loading code while the public architecture uses descriptive component names.
"""
from scenemi.pipeline import (
    MotionPipeline,
    NativeScenePrior,
    NativeObservationEncoder,
    native_motion_loss,
)

__all__ = [
    'MotionPipeline', 'NativeScenePrior', 'NativeObservationEncoder',
    'native_motion_loss',
]
