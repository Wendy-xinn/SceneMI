"""Tests for the retained native30 implementation and its exact old fixtures."""
from pathlib import Path

LEGACY_ROOT = Path(__file__).resolve().parents[2] / 'experiments/offline_camera_retrain_v1'
FIXTURES = Path(__file__).with_name('fixtures')
