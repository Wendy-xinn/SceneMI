"""Physical reference checks admit FP32 roundoff and reject actual changes."""
import unittest

import numpy as np
import torch

from . import exo_control_recheck as audit
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion


class PhysicalReferenceTests(unittest.TestCase):
    def motion(self):
        rest = torch.zeros(1, 22, 3)
        rest[0, :, 1] = torch.arange(22).float() / 30
        motion = reconcile_motion(torch.zeros(1, 3, 201), rest)
        return motion, rest

    def difference(self, a, b, rest):
        return audit.physical_reference_difference(a[0].numpy(), b[0].numpy(), rest[0].numpy())

    def test_physical_roundoff_passes_but_real_translation_fails(self):
        motion, rest = self.motion()
        tiny = reconcile_motion(motion, rest, torch.full((1, 3, 3), 2e-6))
        values = self.difference(motion, tiny, rest)
        audit.require_physical_bounds(values)
        self.assertLess(values['max_root_translation_difference_mm'], .01)
        moved = reconcile_motion(motion, rest, torch.full((1, 3, 3), 1e-3))
        with self.assertRaisesRegex(ValueError, 'exceeds bound'):
            audit.require_physical_bounds(self.difference(motion, moved, rest))

    def test_small_rotation_admitted_but_terminal_joint_rotation_still_guarded(self):
        motion, rest = self.motion()
        rotation = torch.zeros(1, 3, 22, 3)
        rotation[:, :, 21, 2] = np.deg2rad(.005)
        tiny = reconcile_motion(motion, rest, rotation_axis_angle=rotation)
        values = self.difference(motion, tiny, rest)
        audit.require_physical_bounds(values)
        self.assertAlmostEqual(values['max_global_orientation_difference_deg'], .005, places=5)
        rotation[:, :, 21, 2] = np.deg2rad(.02)
        changed = reconcile_motion(motion, rest, rotation_axis_angle=rotation)
        with self.assertRaisesRegex(ValueError, 'orientation'):
            audit.require_physical_bounds(self.difference(motion, changed, rest))

    def test_embedded_fk_channels_and_finite_values_are_independently_checked(self):
        motion, rest = self.motion()
        bad = motion.clone()
        bad[:, :, 135] += .001
        with self.assertRaisesRegex(ValueError, 'embedded'):
            audit.require_physical_bounds(self.difference(motion, bad, rest))
        bad[:, :, 135] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.difference(motion, bad, rest)


if __name__ == '__main__':
    unittest.main()
