import copy
import unittest

import numpy as np
import torch
from torch import nn

from scenemi.native_joint_data import ego_payload, validation_cases
from scenemi.native_joint_training import CONFIG, NativeMotionPrior, configuration, configure_trainable, selection_result
from scenemi.observations import INPUT_KEYS


class NativeJointTests(unittest.TestCase):
    def test_scene_free_model_rejects_scene_before_forward(self):
        model=object.__new__(NativeMotionPrior)
        with self.assertRaisesRegex(ValueError,'no scene input'):
            model.forward(None,None,None,use_scene=True)

    def test_accepted_intervals_tile_without_crossing_rejected_starts(self):
        cases = validation_cases([dict(starts=np.r_[np.arange(10,270),np.arange(450,460)],
                                       dataset='grail',family='stairs')])
        self.assertEqual([c['start'] for c in cases], [10,138,266,269,450,459])

    def test_camera_only_payload_excludes_native_targets_and_scene(self):
        batch = {k:object() for k in INPUT_KEYS+('body_format','sensor_profile')}
        batch.update(target_rest=float('nan'),target_native_betas=float('nan'),occupancy=float('nan'))
        self.assertEqual(set(ego_payload(batch)),set(INPUT_KEYS+('body_format','sensor_profile')))

    def test_partial_unfreeze_preserves_encoder_and_scene_freeze(self):
        model = nn.Module();model.core=nn.Module();model.core.unet=nn.Module()
        model.core.unet.downs=nn.Linear(1,1);model.core.unet.ups=nn.Linear(1,1)
        model.native_body=nn.Module();model.native_body.coefficients=nn.Linear(1,1)
        model.patch_projection=nn.Linear(1,1)
        configure_trainable(model,configuration(CONFIG))
        self.assertFalse(model.core.unet.downs.weight.requires_grad)
        self.assertTrue(model.core.unet.ups.weight.requires_grad)
        self.assertTrue(model.native_body.coefficients.weight.requires_grad)
        self.assertFalse(model.patch_projection.weight.requires_grad)

    def test_combined_improvement_cannot_hide_one_ego_dataset_regression(self):
        c=configuration(CONFIG)
        ego=dict(dataset_macro_mean={'mpjpe_mm':100.},by_dataset={k:{'family_macro_mean':{'mpjpe_mm':100.}}
                 for k in ('egobody','trumans','grail')})
        metrics=dict(mpjpe_mm=50.,root_error_mm=40.,tracked_visible_mpjpe_mm=50.,
                     repair_region_mpjpe_mm=100.,max_change_from_initial_mm=0.)
        summary=dict(family_macro_mean=copy.deepcopy(metrics),by_dataset={k:{'family_macro_mean':copy.deepcopy(metrics)}
                     for k in ('egobody','trumans','grail')})
        exo={'conditions':{v:{name:copy.deepcopy(summary) for name in ('initial','repair')}
                           for v in ('identity','tracking','occlusion','gap')}}
        amass={'family_macro_mean':{'mpjpe_mm':153.}}
        baseline=copy.deepcopy(dict(ego=ego,exo=exo,amass_ego=amass))
        ego['by_dataset']['trumans']['family_macro_mean']['mpjpe_mm']=101.
        ego['dataset_macro_mean']['mpjpe_mm']=80.
        result=selection_result(ego,exo,amass,baseline,c,6000)
        self.assertLess(result['selection_score'],1.)
        self.assertFalse(result['selection_eligible'])
        self.assertFalse(result['checks']['ego_trumans_noninferiority'])


if __name__=='__main__':unittest.main()
