"""Protocol guards independent of the machine's AMASS/body-model paths."""
import unittest
from copy import deepcopy
from unittest.mock import patch

from scenemi.joint_training import configuration, CONFIG, selection_result


class JointProtocolTests(unittest.TestCase):
    def test_equal_exposure_requires_110k_with_four_examples_per_mode(self):
        warm=configuration(CONFIG,'warm');scratch=configuration(CONFIG,'scratch')
        self.assertEqual(scratch['updates']*4,55000*8)
        self.assertEqual(warm['updates'],20000)
        self.assertEqual(warm['training']['sample_order'],['ego','exo']*4)
        self.assertEqual(scratch['initialization'],'from_scratch_no_pretrained_weights')

    def test_lower_combined_score_cannot_hide_ego_regression(self):
        config=configuration(CONFIG,'warm')
        baseline_ego={'family_macro_mean':{'mpjpe_mm':100.}}
        metrics=dict(mpjpe_mm=50.,root_error_mm=40.,tracked_visible_mpjpe_mm=50.,
                     repair_region_mpjpe_mm=100.,max_change_from_initial_mm=0.)
        exo={'conditions':{c:{v:{'family_macro_mean':deepcopy(metrics)}
            for v in ('initial','repair')} for c in ('identity','tracking','occlusion','gap')}}
        baseline_exo=deepcopy(exo)
        for c in ('tracking','occlusion','gap'):
            exo['conditions'][c]['repair']['family_macro_mean']['repair_region_mpjpe_mm']=80.
        with patch('pathlib.Path.read_text',side_effect=[__import__('json').dumps(baseline_ego),
                                                       __import__('json').dumps(baseline_exo)]):
            result=selection_result({'family_macro_mean':{'mpjpe_mm':101.}},exo,config,5000)
        self.assertLess(result['selection_score'],1.)
        self.assertFalse(result['checks']['ego_parent_noninferiority'])
        self.assertFalse(result['selection_eligible'])


if __name__=='__main__':unittest.main()
