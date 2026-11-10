import unittest
import numpy as np

from scenemi.scene_repair_training import StageDataset, paired_dataset_interval


class NativeRepairStageTests(unittest.TestCase):
    def test_validation_windows_do_not_cross_rejected_motion_ranges(self):
        dataset=object.__new__(StageDataset)
        accepted=np.r_[np.arange(10,270),np.arange(450,460)]
        dataset.records=[dict(starts=accepted,dataset='grail',family='example')]
        cases=dataset.validation_cases()
        self.assertEqual([c['start'] for c in cases],[10,138,266,269,450,459])
        self.assertTrue(all(c['start'] in accepted for c in cases))

    def test_large_corpus_does_not_dominate_paired_interval(self):
        rows=[]
        for corpus,families,delta in [('egobody',1,0.),('trumans',1,0.),('grail',20,9.)]:
            for family in range(families):
                for variant,value in [('initial',20.),('repair',20.+delta)]:
                    rows.append(dict(condition='gap',variant=variant,identity={'dataset':corpus},family=str(family),
                                     metrics={'repair_region_mpjpe_mm':value}))
        result=paired_dataset_interval(rows,'gap','initial',np.random.default_rng(123))
        self.assertEqual(result['delta_repair_minus_control_mm'],3.)
        self.assertEqual(result['percentile_95_interval_mm'],[3.,3.])


if __name__=='__main__':unittest.main()
