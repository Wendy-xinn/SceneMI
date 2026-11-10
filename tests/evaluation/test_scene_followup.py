import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch

from scenemi.scene_followup import change_scene, donor_indices, geometry_indices, read_ledger
from scenemi.scene_evaluation import SCENE_KEYS
from scenemi.geometry.scene_quality import supported_stance, masked_mean


class FollowupTests(unittest.TestCase):
    def test_empty_scene_removes_all_geometry_without_touching_camera_or_labels(self):
        batch={key:torch.ones(2,3) for key in SCENE_KEYS}
        batch['local_history_valid']=torch.ones(2,dtype=torch.bool)
        batch['camera']=torch.randn(2,9);batch['target_motion']=torch.randn(2,201)
        original={key:value.clone() for key,value in batch.items()}
        changed=change_scene(batch,'empty')
        for key in ('camera','target_motion'):self.assertIs(changed[key],batch[key])
        for key in SCENE_KEYS:
            expected=-1 if key.endswith('owner') else 0
            self.assertTrue((changed[key]==expected).all())
        for key in batch:self.assertTrue(torch.equal(batch[key],original[key]))

    def test_mismatch_moves_consistent_scene_bundle_and_never_donor_person(self):
        a={key:torch.zeros(2,3) for key in SCENE_KEYS};a['camera']=torch.zeros(2,9)
        b={key:torch.ones(2,3) for key in SCENE_KEYS};b['camera']=torch.ones(2,9)
        changed=change_scene(a,'mismatched',b)
        self.assertIs(changed['camera'],a['camera'])
        for key in SCENE_KEYS:self.assertIs(changed[key],b[key])
        with self.assertRaises(ValueError):change_scene(a,'mismatched')

    def test_donors_cannot_share_family_or_cross_action(self):
        rows=[dict(identity=dict(dataset='grail',category=c,family=f))
              for c in ('sitting','slope') for f in ('a','a','b','c')]
        for i,j in enumerate(donor_indices(rows)):
            a,b=rows[i]['identity'],rows[j]['identity']
            self.assertEqual(a['category'],b['category']);self.assertNotEqual(a['family'],b['family'])
        with self.assertRaises(ValueError):donor_indices(rows[:2])

    def test_geometry_cases_are_seeded_balanced_and_family_distinct(self):
        rows=[dict(identity=dict(dataset='grail',category=c,family=f))
              for c in ('curb','sitting','slope','stair_p1','stair_p2') for f in ('a','a','b','c')]
        policy=dict(geometry_seed=41,geometry_cases_per_category=2)
        ids=geometry_indices(rows,policy)
        self.assertEqual(ids,geometry_indices(rows,policy));self.assertEqual(len(ids),10)
        for c in ('curb','sitting','slope','stair_p1','stair_p2'):
            self.assertEqual(len({rows[i]['identity']['family'] for i in ids if rows[i]['identity']['category']==c}),2)

    def test_support_masks_reject_swing_fast_motion_and_single_frame_grazes(self):
        distance=np.zeros((9,2));centres=np.zeros((9,2,3))
        distance[:,1]=.1;distance[4,0]=.1
        mask=supported_stance(distance,centres)
        self.assertFalse(mask[:,1].any());self.assertFalse(mask[3:6,0].any())
        self.assertTrue(mask[2,0]);self.assertTrue(mask[7,0])
        centres[:,0,0]=np.arange(9)*.1
        self.assertFalse(supported_stance(np.zeros((9,2)),centres)[:,0].any())
        self.assertIsNone(masked_mean(distance,np.zeros_like(distance,dtype=bool)))

    def test_interrupted_final_write_can_resume_without_masking_earlier_corruption(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'metrics.jsonl';path.write_text('{"case_id":0}\n{"case_id":')
            self.assertEqual(read_ledger(path),[dict(case_id=0)])
            self.assertEqual(len(list(Path(folder).glob('interrupted_metrics_tail_*'))),1)
            self.assertEqual(read_ledger(path),[dict(case_id=0)])
            path.write_text('{"case_id":0}')
            self.assertEqual(read_ledger(path),[dict(case_id=0)])
            self.assertTrue(path.read_text().endswith('\n'))
            path.write_text('{broken}\n{"case_id":1}\n')
            with self.assertRaises(json.JSONDecodeError):read_ledger(path)


if __name__=='__main__':unittest.main()
