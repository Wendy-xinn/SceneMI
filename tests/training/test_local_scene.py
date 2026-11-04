"""Input visibility and test identity must survive the new adapters."""
import unittest
import torch

from scenemi.benchmark import validate_cases
from scenemi.local_model import nearest_observed
from scenemi.data.local_scene import LOCAL_INPUT_KEYS


class LocalSceneTests(unittest.TestCase):
    def test_unknown_and_person_points_cannot_trigger_neighbours(self):
        anchors=torch.zeros(2,3,3)
        history=torch.tensor([[0.,0.,0.],[float('nan'),0.,0.]])
        frame=torch.zeros(2,2,3)
        owners=torch.tensor([[100,-1],[100,-1]])
        _,_,valid=nearest_observed(anchors,history,torch.tensor([False,True]),frame,owners,k=3)
        self.assertFalse(valid.any())

    def test_current_frame_never_looks_ahead(self):
        anchors=torch.zeros(2,3,3)
        history=torch.empty(0,3); mask=torch.empty(0,dtype=torch.bool)
        frame=torch.ones(2,2,3)*10
        frame[0,0]=torch.tensor([.1,0.,0.])
        owners=torch.zeros(2,2,dtype=torch.long)
        before=nearest_observed(anchors,history,mask,frame,owners,k=2)
        frame[1]=0
        after=nearest_observed(anchors,history,mask,frame,owners,k=2)
        for left,right in zip(before,after):self.assertTrue(torch.equal(left[0],right[0]))
        self.assertFalse(before[2][1].any());self.assertTrue(after[2][1].any())

    def test_native_labels_are_not_local_inputs(self):
        for key in ('motion','target_rest','target_template','target_native_betas','native_body'):
            self.assertNotIn(key,LOCAL_INPUT_KEYS)

    def test_test_manifest_cannot_be_changed_or_truncated(self):
        original=[dict(corpus='SSM_synced',case_id=0),dict(corpus='Transitions_mocap',case_id=1)]
        validate_cases(original,original)
        with self.assertRaises(ValueError):validate_cases(original[:1],original)
        with self.assertRaises(ValueError):validate_cases([dict(corpus='HumanEva')],[dict(corpus='HumanEva')])


if __name__=='__main__':unittest.main()
