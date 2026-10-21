"""Strict admission, saved-source mismatch and durable-resume tests (CPU only)."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import final_metric_recheck as audit


class ExactInputAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entry = dict(case=7, mode='exo', dataset='rich', recording='r', sequence_id='r', start=0,
                          source_first=5, case_seed=91)
        self.plan = dict(protocol=audit.final.PROTOCOL, state='frozen', split='full-validation',
            scene_variants=['scene_on', 'scene_off'], regimes=['noisy'], planned_predictions=2,
            unique_mode_windows=1, cases=[self.entry], selection=dict(checkpoint_sha256='c'*64),
            selection_manifest_sha256='s'*64)
        self.sample = dict(motion=np.zeros((3,201), np.float32), rest=np.ones((22,3), np.float32))
        self.observed = dict(trajectory=np.ones((3,22,9), np.float32),
                             observation_meta=np.ones((3,22,4), np.float32))
        self.reference = np.zeros((3,201), np.float32)
        self.identity = dict(sequence_id='r', source_start_30fps=5, mode='exo')
        self.rows = []
        for scene in audit.final.SCENES:
            path = self.root/(scene+'.npz')
            np.savez(path, truth=self.sample['motion'], rest=self.sample['rest'],
                     observations=self.observed['trajectory'], observation_meta=self.observed['observation_meta'],
                     reference_motion=self.reference, identity_json=np.array(json.dumps(self.identity)),
                     input_sha256=np.array('i'*64), checkpoint_sha256=np.array('c'*64),
                     case_seed=np.array(91), use_scene=np.array(scene=='scene_on'))
            self.rows.append(dict(**{k:self.entry[k] for k in ('case',*audit.IDENTITY_KEYS)},
                seed=91, regime='noisy', scene_variant=scene, input_sha256='i'*64,
                selection_sha256='s'*64, checkpoint_sha256='c'*64,
                artifact=str(path), artifact_sha256=audit.sha256(path)))

    def compare(self, **changes):
        values=dict(entry=self.entry, regime='noisy', pair=self.rows, sample=self.sample,
                    identity=self.identity, observed=self.observed, reference=self.reference, input_sha256='i'*64)
        values.update(changes)
        return audit.compare_saved(**values)

    def test_complete_pair_is_admitted_but_missing_duplicate_and_test_are_rejected(self):
        self.assertEqual(len(audit.admit_rows(self.plan,self.rows)),1)
        for rows in (self.rows[:1], [self.rows[0],self.rows[0]]):
            with self.assertRaises(ValueError):audit.admit_rows(self.plan,rows)
        with self.assertRaises(ValueError):audit.admit_rows(dict(self.plan,split='rich-test'),self.rows)

    def test_original_identity_seed_model_and_scene_input_are_bound(self):
        for key,value in (('seed',92),('start',64),('checkpoint_sha256','x'*64),('input_sha256','x'*64)):
            rows=copy.deepcopy(self.rows);rows[1][key]=value
            with self.assertRaises(ValueError):audit.admit_rows(self.plan,rows)

    def test_exact_sources_pass_without_a_prediction_member(self):
        result=self.compare()
        self.assertEqual(result['status'],'exact_pass')
        self.assertTrue(result['source_arrays_exact'])
        self.assertEqual(result['checked_scene_artifacts'],2)

    def test_one_ulp_reference_and_raw_input_mismatches_are_not_tolerated(self):
        reference=self.reference.copy();reference[1,3]=np.nextafter(np.float32(0),np.float32(1))
        with self.assertRaisesRegex(ValueError,'reference_motion: exact mismatch'):self.compare(reference=reference)
        observed=copy.deepcopy(self.observed);observed['observation_meta'][0,0,0]=0
        with self.assertRaisesRegex(ValueError,'observation_meta: exact mismatch'):self.compare(observed=observed)
        with self.assertRaisesRegex(ValueError,'fingerprint'):self.compare(input_sha256='x'*64)

    def test_artifact_change_is_rejected_before_array_comparison(self):
        Path(self.rows[0]['artifact']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'artifact changed'):self.compare()

    def test_resume_accepts_only_exact_bound_records_and_recovers_own_tail(self):
        row=self.compare();row['audit_plan_sha256']='a'*64
        path=self.root/'audit.jsonl';audit.append_record(path,row)
        expected={(7,'noisy'):(self.entry,self.rows)}
        with path.open('ab') as stream:stream.write(b'{"interrupted":')
        kept=audit.resume_rows(path,'a'*64,expected,resume=True)
        self.assertEqual(set(kept),{(7,'noisy')})
        self.assertTrue(path.read_bytes().endswith(b'\n'))
        self.assertTrue((self.root/'TAIL_RECOVERY.json').exists())
        with self.assertRaises(ValueError):audit.resume_rows(path,'b'*64,expected,resume=True)
        with self.assertRaises(ValueError):audit.resume_rows(path,'a'*64,expected,resume=False)
        audit.append_record(path,row)
        with self.assertRaisesRegex(ValueError,'duplicate'):audit.resume_rows(path,'a'*64,expected,resume=True)


if __name__ == '__main__':unittest.main()
