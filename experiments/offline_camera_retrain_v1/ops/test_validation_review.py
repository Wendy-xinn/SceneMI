"""CPU regression checks for paired VAL identity, metrics and durable resume."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from . import validation_review as review


def fake_cohort():
    config={'validation':{'seed':123}}
    data=SimpleNamespace(split='validation',members={g:[] for g in ('rich','trumans','camera_wearer','interactee')})
    entries=[]
    for group,n in [('rich',27),('trumans',1),('camera_wearer',1)]:
        for i in range(n):
            meta=dict(split='validation',sequence_id=f'{group}-{i}',recording=f'{group}-record{i}',
                      source_first=1000,path=f'/fake/{group}/{i}')
            data.members[group].append((meta,set(range(0,4096,64))))
    def add(group,member,start):
        meta,_=data.members[group][member];source_first=meta['source_first']+start
        seedkey=json.dumps([group,'ego',meta['sequence_id'],source_first],separators=(',',':'))
        seed=(123+int.from_bytes(hashlib.sha256(seedkey.encode()).digest()[:4],'little'))%2**32
        entries.append(dict(case=len(entries),group=group,dataset='egobody' if group=='camera_wearer' else group,
            mode='ego',joints=[15],member=member,start=start,path=meta['path'],source_first=source_first,
            sequence_id=meta['sequence_id'],recording=meta['recording'],case_seed=seed))
    for i in range(27):add('rich',i,0)
    for i in range(179):add('rich',i%27,(i//27+1)*64)
    for group in ('trumans','camera_wearer'):
        for i in range(50):add(group,0,i*64)
    data.starts_for=lambda group,member,mode:data.members[group][member][1]
    return data,entries,config


class CohortIdentityTests(unittest.TestCase):
    def test_registered_counts_and_initial_recording_coverage(self):
        data,entries,config=fake_cohort()
        coverage=review.validate_entries(data,entries,config)
        self.assertEqual(coverage['windows'],dict(rich=206,trumans=50,egobody=50))
        self.assertEqual(coverage['recordings']['rich'],27)
        order=review.schedule(entries)
        self.assertEqual([(e['case'],regime,label) for e,regime,label in order[:4]],
                         [(0,'clean','candidate41k'),(0,'clean','final55k'),(0,'noisy','candidate41k'),(0,'noisy','final55k')])
        self.assertEqual(len(order),1224)

    def test_wrong_path_member_seed_or_clock_cannot_pass(self):
        data,entries,config=fake_cohort()
        for key,value in [('path','/different'),('member',1),('case_seed',0),('source_first',0),('start',1),('dataset','trumans')]:
            with self.subTest(key=key):
                bad=copy.deepcopy(entries);bad[0][key]=value
                with self.assertRaises(ValueError):review.validate_entries(data,bad,config)
        data.split='train'
        with self.assertRaises(ValueError):review.validate_entries(data,entries,config)

    def test_no_duplicate_or_false_initial_recording_coverage(self):
        data,entries,config=fake_cohort()
        bad=copy.deepcopy(entries);bad[28]=dict(bad[27],case=28)
        with self.assertRaises(ValueError):review.validate_entries(data,bad,config)
        bad=copy.deepcopy(entries);bad[26],bad[27]=bad[27],bad[26]
        for i,row in enumerate(bad):row['case']=i
        with self.assertRaises(ValueError):review.validate_entries(data,bad,config)


class DurableResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name)
        self.entry=fake_cohort()[1][0]
        self.plan=dict(model_order=list(review.LABELS),models={label:dict(checkpoint_sha256=label,expected_step=step)
                     for label,step in zip(review.LABELS,(41000,55000))})
        artifact=self.out/'motions/case0000_clean_candidate41k.npz';artifact.parent.mkdir();artifact.write_bytes(b'committed-array-bytes')
        self.row=dict(case=0,regime='clean',model='candidate41k',seed=self.entry['case_seed'],checkpoint_step=41000,
            checkpoint_sha256='candidate41k',artifact=str(artifact),artifact_sha256=review.sha256(artifact),
            **{k:self.entry[k] for k in ('dataset','group','recording','sequence_id','source_first','start')})

    def tearDown(self):self.tmp.cleanup()

    def put(self,rows,tail=b''):
        (self.out/'rows.jsonl').write_bytes(b''.join(json.dumps(row).encode()+b'\n' for row in rows)+tail)

    def test_only_incomplete_last_commit_is_truncated_and_preserved(self):
        tail=b'{"case":1';self.put([self.row],tail)
        self.assertEqual(review.committed_rows(self.out,self.plan,[self.entry]),[self.row])
        self.assertTrue((self.out/'rows.jsonl').read_bytes().endswith(b'\n'))
        digest=hashlib.sha256(tail).hexdigest()
        self.assertEqual((self.out/('TRUNCATED_ROW_TAIL_'+digest+'.bin')).read_bytes(),tail)

    def test_duplicate_identity_checkpoint_and_external_artifact_are_rejected(self):
        self.put([self.row,self.row])
        with self.assertRaises(ValueError):review.committed_rows(self.out,self.plan,[self.entry])
        for key,value in [('seed',1),('checkpoint_step',55000),('checkpoint_sha256','changed'),('artifact','/tmp/outside.npz')]:
            row=dict(self.row);row[key]=value;self.put([row])
            with self.subTest(key=key),self.assertRaises(ValueError):review.committed_rows(self.out,self.plan,[self.entry])
        self.put([self.row]);Path(self.row['artifact']).write_bytes(b'changed')
        with self.assertRaises(ValueError):review.committed_rows(self.out,self.plan,[self.entry])


class FormulaTests(unittest.TestCase):
    def saved(self):
        from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
        rng=np.random.default_rng(13);T=24
        rest=rng.normal(size=(22,3)).astype(np.float32)*.1;rest[0]=0
        truth=np.zeros((T,201),np.float32);prediction=truth.copy();prediction[:,0]=.05
        with torch.no_grad():joints=forward_kinematics(torch.from_numpy(truth)[None],torch.from_numpy(rest)[None])[0].numpy()
        observed=np.zeros((T,22,9),np.float32);observed[:,15,:3]=joints[:,15]/2
        meta=np.zeros((T,22,4),np.float32);meta[:,15]=1
        return dict(prediction=prediction,truth=truth,rest=rest,observations=observed,observation_meta=meta)

    def test_translation_world_root_relative_and_valid_denominators(self):
        saved=self.saved();metrics,counts=review.metrics_for(saved);direct=review.independent_core(saved)
        self.assertAlmostEqual(metrics['w_mpjpe_mm'],100,places=3)
        self.assertLess(metrics['root_relative_mpjpe_mm'],1e-3)
        self.assertLess(metrics['pa_mpjpe_mm'],1e-3)
        self.assertEqual(counts['visible_mpjpe_mm'],24)
        self.assertEqual(counts['hidden_mpjpe_mm'],24*21)
        self.assertEqual(counts['foot_joint_speed_gt_low_m_s'],23*2)
        for key,value in direct.items():self.assertAlmostEqual(value,metrics[key],places=3,msg=key)
        self.assertFalse(torch.cuda.is_initialized())

    def test_no_valid_observation_is_explicit_missing(self):
        saved=self.saved();saved['observation_meta'][:]=0
        metrics,counts=review.metrics_for(saved)
        self.assertIsNone(metrics['visible_mpjpe_mm']);self.assertEqual(counts['visible_mpjpe_mm'],0)


class InputDeviceTests(unittest.TestCase):
    def test_recursive_native_skin_assets_are_checked_before_inference(self):
        known=dict(trajectory=torch.ones(2),skin_assets=[dict(extra_rotations=torch.ones(3),nested=(torch.zeros(1),))],
                   surface_windows=[object()])
        review.assert_input_device(known,'cpu')
        known['skin_assets'][0]['extra_rotations']=torch.empty(3,device='meta')
        with self.assertRaisesRegex(ValueError,'skin_assets.*extra_rotations'):
            review.assert_input_device(known,'cpu')
        self.assertFalse(torch.cuda.is_initialized())

    def test_fingerprint_includes_nested_skinning_tensors(self):
        known=dict(trajectory=torch.ones(2),skin_assets=[dict(extra_rotations=torch.ones(3))])
        first=review.tensor_fingerprint(known)
        known['skin_assets'][0]['extra_rotations'][0]=2
        self.assertNotEqual(first,review.tensor_fingerprint(known))

    def test_builder_device_is_explicit_without_initializing_cuda(self):
        config=review.configurations(review.DEFAULT_OUT)
        self.assertEqual(review.observation_builder(config).skin_cache.device,'cpu')
        self.assertEqual(review.observation_builder(config,device='cuda').skin_cache.device,'cuda')
        self.assertFalse(torch.cuda.is_initialized())


class AggregationTests(unittest.TestCase):
    def row(self,model,case,recording,value,count=1,dataset='rich',regime='clean'):
        return dict(model=model,case=case,recording=recording,dataset=dataset,regime=regime,
            input_sha256='same'+str(case)+regime,metrics=dict(w_mpjpe_mm=value),counts=dict(w_mpjpe_mm=count))

    def test_window_count_weighted_and_recording_macro_are_distinct(self):
        rows=[self.row('candidate41k',0,'A',0,1),self.row('candidate41k',1,'A',100,3),self.row('candidate41k',2,'B',200,2)]
        result=review.aggregate(rows)['candidate41k/rich/clean']['metrics']['w_mpjpe_mm']
        self.assertEqual(result['window_mean'],100)
        self.assertAlmostEqual(result['count_weighted'],700/6)
        self.assertEqual(result['recording_macro'],137.5)
        self.assertEqual(result['recordings'],2)

    def test_equal_dataset_macro_and_pairing_excludes_unpaired_windows(self):
        rows=[]
        for dataset,value in [('rich',10),('trumans',40),('egobody',100)]:
            for label,delta in [('candidate41k',0),('final55k',-1)]:
                rows.append(self.row(label,0,dataset,value+delta,dataset=dataset))
        groups=review.aggregate(rows)
        self.assertEqual(review.equal_dataset_macro(groups)['candidate41k/clean']['w_mpjpe_mm'],50)
        # Case indices are globally unique in real cohorts.
        pair=[self.row('candidate41k',0,'A',100),self.row('final55k',0,'A',80),self.row('candidate41k',1,'A',999)]
        actual=review.paired_recording_differences(pair)
        self.assertEqual(actual[0]['paired_windows'],1)
        self.assertEqual(actual[0]['metrics']['w_mpjpe_mm']['delta_final_minus_candidate'],-20)
        pair[1]['input_sha256']='different'
        with self.assertRaises(ValueError):review.paired_recording_differences(pair)


class LifecycleTests(unittest.TestCase):
    def test_cpu_child_runs_only_after_job_lock_is_released(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'PLAN.json').write_text('{}')
            def child(*args,**kwargs):
                with (out/'job.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertIn('recheck',args[0])
            with mock.patch.object(review,'evaluate',return_value=True) as evaluate, mock.patch.object(review.subprocess,'run',side_effect=child):
                review.main(['run','--out',str(out)])
                self.assertTrue(evaluate.call_args.kwargs['resume'])

    def test_competing_command_does_not_overwrite_active_status(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);review.write(out/'STATUS.json',dict(status='sampling'))
            with (out/'job.lock').open('a') as held:
                fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):review.main(['resume','--out',str(out)])
            self.assertEqual(review.read(out/'STATUS.json')['status'],'sampling')

    def test_real_cohort_config_has_expected_original_protocol(self):
        out=review.DEFAULT_OUT
        config=review.configurations(out);cohort=review.read(out/'COHORT.json')
        self.assertEqual(config['validation']['seed'],2026101351)
        self.assertEqual(len(cohort['entries']),306)
        self.assertEqual(len({row['recording'] for row in cohort['entries'][:27]}),27)
        self.assertFalse(cohort['TEST_used'])


if __name__=='__main__':unittest.main()
