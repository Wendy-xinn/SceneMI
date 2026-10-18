"""CPU checks for immutable three-model EXO comparison and future registration."""
import copy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import adapter_evaluation as review


class CohortTests(unittest.TestCase):
    def test_real_frozen_cohort_scope_and_model_interleaving(self):
        cohort=review.read(review.DEFAULT_OUT/'COHORT.json');entries=cohort['entries']
        self.assertEqual(len(entries),64)
        self.assertEqual(len({(r['dataset'],r['recording']) for r in entries}),64)
        self.assertEqual(sum(r['panel']=='new55' for r in entries),55)
        self.assertTrue(all(r['dataset']=='rich' and r['panel']=='new55' for r in entries[:24]))
        schedule=review.schedule(entries,list(review.LABELS))
        self.assertEqual(len(schedule),384)
        self.assertEqual([(r['case'],regime,label) for r,regime,label in schedule[:3]],
                         [(0,'clean',label) for label in review.LABELS])

    def test_inference_signature_allows_training_policy_but_rejects_physics(self):
        config=review.common.configurations(review.DEFAULT_OUT)
        adapter=copy.deepcopy(config);adapter.update(training_mode='exo',adapters={'rank':16},optimizer={'lr':1e-4},adapter_stage={'quotas':'balanced'})
        self.assertEqual(review.inference_signature(config),review.inference_signature(adapter))
        adapter['physical_guidance']['enabled']=False
        self.assertNotEqual(review.inference_signature(config),review.inference_signature(adapter))


class ModelRegistrationTests(unittest.TestCase):
    def test_future_adapter_requires_final_CPU_report_and_original_base(self):
        config={'frames':192};declaration=dict(run='/stage',checkpoint='/stage/training/last.pt',declared_config_sha256='same')
        check=dict(status='independent_CPU_primary_metrics_passed',EGO_serialized_outputs_exact=True,CUDA_initialized=False,
                   frozen_base_sha256=review.BASE_HASH,final_checkpoint_sha256='same')
        effective=dict(config=config,code_sha256={'/source.py':'same'})
        def documents(path):
            return check if Path(path).name=='CPU_RECHECK.json' else effective if Path(path).name=='effective_config.json' else config
        with mock.patch.object(review,'sha256',return_value='same'),mock.patch.object(review,'read',side_effect=documents):
            digest,evidence=review.verify_adapter_record(declaration,config)
            self.assertEqual(digest,'same');self.assertIn('stage_source_sha256',evidence)
            for key,value in [('status','sampling'),('EGO_serialized_outputs_exact',False),('CUDA_initialized',True),('frozen_base_sha256','other'),('final_checkpoint_sha256','other')]:
                original=check[key];check[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):review.verify_adapter_record(declaration,config)
                check[key]=original

    def test_registered_weights_cannot_change_after_future_is_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'candidate.pt';path.write_bytes(b'exact-final-model')
            declared=dict(checkpoint=str(path),checkpoint_sha256=review.sha256(path))
            review.verify_registered(declared)
            path.write_bytes(b'later-model')
            with self.assertRaises(ValueError):review.verify_registered(declared)

    def test_resume_can_contain_only_base_v1_but_not_unregistered_v2(self):
        entry=review.read(review.DEFAULT_OUT/'COHORT.json')['entries'][0]
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);folder=out/'motions';folder.mkdir()
            registry={label:dict(checkpoint_sha256=label,expected_step=41000 if label=='base41k' else 1000) for label in review.LABELS[:2]}
            artifact=folder/'case0000_clean_base41k.npz';artifact.write_bytes(b'motion')
            row=dict(case=0,regime='clean',model='base41k',seed=entry['case_seed'],checkpoint_step=41000,checkpoint_sha256='base41k',
                artifact=str(artifact),artifact_sha256=review.sha256(artifact),panel=entry['panel'],**{k:entry[k] for k in ('dataset','group','recording','sequence_id','source_first','start')})
            (out/'rows.jsonl').write_text(json.dumps(row)+'\n')
            self.assertEqual(review.committed_rows(out,{},[entry],registry),[row])
            row['panel']='old9';(out/'rows.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):review.committed_rows(out,{},[entry],registry)


class AggregationTests(unittest.TestCase):
    def row(self,label,case,dataset,panel,hidden,count):
        return dict(model=label,case=case,dataset=dataset,recording=f'{dataset}{case}',panel=panel,regime='clean',
            input_sha256=f'same{case}',metrics=dict(hidden_mpjpe_mm=hidden,w_mpjpe_mm=hidden,visible_mpjpe_mm=1.,foot_joint_speed_gt_low_m_s=.1),
            counts=dict(hidden_mpjpe_mm=count,w_mpjpe_mm=4224,visible_mpjpe_mm=20,foot_joint_speed_gt_low_m_s=30))

    def test_old9_never_leaks_into_new55_and_macro_equalizes_recordings(self):
        rows=[self.row('base41k',0,'rich','new55',10,1),self.row('base41k',1,'rich','new55',30,3),
              self.row('base41k',2,'trumans','new55',40,1),self.row('base41k',3,'egobody','new55',60,1),
              self.row('base41k',4,'rich','old9',1000,1)]
        groups,macro=review.aggregate(rows)
        rich=groups['new55/base41k/rich/clean']
        self.assertEqual(rich['windows'],2)
        self.assertEqual(rich['metrics']['hidden_mpjpe_mm']['recording_macro'],20)
        self.assertEqual(rich['metrics']['hidden_mpjpe_mm']['count_weighted'],25)
        self.assertEqual(macro['new55/base41k/clean']['hidden_mpjpe_mm'],40)
        self.assertEqual(groups['old9/base41k/rich/clean']['windows'],1)

    def test_per_recording_comparison_rejects_different_inputs_or_counts(self):
        a=self.row('base41k',0,'rich','new55',100,3);b=self.row('v1adapter1000',0,'rich','new55',70,3)
        groups,macro=review.aggregate([a,b]);paired,guards=review.comparisons([a,b],groups,macro)
        self.assertEqual(paired[0]['metrics']['hidden_mpjpe_mm']['delta'],-30)
        self.assertTrue(next(g for g in guards if g['metric']=='hidden_mpjpe_mm')['passed'])
        b['input_sha256']='changed'
        with self.assertRaises(ValueError):review.comparisons([a,b],groups,macro)


class LifecycleTests(unittest.TestCase):
    def test_complete_full_job_starts_fresh_CPU_only_after_unlock(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'PLAN.json').write_text('{}')
            def child(command,**kwargs):
                with (out/'job.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertEqual(command[3],'recheck')
            with mock.patch.object(review,'evaluate',return_value=True),mock.patch.object(review.subprocess,'run',side_effect=child):
                review.main(['run','--out',str(out)])

    def test_subset_completion_does_not_claim_full_or_start_CPU(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'PLAN.json').write_text('{}')
            with mock.patch.object(review,'evaluate',return_value=False) as evaluate,mock.patch.object(review.subprocess,'run') as child:
                review.main(['run','--out',str(out),'--models','base41k','v1adapter1000'])
                self.assertEqual(evaluate.call_args.args[1],list(review.LABELS[:2]));child.assert_not_called()

    def test_competing_command_does_not_replace_active_status(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);review.write(out/'STATUS.json',dict(status='sampling'))
            with (out/'job.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):review.main(['resume','--out',str(out)])
            self.assertEqual(review.read(out/'STATUS.json')['status'],'sampling')


if __name__=='__main__':unittest.main()
