"""Denominator, pairing and two-comparator gates for the EXO CPU report."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import exo_control_report as report


def row(recording,value,count=1,*,dataset='rich',model='adapter_final',regime='clean',panel='new55',start=0):
    return dict(dataset=dataset,recording=recording,sequence_id=recording,start=start,source_first=start,
        regime=regime,panel=panel,model=model,
        metrics={key:value for key in report.CORE},counts={key:count for key in report.CORE})


def complete(reference=100.,control=120.,final=90.,visible_final=10.):
    rows=[]
    for dataset in report.DATASETS:
        for regime in report.REGIMES:
            for model,value in zip(report.LABELS,(reference,control,final)):
                current=row(dataset,value,dataset=dataset,model=model,regime=regime)
                current['metrics']['visible_mpjpe_mm']=visible_final if model == 'adapter_final' else 10.
                rows.append(current)
    return rows


class ReductionTests(unittest.TestCase):
    def test_recording_macro_and_pool_keep_distinct_denominators(self):
        rows=[row('a',10.,1),row('a',30.,3,start=1),row('b',100.,1),row('empty',None,0)]
        value=report.metric_summary(rows,'hidden_mpjpe_mm')
        self.assertEqual(value['equal_recording'],62.5)
        self.assertEqual(value['pooled'],40.)
        self.assertEqual(value['count'],5)
        self.assertEqual(value['valid_recordings'],2)
        self.assertEqual(value['valid_windows'],3)
        self.assertEqual(value['total_windows'],4)

    def test_missing_is_not_zero_and_inconsistent_counts_fail(self):
        empty=report.metric_summary([row('a',None,0)],'hidden_mpjpe_mm')
        self.assertIsNone(empty['equal_recording']);self.assertIsNone(empty['pooled'])
        for bad in (row('a',None,1),row('a',0.,0),row('a',float('nan'),1),row('a',1.,True)):
            with self.assertRaises(ValueError):
                report.metric_summary([bad],'hidden_mpjpe_mm')

    def test_macro_weights_datasets_not_recording_or_joint_counts(self):
        rows=complete();rows.append(row('rich2',300.,100,dataset='rich'))
        groups,macros=report.aggregate(rows)
        self.assertEqual(groups['all64/rich/clean/adapter_final']['hidden_mpjpe_mm']['equal_recording'],195.)
        self.assertEqual(macros['all64/clean/adapter_final']['hidden_mpjpe_mm']['equal_recording'],125.)
        self.assertNotEqual(macros['all64/clean/adapter_final']['hidden_mpjpe_mm']['pooled'],125.)

    def test_recording_winrates_exclude_zero_sample_pairs(self):
        rows=complete()
        for model in report.LABELS:
            rows.append(row('empty',None,0,model=model))
        rates=report.paired_winrates(rows)
        chosen=next(r for r in rates if (r['scope'],r['dataset'],r['regime'],r['comparator']) ==
            ('all64','rich','clean','input_reference'))
        self.assertEqual(chosen['metrics']['hidden_mpjpe_mm']['paired_recordings'],1)
        self.assertEqual(chosen['metrics']['hidden_mpjpe_mm']['wins'],1)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            report.paired_winrates(rows+[copy.deepcopy(rows[0])])

    def test_winrates_reject_unpaired_or_denominator_changed_outputs(self):
        rows=complete();rows[0]['counts']['hidden_mpjpe_mm']=5
        with self.assertRaisesRegex(ValueError,'denominator'):
            report.paired_winrates(rows)
        with self.assertRaisesRegex(ValueError,'Missing paired'):
            report.paired_winrates(complete()[1:])

    def test_multiple_windows_cast_one_count_weighted_recording_vote(self):
        rows=[]
        for start,count,final in ((0,1,90.),(1,9,120.)):
            for model,value in zip(report.LABELS,(100.,100.,final)):
                rows.append(row('a',value,count,model=model,start=start))
        chosen=next(r for r in report.paired_winrates(rows) if r['scope'] == 'all64'
            and r['comparator'] == 'input_reference')['metrics']['hidden_mpjpe_mm']
        self.assertEqual(chosen['paired_recordings'],1)
        self.assertEqual(chosen['paired_windows'],2)
        self.assertEqual(chosen['losses'],1)
        self.assertEqual(chosen['mean_delta_mm'],17.)


class GateTests(unittest.TestCase):
    def gate(self,rows,**changes):
        groups,macros=report.aggregate(rows)
        flags=dict(ego_exact=True,inherited_foot_pass=True,cpu_verified=True,base_unchanged=True)
        flags.update(changes)
        return report.usability(groups,macros,**flags)

    def test_success_requires_both_comparators(self):
        self.assertEqual(self.gate(complete())['status'],'pass')
        result=self.gate(complete(reference=50.,control=120.,final=90.))
        self.assertEqual(result['status'],'failed')
        failures=[g for g in result['hidden_macro'] if not g['passed']]
        self.assertEqual({g['comparator'] for g in failures},{'input_reference'})

    def test_clean_and_noisy_checked_separately(self):
        rows=complete()
        for current in rows:
            if current['model'] == 'adapter_final' and current['regime'] == 'noisy':
                current['metrics']['hidden_mpjpe_mm']=99.
        self.assertEqual(self.gate(rows)['status'],'failed')

    def test_macro_gain_cannot_hide_corpus_or_observed_regression(self):
        rows=complete(final=30.)
        target=next(r for r in rows if (r['dataset'],r['regime'],r['model']) == ('rich','clean','adapter_final'))
        target['metrics']['hidden_mpjpe_mm']=106.
        result=self.gate(rows)
        self.assertTrue(all(g['passed'] for g in result['hidden_macro']))
        self.assertEqual(result['status'],'failed')
        self.assertEqual(self.gate(complete(visible_final=11.01))['status'],'failed')

    def test_required_evidence_cannot_be_overridden_by_metric_gains(self):
        for flag in ('ego_exact','inherited_foot_pass','cpu_verified','base_unchanged'):
            self.assertEqual(self.gate(complete(),**{flag:False})['status'],'failed')

    def test_missing_dataset_or_hidden_samples_fail_closed(self):
        rows=[r for r in complete() if r['dataset'] != 'rich']
        self.assertEqual(self.gate(rows)['status'],'failed')
        rows=complete()
        for current in rows:
            if current['dataset'] == 'rich':
                current['metrics']['hidden_mpjpe_mm']=None;current['counts']['hidden_mpjpe_mm']=0
        self.assertEqual(self.gate(rows)['status'],'failed')


class SavedArtifactIntegrationTests(unittest.TestCase):
    def test_full_saved_schema_uses_cpu_arrays_and_rejects_tampering(self):
        """Exercise the real native FK, both archive schemas and audit linkage."""
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary);baseline=run/'baseline'
            for folder in ('prepared','baseline_validation','training','baseline/motions'):
                (run/folder).mkdir(parents=True,exist_ok=True)
            def write(path,value):
                path.write_text(json.dumps(value)+'\n')
            truth=np.zeros((3,201),np.float32);rest=np.zeros((22,3),np.float32)
            observation=np.zeros((3,22,18),np.float32);meta=np.zeros((3,22,8),np.float32)
            meta[:,:11,0]=1
            def motion(value):
                result=truth.copy();result[:,0]=value/2000;return result
            reference,control,final=map(motion,(100.,120.,90.))
            panel=[]
            for i,dataset in enumerate(report.DATASETS):
                panel.append(dict(case=i,dataset=dataset,recording='ego'+dataset,sequence_id='ego'+dataset,
                    start=0,source_first=0,mode='ego',case_seed=100+i,panel='fixed'))
            for dataset,n in zip(report.DATASETS,(27,20,17)):
                for k in range(n):
                    panel.append(dict(case=len(panel),dataset=dataset,recording=dataset+str(k),
                        sequence_id=dataset+str(k),start=0,source_first=0,mode='exo',case_seed=100+len(panel),
                        panel='old9' if k<3 else 'new55'))
            saved_before={'step':np.array(0)};saved_after={'step':np.array(1)}
            record_before=dict(step=0,cases=[]);record_after=dict(step=1,cases=[]);base_rows=[]
            for i,entry in enumerate(panel):
                for target in (saved_before,saved_after):
                    target[f'{i}_truth']=truth;target[f'{i}_rest']=rest
                for regime in report.REGIMES:
                    metrics={}
                    for label,prediction in zip(report.LABELS,(reference,control,final)):
                        payload=dict(prediction=prediction,truth=truth,rest=rest,observations=observation,observation_meta=meta)
                        values,counts=report.motion_metrics(payload)
                        metrics[label]={key:dict(value=values[name],count=counts[name]) for name,key in
                            [('w_mpjpe_mm','mpjpe_mm'),('visible_mpjpe_mm','conditioned_mpjpe_mm'),('hidden_mpjpe_mm','unconditioned_mpjpe_mm')]}
                    for target in (saved_before,saved_after):
                        target[f'{i}_{regime}_observations']=observation
                        target[f'{i}_{regime}_observation_meta']=meta
                        target[f'{i}_{regime}_reference_motion']=reference
                    saved_before[f'{i}_{regime}_prediction']=control
                    saved_after[f'{i}_{regime}_prediction']=control if entry['mode'] == 'ego' else final
                    record_before['cases'].append(dict(entry,regime=regime,metrics=metrics['control3000'],reference_metrics=metrics['input_reference']))
                    record_after['cases'].append(dict(entry,regime=regime,metrics=metrics['adapter_final'],reference_metrics=metrics['input_reference']))
                    if entry['mode'] != 'exo':continue
                    for label,prediction in zip((*report.LABELS[:2],'base41k'),(reference,control,control)):
                        path=baseline/'motions'/f'{i}_{regime}_{label}.npz'
                        np.savez_compressed(path,prediction=prediction,truth=truth,rest=rest,
                            observations=observation,observation_meta=meta,reference_motion=reference)
                        base_rows.append(dict(entry,regime=regime,model=label,seed=entry['case_seed'],
                            artifact=str(path),artifact_sha256=report.sha256(path)))
            for folder,record,saved in [('baseline_validation',record_before,saved_before),('training',record_after,saved_after)]:
                write(run/folder/'validation.jsonl',record)
                np.savez_compressed(run/folder/'validation_latest.npz',**saved)
            write(run/'config.json',dict(adaptation=dict(updates=1,reference_policy='exo-local-so3-reference-gap-fill-v1'),physical_guidance=dict(enabled=False)))
            write(run/'STUDY.json',{})
            write(run/'prepared/validation_plan.json',dict(plan=panel))
            write(run/'prepared/effective_config.json',dict(adaptation_parent=dict(base_model_sha256='base')))
            checkpoint=run/'training/last.pt';checkpoint.write_bytes(b'test checkpoint bytes')
            cpu=dict(status='independent_CPU_primary_metrics_passed',CUDA_initialized=False,GPU_used_for_recheck=False,
                matched_input_arrays=True,EGO_serialized_outputs_exact=True,frozen_base_sha256='base',
                final_checkpoint_sha256=report.sha256(checkpoint),files_sha256={str(run/folder/name):report.sha256(run/folder/name)
                    for folder in ('baseline_validation','training') for name in ('validation.jsonl','validation_latest.npz')})
            write(run/'CPU_RECHECK.json',cpu)
            gates=dict(groups=[dict(foot_speed_guard_pass=True,motion_guards=dict(stride=dict(passed=True))) for _ in range(6)])
            write(run/'ACCEPTANCE.json',gates)
            write(run/'COMPARISON.json',dict(CPU_recheck_sha256=report.sha256(run/'CPU_RECHECK.json'),acceptance=gates))
            write(baseline/'PLAN.json',dict(planned_rows=384))
            (baseline/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in base_rows))
            write(baseline/'CPU_RECHECK.json',dict(status='independent_cpu_passed',rows_sha256=report.sha256(baseline/'rows.jsonl')))
            result=report.render(run,baseline)
            self.assertEqual(result['usability']['status'],'pass')
            self.assertEqual(result['matched_standalone_reference_control_arrays'],256)
            self.assertEqual(len(result['rows']),384)
            self.assertFalse(torch.cuda.is_initialized())
            self.assertTrue((run/'USABILITY_zh.md').exists())
            victim=Path(base_rows[0]['artifact']);victim.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Standalone artifact changed'):
                report.render(run,baseline)


if __name__ == '__main__':
    unittest.main()
