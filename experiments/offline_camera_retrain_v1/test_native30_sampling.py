"""Sampling coverage, overlap, exact resume and outcome-independent VAL panels."""
import copy
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch
import tempfile
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.native30_sampling import StratifiedRecordingSampler, PROTOCOL
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_validation import monitoring_plan, aggregate_monitoring, validate
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config, sample_item

OPTIONS=dict(protocol=PROTOCOL,mode_block_size=10,recording_cooldown=8,recent_windows_per_target=4)
PLAN=[dict(group=g,mode=m,probability=p*q) for m,p in [('ego',.7),('exo',.3)]
      for g,q in [('trumans',.6),('camera_wearer',.15),('interactee',.15),('rich',.1)]]


class Fixture:
    def __init__(self,records=16,split='train'):
        self.split=split;self.frames=192;self.rng=np.random.default_rng(100)
        self.members={g:[] for g in ('trumans','camera_wearer','interactee','rich')}
        for group in self.members:
            for i in range(records):
                key='egobody' if group in ('camera_wearer','interactee') else group
                meta=dict(sequence_id=f'{group}_{i:03}',recording=f'{key}_{i:03}',source_first=0,
                          path=f'{group}/{i}',allowed_modes=['ego','exo'])
                # One much longer record should not dominate the recording cycle.
                self.members[group].append((meta,np.arange(0,6400 if i==0 else 640,64)))
    def starts_for(self,g,i,mode):return self.members[g][i][1]
    def sample(self,g,mode,**kwargs):return dict(),dict(group=g,mode=mode,**kwargs)
    sampler_state=Native30Data.sampler_state
    load_sampler_state=Native30Data.load_sampler_state


class SamplingTests(unittest.TestCase):
    def test_exact_mode_blocks_and_dataset_quotas(self):
        s=StratifiedRecordingSampler(Fixture(),PLAN,OPTIONS,55);counts=Counter()
        for _ in range(100):
            block=[s.draw() for _ in range(10)]
            self.assertEqual(Counter(v['mode'] for v in block),{'ego':7,'exo':3})
            counts.update((v['group'],v['mode']) for v in block)
        for row in PLAN:
            self.assertLessEqual(abs(counts[row['group'],row['mode']]-1000*row['probability']),1)

    def test_recording_cycles_remove_length_bias_and_avoid_recent_records(self):
        s=StratifiedRecordingSampler(Fixture(),PLAN,OPTIONS,88);recent=[];records=Counter()
        for _ in range(1200):
            v=s.draw()
            if not v['cooldown_fallback']:self.assertNotIn(v['recording_key'],recent)
            recent=(recent+[v['recording_key']])[-OPTIONS['recording_cooldown']:]
            if v['group']=='trumans' and v['mode']=='ego':records[v['recording_key']]+=1
        self.assertEqual(len(records),16)
        self.assertLessEqual(max(records.values())-min(records.values()),1)

    def test_nonoverlap_history_across_modes_and_logged_fallback(self):
        s=StratifiedRecordingSampler(Fixture(records=1),PLAN,OPTIONS,4);history={};fallback=False
        for _ in range(200):
            v=s.draw();key=(v['group'],v['recording_key']);current=(v['start'],v['start']+192)
            if not v['overlap_fallback']:
                for lo,hi in history.get(key,[]):self.assertTrue(current[1]<=lo or current[0]>=hi)
            history[key]=(history.get(key,[])+[current])[-4:];fallback |= v['cooldown_fallback']
        self.assertTrue(fallback)

    def test_resume_replays_group_record_window_and_data_yaw_rng(self):
        data=Fixture();data.training_sampler=StratifiedRecordingSampler(data,PLAN,OPTIONS,44)
        for _ in range(73):data.training_sampler.draw();data.rng.random()
        state=data.sampler_state();expected=[(data.training_sampler.draw(),data.rng.random()) for _ in range(97)]
        restored=Fixture();restored.load_sampler_state(state)
        self.assertEqual(expected,[(restored.training_sampler.draw(),restored.rng.random()) for _ in range(97)])
        # Snapshot must not alias queues that have subsequently advanced.
        self.assertEqual(state['recording_sampler']['draws'],73)

    def test_resume_rejects_changed_windows_or_policy(self):
        s=StratifiedRecordingSampler(Fixture(),PLAN,OPTIONS,1);state=s.state_dict()
        changed=Fixture();changed.members['rich'][0][1][0]=32
        with self.assertRaisesRegex(ValueError,'changed'):
            StratifiedRecordingSampler(changed,PLAN,OPTIONS,1).load_state_dict(state)
        changed=Fixture();changed.members['rich'][0][0]['source_first']=100
        with self.assertRaisesRegex(ValueError,'changed'):
            StratifiedRecordingSampler(changed,PLAN,OPTIONS,1).load_state_dict(state)

    def test_training_sampler_rejects_val_and_invalid_mode_ratio(self):
        with self.assertRaises(ValueError):StratifiedRecordingSampler(Fixture(split='validation'),PLAN,OPTIONS,1)
        with self.assertRaises(ValueError):StratifiedRecordingSampler(Fixture(),PLAN,dict(OPTIONS,mode_block_size=3),1)

    def test_sample_item_uses_head_only_and_explicit_member_start(self):
        data=Fixture();builder=Mock();builder.build.side_effect=lambda d,s,i,m,j,**kw:(s,None,None)
        config=dict(sampler=OPTIONS,seed=1,ego_head_only_probability=1.,scene_relations={})
        for _ in range(10):
            item,identity=sample_item(data,builder,config,PLAN,np.random.default_rng(11))
            self.assertIn('member',identity);self.assertIn('start',identity)
            joints=builder.build.call_args.args[4]
            self.assertEqual(joints,(15,) if identity['mode']=='ego' else None)

    def test_bank_uses_same_sampler_and_registered_corpus_plan(self):
        data=Fixture();ids=[]
        for g,members in data.members.items():
            for meta,starts in members[:4]:
                for mode in ('ego','exo'):
                    ids.append(dict(meta,group=g,mode=mode,start=int(starts[0])))
        s=StratifiedRecordingSampler.for_bank(ids,192,PLAN,OPTIONS,8)
        for _ in range(30):
            v=s.draw();meta,_=s.data.members[v['group']][v['member']]
            self.assertEqual(ids[meta['bank_index']]['mode'],v['mode'])


class MonitoringTests(unittest.TestCase):
    def config(self):
        return dict(ego_head_only_probability=1.,validation=dict(seed=71,interval_steps=1000,
            subset_protocol='fixed-plus-rotating-recordings-v1',fixed_recordings_per_dataset_mode=1,
            rotating_recordings_per_dataset_mode=1))

    def test_fixed_panel_stable_rotating_recordings_change_and_no_overlap(self):
        data=Fixture(split='validation');config=self.config();previous=None;seen={}
        def identities(rows):return [(v['group'],v['mode'],v['sequence_id'],v['start'],v['case_seed']) for v in rows if v['panel']=='fixed']
        anchor=identities(monitoring_plan(data,config,1))
        for round_index in range(15):
            plan=monitoring_plan(data,config,round_index*1000 or 1)
            self.assertEqual(len(plan),12);self.assertEqual(anchor,identities(plan))
            for v in plan:
                self.assertEqual(v['joints'],(15,) if v['mode']=='ego' else None)
                if v['panel']=='rotating':
                    key=(v['dataset'],v['mode']);seen.setdefault(key,set())
                    self.assertNotIn(v['recording'],seen[key]);seen[key].add(v['recording'])
                    fixed=next(x for x in plan if x['dataset']==v['dataset'] and x['mode']==v['mode'] and x['panel']=='fixed')
                    self.assertNotEqual(v['recording'],fixed['recording'])
            previous=plan
        self.assertEqual(plan,monitoring_plan(data,config,14000))

    def test_val_plan_preserves_sampler_rng_and_rejects_test(self):
        data=Fixture(split='validation');state=copy.deepcopy(data.rng.bit_generator.state)
        monitoring_plan(data,self.config(),1000)
        self.assertEqual(data.rng.bit_generator.state,state)
        with self.assertRaises(ValueError):monitoring_plan(Fixture(split='test'),self.config(),1000)

    def test_summary_is_count_weighted_and_separates_panels_modes_regimes(self):
        rows=[dict(panel=p,mode=m,regime=r,metrics=dict(mpjpe_mm=dict(value=v,count=n)))
              for p,m,r,v,n in [('fixed','ego','clean',10.,1),('fixed','ego','clean',20.,3),
                                ('rotating','ego','clean',500.,1),('fixed','exo','noisy',77.,2)]]
        result=aggregate_monitoring(rows)
        self.assertEqual(result['fixed/ego/clean/scene_on']['mpjpe_mm'],dict(value=17.5,count=4))
        self.assertEqual(len(result),3)

    def test_new_config_head_only_and_registered_monitoring(self):
        c=read_config(Path(__file__).with_name('native30_recording_headonly_config.json'))
        self.assertEqual(c['ego_head_only_probability'],1.)
        self.assertFalse(c['observation']['input_noise']['ego_wrists']['enabled'])
        self.assertEqual(c['sampling']['ddim_steps'],20)
        self.assertEqual(c['scene']['scales_m'],[.08,.25,.6])

    def test_validation_orchestration_pairs_scene_checks_and_preserves_training_rng(self):
        data=Fixture(split='validation');config=self.config()
        config['sampling']=dict(ddim_steps=20)
        config['validation'].update(scene_ablation_interval_steps=5000,scope='monitoring')
        data.sample=lambda group,mode,**kwargs:(dict(motion=np.zeros((192,201),np.float32),rest=np.zeros((22,3),np.float32)),dict(group=group,mode=mode,**kwargs))
        builder=Mock();builder.noise_config={};builder.skin_cache=None;builder.mounts={}
        def build(*args,**kwargs):
            return dict(trajectory=np.zeros((192,22,9),np.float32),observation_meta=np.ones((192,22,5),np.float32)),None,None
        builder.build.side_effect=build
        prediction=torch.zeros(1,192,201);model=torch.nn.Linear(1,1);model.train()
        seeds=[]
        def sample(*args,**kwargs):
            seeds.append((kwargs['seed'],kwargs.get('use_scene',True)))
            torch.rand(1);return prediction
        state=torch.get_rng_state().clone()
        base='experiments.offline_camera_retrain_v1.native30_validation.'
        with tempfile.TemporaryDirectory() as folder, \
             patch(base+'collate_observations',return_value=dict()), \
             patch(base+'model_inputs',return_value=dict(reference_motion=prediction)), \
             patch(base+'sample_motion',side_effect=sample), \
             patch(base+'raw_metrics',return_value=dict(mpjpe_mm=dict(value=3.,count=1))), \
             patch('experiments.offline_camera_retrain_v1.native30_observations.Native30ObservationBuilder',return_value=builder):
            rows=validate(model,data,builder,config,folder,5000)
            self.assertEqual(len(rows),36)
            self.assertEqual(sum(r.get('scene_variant')=='scene_off' for r in rows),12)
            self.assertTrue((Path(folder)/'validation_latest.npz').exists())
        self.assertTrue(model.training);self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for seed,scene_on in seeds:
            if not scene_on:self.assertIn((seed,True),seeds)


if __name__=='__main__':unittest.main()
