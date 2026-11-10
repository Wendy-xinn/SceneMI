"""Checkpoint coverage, metric weighting and final selection must survive saving."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scenemi.scene_selection import (aggregate_scene_rows, capture_candidate, finalists,
                                    validation_manifests)


class SmallValidation:
    split='validation'

    def __init__(self):
        self.records=[]
        self.groups={n:{} for n in ('egobody','trumans','grail')}
        for dataset in ('egobody','trumans'):
            self.add(dataset,dataset+'_family',dataset,2)
        for category in ('curb','sitting','slope','stair_p1','stair_p2'):
            for i in range(5):
                self.add('grail',f'{category}_{i}',category,2)
        self.families={n:sorted(g) for n,g in self.groups.items()}

    def add(self,dataset,family,category,count):
        ids=[]
        for _ in range(count):
            ids.append(len(self.records))
            self.records.append(dict(dataset=dataset,family=family,category=category,starts=np.arange(129)))
        self.groups[dataset][family]=ids

    def metadata(self,record):
        return record


class SceneSelectionTests(unittest.TestCase):
    def test_nested_coverage_of_all_families_and_native_recordings(self):
        data=SmallValidation()
        manifest=validation_manifests(data,777,2)
        self.assertEqual(manifest,validation_manifests(data,777,2))
        self.assertEqual(manifest['coverage']['quick']['grail_category_windows'],
                         dict.fromkeys(('curb','sitting','slope','stair_p1','stair_p2'),2))
        for dataset in data.families:
            actual={r['family'] for r in manifest['comprehensive'] if r['dataset']==dataset}
            self.assertEqual(actual,set(data.families[dataset]))
        native={i for i,r in enumerate(data.records) if r['dataset']!='grail'}
        self.assertEqual(native,{r['member'] for r in manifest['quick'] if r['dataset']!='grail'})
        quick={(r['member'],r['start']) for r in manifest['quick']}
        full={(r['member'],r['start']) for r in manifest['comprehensive']}
        self.assertTrue(quick.issubset(full))

    def test_single_window_recording_is_not_duplicated(self):
        data=SmallValidation()
        data.records[0]['starts']=np.array([0])
        manifest=validation_manifests(data,777,2)
        self.assertEqual(sum(r['member']==0 for r in manifest['comprehensive']),1)

    def test_training_or_missing_action_category_is_rejected(self):
        data=SmallValidation();data.split='train'
        with self.assertRaises(ValueError):validation_manifests(data,777)
        data.split='validation';data.families['grail']=[f for f in data.families['grail'] if not f.startswith('sitting')]
        with self.assertRaises(ValueError):validation_manifests(data,777)

    def test_more_stair_families_cannot_overwhelm_other_actions(self):
        def row(dataset,family,category,value):
            return dict(identity=dict(dataset=dataset,corpus=dataset,family=family,category=category),
                        metrics=dict(mpjpe_mm=value))
        rows=[row('egobody','ego','egobody',100),row('trumans','tru','trumans',200)]
        for i,category in enumerate(('curb','sitting','slope','stair_p1','stair_p2')):
            rows.append(row('grail',category,category,(i+1)*10))
        before=aggregate_scene_rows(rows)['selection_score_mm']
        rows.extend(row('grail',f'stair_extra_{i}','stair_p2',50) for i in range(100))
        self.assertEqual(before,110.)
        self.assertEqual(aggregate_scene_rows(rows)['selection_score_mm'],before)

    def test_final_checkpoint_is_always_a_finalist(self):
        rows=[dict(step=i*1000,selection_score_mm=i) for i in range(6)]
        self.assertEqual(finalists(rows,5000),[0,1000,2000,5000])
        with self.assertRaises(ValueError):finalists(rows,6000)
        with self.assertRaises(ValueError):finalists(rows+rows[:1],5000)

    def test_atomic_last_replacement_cannot_overwrite_retained_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);last=root/'last.pt';last.write_bytes(b'first')
            saved=capture_candidate(last,root/'selection',1000,'sealed')
            replacement=root/'replacement';replacement.write_bytes(b'second');os.replace(replacement,last)
            self.assertEqual(saved.read_bytes(),b'first')
            with self.assertRaises(ValueError):capture_candidate(last,root/'selection',1000,'sealed')

    def test_comprehensive_selection_can_reverse_quick_ranking(self):
        from scenemi.local_training import train
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'preflight').mkdir()
            c=dict(seed=1,grail_review='accepted',training=dict(planned_updates=3000,
                checkpoint_every=1000,validation_every=1000),optimizer=dict(lr=.001,weight_decay=0.),
                prior_step=50000,prior_checkpoint='prior.pt',prior_sha256='prior')
            seal=dict(sha256='seal',config=c)
            for device in ('cpu','cuda'):
                (out/'preflight'/f'{device}.json').write_text(json.dumps(dict(
                    state='passed',contract_sha256='seal',resumable_checkpoint_bytes=1000)))
            model=torch.nn.Linear(1,1)
            with torch.no_grad():model.weight.zero_()
            quick=[{}];full=[{},{}]
            manifests=dict(quick=quick,comprehensive=full,coverage={})
            def update(model,optimizer,data,rng,config,device,step):
                with torch.no_grad():model.weight.fill_(step)
                return dict(losses=dict(total=0.),identities=[dict(dataset='trumans')])
            def evaluate(model,data,cases,config,device,progress=None):
                step=int(model.weight.item())
                score=step/1000+1 if cases is quick else {0:100.,1000:50.,2000:10.,3000:20.}[step]
                return dict(selection_score_mm=score,rows=[])
            with patch('scenemi.local_training.update',side_effect=update),\
                    patch('scenemi.local_training.signal.signal'):
                with patch('scenemi.local_training.evaluate',side_effect=RuntimeError('interrupted validation')):
                    with self.assertRaises(RuntimeError):
                        train(model,None,None,manifests,c,'cpu',out,seal,False)
                with patch('scenemi.local_training.evaluate',side_effect=evaluate):
                    result=train(model,None,None,manifests,c,'cpu',out,seal,True)
            self.assertEqual(result['state'],'complete')
            self.assertEqual(result['best']['step'],2000)
            self.assertFalse(result['best']['provisional'])
            best=torch.load(out/'selection/best.pt',weights_only=False)
            last=torch.load(out/'training/last.pt',weights_only=False)
            self.assertEqual(best['step'],2000);self.assertEqual(last['step'],3000)
            self.assertEqual(float(best['model']['weight']),2000.)
            self.assertEqual(len(list((out/'selection/candidates').glob('*.pt'))),4)


if __name__=='__main__':unittest.main()
