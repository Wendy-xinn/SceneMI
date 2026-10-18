"""CPU tests for exact EXO recording independence and outcome-free selection."""
import unittest

from .exo_cohort import case_seed, select


class ExoCohortTest(unittest.TestCase):
    def fixture(self):
        config = dict(validation=dict(seed=2026101351)); entries=[]; previous=[]
        for dataset in ('rich','trumans','egobody'):
            for recording in range(6):
                roles = ('camera_wearer','interactee') if dataset=='egobody' else (dataset,)
                for role in roles:
                    for start in (0,64,128,192):
                        row=dict(dataset=dataset,group=role,mode='exo',recording=f'{dataset}_{recording}',
                            sequence_id=f'{dataset}_{recording}_{role}',source_bundle_ordinal=recording,
                            path=f'/{dataset}_{recording}_{role}',start=start,source_first=1000+start)
                        entries.append(row)
                        if recording<3 and start==64 and role==roles[recording%len(roles)]:
                            previous.append(dict(row,case_seed=case_seed(config,row)))
        return config,entries,previous

    def test_independence_regression_exact_and_upper_middle(self):
        config,entries,previous=self.fixture();selected=select(entries,previous,config,dict(rich=6,trumans=5,egobody=6))
        self.assertEqual(len(selected),17)
        self.assertEqual(len({(r['dataset'],r['recording']) for r in selected}),17)
        self.assertTrue(all(r['dataset']=='rich' and r['panel']=='new55' for r in selected[:3]))
        self.assertTrue(all(r['start']==128 for r in selected if r['panel']=='new55'))
        old={(r['group'],r['sequence_id'],r['start']):r['case_seed'] for r in previous}
        self.assertEqual({(r['group'],r['sequence_id'],r['start']):r['case_seed'] for r in selected if r['panel']=='old9'},old)
        self.assertEqual(selected,select(list(reversed(entries)),previous,config,dict(rich=6,trumans=5,egobody=6)))

    def test_missing_old_recording_refused_no_substitute(self):
        config,entries,previous=self.fixture()
        entries=[r for r in entries if r['recording']!='rich_0']
        with self.assertRaises(ValueError):select(entries,previous,config,dict(rich=6,trumans=5,egobody=6))

    def test_changed_prior_seed_refused(self):
        config,entries,previous=self.fixture();previous[0]['case_seed']+=1
        with self.assertRaises(ValueError):select(entries,previous,config,dict(rich=6,trumans=5,egobody=6))


if __name__=='__main__':unittest.main()
