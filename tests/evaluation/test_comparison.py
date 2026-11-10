"""Family-paired uncertainty must retain the scene selection weights."""
import copy
import unittest

from scenemi.comparison import METRICS, paired_comparison


class ComparisonTests(unittest.TestCase):
    @staticmethod
    def records():
        def row(dataset,family,value,category=None):
            return dict(identity=dict(dataset=dataset,corpus=dataset,family=family,
                category=category or dataset,split='validation',synthetic_chair=False),
                metrics={m:value for m in METRICS})
        candidate=[row('egobody','a',10)]*7+[row('egobody','b',30),row('trumans','t',50)]
        candidate += [row('grail',c,v,c) for c,v in
                      [('curb',10),('sitting',30),('slope',50),('stair_p1',70),('stair_p2',90)]]
        baseline=[copy.deepcopy(r) for r in candidate]
        for r in baseline:
            r['metrics']={k:v+2 for k,v in r['metrics'].items()}
        return dict(rows=baseline,selection_score_mm=42),dict(rows=candidate,selection_score_mm=40)

    def test_window_duplication_does_not_reweight_families_or_categories(self):
        a,b=self.records();result=paired_comparison(a,b,draws=100)
        self.assertEqual(result['source_families'],8)
        self.assertEqual(result['per_dataset']['egobody']['candidate']['mpjpe_mm'],20)
        self.assertEqual(result['dataset_macro']['baseline_minus_candidate']['mpjpe_mm'],2)
        self.assertEqual(result['dataset_macro']['paired_family_bootstrap_95pct']['mpjpe_mm'],[2,2])

    def test_reordered_identities_are_rejected(self):
        a,b=self.records();b['rows'].reverse()
        with self.assertRaises(ValueError):
            paired_comparison(a,b,draws=100)

    def test_independent_score_check_detects_changed_aggregation(self):
        a,b=self.records();b['selection_score_mm']=41
        with self.assertRaises(ValueError):
            paired_comparison(a,b,draws=100)


if __name__=='__main__':
    unittest.main()
