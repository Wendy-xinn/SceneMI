import unittest

from scenemi.relation_evaluation import summarize_variants


class RelationEvaluationTests(unittest.TestCase):
    def test_variant_summary_keeps_one_copy_of_rows_and_category_weights(self):
        ledger=[]
        cases=[('egobody','e',10.),('trumans','t',20.),
               ('grail','sitting',30.),('grail','stair_p1',50.),('grail','stair_p1',70.)]
        for i,(dataset,category,error) in enumerate(cases):
            identity=dict(dataset=dataset,corpus=dataset,category=category,family=str(i))
            ledger.append(dict(variants={name:dict(identity=identity,metrics=dict(mpjpe_mm=error+offset))
                for name,offset in [('correct',0.),('empty',3.),('mismatched',1.)]}))
        result=summarize_variants(ledger)
        self.assertEqual(result['correct']['rows'],[r['variants']['correct'] for r in ledger])
        # Five-category GRAIL weighting uses present category family means,
        # rather than giving the two stair families twice the sitting weight.
        self.assertAlmostEqual(result['correct']['selection_score_mm'],(10+20+(30+60)/2)/3)
        self.assertAlmostEqual(result['empty']['selection_score_mm']-result['correct']['selection_score_mm'],3.)


if __name__=='__main__':unittest.main()
