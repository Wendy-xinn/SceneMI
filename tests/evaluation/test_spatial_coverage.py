"""Coverage counts distinguish body/foot evidence, unknown maps and ground."""
import numpy as np

from scenemi.refinement.coverage import combine_distributions, summarize
from scenemi.refinement.spatial_audit import distribution


def test_aggregate_counts_weight_anchor_numbers_and_retain_unknowns():
    first=distribution([.1])
    second=distribution([.3,.5,.9,np.inf])
    result=combine_distributions([first,second])
    assert result['count']==5 and result['unknown']==1
    assert result['counts']==[1,1,1,1]
    assert result['within_20cm_fraction']==.2
    assert 'quantiles_m' not in result


def test_window_minimum_does_not_imply_foot_coverage_or_a_known_ground_plane():
    distances=dict(full_static=np.array([.1,1.,2.]),predicted_feet_full_static=np.array([1.,2.]))
    row=dict(case=dict(dataset='trumans'),identity=dict(family='one-recording'),originally_zero_hit=True,
        distance={k:distribution(v) for k,v in distances.items()},minimum_m={k:float(v.min()) for k,v in distances.items()},
        ground=dict(sampled_foot_anchors=108,valid=0,ambiguous=2,valid_without_20cm_neighbour=0))
    result=summarize([row]);group=result['by_source']['trumans']['all']
    assert group['distance']['full_static']['window_minimum_distribution']['counts']==[1,0,0,0]
    assert group['distance']['predicted_feet_full_static']['window_minimum_distribution']['counts']==[0,0,0,1]
    assert group['ground']['valid']==0 and group['ground']['ambiguous']==2
    assert not result['amass']['scene_available']
    assert not result['new_model_performance_measured'] and result['gt_body_lookup_diagnostic_only']
