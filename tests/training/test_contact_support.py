"""Loss mass, independent evidence, bounded support and restored motion targets."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scenemi.config import ROOT
from scenemi.data.support_cache import PreparedSupportIndex
from scenemi.geometry.spatial_scene import SceneView
from scenemi.geometry.support_evidence import EvidenceQueries, query_evidence
from scenemi.refinement.contact_training import load_config
from scenemi.refinement.support_objectives import (weighted_group, normalize_group,
    geometry_components, components, EffectiveBatchGradients, combine)
from tests.model.test_support_surfaces import plane, index_for, policy
from tests.training.test_support_cache import prepared
from tests.training.test_spatial_execution import equal_tree
from tests.training.test_spatial_refinement import fixture


def objective_policy():
    return copy.deepcopy(load_config(ROOT/'configs/ego_scene_contact_refinement.yaml')['objective'])


def candidate_view(points, *, fitted=True, reliable=False):
    index = index_for(points)
    def query(anchors):
        shape = anchors.shape[:-1]
        return dict(height=np.zeros(shape,np.float32), normal=np.broadcast_to([0.,1.,0.],anchors.shape).copy(),
            confidence=np.full(shape,.8,np.float32), valid=np.full(shape,fitted,bool),
            reliable=np.full(shape,reliable,bool), ambiguous=np.zeros(shape,bool), fit_error=np.zeros(shape,np.float32))
    return SceneView(PreparedSupportIndex(index,SimpleNamespace(query=query)),np.zeros(3),np.eye(3))


def test_invalid_padding_and_uniform_confidence_do_not_dilute_valid_gradients():
    original=torch.tensor([1.,3.],requires_grad=True)
    first=weighted_group(original.square(),torch.tensor([.1,.1]),torch.ones(2,dtype=torch.bool))
    loss,_=normalize_group([first],minimum=2)
    gradient,=torch.autograd.grad(loss,original)
    padded=torch.tensor([1.,3.]+[100.]*100,requires_grad=True)
    second=weighted_group(padded.square(),torch.tensor([.01,.01]+[0.]*100),torch.ones(102,dtype=torch.bool))
    other,_=normalize_group([second],minimum=2)
    other.backward()
    torch.testing.assert_close(loss,other)
    torch.testing.assert_close(gradient,padded.grad[:2])
    assert not padded.grad[2:].any()


def test_effective_batch_normalization_is_independent_of_chunk_boundaries():
    values=torch.tensor([1.,2.,10.,20.],requires_grad=True);weights=torch.tensor([1.,.2,0.,.8])
    whole,_=normalize_group([weighted_group(values,weights,torch.ones(4,dtype=torch.bool))],minimum=3)
    chunks=[weighted_group(values[:1],weights[:1],torch.ones(1,dtype=torch.bool)),
            weighted_group(values[1:],weights[1:],torch.ones(3,dtype=torch.bool))]
    split,_=normalize_group(chunks,minimum=3)
    assert whole==split


@pytest.mark.parametrize('weights',[[0.,0.],[1.,0.]])
def test_empty_and_too_small_groups_have_exactly_zero_loss_and_gradient(weights):
    value=torch.tensor([2.,4.],requires_grad=True)
    loss,stats=normalize_group([weighted_group(value.square(),torch.tensor(weights),torch.ones(2,dtype=torch.bool))],minimum=2)
    loss.backward()
    assert loss==0 and not value.grad.any() and stats['enabled']==0


def test_two_surfaces_do_not_count_one_anchor_twice_for_minimum_sample_gate():
    values=torch.ones(1,2,requires_grad=True)
    group=weighted_group(values,torch.ones_like(values),torch.ones(1,dtype=torch.bool),sample_axes=2)
    loss,stats=normalize_group([group],minimum=2)
    assert loss==0 and stats['valid_count']==1


@pytest.mark.parametrize('measured,fitted,reliable',[(True,True,True),(True,True,False),(False,True,True),(False,False,False)])
def test_measured_and_fitted_evidence_preserve_the_four_independent_states(measured,fitted,reliable):
    points=plane() if measured else plane(.4)
    view=candidate_view(points,fitted=fitted,reliable=reliable)
    anchors=np.array([[[0.,.01,0.]]],np.float32)
    result=query_evidence(view,anchors,{'query':{'near_radius_m':.2},'ground':{}},ground_mask=np.ones(1,bool))
    assert result['observed_geometry']['valid'].item()==measured
    assert result['support_surface']['valid'].item()==fitted
    assert result['support_surface']['reliable'].item()==reliable


def geometry_fixture(*, reliable=False):
    model,batch,points,_=fixture()
    view=candidate_view(plane(),reliable=reliable)
    batch['scene_views']=(view,)
    return model,batch,points


def test_uncertain_fit_preserves_close_measured_support_but_wall_and_far_points_do_not_force_contact():
    model,batch,points=geometry_fixture(reliable=False)
    details=dict(scene_active=torch.ones(1),contact_logits=torch.zeros(1,8,67,requires_grad=True))
    groups,stats=geometry_components(model,points.flatten(2),details,batch,objective_policy())
    assert groups['support']['count']>0 and stats['measured_support_valid_count'].sum()>0
    assert stats['fitted_support_valid_count'].sum()==0
    for point,normal in (([0.,.15,0.],[0.,1.,0.]),([.01,0.,0.],[1.,0.,0.])):
        index=index_for(np.asarray([point],np.float32));index.normals[:]=normal
        view=SceneView(PreparedSupportIndex(index,batch['scene_views'][0].index.atlas),np.zeros(3),np.eye(3))
        batch['scene_views']=(view,)
        groups,_=geometry_components(model,points.flatten(2),details,batch,objective_policy())
        assert groups['support']['count']==0 and groups['contact']['count']==0


def test_both_support_sources_contribute_without_contact_probability_gating_geometry():
    model,batch,points=geometry_fixture(reliable=True)
    prediction=points.clone();prediction[...,1]=.1;prediction=prediction.flatten(2).requires_grad_()
    logits=torch.full((1,8,67),-20.,requires_grad=True)
    details=dict(scene_active=torch.ones(1),contact_logits=logits)
    groups,stats=geometry_components(model,prediction,details,batch,objective_policy())
    assert stats['measured_support_valid_count'].sum()>0 and stats['fitted_support_valid_count'].sum()>0
    loss,_=normalize_group([groups['support']],minimum=1)
    loss.backward()
    assert prediction.grad.abs().max()>0 and logits.grad is None


def test_collision_requeries_final_body_and_differentiates_to_that_body(monkeypatch):
    from scenemi.refinement import support_objectives
    model,batch,points=geometry_fixture(reliable=True)
    final=points.clone();final[...,1]=-.1;final=final.flatten(2).requires_grad_()
    calls=[];original=support_objectives.evidence
    def record(model,anchors,batch,**options):
        calls.append((anchors.detach().clone(),options))
        result=original(model,anchors,batch,**options)
        assert not result['observed_geometry']['centre'].requires_grad
        return result
    monkeypatch.setattr(support_objectives,'evidence',record)
    details=dict(scene_active=torch.ones(1),contact_logits=torch.zeros(1,8,67))
    groups,_=geometry_components(model,final,details,batch,objective_policy())
    stride=model.geometry_policy['loss_frame_stride']
    torch.testing.assert_close(calls[0][0],final[:,::stride].reshape(1,-1,67,3))
    assert calls[0][1]['dynamic'] is True
    assert calls[1][1]['dynamic'] is False
    assert not torch.equal(calls[0][0],calls[1][0])
    loss,_=normalize_group([groups['penetration']],minimum=1)
    loss.backward()
    assert loss>0 and (final.grad.reshape(1,8,67,3)[:,::stride,:,1]<0).all()


def test_process_evidence_matches_serial_and_reuses_existing_fit_cache(tmp_path):
    index,cache,_=prepared(tmp_path,plane())
    view=SceneView(PreparedSupportIndex(index,cache.load(index.execution_reference,index)),np.zeros(3),np.eye(3))
    from tests.model.test_spatial_scene import policy as scene_policy
    anchors=np.zeros((4,2,3),np.float32);anchors[...,1]=.03
    options=dict(ground_mask=np.ones(2,bool));expected=query_evidence(view,anchors,scene_policy(),**options)
    with EvidenceQueries(1,tmp_path/'indices',tmp_path/'support',policy()) as executor:
        actual=executor.submit('evidence',view,anchors,scene_policy(),**options).result(timeout=60)
        equal_tree(expected,actual)


def test_restored_lower_fk_penalizes_static_lower_body_error_without_velocity_change(monkeypatch):
    from tests.model.test_scene_refinement import fixture as body_fixture
    from scenemi.refinement.model import apply_rotations
    model,raw,details,public,batch,_=body_fixture()
    omega=torch.zeros(*raw.shape[:2],22,3,dtype=raw.dtype);omega[:,:,4,0]=.1;omega.requires_grad_()
    prediction=apply_rotations(raw,omega,details['rest'])
    extra=dict(details,scene_active=torch.ones(len(raw)),rotation_residual=omega,
        anchor_near_hit=torch.ones(len(raw),raw.shape[1],67,dtype=torch.bool),
        ground_valid=torch.zeros(len(raw),raw.shape[1],67,dtype=torch.bool),
        anchor_hit=torch.ones(len(raw),raw.shape[1],67,dtype=torch.bool))
    monkeypatch.setattr('scenemi.refinement.support_objectives.geometry_components',lambda *args: ({},{}))
    p=objective_policy();p['motion']={k:float(k=='lower_fk') for k in p['motion']}
    local,_,stats=components(model,prediction,extra,batch,p)
    local['motion'].backward()
    assert stats['motion_lower_fk']>0 and stats['motion_lower_velocity']<1e-20
    assert omega.grad[:,:,4].abs().sum()>0


def test_new_protocol_keeps_the_old_run_and_network_schedule_intact():
    from scenemi.refinement.support_training import load_config as retained_config
    old=retained_config(ROOT/'configs/ego_scene_support_refinement.yaml')
    new=load_config(ROOT/'configs/ego_scene_contact_refinement.yaml')
    for key in ('refiner','optimizer','data','training','sampling','spatial','support'):
        assert old[key]==new[key]
    assert old['output']!=new['output'] and new['objective']['motion']['lower_fk']==.25


def test_omitted_runtime_section_still_uses_worker_owned_map_references(monkeypatch):
    from scenemi.refinement import contact_training
    config = load_config(ROOT/'configs/ego_scene_contact_refinement.yaml')
    assert config['query_runtime'] == contact_training.DEFAULT_QUERY_RUNTIME
    calls = []
    monkeypatch.setattr(contact_training, 'ReferenceSupportData',
        lambda config, paths, split: calls.append(split) or 'reference-only')
    # Older YAML without the new block must not select a full parent KD-tree.
    assert contact_training.data_for({}, {}, 'train') == 'reference-only'
    assert calls == ['train']


def test_device_transfer_failure_closes_preflight_query_workers(tmp_path, monkeypatch):
    from scenemi.refinement import contact_training
    closed = []
    class FailingModel:
        def to(self, device):
            raise RuntimeError('synthetic device transfer failure')
        def close_queries(self):
            closed.append(True)
    monkeypatch.setattr(contact_training, 'deterministic', lambda *args: None)
    monkeypatch.setattr(contact_training, 'contract', lambda config: {})
    monkeypatch.setattr(contact_training, 'verify_contract', lambda seal: None)
    monkeypatch.setattr(contact_training, 'model_for', lambda *args: FailingModel())
    with pytest.raises(RuntimeError, match='synthetic device transfer failure'):
        contact_training.preflight(dict(seed=1, output=str(tmp_path)), {}, 'cpu')
    assert closed == [True]


def test_cleanup_attempts_every_resource_and_keeps_original_failure():
    from scenemi.refinement.contact_training import _close_resources
    closed = []
    def failing_reporter():
        raise RuntimeError('synthetic reporter close failure')
    def workers():
        closed.append(True)
    with pytest.raises(ValueError, match='original training failure'):
        try:
            raise ValueError('original training failure')
        finally:
            _close_resources(failing_reporter, workers)
    assert closed == [True]
    with pytest.raises(RuntimeError, match='synthetic reporter close failure'):
        _close_resources(failing_reporter, workers)
    assert closed == [True, True]


def test_context_only_is_retained_as_context_but_does_not_enable_motion_change():
    from tests.model.test_spatial_refinement import fixture as model_fixture
    from scenemi.refinement.contact_model import ContactSupportPipeline
    old,raw,details,public,policy=model_fixture()
    from tests.model.test_scene_refinement import policies
    model=ContactSupportPipeline(old,policies()[0],policy).double()
    torch.nn.init.normal_(model.refiner.rotation.weight,std=.1)
    output,extra=model.refine(raw,details,public)
    assert extra['evidence_states']['context'].any()
    assert not extra['anchor_near_hit'].any() and not extra['ground_valid'].any()
    assert not extra['rotation_residual'].any()
    assert torch.equal(output,model.locate(raw,details))


def test_near_observation_remains_usable_when_fitted_support_is_unknown():
    from tests.model.test_spatial_refinement import fixture as model_fixture
    from scenemi.refinement.contact_model import ContactSupportPipeline
    from tests.model.test_scene_refinement import policies
    old,raw,details,public,policy=model_fixture()
    model=ContactSupportPipeline(old,policies()[0],policy).double()
    torch.nn.init.normal_(model.refiner.rotation.weight,std=.1)
    view=candidate_view(plane(),fitted=False,reliable=False)
    output,extra=model.refine(raw,details,dict(public,scene_views=(view,view)))
    assert extra['anchor_near_hit'].any() and not extra['ground_valid'].any()
    assert extra['rotation_residual'].abs().max()>0 and torch.isfinite(output).all()


def test_bounded_memory_gradient_accumulation_matches_full_graph_batch_objective():
    p=objective_policy()
    p['normalization']['minimum_valid_samples']={k:2 for k in ('contact','support','sliding','penetration')}
    x=torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
    weights=torch.tensor([.2,0.,.1,.9],dtype=torch.float64)
    def chunk(parameter,first,last):
        y=x[first:last]*parameter
        local=dict(motion=(y-1).square().mean(),residual=y.square().mean())
        groups={name:weighted_group((y-(i+1)).square(),weights[first:last],
            torch.ones(last-first,dtype=torch.bool)) for i,name in enumerate(('penetration','support','sliding','contact'))}
        return last-first,local,groups
    reference=torch.nn.Parameter(torch.tensor(.4,dtype=torch.float64))
    expected,_=combine([chunk(reference,0,1),chunk(reference,1,4)],p)
    expected['total'].backward()
    actual=torch.nn.Parameter(reference.detach().clone())
    accumulation=EffectiveBatchGradients([('rotation.weight',actual)],p,len(x))
    for first,last in ((0,1),(1,4)):
        n,local,groups=chunk(actual,first,last)
        accumulation.add(n,local,groups)
    got,stats=accumulation.finish()
    torch.testing.assert_close(got['total'],expected['total'],atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(actual.grad,reference.grad,atol=1e-12,rtol=1e-12)
    assert stats['support_enabled']==1 and stats['support_valid_count']==3
    assert stats['weighted_gradient_norm_support']>0


def test_scene_free_chunk_with_constant_geometric_losses_has_zero_gradient():
    p=objective_policy();parameter=torch.nn.Parameter(torch.tensor(.4))
    local=dict(motion=parameter*0,residual=parameter*0)
    groups={name:weighted_group(torch.zeros(4),torch.zeros(4),torch.zeros(4,dtype=torch.bool))
            for name in ('penetration','support','sliding','contact')}
    accumulated=EffectiveBatchGradients([('rotation.weight',parameter)],p,4)
    accumulated.add(4,local,groups)
    terms,statistics=accumulated.finish()
    assert terms['total']==0 and parameter.grad==0
    assert statistics['weighted_gradient_norm_support']==0
