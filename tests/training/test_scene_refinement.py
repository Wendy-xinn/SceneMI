"""Configuration/integration contracts without loading datasets or weights."""
import copy

import pytest
import yaml
import json
import torch

from scenemi.config import ROOT
from scenemi.refinement.training import load_config, restore_selection


def test_pinned_base_four_source_batch_and_real_sampling():
    c=load_config(ROOT/'configs/ego_scene_refinement.yaml')
    assert c['continuation']['step']==22000
    assert c['continuation']['checkpoint']=='runs/ego_only_clean_extension/selection/best.pt'
    assert c['data']['datasets']==['amass','egobody','trumans','grail']
    assert c['data']['excluded_datasets']==['rich']
    assert c['training']['effective_batch_size']==128
    assert c['training']['microbatch_size']==32
    assert c['sampling']['ddim_steps']==20
    assert c['runtime']['allocator_gib'] is None
    assert c['model']['scene_modules_trainable'] is False
    assert c['output']!='runs/ego_generalization_abc'


@pytest.mark.parametrize('field,value', [('width',95),('heads',0),('blocks',0),
    ('lower_limit_deg',180),('root_yaw_limit_deg',float('nan'))])
def test_invalid_architecture_is_rejected(tmp_path,field,value):
    c=yaml.safe_load((ROOT/'configs/ego_scene_refinement.yaml').read_text())
    c['refiner'][field]=value
    file=tmp_path/'refiner.yaml';file.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):load_config(file)


def validation_record(root,step,score,eligible=True):
    p=root/f'selection/step_{step:06d}';p.mkdir(parents=True,exist_ok=True)
    (p/'RESULT.json').write_text(json.dumps(dict(state='complete',selection_score_mm=score,guards=dict(head=eligible,jerk=True))))


def test_selection_recovers_completed_validation_before_best_publication(tmp_path):
    validation_record(tmp_path,0,100.)
    validation_record(tmp_path,500,90.)
    (tmp_path/'training').mkdir()
    torch.save(dict(step=0),tmp_path/'selection/best.pt')
    torch.save(dict(step=500),tmp_path/'training/last.pt')
    selected=restore_selection(tmp_path,500)
    assert selected['step']==500
    assert torch.load(tmp_path/'selection/best.pt',weights_only=True)['step']==500


def test_selection_ignores_future_steps_and_guard_failures(tmp_path):
    validation_record(tmp_path,0,100.)
    validation_record(tmp_path,500,90.,eligible=False)
    validation_record(tmp_path,1000,80.)
    torch.save(dict(step=0),tmp_path/'selection/best.pt')
    assert restore_selection(tmp_path,500)['step']==0


def test_missing_selected_weights_never_fall_back_silently(tmp_path):
    validation_record(tmp_path,0,100.)
    validation_record(tmp_path,500,90.)
    (tmp_path/'training').mkdir()
    torch.save(dict(step=0),tmp_path/'selection/best.pt')
    torch.save(dict(step=600),tmp_path/'training/last.pt')
    with pytest.raises(ValueError,match='unavailable'):
        restore_selection(tmp_path,600)
