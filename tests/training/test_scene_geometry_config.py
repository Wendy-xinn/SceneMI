from pathlib import Path
import pytest
import yaml

from scenemi.scene_geometry_training import load_config


def test_variants_share_parent_budget_and_only_change_declared_geometry():
    root=Path(__file__).resolve().parents[2]
    configs=[load_config(root/'configs'/f'{name}.yaml') for name in
        ('ego_scene_geometry_22k','ego_scene_query_22k','ego_geometry_free_adapter_22k')]
    assert all(c['continuation']==configs[0]['continuation'] for c in configs)
    assert all(c['training']==configs[0]['training'] and c['geometry']==configs[0]['geometry'] for c in configs)
    assert configs[0]['scene_stage']['geometry_losses']
    assert not configs[1]['scene_stage']['geometry_losses']
    assert configs[2]['scene_stage']['scene_blind']
