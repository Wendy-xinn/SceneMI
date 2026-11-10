import torch

from scenemi.geometry.surface_query import nearest,query_surface


def test_nearest_masks_nonprefix_and_empty_observations():
    points=torch.tensor([[[0.,0,0],[2,0,0],[1,0,0]],[[0.,0,0],[1,0,0],[2,0,0]]])
    query=torch.tensor([[[.1,0,0],[1.1,0,0]],[[.1,0,0],[1.1,0,0]]])
    valid=torch.tensor([[False,True,True],[False,False,False]])
    index,distance=nearest(query,points,valid)
    assert index[0].tolist()==[2,2]
    assert torch.allclose(distance[0],torch.tensor([.9,.1]))
    assert torch.isinf(distance[1]).all()


def test_dynamic_lookup_uses_only_current_valid_object_points():
    points=torch.tensor([[[0.,0,0]]]);normal=torch.tensor([[[0.,1,0]]])
    batch=dict(static_points=points,static_normals=normal,static_valid=torch.ones(1,1,dtype=torch.bool),
        static_normal_valid=torch.ones(1,1,dtype=torch.bool),
        frame_scene_points=torch.tensor([[[[0.,.1,0]],[[0.,.1,0]]]]),
        frame_scene_owner=torch.tensor([[[1],[100]]]),
        frame_normals=normal[:,None].expand(-1,2,-1,-1),frame_normal_valid=torch.ones(1,2,1,dtype=torch.bool))
    anchors=torch.tensor([[[[0.,.11,0]],[[0.,.11,0]]]])
    patch=query_surface(anchors,batch,dict(query_radius_m=.2))
    assert patch['owner'].flatten().tolist()==[1,0]
    assert torch.allclose(patch['distance'].flatten(),torch.tensor([.01,.11]))
