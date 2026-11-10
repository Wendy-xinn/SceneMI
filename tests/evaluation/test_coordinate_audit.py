from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from scenemi.coordinate_audit import paired_error, restore_world
from scenemi.data.amass_dataset import camera_anchor
from scenemi.data.rigid_world import RigidAmassSource, RigidSceneSource
from scenemi.geometry.body import AMASS_ZUP_TO_YUP, encode_body, PARENTS
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def fixture_scene():
    rng = np.random.default_rng(38)
    rest = np.zeros((22,3))
    rest[0] = [.1, .9, -.03]
    for j in range(1,22):
        rest[j] = rest[PARENTS[j]] + rng.normal(size=3)*.1
    local = Rotation.random(3*22, random_state=rng).as_matrix().reshape(3,22,3,3)
    motion = encode_body(local, rng.normal(size=(3,3)), rest)
    original = dict(rest=rest, motion_world=motion,
        camera_position_scenemi_yup=rng.normal(size=(3,3)),
        camera_rotation_scenemi_yup=local[:,0],
        static_full=rng.normal(size=(7,3)), static_points=rng.normal(size=(5,3)),
        static_normals=rng.normal(size=(5,3)),
        frame_points=rng.normal(size=(3,4,3)), frame_owner=np.zeros((3,4),int),
        frame_normals=rng.normal(size=(3,4,3)))
    original['frame_owner'][1,2] = -1
    return original


def test_raw_scene_transform_preserves_fk_and_intrinsics():
    original = fixture_scene()
    rotation = Rotation.from_euler('y',1.7).as_matrix(); translation = np.array([7.,0.,-3.])
    base = SimpleNamespace(cached=lambda row:original)
    transformed = RigidSceneSource(base,rotation,translation).cached({'path':'synthetic'})
    def fk(value):
        return forward_kinematics(torch.tensor(value['motion_world'])[None],
            torch.tensor(value['rest'],dtype=torch.tensor(value['motion_world']).dtype)[None]).numpy()[0]
    np.testing.assert_allclose(fk(transformed), fk(original)@rotation.T+translation, atol=1e-6)
    np.testing.assert_allclose(transformed['static_normals'], original['static_normals']@rotation.T, atol=1e-12)
    np.testing.assert_array_equal(transformed['frame_points'][1,2], np.zeros(3))
    np.testing.assert_array_equal(transformed['motion_world'][:,9:135], original['motion_world'][:,9:135])
    np.testing.assert_array_equal(transformed['rest'], original['rest'])


def test_amass_yup_world_transform_applied_in_zup_source():
    rng = np.random.default_rng(19)
    original = dict(poses=rng.normal(size=(3,66))*.2, joints=rng.normal(size=(3,22,3)), betas=np.zeros(16))
    rotation = Rotation.from_euler('y',-.9).as_matrix(); translation = np.array([-4.,0.,6.])
    base = SimpleNamespace(load=lambda *args:original)
    changed = RigidAmassSource(base,rotation,translation).load('synthetic','sha')
    z = AMASS_ZUP_TO_YUP
    np.testing.assert_allclose(changed['joints']@z.T, (original['joints']@z.T)@rotation.T+translation, atol=1e-12)
    np.testing.assert_allclose(z@Rotation.from_rotvec(changed['poses'][:,:3]).as_matrix(),
        rotation@z@Rotation.from_rotvec(original['poses'][:,:3]).as_matrix(), atol=1e-12)
    np.testing.assert_array_equal(changed['poses'][:,3:], original['poses'][:,3:])


def test_world_inverse_and_camera_canonicalization():
    rng = np.random.default_rng(9)
    points = rng.normal(size=(5,22,3)); origin = np.array([1.,0.,-2.])
    head = Rotation.from_rotvec([.1,1.2,-.2]).as_matrix()
    anchor = camera_anchor(head)
    rotation = Rotation.from_euler('y',-1.9).as_matrix(); translation = np.array([8.,0.,-7.])
    other_anchor = camera_anchor(rotation@head)
    new_origin = origin@rotation.T+translation
    canonical = (points-origin)@anchor.T
    changed = (points@rotation.T+translation-new_origin)@other_anchor.T
    np.testing.assert_allclose(canonical,changed,atol=1e-12)
    f1 = dict(origin=origin,anchor=anchor,ground=-1.6)
    f2 = dict(origin=new_origin,anchor=other_anchor,ground=-1.6)
    canonical[...,1] += 1.6; changed[...,1] += 1.6
    np.testing.assert_allclose(restore_world(canonical,f1),points,atol=1e-12)
    assert paired_error(canonical,changed,f1,f2,rotation,translation)['max_mm'] < 1e-10
    assert paired_error(canonical,changed+.01,f1,f2,rotation,translation)['mean_mm'] > 10


def test_reject_vertical_translation_and_reflection():
    for rotation,translation in ((np.eye(3),[0,1,0]),(np.diag([1,1,-1]),[0,0,0])):
        with pytest.raises(ValueError):
            RigidSceneSource(None,rotation,translation)
