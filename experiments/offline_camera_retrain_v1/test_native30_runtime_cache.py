"""Persistent cache isolation, rigid augmentation, and label-only contracts."""
import tempfile
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.offline_camera_retrain_v1.native30_memory import ObservedStaticMemory
from experiments.offline_camera_retrain_v1.native30_support_cache import TrainingSupportCache,ARRAY_KEYS
from experiments.offline_camera_retrain_v1.native30_observations import INPUT_KEYS
from experiments.offline_camera_retrain_v1.test_native30_query_runtime import fixture
from experiments.offline_camera_retrain_v1.native30_cache_io import verified_array,verified_sha256


class RuntimeCacheTests(unittest.TestCase):
    def test_buffered_checksum_fallback_still_requires_original_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'array.npy';np.save(path,np.arange(12))
            expected=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch('experiments.offline_camera_retrain_v1.native30_cache_io.mapped_sha256',return_value='wrong'):
                self.assertEqual(verified_sha256(path,expected),expected)
                with self.assertRaisesRegex(ValueError,'mismatch'):verified_sha256(path,'unknown')

    def test_drvfs_array_mirror_publishes_only_verified_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source.npy';np.save(path,np.arange(12,dtype=np.float32))
            expected=hashlib.sha256(path.read_bytes()).hexdigest();root=Path(folder)/'mirror'
            with patch.object(Path,'resolve',return_value=Path('/mnt/e/source.npy')):
                actual=verified_array(path,expected,root)
                self.assertTrue(np.array_equal(actual,np.arange(12)))
                self.assertTrue((root/(expected+'.npy')).exists())
                self.assertEqual(hashlib.sha256((root/(expected+'.npy')).read_bytes()).hexdigest(),expected)
                with self.assertRaisesRegex(ValueError,'mismatch'):verified_array(path,'wrong',root)
                self.assertFalse((root/'wrong.npy').exists())

    def test_static_map_roundtrip_and_observer_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            meta=dict(path=folder,split='train',group='rich',sequence_id='a',scene_id='room',source_first=0,
                      exo_camera=dict(view_id='001'),allowed_modes=['ego','exo'],output_sha256={'x':'immutable'})
            points=np.array([[[0.,0.,0.],[.03,0,0]],[[0.,0.,0.],[.06,0,0]]],np.float32)
            arrays=dict(visible_frame_points_scenemi_yup=points,visible_frame_owner=np.zeros((2,2),int),
                        exo_frame_points_world=points+1,exo_frame_owner=np.zeros((2,2),int),
                        source_frame_ids=np.arange(2),observation_valid=np.ones(2,bool),
                        camera_position_scenemi_yup=np.tile([0,1,0],(2,1)),exo_sensor_to_world=np.tile(np.eye(4),(2,1,1)))
            load=lambda _,key:arrays[key]
            root=Path(folder)/'cache'
            first=ObservedStaticMemory(load,[meta],cache_root=root)
            ego=first._map(meta,'ego');exo=first._map(meta,'exo')
            self.assertFalse(np.array_equal(ego[0],exo[0]))
            second=ObservedStaticMemory(load,[meta],cache_root=root)
            for mode,expected in [('ego',ego),('exo',exo)]:
                actual=second._map(meta,mode)
                self.assertTrue(all(np.array_equal(a,b) for a,b in zip(actual,expected)))
            self.assertEqual(second.cache_hits,2);self.assertEqual(second.cache_misses,0)

    def test_training_labels_transform_and_reuse_without_truth_in_input(self):
        with tempfile.TemporaryDirectory() as folder:
            identity=dict(path=folder,split='train',group='rich',sequence_id='a',mode='ego',start=0,
                          anchor=np.eye(3).tolist(),origin=[0,0,0],native_body={'shape':'a'},output_sha256={'x':'hash'})
            data=SimpleNamespace(split='train',runtime_cache_root=Path(folder),frames=3,
                members={'rich':[(identity,np.array([0,64]))]},quality_config={},
                scene_memory=SimpleNamespace(entries=[identity],protocol='offline',voxel=.025))
            data.sample=Mock(return_value=({},identity));builder=Mock();builder.build.return_value=({},None,None)
            targets={k:np.zeros((3,22),np.float32) for k in ARRAY_KEYS}
            for key in ('points','normals','velocities','vertices'):targets[key]=np.ones((3,67,3),np.float32)
            cache=TrainingSupportCache(data,builder)
            with patch('experiments.offline_camera_retrain_v1.native30_support_cache.training_support_targets',return_value=targets):
                first=cache.get(identity)
                turn=Rotation.from_rotvec([0,.8,0]).as_matrix()
                other=dict(identity,anchor=turn.tolist())
                rotated=cache.get(other)
                self.assertEqual(data.sample.call_count,1)
                self.assertTrue(np.allclose(rotated['points'],first['points']@turn.T))
                self.assertTrue(np.array_equal(rotated['target'],first['target']))
                self.assertNotEqual(cache.path(identity),cache.path(dict(identity,start=64)))
                self.assertNotEqual(cache.path(identity),cache.path(dict(identity,mode='exo')))
                self.assertFalse(any(k.startswith('support_') for k in INPUT_KEYS))

    def test_validation_cannot_create_weak_training_label_cache(self):
        with self.assertRaisesRegex(ValueError,'TRAIN-only'):
            TrainingSupportCache(SimpleNamespace(split='validation'),None)

    def test_nearest_boundary_ties_and_many_realistic_queries(self):
        scene,probes,rotations=fixture()
        for dtype in (np.float32,np.float64):
            expected=scene.query_reference(probes.astype(dtype),rotations)
            actual=scene.query_nearest(probes.astype(dtype),rotations)
            for key in expected:
                self.assertTrue(np.array_equal(expected[key][:,:,2,0],actual[key][:,:,2,0]),key)

if __name__=='__main__':unittest.main()
