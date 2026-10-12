"""Official split boundaries and chair persistence across motion windows."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.offline_camera_retrain_v1.native30_splits import (
    official_recordings, official_sources,
    split_summary, validate_official_bundle)
from experiments.offline_camera_retrain_v1.native30_export import inventory
from experiments.offline_camera_retrain_v1.native30_geometry import Native30SurfaceWindow
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_memory import ObservedStaticPrefix
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder


class OfficialSplitTests(unittest.TestCase):
    def test_official_counts_and_all_legacy_scopes_reassigned(self):
        summary=split_summary()
        self.assertEqual([summary[s]['official_recordings'] for s in ('train','validation','test')],[65,17,43])
        self.assertEqual([summary[s]['available_recordings'] for s in ('train','validation','test')],[65,17,43])
        rows=official_sources(); official=official_recordings()
        self.assertEqual(len(rows),384)
        self.assertTrue(any(r['legacy_split']=='test' and r['split']=='train' for r in rows))
        for r in rows:
            self.assertEqual(r['split'],official[r['recording']])
            self.assertEqual(r['pose_profile_split'],r['legacy_split'])
        sets=[{r['recording'] for r in rows if r['split']==s} for s in ('train','validation','test')]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

    def test_training_rejects_relabelled_test_and_legacy(self):
        row=next(r for r in official_sources() if r['split']=='test')
        validate_official_bundle(row,'test')
        with self.assertRaisesRegex(ValueError,'wrong official split'):
            validate_official_bundle(dict(row,split='train'),'train')
        with self.assertRaisesRegex(ValueError,'newly indexed'):
            validate_official_bundle(dict(row,split_protocol='legacy-internal-scene-v1'),'test')

    def test_rich_test_does_not_substitute_validation(self):
        self.assertFalse(any(r['group']=='rich' for r in inventory('test')))
        self.assertEqual({r['sequence_id'] for r in inventory('val')},
                         {r['sequence_id'] for r in inventory('validation')})

    def test_exo_only_recordings_do_not_enter_ego_sampling(self):
        rows=[r for r in official_sources() if r['allowed_modes']==['exo']]
        self.assertEqual(len(rows),4)
        self.assertEqual({r['split'] for r in rows},{'train','test'})
        data=Native30Data.__new__(Native30Data)
        data.members={g:[] for g in ('trumans','camera_wearer','interactee','rich')}
        data.members['camera_wearer']=[(dict(allowed_modes=['ego','exo']),np.arange(4)),
                                       (dict(allowed_modes=['exo']),np.arange(1000))]
        plan=data.sampling_plan()
        self.assertAlmostEqual(sum(r['probability'] for r in plan if r['mode']=='ego'),.7)
        self.assertAlmostEqual(sum(r['probability'] for r in plan if r['mode']=='exo'),.3)
        with self.assertRaisesRegex(ValueError,'unavailable'):
            data.sample('camera_wearer','ego',member=1)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.meta=dict(path='/observations/partA',split='train',group='camera_wearer',
            dataset='egobody',recording='recording_A',sequence_id='partA',source_first=0,
            start=2,origin=[0,0,0],anchor=np.eye(3).tolist(),frames=6,
            native_body=dict(model='smpl',gender='male',betas=[0.]*10,scale=1.),actual_virtual_mount=None)
        # Chair observed only before the requested two-frame motion window.
        p=np.zeros((6,3,3),np.float32); owner=np.full((6,3),-1,np.int32)
        p[0,0]=[.2,0,0]; owner[0,0]=0
        p[:,1]=np.c_[np.arange(6)+2,np.zeros((6,2))];owner[:,1]=1
        p[5,2]=[9,0,0];owner[5,2]=0  # future chair must remain unknown
        self.arrays=dict(visible_frame_points_scenemi_yup=p,visible_frame_owner=owner,
            exo_frame_points_world=p.copy()+[0,0,1],exo_frame_owner=owner.copy(),
            camera_position_scenemi_yup=np.zeros((6,3)),source_frame_ids=np.arange(6),
            observation_valid=np.ones(6,bool),exo_sensor_to_world=np.eye(4),
            exo_intrinsics=np.ones(4),exo_trajectory_world=np.zeros((6,22,3)),
            exo_position_available=np.ones((6,22)),exo_position_confidence=np.ones((6,22)))
        self.opened=[]

    def load(self,folder,key):
        self.opened.append(key)
        self.assertNotIn(key,('motion_world','contact_target','person_0_vertices'))
        return self.arrays[key]

    def test_chair_survives_window_and_dynamic_has_no_ghost(self):
        memory=ObservedStaticPrefix(self.load,[self.meta])
        static=memory.window(self.meta,'ego',2)
        np.testing.assert_allclose(static[0],[[.2,0,0]])
        p=self.arrays['visible_frame_points_scenemi_yup'][2:4]
        o=self.arrays['visible_frame_owner'][2:4]
        scene=Native30SurfaceWindow(p,o,np.zeros((2,3)),static_memory=static)
        for t,cloud in enumerate(scene.dynamic):
            np.testing.assert_allclose(cloud['points'],[[4+t,0,0]])
        query=scene.query(np.broadcast_to([.2,0,0],(2,1,3)),np.broadcast_to(np.eye(3),(2,1,3,3)))
        self.assertTrue(query['valid'][:,:,0].any())
        self.assertTrue((scene.static['owners']==0).all())
        self.assertNotIn(9,scene.points[:,0])

    def test_random_window_order_future_filter_and_world_anchor(self):
        memory=ObservedStaticPrefix(self.load,[self.meta])
        later=dict(self.meta,start=4)
        self.assertEqual(len(memory.window(later,'ego',2)[0]),2)
        early=dict(self.meta,start=0)
        self.assertEqual(len(memory.window(early,'ego',2)[0]),1)
        a=Rotation.from_rotvec([0,1.2,0]).as_matrix();origin=np.array([1,2,3])
        shifted=dict(self.meta,anchor=a.tolist(),origin=origin.tolist())
        p,c=memory.window(shifted,'ego',2)
        np.testing.assert_allclose(p@a+origin,[[.2,0,0]],atol=1e-6)
        np.testing.assert_allclose(c@a+origin,[[0,0,0]],atol=1e-6)

    def test_prior_source_segment_persists_and_observers_are_isolated(self):
        part_b=dict(self.meta,path='/observations/partB',sequence_id='partB',source_first=6,start=0)
        other=dict(self.meta,path='/observations/other',group='interactee')
        def load(path,key):
            value=self.load(path,key).copy()
            if path==part_b['path']:
                if key=='source_frame_ids':value+=6
                if key.endswith('_owner'):value[value==0]=-1
            if path==other['path'] and key.endswith('_points_scenemi_yup'):value+=100
            return value
        memory=ObservedStaticPrefix(load,[self.meta,part_b,other])
        # PartB sees no static surfaces itself, but the chair from PartA survives.
        p,_=memory.window(part_b,'ego',2)
        np.testing.assert_allclose(p[:,0],[.2,9])
        exo,_=memory.window(self.meta,'exo',2)
        np.testing.assert_allclose(exo,[[.2,0,1]])
        other_map,_=memory.window(other,'ego',2)
        np.testing.assert_allclose(other_map,[[100.2,100,100]],atol=1e-5)

    def test_invalid_history_and_target_skin_are_not_memory(self):
        self.arrays['observation_valid'][0]=False
        memory=ObservedStaticPrefix(self.load,[self.meta])
        self.assertEqual(len(memory.window(self.meta,'ego',2)[0]),0)
        self.arrays['visible_frame_owner'][1,0]=100
        with self.assertRaisesRegex(ValueError,'Target body'):
            ObservedStaticPrefix(self.load,[self.meta]).window(self.meta,'ego',2)

    def test_builder_uses_prefix_in_actual_model_condition(self):
        class Input:
            pass
        data=Input();data.load=self.load;data.scene_memory=ObservedStaticPrefix(self.load,[self.meta])
        class Skin:
            def get(self,profile):return None
        camera=np.tile([0,0,0,1,0,0,0,1,0],(2,1)).astype(np.float32)
        sample=dict(camera=camera,rest=np.zeros((22,3),np.float32),
            body_type=np.array([1,0],np.float32),body_scale=np.ones(1,np.float32))
        prepared,scene,_=Native30ObservationBuilder(Skin(),{}).build(data,sample,self.meta,'exo')
        np.testing.assert_allclose(scene.points,[[.2,0,1]])
        self.assertGreater(float(prepared['occupancy'].sum()),0)
        self.assertNotIn('motion',prepared)


if __name__=='__main__':
    unittest.main()
