"""Full recording coverage and external observer integration contracts."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from experiments.offline_camera_retrain_v1.native30_splits import official_sources
from experiments.offline_camera_retrain_v1.native30_export import inventory,rows,PROTOCOL
from experiments.offline_camera_retrain_v1.data import SOURCE
from experiments.offline_camera_retrain_v1.native30_data import Native30Data,exo_window_starts
from experiments.offline_camera_retrain_v1.native30_exo import external_transforms,external_intrinsics
from experiments.offline_camera_retrain_v1.native30_memory import observer_stream_key
from experiments.offline_camera_retrain_v1.native30_window_index import valid_window_grid,PROTOCOL as WINDOW_PROTOCOL
from experiments.offline_camera_retrain_v1.native30_cache_io import mapped_sha256


class ThreeCorpusTests(unittest.TestCase):
    def test_window_grid_preserves_original_offsets_around_invalid_frames(self):
        rng=np.random.default_rng(27)
        for n in (1,16,28):
            valid=rng.random(n)>.15
            for frames in (1,4,7):
                for stride in (1,4,8):
                    expected=[s for s in range(0,max(0,n-frames+1),stride) if valid[s:s+frames].all()]
                    np.testing.assert_array_equal(valid_window_grid(valid,frames,stride),expected)
        valid=np.ones(12,bool);valid[0]=False
        np.testing.assert_array_equal(valid_window_grid(valid,4,4),[4,8])
        for invalid in (0,-1,True,1.5):
            with self.assertRaises(ValueError):valid_window_grid(valid,4,invalid)

    def test_persisted_windows_bind_grid_masks_and_sources_without_truncating_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'bundle';folder.mkdir()
            meta=dict(protocol=PROTOCOL,fps=30,split='validation',group='trumans',
                sequence_id='source',bundle_id='source',frames=12,input_sha256='fixture',
                diagnostic_only=True,source_first=5,source_last=16,allowed_modes=['ego','exo'])
            (folder/'metadata.json').write_text(json.dumps(meta))
            manifest=root/'manifest.jsonl';manifest.write_text(json.dumps(dict(meta,path=str(folder)))+'\n')
            np.save(folder/'source_frame_ids.npy',np.arange(5,17))
            np.save(folder/'observation_valid.npy',np.ones(12,bool))
            available=np.zeros((12,22),bool);available[0,15]=True;available[-1,20]=True
            np.save(folder/'exo_position_available.npy',available)
            meta['output_sha256']={p.name:mapped_sha256(p) for p in folder.glob('*.npy')}
            (folder/'metadata.json').write_text(json.dumps(meta))
            manifest.write_text(json.dumps(dict(meta,path=str(folder)))+'\n')
            spec=dict(protocol=WINDOW_PROTOCOL,status='complete',frames=4,window_stride=4,
                exo_max_empty_frames=3,source_manifest_sha256=mapped_sha256(manifest),
                bundle_paths=[str(folder)],files={})
            for mode,starts in [('ego',[0,4,8]),('exo',[0,8])]:
                file=root/f'validation_{mode}.npy';np.save(file,np.c_[np.zeros(len(starts),np.int64),starts])
                spec['files'][f'validation/{mode}']=dict(file=file.name,windows=len(starts),sha256=mapped_sha256(file))
            index=root/'dataset.json';index.write_text(json.dumps(spec))
            options=dict(diagnostic=True,frames=4,window_stride=4,exo_max_empty_frames=3,window_index=index)
            data=Native30Data(manifest,'validation',**options)
            np.testing.assert_array_equal(data.starts_for('trumans',0,'ego'),[0,4,8])
            np.testing.assert_array_equal(data.starts_for('trumans',0,'exo'),[0,8])
            self.assertEqual(data.scene_memory.entries[0]['frames'],12)
            with self.assertRaisesRegex(ValueError,'mismatch'):
                Native30Data(manifest,'validation',**dict(options,window_stride=8))
            # A damaged index is caught even though its descriptor still names
            # the same source and number of windows.
            file=root/'validation_exo.npy';np.save(file,np.array([[0,0],[0,4]],np.int64))
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                Native30Data(manifest,'validation',**options)
            # Updating the digest cannot make an unobserved window valid.
            spec['files']['validation/exo']['sha256']=mapped_sha256(file);index.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'validity/visibility'):
                Native30Data(manifest,'validation',**options)
            manifest.write_text(manifest.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'manifest mismatch'):
                Native30Data(manifest,'validation',**options)

    def test_long_all_joint_gaps_are_deferred_with_exact_window_boundaries(self):
        rng=np.random.default_rng(19)
        for frames in (4,192):
            for limit in (0,2,18,frames):
                for _ in range(12):
                    observed=rng.random(frames+35)>.3
                    observed[:19]=False;observed[-19:]=False
                    starts=np.arange(len(observed)-frames+1)
                    expected=[]
                    for start in starts:
                        run=longest=0
                        for value in observed[start:start+frames]:
                            run=0 if value else run+1;longest=max(longest,run)
                        if observed[start:start+frames].any() and longest<=limit:expected.append(start)
                    np.testing.assert_array_equal(exo_window_starts(observed,starts,frames,limit),expected)
        # Missing body parts remain eligible when another joint is observed.
        np.testing.assert_array_equal(exo_window_starts(np.ones(200,bool),np.arange(9),192,18),np.arange(9))
        for invalid in (-1,True,1.5):
            with self.assertRaises(ValueError):exo_window_starts(np.ones(4,bool),[0],4,invalid)

    def test_entirely_unobserved_windows_are_excluded_but_raw_memory_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'bundle';folder.mkdir()
            meta=dict(protocol=PROTOCOL,fps=30,split='validation',group='trumans',
                sequence_id='source',bundle_id='source',frames=12,input_sha256='fixture',
                diagnostic_only=True,source_first=0,source_last=11,allowed_modes=['exo'])
            (folder/'metadata.json').write_text(json.dumps(meta))
            manifest=root/'manifest.jsonl';manifest.write_text(json.dumps(dict(meta,path=str(folder)))+'\n')
            np.save(folder/'source_frame_ids.npy',np.arange(12));np.save(folder/'observation_valid.npy',np.ones(12,bool))
            available=np.zeros((12,22),bool);available[0,15]=True;available[-1,20]=True
            np.save(folder/'exo_position_available.npy',available)
            meta['output_sha256']={p.name:mapped_sha256(p) for p in folder.glob('*.npy')}
            (folder/'metadata.json').write_text(json.dumps(meta))
            manifest.write_text(json.dumps(dict(meta,path=str(folder)))+'\n')
            data=Native30Data(manifest,'validation',diagnostic=True,frames=4)
            np.testing.assert_array_equal(data.starts_for('trumans',0,'exo'),[0,8])
            self.assertEqual(data.scene_memory.entries[0]['frames'],12)
            # Construction needs no motion/contact labels to derive availability.
            self.assertFalse((folder/'motion_world.npy').exists())

    def test_every_raw_egobody_frame_has_exactly_one_scoped_observer_bundle(self):
        scoped=official_sources()
        for raw in rows(SOURCE/'sequences.jsonl'):
            if raw['dataset']!='egobody':continue
            parts=sorted((r['source_first_frame'],r['source_last_frame_exclusive']) for r in scoped
                if r['recording']==raw['recording_id'] and r['role']==raw['role'] and 'exo' in r['allowed_modes'])
            self.assertEqual(parts[0][0],raw['source_first_frame'])
            self.assertEqual(parts[-1][1],raw['source_last_frame_inclusive']+1)
            self.assertTrue(all(a[1]==b[0] for a,b in zip(parts,parts[1:])))

    def test_all_corpora_have_external_modes(self):
        for split in ('train','validation'):
            r=inventory(split)
            self.assertEqual({x['group'] for x in r if 'exo' in x['allowed_modes']},
                             {'trumans','rich','camera_wearer','interactee'})

    def test_both_modes_cover_all_available_corpora_and_preserve_total_mix(self):
        data=Native30Data.__new__(Native30Data)
        data.members={g:[(dict(allowed_modes=['ego','exo']),np.arange(n))]
                      for g,n in [('trumans',10000),('camera_wearer',100),('interactee',100),('rich',200)]}
        plan=data.sampling_plan()
        for mode,p in [('ego',.7),('exo',.3)]:
            self.assertAlmostEqual(sum(r['probability'] for r in plan if r['mode']==mode),p)
            self.assertEqual({r['group'] for r in plan if r['mode']==mode},set(data.members))
        # Square-root corpus weighting avoids raw-duration monopoly.
        self.assertLess(next(r['probability'] for r in plan if r['mode']=='exo' and r['group']=='trumans'),.27)

    def test_moving_and_static_cameras_use_the_observation_clock(self):
        matrices=np.broadcast_to(np.eye(4),(10,4,4)).copy();matrices[:,0,3]=np.arange(10)
        k=np.c_[np.arange(10),np.ones((10,3))]
        load=lambda folder,key:matrices if key=='exo_sensor_to_world' else k
        np.testing.assert_array_equal(external_transforms(load,'x',3,4)[:,0,3],np.arange(3,7))
        np.testing.assert_array_equal(external_intrinsics(load,'x',3,4)[:,0],np.arange(3,7))
        static=lambda folder,key:np.eye(4) if key=='exo_sensor_to_world' else np.ones(4)
        self.assertEqual(external_transforms(static,'x',3,4).shape,(4,4,4))

    def test_external_views_never_exchange_static_maps(self):
        meta=dict(split='train',group='rich',sequence_id='sequence',world_frame_id='world')
        a=dict(meta,exo_camera=dict(view_id='rich_000'))
        b=dict(meta,exo_camera=dict(view_id='rich_001'))
        self.assertNotEqual(observer_stream_key(a,'exo'),observer_stream_key(b,'exo'))
        self.assertEqual(observer_stream_key(a,'ego'),observer_stream_key(b,'ego'))


if __name__=='__main__':unittest.main()
