"""Held-out RICH keeps person clocks, masks and train/test isolation intact."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scenemi.data.amass_prepare import sha256
from scenemi.data.heldout_geometry import RichTestSelection, HeldoutPreparedScene, HeldoutGeometryData
from scenemi.heldout_evaluation import aggregate


class HeldoutGeometryTest(unittest.TestCase):
    def selection(self, root, *, broken_clock=False):
        source = root/'person'; source.mkdir()
        valid = np.ones(256,bool); valid[130] = False
        clock = np.arange(5,261)
        if broken_clock:
            clock[200] += 1
        hashes = {}
        for name,value in [('observation_valid',valid),('source_frame_ids',clock)]:
            file = source/(name+'.npy'); np.save(file,value)
            hashes[file.name] = sha256(file)
        meta = dict(dataset='rich',group='rich',split='test',fps=30,frames=256,
            source_first=5,source_last=260,allowed_modes=['ego'],diagnostic_only=False,
            rich_test_protocol='rich-test-all-subjects-native30-ego-v1',
            native_body=dict(model='smplx'), source_sequence_id='recording',
            sequence_id='recording__person',subject_id='person',output_sha256=hashes)
        (source/'metadata.json').write_text(json.dumps(meta))
        # A different dataset's test path must never be opened by this adapter.
        rows = [dict(dataset='egobody',split='test',path='/missing/other/dataset'),dict(meta,path=str(source))]
        manifest = root/'manifest.jsonl'
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        policy = dict(manifest_sha256=sha256(manifest),recordings=1,person_streams=1,source_frames=256)
        return RichTestSelection(manifest,root/'verified',policy)

    def geometry(self):
        return dict(geometry=dict(voxel_m=.04,static_points=32,frame_points=4,
            normal_neighbours=16,normal_radius_m=.1,minimum_neighbours=6,
            minimum_spread_m=.005,maximum_planarity_ratio=.03))

    def test_no_window_crosses_missing_person_and_no_tail_is_added(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); selection = self.selection(root)
            data = HeldoutPreparedScene(selection,[0,0,0],self.geometry(),root/'cache',64)
            self.assertEqual(data.records[0]['starts'].tolist(),[0,1,2])
            self.assertEqual([c['start'] for c in data.cases],[0])
            self.assertEqual(data.inventory[0]['covered_frames'],128)
            with self.assertRaisesRegex(ValueError,'test-only'):
                selection.metadata(data.records[0],'train')
        with self.assertRaisesRegex(RuntimeError,'cannot be used for training'):
            HeldoutGeometryData.pairs(None)

    def test_compressed_or_nonmonotonic_source_clock_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); selection = self.selection(root,broken_clock=True)
            with self.assertRaisesRegex(ValueError,'clock'):
                HeldoutPreparedScene(selection,[0,0,0],self.geometry(),root/'cache',64)

    def test_window_and_recording_reductions_remain_distinct(self):
        rows = [dict(case=dict(family=f,sequence_id=f),metrics=dict(mpjpe_mm=v))
            for f,v in [('a',0),('a',0),('b',9)]]
        result = aggregate(rows)
        self.assertEqual(result['window_mean']['mpjpe_mm'],3)
        self.assertEqual(result['recording_mean']['mpjpe_mm'],4.5)


if __name__ == '__main__':
    unittest.main()
