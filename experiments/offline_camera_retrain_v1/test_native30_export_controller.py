"""Crash recovery must preserve evidence and never silently skip a source."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.offline_camera_retrain_v1.export_native30_all import copy_verified,isolated_worker,verify_existing,recover_preserved_read
from experiments.offline_camera_retrain_v1.native30_export import PROTOCOL,export_fingerprint
from experiments.offline_camera_retrain_v1.native30_cache_io import atomic_text,read_json_retry
from experiments.offline_camera_retrain_v1.native30_cache_io import mapped_sha256,verified_sha256

POPEN='experiments.offline_camera_retrain_v1.export_native30_all.subprocess.Popen'


class ExportRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.row=dict(split='train',sequence_id='source')

    def bundle(self):
        folder=self.root/'train/source';folder.mkdir(parents=True)
        payload=b'committed cache';(folder/'array.npy').write_bytes(payload)
        meta=dict(protocol=PROTOCOL,export_fingerprint=export_fingerprint(),
                  output_sha256={'array.npy':hashlib.sha256(payload).hexdigest()})
        (folder/'metadata.json').write_text(json.dumps(meta));return folder,meta

    def test_verified_source_never_recomputes(self):
        folder,meta=self.bundle()
        with patch(POPEN,side_effect=AssertionError('Valid source must not be recomputed')),patch(
            'experiments.offline_camera_retrain_v1.native30_cache_io.mapped_sha256',
            side_effect=['stale',meta['output_sha256']['array.npy']]):
            self.assertEqual(isolated_worker(self.row,self.root),dict(path=str(folder),**meta))

    def test_corruption_is_preserved_but_stale_protocol_is_rejected(self):
        folder,meta=self.bundle();(folder/'array.npy').write_bytes(b'corrupt')
        self.assertIsNone(verify_existing(self.row,self.root))
        quarantined=list((self.root/'_interrupted_bundles/train').iterdir())
        self.assertEqual(len(quarantined),1)
        self.assertEqual((quarantined[0]/'array.npy').read_bytes(),b'corrupt')
        folder,meta=self.bundle();meta['protocol']='different protocol'
        (folder/'metadata.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'Stale native30'):
            verify_existing(self.row,self.root)
        self.assertTrue(folder.exists())

    def test_native_signal_retries_in_new_process_and_returns_same_source(self):
        calls=[]
        class Child:
            def __init__(child,command,**kwargs):
                child.pid=999999;calls.append(command)
                if len(calls)==2:Path(command[-1]).write_text(json.dumps({'sequence_id':'source'}))
            def wait(child):return -11 if len(calls)==1 else 0
        with patch(POPEN,Child):
            self.assertEqual(isolated_worker(self.row,self.root),{'sequence_id':'source'})
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0],calls[1])

    def test_repeated_native_crash_is_failure_not_success_or_infinite_retry(self):
        calls=[]
        class Child:
            def __init__(child,command,**kwargs):child.pid=999999;calls.append(command)
            def wait(child):return -11
        with patch(POPEN,Child),self.assertRaisesRegex(RuntimeError,'exited -11: source'):
            isolated_worker(self.row,self.root)
        self.assertEqual(len(calls),3)

    def test_transient_replace_and_reader_gap_recover_without_partial_json(self):
        path=self.root/'progress.json';replace=Path.replace;attempts=[]
        def interrupted_replace(source,destination):
            attempts.append(1)
            if len(attempts)==1:raise FileNotFoundError('Transient DrvFS rename')
            return replace(source,destination)
        with patch.object(Path,'replace',interrupted_replace):
            atomic_text(path,json.dumps({'complete':True}))
        self.assertEqual(read_json_retry(path),{'complete':True})
        read=Path.read_text;attempts.clear()
        def interrupted_read(source,*args,**kwargs):
            attempts.append(1)
            if len(attempts)==1:raise FileNotFoundError('Transient reader gap')
            return read(source,*args,**kwargs)
        with patch.object(Path,'read_text',interrupted_read):
            self.assertEqual(read_json_retry(path),{'complete':True})

    def test_copied_bytes_are_rewritten_after_failed_verification(self):
        source=self.root/'source.npy';destination=self.root/'copy.npy'
        source.write_bytes(b'original cache bytes');checks=[]
        def interrupted_hash(path):
            if Path(path)==destination:
                checks.append(1)
                if len(checks)==1:destination.write_bytes(b'corrupted copy')
            return mapped_sha256(path)
        with patch('experiments.offline_camera_retrain_v1.export_native30_all.mapped_sha256',interrupted_hash):
            copy_verified(source,destination)
        self.assertEqual(destination.read_bytes(),source.read_bytes())
        self.assertEqual(len(checks),2)

    def test_persistently_bad_copy_cannot_be_committed(self):
        source=self.root/'source.npy';destination=self.root/'copy.npy'
        source.write_bytes(b'original cache bytes');checks=[]
        def bad_hash(path):
            if Path(path)==destination:checks.append(1);return 'mismatch'
            return mapped_sha256(path)
        with patch('experiments.offline_camera_retrain_v1.export_native30_all.mapped_sha256',bad_hash):
            with self.assertRaisesRegex(ValueError,'Copied cache bytes differ'):
                copy_verified(source,destination)
        self.assertEqual(len(checks),3)

    def test_stale_hash_read_refreshes_without_accepting_wrong_digest(self):
        with patch('experiments.offline_camera_retrain_v1.native30_cache_io.mapped_sha256',
                   side_effect=['stale','expected']) as read:
            self.assertEqual(verified_sha256(self.root/'array.npy','expected'),'expected')
        self.assertEqual(read.call_args_list[0].kwargs,dict(refresh=False))
        self.assertEqual(read.call_args_list[1].kwargs,dict(refresh=True))

    def test_persistent_hash_mismatch_remains_a_failure(self):
        # The independent buffered fallback also needs real, mismatching bytes.
        (self.root/'array.npy').write_bytes(b'wrong source bytes')
        with patch('experiments.offline_camera_retrain_v1.native30_cache_io.mapped_sha256',
                   return_value='wrong') as read,self.assertRaisesRegex(ValueError,'Cache SHA256 mismatch'):
            verified_sha256(self.root/'array.npy','expected')
        self.assertEqual(read.call_count,3)

    def test_isolated_original_bytes_are_reused_only_when_all_hashes_match(self):
        folder,meta=self.bundle();preserved=self.root/'_interrupted_bundles/train/source_1'
        preserved.parent.mkdir(parents=True);folder.replace(preserved)
        restored=recover_preserved_read(self.row,self.root)
        self.assertEqual(restored,dict(path=str(folder),**meta))
        self.assertTrue(preserved.exists())
        self.assertEqual((folder/'array.npy').read_bytes(),b'committed cache')

    def test_corrupt_or_numerically_invalid_preserved_bytes_are_never_reused(self):
        folder,meta=self.bundle();preserved=self.root/'_interrupted_bundles/train/source_1'
        preserved.parent.mkdir(parents=True);folder.replace(preserved)
        (preserved/'array.npy').write_bytes(b'corrupt')
        self.assertIsNone(recover_preserved_read(self.row,self.root))
        (preserved/'array.npy').write_bytes(b'committed cache')
        preserved.replace(preserved.with_name('source_numerical_2'))
        self.assertIsNone(recover_preserved_read(self.row,self.root))


if __name__=='__main__':unittest.main()
