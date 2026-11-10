from pathlib import Path
import numpy as np
import pytest
from scenemi.data.cache_integrity import digest, publish_exact, amass_arrays, pinned_metadata


def test_repair_requires_original_digest_and_keeps_corrupt_evidence(tmp_path):
    source, target, backup = [tmp_path/x for x in ('original','cache','backup')]
    source.write_bytes(b'original pinned bytes')
    target.write_bytes(b'corrupt bytes')
    publish_exact(source, target, digest(source), backup)
    assert target.read_bytes() == source.read_bytes()
    assert backup.read_bytes() == b'corrupt bytes'


def test_repair_rejects_nonidentical_reconstruction(tmp_path):
    source, target, backup = [tmp_path/x for x in ('rebuilt','cache','backup')]
    source.write_bytes(b'another plausible array')
    target.write_bytes(b'bad cache')
    with pytest.raises(ValueError, match='original bytes'):
        publish_exact(source, target, '0'*64, backup)
    assert target.read_bytes() == b'bad cache' and not backup.exists()


def test_amass_rebuild_preserves_original_single_frame_drop(tmp_path):
    source = tmp_path/'source.npz'
    np.savez(source, root_orient=np.ones((5,3),np.float32), pose_body=np.zeros((5,63),np.float32),
             joints=np.arange(5*22*3,dtype=np.float32).reshape(5,22,3), betas=np.ones((5,16),np.float32))
    arrays = amass_arrays(source, digest(source))
    assert arrays['poses'].shape == (4,66) and arrays['joints'].shape == (4,22,3)
    assert arrays['joints'][0,0,0] == 66
    assert arrays['betas'].shape == (16,)
    with pytest.raises(ValueError, match='NPZ changed'):
        amass_arrays(source, '0'*64)


def test_metadata_restoration_only_uses_an_original_digest_match(tmp_path):
    source = tmp_path/'metadata.json'
    source.write_text('{"original":true}')
    expected = digest(source)
    mirror = tmp_path/'verified'
    pinned_metadata(source, expected, mirror)
    source.write_bytes(b'corrupt')
    records = []
    raw = pinned_metadata(source, expected, mirror, True, records)
    assert raw == b'{"original":true}' and records[0]['restored_sha256'] == expected
    source.write_bytes(b'damaged again')
    (mirror/(expected+'.json')).write_bytes(b'untrusted recovery')
    with pytest.raises(ValueError, match='no verified recovery copy'):
        pinned_metadata(source, expected, mirror, True)
    assert source.read_bytes() == b'damaged again'
