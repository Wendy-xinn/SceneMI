"""Missing frontend estimates must not enter the motion initializer."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scenemi.data.exo_initialization import fill_estimate_gaps


class InitializationTests(unittest.TestCase):
    def test_gap_payload_is_ignored_and_estimated_neighbors_are_preserved(self):
        frames=128
        vectors=np.zeros((frames,22,3))
        vectors[...,1]=np.linspace(0,.8,frames)[:,None]
        rotations=Rotation.from_rotvec(vectors.reshape(-1,3)).as_matrix().reshape(frames,22,3,3)
        root=np.linspace([0.,1.,0.],[1.,1.,2.],frames)
        available=np.ones(frames,bool);available[40:64]=False
        original_rotation,original_root=rotations.copy(),root.copy()
        first=fill_estimate_gaps(rotations,root,available)
        rotations[~available]=np.nan;root[~available]=1e20
        poisoned=fill_estimate_gaps(rotations,root,available)
        np.testing.assert_array_equal(first[0],poisoned[0])
        np.testing.assert_array_equal(first[1],poisoned[1])
        np.testing.assert_array_equal(first[0][available],original_rotation[available])
        np.testing.assert_array_equal(first[1][available],original_root[available])
        np.testing.assert_allclose(first[0],original_rotation,atol=1e-12)
        np.testing.assert_allclose(first[1],original_root,atol=1e-12)

    def test_boundary_gap_holds_nearest_estimate_and_empty_track_is_rejected(self):
        rotations=np.tile(np.eye(3),(128,22,1,1));root=np.arange(384).reshape(128,3).astype(float)
        available=np.ones(128,bool);available[:16]=False;available[-16:]=False
        _,filled=fill_estimate_gaps(rotations,root,available)
        np.testing.assert_array_equal(filled[:16],np.tile(root[16],(16,1)))
        np.testing.assert_array_equal(filled[-16:],np.tile(root[-17],(16,1)))
        with self.assertRaises(ValueError):
            fill_estimate_gaps(rotations,root,np.zeros(128,bool))


if __name__=='__main__':
    unittest.main()
