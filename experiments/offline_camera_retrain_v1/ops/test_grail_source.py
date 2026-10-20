"""CPU checks for source clock, scale, USD transforms, and local pickle compatibility."""
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import grail_source as source


class GrailSourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_usd(self, path):
        from pxr import Usd, UsdGeom, Gf
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        xform = UsdGeom.Xform.Define(stage, "/model")
        xform.AddTranslateOp().Set(Gf.Vec3d(1., 2., 3.))
        stage.SetDefaultPrim(xform.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/model/mesh")
        mesh.CreatePointsAttr([(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        stage.GetRootLayer().Save()

    def make_record(self, n=121):
        folder = self.root / "data" / "sitting"
        for sub in ("recon", "object_usd", "meta"):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        name = "sitting__chair__000"
        pose = np.zeros((n, 165), np.float32)
        pose[:, 0] = np.linspace(0., .5, n)
        trans = np.tile([1., 2., 3.], (n, 1)).astype(np.float32)
        payload = {"human_data": {
            "poses": pose, "trans": trans, "betas": np.zeros((1, 10), np.float32),
            "scale": .5, "mocap_frame_rate": 30, "gender": "neutral"},
            "obj_data": {"obj_R": np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
                         "obj_t": trans, "obj_scale": np.array([[2.], [3.], [4.]])}}
        with (folder / "recon" / (name + ".pkl")).open("wb") as handle:
            pickle.dump(payload, handle)
        self.write_usd(folder / "object_usd" / (name + ".usd"))
        return source.inventory(self.root)[0]

    def test_clock_24_to_30(self):
        original, output, q = source.output_clock(121)
        self.assertEqual(len(output), 151)
        self.assertEqual(output[-1], 5.)
        self.assertEqual(q[-1], 120.)
        self.assertEqual(original[-1], output[-1])
        self.assertEqual(len(source.output_clock(241)[1]), 301)

    def test_no_extrapolation(self):
        original, output, q = source.output_clock(17, 24., 29.)
        self.assertLessEqual(output[-1], original[-1])
        self.assertLessEqual(q[-1], 16.)
        with self.assertRaises(ValueError):
            source.output_clock(1)

    def test_local_numpy_pickle_compatibility(self):
        path = self.root / "array.pkl"
        array = np.arange(9).reshape(3, 3)
        payload = pickle.dumps(array, protocol=2)
        payload = payload.replace(b"numpy.core.multiarray\n", b"numpy._core.multiarray\n")
        path.write_bytes(payload)
        before = set(sys.modules)
        np.testing.assert_array_equal(source.load_pickle(path), array)
        self.assertNotIn("numpy._core.multiarray", set(sys.modules) - before)

    def test_slerp_wraparound(self):
        aa = np.array([[0., 0., np.deg2rad(170)], [0., 0., np.deg2rad(-170)]])
        out = source.rotation_sample(aa, [0., .5, 1.])
        self.assertAlmostEqual(abs(float(out[1, 2])), np.pi, places=6)
        np.testing.assert_allclose(Rotation.from_rotvec(out[[0, 2]]).as_matrix(),
                                   Rotation.from_rotvec(aa).as_matrix(), atol=2e-7)

    def test_usd_applies_prim_transform_once(self):
        path = self.root / "test.usda"
        self.write_usd(path)
        vertices, faces, metadata = source.usd_mesh(path)
        np.testing.assert_array_equal(vertices[0], [1., 2., 3.])
        np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3]])
        self.assertEqual(metadata["up_axis"], "Z")

    def test_record_scale_clock_and_world(self):
        row = self.make_record()
        result = source.load_record(row)
        self.assertEqual(len(result["params"]["body_pose"]), 151)
        self.assertEqual(result["params"]["betas"].shape, (151, 10))
        np.testing.assert_array_equal(result["params"]["transl"][0], [2., 4., 6.])
        np.testing.assert_array_equal(result["object_vertices"][0], [2., 6., 12.])
        np.testing.assert_array_equal(result["object_positions"][0], [1., 3., -2.])
        np.testing.assert_array_equal(result["object_rotations"][0], source.WORLD_ROTATION)
        self.assertTrue(result["quality"]["raw_fps_conflicts_with_override"])
        self.assertFalse(result["quality"]["body_template_verified"])
        self.assertFalse(result["quality"]["reconstruction_is_physics_validated"])
        self.assertEqual(result["source_provenance"]["synthetic_ground"]["scenemi_world_plane"], [0., 1., 0., 0.])
        self.assertEqual(len(result["input_sha256"]), 64)
        # Analytic scale around source translation: importer divided transl first.
        local = np.array([.2, .4, .8])
        trans = np.array([1., 2., 3.])
        actual = .5 * (local + result["params"]["transl"][0])
        expected = .5 * local + trans
        np.testing.assert_allclose(actual, expected)

    def test_inventory_does_not_create_splits_or_load_pickle(self):
        row = self.make_record()
        Path(row["recon_path"]).write_bytes(b"not a valid pickle")
        rows = source.inventory(self.root)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["split"])
        self.assertEqual(rows[0]["object_family_id"], "sitting__chair")

    def test_invalid_object_rotation_rejected(self):
        row = self.make_record()
        payload = source.load_pickle(row["recon_path"])
        payload["obj_data"]["obj_R"][0, 0, 0] = -1.
        with Path(row["recon_path"]).open("wb") as handle:
            pickle.dump(payload, handle)
        with self.assertRaisesRegex(ValueError, "proper rotations"):
            source.load_record(row)


if __name__ == "__main__":
    unittest.main()
