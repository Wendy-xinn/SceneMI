import unittest

from experiments.offline_camera_retrain_v1.ops.grail_inventory import canonical_asset_family


class CanonicalAssetFamilyTests(unittest.TestCase):
    def test_up_down_results_share_asset(self):
        self.assertEqual(
            canonical_asset_family("results_collection/stairs_down/generation/mesh/generated/chair/model.obj"),
            canonical_asset_family("../results_collection/stairs_up/generation/mesh/generated/chair/model.obj"))

    def test_namespace_is_retained(self):
        self.assertNotEqual(canonical_asset_family("x/mesh/SynStairsV2/stairs_072/model.obj"),
                            canonical_asset_family("x/mesh/SynStairsV3/stairs_072/model.obj"))

    def test_uses_last_mesh_component(self):
        self.assertEqual(canonical_asset_family("x/mesh/previous/mesh/namespace/chair/model.obj"),
                         "grail/namespace/chair/model.obj")

    def test_rejects_unresolved_and_traversal(self):
        for path in (None, "chair/model.obj", "x/mesh/chair/model.obj", "x/mesh/ns/../chair/model.obj"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                canonical_asset_family(path)


if __name__ == "__main__":
    unittest.main()
