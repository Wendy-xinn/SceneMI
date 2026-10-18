"""Reject misleading playback pairs and bind tiny saved archives to identity."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.offline_camera_retrain_v1.ops import motion_viewer as viewer


class PairingTests(unittest.TestCase):
    def arrays(self):
        return dict(truth=np.zeros((2, 201), np.float32), rest=np.zeros((22, 3), np.float32),
            clean_prediction=np.zeros((2, 201), np.float32), noisy_prediction=np.zeros((2, 201), np.float32),
            clean_observations=np.zeros((2, 22, 9), np.float32), noisy_observations=np.zeros((2, 22, 9), np.float32),
            clean_observation_meta=np.zeros((2, 22, 5), np.float32), noisy_observation_meta=np.zeros((2, 22, 5), np.float32))

    def metadata(self, mode="exo", step=0):
        return dict(identity=dict(mode=mode, sequence_id="same-recording", start=100,
            origin=[0., 0., 0.], anchor=np.eye(3).tolist()), step=step)

    def case(self):
        return dict(label="paired fixture", models=[
            dict(key="base41k", label="before", case=dict(label="before")),
            dict(key="adapter_v2", label="after", case=dict(label="after"))])

    def prepare(self, arrays, metadata):
        values = list(zip(arrays, metadata))
        with patch.object(viewer, "load_source", side_effect=values), patch.object(viewer, "prepare_case",
                side_effect=lambda case, *_: dict(case, cache="unused", step=0)):
            return viewer.prepare_review_case(self.case(), Path("."), {}, Path("unused"))

    def test_same_exo_inputs_allow_stage_zero_and_thousand_without_equal_predictions(self):
        before, after = self.arrays(), self.arrays()
        after["clean_prediction"] += 1
        result = self.prepare([before, after], [self.metadata(step=0), self.metadata(step=1000)])
        self.assertTrue(result["paired_inputs_verified"])
        self.assertEqual([entry["mode"] for entry in result["models"]], ["exo", "exo"])

    def test_each_paired_target_profile_or_condition_mismatch_is_rejected(self):
        for key in ("truth", "rest", "clean_observations", "noisy_observations",
                    "clean_observation_meta", "noisy_observation_meta"):
            with self.subTest(key=key):
                before, after = self.arrays(), self.arrays()
                after[key].flat[0] = 1
                with self.assertRaisesRegex(ValueError, "different conditions: " + key):
                    self.prepare([before, after], [self.metadata(), self.metadata(step=1000)])

    def test_same_mode_different_window_identity_is_rejected(self):
        after = self.metadata(step=1000)
        after["identity"]["start"] += 1
        with self.assertRaisesRegex(ValueError, "different observer/window identities"):
            self.prepare([self.arrays(), self.arrays()], [self.metadata(), after])

    def test_dual_mode_same_anchored_target_allowed_but_different_gt_or_rest_rejected(self):
        result = self.prepare([self.arrays(), self.arrays()], [self.metadata("ego", 41000), self.metadata("exo", 1000)])
        self.assertEqual([entry["mode"] for entry in result["models"]], ["ego", "exo"])
        for key in ("truth", "rest"):
            with self.subTest(key=key):
                ego, exo = self.arrays(), self.arrays()
                exo[key].flat[0] = 1
                with self.assertRaisesRegex(ValueError, "not the same anchored target/window"):
                    self.prepare([ego, exo], [self.metadata("ego", 41000), self.metadata("exo", 1000)])


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.identity = dict(mode="exo", sequence_id="fixture", start=0,
            native_body=dict(model="smpl", gender="neutral"))
        self.sha = "a" * 64
        self.common = dict(identity_json=np.asarray(json.dumps(self.identity)),
            truth=np.zeros((2, 201), np.float32), rest=np.zeros((22, 3), np.float32),
            prediction=np.zeros((2, 201), np.float32), observations=np.zeros((2, 22, 9), np.float32),
            observation_meta=np.zeros((2, 22, 5), np.float32),
            checkpoint_sha256=np.asarray(self.sha), step=np.asarray(1000))

    def write_case(self, *, step=1000, checkpoint=True, references=True, overrides=None):
        archives = {}
        for index, regime in enumerate(("clean", "noisy")):
            values = dict(self.common)
            values["step"] = np.asarray(step)
            if references:
                values["reference_motion"] = np.full((2, 201), 3 + index, np.float32)
            values.update((overrides or {}).get(regime, {}))
            path = self.root / (regime + ".npz")
            np.savez(path, **values)
            archives[regime] = path.name
        case = dict(label="fixture", archives=archives, step=step)
        if checkpoint:
            record = self.root / "checkpoint.json"
            record.write_text(json.dumps(dict(step=step, checkpoint_sha256=self.sha)))
            case["checkpoint_record"] = record.name
        return case

    def test_saved_reference_motion_fields_are_preserved_for_both_regimes(self):
        case = self.write_case()
        arrays, metadata = viewer.load_source(case, self.root)
        np.testing.assert_array_equal(arrays["clean_reference_motion"], np.full((2, 201), 3))
        np.testing.assert_array_equal(arrays["noisy_reference_motion"], np.full((2, 201), 4))
        self.assertEqual(metadata["checkpoint_identity_record"]["checkpoint_sha256"], self.sha)

    def test_multicase_archive_reference_fields_are_read_using_case_index(self):
        values = dict(identities_json=np.asarray(json.dumps([self.identity])), step=np.asarray(41000))
        for key in ("truth", "rest"):
            values["0_" + key] = self.common[key]
        for index, regime in enumerate(("clean", "noisy")):
            for key in ("prediction", "observations", "observation_meta"):
                values[f"0_{regime}_{key}"] = self.common[key]
            values[f"0_{regime}_reference_motion"] = np.full((2, 201), 6 + index, np.float32)
        path = self.root / "validation.npz"
        np.savez(path, **values)
        arrays, metadata = viewer.load_source(dict(label="case-index", archive=path.name, case_index=0, step=41000), self.root)
        self.assertEqual(metadata["step"], 41000)
        np.testing.assert_array_equal(arrays["clean_reference_motion"], np.full((2, 201), 6))
        np.testing.assert_array_equal(arrays["noisy_reference_motion"], np.full((2, 201), 7))

    def test_saved_reference_prevents_any_fallback_reference_reconstruction(self):
        import torch
        case = self.write_case()

        def decode(motion, *_):
            length = len(motion)
            return np.zeros((length, 3, 3), np.float32), np.array([[0, 1, 2]], np.int32), np.zeros((length, 22, 3), np.float32)

        def fk(motion, *_):
            return torch.zeros((motion.shape[0], motion.shape[1], 22, 3))

        model = SimpleNamespace(cpu=lambda: SimpleNamespace())
        with patch("experiments.offline_camera_retrain_v1.ops.viewer_geometry.reference_motion_from_inputs",
                    side_effect=AssertionError("Saved reference must take priority")), \
                patch("experiments.offline_camera_retrain_v1.native_body_mesh.decode_native_mesh", side_effect=decode), \
                patch("experiments.offline_camera_retrain_v1.export_rest_joints.load_model", return_value=model), \
                patch("experiments.offline_camera_retrain_v1.supervision.forward_kinematics", side_effect=fk), \
                patch.object(viewer, "observed_scene", return_value=({"visible_static": np.empty((0, 3), np.float32)}, {"status": "available"})), \
                patch("experiments.offline_camera_retrain_v1.ops.viewer_geometry.build_scene_layers", return_value=({}, {})), \
                patch.object(viewer, "existing_array", side_effect=FileNotFoundError("No optional source visibility fixture")), \
                patch.object(viewer, "reference_scene", return_value=(dict(scene_vertices=np.empty((0, 3), np.float32),
                    scene_faces=np.empty((0, 3), np.int32)), {"status": "available"})):
            prepared = viewer.prepare_case(case, self.root, {"scene": {}}, self.root / "cache")
        with np.load(prepared["cache"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["clean_reference_motion"], np.full((2, 201), 3))
            metadata = json.loads(str(saved["metadata_json"]))
        self.assertEqual(metadata["reference_input"]["clean"]["kind"], "original_saved_observation_reference")

    def test_requested_step_archive_step_and_checkpoint_record_must_agree(self):
        case = self.write_case()
        case["step"] = 999
        with self.assertRaisesRegex(ValueError, "requested step"):
            viewer.load_source(case, self.root)
        case = self.write_case(overrides={"noisy": {"step": np.asarray(999)}})
        with self.assertRaisesRegex(ValueError, "record step differs"):
            viewer.load_source(case, self.root)

    def test_checkpoint_sha_is_checked_for_each_archive(self):
        case = self.write_case(overrides={"noisy": {"checkpoint_sha256": np.asarray("b" * 64)}})
        with self.assertRaisesRegex(ValueError, "record SHA differs"):
            viewer.load_source(case, self.root)

    def test_checkpoint_model_and_guidance_manifest_fields_are_checked(self):
        case = self.write_case(overrides={regime: {"model": np.asarray("base41k"), "guidance": np.asarray("original")}
            for regime in ("clean", "noisy")})
        case["model"] = "different-model"
        with self.assertRaisesRegex(ValueError, "manifest model differs"):
            viewer.load_source(case, self.root)
        case["model"], case["guidance"] = "base41k", "different-guidance"
        with self.assertRaisesRegex(ValueError, "manifest guidance differs"):
            viewer.load_source(case, self.root)

    def prepared_fixture(self):
        source = self.write_case()
        arrays, metadata = viewer.load_source(source, self.root)
        folder = self.root / "cache"
        folder.mkdir(exist_ok=True)
        cache, record = folder / "playback.npz", folder / "MANIFEST.json"
        metadata["viewer_sha256"] = "historical UI source; no geometry rebuild required"
        np.savez(cache, **arrays, metadata_json=np.asarray(json.dumps(metadata)))
        record.write_text(json.dumps(metadata))
        source.update(cache=str(cache), cache_manifest=str(record))
        manifest = dict(schema=viewer.SCHEMA, cases=[dict(label="prepared fixture",
            paired_inputs_verified=True, models=[dict(key="base41k", label="base", mode="exo", case=source)])])
        return manifest, arrays, metadata

    def test_serve_prepared_accepts_old_ui_source_when_original_arrays_still_match(self):
        manifest, _, _ = self.prepared_fixture()
        result = viewer.validate_prepared_cases(manifest, self.root)
        self.assertIs(result, manifest["cases"])

    def test_serve_prepared_rejects_changed_source_or_cache_motion_or_reference(self):
        for kind in ("source", "prediction", "reference"):
            with self.subTest(kind=kind):
                manifest, arrays, metadata = self.prepared_fixture()
                if kind == "source":
                    source = self.root / "clean.npz"
                    with np.load(source, allow_pickle=False) as saved:
                        values = {key: saved[key].copy() for key in saved.files}
                    values["prediction"][0, 0] += 1
                    np.savez(source, **values)
                    expected = "metadata differs: sources"
                else:
                    key = "clean_prediction" if kind == "prediction" else "clean_reference_motion"
                    arrays[key][0, 0] += 1
                    np.savez(self.root / "cache/playback.npz", **arrays,
                        metadata_json=np.asarray(json.dumps(metadata)))
                    expected = "array differs: clean_prediction" if kind == "prediction" else "input reference differs"
                with self.assertRaisesRegex(ValueError, expected):
                    viewer.validate_prepared_cases(manifest, self.root)


if __name__ == "__main__":
    unittest.main()
