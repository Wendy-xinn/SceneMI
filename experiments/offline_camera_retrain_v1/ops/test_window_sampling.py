"""CPU checks for unbiased window epochs and uninterrupted checkpoint replay."""
import copy
from collections import Counter
import unittest
from unittest import mock

import numpy as np

from . import window_sampling as sampling


OPTIONS = dict(protocol=sampling.PROTOCOL, training_mode="exo")


class Fixture:
    def __init__(self, split="train"):
        self.split = split
        self.frames = 192
        self.rng = np.random.default_rng(11)
        self.members = {group: [] for group in ("rich", "trumans", "camera_wearer", "interactee")}
        for group, sizes in (("rich", [1, 2]), ("trumans", [9, 5]),
                             ("camera_wearer", [3]), ("interactee", [3])):
            for index, size in enumerate(sizes):
                recording = "shared_ego_recording" if group in ("camera_wearer", "interactee") else f"{group}_{index}"
                meta = dict(sequence_id=f"{group}_{index}", recording=recording,
                            path=f"/{group}/{index}", source_first=20,
                            input_sha256="fixed-source", allowed_modes=["ego", "exo"])
                self.members[group].append((meta, np.arange(size, dtype=np.int64) * 64))

    def starts_for(self, group, member, mode):
        return self.members[group][member][1]

    def sample(self, group, mode, **kwargs):
        return {}, dict(group=group, mode=mode, yaw=float(self.rng.random()), **kwargs)


def identity(row):
    return row["group"], row["mode"], row["member"], row["start"]


class NaturalWindowTests(unittest.TestCase):
    def make(self, data=None, seed=17):
        data = Fixture() if data is None else data
        return sampling.NaturalWindowSampler(data, sampling.natural_training_plan(data), OPTIONS, seed)

    def test_complete_epochs_visit_every_window_once_with_natural_counts(self):
        sampler = self.make()
        expected = {identity(row) for row in sampler._rows}
        orders = []
        for _ in range(3):
            rows = [sampler.draw() for _ in range(sampler.total_windows)]
            self.assertEqual({identity(row) for row in rows}, expected)
            self.assertEqual(len(rows), len(expected))
            self.assertEqual(Counter(row["group"] for row in rows),
                             dict(rich=3, trumans=14, camera_wearer=3, interactee=3))
            self.assertTrue(all(row["cooldown_fallback"] is None and row["overlap_fallback"] is None for row in rows))
            orders.append([identity(row) for row in rows])
        self.assertNotEqual(orders[0], orders[1])
        coverage = sampler.coverage_summary()
        self.assertEqual(coverage["completed_cycles"], 3)
        self.assertEqual(coverage["unique_windows_seen"], 23)
        self.assertEqual(coverage["repeated_presentations"], 46)
        self.assertEqual(coverage["min_presentations_per_window"], 3)
        self.assertEqual(coverage["max_presentations_per_window"], 3)
        # Two EgoBody targets share one physical recording, not two.
        self.assertEqual(coverage["physical_recordings"], 5)

    def test_resume_before_at_and_after_epoch_boundary_is_exact(self):
        for prefix in (0, 1, 22, 23, 24, 71):
            with self.subTest(prefix=prefix):
                sampler = self.make()
                for _ in range(prefix):
                    sampler.draw()
                state = sampler.state_dict()
                restored = self.make(seed=999)
                restored.load_state_dict(state)
                self.assertEqual([sampler.draw() for _ in range(90)],
                                 [restored.draw() for _ in range(90)])
                self.assertEqual(sampler.state_dict(), restored.state_dict())
                self.assertEqual(state["draws"], prefix)

    def test_rejects_wrong_population_proportions_and_non_train(self):
        data = Fixture()
        wrong = sampling.natural_training_plan(data)
        wrong[0]["probability"] += .1
        with self.assertRaisesRegex(ValueError, "natural eligible"):
            sampling.NaturalWindowSampler(data, wrong, OPTIONS, 1)
        with self.assertRaisesRegex(ValueError, "VAL/test"):
            self.make(Fixture(split="validation"))
        with self.assertRaisesRegex(ValueError, "EXO-only"):
            sampling.validate_sampling_options(OPTIONS, 1.)
        with self.assertRaisesRegex(ValueError, "cooldown"):
            sampling.validate_sampling_options(dict(OPTIONS, recording_cooldown=8), 0.)

    def test_population_uses_only_eligible_exo_starts_and_rejects_duplicate_slots(self):
        data = Fixture()
        data.members["rich"][0][0]["allowed_modes"] = ["ego"]
        data.members["trumans"][1] = (data.members["trumans"][1][0], np.array([], dtype=np.int64))
        sampler = self.make(data)
        self.assertEqual(sampler.total_windows, 17)
        for _ in range(17):
            row = sampler.draw()
            self.assertNotEqual((row["group"], row["member"]), ("rich", 0))
            self.assertNotEqual((row["group"], row["member"]), ("trumans", 1))
        invalid = Fixture()
        invalid.members["rich"][1][1][1] = 0
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.make(invalid)

    def test_resume_rejects_changed_source_or_window_without_mutating_state(self):
        state = self.make().state_dict()
        for field in ("start", "input_sha256", "source_first"):
            changed = Fixture()
            if field == "start":
                changed.members["rich"][0][1][0] += 1
            else:
                changed.members["rich"][0][0][field] = "changed"
            target = self.make(changed)
            before = target.state_dict()
            with self.assertRaisesRegex(ValueError, "changed"):
                target.load_state_dict(state)
            self.assertEqual(before, target.state_dict())

    def test_corrupt_checkpoint_coverage_and_permutation_are_rejected(self):
        sampler = self.make()
        sampler.draw()
        original = sampler.state_dict()
        for kind in ("duplicate", "count", "cursor", "fractional"):
            state = copy.deepcopy(original)
            if kind == "duplicate":
                state["permutation"][0] = state["permutation"][1]
            elif kind == "count":
                state["window_draw_counts"][0] += 1
            elif kind == "cursor":
                state["cursor"] += 1
            else:
                state["permutation"][0] = .5
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                sampler.load_state_dict(state)
            self.assertEqual(original, sampler.state_dict())

    def test_scoped_install_restores_legacy_bindings_on_exception(self):
        from experiments.offline_camera_retrain_v1 import native30_sampling as legacy
        previous = legacy.StratifiedRecordingSampler, legacy.validate_sampling_options
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with sampling.install_natural_window_sampler():
                self.assertIs(legacy.StratifiedRecordingSampler, sampling.NaturalWindowSampler)
                legacy.validate_sampling_options(OPTIONS, 0.)
                with self.assertRaisesRegex(RuntimeError, "cannot nest"):
                    with sampling.install_natural_window_sampler():
                        pass
                raise RuntimeError("sentinel")
        self.assertEqual(previous, (legacy.StratifiedRecordingSampler, legacy.validate_sampling_options))

    def test_real_sample_item_and_native_data_resume_preserve_draws_and_yaw(self):
        from experiments.offline_camera_retrain_v1.native30_data import Native30Data
        from experiments.offline_camera_retrain_v1.prepare_native30 import sample_item
        fixture = Fixture()
        fixture.sampler_state = Native30Data.sampler_state.__get__(fixture)
        plan = sampling.natural_training_plan(fixture)
        config = dict(sampler=OPTIONS, seed=55, ego_head_only_probability=1., scene_relations={})
        builder = mock.Mock()
        builder.build.side_effect = lambda data, sample, ident, mode, joints, **kwargs: (sample, None, None)
        condition_rng = np.random.default_rng(37)
        with sampling.install_natural_window_sampler():
            for _ in range(17):
                sample_item(fixture, builder, config, plan, condition_rng)
            state = fixture.sampler_state()
            expected = [sample_item(fixture, builder, config, plan, condition_rng)[1] for _ in range(80)]
            restored = Fixture()
            Native30Data.load_sampler_state(restored, state)
            actual = [sample_item(restored, builder, config, plan, condition_rng)[1] for _ in range(80)]
        self.assertEqual(expected, actual)
        self.assertIsNone(actual[0]["sampling"]["overlap_fallback"])
        self.assertEqual(state["recording_sampler"]["draws"], 17)
        self.assertIsNone(builder.build.call_args.args[4])


if __name__ == "__main__":
    unittest.main()
