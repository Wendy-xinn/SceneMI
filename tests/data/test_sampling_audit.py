import unittest

from scenemi.data.sampling_audit import covered_frames, intervals_for_starts


class SamplingCoverageTests(unittest.TestCase):
    def test_overlaps_and_repeated_windows_count_each_source_frame_once(self):
        self.assertEqual(covered_frames([0, 0, 64, 128], 256), 256)
        self.assertEqual(intervals_for_starts([128, 0, 64, 0]), [[0, 256]])

    def test_rejected_gaps_are_not_filled_between_separate_windows(self):
        self.assertEqual(covered_frames([0, 256], 384), 256)
        self.assertEqual(intervals_for_starts([0, 256]), [[0, 128], [256, 384]])

    def test_out_of_recording_windows_fail_instead_of_being_clipped(self):
        with self.assertRaises(ValueError):
            covered_frames([129], 256)
        with self.assertRaises(ValueError):
            covered_frames([-1], 256)


if __name__=='__main__':
    unittest.main()
