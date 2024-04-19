import os
import sys
import unittest


try:
    from asr.asr import select_word_groups
    from asr.asr import split_long_segments
    from asr.asr import strip_segments
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from asr.asr import select_word_groups
    from asr.asr import split_long_segments
    from asr.asr import strip_segments


class TestASR(unittest.TestCase):
    def test_strip_segments_pos01(self):
        max_sound_duration = 5.5
        input_segments = [(0.1, 0.9), (0.95, 3.0), (3.0, 5.0)]
        target_segments = [(0.1, 0.9), (0.95, 3.0), (3.0, 5.0)]
        predicted_segments = strip_segments(input_segments, max_sound_duration)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(target_segments))
        for idx in range(len(target_segments)):
            self.assertIsInstance(predicted_segments[idx], tuple)
            self.assertEqual(len(predicted_segments[idx]), 2)
            self.assertAlmostEqual(predicted_segments[idx][0], target_segments[idx][0])
            self.assertAlmostEqual(predicted_segments[idx][1], target_segments[idx][1])

    def test_strip_segments_pos02(self):
        max_sound_duration = 5.5
        input_segments = [(-0.1, 0.9), (0.95, 3.0), (3.0, 5.0)]
        target_segments = [(0.0, 0.9), (0.95, 3.0), (3.0, 5.0)]
        predicted_segments = strip_segments(input_segments, max_sound_duration)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(target_segments))
        for idx in range(len(target_segments)):
            self.assertIsInstance(predicted_segments[idx], tuple)
            self.assertEqual(len(predicted_segments[idx]), 2)
            self.assertAlmostEqual(predicted_segments[idx][0], target_segments[idx][0])
            self.assertAlmostEqual(predicted_segments[idx][1], target_segments[idx][1])

    def test_strip_segments_pos03(self):
        max_sound_duration = 5.5
        input_segments = [(0.1, 0.9), (0.95, 3.0), (3.0, 5.8)]
        target_segments = [(0.1, 0.9), (0.95, 3.0), (3.0, 5.5)]
        predicted_segments = strip_segments(input_segments, max_sound_duration)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(target_segments))
        for idx in range(len(target_segments)):
            self.assertIsInstance(predicted_segments[idx], tuple)
            self.assertEqual(len(predicted_segments[idx]), 2)
            self.assertAlmostEqual(predicted_segments[idx][0], target_segments[idx][0])
            self.assertAlmostEqual(predicted_segments[idx][1], target_segments[idx][1])

    def test_split_long_segments_pos01(self):
        max_segment_size = 3
        input_segments = [(0.1, 0.9), (0.95, 3.0), (3.0, 5.0)]
        target_segments = [(0.1, 0.9), (0.95, 3.0), (3.0, 5.0)]
        predicted_segments = split_long_segments(input_segments, max_segment_size)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(target_segments))
        for idx in range(len(target_segments)):
            self.assertIsInstance(predicted_segments[idx], tuple)
            self.assertEqual(len(predicted_segments[idx]), 2)
            self.assertAlmostEqual(predicted_segments[idx][0], target_segments[idx][0])
            self.assertAlmostEqual(predicted_segments[idx][1], target_segments[idx][1])

    def test_split_long_segments_pos02(self):
        max_segment_size = 2
        input_segments = [(0.1, 0.9), (0.95, 3.0), (3.0, 5.0)]
        target_segments = [(0.1, 0.9), (1.1, 2.85), (3.0, 5.0)]
        predicted_segments = split_long_segments(input_segments, max_segment_size)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(target_segments), msg=f'{predicted_segments}')
        for idx in range(len(target_segments)):
            self.assertIsInstance(predicted_segments[idx], tuple)
            self.assertEqual(len(predicted_segments[idx]), 2)
            self.assertAlmostEqual(predicted_segments[idx][0], target_segments[idx][0])
            self.assertAlmostEqual(predicted_segments[idx][1], target_segments[idx][1])

    def test_split_long_segments_pos03(self):
        max_segment_size = 2
        input_segments = [(0.1, 0.9), (0.95, 4.0), (4.0, 6.0)]
        target_segments = [(0.1, 0.9), (1.1, 2.475), (2.475, 3.85), (4.0, 6.0)]
        predicted_segments = split_long_segments(input_segments, max_segment_size)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(target_segments), msg=f'{predicted_segments}')
        for idx in range(len(target_segments)):
            self.assertIsInstance(predicted_segments[idx], tuple)
            self.assertEqual(len(predicted_segments[idx]), 2)
            self.assertAlmostEqual(predicted_segments[idx][0], target_segments[idx][0])
            self.assertAlmostEqual(predicted_segments[idx][1], target_segments[idx][1])

    def test_select_word_groups_pos01(self):
        segment_size = 2
        words = [(0.1, 0.5), (0.7, 1.0), (1.1, 2.3), (2.7, 2.8), (3.6, 3.8), (3.8, 4.0)]
        target_groups = [[(0.1, 0.5)], [(0.7, 1.0), (1.1, 2.3)], [(2.7, 2.8)], [(3.6, 3.8), (3.8, 4.0)]]
        predicted_groups = select_word_groups(words, segment_size)
        self.assertIsInstance(predicted_groups, list)
        self.assertEqual(len(predicted_groups), len(target_groups))
        for group_idx in range(len(target_groups)):
            self.assertIsInstance(predicted_groups[group_idx], list)
            self.assertEqual(len(predicted_groups[group_idx]), len(target_groups[group_idx]))
            for word_idx in range(len(target_groups[group_idx])):
                self.assertIsInstance(predicted_groups[group_idx][word_idx], tuple)
                self.assertEqual(len(predicted_groups[group_idx][word_idx]), 2)
                self.assertAlmostEqual(predicted_groups[group_idx][word_idx][0], target_groups[group_idx][word_idx][0])
                self.assertAlmostEqual(predicted_groups[group_idx][word_idx][1], target_groups[group_idx][word_idx][1])

    def test_select_word_groups_pos02(self):
        segment_size = 2
        words = [(0.1, 0.5), (0.7, 1.0)]
        target_groups = [[(0.1, 0.5), (0.7, 1.0)]]
        predicted_groups = select_word_groups(words, segment_size)
        self.assertIsInstance(predicted_groups, list)
        self.assertEqual(len(predicted_groups), len(target_groups))
        for group_idx in range(len(target_groups)):
            self.assertIsInstance(predicted_groups[group_idx], list)
            self.assertEqual(len(predicted_groups[group_idx]), len(target_groups[group_idx]))
            for word_idx in range(len(target_groups[group_idx])):
                self.assertIsInstance(predicted_groups[group_idx][word_idx], tuple)
                self.assertEqual(len(predicted_groups[group_idx][word_idx]), 2)
                self.assertAlmostEqual(predicted_groups[group_idx][word_idx][0], target_groups[group_idx][word_idx][0])
                self.assertAlmostEqual(predicted_groups[group_idx][word_idx][1], target_groups[group_idx][word_idx][1])

    def test_select_word_groups_pos03(self):
        segment_size = 2
        words = [(0.1, 0.5), (3.7, 4.0)]
        target_groups = [[(0.1, 0.5)], [(3.7, 4.0)]]
        predicted_groups = select_word_groups(words, segment_size)
        self.assertIsInstance(predicted_groups, list)
        self.assertEqual(len(predicted_groups), len(target_groups))
        for group_idx in range(len(target_groups)):
            self.assertIsInstance(predicted_groups[group_idx], list)
            self.assertEqual(len(predicted_groups[group_idx]), len(target_groups[group_idx]))
            for word_idx in range(len(target_groups[group_idx])):
                self.assertIsInstance(predicted_groups[group_idx][word_idx], tuple)
                self.assertEqual(len(predicted_groups[group_idx][word_idx]), 2)
                self.assertAlmostEqual(predicted_groups[group_idx][word_idx][0], target_groups[group_idx][word_idx][0])
                self.assertAlmostEqual(predicted_groups[group_idx][word_idx][1], target_groups[group_idx][word_idx][1])

    def test_select_word_groups_pos04(self):
        segment_size = 2
        words = [(0.1, 4.0)]
        target_groups = [[(0.1, 4.0)]]
        predicted_groups = select_word_groups(words, segment_size)
        self.assertIsInstance(predicted_groups, list)
        self.assertEqual(len(predicted_groups), len(target_groups))
        for group_idx in range(len(target_groups)):
            self.assertIsInstance(predicted_groups[group_idx], list)
            self.assertEqual(len(predicted_groups[group_idx]), len(target_groups[group_idx]))
            for word_idx in range(len(target_groups[group_idx])):
                self.assertIsInstance(predicted_groups[group_idx][word_idx], tuple)
                self.assertEqual(len(predicted_groups[group_idx][word_idx]), 2)
                self.assertAlmostEqual(predicted_groups[group_idx][word_idx][0], target_groups[group_idx][word_idx][0])
                self.assertAlmostEqual(predicted_groups[group_idx][word_idx][1], target_groups[group_idx][word_idx][1])

    def test_select_word_groups_pos05(self):
        segment_size = 2
        words = []
        target_groups = [[]]
        predicted_groups = select_word_groups(words, segment_size)
        self.assertIsInstance(predicted_groups, list)
        self.assertEqual(len(predicted_groups), len(target_groups))


if __name__ == '__main__':
    unittest.main(verbosity=2)
