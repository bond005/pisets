import os
import sys
import unittest


try:
    from asr.asr import select_word_groups
    from asr.asr import strip_segments
    from asr.asr import join_short_segments_to_long_ones
    from asr.asr import find_repeated_tokens, find_tokens_in_text, remove_oscillatory_hallucinations
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from asr.asr import select_word_groups
    from asr.asr import strip_segments
    from asr.asr import join_short_segments_to_long_ones
    from asr.asr import find_repeated_tokens, find_tokens_in_text, remove_oscillatory_hallucinations


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

    def test_find_repeated_tokens_pos01(self):
        tokens = ['мама', 'мыла', 'раму']
        self.assertIsNone(find_repeated_tokens(tokens, 0))

    def test_find_repeated_tokens_pos02(self):
        tokens = ['мама', 'мыла', 'раму']
        self.assertIsNone(find_repeated_tokens(tokens, 2))

    def test_find_repeated_tokens_pos03(self):
        tokens = ['мама', 'мыла', 'раму', 'раму', 'раму', 'раму', 'раму']
        true_bounds = (2, 7)
        res = find_repeated_tokens(tokens, 0)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], int)
        self.assertIsInstance(res[1], int)
        self.assertEqual(res, true_bounds)

    def test_find_repeated_tokens_pos04(self):
        tokens = ['мама', 'мама', 'мама', 'мама', 'мыла', 'раму', 'раму', 'раму', 'раму', 'раму']
        true_bounds = (0, 4)
        res = find_repeated_tokens(tokens, 0)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], int)
        self.assertIsInstance(res[1], int)
        self.assertEqual(res, true_bounds)

    def test_find_repeated_tokens_pos05(self):
        tokens = ['мама', 'мама', 'мама', 'мама', 'мыла', 'раму', 'раму', 'раму', 'раму', 'раму']
        true_bounds = (5, 10)
        res = find_repeated_tokens(tokens, 2)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], int)
        self.assertIsInstance(res[1], int)
        self.assertEqual(res, true_bounds)

    def test_find_repeated_tokens_pos06(self):
        tokens = ['мама', 'мыла', 'раму', 'раму', 'раму', 'раму', 'раму']
        self.assertIsNone(find_repeated_tokens(tokens, 4))

    def test_find_tokens_in_text_pos01(self):
        s = 'Мама мыла раму.'
        tokens = ['мама', 'мыла']
        true_bounds = (0, 9)
        res = find_tokens_in_text(s, tokens)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], int)
        self.assertIsInstance(res[1], int)
        self.assertEqual(res, true_bounds)

    def test_find_tokens_in_text_pos02(self):
        s = 'Мама, мы все тяжело больны. Мама, я знаю, мы все сошли с ума.'
        tokens = ['мама', 'мы']
        true_bounds = (0, 8)
        res = find_tokens_in_text(s, tokens)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], int)
        self.assertIsInstance(res[1], int)
        self.assertEqual(res, true_bounds)

    def test_find_tokens_in_text_pos03(self):
        s = 'Мама, мы все тяжело больны. Мама, я знаю, мы все сошли с ума.'
        tokens = ['мы', 'все']
        true_bounds = (6, 12)
        res = find_tokens_in_text(s, tokens, 4)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], int)
        self.assertIsInstance(res[1], int)
        self.assertEqual(res, true_bounds)

    def test_find_tokens_in_text_pos04(self):
        s = 'Мама, мы все тяжело больны. Мама, я знаю, мы все сошли с ума.'
        tokens = ['мы', 'все']
        true_bounds = (42, 48)
        res = find_tokens_in_text(s, tokens, 8)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], int)
        self.assertIsInstance(res[1], int)
        self.assertEqual(res, true_bounds)

    def test_find_tokens_in_text_pos05(self):
        s = 'Мама, мы все тяжело больны. Мама, я знаю, мы все сошли с ума.'
        tokens = ['с']
        true_bounds = (55, 56)
        res = find_tokens_in_text(s, tokens)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], int)
        self.assertIsInstance(res[1], int)
        self.assertEqual(res, true_bounds)

    def test_find_tokens_in_text_neg01(self):
        s = 'Мама мыла раму.'
        tokens = ['мама', 'мыла']
        with self.assertRaises(RuntimeError):
            _ = find_tokens_in_text(s, tokens, 3)

    def test_find_tokens_in_text_neg02(self):
        s = 'Мама мыла раму.'
        tokens = ['мама', 'мы']
        with self.assertRaises(RuntimeError):
            _ = find_tokens_in_text(s, tokens)

    def test_find_tokens_in_text_neg03(self):
        s = 'Мама мыла раму.'
        tokens = []
        with self.assertRaises(RuntimeError):
            _ = find_tokens_in_text(s, tokens)

    def test_find_tokens_in_text_neg04(self):
        s = ''
        tokens = ['мама', 'мыла']
        with self.assertRaises(RuntimeError):
            _ = find_tokens_in_text(s, tokens)

    def test_remove_oscillatory_hallucinations_pos01(self):
        source_text = 'Сто процедуры безопасности'
        true_text = 'Сто процедуры безопасности'
        res = remove_oscillatory_hallucinations(source_text)
        self.assertIsInstance(res, str)
        self.assertEqual(res, true_text)

    def test_remove_oscillatory_hallucinations_pos02(self):
        source_text = ('Сто процедуры безопасности безопасности безопасности безопасности безопасности безопасности '
                       'безопасности безопасности безопасности безопасности безопасности безопасности безопасности '
                       'безопасности безопасности безопасности безопасности безопасности безопасности безопасности '
                       'безопасности безопасности безопасности безопасности безопасности безопасности безопасности '
                       'безопасности безопасности безопасности безопасности безопасности безопасности безопасности '
                       'безопасности безопасности безопасности безопасности безопасности безопасности безопасности '
                       'безопасности безопасности безопасности безопасности безопасности безопасности безопасности '
                       'безопасности безопасности безопасности безопасности безопасности безопасности безопасности '
                       'безопасности безопасности безопасности безопасности безопасности безопасности безопасности '
                       'безопасности безопасности безопасности, безопасности.')
        true_text = 'Сто процедуры безопасности.'
        res = remove_oscillatory_hallucinations(source_text)
        self.assertIsInstance(res, str)
        self.assertEqual(res, true_text)

    def test_join_short_segments_to_long_ones_pos01(self):
        source_segments = [(0.5, 2.5), (2.7, 3.92), (5.0, 7.5)]
        true_segments = [(0.5, 2.5), (2.7, 3.92), (5.0, 7.5)]
        predicted_segments = join_short_segments_to_long_ones(source_segments, 1)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(true_segments))
        for idx in range(len(true_segments)):
            self.assertIsInstance(predicted_segments[idx], tuple)
            self.assertEqual(len(predicted_segments[idx]), 2)
            self.assertAlmostEqual(predicted_segments[idx][0], true_segments[idx][0], delta=1e-6)
            self.assertAlmostEqual(predicted_segments[idx][1], true_segments[idx][1], delta=1e-6)

    def test_join_short_segments_to_long_ones_pos02(self):
        source_segments = [(0.5, 1.1), (2.7, 3.92), (5.0, 7.5)]
        true_segments = [(0.5, 1.1), (2.7, 3.92), (5.0, 7.5)]
        predicted_segments = join_short_segments_to_long_ones(source_segments, 1)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(true_segments))
        for idx in range(len(true_segments)):
            self.assertIsInstance(predicted_segments[idx], tuple)
            self.assertEqual(len(predicted_segments[idx]), 2)
            self.assertAlmostEqual(predicted_segments[idx][0], true_segments[idx][0], delta=1e-6)
            self.assertAlmostEqual(predicted_segments[idx][1], true_segments[idx][1], delta=1e-6)

    def test_join_short_segments_to_long_ones_pos03(self):
        source_segments = [(0.5, 1.1), (1.7, 3.92), (5.0, 7.5)]
        true_segments = [(0.5, 3.92), (5.0, 7.5)]
        predicted_segments = join_short_segments_to_long_ones(source_segments, 1)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(true_segments))
        for idx in range(len(true_segments)):
            self.assertIsInstance(predicted_segments[idx], tuple)
            self.assertEqual(len(predicted_segments[idx]), 2)
            self.assertAlmostEqual(predicted_segments[idx][0], true_segments[idx][0], delta=1e-6)
            self.assertAlmostEqual(predicted_segments[idx][1], true_segments[idx][1], delta=1e-6)

    def test_join_short_segments_to_long_ones_pos04(self):
        source_segments = [(0.5, 2.5), (2.7, 2.92), (5.0, 7.5)]
        true_segments = [(0.5, 2.92), (5.0, 7.5)]
        predicted_segments = join_short_segments_to_long_ones(source_segments, 1)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(true_segments))
        for idx in range(len(true_segments)):
            self.assertIsInstance(predicted_segments[idx], tuple)
            self.assertEqual(len(predicted_segments[idx]), 2)
            self.assertAlmostEqual(predicted_segments[idx][0], true_segments[idx][0], delta=1e-6)
            self.assertAlmostEqual(predicted_segments[idx][1], true_segments[idx][1], delta=1e-6)

    def test_join_short_segments_to_long_ones_pos05(self):
        source_segments = [(0.5, 2.5), (2.7, 2.92), (3.0, 7.5)]
        true_segments = [(0.5, 2.5), (2.7, 7.5)]
        predicted_segments = join_short_segments_to_long_ones(source_segments, 1)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(true_segments))
        for idx in range(len(true_segments)):
            self.assertIsInstance(predicted_segments[idx], tuple)
            self.assertEqual(len(predicted_segments[idx]), 2)
            self.assertAlmostEqual(predicted_segments[idx][0], true_segments[idx][0], delta=1e-6)
            self.assertAlmostEqual(predicted_segments[idx][1], true_segments[idx][1], delta=1e-6)

    def test_join_short_segments_to_long_ones_pos06(self):
        source_segments = [(0.5, 2.5), (2.7, 3.92), (4.0, 4.3)]
        true_segments = [(0.5, 2.5), (2.7, 4.3)]
        predicted_segments = join_short_segments_to_long_ones(source_segments, 1)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(true_segments))
        for idx in range(len(true_segments)):
            self.assertIsInstance(predicted_segments[idx], tuple)
            self.assertEqual(len(predicted_segments[idx]), 2)
            self.assertAlmostEqual(predicted_segments[idx][0], true_segments[idx][0], delta=1e-6)
            self.assertAlmostEqual(predicted_segments[idx][1], true_segments[idx][1], delta=1e-6)

    def test_join_short_segments_to_long_ones_pos07(self):
        source_segments = [(0.5, 2.5), (2.7, 3.92), (5.0, 5.5)]
        true_segments = [(0.5, 2.5), (2.7, 3.92), (5.0, 5.5)]
        predicted_segments = join_short_segments_to_long_ones(source_segments, 1)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(true_segments))
        for idx in range(len(true_segments)):
            self.assertIsInstance(predicted_segments[idx], tuple)
            self.assertEqual(len(predicted_segments[idx]), 2)
            self.assertAlmostEqual(predicted_segments[idx][0], true_segments[idx][0], delta=1e-6)
            self.assertAlmostEqual(predicted_segments[idx][1], true_segments[idx][1], delta=1e-6)

    def test_join_short_segments_to_long_ones_pos08(self):
        source_segments = [(0.5, 0.6)]
        true_segments = [(0.5, 0.6)]
        predicted_segments = join_short_segments_to_long_ones(source_segments, 1)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(true_segments))
        for idx in range(len(true_segments)):
            self.assertIsInstance(predicted_segments[idx], tuple)
            self.assertEqual(len(predicted_segments[idx]), 2)
            self.assertAlmostEqual(predicted_segments[idx][0], true_segments[idx][0], delta=1e-6)
            self.assertAlmostEqual(predicted_segments[idx][1], true_segments[idx][1], delta=1e-6)

    def test_join_short_segments_to_long_ones_pos09(self):
        source_segments = []
        true_segments = []
        predicted_segments = join_short_segments_to_long_ones(source_segments, 1)
        self.assertIsInstance(predicted_segments, list)
        self.assertEqual(len(predicted_segments), len(true_segments))


if __name__ == '__main__':
    unittest.main(verbosity=2)
