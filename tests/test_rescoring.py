import os
import sys
import unittest


try:
    from rescoring.rescoring import align, levenshtein
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from rescoring.rescoring import align, levenshtein


class TestRescoring(unittest.TestCase):
    def test_levenshtein01(self):
        s1 = 'нейросетиэтохорошо'
        s2 = 'нейросетиэтохорошо'
        true_matches = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10),
                        (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17)]
        calculated_matches = levenshtein(s1, s2)
        self.assertIsInstance(calculated_matches, list)
        self.assertEqual(len(true_matches), len(calculated_matches))
        self.assertEqual(calculated_matches, true_matches)

    def test_levenshtein02(self):
        s1 = 'нейросетиэтохорошо'
        s2 = 'нейросетиэтодахорошо'
        true_matches = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10),
                        (11, 11), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16), (15, 17), (16, 18), (17, 19)]
        calculated_matches = levenshtein(s1, s2)
        self.assertIsInstance(calculated_matches, list)
        self.assertEqual(len(true_matches), len(calculated_matches))
        self.assertEqual(calculated_matches, true_matches)

    def test_levenshtein03(self):
        s1 = 'нейросетиэтохорошо'
        s2 = 'нейросетиэтооченьхорошо'
        calculated_matches = levenshtein(s1, s2)
        self.assertIsInstance(calculated_matches, list)
        prev_val0 = 0
        prev_val1 = 0
        for idx, val in enumerate(calculated_matches):
            self.assertIsInstance(val, tuple, msg=f'Match {idx}: {val}')
            self.assertEqual(len(val), 2, msg=f'Match {idx}: {val}')
            self.assertTrue((val[0] >= 0) and (val[0] < len(s1)), msg=f'Match {idx}: {val}')
            self.assertTrue((val[1] >= 0) and (val[1] < len(s2)), msg=f'Match {idx}: {val}')
            self.assertLessEqual(prev_val0, val[0], msg=f'Match {idx}: {val}')
            self.assertLessEqual(prev_val1, val[1], msg=f'Match {idx}: {val}')
            prev_val0 = val[0]
            prev_val1 = val[1]
        self.assertEqual(len(set([val[0] for val in calculated_matches])), len(s1))
        self.assertEqual(len(set([val[1] for val in calculated_matches])), len(s2))

    def test_levenshtein04(self):
        s2 = 'нейросетиэтохорошо'
        s1 = 'нейросетиэтооченьхорошо'
        calculated_matches = levenshtein(s1, s2)
        self.assertIsInstance(calculated_matches, list)
        prev_val0 = 0
        prev_val1 = 0
        for idx, val in enumerate(calculated_matches):
            self.assertIsInstance(val, tuple, msg=f'Match {idx}: {val}')
            self.assertEqual(len(val), 2, msg=f'Match {idx}: {val}')
            self.assertTrue((val[0] >= 0) and (val[0] < len(s1)), msg=f'Match {idx}: {val}')
            self.assertTrue((val[1] >= 0) and (val[1] < len(s2)), msg=f'Match {idx}: {val}')
            self.assertLessEqual(prev_val0, val[0], msg=f'Match {idx}: {val}')
            self.assertLessEqual(prev_val1, val[1], msg=f'Match {idx}: {val}')
            prev_val0 = val[0]
            prev_val1 = val[1]
        self.assertEqual(len(set([val[0] for val in calculated_matches])), len(s1))
        self.assertEqual(len(set([val[1] for val in calculated_matches])), len(s2))

    def test_align_01(self):
        src_words = [('нейросети', 0.0, 1.0), ('это', 1.2, 2.2), ('хорошо', 2.3, 3.5)]
        rescored_words = ['нейросети', 'это', 'хорошо']
        true_words = [('нейросети', 0.0, 1.0), ('это', 1.2, 2.2), ('хорошо', 2.3, 3.5)]
        self.assertEqual(align(src_words, rescored_words), true_words)

    def test_align_02(self):
        src_words = [('нейросети', 0.0, 1.0), ('это', 1.2, 2.2), ('хорошо', 2.3, 3.5)]
        rescored_words = ['нейросети', 'это', 'очень', 'хорошо']
        true_words = [('нейросети', 0.0, 1.0), ('это', 1.2, 2.2), ('очень', 1.2, 2.2), ('хорошо', 2.3, 3.5)]
        calculated_words = align(src_words, rescored_words)
        self.assertEqual(calculated_words, true_words)

    def test_align_03(self):
        src_words = [('нейросети', 0.0, 1.0), ('это', 1.2, 2.2), ('очень', 2.3, 3.5), ('хорошо', 3.55, 4.5)]
        rescored_words = ['нейросети', 'это', 'хорошо']
        true_words = [('нейросети', 0.0, 1.0), ('это', 1.2, 3.5), ('хорошо', 3.55, 4.5)]
        self.assertEqual(align(src_words, rescored_words), true_words)


if __name__ == '__main__':
    unittest.main(verbosity=2)
