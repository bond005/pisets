import os
import re
import sys
import unittest

import nltk

try:
    from normalization.normalization import calculate_sentence_bounds, tokenize_text, check_language, dtw
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from normalization.normalization import calculate_sentence_bounds, tokenize_text, check_language, dtw


class TestNormalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        nltk.download('punkt')

    def test_normalization_01(self):
        asr_res = [
            ('мама', 0.0, 0.5),
            ('мыла', 0.53, 0.92),
            ('раму', 1.0, 1.35),
            ('папа', 1.35, 1.57),
            ('пил', 1.575, 1.9),
            ('пиво', 1.9, 2.09),
            ('елочка', 2.1, 2.4),
            ('гори', 2.45, 2.65)
        ]
        source_sentences = ['Мама мыла раму.', 'Папа пил пиво?', 'Ёлочка, гори!']
        true_sentences = [
            ('Мама мыла раму.', 0.0, 1.35),
            ('Папа пил пиво?', 1.35, 2.09),
            ('Ёлочка, гори!', 2.1, 2.65)
        ]
        calculated_sentences = calculate_sentence_bounds(asr_res, source_sentences)
        self.assertIsInstance(calculated_sentences, list)
        self.assertEqual(len(calculated_sentences), len(true_sentences))
        for sent_idx in range(len(true_sentences)):
            self.assertIsInstance(calculated_sentences[sent_idx], tuple)
            self.assertEqual(len(calculated_sentences[sent_idx]), 3)
            self.assertIsInstance(calculated_sentences[sent_idx][0], str)
            self.assertIsInstance(calculated_sentences[sent_idx][1], float)
            self.assertIsInstance(calculated_sentences[sent_idx][2], float)
            self.assertEqual(calculated_sentences[sent_idx][0], true_sentences[sent_idx][0])
            self.assertAlmostEqual(calculated_sentences[sent_idx][1], true_sentences[sent_idx][1])
            self.assertAlmostEqual(calculated_sentences[sent_idx][2], true_sentences[sent_idx][2])

    def test_normalization_02(self):
        asr_res = [
            ('мамы', 0.0, 0.5),
            ('мыла', 0.53, 0.92),
            ('раму', 1.0, 1.35),
            ('папа', 1.35, 1.57),
            ('пил', 1.575, 1.9),
            ('пива', 1.9, 2.09),
            ('елочка', 2.1, 2.4),
            ('гори', 2.45, 2.65)
        ]
        source_sentences = ['Мама мыла раму.', 'Папа пил пиво?', 'Ёлочка, гори!']
        true_sentences = [
            ('Мама мыла раму.', 0.0, 1.35),
            ('Папа пил пиво?', 1.35, 2.09),
            ('Ёлочка, гори!', 2.1, 2.65)
        ]
        calculated_sentences = calculate_sentence_bounds(asr_res, source_sentences)
        self.assertIsInstance(calculated_sentences, list)
        self.assertEqual(len(calculated_sentences), len(true_sentences))
        for sent_idx in range(len(true_sentences)):
            self.assertIsInstance(calculated_sentences[sent_idx], tuple)
            self.assertEqual(len(calculated_sentences[sent_idx]), 3)
            self.assertIsInstance(calculated_sentences[sent_idx][0], str)
            self.assertIsInstance(calculated_sentences[sent_idx][1], float)
            self.assertIsInstance(calculated_sentences[sent_idx][2], float)
            self.assertEqual(calculated_sentences[sent_idx][0], true_sentences[sent_idx][0])
            self.assertAlmostEqual(calculated_sentences[sent_idx][1], true_sentences[sent_idx][1])
            self.assertAlmostEqual(calculated_sentences[sent_idx][2], true_sentences[sent_idx][2])

    def test_tokenize_text_ru(self):
        text = 'Мама мыла раму. Папа пил пиво? Ёлочка, гори!'
        true_sentences = ['Мама мыла раму.', 'Папа пил пиво?', 'Ёлочка, гори!']
        calculated_sentences = tokenize_text(text, lang='ru')
        self.assertIsInstance(calculated_sentences, list)
        self.assertEqual(len(true_sentences), len(calculated_sentences))
        self.assertEqual(true_sentences, calculated_sentences)

    def test_tokenize_text_en(self):
        text = 'This is a sentence. This is another sentence, isn\'t it? Yes, of course!'
        true_sentences = ['This is a sentence.', 'This is another sentence, isn\'t it?', 'Yes, of course!']
        calculated_sentences = tokenize_text(text, lang='en')
        self.assertIsInstance(calculated_sentences, list)
        self.assertEqual(len(true_sentences), len(calculated_sentences))
        self.assertEqual(true_sentences, calculated_sentences)

    def test_check_language_ru1(self):
        lang = check_language('Russian')
        true_lang = 'ru'
        self.assertEqual(true_lang, lang)

    def test_check_language_ru2(self):
        lang = check_language(' rus')
        true_lang = 'ru'
        self.assertEqual(true_lang, lang)

    def test_check_language_ru3(self):
        lang = check_language('ru')
        true_lang = 'ru'
        self.assertEqual(true_lang, lang)

    def test_check_language_en1(self):
        lang = check_language('English\t')
        true_lang = 'en'
        self.assertEqual(true_lang, lang)

    def test_check_language_en2(self):
        lang = check_language('ENG')
        true_lang = 'en'
        self.assertEqual(true_lang, lang)

    def test_check_language_en3(self):
        lang = check_language('en')
        true_lang = 'en'
        self.assertEqual(true_lang, lang)

    def test_dtw_pos01(self):
        s1 = 'abcbdd'
        s2 = 'aabbbddd'
        true_rule_1 = [0, 0, 1, 2, 3, 4, 5, 5]
        true_rule_2 = [0, 1, 2, 3, 4, 5, 6, 7]
        rules = dtw(s1, s2)
        self.assertIsInstance(rules, tuple)
        self.assertEqual(len(rules), 2)
        self.assertEqual(true_rule_1, rules[0])
        self.assertEqual(true_rule_2, rules[1])

    def test_dtw_pos02(self):
        s2 = 'abcbdd'
        s1 = 'aabbbddd'
        true_rule_2 = [0, 0, 1, 2, 3, 4, 5, 5]
        true_rule_1 = [0, 1, 2, 3, 4, 5, 6, 7]
        rules = dtw(s1, s2)
        self.assertIsInstance(rules, tuple)
        self.assertEqual(len(rules), 2)
        self.assertEqual(true_rule_1, rules[0])
        self.assertEqual(true_rule_2, rules[1])

    def test_dtw_pos03(self):
        s1 = 'a'
        s2 = 'aabbbddd'
        true_rule_1 = [0, 0, 0, 0, 0, 0, 0, 0]
        true_rule_2 = [0, 1, 2, 3, 4, 5, 6, 7]
        rules = dtw(s1, s2)
        self.assertIsInstance(rules, tuple)
        self.assertEqual(len(rules), 2)
        self.assertEqual(true_rule_1, rules[0])
        self.assertEqual(true_rule_2, rules[1])

    def test_dtw_neg01(self):
        with self.assertRaisesRegex(ValueError, re.escape(r'Both sequences are empty!')):
            _ = dtw('', '')

    def test_dtw_neg02(self):
        with self.assertRaisesRegex(ValueError, re.escape(r'Left sequence is empty!')):
            _ = dtw('', 'a')

    def test_dtw_neg03(self):
        with self.assertRaisesRegex(ValueError, re.escape(r'Right sequence is empty!')):
            _ = dtw('a', '')


if __name__ == '__main__':
    unittest.main(verbosity=2)
