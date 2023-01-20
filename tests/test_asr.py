import os
import re
import sys
import unittest

import torch

try:
    from asr.asr import check_annotations_and_hotwords
    from asr.asr import decode_for_evaluation
    from asr.asr import recognize, initialize_model
    from wav_io.wav_io import load_sound
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from asr.asr import check_annotations_and_hotwords
    from asr.asr import decode_for_evaluation
    from asr.asr import recognize, initialize_model
    from wav_io.wav_io import load_sound


class TestASR(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if torch.cuda.is_available():
            cls.cuda_is_used = True
            torch.cuda.empty_cache()
        else:
            cls.cuda_is_used = False
        cls.sound = load_sound(os.path.join(os.path.dirname(__file__), 'testdata', 'test_sound_ru.wav'))
        cls.yet_another_sound = load_sound(os.path.join(os.path.dirname(__file__), 'testdata', 'mono_sound.wav'))
        cls.processor, cls.model = initialize_model()

    def test_recognize_pos01(self):
        res = recognize(
            mono_sound=self.sound,
            processor=self.processor,
            model=self.model
        )
        true_words = ['нейронные', 'сети', 'это', 'хорошо']
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), len(true_words))
        prev_pos = 0.0
        for idx in range(len(true_words)):
            self.assertIsInstance(res[idx], tuple)
            self.assertEqual(len(res[idx]), 3)
            self.assertIsInstance(res[idx][0], str)
            self.assertIsInstance(res[idx][1], float)
            self.assertIsInstance(res[idx][2], float)
            self.assertEqual(res[idx][0], true_words[idx])
            self.assertGreaterEqual(res[idx][1], prev_pos)
            self.assertLess(res[idx][1], res[idx][2])
            prev_pos = res[idx][2]

    def test_recognize_pos02(self):
        res = recognize(
            mono_sound=self.sound,
            processor=self.processor,
            model=self.model,
            alpha=0.565,
            beta=0.148,
            hotword_weight=2.0,
            hotwords=['нейронные', 'сети']
        )
        true_words = ['нейронные', 'сети', 'это', 'хорошо']
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), len(true_words))
        prev_pos = 0.0
        for idx in range(len(true_words)):
            self.assertIsInstance(res[idx], tuple)
            self.assertEqual(len(res[idx]), 3)
            self.assertIsInstance(res[idx][0], str)
            self.assertIsInstance(res[idx][1], float)
            self.assertIsInstance(res[idx][2], float)
            self.assertEqual(res[idx][0], true_words[idx])
            self.assertGreaterEqual(res[idx][1], prev_pos)
            self.assertLess(res[idx][1], res[idx][2])
            prev_pos = res[idx][2]

    def test_check_annotations_and_hotwords_pos01(self):
        annotations = [
            'Нейронные сети — это хорошо!',
            'Нейросеть является эффективным способом построения иерархии представлений.',
            'Первые попытки формализации искусственного нейрона были предприняты в сороковых годах двадцатого века.'
        ]
        check_annotations_and_hotwords(list_of_texts=annotations, additional='annotation')
        self.assertTrue(True)

    def test_check_annotations_and_hotwords_neg01(self):
        annotations = [
            'Нейронные сети - это хорошо!',
            'Нейросеть является эффективным способом построения иерархии представлений.',
            'Первые попытки формализации искусственного нейрона были предприняты в 40-х годах двадцатого века.'
        ]
        true_err_msg = 'The 3rd annotation "Первые попытки формализации искусственного нейрона были предприняты ' \
                       'в 40-х годах двадцатого века." is wrong! The text should not include any digit! ' \
                       'All numerals have to be written in letters: for example, "42" should look like ' \
                       '"forty two" in English or "сорок два" in Russian.'
        with self.assertRaisesRegex(ValueError, re.escape(true_err_msg)):
            check_annotations_and_hotwords(list_of_texts=annotations, additional='annotation')

    def test_check_annotations_and_hotwords_neg02(self):
        annotations = [
            'Нейронные сети - это хорошо!',
            'Первые попытки формализации искусственного нейрона были предприняты в сороковых годах 20 века.',
            'Нейросеть является эффективным способом построения иерархии представлений.',
        ]
        true_err_msg = 'The 2nd annotation "Первые попытки формализации искусственного нейрона были предприняты ' \
                       'в сороковых годах 20 века." is wrong! The text should not include any digit! ' \
                       'All numerals have to be written in letters: for example, "42" should look like ' \
                       '"forty two" in English or "сорок два" in Russian.'
        with self.assertRaisesRegex(ValueError, re.escape(true_err_msg)):
            check_annotations_and_hotwords(list_of_texts=annotations, additional='annotation')

    def test_check_annotations_and_hotwords_neg03(self):
        annotations = [
            'Нейронные сети - это хорошо!',
            'Нейросети — это хорошо, поскольку они являются эффективным инструментом распознавания речи.',
            'Нейросеть является эффективным способом построения иерархии представлений.',
            'Первые попытки формализации искусственного нейрона были предприняты в 40-х годах двадцатого века.'
        ]
        true_err_msg = 'The 4th annotation "Первые попытки формализации искусственного нейрона были предприняты ' \
                       'в 40-х годах двадцатого века." is wrong! The text should not include any digit! ' \
                       'All numerals have to be written in letters: for example, "42" should look like ' \
                       '"forty two" in English or "сорок два" in Russian.'
        with self.assertRaisesRegex(ValueError, re.escape(true_err_msg)):
            check_annotations_and_hotwords(list_of_texts=annotations, additional='annotation')

    def test_decode_for_evaluation(self):
        re_for_digits = re.compile(r'\d+')
        re_for_russian_letters = re.compile(r'^[абвгдежзийклмнопрстуфхцчшщъыьэюя]+[ абвгдежзийклмнопрстуфхцчшщъыьэюя]*'
                                            r'[абвгдежзийклмнопрстуфхцчшщъыьэюя]+$')
        inputs = self.processor(self.sound, sampling_rate=16_000,
                                return_tensors="pt", padding=True)
        if self.cuda_is_used:
            inputs = inputs.to('cuda')
        with torch.no_grad():
            predicted = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        del inputs
        if self.cuda_is_used:
            logits1 = predicted.to('cpu').numpy()[0]
        else:
            logits1 = predicted.numpy()[0]
        del predicted
        inputs = self.processor(self.yet_another_sound, sampling_rate=16_000,
                                return_tensors="pt", padding=True)
        if self.cuda_is_used:
            inputs = inputs.to('cuda')
        with torch.no_grad():
            predicted = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        del inputs
        if self.cuda_is_used:
            logits2 = predicted.to('cpu').numpy()[0]
        else:
            logits2 = predicted.numpy()[0]
        del predicted
        logits = [logits1, logits2]
        recognized_texts = decode_for_evaluation(
            processor=self.processor,
            evaluation_logits=logits,
            alpha=0.565,
            beta=0.148,
            hotword_weight=2.0,
            hotwords=['нейронные', 'сети']
        )
        self.assertIsInstance(recognized_texts, list)
        self.assertEqual(len(recognized_texts), 2)
        self.assertIsInstance(recognized_texts[0], str)
        self.assertIsInstance(recognized_texts[1], str)
        self.assertEqual(recognized_texts[0], ' '.join(recognized_texts[0].strip().lower().split()))
        self.assertEqual(recognized_texts[1], ' '.join(recognized_texts[1].strip().lower().split()))
        self.assertGreater(len(recognized_texts[0]), 0)
        self.assertGreater(len(recognized_texts[1]), 0)
        self.assertNotEqual(recognized_texts[0], recognized_texts[1])
        self.assertIsNone(re_for_digits.search(recognized_texts[0]))
        self.assertIsNone(re_for_digits.search(recognized_texts[1]))
        self.assertIsNotNone(re_for_russian_letters.search(recognized_texts[0]), msg=recognized_texts[0])
        self.assertIsNotNone(re_for_russian_letters.search(recognized_texts[1]), msg=recognized_texts[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
