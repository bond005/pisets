import os
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


class TestEnglishASR(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if torch.cuda.is_available():
            cls.cuda_is_used = True
            torch.cuda.empty_cache()
        else:
            cls.cuda_is_used = False
        cls.sound = load_sound(os.path.join(os.path.dirname(__file__), 'testdata', 'test_sound_en.wav'))
        cls.processor_en, cls.model_en = initialize_model('en')

    def test_recognize_pos01(self):
        res = recognize(
            mono_sound=self.sound,
            processor=self.processor_en,
            model=self.model_en
        )
        true_words = ['neural', 'networks', 'are', 'good']
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
        self.assertLessEqual(prev_pos, self.sound.shape[0] / 16000.0)
        self.assertGreater(prev_pos, 0.5 * (self.sound.shape[0] / 16000.0))

    def test_recognize_pos02(self):
        res = recognize(
            mono_sound=self.sound,
            processor=self.processor_en,
            model=self.model_en,
            alpha=0.565,
            beta=0.148,
            hotword_weight=2.0,
            hotwords=['neural', 'networks']
        )
        true_words = ['neural', 'networks', 'are', 'good']
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
        self.assertLessEqual(prev_pos, self.sound.shape[0] / 16000.0)
        self.assertGreater(prev_pos, 0.5 * (self.sound.shape[0] / 16000.0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
