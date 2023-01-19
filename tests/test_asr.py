import os
import sys
import unittest

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
        cls.sound = load_sound(os.path.join(os.path.dirname(__file__), 'testdata', 'test_sound_ru.wav'))
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
