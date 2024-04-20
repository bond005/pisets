import os
import sys
import unittest

from nltk import wordpunct_tokenize
import torch

try:
    from asr.asr import initialize_model_for_speech_recognition
    from asr.asr import initialize_model_for_speech_segmentation
    from asr.asr import transcribe
    from asr.asr import TARGET_SAMPLING_FREQUENCY
    from wav_io.wav_io import load_sound
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from asr.asr import initialize_model_for_speech_recognition
    from asr.asr import initialize_model_for_speech_segmentation
    from asr.asr import transcribe
    from asr.asr import TARGET_SAMPLING_FREQUENCY
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
        cls.segmenter = initialize_model_for_speech_segmentation(
            'en',
            'facebook/wav2vec2-base-960h'
        )
        cls.recognizer = initialize_model_for_speech_recognition(
            'en',
            'openai/whisper-small'
        )

    def test_recognize_pos01(self):
        res = transcribe(
            mono_sound=self.sound,
            segmenter=self.segmenter,
            asr=self.recognizer,
            max_segment_size=5
        )
        true_words = ['neural', 'networks', 'are', 'good']
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 1)
        self.assertIsInstance(res[0], tuple)
        self.assertEqual(len(res[0]), 3)
        self.assertIsInstance(res[0][0], float)
        self.assertIsInstance(res[0][1], float)
        self.assertIsInstance(res[0][2], str)
        self.assertLessEqual(0.0, res[0][0])
        self.assertLess(res[0][0], res[0][1])
        self.assertLessEqual(res[0][1], self.sound.shape[0] / TARGET_SAMPLING_FREQUENCY)
        predicted_words = list(filter(lambda it: it.isalnum(), wordpunct_tokenize(res[0][2].lower())))
        self.assertEqual(predicted_words, true_words)


if __name__ == '__main__':
    unittest.main(verbosity=2)
