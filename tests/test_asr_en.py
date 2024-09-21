import os
import sys
import unittest
import warnings

from nltk import wordpunct_tokenize
import numpy as np
import torch

try:
    from asr.asr import initialize_model_for_speech_recognition
    from asr.asr import initialize_model_for_speech_classification
    from asr.asr import initialize_model_for_speech_segmentation
    from asr.asr import transcribe
    from asr.asr import TARGET_SAMPLING_FREQUENCY
    from wav_io.wav_io import load_sound
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from asr.asr import initialize_model_for_speech_recognition
    from asr.asr import initialize_model_for_speech_classification
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
        cls.sound_without_speech = load_sound(os.path.join(os.path.dirname(__file__), 'testdata', 'test_silence.wav'))
        segmenter_name = os.path.join(os.path.dirname(__file__), 'testdata', 'model_en', 'wav2vec2')
        try:
            cls.segmenter = initialize_model_for_speech_segmentation(
                'en',
                segmenter_name
            )
        except Exception as err:
            warnings.warn(f'The segmenter is not loaded from the "{segmenter_name}": {str(err)}')
            cls.segmenter = initialize_model_for_speech_segmentation(
                'en',
                'facebook/wav2vec2-base-960h'
            )
        vad_name = os.path.join(os.path.dirname(__file__), 'testdata', 'model_en', 'ast')
        try:
            cls.vad = initialize_model_for_speech_classification(
                vad_name
            )
        except Exception as err:
            warnings.warn(f'The voice activity detector is not loaded from the "{vad_name}": {str(err)}')
            cls.vad = initialize_model_for_speech_classification()
        recognizer_name = os.path.join(os.path.dirname(__file__), 'testdata', 'model_en', 'whisper')
        try:
            cls.recognizer = initialize_model_for_speech_recognition(
                'en',
                recognizer_name
            )
        except Exception as err:
            warnings.warn(f'The recognizer is not loaded from the "{recognizer_name}": {str(err)}')
            cls.recognizer = initialize_model_for_speech_recognition(
                'en',
                'openai/whisper-medium'
            )

    def test_recognize_pos01(self):
        res = transcribe(
            mono_sound=self.sound,
            segmenter=self.segmenter,
            voice_activity_detector=self.vad,
            asr=self.recognizer,
            min_segment_size=1,
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

    def test_recognize_pos02(self):
        res = transcribe(
            mono_sound=self.sound_without_speech,
            segmenter=self.segmenter,
            voice_activity_detector=self.vad,
            asr=self.recognizer,
            min_segment_size=1,
            max_segment_size=5
        )
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 0)

    def test_recognize_pos03(self):
        res = transcribe(
            mono_sound=np.zeros((160_000,), dtype=np.float32),
            segmenter=self.segmenter,
            voice_activity_detector=self.vad,
            asr=self.recognizer,
            min_segment_size=1,
            max_segment_size=5
        )
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
