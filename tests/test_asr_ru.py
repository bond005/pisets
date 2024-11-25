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


class TestRussianASR(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if torch.cuda.is_available():
            cls.cuda_is_used = True
            torch.cuda.empty_cache()
        else:
            cls.cuda_is_used = False
        cls.sound = load_sound(os.path.join(os.path.dirname(__file__), 'testdata', 'test_sound_ru.wav'))
        cls.sound_without_speech = load_sound(os.path.join(os.path.dirname(__file__), 'testdata', 'test_silence.wav'))
        segmenter_name = os.path.join(os.path.dirname(__file__), 'testdata', 'model_ru', 'wav2vec2')
        try:
            cls.segmenter = initialize_model_for_speech_segmentation(
                'ru',
                segmenter_name
            )
        except Exception as err:
            warnings.warn(f'The segmenter is not loaded from the "{segmenter_name}": {str(err)}')
            cls.segmenter = initialize_model_for_speech_segmentation(
                'ru',
                'bond005/wav2vec2-base-ru-birm'
            )
        vad_name = os.path.join(os.path.dirname(__file__), 'testdata', 'model_en', 'ast')
        try:
            cls.vad = initialize_model_for_speech_classification(
                vad_name
            )
        except Exception as err:
            warnings.warn(f'The voice activity detector is not loaded from the "{vad_name}": {str(err)}')
            cls.vad = initialize_model_for_speech_classification()
        recognizer_name = os.path.join(os.path.dirname(__file__), 'testdata', 'model_ru', 'whisper')
        try:
            cls.recognizer = initialize_model_for_speech_recognition(
                'ru',
                recognizer_name
            )
        except Exception as err:
            warnings.warn(f'The recognizer is not loaded from the "{recognizer_name}": {str(err)}')
            cls.recognizer = initialize_model_for_speech_recognition(
                'ru',
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
        true_words = ['нейронные', 'сети', 'это', 'хорошо']
        predicted_text = ' '.join([r.transcription for r in res])
        self.assertEqual(len(res), 1)
        self.assertLessEqual(0.0, res[0].start)
        self.assertLess(res[0].start, res[0].end)
        self.assertLessEqual(res[0].end, self.sound.shape[0] / TARGET_SAMPLING_FREQUENCY)
        predicted_words = list(filter(lambda it: it.isalnum(), wordpunct_tokenize(predicted_text)))
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
