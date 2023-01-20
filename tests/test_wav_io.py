import os
import re
import sys
import unittest
import tempfile

import numpy as np

try:
    from wav_io.wav_io import load_sound, transform_to_wavpcm
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from wav_io.wav_io import load_sound, transform_to_wavpcm


class TestWavIO(unittest.TestCase):
    def setUp(self):
        self.mono_fname = os.path.join(os.path.dirname(__file__), 'testdata', 'mono_sound.wav')
        self.stereo_fname = os.path.join(os.path.dirname(__file__), 'testdata', 'stereo_sound.wav')
        self.empty_fname = os.path.join(os.path.dirname(__file__), 'testdata', 'empty_mono.wav')
        self.incorrect_fname = os.path.join(os.path.dirname(__file__), 'testdata', 'incorrect_sampling_freq.wav')
        self.mpeg_fname = os.path.join(os.path.dirname(__file__), 'testdata', 'test_mpeg.m4a')
        self.wav_from_mpeg_fname = os.path.join(os.path.dirname(__file__), 'testdata', 'test_wav_from_mpeg.wav')
        self.unknown_sound = os.path.join(os.path.dirname(__file__), 'testdata', 'unknown_sound')
        self.not_sound = os.path.join(os.path.dirname(__file__), 'testdata', 'notsound.wav')
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as fp:
            self.tmp_fname = fp.name

    def tearDown(self) -> None:
        if hasattr(self, 'tmp_fname'):
            if os.path.isfile(self.tmp_fname):
                os.remove(self.tmp_fname)

    def test_load_mono(self):
        loaded = load_sound(self.mono_fname)
        self.assertIsInstance(loaded, np.ndarray)
        self.assertEqual(len(loaded.shape), 1)
        self.assertGreater(loaded.shape[0], 16_000)
        self.assertGreater(np.min(loaded), -1.0)
        self.assertLess(np.max(loaded), 1.0)
        self.assertGreater(np.max(loaded), 0.0)
        self.assertLess(np.min(loaded), 0.0)

    def test_load_stereo(self):
        loaded = load_sound(self.stereo_fname)
        self.assertIsInstance(loaded, tuple)
        self.assertEqual(len(loaded), 2)
        channel_idx = 0
        self.assertIsInstance(loaded[channel_idx], np.ndarray)
        self.assertEqual(len(loaded[channel_idx].shape), 1)
        self.assertGreater(loaded[channel_idx].shape[0], 16_000)
        self.assertGreater(np.min(loaded[channel_idx]), -1.0)
        self.assertLess(np.max(loaded[channel_idx]), 1.0)
        self.assertGreater(np.max(loaded[channel_idx]), 0.0)
        self.assertLess(np.min(loaded[channel_idx]), 0.0)
        channel_idx = 1
        self.assertIsInstance(loaded[channel_idx], np.ndarray)
        self.assertEqual(len(loaded[channel_idx].shape), 1)
        self.assertGreater(loaded[channel_idx].shape[0], 16_000)
        self.assertGreater(np.min(loaded[channel_idx]), -1.0)
        self.assertLess(np.max(loaded[channel_idx]), 1.0)
        self.assertGreater(np.max(loaded[channel_idx]), 0.0)
        self.assertLess(np.min(loaded[channel_idx]), 0.0)

    def test_load(self):
        mono = load_sound(self.mono_fname)
        stereo = load_sound(self.stereo_fname)
        eps = np.max(np.abs(mono)) / 500.0
        self.assertLess(np.max(np.abs(mono - (stereo[0] + stereo[1]) / 2.0)), eps)

    def test_load_empty(self):
        loaded = load_sound(self.empty_fname)
        self.assertIsNone(loaded)

    def test_load_incorrect(self):
        with self.assertRaises(ValueError):
            _ = load_sound(self.incorrect_fname)

    def test_transform_to_wavpcm_pos01(self):
        transform_to_wavpcm(self.mpeg_fname, self.tmp_fname)
        true_sound = load_sound(self.wav_from_mpeg_fname)
        converted_sound = load_sound(self.tmp_fname)
        self.assertIsInstance(converted_sound, np.ndarray)
        self.assertEqual(len(converted_sound.shape), 1)
        self.assertEqual(true_sound.shape, converted_sound.shape)
        max_val = float(np.max(np.abs(true_sound)))
        eps = max_val / 100.0
        diff = float(np.max(np.abs(converted_sound - true_sound)))
        self.assertLess(diff, eps)

    def test_transform_to_wavpcm_neg01(self):
        true_err_msg = f'The file "nonexisted.wav" does not exist!'
        with self.assertRaisesRegex(IOError, re.escape(true_err_msg)):
            transform_to_wavpcm('nonexisted.wav', self.tmp_fname)

    def test_transform_to_wavpcm_neg02(self):
        true_err_msg = f'The extension of the file "{self.unknown_sound}" is unknown. ' \
                       f'So, I cannot determine a format of this sound file.'
        with self.assertRaisesRegex(ValueError, re.escape(true_err_msg)):
            transform_to_wavpcm(self.unknown_sound, self.tmp_fname)

    def test_transform_to_wavpcm_neg03(self):
        true_err_msg = f'The file "{self.not_sound}" cannot be opened. '
        with self.assertRaisesRegex(IOError, re.escape(true_err_msg) + r'.+'):
            transform_to_wavpcm(self.not_sound, self.tmp_fname)


if __name__ == '__main__':
    unittest.main(verbosity=2)
