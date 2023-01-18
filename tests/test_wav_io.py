import os
import sys
import unittest

import numpy as np

try:
    from wav_io.wav_io import load_sound
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from wav_io.wav_io import load_sound


class TestWavIO(unittest.TestCase):
    def setUp(self):
        self.mono_fname = os.path.join(os.path.dirname(__file__), 'testdata', 'mono_sound.wav')
        self.stereo_fname = os.path.join(os.path.dirname(__file__), 'testdata', 'stereo_sound.wav')
        self.empty_fname = os.path.join(os.path.dirname(__file__), 'testdata', 'empty_mono.wav')
        self.incorrect_fname = os.path.join(os.path.dirname(__file__), 'testdata',
                                            'incorrect_sampling_freq.wav')

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
