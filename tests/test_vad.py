import os
import sys
import unittest

import numpy as np
from webrtcvad import Vad

try:
    from vad.vad import sound_to_bytes, calculate_voice_probabilities, split_long_sound
    from vad.vad import stick_subsounds, stick_short_subsounds_to_longer_neighbours
    from wav_io.wav_io import load_sound
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from vad.vad import sound_to_bytes, calculate_voice_probabilities, split_long_sound
    from vad.vad import stick_subsounds, stick_short_subsounds_to_longer_neighbours
    from wav_io.wav_io import load_sound


def find_subarray(full_array: np.ndarray, sub_array: np.ndarray, start_pos: int) -> int:
    found_idx = -1
    if full_array.shape[0] == 0:
        return found_idx
    if sub_array.shape[0] == 0:
        return found_idx
    if sub_array.shape[0] > (full_array.shape[0] - start_pos):
        return found_idx
    for idx in range(start_pos, full_array.shape[0] - sub_array.shape[0] + 1):
        cur_subarray = full_array[idx:(idx + sub_array.shape[0])]
        if cur_subarray.shape != sub_array.shape:
            break
        if np.max(np.abs(cur_subarray - sub_array)) < 1e-6:
            found_idx = idx
            break
    return found_idx


class TestVAD(unittest.TestCase):
    def setUp(self):
        self.sound = load_sound(os.path.join(os.path.dirname(__file__), 'testdata', 'mono_sound.wav'))
        self.silence = load_sound(os.path.join(os.path.dirname(__file__), 'testdata', 'silence.wav'))
        self.vad_ensemble = [Vad(0), Vad(1), Vad(2), Vad(3)]

    def tearDown(self) -> None:
        del self.sound
        del self.silence
        del self.vad_ensemble

    def test_silence(self):
        probabilities = calculate_voice_probabilities(self.silence, self.vad_ensemble)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(len(probabilities.shape), 1)
        self.assertEqual(probabilities.shape[0], 98)
        self.assertAlmostEqual(np.min(probabilities), 0.0)
        self.assertAlmostEqual(np.max(probabilities), 0.0)

    def test_sound(self):
        probabilities = calculate_voice_probabilities(self.sound, self.vad_ensemble)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(len(probabilities.shape), 1)
        self.assertGreater(probabilities.shape[0], 98)
        self.assertLessEqual(np.max(probabilities), 1.0)
        self.assertGreaterEqual(np.min(probabilities), 0.0)
        self.assertGreater(np.max(probabilities), np.min(probabilities))

    def test_sound_to_bytes_pos01(self):
        input_sound = np.zeros((480,), dtype=np.float32)
        target_bytes = b'\x00\x00' * 480
        calculated_bytes = sound_to_bytes(input_sound)
        self.assertIsInstance(calculated_bytes, bytes)
        self.assertEqual(calculated_bytes, target_bytes)

    def test_sound_to_bytes_neg01(self):
        input_sound = np.zeros((400,), dtype=np.float32)
        with self.assertRaises(ValueError):
            _ = sound_to_bytes(input_sound)

    def test_split_long_sound(self):
        res = split_long_sound(self.sound, self.vad_ensemble, 3 * 16_000)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        subsounds, bounds_of_subsounds = res
        self.assertIsInstance(subsounds, list)
        self.assertEqual(len(subsounds), 5)
        self.assertIsInstance(bounds_of_subsounds, list)
        self.assertEqual(len(subsounds), len(bounds_of_subsounds))
        sumlen = 0
        for idx, cur in enumerate(subsounds):
            self.assertIsInstance(cur, np.ndarray, msg=f'Subsound {idx} is wrong!')
            self.assertEqual(len(cur.shape), 1, msg=f'Subsound {idx} is wrong!')
            self.assertGreater(cur.shape[0], 0, msg=f'Subsound {idx} is wrong!')
            self.assertLess(cur.shape[0], 4 * 16_000, msg=f'Subsound {idx} is wrong!')
            self.assertIsInstance(bounds_of_subsounds[idx], tuple,
                                  msg=f'Bounds of the subsound {idx} are wrong!')
            self.assertEqual(len(bounds_of_subsounds[idx]), 2,
                                 msg=f'Bounds of the subsound {idx} are wrong!')
            self.assertGreaterEqual(bounds_of_subsounds[idx][0], 0,
                                    msg=f'Bounds of the subsound {idx} are wrong!')
            self.assertLessEqual(bounds_of_subsounds[idx][1], self.sound.shape[0],
                                 msg=f'Bounds of the subsound {idx} are wrong!')
            self.assertLess(bounds_of_subsounds[idx][0], bounds_of_subsounds[idx][1],
                            msg=f'Bounds of the subsound {idx} are wrong!')
            sumlen += cur.shape[0]
        self.assertGreaterEqual(sumlen, self.sound.shape[0])
        self.assertLess(sumlen, round(1.1 * self.sound.shape[0]))
        start_pos = 0
        for idx, cur in enumerate(subsounds):
            found_idx = find_subarray(full_array=self.sound, sub_array=cur, start_pos=start_pos)
            self.assertGreater(found_idx, -1, msg=f'Subsound {idx} is wrong! (start_pos = {start_pos})')
            self.assertEqual((found_idx, found_idx + cur.shape[0]), bounds_of_subsounds[idx],
                             msg=f'Subsound {idx} does not correspond to its bounds!')
            start_pos = found_idx + cur.shape[0] - 1000

    def test_stick_subsounds_pos01(self):
        lengths = [3200, 800, 2200, 400, 650, 1800, 16000]
        bounds_of_subsounds = [
            (0, 3200),
            (3000, 3800),
            (3750, 5950),
            (5900, 6300),
            (6290, 6940),
            (6800, 8600),
            (8500, 24500)
        ]
        source_sound = np.random.normal(loc=0.0, scale=1.0,
                                        size=bounds_of_subsounds[-1][1] - bounds_of_subsounds[0][0])
        subsounds = [source_sound[bounds_of_subsounds[idx][0]:(bounds_of_subsounds[idx][0] + lengths[idx])]
                     for idx in range(len(lengths))]
        true_new_subsound_bounds = (0, 3200)
        true_new_subsound = source_sound[0:3200]
        res = stick_subsounds(subsounds=subsounds, bounds_of_subsounds=bounds_of_subsounds, indices=(0, 1))
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], np.ndarray)
        self.assertIsInstance(res[1], tuple)
        self.assertEqual(res[1], true_new_subsound_bounds)
        self.assertEqual(res[0].shape, true_new_subsound.shape)
        self.assertLess(np.max(np.abs(res[0] - true_new_subsound)), 1e-5)

    def test_stick_subsounds_pos02(self):
        lengths = [3200, 800, 2200, 400, 650, 1800, 16000]
        bounds_of_subsounds = [
            (0, 3200),
            (3000, 3800),
            (3750, 5950),
            (5900, 6300),
            (6290, 6940),
            (6800, 8600),
            (8500, 24500)
        ]
        source_sound = np.random.normal(loc=0.0, scale=1.0,
                                        size=bounds_of_subsounds[-1][1] - bounds_of_subsounds[0][0])
        subsounds = [source_sound[bounds_of_subsounds[idx][0]:(bounds_of_subsounds[idx][0] + lengths[idx])]
                     for idx in range(len(lengths))]
        true_new_subsound_bounds = (5900, 8600)
        true_new_subsound = source_sound[5900:8600]
        res = stick_subsounds(subsounds=subsounds, bounds_of_subsounds=bounds_of_subsounds, indices=(3, 6))
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], np.ndarray)
        self.assertIsInstance(res[1], tuple)
        self.assertEqual(res[1], true_new_subsound_bounds)
        self.assertEqual(res[0].shape, true_new_subsound.shape)
        self.assertLess(np.max(np.abs(res[0] - true_new_subsound)), 1e-5)

    def test_stick_short_subsounds_to_longer_neighbours_pos01(self):
        lengths = [3200, 800, 2200, 400, 650, 1800, 16000]
        bounds_of_subsounds = [
            (0, 3200),
            (3000, 3800),
            (3750, 5950),
            (5900, 6300),
            (6290, 6940),
            (6800, 8600),
            (8500, 24500)
        ]
        source_sound = np.random.normal(loc=0.0, scale=1.0,
                                        size=bounds_of_subsounds[-1][1] - bounds_of_subsounds[0][0])
        subsounds = [source_sound[bounds_of_subsounds[idx][0]:(bounds_of_subsounds[idx][0] + lengths[idx])]
                     for idx in range(len(lengths))]
        true_bounds_of_subsounds = [
            (0, 3200),
            (3000, 5950),
            (5900, 8600),
            (8500, 24500)
        ]
        true_subsounds = [source_sound[cur[0]:cur[1]] for cur in true_bounds_of_subsounds]
        res = stick_short_subsounds_to_longer_neighbours(subsounds, bounds_of_subsounds)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], list)
        self.assertIsInstance(res[1], list)
        self.assertEqual(len(res[0]), len(res[1]))
        self.assertEqual(len(res[1]), len(true_bounds_of_subsounds))
        self.assertEqual(res[1], true_bounds_of_subsounds)
        for idx in range(len(true_bounds_of_subsounds)):
            self.assertIsInstance(res[0][idx], np.ndarray)
            self.assertEqual(res[0][idx].shape, true_subsounds[idx].shape)
            self.assertLess(np.max(np.abs(res[0][idx] - true_subsounds[idx])), 1e-5)

    def test_stick_short_subsounds_to_longer_neighbours_pos02(self):
        lengths = [3200, 2950, 2700, 16000]
        bounds_of_subsounds = [
            (0, 3200),
            (3000, 5950),
            (5900, 8600),
            (8500, 24500)
        ]
        source_sound = np.random.normal(loc=0.0, scale=1.0,
                                        size=bounds_of_subsounds[-1][1] - bounds_of_subsounds[0][0])
        subsounds = [source_sound[bounds_of_subsounds[idx][0]:(bounds_of_subsounds[idx][0] + lengths[idx])]
                     for idx in range(len(lengths))]
        true_bounds_of_subsounds = [
            (0, 3200),
            (3000, 5950),
            (5900, 8600),
            (8500, 24500)
        ]
        true_subsounds = [source_sound[cur[0]:cur[1]] for cur in true_bounds_of_subsounds]
        res = stick_short_subsounds_to_longer_neighbours(subsounds, bounds_of_subsounds)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], list)
        self.assertIsInstance(res[1], list)
        self.assertEqual(len(res[0]), len(res[1]))
        self.assertEqual(len(res[1]), len(true_bounds_of_subsounds))
        self.assertEqual(res[1], true_bounds_of_subsounds)
        for idx in range(len(true_bounds_of_subsounds)):
            self.assertIsInstance(res[0][idx], np.ndarray)
            self.assertEqual(res[0][idx].shape, true_subsounds[idx].shape)
            self.assertLess(np.max(np.abs(res[0][idx] - true_subsounds[idx])), 1e-5)

    def test_stick_short_subsounds_to_longer_neighbours_pos03(self):
        lengths = [1200]
        bounds_of_subsounds = [
            (0, 1200),
        ]
        source_sound = np.random.normal(loc=0.0, scale=1.0,
                                        size=bounds_of_subsounds[-1][1] - bounds_of_subsounds[0][0])
        subsounds = [source_sound[bounds_of_subsounds[idx][0]:(bounds_of_subsounds[idx][0] + lengths[idx])]
                     for idx in range(len(lengths))]
        true_bounds_of_subsounds = [
            (0, 1200),
        ]
        true_subsounds = [source_sound[cur[0]:cur[1]] for cur in true_bounds_of_subsounds]
        res = stick_short_subsounds_to_longer_neighbours(subsounds, bounds_of_subsounds)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], list)
        self.assertIsInstance(res[1], list)
        self.assertEqual(len(res[0]), len(res[1]))
        self.assertEqual(len(res[1]), len(true_bounds_of_subsounds))
        self.assertEqual(res[1], true_bounds_of_subsounds)
        for idx in range(len(true_bounds_of_subsounds)):
            self.assertIsInstance(res[0][idx], np.ndarray)
            self.assertEqual(res[0][idx].shape, true_subsounds[idx].shape)
            self.assertLess(np.max(np.abs(res[0][idx] - true_subsounds[idx])), 1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
