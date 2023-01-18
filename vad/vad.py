from typing import List, Tuple

import numpy as np
from webrtcvad import Vad


MIN_SOUND_LENGTH = 1600  # 100 milliseconds with sampling frequency = 16000 Hz


def sound_to_bytes(sound: np.ndarray) -> bytes:
    possible_lengths = {160, 320, 480}
    if len(sound.shape) != 1:
        err_msg = f'The sound channel number is wrong! Expected 1, got {len(sound.shape)}.'
        raise ValueError(err_msg)
    if sound.shape[0] not in possible_lengths:
        msecs = sound.shape[0] / 16.0
        err_msg = f'The sound length is wrong! Expected 10, 20 or 30 milliseconds, got {msecs} milliseconds.'
        raise ValueError(err_msg)
    res = np.asarray(sound * 32768.0, dtype=np.int16)
    return res.tobytes()


def calculate_voice_probabilities(sound: np.ndarray, vad_ensemble: List[Vad]) -> np.ndarray:
    window_size = 480
    shift_size = 160
    number_of_windows = (sound.shape[0] - window_size) // shift_size + 1
    probabilities = np.zeros((number_of_windows,), dtype=np.float32)
    for window_idx in range(number_of_windows):
        window_start = window_idx * shift_size
        window_end = min(sound.shape[0], window_start + window_size)
        cur_window = np.zeros((window_size,), dtype=np.float32)
        if (window_end - window_start) < window_size:
            cur_window[:(window_end - window_start)] = sound[window_start:window_end]
        else:
            cur_window = sound[window_start:window_end]
        buffer = sound_to_bytes(cur_window)
        for cur_vad in vad_ensemble:
            probabilities[window_idx] += (1.0 if cur_vad.is_speech(buffer, 16_000) else 0.0)
        probabilities[window_idx] /= float(len(vad_ensemble))
        del cur_window
    return probabilities


def start_to_time(start_pos: int) -> int:
    return start_pos * 160


def end_to_time(end_pos: int) -> int:
    return (end_pos - 1) * 160 + 480


def split_long_sound(sound: np.ndarray, vad_ensemble: List[Vad],
                     max_sound_len: int = 16_000 * 50) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    speech_probabilities = calculate_voice_probabilities(sound, vad_ensemble)
    start_pos = 0
    best_pos = 0
    subsounds = []
    bounds_of_subsounds = []
    for cur_pos in range(1, speech_probabilities.shape[0]):
        if (cur_pos - start_pos) >= (max_sound_len // 160):
            if best_pos > start_pos:
                cur_bounds = (
                    start_to_time(start_pos),
                    end_to_time(best_pos + 1)
                )
                subsounds.append(sound[cur_bounds[0]:cur_bounds[1]])
                start_pos = best_pos
            else:
                cur_bounds = (
                    start_to_time(start_pos),
                    end_to_time(cur_pos)
                )
                subsounds.append(sound[cur_bounds[0]:cur_bounds[1]])
                start_pos = cur_pos - 1
                best_pos = start_pos
            bounds_of_subsounds.append(cur_bounds)
        else:
            if speech_probabilities[cur_pos] <= speech_probabilities[best_pos]:
                best_pos = cur_pos
    cur_bounds = (
        start_to_time(start_pos),
        sound.shape[0]
    )
    subsounds.append(sound[cur_bounds[0]:cur_bounds[1]])
    bounds_of_subsounds.append(cur_bounds)
    return subsounds, bounds_of_subsounds


def stick_subsounds(subsounds: List[np.ndarray], bounds_of_subsounds: List[Tuple[int, int]],
                    indices: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    subsound_start = bounds_of_subsounds[indices[0]][0]
    subsound_end = bounds_of_subsounds[indices[1] - 1][1]
    new_subsound = np.empty((subsound_end - subsound_start,), dtype=np.float32)
    for idx in range(indices[0], indices[1]):
        start = bounds_of_subsounds[idx][0] - subsound_start
        end = bounds_of_subsounds[idx][1] - subsound_start
        new_subsound[start:end] = subsounds[idx]
    return new_subsound, (subsound_start, subsound_end)


def stick_short_subsounds_to_longer_neighbours(
        subsounds: List[np.ndarray],
        bounds_of_subsounds: List[Tuple[int, int]]
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    n = len(subsounds)
    if n < 2:
        return subsounds, bounds_of_subsounds
    indices_of_sticked_subsounds = []
    start_idx = -1
    for idx in range(n):
        if subsounds[idx].shape[0] <= MIN_SOUND_LENGTH:
            if start_idx < 0:
                start_idx = idx
        else:
            if start_idx >= 0:
                subsound_start = bounds_of_subsounds[start_idx][0]
                subsound_end = bounds_of_subsounds[idx - 1][1]
                if (subsound_end - subsound_start) <= MIN_SOUND_LENGTH:
                    indices_of_sticked_subsounds.append((start_idx, idx + 1))
                else:
                    indices_of_sticked_subsounds.append((start_idx, idx))
                    indices_of_sticked_subsounds.append((idx, idx + 1))
                start_idx = -1
            else:
                indices_of_sticked_subsounds.append((idx, idx + 1))
    if start_idx >= 0:
        subsound_start = bounds_of_subsounds[start_idx][0]
        subsound_end = n
        if (subsound_end - subsound_start) <= MIN_SOUND_LENGTH:
            if len(indices_of_sticked_subsounds) > 0:
                indices_of_sticked_subsounds[-1] = (
                    indices_of_sticked_subsounds[-1][0],
                    n
                )
            else:
                indices_of_sticked_subsounds.append((start_idx, n))
        else:
            indices_of_sticked_subsounds.append((start_idx, n))
    new_subsounds = []
    new_bounds_of_subsounds = []
    for indices in indices_of_sticked_subsounds:
        subsound, bounds_of_subsound = stick_subsounds(subsounds, bounds_of_subsounds, indices)
        new_subsounds.append(subsound)
        new_bounds_of_subsounds.append(bounds_of_subsound)
    return new_subsounds, new_bounds_of_subsounds
