import os.path
from typing import Tuple, Union
import wave

import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


def load_sound(fname: str) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], None]:
    with wave.open(fname, 'rb') as fp:
        n_channels = fp.getnchannels()
        if n_channels not in {1, 2}:
            err_msg = f'The channels number of the WAV sound is wrong! ' \
                      f'Expected 1 or 2, got {n_channels}.'
            raise ValueError(err_msg)
        fs = fp.getframerate()
        if fs != 16_000:
            err_msg = f'The sampling frequency the WAV sound is wrong! ' \
                      f'Expected 16000 Hz, got {fs} Hz.'
            raise ValueError(err_msg)
        bytes_per_sample = fp.getsampwidth()
        if bytes_per_sample not in {1, 2}:
            err_msg = f'The sample width of the WAV sound is wrong! ' \
                      f'Expected 1 or 2, got {bytes_per_sample}.'
            raise ValueError(err_msg)
        sound_length = fp.getnframes()
        sound_bytes = fp.readframes(sound_length)
    if sound_length == 0:
        return None
    if bytes_per_sample == 1:
        data = np.frombuffer(sound_bytes, dtype=np.uint8)
    else:
        data = np.frombuffer(sound_bytes, dtype=np.int16)
    if len(data.shape) != 1:
        err_msg = f'The loaded data is wrong! Expected 1-d array, got {len(data.shape)}-d one.'
        raise ValueError(err_msg)
    if n_channels == 1:
        if bytes_per_sample == 1:
            sound = (data.astype(np.float32) - 128.0) / 128.0
        else:
            sound = data.astype(np.float32) / 32768.0
    else:
        channel1 = data[0::2]
        channel2 = data[1::2]
        if bytes_per_sample == 1:
            sound = (
                (channel1.astype(np.float32) - 128.0) / 128.0,
                (channel2.astype(np.float32) - 128.0) / 128.0
            )
        else:
            sound = (
                channel1.astype(np.float32) / 32768.0,
                channel2.astype(np.float32) / 32768.0
            )
    return sound


def transform_to_wavpcm(src_fname: str, dst_fname: str) -> None:
    found_idx = src_fname.rfind('.')
    if found_idx < 0:
        err_msg = f'The extension of the file "{src_fname}" is unknown. ' \
                  f'So, I cannot determine a format of this sound file.'
        raise ValueError(err_msg)
    if not os.path.isfile(src_fname):
        err_msg = f'The file "{src_fname}" does not exist!'
        raise IOError(err_msg)
    source_audio_extension = dst_fname[(found_idx + 1):]
    try:
        audio = AudioSegment.from_file(src_fname, format=source_audio_extension)
    except CouldntDecodeError as e1:
        audio = None
        additional_err_msg = str(e1)
    except BaseException as e2:
        audio = None
        additional_err_msg = str(e2)
    else:
        additional_err_msg = ''
    if audio is None:
        err_msg = f'The file "{src_fname}" cannot be opened.'
        if additional_err_msg != '':
            err_msg += f' {additional_err_msg}'
        raise IOError(err_msg)
    if audio.channels != 1:
        audio.set_channels(1)
    if audio.frame_rate != 16_000:
        audio.set_frame_rate(16_000)
    if audio.frame_width != 2:
        audio.set_sample_width(2)
    audio.export(dst_fname, format='wav')
