from typing import List, Tuple

import numpy as np
import torch
from transformers import pipeline, Pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig

from wav_io.wav_io import TARGET_SAMPLING_FREQUENCY


MIN_SOUND_LENGTH: int = 1600
WHISPER_NUM_BEAMS: int = 5


def initialize_model_for_speech_segmentation(language: str = 'ru') -> Pipeline:
    if language == 'ru':
        model_name = 'bond005/wav2vec2-large-ru-golos'
    else:
        model_name = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'
    if torch.cuda.is_available():
        segmenter = pipeline(
            'automatic-speech-recognition', model=model_name,
            chunk_length_s=10, stride_length_s=(4, 2),
            device='cuda:0'
        )
    else:
        segmenter = pipeline(
            'automatic-speech-recognition', model=model_name,
            chunk_length_s=10, stride_length_s=(4, 2)
        )
    return segmenter


def select_word_groups(words: List[Tuple[float, float]], segment_size: int) -> List[List[Tuple[float, float]]]:
    if len(words) < 2:
        return [words]
    if words[-1][1] - words[0][0] < segment_size:
        return [words]
    if len(words) == 2:
        return [words[0:1], words[1:2]]
    max_pause = words[1][0] - words[0][1]
    best_word_idx = 0
    for word_idx in range(1, len(words) - 1):
        new_pause = words[word_idx + 1][0] - words[word_idx][1]
        if new_pause > max_pause:
            max_pause = new_pause
            best_word_idx = word_idx
    left_frame = words[0:(best_word_idx + 1)]
    right_frame = words[(best_word_idx + 1):]
    if (left_frame[-1][1] - left_frame[0][0]) < segment_size:
        word_groups = [left_frame]
    else:
        word_groups = select_word_groups(left_frame, segment_size)
    if (right_frame[-1][1] - right_frame[0][0]) < segment_size:
        word_groups += [right_frame]
    else:
        word_groups += select_word_groups(right_frame, segment_size)
    return word_groups


def split_long_segments(segments: List[Tuple[float, float]], max_segment_size: int) -> List[Tuple[float, float]]:
    new_segments = []
    for segment_start, segment_end in segments:
        if (segment_end - segment_start) <= max_segment_size:
            new_segments.append((segment_start, segment_end))
        else:
            segment_start_ = segment_start + 0.15
            segment_end_ = segment_end - 0.15
            if (segment_end_ - segment_start_) <= max_segment_size:
                new_segments.append((segment_start_, segment_end_))
            else:
                div = 2
                while ((segment_end_ - segment_start_) / float(div)) > max_segment_size:
                    div += 1
                new_segment_len = (segment_end_ - segment_start_) / float(div)
                segment_start__ = segment_start_
                segment_end__ = segment_start__ + new_segment_len
                new_segments.append((segment_start__, segment_end__))
                for _ in range(div - 1):
                    segment_start__ = segment_end__
                    segment_end__ = segment_start__ + new_segment_len
                    new_segments.append((segment_start__, segment_end__))
    return new_segments


def strip_segments(segments: List[Tuple[float, float]], max_sound_duration: float) -> List[Tuple[float, float]]:
    return [(max(0.0, it[0]), min(it[1], max_sound_duration)) for it in segments]


def segmentate_sound(mono_sound: np.ndarray, segmenter: Pipeline,
                     max_segment_size: int) -> List[Tuple[float, float]]:
    if not isinstance(mono_sound, np.ndarray):
        err_msg = f'The sound is wrong! Expected {type(np.array([1, 2]))}, got {type(mono_sound)}.'
        raise ValueError(err_msg)
    if len(mono_sound.shape) != 1:
        err_msg = f'The sound channel number is wrong! Expected 1, got {len(mono_sound.shape)}.'
        raise ValueError(err_msg)
    if mono_sound.shape[0] <= MIN_SOUND_LENGTH:
        err_msg = f'The sound length = {mono_sound.shape[0]} is too short. ' \
                  f'The expected length should be greater than {MIN_SOUND_LENGTH}.'
        raise ValueError(err_msg)

    output = segmenter(mono_sound, return_timestamps='word')
    word_bounds = [(float(it['timestamp'][0]), float(it['timestamp'][1])) for it in output['chunks']]
    if len(word_bounds) < 1:
        return []
    if len(word_bounds) == 1:
        segment_start = word_bounds[0][0] - 0.15
        segment_end = word_bounds[0][1] + 0.15
        return strip_segments(split_long_segments([(segment_start, segment_end)], max_segment_size),
                              mono_sound.shape[0] / TARGET_SAMPLING_FREQUENCY)
    if (word_bounds[-1][1] - word_bounds[0][0]) <= max_segment_size:
        segment_start = word_bounds[0][0] - 0.15
        segment_end = word_bounds[-1][1] + 0.15
        return strip_segments(split_long_segments([(segment_start, segment_end)], max_segment_size),
                              mono_sound.shape[0] / TARGET_SAMPLING_FREQUENCY)
    word_groups = select_word_groups(word_bounds, max_segment_size)
    segments = split_long_segments(
        [(cur_group[0][0] - 0.15, cur_group[-1][0] + 0.15) for cur_group in word_groups],
        max_segment_size
    )

    segments = strip_segments(segments, mono_sound.shape[0] / TARGET_SAMPLING_FREQUENCY)
    n_segments = len(segments)

    if n_segments > 1:
        for idx in range(1, n_segments):
            if segments[idx - 1][1] > segments[idx][0]:
                overlap = segments[idx - 1][1] - segments[idx][0]
                segments[idx - 1] = (segments[idx - 1][0], segments[idx - 1][1] - overlap / 2.0)
                segments[idx] = (segments[idx][0] + overlap / 2.0, segments[idx][1])

    return segments


def recognize_sound(mono_sound: np.ndarray, processor: WhisperProcessor, model: WhisperForConditionalGeneration,
                    config: GenerationConfig, lang: str) -> str:
    if not isinstance(mono_sound, np.ndarray):
        err_msg = f'The sound is wrong! Expected {type(np.array([1, 2]))}, got {type(mono_sound)}.'
        raise ValueError(err_msg)
    if len(mono_sound.shape) != 1:
        err_msg = f'The sound channel number is wrong! Expected 1, got {len(mono_sound.shape)}.'
        raise ValueError(err_msg)
    if mono_sound.shape[0] <= MIN_SOUND_LENGTH:
        err_msg = f'The sound length = {mono_sound.shape[0]} is too short. ' \
                  f'The expected length should be greater than {MIN_SOUND_LENGTH}.'
        raise ValueError(err_msg)

    inputs = processor(mono_sound, return_tensors='pt')
    input_features = inputs.input_features
    del inputs
    with torch.no_grad():
        generated_ids = model.generate(inputs=input_features.to(model.device), generation_config=config,
                                       num_beams=WHISPER_NUM_BEAMS, task='transcribe', language=lang)
    del input_features
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    del generated_ids
    return transcription
