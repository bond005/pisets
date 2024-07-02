import logging
from typing import List, Optional, Tuple, Union

import asyncio
import math
from nltk import wordpunct_tokenize
import numpy as np
from tqdm import trange
import torch
from transformers import pipeline, Pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig

from wav_io.wav_io import TARGET_SAMPLING_FREQUENCY


MIN_SOUND_LENGTH: int = 1600
WHISPER_NUM_BEAMS: int = 5
MINIBATCH_SIZE: int = 4
OSCILLATORY_HALLUCINATION_MIN_SIZE: int = 4
asr_logger = logging.getLogger(__name__)


def find_repeated_tokens(tokens: List[str], start_pos: int = 0) -> Union[Tuple[int, int], None]:
    token_idx = start_pos
    n_tokens = len(tokens)
    repeated_group_start = -1
    repeated_group_end = -1
    while token_idx < (n_tokens - 1):
        if tokens[token_idx + 1] == tokens[token_idx]:
            repeated_group_start = token_idx
            repeated_group_end = token_idx + 2
            while repeated_group_end < n_tokens:
                if tokens[repeated_group_end] != tokens[repeated_group_start]:
                    break
                repeated_group_end += 1
            if (repeated_group_end - repeated_group_start) >= OSCILLATORY_HALLUCINATION_MIN_SIZE:
                break
            token_idx += 1
            repeated_group_start = -1
            repeated_group_end = -1
        else:
            token_idx += 1
    if (repeated_group_start < 0) or (repeated_group_end < 0):
        return None
    return repeated_group_start, repeated_group_end


def find_tokens_in_text(source_text: str, tokens: List[str], start_pos: int = 0) -> Tuple[int, int]:
    if len(tokens) == 0:
        err_msg = 'The tokens list is empty!'
        asr_logger.error(err_msg)
        raise RuntimeError(err_msg)
    if len(source_text) == 0:
        err_msg = 'The source text is empty!'
        asr_logger.error(err_msg)
        raise RuntimeError(err_msg)
    source_text_ = source_text.lower()
    found_idx = source_text_[start_pos:].find(tokens[0])
    if found_idx < 0:
        err_msg = f'The token {tokens[0]} does not found in the text {source_text}'
        asr_logger.error(err_msg)
        raise RuntimeError(err_msg)
    token_start_pos = found_idx + start_pos
    token_end_pos = token_start_pos + len(tokens[0])
    ok = True
    if token_start_pos > 0:
        if source_text_[token_start_pos - 1].isalnum():
            ok = False
    if ok:
        if token_end_pos < len(source_text_):
            if source_text_[token_end_pos].isalnum():
                ok = False
    while not ok:
        found_idx = source_text_[token_end_pos:].find(tokens[0])
        if found_idx < 0:
            err_msg = f'The token {tokens[0]} does not found in the text {source_text}'
            asr_logger.error(err_msg)
            raise RuntimeError(err_msg)
        token_start_pos = found_idx + token_end_pos
        token_end_pos = token_start_pos + len(tokens[0])
        ok = True
        if token_start_pos > 0:
            if source_text_[token_start_pos - 1].isalnum():
                ok = False
        if ok:
            if token_end_pos < len(source_text_):
                if source_text_[token_end_pos].isalnum():
                    ok = False
    if len(tokens) < 2:
        return token_start_pos, token_end_pos
    return token_start_pos, find_tokens_in_text(source_text, tokens[1:], token_end_pos)[1]


def remove_oscillatory_hallucinations(input_transcription: str) -> str:
    tokens = wordpunct_tokenize(input_transcription.lower())
    tokens_without_punctuation = list(filter(lambda it: it.isalnum(), tokens))
    if len(tokens_without_punctuation) >= OSCILLATORY_HALLUCINATION_MIN_SIZE:
        hallucination_bounds = find_repeated_tokens(tokens_without_punctuation)
        if hallucination_bounds is None:
            output_transcription = ' '.join(input_transcription.strip().split())
        else:
            start_pos, end_pos = find_tokens_in_text(
                input_transcription,
                tokens_without_punctuation[hallucination_bounds[0]:hallucination_bounds[1]]
            )
            first_token_start, first_token_end = find_tokens_in_text(
                input_transcription,
                tokens_without_punctuation[hallucination_bounds[0]:(hallucination_bounds[0] + 1)]
            )
            output_transcription = input_transcription[0:first_token_end] + input_transcription[end_pos:]
            output_transcription = remove_oscillatory_hallucinations(' '.join(output_transcription.strip().split()))
    else:
        output_transcription = ' '.join(input_transcription.strip().split())
    return output_transcription


def check_language(lang: str) -> str:
    lang_ = ' '.join(list(filter(lambda it: it.isalnum(), wordpunct_tokenize(lang)))).lower()
    if lang_ in {'en', 'eng', 'engl', 'english'}:
        language_name = 'en'
    else:
        language_name = 'ru'
        if lang_ not in {'ru', 'russ', 'rus', 'russian'}:
            err_msg = f'The language {lang} is not supported!'
            asr_logger.error(err_msg)
            raise RuntimeError(err_msg)
    return language_name


def initialize_model_for_speech_segmentation(language: str = 'ru', model_info: Optional[str] = None) -> Pipeline:
    if model_info is not None:
        model_name = model_info
    else:
        if language == 'ru':
            model_name = 'bond005/wav2vec2-large-ru-golos'
        else:
            model_name = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'
    try:
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
    except Exception as err:
        asr_logger.error(str(err))
        raise
    return segmenter


def initialize_model_for_speech_recognition(language: str = 'ru', model_info: Optional[str] = None) -> (
        Tuple)[WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig]:
    if model_info is not None:
        model_name = model_info
    else:
        if language == 'ru':
            model_name = 'bond005/whisper-large-v2-ru-podlodka'
        else:
            model_name = 'openai/whisper-large-v2'
    try:
        processor = WhisperProcessor.from_pretrained(
            model_name, language='Russian' if (language == 'ru') else 'English',
            task='transcribe'
        )
    except Exception as err:
        asr_logger.error(str(err))
        raise
    config = GenerationConfig.from_pretrained(model_name)
    if config.num_beams is None:
        config.num_beams = WHISPER_NUM_BEAMS
    elif config.num_beams < WHISPER_NUM_BEAMS:
        config.num_beams = WHISPER_NUM_BEAMS
    config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language='Russian' if (language == 'ru') else 'English',
        task='transcribe'
    )
    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    except Exception as err:
        asr_logger.error(str(err))
        raise
    model.config.forced_decoder_ids = config.forced_decoder_ids
    if torch.cuda.is_available():
        model = model.cuda()
    return processor, model, config


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
        asr_logger.error(err_msg)
        raise ValueError(err_msg)
    if len(mono_sound.shape) != 1:
        err_msg = f'The sound channel number is wrong! Expected 1, got {len(mono_sound.shape)}.'
        asr_logger.error(err_msg)
        raise ValueError(err_msg)
    if mono_sound.shape[0] <= MIN_SOUND_LENGTH:
        err_msg = f'The sound length = {mono_sound.shape[0]} is too short. ' \
                  f'The expected length should be greater than {MIN_SOUND_LENGTH}.'
        asr_logger.error(err_msg)
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


def recognize_sounds(sounds: List[np.ndarray], processor: WhisperProcessor, model: WhisperForConditionalGeneration,
                     config: GenerationConfig) -> List[str]:
    for idx, val in enumerate(sounds):
        if not isinstance(val, np.ndarray):
            err_msg = f'The sound {idx} is wrong! Expected {type(np.array([1, 2]))}, got {type(val)}.'
            asr_logger.error(err_msg)
            raise ValueError(err_msg)
        if len(val.shape) != 1:
            err_msg = f'The sound {idx} channel number is wrong! Expected 1, got {len(val.shape)}.'
            asr_logger.error(err_msg)
            raise ValueError(err_msg)

    all_transcriptions = []
    n_batches = math.ceil(len(sounds) / MINIBATCH_SIZE)
    if n_batches > 1:
        for batch_idx in trange(n_batches):
            batch_start = batch_idx * MINIBATCH_SIZE
            batch_end = min(len(sounds), batch_start + MINIBATCH_SIZE)
            inputs = processor(sounds[batch_start:batch_end], return_tensors='pt',
                               sampling_rate=TARGET_SAMPLING_FREQUENCY)
            input_features = inputs.input_features
            del inputs
            with torch.no_grad():
                generated_ids = model.generate(inputs=input_features.to(model.device).half(), generation_config=config)
            del input_features
            transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            del generated_ids
            all_transcriptions += transcriptions
            del transcriptions
    else:
        inputs = processor(sounds, return_tensors='pt',
                           sampling_rate=TARGET_SAMPLING_FREQUENCY)
        input_features = inputs.input_features
        del inputs
        with torch.no_grad():
            generated_ids = model.generate(inputs=input_features.to(model.device).half(), generation_config=config)
        del input_features
        all_transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        del generated_ids
    return [remove_oscillatory_hallucinations(it) for it in all_transcriptions]


async def transcribe(mono_sound: np.ndarray, segmenter: Pipeline,
               asr: Tuple[WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig],
               max_segment_size: int) -> List[Tuple[float, float, str]]:
    speech_segments = segmentate_sound(mono_sound, segmenter, max_segment_size)
    asr_logger.info(f'The speech sound is divided into {len(speech_segments)} segments.')
    speech_segments_ = []
    segment_lengths = []
    for it in speech_segments:
        segment_lengths.append(it[1] - it[0])
        segment_start = min(round(it[0] * TARGET_SAMPLING_FREQUENCY), mono_sound.shape[0])
        segment_end = min(round(it[1] * TARGET_SAMPLING_FREQUENCY), mono_sound.shape[0])
        if segment_start >= segment_end:
            err_msg = f'Segments {speech_segments} are wrong!'
            asr_logger.error(err_msg)
            raise RuntimeError(err_msg)
        speech_segments_.append((segment_start, segment_end))
    segment_lengths = sorted([(it[1] - it[0]) for it in speech_segments])
    mean_length = float(np.mean(segment_lengths))
    asr_logger.info(f'Minimal segment duration is {segment_lengths[0]} seconds.')
    asr_logger.info(f'Maximal segment duration is {segment_lengths[-1]} seconds.')
    asr_logger.info(f'Median segment duration is {segment_lengths[len(segment_lengths) // 2]} seconds.')
    asr_logger.info(f'Mean segment duration is {mean_length} seconds.')
    recognized_transcriptions = recognize_sounds(
        sounds=[mono_sound[bounds[0]:bounds[1]] for bounds in speech_segments_],
        processor=asr[0],
        model=asr[1],
        config=asr[2]
    )
    del speech_segments_
    results = list(filter(
        lambda it2: len(it2[2]) > 0,
        map(
            lambda it: (it[0][0], it[0][1], it[1].strip()),
            zip(speech_segments, recognized_transcriptions)
        )
    ))
    return results
