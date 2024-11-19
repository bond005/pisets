import copy
import gc
import logging
from typing import List, Optional, Tuple, Union

from nltk import wordpunct_tokenize
import numpy as np
from tqdm import tqdm
import torch
from transformers import pipeline, Pipeline

from utils.utils import time_to_str
from wav_io.wav_io import TARGET_SAMPLING_FREQUENCY


MIN_SOUND_LENGTH: int = 1600
OSCILLATORY_HALLUCINATION_MIN_SIZE: int = 4
asr_logger = logging.getLogger(__name__)


def find_repeated_tokens(tokens: List[str], start_pos: int = 0) -> Union[Tuple[int, int], None]:
    """
    Accepts a list of strings, usually words. Finds the first occurence of
    OSCILLATORY_HALLUCINATION_MIN_SIZE or more equal words in `tokens[start_pos:]`.
    
    Output: the start (inclusive) and end (exclusive) indices of the found subsequence, or None
    if no such subsequence is found.
    """
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
    """
    Arguments:
    - source_text: a text
    - tokens: a sequence of lowercase words
    - start_pos: an index to search from

    Returns start and end positions of the word sequence in the source text, excluding punctuation.
    Raises RuntimeError if no such positions found.
    """
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
        err_msg = f'The token {tokens[0]} was not found in the text {source_text}'
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
            err_msg = f'The token {tokens[0]} was not found in the text {source_text}'
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
    """
    Searches for a subsequence of OSCILLATORY_HALLUCINATION_MIN_SIZE or more consecutive repetitions
    of the same word. If found, removes the repetitions. Also strips the input string and performs a
    space deduplication.

    TODO consider fixing cases when both calls to `find_tokens_in_text` attend to different fragments:
    ```
    text = 'What a mess, someone stole the words! A-a-a-a!'
    remove_oscillatory_hallucinations(text)
    >>> 'What a-a!'
    """
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
    """
    Loads and returns a specified Pipeline to use for speech segmentation.
    
    Arguments:
    - `model_info` can be any Hugging Face model name or path for 'automatic-speech-recognition' pipeline.
        A reasonable option is 'bond005/wav2vec2-base-ru-birm' (used in tests).
    - `language` is used only if `model_info` is not provided to load a default pipeline:
        - for language='ru': 'bond005/wav2vec2-large-ru-golos'
        - for language='en': 'jonatasgrosman/wav2vec2-large-xlsr-53-english'.
    
    Returned value: an AutomaticSpeechRecognitionPipeline, to be called on mono sound with rate 16_000
    and argument `return_timestamps='word'`. In Pisets, only the output timestamps are used, not the
    transcribed speech.

    Example:
    ```
    import librosa
    waveform, _ = librosa.load('tests/testdata/test_sound_ru.wav', sr=None)
    segmenter = initialize_model_for_speech_segmentation()
    segmenter(waveform, return_timestamps='word')

    >>> {
      'text': 'нейронные сети это хорошо',
      'chunks': [
        {'text': 'нейронные', 'timestamp': (0.44, 0.98)},
        {'text': 'сети', 'timestamp': (1.08, 1.32)},
        {'text': 'это', 'timestamp': (1.7, 1.82)},
        {'text': 'хорошо', 'timestamp': (1.88, 2.24)}
      ]
    }
    ```
    """
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
                chunk_length_s=10, stride_length_s=(4, 2), device='cuda:0'
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


def initialize_model_for_speech_classification(model_info: Optional[str] = None) -> Pipeline:
    """
    Loads and returns a specified Pipeline to use for voice activity detection (VAD).
    
    Arguments:
    - `model_info` can be any Hugging Face model name or path for 'audio-classification' pipeline.
      By default, loads 'MIT/ast-finetuned-audioset-10-10-0.4593' (a default VAD for Pisets).
    
    Returned value: An AudioClassificationPipeline, to be called on mono sound with rate 16_000 and returns
    a list of classes. Pisets considers a sound as speech if the top class name contains a word "speech".

    Example:
    ```
    import librosa
    waveform, _ = librosa.load('tests/testdata/test_sound_ru.wav', sr=None)
    vad = initialize_model_for_speech_classification()
    vad(waveform)

    >>> [
      {'score': 0.6382841467857361, 'label': 'Speech'},
      {'score': 0.024410240352153778, 'label': 'Animal'},
      {'score': 0.02063129097223282, 'label': 'Inside, small room'},
      {'score': 0.018520403653383255, 'label': 'Insect'},
      {'score': 0.01799442060291767, 'label': 'Owl'}
    ]
    ```
    """
    if model_info is not None:
        model_name = model_info
    else:
        model_name = 'MIT/ast-finetuned-audioset-10-10-0.4593'
    try:
        if torch.cuda.is_available():
            classifier = pipeline(
                'audio-classification', model=model_name, device='cuda:0'
            )
        else:
            classifier = pipeline(
                'audio-classification', model=model_name
            )
    except Exception as err:
        asr_logger.error(str(err))
        raise
    return classifier


def initialize_model_for_speech_recognition(language: str = 'ru', model_info: Optional[str] = None) -> Pipeline:
    """
    Loads and returns a specified Pipeline to use for speech recognition.
    
    Arguments:
    - `model_info` can be any Hugging Face model name or path for 'automatic-speech-recognition' pipeline.
    - `language` is used only if `model_info` is not provided to load a default pipeline:
        - for language='ru': 'bond005/whisper-large-v3-ru-podlodka'
        - for language='en': 'openai/whisper-large-v3'.
    
    Returned value: an AutomaticSpeechRecognitionPipeline, to be called on mono sound with rate 16_000.

    Example:
    ```
    import librosa
    waveform, _ = librosa.load('tests/testdata/test_sound_ru.wav', sr=None)
    recognizer = initialize_model_for_speech_recognition()
    recognizer(waveform)
    
    >>> {'text': 'Нейронные сети — это хорошо.'}
    ```
    """
    if model_info is not None:
        model_name = model_info
    else:
        if language == 'ru':
            model_name = 'bond005/whisper-large-v3-ru-podlodka'
        else:
            model_name = 'openai/whisper-large-v3'
    try:
        if torch.cuda.is_available():
            recognizer = pipeline(
                'automatic-speech-recognition', model=model_name,
                chunk_length_s=20, stride_length_s=(4, 2),
                device='cuda:0', model_kwargs={'attn_implementation': 'sdpa'}, torch_dtype=torch.float16
            )
        else:
            recognizer = pipeline(
                'automatic-speech-recognition', model=model_name,
                chunk_length_s=20, stride_length_s=(4, 2)
            )
    except Exception as err:
        asr_logger.error(str(err))
        raise
    return recognizer


def select_word_groups(
    words: List[Tuple[float, float]],
    segment_size: float
) -> List[List[Tuple[float, float]]]:
    """
    Accepts a list of consecutive segments, each segment is a tuple (start_time, end_time).

    Iteratively splits the list of segments into "left" and "right" part by the largest pause between segments,
    then splits both parts the same way, and so on. A list of segments is splitted only if its total length
    (from the beginning of the first segment to the end of the last segment) is no shorter than `segment_size`,
    otherwise it is not splitted.

    The result is a list of groups, each group is a list of segments. The result may be an empty list,
    oherwise each group is guaranteed to be not empty.

    Example:
    ```
    A, B, C, D, E, F = (3, 4), (4, 9), (12, 14), (14.5, 18), (18, 20), (28, 29)

    result = select_word_groups([A, B, C, D, E, F], segment_size=9)
    assert result == [[A, B], [C, D, E], [F]]

    result = select_word_groups([A, B, C, D, E, F], segment_size=0)
    assert result == [[A], [B], [C], [D], [E], [F]]

    result = select_word_groups([A], segment_size=0)
    assert result == [[A]]

    result = select_word_groups([], segment_size=0)
    assert result == [[]]
    ```
    """
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


def strip_segments(
    segments: List[Tuple[float, float]],
    max_sound_duration: float
) -> List[Tuple[float, float]]:
    """
    Clips tuples (start_time, end_time) between (0, max_sound_duration).
    """
    return [(max(0.0, it[0]), min(it[1], max_sound_duration)) for it in segments]


def join_short_segments_to_long_ones(
    segments: List[Tuple[float, float]],
    min_segment_size: float
) -> List[Tuple[float, float]]:
    """
    Iterates segments from left to right and merges two segments if two conditions are met:
    1) The segment is shorter than `min_segment_size`
    2) The pause between it and a neigbour segment is shorter than `min_segment_size`

    If segment is short, not the first, not the last, and both pauses around it are shorter
    than `min_segment_size`, then merges along the shorter pause.
    """
    n_segments = len(segments)

    if n_segments > 1:
        new_segments = copy.deepcopy(segments)

        segment_idx = 0
        while segment_idx < len(new_segments):
            segment_start, segment_end = new_segments[segment_idx]
            if (segment_end - segment_start) < min_segment_size:
                if (segment_idx > 0) and (segment_idx < len(new_segments) - 1):
                    distance_to_left = segment_start - new_segments[segment_idx - 1][1]
                    distance_to_right = new_segments[segment_idx + 1][0] - segment_end
                    if distance_to_left < distance_to_right:
                        if distance_to_left < min_segment_size:
                            new_segments[segment_idx - 1] = (
                                new_segments[segment_idx - 1][0],
                                segment_end
                            )
                            _ = new_segments.pop(segment_idx)
                        else:
                            segment_idx += 1
                    else:
                        if distance_to_right < min_segment_size:
                            new_segments[segment_idx + 1] = (
                                segment_start,
                                new_segments[segment_idx + 1][1]
                            )
                            _ = new_segments.pop(segment_idx)
                        else:
                            segment_idx += 1
                elif segment_idx > 0:
                    distance_to_left = segment_start - new_segments[segment_idx - 1][1]
                    if distance_to_left < min_segment_size:
                        new_segments[segment_idx - 1] = (
                            new_segments[segment_idx - 1][0],
                            segment_end
                        )
                        _ = new_segments.pop(segment_idx)
                    else:
                        segment_idx += 1
                else:
                    distance_to_right = new_segments[segment_idx + 1][0] - segment_end
                    if distance_to_right < min_segment_size:
                        new_segments[segment_idx + 1] = (
                            segment_start,
                            new_segments[segment_idx + 1][1]
                        )
                        _ = new_segments.pop(segment_idx)
                    else:
                        segment_idx += 1
            else:
                segment_idx += 1
    else:
        new_segments = segments

    return new_segments


def segment_sound(
    mono_sound: np.ndarray,
    segmenter: Pipeline,
    min_segment_size: float,
    max_segment_size: float,
    indent_for_silence: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Arguments:
    - mono_sound: 1D waveform with rate 16_000 (equals wav_io.TARGET_SAMPLING_FREQUENCY),
      no shorter than asr.MIN_SOUND_LENGTH.
    - segmenter: an AutomaticSpeechRecognitionPipeline that can return word timestamps. See
      `initialize_model_for_speech_segmentation` for details.
    - min_segment_size: see below
    - max_segment_size: see below
    - indent_for_silence: see below

    Output: a list of tuples (start_time, end_time) for all found utterances, can be empty.
    
    Performs the following actions:
    1) Obtains speech segment boundaries by applying `segmenter` to `mono_sound`.
    2) Performs `select_word_groups` with `max_segment_size` argment and then merge each
      group into one segment. Thus we join adjacent segments with short pauses between, but
      do not exceed `max_segment_size` (if the initial segments do not exceed it).
    3) Adds a padding `indent_for_silence` (in seconds) added around each obtained segment,
      but without going beyond the input audio boundaries. Note that we just expand the borders,
      but we do not check that padding is a silence. If the segments start to overlap after
      adding a padding, we shift the overlapping borders so that the end of the previous sement
      is a start of the next segment. (TODO consider that using the same silence fragment is
      not a bad overlap, we should not fix this).
    4) Merges segments using `join_short_segments_to_long_ones`. A segment is merged with a
      neighbour if it is shorter than `min_segment_size` and the pause between them is also
      shorter than `min_segment_size`.

    NOTE: is not guaranteed that all output segments are larger than `min_segment_size`,
    if total length returned by `segmenter` (from the start of the first segment to the end of
    the last segment), including `indent_for_silence`, is not enough.

    NOTE: is not guaranteed that all output segments are smaller than `max_segment_size`,
    because a single segment returned by `segmenter` can exceed `max_segment_size`, especially
    after expanding segments with `indent_for_silence`. Another reason is that 4-th step (joining
    short segments to long ones) may produce segments that exceed `max_segment_size`. For example,
    if A = (0, 0.75), B = (1.5, 6), they will be joined to (0, 6), exceeding `max_segment_size=5`,
    while both segments do not.
    """
    if not isinstance(mono_sound, np.ndarray):
        err_msg = f'The sound is wrong! Expected {type(np.array([1, 2]))}, got {type(mono_sound)}.'
        asr_logger.error(err_msg)
        raise ValueError(err_msg)
    if len(mono_sound.shape) != 1:
        err_msg = f'The sound channel number is wrong! Expected 1, got {len(mono_sound)}.'
        asr_logger.error(err_msg)
        raise ValueError(err_msg)
    if mono_sound.shape[0] <= MIN_SOUND_LENGTH:
        err_msg = f'The sound length = {mono_sound.shape[0]} is too short. ' \
                  f'The expected length should be greater than {MIN_SOUND_LENGTH}.'
        asr_logger.error(err_msg)
        raise ValueError(err_msg)

    output = segmenter(mono_sound, return_timestamps='word')
    gc.collect()
    torch.cuda.empty_cache()

    word_bounds = [(float(it['timestamp'][0]), float(it['timestamp'][1])) for it in output['chunks']]
    if len(word_bounds) < 1:
        return []
    if len(word_bounds) == 1:
        segment_start = word_bounds[0][0] - indent_for_silence
        segment_end = word_bounds[0][1] + indent_for_silence
        return strip_segments([(segment_start, segment_end)],
                              mono_sound.shape[0] / TARGET_SAMPLING_FREQUENCY)
    if (word_bounds[-1][1] - word_bounds[0][0]) <= max_segment_size:
        segment_start = word_bounds[0][0] - indent_for_silence
        segment_end = word_bounds[-1][1] + indent_for_silence
        return strip_segments([(segment_start, segment_end)],
                              mono_sound.shape[0] / TARGET_SAMPLING_FREQUENCY)
    word_groups = select_word_groups(word_bounds, max_segment_size)

    segments = strip_segments(
        [(cur_group[0][0] - indent_for_silence, cur_group[-1][1] + indent_for_silence) for cur_group in word_groups],
        mono_sound.shape[0] / TARGET_SAMPLING_FREQUENCY
    )
    n_segments = len(segments)

    if n_segments > 1:
        for idx in range(1, n_segments):
            if segments[idx - 1][1] > segments[idx][0]:
                overlap = segments[idx - 1][1] - segments[idx][0]
                segments[idx - 1] = (segments[idx - 1][0], segments[idx - 1][1] - overlap / 2.0)
                segments[idx] = (segments[idx][0] + overlap / 2.0, segments[idx][1])

    return join_short_segments_to_long_ones(segments, min_segment_size)


def is_speech(sound: np.ndarray, classifier: Pipeline) -> bool:
    """
    Checks if top class, according to the classifier, contains a word "speech".

    Arguments:
    - sound: 1D waveform with rate 16_000 (equals wav_io.TARGET_SAMPLING_FREQUENCY)
    - classifier: an AudioClassificationPipeline that can classify audios. See
      `initialize_model_for_speech_classification` for details.
    """
    output = classifier(sound)
    if len(output) > 0:
        class_label = output[0]['label']
        contains_speech = ('speech' in set(wordpunct_tokenize(class_label.lower())))
    else:
        contains_speech = False
    return contains_speech


def recognize_sounds(sounds: List[np.ndarray], recognizer: Pipeline) -> List[str]:
    """
    Arguments:
    - mono_sound: a list of 1D waveforms with rate 16_000 (equals wav_io.TARGET_SAMPLING_FREQUENCY)
    - recognizer: an AutomaticSpeechRecognitionPipeline that can return transcriptions. See
      `initialize_model_for_speech_recognition` for details.

    Sequentially applies `recognizer` to each sound, gather a list of the resulting transcriptions
    (a "text" field in `recognizer` output). Further, calls `remove_oscillatory_hallucinations` for
    each transcription, and returns the list.
    """
    for idx, val in enumerate(sounds):
        if not isinstance(val, np.ndarray):
            err_msg = f'The sound {idx} is wrong! Expected {type(np.array([1, 2]))}, got {type(val)}.'
            asr_logger.error(err_msg)
            raise ValueError(err_msg)
        if len(val.shape) != 1:
            err_msg = f'The sound {idx} channel number is wrong! Expected 1, got {len(val)}.'
            asr_logger.error(err_msg)
            raise ValueError(err_msg)

    all_transcriptions = []
    for cur_sound in tqdm(sounds):
        all_transcriptions.append(recognizer(cur_sound)['text'])
        gc.collect()
        torch.cuda.empty_cache()
    return [remove_oscillatory_hallucinations(it) for it in all_transcriptions]


def transcribe(
    mono_sound: np.ndarray,
    segmenter: Pipeline,
    voice_activity_detector: Pipeline,
    asr: Pipeline,
    min_segment_size: float,
    max_segment_size: float
) -> List[Tuple[float, float, str]]:
    """
   1) Processes a sound with `segmenter`
   2) Adds padding around each segment
   3) Merges short segments into longer ones
   4) Applies `voice_activity_detector` to filter out non-speech segments
   5) Applies `asr` to obtain final transcriptions for each segment
   6) Removes oscillatory hallucinations from the final transcriptions
   7) Returns only the segments with non-empty resulting transcriptions

    Arguments:
    - mono_sound: 1D waveform with rate 16_000 (equals wav_io.TARGET_SAMPLING_FREQUENCY),
      no shorter than asr.MIN_SOUND_LENGTH.
    - segmenter: an AutomaticSpeechRecognitionPipeline that can return word timestamps. See
      `initialize_model_for_speech_segmentation` for details.
    - voice_activity_detector: an AudioClassificationPipeline that can classify audios. See
      `initialize_model_for_speech_classification` for details.
    - asr: an AutomaticSpeechRecognitionPipeline that can return transcriptions. See
      `initialize_model_for_speech_recognition` for details.
    - min_segment_size: a parameter for segment processing, see `segment_sound` for details.
    - max_segment_size: a parameter for segment processing, see `segment_sound` for details.

    Output: a list of tuples (start_time, end_time, transcription) for all found utterances,
    can be empty.

    TODO when calling `voice_activity_detector` and `asr`, process all segments at once as
    batch to improve performance, especially for long audios. However, need to calculate
    max batch size to fit into GPU memory, or this help:
    https://discuss.pytorch.org/t/is-it-possible-to-execute-two-modules-in-parallel-in-pytorch/54866
    """
    sound_segments = segment_sound(mono_sound, segmenter, min_segment_size, max_segment_size)
    asr_logger.info(f'The speech sound is divided into {len(sound_segments)} segments.')
    if len(sound_segments) == 0:
        return []
    sound_segments_ = []
    for it in sound_segments:
        segment_start = min(round(it[0] * TARGET_SAMPLING_FREQUENCY), mono_sound.shape[0])
        segment_end = min(round(it[1] * TARGET_SAMPLING_FREQUENCY), mono_sound.shape[0])
        if segment_start >= segment_end:
            err_msg = f'Segments {sound_segments} are wrong!'
            asr_logger.error(err_msg)
            raise RuntimeError(err_msg)
        sound_segments_.append((segment_start, segment_end))
    segment_lengths = sorted([(it[1] - it[0]) for it in sound_segments])
    mean_length = float(np.mean(segment_lengths))
    min_segment_idx = -1
    min_segment_length = None
    for idx, val in enumerate(sound_segments):
        cur_segment_length = val[1] - val[0]
        if min_segment_length is None:
            min_segment_idx = idx
            min_segment_length = cur_segment_length
        elif cur_segment_length < min_segment_length:
            min_segment_idx = idx
            min_segment_length = cur_segment_length
    minimal_segment_description = ('[' + time_to_str(sound_segments[min_segment_idx][0]) + ' - ' +
                                   time_to_str(sound_segments[min_segment_idx][1]) + ']')
    info_msg = (f'Minimal segment duration is {segment_lengths[0]} seconds. '
                f'This segment is {minimal_segment_description}.')
    asr_logger.info(info_msg)
    asr_logger.info(f'Maximal segment duration is {segment_lengths[-1]} seconds.')
    asr_logger.info(f'Median segment duration is {segment_lengths[len(segment_lengths) // 2]} seconds.')
    asr_logger.info(f'Mean segment duration is {mean_length} seconds.')
    del segment_lengths
    segments_with_speech = []
    sounds_with_speech = []
    for idx in range(len(sound_segments)):
        bounds = sound_segments_[idx]
        if is_speech(sound=mono_sound[bounds[0]:bounds[1]], classifier=voice_activity_detector):
            sounds_with_speech.append(mono_sound[bounds[0]:bounds[1]])
            segments_with_speech.append(sound_segments[idx])
    asr_logger.info(f'{len(sounds_with_speech)} of {len(sound_segments)} segments contain a human speech.')
    del sound_segments_, sound_segments
    if len(sounds_with_speech) == 0:
        return []
    recognized_transcriptions = recognize_sounds(
        sounds=sounds_with_speech,
        recognizer=asr
    )
    del sounds_with_speech
    results = list(filter(
        lambda it2: len(it2[2]) > 0,
        map(
            lambda it: (it[0][0], it[0][1], it[1].strip()),
            zip(segments_with_speech, recognized_transcriptions)
        )
    ))
    return results
