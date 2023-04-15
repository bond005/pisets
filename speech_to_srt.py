from argparse import ArgumentParser
import codecs
import logging
import os
import sys
import tempfile

import numpy as np
from wav_io.wav_io import transform_to_wavpcm, load_sound
from wav_io.wav_io import TARGET_SAMPLING_FREQUENCY
from vad.vad import initialize_vad_ensemble, split_long_sound
from asr.asr import recognize, initialize_model
from rescoring.rescoring import initialize_rescorer, rescore
from normalization.normalization import check_language, initialize_normalizer
from normalization.normalization import tokenize_text, normalize_text, calculate_sentence_bounds
from utils.utils import time_to_str


speech_to_srt_logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument('--lang', dest='language', type=str, required=False, default='ru',
                        help='The language of input speech (Russian or English).')
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input sound file name or YouTube URL.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output SubRip file name.')
    parser.add_argument('-r', '--rescorer', dest='rescorer', action='store_true',
                        help='The necessity to use the T5 transformer as a rescorer.')
    parser.add_argument('-f', '--frame', dest='sound_frame', type=int, default=50, required=False,
                        help='The maximum size of the sound frame (in seconds).')
    args = parser.parse_args()

    tokenizer_for_rescorer = None
    model_of_rescorer = None

    frame_size = args.sound_frame
    if (frame_size <= 10) or (frame_size >= 70):
        err_msg = f'The maximum size of the sound frame has a wrong value! ' \
                  f'Expected an integer greater than 10 and less than 70, got {frame_size}.'
        speech_to_srt_logger.error(err_msg)
        raise IOError(err_msg)

    audio_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(audio_fname):
        err_msg = f'The file "{audio_fname}" does not exist!'
        speech_to_srt_logger.error(err_msg)
        raise IOError(err_msg)

    try:
        language_name = check_language(args.language)
    except BaseException as ex:
        err_msg = str(ex)
        speech_to_srt_logger.error(err_msg)
        raise

    output_srt_fname = os.path.normpath(args.output_name)
    output_srt_dir = os.path.dirname(output_srt_fname)
    if len(output_srt_dir) > 0:
        if not os.path.isdir(output_srt_dir):
            err_msg = f'The directory "{output_srt_dir}" does not exist!'
            speech_to_srt_logger.error(err_msg)
            raise IOError(err_msg)
    if len(os.path.basename(output_srt_fname).strip()) == 0:
        err_msg = f'The file name "{output_srt_fname}" is incorrect!'
        speech_to_srt_logger.error(err_msg)
        raise IOError(err_msg)

    tmp_wav_name = ''
    try:
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.wav') as fp:
            tmp_wav_name = fp.name
        try:
            transform_to_wavpcm(audio_fname, tmp_wav_name)
        except BaseException as ex:
            err_msg = str(ex)
            speech_to_srt_logger.error(err_msg)
            raise
        speech_to_srt_logger.info(f'The sound "{audio_fname}" is converted to the "{tmp_wav_name}".')
        try:
            input_sound = load_sound(tmp_wav_name)
        except BaseException as ex:
            err_msg = str(ex)
            speech_to_srt_logger.error(err_msg)
            raise
        speech_to_srt_logger.info(f'The sound is "{tmp_wav_name}" is loaded.')
    finally:
        if os.path.isfile(tmp_wav_name):
            os.remove(tmp_wav_name)
            speech_to_srt_logger.info(f'The sound is "{tmp_wav_name}" is removed.')

    if input_sound is None:
        speech_to_srt_logger.info(f'The sound "{audio_fname}" is empty.')
        sentences_with_bounds = []
    else:
        if not isinstance(input_sound, np.ndarray):
            speech_to_srt_logger.info(f'The sound "{audio_fname}" is stereo.')
            input_sound = (input_sound[0] + input_sound[1]) / 2.0
        speech_to_srt_logger.info(f'The total duration of the sound "{audio_fname}" is '
                                  f'{time_to_str(input_sound.shape[0] / TARGET_SAMPLING_FREQUENCY)}.')

        try:
            processor, model = initialize_model(language_name)
        except BaseException as ex:
            err_msg = str(ex)
            speech_to_srt_logger.error(err_msg)
            raise
        speech_to_srt_logger.info('The model and processor are loaded.')

        if args.rescorer:
            try:
                tokenizer_for_rescorer, model_of_rescorer = initialize_rescorer(language_name)
            except BaseException as ex:
                err_msg = str(ex)
                speech_to_srt_logger.error(err_msg)
                raise
            speech_to_srt_logger.info('The rescorer is loaded.')

        try:
            vad = initialize_vad_ensemble()
        except BaseException as ex:
            err_msg = str(ex)
            speech_to_srt_logger.error(err_msg)
            raise
        speech_to_srt_logger.info('The VAD is initialized.')

        try:
            text_normalizer = initialize_normalizer()
        except BaseException as ex:
            err_msg = str(ex)
            speech_to_srt_logger.error(err_msg)
            raise
        speech_to_srt_logger.info('The text normalizer is initialized.')

        try:
            sound_frames, frame_bounds = split_long_sound(input_sound, vad, max_sound_len=frame_size * 16_000)
        except BaseException as ex:
            err_msg = str(ex)
            speech_to_srt_logger.error(err_msg)
            raise
        speech_to_srt_logger.info(f'The sound "{audio_fname}" is divided into {len(sound_frames)} shorter frames.')
        try:
            words = recognize(sound_frames[0], processor, model)
        except BaseException as ex:
            err_msg = str(ex)
            speech_to_srt_logger.error(err_msg)
            raise
        speech_to_srt_logger.info(f'The sound frame 1 is recognized '
                                  f'(the frame duration is {sound_frames[0].shape[0] / TARGET_SAMPLING_FREQUENCY}).')
        if (tokenizer_for_rescorer is not None) and (model_of_rescorer is not None):
            try:
                words = rescore(words, tokenizer_for_rescorer, model_of_rescorer)
            except BaseException as ex:
                err_msg = str(ex)
                speech_to_srt_logger.error(err_msg)
                raise
            speech_to_srt_logger.info('The sound frame 1 is rescored.')
        try:
            sentences = tokenize_text(
                s=normalize_text(
                    s=' '.join([cur[0] for cur in words]),
                    normalizer=text_normalizer,
                    lang=language_name
                ),
                lang=language_name
            )
        except BaseException as ex:
            err_msg = str(ex)
            speech_to_srt_logger.error(err_msg)
            raise
        try:
            sentences_with_bounds = calculate_sentence_bounds(
                asr_result=words,
                sentences=sentences
            )
        except BaseException as ex:
            err_msg = str(ex)
            speech_to_srt_logger.error(err_msg)
            raise
        del words, sentences
        speech_to_srt_logger.info('The sound frame 1 is normalized and tokenized.')

        for counter, (cur_frame, frame_bounds) in enumerate(zip(sound_frames[1:], frame_bounds[1:])):
            try:
                words_ = recognize(cur_frame, processor, model)
            except BaseException as ex:
                err_msg = str(ex)
                speech_to_srt_logger.error(err_msg)
                raise
            speech_to_srt_logger.info(f'The sound frame {counter + 2} is recognized '
                                      f'(the frame duration is {cur_frame.shape[0] / TARGET_SAMPLING_FREQUENCY}).')
            if (tokenizer_for_rescorer is not None) and (model_of_rescorer is not None):
                try:
                    words_ = rescore(words_, tokenizer_for_rescorer, model_of_rescorer)
                except BaseException as ex:
                    err_msg = str(ex)
                    speech_to_srt_logger.error(err_msg)
                    raise
                speech_to_srt_logger.info(f'The sound frame {counter + 2} is rescored.')
            frame_start = frame_bounds[0] / TARGET_SAMPLING_FREQUENCY
            words = []
            for word_text, word_start, word_end in words_:
                words.append((
                    word_text,
                    word_start + frame_start,
                    word_end + frame_start
                ))
            del words_
            try:
                sentences = tokenize_text(
                    s=normalize_text(
                        s=' '.join([cur[0] for cur in words]),
                        normalizer=text_normalizer,
                        lang=language_name
                    ),
                    lang=language_name
                )
            except BaseException as ex:
                err_msg = str(ex)
                speech_to_srt_logger.error(err_msg)
                raise
            try:
                sentences_with_bounds += calculate_sentence_bounds(
                    asr_result=words,
                    sentences=sentences
                )
            except BaseException as ex:
                err_msg = str(ex)
                speech_to_srt_logger.error(err_msg)
                raise
            del words, sentences
            speech_to_srt_logger.info(f'The sound frame {counter + 2} is normalized and tokenized.')

    with codecs.open(output_srt_fname, mode='w', encoding='utf-8') as fp:
        for counter, (sentence_text, sent_start, sent_end) in enumerate(sentences_with_bounds):
            fp.write(f'{counter + 1}\n')
            fp.write(f'{time_to_str(sent_start)} --> {time_to_str(sent_end)}\n')
            fp.write(f'{sentence_text}\n\n')


if __name__ == '__main__':
    speech_to_srt_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    speech_to_srt_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('speech_to_srt.log')
    file_handler.setFormatter(formatter)
    speech_to_srt_logger.addHandler(file_handler)
    main()
