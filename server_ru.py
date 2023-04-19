import logging
import os
import tempfile

from flask import Flask, request, jsonify
import numpy as np

from wav_io.wav_io import transform_to_wavpcm, load_sound
from wav_io.wav_io import TARGET_SAMPLING_FREQUENCY
from vad.vad import initialize_vad_ensemble, split_long_sound
from asr.asr import recognize, initialize_model
from rescoring.rescoring import initialize_rescorer, rescore
from normalization.normalization import initialize_normalizer
from normalization.normalization import tokenize_text, normalize_text, calculate_sentence_bounds
from utils.utils import time_to_str


speech_to_srt_logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
app = Flask(__name__)

FRAME_SIZE = 50
LANGUAGE_NAME = 'ru'

try:
    processor, model = initialize_model(LANGUAGE_NAME)
except BaseException as ex:
    err_msg = str(ex)
    speech_to_srt_logger.error(err_msg)
    raise
speech_to_srt_logger.info('The model and processor are loaded.')

try:
    tokenizer_for_rescorer, model_of_rescorer = initialize_rescorer(LANGUAGE_NAME)
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


@app.route('/ready')
def ready():
    return 'OK'


@app.route('/transcribe', methods=['POST'])
def transcribe():
    speech_to_srt_logger.info('~~~Recognition process started~~~')
    if 'audio' not in request.files:
        speech_to_srt_logger.error('400: No audiofile part in the request')
        resp = jsonify({'message': 'No audiofile part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['audio']
    if file.filename == '':
        speech_to_srt_logger.error('400: No audio file provided for upload')
        resp = jsonify({'message': 'No audio file provided for upload'})
        resp.status_code = 400
        return resp

    point_pos = file.filename.rfind('.')
    if point_pos > 0:
        src_file_ext = file.filename[(point_pos + 1):]
    else:
        src_file_ext = ''
    if len(src_file_ext) == 0:
        speech_to_srt_logger.error('400: Unknown type of the file provided for upload')
        resp = jsonify({'message': 'Unknown type of the file provided for upload'})
        resp.status_code = 400
        return resp
    tmp_audio_name = ''
    tmp_wav_name = ''
    err_msg = ''
    input_sound = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.' + src_file_ext) as fp:
            tmp_audio_name = fp.name
        file.save(tmp_audio_name)
        speech_to_srt_logger.info(f'The sound "{file.filename}" is saved to the "{tmp_audio_name}".')
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.wav') as fp:
            tmp_wav_name = fp.name
        try:
            transform_to_wavpcm(tmp_audio_name, tmp_wav_name)
        except BaseException as ex:
            err_msg = str(ex)
            speech_to_srt_logger.error(err_msg)
        if len(err_msg) == 0:
            speech_to_srt_logger.info(f'The sound "{file.filename}" is converted to the "{tmp_wav_name}".')
            try:
                input_sound = load_sound(tmp_wav_name)
            except BaseException as ex:
                err_msg = str(ex)
                speech_to_srt_logger.error(err_msg)
            if len(err_msg) == 0:
                speech_to_srt_logger.info(f'The sound is "{tmp_wav_name}" is loaded.')
    finally:
        if os.path.isfile(tmp_audio_name):
            os.remove(tmp_audio_name)
            speech_to_srt_logger.info(f'The sound is "{tmp_audio_name}" is removed.')
        if os.path.isfile(tmp_wav_name):
            os.remove(tmp_wav_name)
            speech_to_srt_logger.info(f'The sound is "{tmp_wav_name}" is removed.')
    if (len(err_msg) > 0) or (input_sound is None):
        if len(err_msg) == 0:
            err_msg = 'Audio file is empty.'
            speech_to_srt_logger.error('400: ' + err_msg)
            resp = jsonify({'message': err_msg})
        else:
            resp = jsonify({'message': err_msg})
        resp.status_code = 400
        return resp

    if not isinstance(input_sound, np.ndarray):
        speech_to_srt_logger.info(f'The sound "{file.filename}" is stereo.')
        input_sound = (input_sound[0] + input_sound[1]) / 2.0
    speech_to_srt_logger.info(f'The total duration of the sound "{file.filename}" is '
                              f'{time_to_str(input_sound.shape[0] / TARGET_SAMPLING_FREQUENCY)}.')

    try:
        sound_frames, frame_bounds = split_long_sound(input_sound, vad, max_sound_len=FRAME_SIZE * 16_000)
    except BaseException as ex:
        err_msg = str(ex)
        speech_to_srt_logger.error('400: ' + err_msg)
        resp = jsonify({'message': err_msg})
        resp.status_code = 400
        return resp
    speech_to_srt_logger.info(f'The sound "{file.filename}" is divided into {len(sound_frames)} shorter frames.')
    try:
        words = recognize(sound_frames[0], processor, model)
    except BaseException as ex:
        err_msg = str(ex)
        speech_to_srt_logger.error('400: ' + err_msg)
        resp = jsonify({'message': err_msg})
        resp.status_code = 400
        return resp
    speech_to_srt_logger.info(f'The sound frame 1 is recognized '
                              f'(the frame duration is {sound_frames[0].shape[0] / TARGET_SAMPLING_FREQUENCY}).')
    if (tokenizer_for_rescorer is not None) and (model_of_rescorer is not None):
        try:
            words = rescore(words, tokenizer_for_rescorer, model_of_rescorer)
        except BaseException as ex:
            err_msg = str(ex)
            speech_to_srt_logger.error('400: ' + err_msg)
            resp = jsonify({'message': err_msg})
            resp.status_code = 400
            return resp
        speech_to_srt_logger.info('The sound frame 1 is rescored.')
    try:
        sentences = tokenize_text(
            s=normalize_text(
                s=' '.join([cur[0] for cur in words]),
                normalizer=text_normalizer,
                lang=LANGUAGE_NAME
            ),
            lang=LANGUAGE_NAME
        )
    except BaseException as ex:
        err_msg = str(ex)
        speech_to_srt_logger.error('400: ' + err_msg)
        resp = jsonify({'message': err_msg})
        resp.status_code = 400
        return resp
    try:
        sentences_with_bounds = calculate_sentence_bounds(
            asr_result=words,
            sentences=sentences
        )
    except BaseException as ex:
        err_msg = str(ex)
        speech_to_srt_logger.error('400: ' + err_msg)
        resp = jsonify({'message': err_msg})
        resp.status_code = 400
        return resp
    del words, sentences
    speech_to_srt_logger.info('The sound frame 1 is normalized and tokenized.')

    for counter, (cur_frame, frame_bounds) in enumerate(zip(sound_frames[1:], frame_bounds[1:])):
        try:
            words_ = recognize(cur_frame, processor, model)
        except BaseException as ex:
            err_msg = str(ex)
            speech_to_srt_logger.error('400: ' + err_msg)
            resp = jsonify({'message': err_msg})
            resp.status_code = 400
            return resp
        speech_to_srt_logger.info(f'The sound frame {counter + 2} is recognized '
                                  f'(the frame duration is {cur_frame.shape[0] / TARGET_SAMPLING_FREQUENCY}).')
        if (tokenizer_for_rescorer is not None) and (model_of_rescorer is not None):
            try:
                words_ = rescore(words_, tokenizer_for_rescorer, model_of_rescorer)
            except BaseException as ex:
                err_msg = str(ex)
                speech_to_srt_logger.error('400: ' + err_msg)
                resp = jsonify({'message': err_msg})
                resp.status_code = 400
                return resp
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
                    lang=LANGUAGE_NAME
                ),
                lang=LANGUAGE_NAME
            )
        except BaseException as ex:
            err_msg = str(ex)
            speech_to_srt_logger.error('400: ' + err_msg)
            resp = jsonify({'message': err_msg})
            resp.status_code = 400
            return resp
        try:
            sentences_with_bounds += calculate_sentence_bounds(
                asr_result=words,
                sentences=sentences
            )
        except BaseException as ex:
            err_msg = str(ex)
            speech_to_srt_logger.error('400: ' + err_msg)
            resp = jsonify({'message': err_msg})
            resp.status_code = 400
            return resp
        del words, sentences
        speech_to_srt_logger.info(f'The sound frame {counter + 2} is normalized and tokenized.')

    output_text = ''
    for counter, (sentence_text, sent_start, sent_end) in enumerate(sentences_with_bounds):
        output_text += f'{counter + 1}\n'
        output_text += f'{time_to_str(sent_start)} --> {time_to_str(sent_end)}\n'
        output_text += f'{sentence_text}\n\n'
    resp = jsonify(output_text)
    resp.status_code = 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8040)
