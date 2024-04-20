import logging
import os
import tempfile
from typing import List

from flask import Flask, request, jsonify
import numpy as np

from wav_io.wav_io import transform_to_wavpcm, load_sound
from wav_io.wav_io import TARGET_SAMPLING_FREQUENCY
from asr.asr import initialize_model_for_speech_recognition
from asr.asr import initialize_model_for_speech_segmentation
from asr.asr import transcribe as transcribe_speech
from utils.utils import time_to_str


speech_to_srt_logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
app = Flask(__name__)

FRAME_SIZE = 20
LANGUAGE_NAME = 'ru'
old_sounds: List[np.ndarray] = []

try:
    segmenter = initialize_model_for_speech_segmentation(LANGUAGE_NAME)
except BaseException as ex:
    err_msg = str(ex)
    speech_to_srt_logger.error(err_msg)
    raise
speech_to_srt_logger.info('The Wav2Vec2-based segmenter is loaded.')

try:
    asr = initialize_model_for_speech_recognition(LANGUAGE_NAME)
except BaseException as ex:
    err_msg = str(ex)
    speech_to_srt_logger.error(err_msg)
    raise
speech_to_srt_logger.info('The Whisper-based ASR is initialized.')


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
    speech_to_srt_logger.info(f'The total duration of the new sound "{file.filename}" is '
                              f'{time_to_str(input_sound.shape[0] / TARGET_SAMPLING_FREQUENCY)}.')

    if len(old_sounds) > 0:
        input_sound = np.concatenate(old_sounds + [input_sound])
        speech_to_srt_logger.info(f'The total duration of the united sound is '
                                  f'{time_to_str(input_sound.shape[0] / TARGET_SAMPLING_FREQUENCY)}.')

    output_text = ''
    if input_sound is None:
        speech_to_srt_logger.info(f'The sound "{file.filename}" is empty.')
    else:
        texts_with_timestamps = transcribe_speech(input_sound, segmenter, asr, FRAME_SIZE)
        for _, _, sentence_text in texts_with_timestamps[:-1]:
            output_text += (' ' + sentence_text)
        output_text = ' '.join(output_text.split())
        old_sounds.clear()
        if len(texts_with_timestamps) > 0:
            last_segment_start = round(texts_with_timestamps[-1][0] * TARGET_SAMPLING_FREQUENCY)
            old_sounds.append(input_sound[-last_segment_start:])

    resp = jsonify(output_text)
    resp.status_code = 200
    return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8040)
