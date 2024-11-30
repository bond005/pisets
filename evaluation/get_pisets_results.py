import os
from pathlib import Path
import json
import dataclasses

import librosa
import pysrt
import numpy as np
import pandas as pd
from datasets import load_dataset, Audio

from IPython.display import clear_output
from wav_io.wav_io import load_sound
from asr.asr import *
from asr.comparison import *

segmenter_no_lm = initialize_model_for_speech_segmentation('ru', 'bond005/wav2vec2-large-ru-golos')
segmenter_lm = initialize_model_for_speech_segmentation('ru', 'bond005/wav2vec2-large-ru-golos-with-lm')
vad = initialize_model_for_speech_classification()
asr_whisper_large_v2 = initialize_model_for_speech_recognition('ru', 'openai/whisper-large-v2')
asr_whisper_large_v3 = initialize_model_for_speech_recognition('ru', 'openai/whisper-large-v3')

max_len = None

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

input_dir = Path('/home/oleg/pisets_test_set/')
output_dir = Path('/home/oleg/pisets_test_results/')

for audio_path in input_dir.glob('*.wav'):
    # if (srt_path := audio_path.with_suffix('.srt')).is_file():
    #     truth = ' '.join([sub.text for sub in pysrt.open(srt_path)])
    # else:
    #     with open(audio_path.with_suffix('.txt')) as f:
    #         truth = f.read()

    name = audio_path.stem

    waveform, _ = librosa.load(audio_path, sr=16_000)

    for mode_name, args, kwargs in (
        ('nolm_whisperV3_1_20', (segmenter_no_lm, vad, asr_whisper_large_v3), dict(min_segment_size=1, max_segment_size=20)),
        ('lm_whisperV2_1_20', (segmenter_lm, vad, asr_whisper_large_v2), dict(min_segment_size=1, max_segment_size=20)),
        ('lm_whisperV3_15_25_stretch', (segmenter_lm, vad, asr_whisper_large_v3), dict(min_segment_size=15, max_segment_size=25, stretch=(3, 4))),
        ('lm_whisperV3_1_20_stretch', (segmenter_lm, vad, asr_whisper_large_v3), dict(min_segment_size=1, max_segment_size=20, stretch=(3, 4))),
        ('lm_whisperV3_1_30_stretch', (segmenter_lm, vad, asr_whisper_large_v3), dict(min_segment_size=1, max_segment_size=30, stretch=(3, 4))),
    ):
        print(name, mode_name)
        output = transcribe(waveform[:max_len], *args, **kwargs)
        with open(output_dir / f'{name}_{mode_name}.json', 'w') as f:
            json.dump(output, f, cls=EnhancedJSONEncoder)
        # print(' '.join(x.transcription for x in output))