import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

try:
    from vad.vad import MIN_SOUND_LENGTH
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from vad.vad import MIN_SOUND_LENGTH


def initialize_model() -> Tuple[Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC]:
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'asr')
    if not os.path.isdir(model_path):
        raise ValueError(f'The directory "{model_path}" does not exist!')
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    return processor, model


def recognize(mono_sound: np.ndarray, processor: Wav2Vec2ProcessorWithLM,
              model: Wav2Vec2ForCTC) -> List[Tuple[str, float, float]]:
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
    inputs = processor(mono_sound, sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits_ = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    logits = logits_.numpy()
    del logits_, inputs
    if len(logits.shape) not in {2, 3}:
        err_msg = f'Logits matrix has incorrect shape! ' \
                  f'Expected 2-D or 3-D array, got {len(logits.shape)}-D one.'
        raise ValueError(err_msg)
    if len(logits.shape) != 2:
        if logits.shape[0] != 1:
            err_msg = f'The first dimenstion of the logits matrix is incorrect! ' \
                      f'Expected 1, got {logits.shape[0]}.'
            raise ValueError(err_msg)
        logits = logits[0]
    res = processor.decode(logits=logits, lm_score_boundary=False, output_word_offsets=True)
    time_offset = model.config.inputs_to_logits_ratio / processor.current_processor.sampling_rate
    word_offsets = [
        (
            d["word"],
            round(d["start_offset"] * time_offset, 3),
            round(d["end_offset"] * time_offset, 3),
        )
        for d in res.word_offsets
    ]
    return word_offsets
