import os
import re
import sys
from typing import Dict, List, Tuple, Union

import jiwer
import jiwer.transforms as tr
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

try:
    from vad.vad import MIN_SOUND_LENGTH
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from vad.vad import MIN_SOUND_LENGTH


BEAM_WIDTH = 20


class ReplaceYo(tr.AbstractTransform):
    def process_string(self, s: str):
        return s.replace('ё', 'е')

    def process_list(self, inp: List[str]):
        outp = []
        for sentence in inp:
            outp.append(sentence.replace('ё', 'е'))
        return outp


def initialize_model(language: str = 'ru') -> Tuple[Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC]:
    if language == 'ru':
        model_name = 'bond005/wav2vec2-large-ru-golos-with-lm'
    else:
        model_name = 'patrickvonplaten/wav2vec2-large-960h-lv60-self-4-gram'
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')
    return processor, model


def recognize(mono_sound: np.ndarray, processor: Wav2Vec2ProcessorWithLM,
              model: Wav2Vec2ForCTC, alpha: float = None, beta: float = None,
              hotword_weight: float = None, hotwords: List[str] = None,
              verbose: bool=False) -> List[Tuple[str, float, float]]:
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
    if hotword_weight is None:
        if hotwords is not None:
            err_msg = 'Hotwords are specified, but the hotword weight is not specified!'
            raise ValueError(err_msg)
    else:
        if hotwords is None:
            err_msg = 'The hotword weight is specified, but hotwords are not specified!'
            raise ValueError(err_msg)
    if alpha is None:
        if beta is not None:
            err_msg = 'The beta is specified, but the alpha is not specified!'
            raise ValueError(err_msg)
    else:
        if beta is None:
            err_msg = 'The alpha is specified, but the beta is not specified!'
            raise ValueError(err_msg)

    cuda_is_used = torch.cuda.is_available()
    inputs = processor(mono_sound, sampling_rate=16_000, return_tensors="pt", padding=True)
    if cuda_is_used:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        logits_ = model(**inputs).logits
    if cuda_is_used:
        logits = logits_.to('cpu').numpy()
    else:
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
    if (alpha is None) and (hotword_weight is None):
        res = processor.decode(logits=logits, lm_score_boundary=True, output_word_offsets=True, beam_width=BEAM_WIDTH)
        if verbose:
            print('Logits are decoded without any hotword.')
    elif (alpha is None) and (hotword_weight is not None):
        res = processor.decode(logits=logits, lm_score_boundary=True, output_word_offsets=True, beam_width=BEAM_WIDTH,
                               hotwords=hotwords, hotword_weight=hotword_weight)
        if verbose:
            print(f'Logits are decoded with hotwords (the hotword weight is {hotword_weight}).')
    elif (alpha is not None) and (hotword_weight is None):
        res = processor.decode(logits=logits, lm_score_boundary=True, output_word_offsets=True, beam_width=BEAM_WIDTH,
                               alpha=alpha, beta=beta)
        if verbose:
            print('Logits are decoded without any hotword.')
    else:
        res = processor.decode(logits=logits, lm_score_boundary=True, output_word_offsets=True, beam_width=BEAM_WIDTH,
                               alpha=alpha, beta=beta, hotwords=hotwords, hotword_weight=hotword_weight)
        if verbose:
            print(f'Logits are decoded with hotwords (the hotword weight is {hotword_weight}).')
    time_offset = model.config.inputs_to_logits_ratio / processor.current_processor.sampling_rate
    word_offsets = [
        (
            d["word"].lower(),
            round(d["start_offset"] * time_offset, 3),
            round(d["end_offset"] * time_offset, 3),
        )
        for d in res.word_offsets
    ]
    return word_offsets


def check_annotations_and_hotwords(list_of_texts: List[str], additional: str = None) -> None:
    re_for_digits = re.compile(r'\d+')
    for idx, text in enumerate(list_of_texts):
        search_res = re_for_digits.search(text)
        if search_res is not None:
            text_number = f'{idx + 1}'
            if len(text_number) > 0:
                if (len(text_number) > 1) and (text_number[-2] == '1'):
                    text_number += 'th'
                else:
                    if text_number[-1] == '1':
                        text_number += 'st'
                    elif text_number[-1] == '2':
                        text_number += 'nd'
                    elif text_number[-1] == '3':
                        text_number += 'rd'
                    else:
                        text_number += 'th'
            if additional is None:
                text_number += ' text'
            elif len(additional.strip()) == 0:
                text_number += ' text'
            else:
                text_number += f' {additional.strip()}'
            err_msg = f'The {text_number} "{text}" is wrong! The text should not include any digit! ' \
                      f'All numerals have to be written in letters: for example, "42" should look like ' \
                      f'"forty two" in English or "сорок два" in Russian.'
            raise ValueError(err_msg)


def build_transforms_for_wer() -> jiwer.Compose:
    transformation_for_wer = jiwer.Compose([
        tr.ToLowerCase(),
        tr.RemovePunctuation(),
        tr.RemoveWhiteSpace(replace_by_space=True),
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
        ReplaceYo(),
        tr.ReduceToListOfListOfWords(word_delimiter=' ')
    ])
    return transformation_for_wer


def decode_for_evaluation(processor: Wav2Vec2ProcessorWithLM, evaluation_logits: List[np.ndarray],
                          alpha: float, beta: float,
                          hotword_weight: float = None, hotwords: List[str] = None) -> List[str]:
    res = []
    if hotword_weight is None:
        for cur in evaluation_logits:
            predicted = processor.decode(
                logits=cur, lm_score_boundary=True, output_word_offsets=False, beam_width=BEAM_WIDTH,
                alpha=alpha, beta=beta
            ).text
            if isinstance(predicted, list):
                res += [cur.lower() for cur in predicted]
            else:
                res.append(predicted.lower())
    else:
        for cur in evaluation_logits:
            predicted = processor.decode(
                logits=cur, lm_score_boundary=True, output_word_offsets=False, beam_width=BEAM_WIDTH,
                alpha=alpha, beta=beta, hotwords=hotwords, hotword_weight=hotword_weight
            ).text
            if isinstance(predicted, list):
                res += [cur.lower() for cur in predicted]
            else:
                res.append(predicted.lower())
    return res


def find_best_inference_hyperparams(processor: Wav2Vec2ProcessorWithLM,
                                    evaluation_logits: List[np.ndarray], evaluation_annotations: List[str],
                                    hotwords: List[str]=None) -> Dict[str, float]:
    check_annotations_and_hotwords(evaluation_annotations, 'annotation')
    if hotwords is not None:
        check_annotations_and_hotwords(evaluation_annotations, 'hotword')

    space = [Real(1e-5, 20.0, "log-uniform", name='alpha'),
             Real(1e-5, 20.0, "log-uniform", name='beta')]
    list_of_transformations = build_transforms_for_wer()
    if hotwords is not None:
        space.append(Real(1.0, 100.0, "log-uniform", name='hotword_weight'))

        @use_named_args(space)
        def objective_f(alpha: float, beta: float, hotword_weight: float) -> float:
            predicted = decode_for_evaluation(processor, evaluation_logits, alpha, beta, hotword_weight, hotwords)
            return jiwer.wer(
                truth=evaluation_annotations,
                hypothesis=predicted,
                truth_transform=list_of_transformations,
                hypothesis_transform=list_of_transformations
            )
    else:
        @use_named_args(space)
        def objective_f(alpha: float, beta: float) -> float:
            predicted = decode_for_evaluation(processor, evaluation_logits, alpha, beta)
            return jiwer.wer(
                truth=evaluation_annotations,
                hypothesis=predicted,
                truth_transform=list_of_transformations,
                hypothesis_transform=list_of_transformations
            )

    res_gp = gp_minimize(
        objective_f, space,
        n_calls=256, n_random_starts=32,
        n_restarts_optimizer=8, random_state=42,
        verbose=True, n_jobs=1
    )
    best_parameters = {
        'alpha': float(res_gp.x[0]),
        'beta': float(res_gp.x[1]),
    }
    if hotwords is not None:
        best_parameters['hotword_weight'] = float(res_gp.x[2])
    return best_parameters
