import os
from typing import List, Tuple

from nltk import wordpunct_tokenize, sent_tokenize
from rusenttokenize import ru_sent_tokenize
import torch


def initialize_normalizer():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'normalization',
                              'snakers4_silero-models')
    if not os.path.isdir(model_path):
        raise ValueError(f'The directory "{model_path}" does not exist!')
    _, _, _, _, apply_te = torch.hub.load(repo_or_dir=model_path, model='silero_te', source='local')
    return apply_te


def check_language(lang: str) -> str:
    lang_ = 'ru'
    if lang.lower().strip() in {'en', 'eng', 'english'}:
        lang_ = 'en'
    elif lang.lower().strip() not in {'ru', 'rus', 'russian'}:
        err_msg = f'The language "{lang}" is not supported!'
        raise ValueError(err_msg)
    return lang_


def normalize_text(s: str, normalizer, lang: str = 'ru') -> str:
    return normalizer(s, lan=check_language(lang))


def tokenize_text(s: str, lang: str = 'ru') -> List[str]:
    if check_language(lang) == 'ru':
        res = ru_sent_tokenize(s)
    else:
        res = sent_tokenize(s)
    return res


def calculate_sentence_bounds(asr_result: List[Tuple[str, float, float]],
                              sentences: List[str]) -> List[Tuple[str, float, float]]:
    source_text = ''
    source_indices = []
    for idx, cur in enumerate(asr_result):
        source_text += (' ' + cur[0].replace('ё', 'е'))
        source_indices += [idx for _ in range(len(cur[0]) + 1)]
    start_pos = 0
    sentences_with_bounds = []
    for cur_sent in sentences:
        simplified_sent = ' '.join(
            list(filter(lambda it: it.isalnum(), wordpunct_tokenize(cur_sent)))
        ).lower().replace('ё', 'е')
        found_idx = source_text[start_pos:].find(simplified_sent)
        if found_idx < 0:
            err_msg = ''
            raise ValueError(err_msg)
        found_idx += start_pos
        word_indices = set()
        for char_idx in range(found_idx, found_idx + len(simplified_sent)):
            word_indices.add(source_indices[char_idx])
        word_indices = sorted(list(word_indices))
        sent_start = asr_result[word_indices[0]][1]
        sent_end = asr_result[word_indices[-1]][2]
        sentences_with_bounds.append(
            (
                cur_sent,
                sent_start,
                sent_end
            )
        )
        start_pos = found_idx + len(simplified_sent)
        del word_indices
    return sentences_with_bounds
