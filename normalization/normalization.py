import os
from typing import Any, List, Tuple

from nltk import wordpunct_tokenize, sent_tokenize
import numpy as np
from rusenttokenize import ru_sent_tokenize
import torch


def initialize_normalizer() -> Any:
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


def normalize_text(s: str, normalizer: Any, lang: str = 'ru') -> str:
    return normalizer(s, lan=check_language(lang))


def tokenize_text(s: str, lang: str = 'ru') -> List[str]:
    if check_language(lang) == 'ru':
        res = ru_sent_tokenize(s)
    else:
        res = sent_tokenize(s)
    return res


def dtw(left_sequence: str, right_sequence: str) -> Tuple[List[int], List[int]]:
    M = len(left_sequence)
    N = len(right_sequence)
    if (N == 0) and (M == 0):
        raise ValueError('Both sequences are empty!')
    elif M == 0:
        raise ValueError('Left sequence is empty!')
    elif N == 0:
        raise ValueError('Right sequence is empty!')
    D = np.zeros((M + 1, N + 1), dtype=np.int32)
    for i in range(1, M + 1):
        D[i, 0] = i
    for j in range(1, N + 1):
        D[0, j] = j
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            diff = 0 if (left_sequence[i - 1] == right_sequence[j - 1]) else 1
            D[i, j] = min(D[i, j - 1] + 1, D[i - 1, j] + 1, D[i - 1, j - 1] + diff)
    left_rule = [M - 1]
    right_rule = [N - 1]
    i = M
    j = N
    while (i > 1) and (j > 1):
        if D[i - 1, j - 1] <= D[i - 1, j]:
            if D[i - 1, j - 1] <= D[i, j - 1]:
                left_rule.insert(0, i - 2)
                right_rule.insert(0, j - 2)
                i -= 1
                j -= 1
            else:
                left_rule.insert(0, i - 1)
                right_rule.insert(0, j - 2)
                j -= 1
        else:
            if D[i - 1, j] < D[i, j - 1]:
                left_rule.insert(0, i - 2)
                right_rule.insert(0, j - 1)
                i -= 1
            else:
                left_rule.insert(0, i - 1)
                right_rule.insert(0, j - 2)
                j -= 1
    if (i > 1) and (j == 1):
        left_rule.insert(0, i - 2)
        right_rule.insert(0, j - 1)
        i -= 1
        while i > 1:
            left_rule.insert(0, i - 2)
            right_rule.insert(0, j - 1)
            i -= 1
    elif (i == 1) and (j > 1):
        left_rule.insert(0, i - 1)
        right_rule.insert(0, j - 2)
        j -= 1
        while j > 1:
            left_rule.insert(0, i - 1)
            right_rule.insert(0, j - 2)
            j -= 1
    return left_rule, right_rule


def calculate_sentence_bounds(asr_result: List[Tuple[str, float, float]],
                              sentences: List[str]) -> List[Tuple[str, float, float]]:
    if len(asr_result) == 0:
        return []
    if len(' '.join([cur[0] for cur in asr_result]).strip()) == 0:
        return []

    cur_word = asr_result[0][0].replace('ё', 'е')
    text_form_words = cur_word
    indices_of_words = [0 for _ in range(len(cur_word))]
    for idx, cur in enumerate(asr_result[1:]):
        cur_word = ' ' + cur[0].replace('ё', 'е')
        text_form_words += cur_word
        indices_of_words += [idx + 1 for _ in range(len(cur_word))]
    indices_of_words = np.array(indices_of_words, dtype=np.int32)

    simplified_sent = ' '.join(
        list(filter(lambda it: it.isalnum(), wordpunct_tokenize(sentences[0])))
    ).lower().replace('ё', 'е')
    text_from_sentences = simplified_sent
    indices_of_sentences = [0 for _ in range(len(simplified_sent))]
    for idx, cur in enumerate(sentences[1:]):
        simplified_sent = ' '.join(
            list(filter(lambda it: it.isalnum(), wordpunct_tokenize(cur)))
        ).lower().replace('ё', 'е')
        text_from_sentences += simplified_sent
        indices_of_sentences += [idx + 1 for _ in range(len(simplified_sent))]
    indices_of_sentences = np.array(indices_of_sentences, dtype=np.int32)

    word_rule, sent_rule = dtw(text_form_words, text_from_sentences)
    indices_of_words = indices_of_words[word_rule]
    indices_of_sentences = indices_of_sentences[sent_rule]
    assert indices_of_sentences.shape[0] == indices_of_words.shape[0]
    sentence_bounds = []
    sent_start = -1
    prev_sent_id = -1
    for idx in range(indices_of_sentences.shape[0]):
        if indices_of_sentences[idx] != prev_sent_id:
            if sent_start >= 0:
                sentence_bounds.append((sent_start, indices_of_words[idx]))
            prev_sent_id = indices_of_sentences[idx]
            sent_start = indices_of_words[idx]
    assert sent_start >= 0
    sentence_bounds.append((sent_start, len(asr_result)))
    assert len(sentence_bounds) == len(sentences)

    res = []
    for cur_sent, (sent_start, sent_end) in zip(sentences, sentence_bounds):
        res.append(
            (
                cur_sent,
                asr_result[sent_start][1],
                asr_result[sent_end - 1][2]
            )
        )
    return res
