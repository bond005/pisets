from typing import List, Tuple

import numpy as np
import torch
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer


MAX_DURATION_FOR_RESCORER = 12.0


def initialize_rescorer(language: str = 'ru') -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
    if language != 'ru':
        raise ValueError(f'The language "{language}" is not supported!')
    model_name = 'bond005/ruT5-ASR'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')
    return tokenizer, model


def levenshtein(seq1: str, seq2: str) -> List[Tuple[int, int]]:
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=np.int32)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    x = size_x - 1
    y = size_y - 1
    matches = [(x - 1, y - 1)]
    while (x > 1) and (y > 1):
        if matrix[x - 1, y - 1] <= matrix[x - 1, y]:
            if matrix[x - 1, y - 1] <= matrix[x, y - 1]:
                x -= 1
                y -= 1
            else:
                y -= 1
        else:
            if matrix[x - 1, y] < matrix[x, y - 1]:
                x -= 1
            else:
                y -= 1
        matches.insert(0, (x - 1, y - 1))
    if (x > 1) or (y > 1):
        if x == 1:
            y -= 1
            matches.insert(0, (0, y))
            while y >= 1:
                y -= 1
                matches.insert(0, (0, y))
        else:
            x -= 1
            matches.insert(0, (x, 0))
            while x >= 1:
                x -= 1
                matches.insert(0, (x, 0))
    return matches


def align(source_words: List[Tuple[str, float, float]], rescored_words: List[str]) -> List[Tuple[str, float, float]]:
    src_text = ''
    src_indices = []
    for idx, (word, _, _) in enumerate(source_words):
        src_text += word
        src_indices += [idx for _ in range(len(word))]
    rescored_text = ''
    rescored_indices = []
    for idx, word in enumerate(rescored_words):
        rescored_text += word
        rescored_indices += [idx for _ in range(len(word))]
    matches = list(map(
        lambda it: (src_indices[it[0]], rescored_indices[it[1]]),
        levenshtein(src_text, rescored_text)
    ))

    source_word_groups = []
    prev_rescored_idx = 0
    start_src_idx = 0
    end_src_idx = 0
    for src_idx, rescored_idx in matches:
        if rescored_idx == prev_rescored_idx:
            end_src_idx = src_idx
        else:
            assert rescored_idx == (prev_rescored_idx + 1), f'{rescored_idx} != {prev_rescored_idx + 1}'
            assert len(source_word_groups) == prev_rescored_idx, f'{len(source_word_groups)} != {prev_rescored_idx}'
            source_word_groups.append((start_src_idx, end_src_idx + 1))
            prev_rescored_idx = rescored_idx
            start_src_idx = src_idx
            end_src_idx = src_idx
    assert len(source_word_groups) == prev_rescored_idx, f'{len(source_word_groups)} != {prev_rescored_idx}'
    source_word_groups.append((start_src_idx, end_src_idx + 1))
    assert len(source_word_groups) == len(rescored_words), f'{len(source_word_groups)} != {len(rescored_words)}'

    res = []
    for idx, word in enumerate(rescored_words):
        word_group = source_word_groups[idx]
        first_word = source_words[word_group[0]]
        last_word = source_words[word_group[1] - 1]
        start_time = first_word[1]
        end_time = last_word[2]
        res.append((word, start_time, end_time))
    return res


def rescore(words: List[Tuple[str, float, float]], tokenizer: T5Tokenizer,
            model: T5ForConditionalGeneration) -> List[Tuple[str, float, float]]:
    if len(words) == 0:
        return []
    if len(words) < 3:
        return words
    if (words[-1][2] - words[0][1]) > MAX_DURATION_FOR_RESCORER:
        start_idx = 0
        end_idx = 1
        rescored_words = []
        while end_idx < len(words):
            if (words[end_idx][2] - words[start_idx][1]) > MAX_DURATION_FOR_RESCORER:
                rescored_words += rescore(words[start_idx:end_idx], tokenizer, model)
                start_idx = end_idx
            end_idx += 1
        rescored_words += rescore(words[start_idx:end_idx], tokenizer, model)
    else:
        text = ' '.join([cur[0] for cur in words])
        ru_letters = set('аоуыэяеёюибвгдйжзклмнпрстфхцчшщьъ')
        punct = set('.,:/\\?!()[]{};"\'-')
        x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
        max_size = int(x.input_ids.shape[1] * 1.5 + 10)
        min_size = 3
        if x.input_ids.shape[1] <= min_size:
            return words
        out = model.generate(**x, do_sample=False, num_beams=7,
                             max_length=max_size, min_length=min_size)
        res = tokenizer.decode(out[0], skip_special_tokens=True).lower().strip()
        res = ' '.join(res.split())
        postprocessed = ''
        for cur in res:
            if cur.isspace() or (cur in punct):
                postprocessed += ' '
            elif cur in ru_letters:
                postprocessed += cur
        postprocessed = (' '.join(postprocessed.strip().split())).replace('ё', 'е').strip()
        if len(postprocessed) == 0:
            rescored_words = words
        else:
            rescored_words = align(words, postprocessed.split())
    return rescored_words
