import csv
import random
from argparse import ArgumentParser
import codecs
import logging
import os
import sys

import numpy as np
from datasets import load_dataset, disable_caching
from tqdm import tqdm

from llm.autolabeling import generate_answer_with_gigachat
from llm.autolabeling import build_prompt_for_detalization, build_prompt_for_simplification
from llm.autolabeling import gigachat_logger


segmentation_dataset_logger = logging.getLogger(__name__)


def mean_word_length(text: str) -> float:
    return np.mean(list(map(lambda it: len(it), text.split())))


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input dataset for post-ASR correction.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output dataset for segmentation.')
    parser.add_argument('-c', '--credentials', dest='credentials', type=str, required=True,
                        help='The credentials for Gigachat API.')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        err_msg = f'The file "{input_fname}" does not exist!'
        segmentation_dataset_logger.error(err_msg)
        raise IOError(err_msg)

    output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_fname):
        basedir = os.path.dirname(output_fname)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                err_msg = f'The directory "{basedir}" does not exist!'
                segmentation_dataset_logger.error(err_msg)
                raise IOError(err_msg)
    if os.path.basename(output_fname) == os.path.basename(input_fname):
        err_msg = 'The output file is same as the input file!'
        segmentation_dataset_logger.error(err_msg)
        raise ValueError(err_msg)

    disable_caching()
    try:
        input_dataset = load_dataset(
            os.path.dirname(input_fname),
            data_files=os.path.basename(input_fname),
            keep_in_memory=True
        )['train']
    except BaseException as ex:
        err_msg = str(ex)
        segmentation_dataset_logger.error(err_msg)
        raise
    info_msg = (f'The dataset "{input_fname}" is loaded. The column names are: {input_dataset.column_names}. '
                f'The dataset size is {len(input_dataset)}.')
    segmentation_dataset_logger.info(info_msg)
    if 'input' not in input_dataset.column_names:
        err_msg = f'The column "input" is not found in the dataset "{input_fname}".'
        segmentation_dataset_logger.error(err_msg)
        raise ValueError(err_msg)
    if 'target' not in input_dataset.column_names:
        err_msg = f'The column "input" is not found in the dataset "{input_fname}".'
        segmentation_dataset_logger.error(err_msg)
        raise ValueError(err_msg)

    input_texts = []
    instructions = []
    for cur in input_dataset:
        if mean_word_length(cur['target']) > 2.0:
            input_texts.append(cur['target'])
            instructions.append(
                (
                    '<LM>Исправь, пожалуйста, ошибки распознавания речи в следующем тексте. ' + cur['input'],
                    cur['target'] + '</s>'
                )
            )
    input_texts = sorted(list(set(input_texts)))
    del input_dataset
    segmentation_dataset_logger.info(f'There are {len(input_texts)} input texts after deduplication.')
    random.shuffle(input_texts)

    with (codecs.open(output_fname, mode='w', encoding='utf-8') as fp):
        csv_writer = csv.writer(fp, delimiter=',', quotechar='"')
        csv_writer.writerow(['input', 'target'])
        for input_prompt, true_response in instructions:
            csv_writer.writerow([input_prompt, true_response])
        del instructions
        for cur_text in tqdm(input_texts):
            try:
                new_prompt = build_prompt_for_simplification(cur_text)
                try:
                    simpler_text = generate_answer_with_gigachat(new_prompt, args.credentials)
                except BaseException as ex:
                    err_msg = str(ex) + ' ' + new_prompt
                    segmentation_dataset_logger.error(err_msg)
                    raise
                del new_prompt
                simpler_text = ' '.join(simpler_text.strip().split())
                if mean_word_length(simpler_text) <= 2.0:
                    warn_msg = (f'{cur_text} cannot be simplified. The result has a strange tokenization. '
                                f'{simpler_text}')
                    segmentation_dataset_logger.warning(warn_msg)
                elif (len(simpler_text) > 0) and (len(simpler_text) < len(cur_text)):
                    input_prompt = '<LM>Упрости, пожалуйста, следующий текст. ' + ' '.join(cur_text.strip().split())
                    true_response = simpler_text + '</s>'
                    csv_writer.writerow([input_prompt, true_response])
                    del input_prompt, true_response
                else:
                    warn_msg = (f'{cur_text} cannot be simplified. The result is not simpler than original. '
                                f'{simpler_text}')
                    segmentation_dataset_logger.warning(warn_msg)
                new_prompt = build_prompt_for_detalization(cur_text)
                try:
                    long_text = generate_answer_with_gigachat(new_prompt, args.credentials)
                except BaseException as ex:
                    err_msg = str(ex) + ' ' + new_prompt
                    segmentation_dataset_logger.error(err_msg)
                    raise
                del new_prompt
                if ('\n' in long_text) and (len(long_text) > (1.3 * len(cur_text))) and \
                        (mean_word_length(long_text) > 2.0):
                    short_segments = list(filter(
                        lambda it2: len(it2) > 0,
                        map(lambda it1: ' '.join(it1.strip().split()), long_text.split('\n'))
                    ))
                    if len(short_segments) > 1:
                        input_prompt = '<LM>Разбей, пожалуйста, следующий текст на абзацы. ' + ' '.join(short_segments)
                        true_response = '\n'.join(short_segments) + '</s>'
                        csv_writer.writerow([input_prompt, true_response])
                        del input_prompt, true_response
                        new_prompt = build_prompt_for_simplification(' '.join(short_segments))
                        try:
                            annotation = generate_answer_with_gigachat(new_prompt, args.credentials)
                        except BaseException as ex:
                            err_msg = str(ex) + ' ' + new_prompt
                            segmentation_dataset_logger.error(err_msg)
                            raise
                        del new_prompt
                        annotation = ' '.join(annotation.strip().split())
                        if len(annotation) > 0:
                            input_prompt = ('<LM>Выполни саммаризацию и выдели, пожалуйста, основную мысль '
                                            'следующего текста. ')
                            input_prompt += ' '.join(short_segments)
                            true_response = annotation + '</s>'
                            csv_writer.writerow([input_prompt, true_response])
                            del input_prompt, true_response
                        else:
                            warn_msg = f'{" ".join(short_segments)} cannot be simplified. The annotation is empty.'
                            segmentation_dataset_logger.warning(warn_msg)
                    else:
                        warn_msg = f'{cur_text} cannot be detailed into several paragraphs.'
                        segmentation_dataset_logger.warning(warn_msg)
                else:
                    warn_msg = f'{" ".join(cur_text.strip().split())} cannot be detailed into several paragraphs.'
                    if mean_word_length(long_text) <= 2.0:
                        warn_msg += ' The result has a strange tokenization. ' + ' '.join(long_text.split())
                    elif len(long_text) <= (1.3 * len(cur_text)):
                        warn_msg += ' The result is not more detailed than the original. ' + ' '.join(long_text.split())
                    segmentation_dataset_logger.warning(warn_msg)
                    new_prompt = build_prompt_for_simplification(cur_text)
                    try:
                        annotation = generate_answer_with_gigachat(new_prompt, args.credentials)
                    except BaseException as ex:
                        err_msg = str(ex) + ' ' + new_prompt
                        segmentation_dataset_logger.error(err_msg)
                        raise
                    del new_prompt
                    annotation = ' '.join(annotation.strip().split())
                    if (len(annotation) > 0) and (len(annotation) < (len(cur_text) // 2)):
                        input_prompt = ('<LM>Выполни саммаризацию и выдели, пожалуйста, основную мысль '
                                        'следующего текста. ')
                        input_prompt += ' '.join(cur_text.strip().split())
                        true_response = annotation + '</s>'
                        csv_writer.writerow([input_prompt, true_response])
                        del input_prompt, true_response
                    else:
                        warn_msg = f'{cur_text} cannot be simplified. The annotation is too long. {annotation}'
                        segmentation_dataset_logger.warning(warn_msg)
            except Exception as ex:
                segmentation_dataset_logger.warning(str(ex))


if __name__ == '__main__':
    segmentation_dataset_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    segmentation_dataset_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('nlp_dataset.log')
    file_handler.setFormatter(formatter)
    segmentation_dataset_logger.addHandler(file_handler)
    gigachat_logger.addHandler(file_handler)
    gigachat_logger.addHandler(stdout_handler)
    main()
