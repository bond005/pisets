import csv
from argparse import ArgumentParser
import codecs
import logging
import os
import sys

from datasets import load_dataset, disable_caching
from tqdm import tqdm

from llm.llm import initialize_saiga_mistral, generate_answer_with_saiga_mistral
from llm.llm import build_prompt_for_detalization, build_prompt_for_simplification
from llm.llm import saiga_mistral_logger


segmentation_dataset_logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input dataset in the Huggingface-compatible format.')
    parser.add_argument('-c', '--column', dest='column_name', type=str, required=True,
                        help='The text column in the input dataset.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output dataset for segmentation.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The adapter of the Saiga Mistral LLM by Ilya Gusev.')
    parser.add_argument('-b', '--base', dest='base_model_name', type=str, required=False, default=None,
                        help='The base model of the Saiga Mistral LLM by Ilya Gusev.')
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

    column_name = args.column_name.strip()
    if len(column_name) == 0:
        err_msg = 'The column name is empty.'
        segmentation_dataset_logger.error(err_msg)
        raise ValueError(err_msg)

    model_name = os.path.normpath(args.model_name)
    if not os.path.isdir(input_fname):
        err_msg = f'The directory "{model_name}" does not exist!'
        segmentation_dataset_logger.error(err_msg)
        raise IOError(err_msg)
    if os.path.basename(output_fname) == os.path.basename(model_name):
        err_msg = 'The output file is same as the model directory!'
        segmentation_dataset_logger.error(err_msg)
        raise ValueError(err_msg)

    if args.base_model_name is not None:
        base_model_name = os.path.normpath(args.base_model_name)
        if not os.path.isdir(input_fname):
            err_msg = f'The directory "{base_model_name}" does not exist!'
            segmentation_dataset_logger.error(err_msg)
            raise IOError(err_msg)
        if os.path.basename(output_fname) == os.path.basename(base_model_name):
            err_msg = 'The output file is same as the model directory!'
            segmentation_dataset_logger.error(err_msg)
            raise ValueError(err_msg)
    else:
        base_model_name = None

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
    if column_name not in input_dataset.column_names:
        err_msg = f'The column "{column_name}" is not found in the dataset "{input_fname}".'
        segmentation_dataset_logger.error(err_msg)
        raise ValueError(err_msg)

    input_texts = sorted(list(set([cur[column_name] for cur in input_dataset])))
    del input_dataset
    segmentation_dataset_logger.info(f'There are {len(input_texts)} input texts after deduplication.')

    try:
        tokenizer, model, generation = initialize_saiga_mistral(model_name, base_model_name)
    except BaseException as ex:
        err_msg = str(ex)
        segmentation_dataset_logger.error(err_msg)
        raise
    segmentation_dataset_logger.info(f'The Saiga Mistral is loaded from the "{model_name}".')

    with codecs.open(output_fname, mode='w', encoding='utf-8') as fp:
        csv_writer = csv.writer(fp, delimiter=',', quotechar='"')
        csv_writer.writerow(['input', 'target'])
        for cur_text in tqdm(input_texts):
            try:
                simpler_text = generate_answer_with_saiga_mistral(
                    prompt=build_prompt_for_simplification(cur_text),
                    tokenizer=tokenizer,
                    model=model,
                    generation=generation
                )
            except BaseException as ex:
                err_msg = str(ex)
                segmentation_dataset_logger.error(err_msg)
                raise
            simpler_text = ' '.join(simpler_text.strip().split())
            if len(simpler_text) > 0:
                input_prompt = '<LM>Упрости, пожалуйста, следующий текст. ' + ' '.join(cur_text.strip().split())
                true_response = simpler_text + '</s>'
                csv_writer.writerow([input_prompt, true_response])
                del input_prompt, true_response
            else:
                warn_msg = f'{simpler_text} cannot be simplified.'
                segmentation_dataset_logger.warning(warn_msg)
            try:
                long_text = generate_answer_with_saiga_mistral(
                    prompt=build_prompt_for_detalization(cur_text),
                    tokenizer=tokenizer,
                    model=model,
                    generation=generation
                )
            except BaseException as ex:
                err_msg = str(ex)
                segmentation_dataset_logger.error(err_msg)
                raise
            if '\n' in long_text:
                short_segments = list(filter(
                    lambda it2: len(it2) > 0,
                    map(lambda it1: ' '.join(it1.strip().split()), long_text.split('\n'))
                ))
                if len(short_segments) > 1:
                    input_prompt = '<LM>Разбей, пожалуйста, следующий текст на абзацы. ' + ' '.join(short_segments)
                    true_response = '\n'.join(short_segments) + '</s>'
                    csv_writer.writerow([input_prompt, true_response])
                    del input_prompt, true_response
                    try:
                        annotation = generate_answer_with_saiga_mistral(
                            prompt=build_prompt_for_simplification(' '.join(short_segments)),
                            tokenizer=tokenizer,
                            model=model,
                            generation=generation
                        )
                    except BaseException as ex:
                        err_msg = str(ex)
                        segmentation_dataset_logger.error(err_msg)
                        raise
                    annotation = ' '.join(annotation.strip().split())
                    if len(annotation) > 0:
                        input_prompt = ('<LM>Выполни саммаризацию и выдели, пожалуйста, основную мысль '
                                        'следующего текста. ')
                        input_prompt += ' '.join(short_segments)
                        true_response = annotation + '</s>'
                        csv_writer.writerow([input_prompt, true_response])
                        del input_prompt, true_response
                    else:
                        warn_msg = f'{" ".join(short_segments)} cannot be simplified.'
                        segmentation_dataset_logger.warning(warn_msg)
                else:
                    warn_msg = f'{cur_text} cannot be detailed into several paragraphs.'
                    segmentation_dataset_logger.warning(warn_msg)
            else:
                warn_msg = f'{cur_text} cannot be detailed into several paragraphs.'
                segmentation_dataset_logger.warning(warn_msg)


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
    saiga_mistral_logger.addHandler(file_handler)
    saiga_mistral_logger.addHandler(stdout_handler)
    main()
