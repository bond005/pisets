from argparse import ArgumentParser
import codecs
import copy
import csv
import os


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input dataset for multi-task LLM.')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        err_msg = f'The file "{input_fname}" does not exist!'
        raise IOError(err_msg)

    true_header = ['input', 'target']
    loaded_header = []
    instruction_frequences = {}
    with codecs.open(input_fname, mode='r', encoding='utf-8') as fp:
        data_reader = csv.reader(fp, delimiter=',', quotechar='"')
        for row in data_reader:
            if len(row) > 0:
                err_msg = f'{input_fname}: the row is incorrect! {row}'
                if len(loaded_header) == 0:
                    loaded_header = copy.copy(row)
                    if loaded_header != true_header:
                        raise ValueError(err_msg)
                else:
                    if len(row) != len(loaded_header):
                        raise ValueError(err_msg)
                    input_text = row[0]
                    target_text = row[1]
                    if not input_text.startswith('<LM>'):
                        raise ValueError(err_msg)
                    if not target_text.endswith('</s>'):
                        raise ValueError(err_msg)
                    if input_text.startswith('<LM>Исправь, пожалуйста, ошибки распознавания речи в следующем тексте.'):
                        instruction_type = 'ASR Correction'
                    elif input_text.startswith('<LM>Упрости, пожалуйста, следующий текст.'):
                        instruction_type = 'Simplification'
                    elif input_text.startswith('<LM>Разбей, пожалуйста, следующий текст на абзацы.'):
                        instruction_type  = 'Segmentation'
                    elif input_text.startswith('<LM>Выполни саммаризацию и выдели, пожалуйста, основную мысль следующего текста.'):
                        instruction_type = 'Summarization'
                    else:
                        instruction_type = 'Unknown'
                    instruction_frequences[instruction_type] = instruction_frequences.get(instruction_type, 0) + 1
    task_list = sorted(list(instruction_frequences.keys()))
    max_text_width = max([len(it) for it in task_list])
    print('Task types:')
    for it in task_list:
        print('{0:<{1}} {2:>6}'.format(it + ':', max_text_width + 1, instruction_frequences[it]))


if __name__ == '__main__':
    main()
