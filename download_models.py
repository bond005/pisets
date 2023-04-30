import sys

from asr.asr import initialize_model
from rescoring.rescoring import initialize_rescorer
from normalization.normalization import check_language, initialize_normalizer


def main():
    if len(sys.argv) <= 1:
        raise ValueError('The language name is not specified!')
    language_name = check_language(sys.argv[1])

    processor, model = initialize_model(language_name)
    tokenizer_for_rescorer, model_of_rescorer = initialize_rescorer(language_name)
    text_normalizer = initialize_normalizer()


if __name__ == '__main__':
    main()
