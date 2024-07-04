import sys

from asr.asr import initialize_model_for_speech_recognition
from asr.asr import initialize_model_for_speech_segmentation
from asr.asr import check_language


def main():
    if len(sys.argv) <= 1:
        raise ValueError('The language name is not specified!')
    language_name = check_language(sys.argv[1])

    _ = initialize_model_for_speech_segmentation(language_name)
    _ = initialize_model_for_speech_recognition(language_name)


if __name__ == '__main__':
    main()
