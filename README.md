[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bond005/pisets/blob/master/LICENSE)
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)

# pisets

This project represents a python library and service for automatic speech recognition and transcribing in Russian and English.

You can generate subtitles in the [SubRip format](https://en.wikipedia.org/wiki/SubRip) for any audio or video which is supported with [FFmpeg software](https://en.wikipedia.org/wiki/FFmpeg).

The "**pisets**" is Russian word (in Cyrillic, "писец") for denoting a person who writes down the text, including dictation (the corresponding English term is "scribe"). Thus, if you need to make a text transcript of an audio recording of a meeting or seminar, then the artificial "**Pisets**" will help you.

## Installation

This project uses a deep learning, therefore a key dependency is a deep learning framework. I prefer [PyTorch](https://pytorch.org/), and you need to install CPU- or GPU-based build of PyTorch ver. 2.3 or later. You can see more detailed description of dependencies in the `requirements.txt`.

Other important dependencies are:

- [Transformers](https://github.com/huggingface/transformers): a Python library for building neural networks with Transformer architecture;
- [FFmpeg](https://ffmpeg.org): a software for handling video, audio, and other multimedia files.

The first dependency is a well-known Python library, but the second dependency is not only "pythonic". You have to install FFmpeg in your system  as described in the instructions https://ffmpeg.org/download.html.

Also, for installation you need to Python 3.10 or later. I recommend using a new [Python virtual environment](https://docs.python.org/3/glossary.html#term-virtual-environment) witch can be created with [Anaconda](https://www.anaconda.com). To install this project in the selected virtual environment, you should activate this environment and run the following commands in the Terminal:

```shell
git clone https://github.com/bond005/pisets.git
cd pisets
python -m pip install -r requirements.txt
```

To check workability and environment setting correctness you can run the unit tests:

```shell
python -m unittest
```

## Usage

### Command prompt

Usage of the **Pisets** is very simple. You have to write the following command in your command prompt:

```shell
python speech_to_srt.py \
    -i /path/to/your/sound/or/video.m4a \
    -o /path/to/resulted/transcription.srt \
    -m /path/to/local/directory/with/models \
    -lang ru
```

The **1st** argument `-i` specifies the name of the source audio or video in any format supported by FFmpeg. 

The **2st** argument `-o` specifies the name of the resulting SubRip file into which the recognized transcription will be written.

Other arguments are not required. If you do not specify them, then their default values will be used. But I think, that their description matters for any user. So, `-lang` specifies the used language. You can select Russian (*ru*, *rus*, *russian*) or English (*en*, *eng*, *english*). The default language is Russian. Yet another argument `-m` points to the directory with all needed pre-downloaded models. This directory must include several subdirectories, which contain localized models for corresponding languages (`ru` or `en` is supported now). In turn, each language subdirectory includes three more subdirectories corresponding to the three models used:

1) `wav2vec2` (for preliminary speech recognition and segmentation into speech frames);
2) `ast` (for filtering non-speech segments);
3) `whisper` (for final speech recognition).

If you don't specify the argument `-m`, then all needed models will be automatically downloaded from Huggingface hub:

- for Russian:
  1) [bond005/Wav2Vec2-Large-Ru-Golos](https://huggingface.co/bond005/wav2vec2-large-ru-golos),
  2) [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593),
  3) [bond005/whisper-large-v3-ru-podlodka](https://huggingface.co/bond005/whisper-large-v3-ru-podlodka);

- for English:
  1) [jonatasgrosman/wav2vec2-large-xlsr-53-english](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english),
  2) [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593),
  3) [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3).

Also, you can generate the transcription of your audio-record as a DocX file:

```shell
python speech_to_docx.py \
    -i /path/to/your/sound/or/video.m4a \
    -o /path/to/resulted/transcription.docx \
    -m /path/to/local/directory/with/models \
    -lang ru
```

If your computer has CUDA-compatible GPU, and your PyTorch has been correctly installed for this GPU, then the **Pisets** will transcribe your speech very quickly. So, the real-time factor (xRT), defined as the ratio between the time it takes to process the input and the duration of the input, is approximately 0.15 - 0.25 (it depends on the concrete GPU type). But if you use CPU only, then the **Pisets** will calculate your speech transcription significantly slower (xRT is approximately 1.0 - 1.5).

### Docker and REST-API

Installation of the **Pisets** can be difficult, especially for Windows users (in Linux it is trivial). Accordingly, in order to simplify the installation process and hide all the difficulties from the user, I suggest using a docker container that is deployed and runs on any operating system. In this case, audio transmission for recognition and receiving transcription results is carried out by means of the REST API.

You can build the docker container youself:

```shell
docker build -t bond005/pisets:0.2 .
```

But the easiest way is to download the built image from Docker-Hub:

```shell
docker pull bond005/pisets:0.2
```

After building (or pulling) you have to run this docker container:

```shell
docker run --rm --gpus all -p 127.0.0.1:8040:8040 bond005/pisets:0.2
```

Hurray! The docker container is ready for use on GPU, and the **Pisets** will transcribe your speech. You can use the Python client for the **Pisets** service in the script [client_ru_demo.py](https://github.com/bond005/pisets/blob/main/client_ru_demo.py):

```shell
python client_ru_demo.py \
    -i /path/to/your/sound/or/video.m4a \
    -o /path/to/resulted/transcription.docx
```

#### Important notes
The **Pisets** in the abovementioned docker container currently supports only Russian. If you want to transcribe English speech, then you have to use the command-line tool `speech_to_srt.py` or `speech_to_docx.py`.

### Cloud computing

You can open your [personal account](https://lk.sibnn.ai/login?redirect=/) (in Russian) on the [SibNN.AI](https://sibnn.ai/) and upload your audio recordings of any size for their automatic recognition.

In addition, you can try the demo of the cloud **Pisets** without registration on the web-page https://pisets.dialoger.tech (the demo without registration contains a limit on the maximum length of an audio recording of no more than 5 minutes, but allows you to record a signal from a microphone).

## Contact

Ivan Bondarenko - [@Bond_005](https://t.me/Bond_005) - [bond005@yandex.ru](mailto:bond005@yandex.ru)

## Acknowledgment

This project was developed as part of a more fundamental project to create an open source system for automatic transcription and semantic analysis of audio recordings of interviews  in Russian. Many journalists, sociologist and other specialists need to prepare the interview manually, and automatization can help their.

The [Foundation for Assistance to Small Innovative Enterprises](https://fasie.ru) which is Russian governmental non-profit organization supports an unique program to build free and open-source artificial intelligence systems. This programs is known as "Code - Artificial Intelligence" (see https://fasie.ru/press/fund/kod-ai/?sphrase_id=114059 in Russian). The abovementioned project was started within the first stage of the "Code - Artificial Intelligence" program. You can see the first-stage winners list on this web-page: https://fasie.ru/competitions/kod-ai-results (in Russian).

Therefore, I thank The Foundation for Assistance to Small Innovative Enterprises for this support.

## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.
