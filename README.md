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
    -lang ru \
    -f 50
```

The **1st** argument `-i` specifies the name of the source audio or video in any format supported by FFmpeg. 

The **2st** argument `-o` specifies the name of the resulting SubRip file into which the recognized transcription will be written.

Other arguments are not required. If you do not specify them, then their default values will be used. But I think, that their description matters for any user. So, `-lang` specifies the used language. You can select Russian (*ru*, *rus*, *russian*) or English (*en*, *eng*, *english*). The default language is Russian.

`-r` indicates the need for a more smart rescoring of speech hypothesis with a large language model as like as T5. This option is possible for Russian only, but it is important for good quality of generated transcription. Thus, I highly recommend using the option `-r` if you want to transcribe a Russian speech signal.

`-f` sets the maximum duration of the sound frame (in seconds). The fact is that the **Pisets** is designed so that a very long audio signal is divided into smaller sound frames, then these frames are recognized independently, and the recognition results are glued together into a single transcription. The need for such a procedure is due to the architecture of the acoustic neural network. And this argument determines the maximum duration of such frame, as defined above. The default value is 50 seconds, and I don't recommend changing it.

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
1. The **Pisets** in the abovementioned docker container currently supports only Russian. If you want to transcribe English speech, then you have use the command-line tool `speech_to_srt.py`.

2. The docker container and the command-line tool support CPU, but the execution of Pisets on CPU-based machine is very long. I recommend using only a GPU with at least 16 GB of memory.

## Models and algorithms

The **Pisets** transcribes speech signal in four steps:

1. The acoustic deep neural network, based on fine-tuned [Wav2Vec2](https://arxiv.org/abs/2006.11477), performs the primary recognition of the speech signal and calculates the probabilities of the recognized letters. So the result of the first step is a probability matrix.
2. The statistical N-gram language model translates the probability matrix into recognized text using a CTC beam search decoder.
3. The language deep neural network, based on fine-tuned [T5](https://arxiv.org/abs/2010.11934), corrects possible errors and generates the final recognition text in a "pure" form (without punctuations, only in lowercase, and so on). 
4. The last component of the "Pisets" places punctuation marks and capital letters.

The first and the second steps for English speech are implemented with Patrick von Platen's [Wav2Vec2-Base-960h + 4-gram](https://huggingface.co/patrickvonplaten/wav2vec2-large-960h-lv60-self-4-gram), and Russian speech transcribing is based on my [Wav2Vec2-Large-Ru-Golos-With-LM](https://huggingface.co/bond005/wav2vec2-large-ru-golos-with-lm).

The third step is not supported for English speech, but it is based on my [ruT5-ASR](https://huggingface.co/bond005/ruT5-ASR) for Russian speech.

The fourth step is realized on basis of [the multilingual text enhancement model created by Silero](https://github.com/snakers4/silero-models#text-enhancement).

My tests show a strong superiority of the recognition system based on the given scheme over Whisper Medium, and a significant superiority over Whisper Large when transcribing Russian speech. The methodology and test results are open:

- Wav2Vec2 + 3-gram LM + T5-ASR for Russian: https://www.kaggle.com/code/bond005/wav2vec2-ru-lm-t5-eval
- Whisper Medium for Russian: https://www.kaggle.com/code/bond005/whisper-medium-ru-eval

Also, you can see the independent evaluation of my [ Wav2Vec2-Large-Ru-Golos-With-LM](https://huggingface.co/bond005/wav2vec2-large-ru-golos-with-lm) model (without T5-based rescorer) on various Russian speech corpora in comparison with other open Russian speech recognition models: https://alphacephei.com/nsh/2023/01/22/russian-models.html (in Russian). 

## Contact

Ivan Bondarenko - [@Bond_005](https://t.me/Bond_005) - [bond005@yandex.ru](mailto:bond005@yandex.ru)

## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.
