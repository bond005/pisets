[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bond005/pisets/blob/master/LICENSE)
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)

# pisets

This project represents a python library and service for automatic speech recognition and transcribing in Russian and English.

You can generate subtitles in the [SubRip format](https://en.wikipedia.org/wiki/SubRip) for any audio or video which is supported with [FFmpeg software](https://en.wikipedia.org/wiki/FFmpeg).

The "**pisets**" is Russian word (in Cyrillic, "писец") for denoting a person who writes down the text, including dictation (the corresponding English term is "scribe"). Thus, if you need to make a text transcript of an audio recording of a meeting or seminar, then the artificial "**Pisets**" will help you.

## Installation

This project uses a deep learning, therefore a key dependency is a deep learning framework. I prefer [PyTorch](https://pytorch.org/), and you need to install CPU- or GPU-based build of PyTorch ver. 2.0 or later. You can see more detailed description of dependencies in the `requirements.txt`.

Other important dependencies are:

- [KenLM](https://github.com/kpu/kenlm): a statistical N-gram language model inference code;
- [FFmpeg](https://ffmpeg.org): a software for handling video, audio, and other multimedia files.

These dependencies are not only "pythonic". Firstly, you have to build the KenLM C++ library from sources accordingly this recommendation: https://github.com/kpu/kenlm#compiling (it is easy for any Linux user, but it can be a problem for Windows users, because KenLM is not fully cross-platform). Secondly, you have to install FFmpeg in your system  as described in the instructions https://ffmpeg.org/download.html.

Also, for installation you need to Python 3.9 or later. I recommend using a new [Python virtual environment](https://docs.python.org/3/glossary.html#term-virtual-environment) witch can be created with [Anaconda](https://www.anaconda.com) or [venv](https://docs.python.org/3/library/venv.html#module-venv). To install this project in the selected virtual environment, you should activate this environment and run the following commands in the Terminal:

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
    -lang ru \
    -r \
    -f 50
```

The **1st** argument `-i` specifies the name of the source audio or video in any format supported by FFmpeg. 

The **2st** argument `-o` specifies the name of the resulting SubRip file into which the recognized transcription will be written.

Other arguments are not required. If you do not specify them, then their default values will be used. But I think, that their description matters for any user. So, `-lang` specifies the used language. You can select Russian (*ru*, *rus*, *russian*) or English (*en*, *eng*, *english*). The default language is Russian.

`-r` indicates the need for a more smart rescoring of speech hypothesis with a large language model as like as T5. This option is possible for Russian only, but it is important for good quality of generated transcription. Thus, I highly recommend using the option `-r` if you want to transcribe a Russian speech signal.

`-f` sets the maximum duration of the sound frame (in seconds). The fact is that the **Pisets** is designed so that a very long audio signal is divided into smaller sound frames, then these frames are recognized independently, and the recognition results are glued together into a single transcription. The need for such a procedure is due to the architecture of the acoustic neural network. And this argument determines the maximum duration of such frame, as defined above. The default value is 50 seconds, and I don't recommend changing it.

### Docker and REST-API

Installation of the **Pisets** can be difficult, especially for Windows users (in Linux it is trivial). Accordingly, in order to simplify the installation process and hide all the difficulties from the user, I suggest using a docker container that is deployed and runs on any operating system. In this case, audio transmission for recognition and receiving transcription results is carried out by means of the REST API.

You can build the docker container youself:

```shell
docker build -t bond005/pisets:0.1 .
```

But the easiest way is to download the built image from Docker-Hub:

```shell
docker pull bond005/pisets:0.1
```

After building (or pulling) you have to run this docker container:

```shell
docker run -p 127.0.0.1:8040:8040 pisets:0.1
```

Hurray! The docker container is ready for use, and the **Pisets** will transcribe your speech.

## Contact

Ivan Bondarenko - [@Bond_005](https://t.me/Bond_005) - [bond005@yandex.ru](mailto:bond005@yandex.ru)

## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.
