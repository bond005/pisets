[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bond005/pisets/blob/master/LICENSE)
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)

# pisets

This project represents a python library and service for automatic speech recognition and transcribing in Russian and English.

You can generate subtitles in the [SubRip format](https://en.wikipedia.org/wiki/SubRip) for any audio or video which is supported with [FFmpeg software](https://en.wikipedia.org/wiki/FFmpeg).

The "pisets" is Russian word (in Cyrillic, "писец") for denoting a person who writes down the text, including dictation (the corresponding English term is "scribe"). Thus, if you need to make a text transcript of an audio recording of a meeting or seminar, then the artificial "Pisets" will help you.

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

Text...

## Contact

Ivan Bondarenko - [@Bond_005](https://t.me/Bond_005) - [bond005@yandex.ru](mailto:bond005@yandex.ru)

## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.
