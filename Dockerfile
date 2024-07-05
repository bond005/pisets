FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
MAINTAINER Ivan Bondarenko <i.bondarenko@g.nsu.ru>

ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

RUN apt-get update

RUN apt-get install -y apt-utils && \
    apt-get install -y gcc && \
    apt-get install -y make && \
    apt-get install -y autoconf && \
    apt-get install -y automake && \
    apt-get install -y apt-transport-https && \
    apt-get install -y build-essential && \
    apt-get install -y git g++ autoconf-archive libtool && \
    apt-get install -y python3-setuptools python3-dev && \
    apt-get install -y cmake-data && \
    apt-get install -y vim && \
    apt-get install -y wget && \
    apt-get install -y libbz2-dev && \
    apt-get install -y ffmpeg && \
    apt-get install -y tar zip unzip && \
    apt-get install -y zlib1g zlib1g-dev lzma liblzma-dev && \
    apt-get install -y libboost-all-dev

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.0"

RUN python3 --version
RUN pip3 --version

RUN mkdir /usr/src/pisets
RUN mkdir /usr/src/huggingface_cached

COPY ./server_ru.py /usr/src/pisets/server_ru.py
COPY ./download_models.py /usr/src/pisets/download_models.py
COPY ./requirements.txt /usr/src/pisets/requirements.txt
COPY ./asr/ /usr/src/pisets/asr/
COPY ./utils/ /usr/src/pisets/utils/
COPY ./vad/ /usr/src/pisets/vad/
COPY ./wav_io/ /usr/src/pisets/wav_io/

WORKDIR /usr/src/pisets

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

RUN export HF_HOME=/usr/src/huggingface_cached
RUN export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
RUN python -c "from transformers import pipeline; print(pipeline('sentiment-analysis', model='philschmid/tiny-bert-sst2-distilled')('we love you'))"

RUN python3 download_models.py ru

ENTRYPOINT ["python3", "server_ru.py"]
