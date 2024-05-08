FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04
MAINTAINER Ivan Bondarenko <i.bondarenko@g.nsu.ru>

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y apt-utils && \
    apt-get install -y gcc && \
    apt-get install -y make && \
    apt-get install -y autoconf && \
    apt-get install -y automake && \
    apt-get install -y apt-transport-https && \
    apt-get install -y build-essential && \
    apt-get install -y git g++ autoconf-archive libtool && \
    # apt-get install -y python-setuptools python-dev && \
    apt-get install -y python3-setuptools python3-dev && \
    apt-get install -y cmake-data && \
    apt-get install -y vim && \
    apt-get install -y wget && \
    apt-get install -y libbz2-dev && \
    apt-get install -y ffmpeg && \
    apt-get install -y tar zip unzip && \
    apt-get install -y zlib1g zlib1g-dev lzma liblzma-dev && \
    apt-get install -y libboost-all-dev

RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3.tar.gz
RUN tar -zxvf cmake-3.26.3.tar.gz
RUN rm cmake-3.26.3.tar.gz
WORKDIR cmake-3.26.3
RUN apt-get update && apt-get install -y libssl-dev
RUN ./configure
RUN make
RUN make install
WORKDIR ..

RUN git clone https://github.com/kpu/kenlm.git
RUN mkdir -p kenlm/build
WORKDIR kenlm/build
RUN cmake ..
RUN make
RUN make install
WORKDIR ..

RUN mkdir /usr/src/pisets

COPY ./server_ru.py /usr/src/pisets/server_ru.py
COPY ./download_models.py /usr/src/pisets/download_models.py
COPY ./requirements.txt /usr/src/pisets/requirements.txt
COPY ./asr/ /usr/src/pisets/asr/
COPY ./normalization/ /usr/src/pisets/normalization/
COPY ./rescoring/ /usr/src/pisets/rescoring/
COPY ./utils/ /usr/src/pisets/utils/
COPY ./vad/ /usr/src/pisets/vad/
COPY ./wav_io/ /usr/src/pisets/wav_io/
COPY ./models/ /usr/src/pisets/models/

WORKDIR /usr/src/pisets

RUN apt-get update && \
    apt-get install -y python3.8 && apt-get install -y python3-pip && \
    python3 -m pip install --upgrade pip \
    yes | apt-get install build-essential -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg && \
    apt-get install cron nano && set $EDITOR=nano && apt-get install -y cuda-toolkit-11-7

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir jupyter -U && pip install --no-cache-dir jupyterlab
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN python3 -m pip install -r requirements.txt

# RUN python3 download_models.py ru

ENV NV_CUDNN_VERSION 8.5.0.96
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"
ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda11.7"
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
