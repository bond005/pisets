FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
MAINTAINER Ivan Bondarenko <i.bondarenko@g.nsu.ru>

ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Установка необходимых пакетов и очистка кэша
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        gcc \
        make \
        autoconf \
        automake \
        apt-transport-https \
        build-essential \
        git \
        g++ \
        autoconf-archive \
        libtool \
        python3-setuptools \
        python3-dev \
        cmake-data \
        vim \
        wget \
        libbz2-dev \
        ffmpeg \
        tar \
        zip \
        unzip \
        zlib1g \
        zlib1g-dev \
        lzma \
        liblzma-dev \
        libboost-all-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.0"

# Проверка версий Python и pip
RUN python3 --version && pip3 --version

# Создание необходимых директорий
RUN mkdir -p /usr/src/pisets /usr/src/huggingface_cached

# Копирование файлов
COPY ./server_ru.py /usr/src/pisets/
COPY ./download_models.py /usr/src/pisets/
COPY ./requirements.txt /usr/src/pisets/
COPY ./asr/ /usr/src/pisets/asr/
COPY ./utils/ /usr/src/pisets/utils/
COPY ./vad/ /usr/src/pisets/vad/
COPY ./wav_io/ /usr/src/pisets/wav_io/

WORKDIR /usr/src/pisets

# Установка зависимостей Python и очистка кэша pip
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip cache purge

# Установка переменных окружения
ENV HF_HOME=/usr/src/huggingface_cached
ENV PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

# Проверка работы библиотеки transformers
RUN python -c "from transformers import pipeline; print(pipeline('sentiment-analysis', model='philschmid/tiny-bert-sst2-distilled')('we love you'))"

# Запуск скрипта для загрузки моделей
RUN python3 download_models.py ru

ENTRYPOINT ["python3", "server_ru.py"]