FROM python:3.9
MAINTAINER Ivan Bondarenko <i.bondarenko@g.nsu.ru>

RUN apt-get update && \
    yes | apt-get install apt-utils && \
    yes | apt-get install -y gcc && \
    yes | apt-get install -y make && \
    yes | apt-get install -y apt-transport-https && \
    yes | apt-get install -y build-essential && \
    yes | apt-get install git g++ autoconf-archive make libtool && \
    yes | apt-get install python-setuptools python-dev && \
    yes | apt-get install python3-setuptools python3-dev && \
    yes | apt-get install vim && \
    yes | apt-get install libbz2-dev \
    yes | apt-get install ffmpeg

RUN python3 --version
RUN pip3 --version

RUN mkdir /usr/src/pisets

COPY ./server_ru.py /usr/src/pisets/server_ru.py
COPY ./requirements.txt /usr/src/pisets/requirements.txt
COPY ./asr/ /usr/src/pisets/asr/
COPY ./normalization/ /usr/src/pisets/normalization/
COPY ./rescoring/ /usr/src/pisets/rescoring/
COPY ./utils/ /usr/src/pisets/utils/
COPY ./vad/ /usr/src/pisets/vad/
COPY ./wav_io/ /usr/src/pisets/wav_io/
COPY ./models/ /usr/src/pisets/models/

WORKDIR /usr/src/pisets

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

ENTRYPOINT ["python3", "server_ru.py"]
