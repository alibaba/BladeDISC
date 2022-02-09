ARG BASEIMAGE=nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
FROM ${BASEIMAGE}

ARG WHEEL_FILE=blade_disc*.whl

ADD ./docker/scripts /install/scripts
RUN bash /install/scripts/find-fastest-apt.sh

ADD ./build/${WHEEL_FILE}  /install/python/
ADD ./build/disc-replay-main /usr/bin/disc-replay-main

RUN apt-get install -y python3 python3-pip \
    && pip3 install --upgrade pip \
    && ln -s /usr/bin/python3.6 /usr/bin/python \
    && pip install /install/python/${WHEEL_FILE}
