ARG BASEIMAGE=nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
FROM ${BASEIMAGE}

ARG DISC_HOST_TF_VERSION=""
ENV DISC_HOST_TF_VERSION="${DISC_HOST_TF_VERSION}"

COPY docker/scripts /install/scripts

RUN bash /install/scripts/find-fastest-apt.sh && \
    apt-get install -y python3 python3-pip git curl vim libssl-dev wget unzip openjdk-11-jdk && \
    pip3 install --upgrade pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    bash /install/scripts/install-cmake.sh && \
    bash /install/scripts/install-bazel.sh && \
    bash /install/scripts/install-python.sh

ARG DEVICE=cu110
RUN bash /install/scripts/install-tensorrt.sh "${DEVICE}"
RUn bash /install/scripts/patch-cu102.sh "${DEVICE}"

ENV PATH="/opt/cmake/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/TensorRT/lib/:${LD_LIBRARY_PATH}"
