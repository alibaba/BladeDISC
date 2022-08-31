ARG BASEIMAGE=nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
FROM ${BASEIMAGE}

ARG DISC_HOST_TF_VERSION=""
ENV DISC_HOST_TF_VERSION="${DISC_HOST_TF_VERSION}"

ARG PYTHON_VERSION=PYTHON3.6
ENV PYTHON_VERSION=${PYTHON_VERSION}

ENV DEBIAN_FRONTEND noninteractive

ARG ENABLE_FIND_FASTEST_APT_SOURCE=ON
ENV ENABLE_FIND_FASTEST_APT_SOURCE=${ENABLE_FIND_FASTEST_APT_SOURCE}


COPY docker/scripts /install/scripts

RUN bash /install/scripts/find-fastest-apt.sh
RUN apt-get install -y git curl vim libssl-dev wget unzip openjdk-11-jdk
RUN bash /install/scripts/install-cmake.sh
RUN bash /install/scripts/install-bazel.sh
RUN bash /install/scripts/install-python.sh

ARG DEVICE=cu110
RUN bash /install/scripts/install-tensorrt.sh
RUN bash /install/scripts/install-cudnn.sh

ENV PATH="/opt/cmake/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/TensorRT/lib/:/usr/local/cuda/lib64/:${LD_LIBRARY_PATH}"
