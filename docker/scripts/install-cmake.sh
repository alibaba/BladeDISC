#!/bin/bash
set -ex
wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.sh \
  -q -O /tmp/cmake-install.sh && \
  chmod u+x /tmp/cmake-install.sh && \
  mkdir -p /opt/cmake && \
  /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake && \
  rm /tmp/cmake-install.sh


