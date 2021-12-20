#!/bin/bash
set -ex
python3 -m pip install virtualenv numpy
DISC_VENV=/opt/venv_disc

function install_tf115() {
  # install virtualenv: Python3+TensorFlow-GPU 1.15
  pip install -q tensorflow-gpu==1.15
}

function install_tf24() {
  # install virtualenv: Python3+TensorFlow-GPU 2.4.0
  pip install -q tensorflow-gpu==2.4
}

function install_tf115_cpu() {
  # install virtualenv: Python3+TensorFlow 1.15
  pip install -q tensorflow==1.15
}

function install_venv() {
  python3 -m virtualenv ${DISC_VENV}
  source ${DISC_VENV}/bin/activate
  if [[ "$CUDA_VERSION" == "" ]]; then
    # CPU Docker image
    install_tf115_cpu
  elif [[ "$CUDA_VERSION" == 10.0.* ]]; then
    # CUDA10 Docker image
    install_tf115
  elif [[ "$CUDA_VERSION" == 11.0.* ]]; then
    install_tf24
    # CUDA11 Docker image
  else
    echo "unsupported CUDA version: " $CUDA_VERSION
    exit -1
  fi
  pip install pytest pytest-forked
  deactivate
}

install_venv
