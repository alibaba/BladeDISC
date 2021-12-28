#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source ${SCRIPT_DIR}/deploy_docker.sh

# build runtime Docker
docker build -t ${REMOTE_RUNTIME_DOCKER} -f docker/runtime/Dockerfile.tf \
  --build-arg BASEIMAGE=${RUNTIME_BASEIMAGE} .

push_images