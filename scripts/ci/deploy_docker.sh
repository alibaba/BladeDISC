#!/bin/bash
# Copyright 2021 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -e

ALIYUN_DOCKER_ORG=bladedisc
ALIYUN_DOCKER_DOMAIN=registry.cn-shanghai.aliyuncs.com
OFFICIAL_DOCKER_ORG=bladedisc

echo "REMOTE_DEV_DOCKER:" ${REMOTE_DEV_DOCKER}
echo "REMOTE_RUNTIME_DOCKER:" ${REMOTE_RUNTIME_DOCKER}
echo "RUNTIME_BASEIMAGE": ${RUNTIME_BASEIMAGE}
echo "GITHUB_PULL_REQUEST": ${GITHUB_PULL_REQUEST}

function push_image() {
  set -x
  local_tag=$1
  remote_tag=$2
  aliyun_tag=${ALIYUN_DOCKER_DOMAIN}/${ALIYUN_DOCKER_ORG}/${remote_tag}
  official_tag=${OFFICIAL_DOCKER_ORG}/${remote_tag}
  docker tag ${local_tag} ${aliyun_tag}
  docker tag ${local_tag} ${official_tag}
  docker push ${official_tag}
  docker push ${aliyun_tag}
  set +x
}

function docker_login() {
  echo "$DOCKER_PASSWORD" |
      docker login --username "$DOCKER_USERNAME" --password-stdin

  echo "$ALIYUN_DOCKER_PASSWORD" |
      docker login --username "$ALIYUN_DOCKER_USERNAME" --password-stdin ${ALIYUN_DOCKER_DOMAIN}
}

function push_images() {
  docker_login
  push_image ${LOCAL_DEV_DOCKER} ${REMOTE_DEV_DOCKER}
  push_image ${REMOTE_RUNTIME_DOCKER} ${REMOTE_RUNTIME_DOCKER}
}