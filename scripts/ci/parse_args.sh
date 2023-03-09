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



function help() {
  echo "disc.sh --cpu-only --venv /opt/venv_py3_tf115"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu-only)
      CPU_ONLY="--cpu_only"
      shift
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --target_cpu_arch)
      TARGET_CPU_ARCH="--target_cpu_arch $2"
      shift 2
      ;;
    --rocm)
      ROCM="--rocm"
      shift
      ;;
    --dcu)
      DCU="--dcu"
      shift
      ;;
    --rocm_path)
      ROCM_PATH="--rocm_path $2"
      shift 2
      ;;
    -h)
      help
      exit
      ;;
    *)
      echo "empty"
      shift
      ;;
  esac
done
