# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex
if [ -z "$CPU_ONLY"]; then
  # install disc python wheel
  ${VENV_PATH}/bin/python -m pip install ./build/blade_disc*.whl

  pushd examples/TensorFlow/Inference/X86/BERT
  bash download_model.sh
  TAO_ENABLE_FALLBACK=false ${VENV_PATH}/bin/python main.py
  # clean up download files
  rm -rf model
  popd
fi