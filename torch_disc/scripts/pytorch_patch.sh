#!/bin/bash
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


TS_BACKEND_DIR=pytorch/lazy_tensor_core/lazy_tensor_core/csrc/ts_backend
grep -rnw ${TS_BACKEND_DIR} -e "LazyNativeFunctions.h" | awk -F':' '{print $1}' | xargs sed -i 's/.*LazyNativeFunctions.h.*/#include <lazy_tensor_core\/csrc\/ts_backend\/LazyNativeFunctions.h>/g'
grep -rnw ${TS_BACKEND_DIR} -e "LazyLazyIr.h" | awk -F':' '{print $1}' | xargs sed -i 's/.*LazyLazyIr.h.*/#include <lazy_tensor_core\/csrc\/ts_backend\/LazyLazyIr.h>/g'
grep -rnw ${TS_BACKEND_DIR} -e "ts_node.h" | awk -F':' '{print $1}' | xargs sed -i 's/.*ts_node.h.*/#include <lazy\/ts_backend\/ts_node.h>/g'
grep -rnw ${TS_BACKEND_DIR} -e "shape_inference.h" | awk -F':' '{print $1}' | xargs sed -i 's/.*shape_inference.h.*/#include <lazy\/core\/shape_inference.h>/g'