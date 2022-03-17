// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __TENSORRT_PYBIND_FUNCS_H__
#define __TENSORRT_PYBIND_FUNCS_H__

#include <pybind11/pybind11.h>
namespace py = pybind11;
namespace tf_blade {
namespace trt {

void initTensorRTBindings(py::module& m);
}  // namespace trt
}  // namespace tf_blade
#endif  //__TENSORRT_PYBIND_FUNCS_H__
