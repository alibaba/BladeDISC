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

#include <c10/core/Device.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir_dump_util.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/multi_wait.h>
#include <torch/csrc/lazy/core/tensor_impl.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/thread_pool.h>
#include <torch/csrc/lazy/core/util.h>
#include <torch/csrc/lazy/python/python_util.h>
#include <torch/csrc/utils/cuda_lazy_init.h>

#include <vector>

// mhlo conversion
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>

#include "compiler/mlir/converters/mhlo_conversion.h"
#include "lazy_tensor_core/csrc/ts_backend/backend_impl.h"

namespace torch_disc {
namespace {

torch::lazy::BackendDevice GetDeviceOrCurrent(const std::string& device_str) {
  if (device_str.empty()) {
    return torch::lazy::BackendDevice();
  }
  return torch::lazy::atenDeviceToBackendDevice(c10::Device(device_str));
}

std::string GetLazyTensorsDump(
    const std::vector<torch::lazy::LazyTensor>& tensors,
    const std::function<std::string(c10::ArrayRef<torch::lazy::Node*>)>&
        coverter) {
  std::vector<torch::lazy::Node*> nodes;
  std::vector<torch::lazy::Value> values;
  for (auto& tensor : tensors) {
    values.push_back(tensor.GetIrValue());
    nodes.push_back(values.back().node.get());
  }
  return coverter(nodes);
}

std::string GetBackendGraph() {
  auto device = GetDeviceOrCurrent("");
  auto tensors = torch::lazy::LazyGraphExecutor::Get()->GetLiveTensors(&device);
  return torch::lazy::LazyGraphExecutor::Get()->DumpBackendComputation(tensors);
}

void InitLtcModuleBindings(py::module m) {
  m.def("_ltc_init_disc_backend",
        []() { torch_lazy_tensors::compiler::InitTorchScriptBackend(); });
  m.def("_disc_backend_graph", &GetBackendGraph);
  m.def("_ltc_dump_graph", []() {
    auto device = GetDeviceOrCurrent("");
    auto tensors =
        torch::lazy::LazyGraphExecutor::Get()->GetLiveTensors(&device);
    auto coverter = [](c10::ArrayRef<torch::lazy::Node*> nodes) {
      return torch::lazy::DumpUtil::ToDot(nodes);
    };
    return GetLazyTensorsDump(tensors, coverter);
  });
}
void InitLtcBindings(py::module m) { InitLtcModuleBindings(m); }
}  // namespace

}  //  namespace torch_disc

PYBIND11_MODULE(_torch_disc, m) { torch_disc::InitLtcBindings(m); }
