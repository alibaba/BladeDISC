// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file    except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pytorch_blade/ltc/disc_backend/backend_impl.h"

#include <ATen/Functions.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/cache.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/ir_dump_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/ts_backend/ir_builder.h>
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include "pytorch_blade/common_utils/logging.h"
#include "pytorch_blade/common_utils/utils.h"
#include "pytorch_blade/ltc/disc_compiler/disc_compiler.h"

namespace torch_disc {
namespace compiler {

struct CachedExecutable {
  explicit CachedExecutable(ExecutablePtr executable)
      : executable(std::move(executable)) {}

  ExecutablePtr executable;
};

using DiscComputationCache = torch::lazy::
    Cache<torch::lazy::hash_t, CachedExecutable, torch::lazy::HashReducer>;

torch::lazy::BackendImplInterface* GetTSBackendImpl();

using BackendDeviceType = torch::lazy::BackendDeviceType;
using TSData = torch::lazy::TSData;
using IrBuilder = torch::lazy::IrBuilder;

struct TSBackendDeviceType : public BackendDeviceType {
  TSBackendDeviceType() = delete;
  TSBackendDeviceType(c10::DeviceType deviceType) {
    TORCH_CHECK(
        supported_device_types_.find((int8_t)deviceType) !=
        supported_device_types_.end());
    type = (int8_t)deviceType;
  }

  std::string toString() const override {
    return c10::DeviceTypeName((c10::DeviceType)type);
  }

  c10::DeviceType c10Type() const {
    return (c10::DeviceType)type;
  }

 private:
  static const std::set<int8_t> supported_device_types_;
};
const std::set<int8_t> TSBackendDeviceType::supported_device_types_ = {
    (int8_t)at::kCPU,
    (int8_t)at::kCUDA};

class DISCBackendImpl : public torch::lazy::BackendImplInterface {
 public:
  DISCBackendImpl() : default_device_type_(at::kCPU) {
    bool env_use_cuda =
        torch::blade::env::ReadBoolFromEnvVar("LTC_DISC_CUDA", false);
    auto type = env_use_cuda ? at::kCUDA : at::kCPU;

    default_device_type_ = TSBackendDeviceType(type);
    cache_ = std::make_shared<DiscComputationCache>(
        FLAGS_torch_lazy_compilation_cache_size);
  }

  const IrBuilder* GetIrBuilder() const override {
    static const IrBuilder* builder = new torch::lazy::TorchScriptIrBuilder();
    return builder;
  }

  std::unique_ptr<torch::lazy::LoweringContext> CreateLoweringContext(
      const std::string& name,
      torch::lazy::BackendDevice device,
      c10::ArrayRef<torch::lazy::Node*> post_order,
      torch::lazy::Util::EmissionMap emit_status) const override {
    return std::make_unique<torch::lazy::TSLoweringContext>(
        name, device, post_order, emit_status);
  }

  std::unique_ptr<torch::lazy::LoweringContext> CreateLoweringContext(
      const std::string& name,
      torch::lazy::BackendDevice device) const override {
    return std::make_unique<torch::lazy::TSLoweringContext>(name, device);
  }

  std::vector<std::string> GetCompilationDevices(
      const std::string& device,
      c10::ArrayRef<std::string> devices) const override {
    return std::vector<std::string>(devices.begin(), devices.end());
  }

  at::Tensor MakeTensorFromComputationData(
      const torch::lazy::BackendDataPtr data,
      c10::optional<at::ScalarType> logical_scalar_type) const override {
    const auto ts_data = std::static_pointer_cast<TSData>(data);
    return ts_data->data();
  }

  torch::lazy::BackendDataPtr MakeComputationDataFromTensor(
      const at::Tensor& tensor,
      const torch::lazy::Shape& shape,
      const torch::lazy::BackendDevice& device) const override {
    at::TensorOptions options = tensor.options().device(
        default_device_type_.c10Type(), device.ordinal());
    if (tensor.device().type() == default_device_type_.c10Type() &&
        default_device_type_.c10Type() == at::kCUDA) {
      return std::make_shared<TSData>(
          tensor.to(options, /*non_blocking=*/true), shape, device);
    } else if (tensor.device().type() == at::kCPU && tensor.numel() == 1) {
      // calling .item() on singleton cpu tensor is fast, and using fill is a
      // safe, async way to copy cpu to cuda for a single value
      auto device_tensor = at::full(tensor.sizes(), tensor.item(), options);
      return std::make_shared<TSData>(device_tensor, shape, device);
    } else {
      return std::make_shared<TSData>(
          tensor.to(options, /*non_blocking=*/false), shape, device);
    }
  }

  torch::lazy::BackendDataPtr MakeComputationDataFromScalar(
      const at::Scalar& scalar,
      const torch::lazy::BackendDevice& device) const override {
    return std::make_shared<TSData>(scalar, device);
  }

  torch::lazy::BackendDataPtr GetComputationDataFromNode(
      torch::lazy::Node* node) const {
    auto* device_data_node = dynamic_cast<torch::lazy::DeviceData*>(node);
    if (!device_data_node) {
      return nullptr;
    }
    return device_data_node->data();
  }

  std::string GetComputationBackendText(
      const torch::lazy::ComputationPtr computation) const override {
    auto ts_computation =
        static_cast<torch::lazy::TSComputation*>(computation.get());
    return ts_computation->graph()->toString();
  }

  //////////////computation client interfaces///////////////////////

 public:
  torch::lazy::BackendDataPtr CreateDataPlaceholder(
      const torch::lazy::BackendDevice& device,
      const torch::lazy::Shape& shape) const override;

  std::vector<torch::lazy::ComputationPtr> Compile(
      std::vector<torch::lazy::ComputationPtr> instances) const override;

  std::vector<torch::lazy::BackendDataPtr> ExecuteComputation(
      torch::lazy::Computation& computation,
      c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
      const torch::lazy::BackendDevice& device) const override;

  std::shared_ptr<torch::lazy::BackendDeviceType> GetDefaultDeviceType()
      const override {
    return std::make_shared<BackendDeviceType>(default_device_type_);
  }

  at::DeviceType EagerFallbackDeviceType() const override;

  void SetDefaultDeviceType(std::string type) override {
    default_device_type_ = TSBackendDeviceType(c10::Device(type).type());
    // The first CUDA usage could happen via lazy tensors. Initialize CUDA here
    // to account for that, at::scalar_tensor constructor triggers everything we
    // need.
    static auto init_cuda = default_device_type_.c10Type() == at::kCUDA
        ? c10::optional<at::Tensor>(
              at::scalar_tensor(0, at::TensorOptions().device(at::kCUDA)))
        : c10::nullopt;
  }

  std::vector<torch::lazy::BackendDevice> GetBackendDevices() const override;

  torch::lazy::BackendDevice GetBackendDevice(
      c10::Device device) const override;

  void SetRngSeed(size_t seed) const override {
    LOG(FATAL) << "Not implemented yet.";
  }

  // std::map<std::string, Metric> GetMetrics() const override { return {}; }

  // MemoryInfo GetMemoryInfo(const std::string& device) override {
  //   LOG(FATAL) << "Not implemented yet.";
  // }

  void PrepareToExit() const override;

 private:
  TSBackendDeviceType default_device_type_;
  // std::unordered_map<torch::lazy::TSComputation*, ExecutablePtr> cache_;
  std::shared_ptr<DiscComputationCache> cache_;
  // ComputationCache* cache =
  //    new ComputationCache(FLAGS_torch_lazy_compilation_cache_size);
};

torch::lazy::BackendDataPtr DISCBackendImpl::CreateDataPlaceholder(
    const torch::lazy::BackendDevice& device,
    const torch::lazy::Shape& shape) const {
  return std::make_shared<TSData>(shape, device);
}

std::vector<torch::lazy::ComputationPtr> DISCBackendImpl::Compile(
    std::vector<torch::lazy::ComputationPtr> instances) const {
  for (const auto& instance : instances) {
    auto ts_computation =
        static_cast<torch::lazy::TSComputation*>(instance.get());
  }
  return instances;
}

std::vector<torch::lazy::BackendDataPtr> DISCBackendImpl::ExecuteComputation(
    torch::lazy::Computation& computation,
    c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
    const torch::lazy::BackendDevice& device) const {
  auto ts_computation = static_cast<torch::lazy::TSComputation&>(computation);
  bool default_device_is_cuda =
      ((c10::DeviceType)default_device_type_.type == at::kCUDA);

  // Note: hasing graph is just solution for PoC, that supports each LazyTensor
  // topology mapping a exclusive TorchScript Graph.
  // TODO(yancey1989): Cache each Disc cluster separately
  auto disc_hash = torch::lazy::DataHash(
      ts_computation.graph().get(), sizeof(*ts_computation.graph().get()));
  if (cache_->Get(disc_hash)) {
    return cache_->Get(disc_hash)->executable->Run(
        arguments, device, default_device_is_cuda);
  }

  ExecutablePtr executable =
      CompileToDiscExecutable(ts_computation.graph()->copy(), arguments);
  auto result = executable->Run(arguments, device, default_device_is_cuda);
  auto cachedExecutable =
      std::make_shared<CachedExecutable>(std::move(executable));
  cache_->Add(disc_hash, cachedExecutable);
  return result;
}

std::vector<torch::lazy::BackendDevice> DISCBackendImpl::GetBackendDevices()
    const {
  std::vector<torch::lazy::BackendDevice> devices;
  // TODO(whc) figure out how to query available devices from pytorch
  devices.emplace_back(GetBackendDevice(c10::Device(c10::kCPU, 0)));
  devices.emplace_back(GetBackendDevice(c10::Device(c10::kCUDA, 0)));
  return devices;
}

torch::lazy::BackendDevice DISCBackendImpl::GetBackendDevice(
    c10::Device device) const {
  // Note, we ignore the device type specified by the c10::Device since it is
  // expected to be a virtual device (lazy::), but we need to change this when
  // we support lazy as a mode
  return torch::lazy::BackendDevice(GetDefaultDeviceType(), device.index());
}

void DISCBackendImpl::PrepareToExit() const {}

c10::DeviceType DISCBackendImpl::EagerFallbackDeviceType() const {
  // For TS backend, hardware device _is_ eager device
  return (c10::DeviceType)GetDefaultDeviceType()->type;
}

torch::lazy::BackendImplInterface* GetDISCBackendImpl() {
  static compiler::DISCBackendImpl* disc_backend_impl =
      new compiler::DISCBackendImpl();
  return disc_backend_impl;
}

void InitTorchScriptBackend() {
  static std::unique_ptr<torch::lazy::BackendRegistrar> s_registrar;
  VLOG(0) << "Welcome to Disc accelerator based on PyTorch LazyTensor Core!!!";
  s_registrar.reset(
      new torch::lazy::BackendRegistrar(compiler::GetDISCBackendImpl()));
}

} //  namespace compiler
} //  namespace torch_disc
