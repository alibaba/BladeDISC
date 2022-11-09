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

#include "tensorflow/compiler/mlir/xla/ral/context/common_context_impl.h"
#include "tensorflow/compiler/mlir/xla/ral/context/context_util.h"
#include "tensorflow/compiler/mlir/xla/ral/context/pdll_util.h"
#include "tensorflow/compiler/mlir/xla/ral/device/cpu/cpu_driver.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_base.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_helper.h"

//===----------------------------------------------------------------------===//
// Test related kernel
//===----------------------------------------------------------------------===//

namespace tao {
namespace ral {

namespace {

template <typename T>
struct CustomAttrState : public Context::Resource {
  std::mutex mu;
  std::unordered_map<opaque_t, std::unique_ptr<T>> customAttrMap;
};

template <typename T>
T* getOrParseCustomAttr(ExecutionContext* ctx, opaque_t attrPtr,
                        const std::string& name,
                        std::function<std::unique_ptr<T>()> creator) {
  using StateTy = CustomAttrState<T>;
  auto state =
      ctx->getOrCreateResource<StateTy>(name, [&]() { return new StateTy; });
  std::lock_guard<std::mutex> l(state->mu);
  auto it = state->customAttrMap.find(attrPtr);
  if (it == state->customAttrMap.end()) {
    auto value = creator();
    if (!value) return nullptr;
    it = state->customAttrMap.emplace(attrPtr, std::move(value)).first;
  }
  return it->second.get();
}

template <typename T, int N>
MemRefType<T, N> simple_test_fused_add_mul_kernel(ExecutionContext* ctx,
                                                  void* /* streamHandle */,
                                                  MemRefType<T, N> A,
                                                  MemRefType<T, N> B,
                                                  void* customAttrs) {
  auto creator = [&]() {
    uint8_t* buffer = (uint8_t*)customAttrs;
    return parsePDLAttr(buffer);
  };
  auto attr = getOrParseCustomAttr<PDLAttr>(
      ctx, customAttrs, "simple_test_fused_add_mul_kernel", creator);
  if (!attr) {
    ctx->signalError(Context::FAILURE, "fail to parse custom_attrs\n");
  }

  TAO_VLOG(0) << "custom_attr { name = \""
              << attr->template as<DictPDLAttr>()
                     .get("name")
                     .template as<StrPDLAttr>()
                     .getValue()
              << "\" }";

  size_t nElems = Size(A);
  auto driver = ctx->getDriver<cpu::CPUDriver>(cpu::CPUDriver::name());
  auto data = static_cast<T*>(driver->alloc(ctx, nElems * sizeof(T)));
  auto out = assignMemRef<T, N>(data, A.sizes);

  for (size_t i = 0; i < nElems; ++i) {
    data[i] = (A.data[i] + B.data[i]) * (A.data[i] + B.data[i]);
  }

  return out;
}

template <typename T, int N>
std::tuple<MemRefType<T, N>, MemRefType<T, N>>
simple_test_fused_add_mul_kernel_multi_results(ExecutionContext* ctx,
                                               void* /* streamHandle */,
                                               MemRefType<T, N> A,
                                               MemRefType<T, N> B,
                                               void* customAttrs) {
  size_t nElems = Size(A);
  auto driver = ctx->getDriver<cpu::CPUDriver>(cpu::CPUDriver::name());
  auto data0 = static_cast<T*>(driver->alloc(ctx, nElems * sizeof(T)));
  auto data1 = static_cast<T*>(driver->alloc(ctx, nElems * sizeof(T)));
  auto out0 = assignMemRef<T, N>(data0, A.sizes);
  auto out1 = assignMemRef<T, N>(data1, A.sizes);

  for (size_t i = 0; i < nElems; ++i) {
    data0[i] = A.data[i] + B.data[i];
    data1[i] = data0[i] * data0[i];
  }

  return std::make_tuple(out0, out1);
}

}  // namespace

TAO_RAL_API("disc.custom_call.test.tf_fused_add_mul", "cpu",
            simple_test_fused_add_mul_kernel<float, 2>);
TAO_RAL_API("disc.custom_call.test.tf_fused_add_mul_multi_results", "cpu",
            simple_test_fused_add_mul_kernel_multi_results<float, 2>);

}  // namespace ral
}  // namespace tao