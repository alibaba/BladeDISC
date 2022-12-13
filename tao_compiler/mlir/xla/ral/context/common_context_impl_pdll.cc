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

template <typename T, int N>
MemRefType<T, N> simple_test_fused_add_mul_kernel(ExecutionContext* ctx,
                                                  void* /* streamHandle */,
                                                  MemRefType<T, N> A,
                                                  MemRefType<T, N> B,
                                                  void* customAttrs) {
  auto attr =
      getOrParsePDLAttr(ctx, customAttrs, "simple_test_fused_add_mul_kernel");
  if (!attr) {
    ctx->signalError(Context::FAILURE, "fail to parse custom_attrs\n");
  }

  auto& dictAttr = attr->as<DictPDLAttr>();
  std::string name = dictAttr.get("name").template as<StrPDLAttr>().getValue();
  TAO_CHECK(name == "disc.custom_call.test.tf_fused_add_mul");
  bool trueBoolAttr =
      dictAttr.get("trueBoolAttr").template as<BoolPDLAttr>().getValue();
  TAO_CHECK(trueBoolAttr);
  bool falseBoolAttr =
      dictAttr.get("falseBoolAttr").template as<BoolPDLAttr>().getValue();
  TAO_CHECK(!falseBoolAttr);
  int64_t int127 = dictAttr.get("int127").template as<IntPDLAttr>().getValue();
  TAO_CHECK(int127 == 127);
  int64_t intNegative123456 =
      dictAttr.get("intNegative123456").template as<IntPDLAttr>().getValue();
  TAO_CHECK(intNegative123456 == -123456);
  double float0_001 =
      dictAttr.get("float0_001").template as<FloatPDLAttr>().getValue();
  // Note that `0.001` is safe to check exact match even for float type.
  TAO_CHECK(float0_001 == 0.001);

  auto& int64ArrayAttr = dictAttr.get("intArray3int128intNegative32int0")
                             .template as<IntArrayPDLAttr>();
  TAO_CHECK(int64ArrayAttr.size() == 3);
  TAO_CHECK(int64ArrayAttr.get(0) == 128);
  TAO_CHECK(int64ArrayAttr.get(1) == -32);
  TAO_CHECK(int64ArrayAttr.get(2) == 0);

  auto& strArrayAttr =
      dictAttr.get("array2strtest0strtest1").template as<ArrayPDLAttr>();
  TAO_CHECK(strArrayAttr.size() == 2);
  TAO_CHECK(strArrayAttr.get(0).template as<StrPDLAttr>().getValue() ==
            "test0");
  TAO_CHECK(strArrayAttr.get(1).template as<StrPDLAttr>().getValue() ==
            "test1");

  auto& floatArrayAttr =
      dictAttr.get("array3float0_001float23_0floatNegative0_5")
          .template as<ArrayPDLAttr>();
  TAO_CHECK(floatArrayAttr.size() == 3);
  TAO_CHECK(floatArrayAttr.get(0).template as<FloatPDLAttr>().getValue() ==
            0.001);
  TAO_CHECK(floatArrayAttr.get(1).template as<FloatPDLAttr>().getValue() ==
            23.0);
  TAO_CHECK(floatArrayAttr.get(2).template as<FloatPDLAttr>().getValue() ==
            -0.5);

  auto& mixedArrayAttr =
      dictAttr.get("array5intNegative5floatNegative0_5strtestint30float0_5")
          .template as<ArrayPDLAttr>();
  TAO_CHECK(mixedArrayAttr.size() == 5);
  TAO_CHECK(mixedArrayAttr.get(0).template as<IntPDLAttr>().getValue() == -5);
  TAO_CHECK(mixedArrayAttr.get(1).template as<FloatPDLAttr>().getValue() ==
            -0.5);
  TAO_CHECK(mixedArrayAttr.get(2).template as<StrPDLAttr>().getValue() ==
            "test");
  TAO_CHECK(mixedArrayAttr.get(3).template as<IntPDLAttr>().getValue() == 30);
  TAO_CHECK(mixedArrayAttr.get(4).template as<FloatPDLAttr>().getValue() ==
            0.5);

  auto& rank0I64DenseAttr =
      dictAttr.get("rank0I64DenseAttr").template as<DenseElementsPDLAttr>();
  TAO_CHECK(rank0I64DenseAttr.getElementType() == "int");
  TAO_CHECK(rank0I64DenseAttr.getNumBits() == 64);
  TAO_CHECK(rank0I64DenseAttr.getShape().size() == 0);
  TAO_CHECK(rank0I64DenseAttr.getNumElements() == 1);
  TAO_CHECK(*rank0I64DenseAttr.getValue<int64_t>() == 1);

  auto& rank1UI8DenseAttr =
      dictAttr.get("rank1UI8DenseAttr").template as<DenseElementsPDLAttr>();
  TAO_CHECK(rank1UI8DenseAttr.getElementType() == "uint");
  TAO_CHECK(rank1UI8DenseAttr.getNumBits() == 8);
  TAO_CHECK(rank1UI8DenseAttr.getShape().size() == 1);
  TAO_CHECK(rank1UI8DenseAttr.getShape()[0] == 1);
  TAO_CHECK(rank1UI8DenseAttr.getNumElements() == 1);
  TAO_CHECK(*rank1UI8DenseAttr.getValue<uint8_t>() == 1);

  auto& rank2Shape2x3SplatBoolDenseAttr =
      dictAttr.get("rank2Shape2x3SplatBoolDenseAttr")
          .template as<DenseElementsPDLAttr>();
  TAO_CHECK(rank2Shape2x3SplatBoolDenseAttr.getElementType() == "int");
  TAO_CHECK(rank2Shape2x3SplatBoolDenseAttr.getNumBits() == 8);
  TAO_CHECK(rank2Shape2x3SplatBoolDenseAttr.getShape().size() == 2);
  TAO_CHECK(rank2Shape2x3SplatBoolDenseAttr.getShape()[0] == 2);
  TAO_CHECK(rank2Shape2x3SplatBoolDenseAttr.getShape()[1] == 3);
  TAO_CHECK(rank2Shape2x3SplatBoolDenseAttr.getNumElements() == 6);
  for (int i = 0; i < rank2Shape2x3SplatBoolDenseAttr.getNumElements(); ++i) {
    TAO_CHECK(rank2Shape2x3SplatBoolDenseAttr.getValue<int8_t>()[i] == 1);
  }

  auto& rank2Shape2x3SplatFloatDenseAttr =
      dictAttr.get("rank2Shape2x3SplatFloatDenseAttr")
          .template as<DenseElementsPDLAttr>();
  TAO_CHECK(rank2Shape2x3SplatFloatDenseAttr.getElementType() == "float");
  TAO_CHECK(rank2Shape2x3SplatFloatDenseAttr.getNumBits() == 32);
  TAO_CHECK(rank2Shape2x3SplatFloatDenseAttr.getShape().size() == 2);
  TAO_CHECK(rank2Shape2x3SplatFloatDenseAttr.getShape()[0] == 2);
  TAO_CHECK(rank2Shape2x3SplatFloatDenseAttr.getShape()[1] == 3);
  TAO_CHECK(rank2Shape2x3SplatFloatDenseAttr.getNumElements() == 6);
  for (int i = 0; i < rank2Shape2x3SplatFloatDenseAttr.getNumElements(); ++i) {
    TAO_CHECK(rank2Shape2x3SplatFloatDenseAttr.getValue<float>()[i] == -0.01f);
  }

  TAO_VLOG(0) << "custom_attr {\n"
              << "\tname = " << name << "\n"
              << "\ttrueBoolAttr = " << trueBoolAttr << "\n"
              << "\tfalseBoolAttr = " << falseBoolAttr << "\n"
              << "\tint127 = " << int127 << "\n"
              << "\tintNegative123456 = " << intNegative123456 << "\n"
              << "\tfloat0_001 = " << float0_001 << "\n"
              << "}\n";

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
