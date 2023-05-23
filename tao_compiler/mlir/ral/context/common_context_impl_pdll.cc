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

#include "mlir/ral/context/context_util.h"
#include "mlir/ral/context/pdll_util.h"
#include "mlir/ral/device/cpu/cpu_driver.h"
#include "mlir/ral/ral_base.h"
#include "mlir/ral/ral_helper.h"

#if defined(TAO_CPU_ONLY)
#include "mlir/ral/context/common_context_impl_quantization.h"
#endif

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

#if defined(TAO_CPU_ONLY)
  std::string data_format_nhwc =
      dictAttr.get("data_format_nhwc").template as<StrPDLAttr>().getValue();
  TAO_CHECK(data_format_nhwc == "NHWC");
  std::string data_format_nchw =
      dictAttr.get("data_format_nchw").template as<StrPDLAttr>().getValue();
  TAO_CHECK(data_format_nchw == "NCHW");

  std::vector<int32_t> reorderDims(4, 0);
  (void)generateReorderDims<4>(data_format_nchw, reorderDims);
  TAO_CHECK(reorderDims[0] == 2);
  TAO_CHECK(reorderDims[1] == 3);
  TAO_CHECK(reorderDims[2] == 0);
  TAO_CHECK(reorderDims[3] == 1);
  (void)generateReorderDims<4>(data_format_nhwc, reorderDims);
  TAO_CHECK(reorderDims[0] == 1);
  TAO_CHECK(reorderDims[1] == 2);
  TAO_CHECK(reorderDims[2] == 0);
  TAO_CHECK(reorderDims[3] == 3);

  std::string padding_same =
      dictAttr.get("padding_same").template as<StrPDLAttr>().getValue();
  TAO_CHECK(padding_same == "SAME");
  std::string padding_valid =
      dictAttr.get("padding_valid").template as<StrPDLAttr>().getValue();
  TAO_CHECK(padding_valid == "VALID");

  auto& strides1 =
      dictAttr.get("strides1").template as<IntArrayPDLAttr>().getValue();
  auto& strides2 =
      dictAttr.get("strides22").template as<IntArrayPDLAttr>().getValue();
  auto& strides3 =
      dictAttr.get("strides1231").template as<IntArrayPDLAttr>().getValue();
  std::vector<int32_t> strides(2, 0);
  std::vector<std::vector<int64_t>> stridesArrs = {strides1, strides2,
                                                   strides3};
  std::vector<std::vector<int32_t>> stridesResults = {{1, 1}, {2, 2}, {2, 3}};
  for (int i = 0; i < 3; ++i) {
    (void)generateStrides<4>(reorderDims, stridesArrs[i], strides);
    TAO_CHECK(strides[0] == stridesResults[i][0]);
    TAO_CHECK(strides[1] == stridesResults[i][1]);
  }

  auto& dilations1 =
      dictAttr.get("dilations1").template as<IntArrayPDLAttr>().getValue();
  auto& dilations2 =
      dictAttr.get("dilations22").template as<IntArrayPDLAttr>().getValue();
  auto& dilations3 =
      dictAttr.get("dilations1231").template as<IntArrayPDLAttr>().getValue();
  std::vector<int32_t> dilations(2, 0);
  std::vector<std::vector<int64_t>> dilationsArrs = {dilations1, dilations2,
                                                     dilations3};
  std::vector<std::vector<int32_t>> dilationsResults = {{1, 1}, {2, 2}, {2, 3}};
  for (int i = 0; i < 3; ++i) {
    (void)generateDilations<4>(reorderDims, dilationsArrs[i], dilations);
    TAO_CHECK(dilations[0] == dilationsResults[i][0]);
    TAO_CHECK(dilations[1] == dilationsResults[i][1]);
  }

  // generate input with shape: [4, 16, 16, 25] in "NHWC" format
  // and weight with shape: [8, 3, 3, 25] in "OHWI" format
  MemRefType<int8_t, 4> input;
  auto inputSizes = std::initializer_list<int64_t>({4, 16, 16, 25});
  std::copy(inputSizes.begin(), inputSizes.end(), input.sizes);
  MemRefType<int8_t, 4> weight;
  auto weightSizes = std::initializer_list<int64_t>({8, 3, 3, 25});
  std::copy(weightSizes.begin(), weightSizes.end(), weight.sizes);

  MemRefType<int32_t, 1> metadata;
  metadata.sizes[0] = 17;  // 4 * 3 + (4 - 2) * 2 + 1
  int32_t metadataData[metadata.sizes[0]] = {0};
  metadata.data = metadataData;
  // input: NHWC
  // kernel: OHWI
  int32_t metadataResultsData[9][17] = {
      {0, 3, 1, 2, 3, 0, 1, 2, 0, 3, 1, 2, 1, 1, 1, 1, 1},
      {0, 3, 1, 2, 3, 0, 1, 2, 0, 3, 1, 2, 1, 1, 2, 2, 1},
      {0, 3, 1, 2, 3, 0, 1, 2, 0, 3, 1, 2, 1, 1, 2, 3, 1},
      {0, 3, 1, 2, 3, 0, 1, 2, 0, 3, 1, 2, 2, 2, 1, 1, 1},
      {0, 3, 1, 2, 3, 0, 1, 2, 0, 3, 1, 2, 2, 2, 2, 2, 1},
      {0, 3, 1, 2, 3, 0, 1, 2, 0, 3, 1, 2, 2, 2, 2, 3, 1},
      {0, 3, 1, 2, 3, 0, 1, 2, 0, 3, 1, 2, 2, 3, 1, 1, 1},
      {0, 3, 1, 2, 3, 0, 1, 2, 0, 3, 1, 2, 2, 3, 2, 2, 1},
      {0, 3, 1, 2, 3, 0, 1, 2, 0, 3, 1, 2, 2, 3, 2, 3, 1}};
  std::vector<MemRefType<int32_t, 1>> metadataResults(9);
  for (int i = 0; i < 9; ++i)
    metadataResults[i].data = &metadataResultsData[i][0];
  int index = 0;
  for (auto strides : stridesResults) {
    for (auto dilations : dilationsResults) {
      (void)generateMetadata<4>(strides, dilations, reorderDims, true,
                                metadata);
      for (int i = 0; i < metadata.sizes[0]; ++i)
        TAO_CHECK(metadata.data[i] == metadataResults[index].data[i]);
      index++;
    }
  }
  // check output shape with following variable:
  // input: 4x16x16x25 (NHWC)
  // kernel: 8x3x3x25 (OHWI)
  // padding: "same" and "valid"
  // strides: [1, 1], [2, 2], [2, 3]
  // dilations: [1, 1], [2, 2], [2, 3]
  MemRefType<int32_t, 1> padding;
  padding.sizes[0] = 4;  // 2 * 2
  int32_t paddingData[padding.sizes[0]] = {0};
  padding.data = paddingData;
  int32_t paddingResultsData[18][4] = {
      {1, 1, 1, 1}, {0, 1, 0, 1}, {0, 1, 1, 1}, {2, 2, 2, 2}, {1, 2, 1, 2},
      {1, 2, 2, 2}, {2, 2, 3, 3}, {1, 2, 2, 3}, {1, 2, 3, 3}, {0, 0, 0, 0},
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  int64_t resultCheckSizes[18][4] = {
      {4, 16, 16, 8}, {4, 8, 8, 8},   {4, 8, 6, 8},   {4, 16, 16, 8},
      {4, 8, 8, 8},   {4, 8, 6, 8},   {4, 16, 16, 8}, {4, 8, 8, 8},
      {4, 8, 6, 8},   {4, 14, 14, 8}, {4, 7, 7, 8},   {4, 7, 5, 8},
      {4, 12, 12, 8}, {4, 6, 6, 8},   {4, 6, 4, 8},   {4, 12, 10, 8},
      {4, 6, 5, 8},   {4, 6, 4, 8}};
  std::vector<int64_t> resultSizes(4, 0);
  index = 0;
  for (auto padding_str : {"SAME", "VALID"}) {
    for (auto dilations : dilationsResults) {
      for (auto strides : stridesResults) {
        (void)generatePadding<4>(input, weight, padding_str, reorderDims,
                                 strides, dilations, padding);
        for (int i = 0; i < 2; ++i) {
          TAO_CHECK(padding.data[2 * i] == paddingResultsData[index][2 * i]);
          TAO_CHECK(padding.data[2 * i + 1] ==
                    paddingResultsData[index][2 * i + 1]);
        }
        generateResultShape<4>(input, weight, reorderDims, padding, strides,
                               dilations, resultSizes);
        for (int i = 0; i < 4; ++i)
          TAO_CHECK(resultSizes[i] = resultCheckSizes[index][i]);
        index++;
      }
    }
  }

  // check output shape with following variable:
  // input: 4x16x16x25
  // kernel: 3x3x25x8
  // explicit_padding : [0, 0, 1, 2, 3, 4, 0, 0]
  // strides: [1, 1], [2, 2], [2, 3]
  // dilations: [1, 1], [2, 2], [2, 3]
  int resultCheckSizes_1[9][4] = {{4, 17, 21, 8}, {4, 9, 11, 8}, {4, 9, 7, 8},
                                  {4, 15, 19, 8}, {4, 8, 10, 8}, {4, 8, 7, 8},
                                  {4, 15, 17, 8}, {4, 8, 9, 8},  {4, 8, 6, 8}};
  auto& explicit_paddings = dictAttr.get("explicit_paddings")
                                .template as<IntArrayPDLAttr>()
                                .getValue();
  (void)generateExplicitPaddings<4>(reorderDims, explicit_paddings, padding);
  index = 0;
  for (auto dilations : dilationsResults) {
    for (auto strides : stridesResults) {
      generateResultShape<4>(input, weight, reorderDims, padding, strides,
                             dilations, resultSizes);
      for (int i = 0; i < 4; ++i)
        TAO_CHECK(resultSizes[i] == resultCheckSizes_1[index][i]);
      index++;
    }
  }
#endif

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
