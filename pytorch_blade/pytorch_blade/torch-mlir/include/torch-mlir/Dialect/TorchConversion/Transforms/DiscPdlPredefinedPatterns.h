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

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/disc/transforms/disc_pdl_utils.h"
#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#include "tests/torch-disc-pdll/utils.h"

using namespace mlir;
using namespace mlir::torch;

namespace {

// add pre-defined pdll patterns here.
std::string getTorchPredefinedPDLPatterns() {
  std::string preDefinedPatterns;
  // #if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
  preDefinedPatterns += R"pdll(
    Rewrite ConvertToF16(value: Value) -> Value {
      let f16_dtype = op<torch.constant.int> {value = attr<"5">} -> (type<"!torch.int">);
      let old_type = GetTorchTensorType(value);
      let new_type = ConvertTorchTensorElemType(old_type, attr<"\"f16\"">);
      let false_val = op<torch.constant.bool> {value = attr<"false">} -> (type<"!torch.bool">);
      let none_val = op<torch.constant.none> -> (type<"!torch.none">);
      let f16_value = op<torch.aten.to.dtype>(
        value, f16_dtype, false_val, false_val, none_val
      ) -> (new_type);
    
      return f16_value.0;
    }
    
    Rewrite ConvertToF32(value: Value) -> Value {
      let f32_dtype = op<torch.constant.int> {value = attr<"6">} -> (type<"!torch.int">);
      let old_type = GetTorchTensorType(value);
      let new_type = ConvertTorchTensorElemType(old_type, attr<"\"f32\"">);
      let false_val = op<torch.constant.bool> {value = attr<"false">} -> (type<"!torch.bool">);
      let none_val = op<torch.constant.none> -> (type<"!torch.none">);
      let f32_value = op<torch.aten.to.dtype>(
        value, f32_dtype, false_val, false_val, none_val
      ) -> (new_type);
    
      return f32_value.0;
    }

    Pattern TorchGroupNormOpF32 {
      /// match phase: define the pattern
      let eps_attr : Attr;
      let eps = op<torch.constant.float> { value = eps_attr };
      let gn = op<torch.aten.group_norm>(
        input: Value,
        num_group: Value,
        weight: Value,
        bias: Value,
        eps.0,
        cudnn_enabled: Value
      ) -> (old_type: Type);
      CheckNotTorchNone(bias);
      CheckTorchConstantInt(num_group);
      CheckTorchTensorElemType(input, attr<"\"f32\"">);
    
      /// rewrite phase
      rewrite gn with {
        let f16_input = ConvertToF16(input);
        let f16_weight = ConvertToF16(weight);
        let f16_bias = ConvertToF16(bias);
        let f16_output = ConvertToF16(gn.0);
    
        /// 1. create custom call op
        let inputs = PackValue_3(attr<"\"in\"">, f16_input, f16_weight, f16_bias);
        let outputs = PackValue_1(attr<"\"out\"">, f16_output);
        let infos = CreateTorchCustomCall(attr<"\"op\"">, inputs, outputs);
    
        /// 2. set attrs that are used by bladedisc.
        SetAttr(infos.op, attr<"\"call_target_name\"">, attr<"\"ral_pdll_group_norm\"">);
        SetAttr(infos.op, attr<"\"input_placements\"">, attr<"\"x,x,x\"">);
        SetAttr(infos.op, attr<"\"output_placements\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"device\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"input_layouts\"">, attr<"\"NCHW,*,*\"">);
        SetAttr(infos.op, attr<"\"output_layouts\"">, attr<"\"NCHW\"">);
        SetAttr(infos.op, attr<"\"expected_input_layouts\"">, attr<"\"NHWC,*,*\"">);
        SetAttr(infos.op, attr<"\"expected_output_layouts\"">, attr<"\"NHWC\"">);
    
        /// 3. set attrs that are directly passed to the custom call kernel.
        let num_group_attr = ConvertTorchConstantIntToI64Attr(num_group);
        SetCustomAttr(infos.op, attr<"\"num_group\"">, num_group_attr);
        SetCustomAttr(infos.op, attr<"\"eps\"">, eps_attr);
        SetCustomAttr(infos.op, attr<"\"silu\"">, attr<"false">);
    
        let rs = UnpackValue_1(infos.new_outputs);
        let new_output = ConvertToF32(rs);
    
        replace gn with new_output;
      };
    }

    Pattern TorchGroupNormWithSiluOpF32 {
      /// match phase: define the pattern
      let eps_attr : Attr;
      let eps = op<torch.constant.float> { value = eps_attr };
      let gn = op<torch.aten.group_norm>(
        input: Value,
        num_group: Value,
        weight: Value,
        bias: Value,
        eps.0,
        cudnn_enabled: Value
      ) -> (old_type: Type);
      let silu = op<torch.aten.silu>(gn.0);
      CheckNotTorchNone(bias);
      CheckTorchConstantInt(num_group);
      CheckTorchTensorElemType(input, attr<"\"f32\"">);
    
      /// rewrite phase
      rewrite silu with {
        let f16_input = ConvertToF16(input);
        let f16_weight = ConvertToF16(weight);
        let f16_bias = ConvertToF16(bias);
        let f16_output = ConvertToF16(silu.0);
    
        /// 1. create custom call op
        let inputs = PackValue_3(attr<"\"in\"">, f16_input, f16_weight, f16_bias);
        let outputs = PackValue_1(attr<"\"out\"">, f16_output);
        let infos = CreateTorchCustomCall(attr<"\"op\"">, inputs, outputs);
    
        /// 2. set attrs that are used by bladedisc.
        SetAttr(infos.op, attr<"\"call_target_name\"">, attr<"\"ral_pdll_group_norm\"">);
        SetAttr(infos.op, attr<"\"input_placements\"">, attr<"\"x,x,x\"">);
        SetAttr(infos.op, attr<"\"output_placements\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"device\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"input_layouts\"">, attr<"\"NCHW,*,*\"">);
        SetAttr(infos.op, attr<"\"output_layouts\"">, attr<"\"NCHW\"">);
        SetAttr(infos.op, attr<"\"expected_input_layouts\"">, attr<"\"NHWC,*,*\"">);
        SetAttr(infos.op, attr<"\"expected_output_layouts\"">, attr<"\"NHWC\"">);
    
        /// 3. set attrs that are directly passed to the custom call kernel.
        let num_group_attr = ConvertTorchConstantIntToI64Attr(num_group);
        SetCustomAttr(infos.op, attr<"\"num_group\"">, num_group_attr);
        SetCustomAttr(infos.op, attr<"\"eps\"">, eps_attr);
        SetCustomAttr(infos.op, attr<"\"silu\"">, attr<"true">);
    
        let rs = UnpackValue_1(infos.new_outputs);
        let new_output = ConvertToF32(rs);
    
        replace silu with new_output;
      };
    }

    Pattern TorchGroupNormOpF16 {
      /// match phase: define the pattern
      let eps_attr : Attr;
      let eps = op<torch.constant.float> { value = eps_attr };
      let gn = op<torch.aten.group_norm>(
        input: Value,
        num_group: Value,
        weight: Value,
        bias: Value,
        eps.0,
        cudnn_enabled: Value
      ) -> (old_type: Type);
      CheckNotTorchNone(bias);
      CheckTorchConstantInt(num_group);
      CheckTorchTensorElemType(input, attr<"\"f16\"">);
    
      /// rewrite phase
      rewrite gn with {
        /// 1. create custom call op
        let inputs = PackValue_3(attr<"\"in\"">, input, weight, bias);
        let outputs = PackValue_1(attr<"\"out\"">, gn.0);
        let infos = CreateTorchCustomCall(attr<"\"op\"">, inputs, outputs);
    
        /// 2. set attrs that are used by bladedisc.
        SetAttr(infos.op, attr<"\"call_target_name\"">, attr<"\"ral_pdll_group_norm\"">);
        SetAttr(infos.op, attr<"\"input_placements\"">, attr<"\"x,x,x\"">);
        SetAttr(infos.op, attr<"\"output_placements\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"device\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"input_layouts\"">, attr<"\"NCHW,*,*\"">);
        SetAttr(infos.op, attr<"\"output_layouts\"">, attr<"\"NCHW\"">);
        SetAttr(infos.op, attr<"\"expected_input_layouts\"">, attr<"\"NHWC,*,*\"">);
        SetAttr(infos.op, attr<"\"expected_output_layouts\"">, attr<"\"NHWC\"">);
    
        /// 3. set attrs that are directly passed to the custom call kernel.
        let num_group_attr = ConvertTorchConstantIntToI64Attr(num_group);
        SetCustomAttr(infos.op, attr<"\"num_group\"">, num_group_attr);
        SetCustomAttr(infos.op, attr<"\"eps\"">, eps_attr);
        SetCustomAttr(infos.op, attr<"\"silu\"">, attr<"false">);
    
        let rs = UnpackValue_1(infos.new_outputs);
        replace gn with rs;
      };
    } 

    Pattern TorchGroupNormWithSiluOpF16 {
      /// match phase: define the pattern
      let eps_attr : Attr;
      let eps = op<torch.constant.float> { value = eps_attr };
      let gn = op<torch.aten.group_norm>(
        input: Value,
        num_group: Value,
        weight: Value,
        bias: Value,
        eps.0,
        cudnn_enabled: Value
      ) -> (old_type: Type);
      let silu = op<torch.aten.silu>(gn.0);
      CheckNotTorchNone(bias);
      CheckTorchConstantInt(num_group);
      CheckTorchTensorElemType(input, attr<"\"f16\"">);
    
      /// rewrite phase
      rewrite silu with {
        /// 1. create custom call op
        let inputs = PackValue_3(attr<"\"in\"">, input, weight, bias);
        let outputs = PackValue_1(attr<"\"out\"">, silu.0);
        let infos = CreateTorchCustomCall(attr<"\"op\"">, inputs, outputs);
    
        /// 2. set attrs that are used by bladedisc.
        SetAttr(infos.op, attr<"\"call_target_name\"">, attr<"\"ral_pdll_group_norm\"">);
        SetAttr(infos.op, attr<"\"input_placements\"">, attr<"\"x,x,x\"">);
        SetAttr(infos.op, attr<"\"output_placements\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"device\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"input_layouts\"">, attr<"\"NCHW,*,*\"">);
        SetAttr(infos.op, attr<"\"output_layouts\"">, attr<"\"NCHW\"">);
        SetAttr(infos.op, attr<"\"expected_input_layouts\"">, attr<"\"NHWC,*,*\"">);
        SetAttr(infos.op, attr<"\"expected_output_layouts\"">, attr<"\"NHWC\"">);
    
        /// 3. set attrs that are directly passed to the custom call kernel.
        let num_group_attr = ConvertTorchConstantIntToI64Attr(num_group);
        SetCustomAttr(infos.op, attr<"\"num_group\"">, num_group_attr);
        SetCustomAttr(infos.op, attr<"\"eps\"">, eps_attr);
        SetCustomAttr(infos.op, attr<"\"silu\"">, attr<"true">);
    
        let rs = UnpackValue_1(infos.new_outputs);
        replace silu with rs;
      };
    }
    
  )pdll";

  return preDefinedPatterns;
}

} // namespace
