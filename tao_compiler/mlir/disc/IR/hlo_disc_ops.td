/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This is the operation definition file for MHLO DISC ops.

#ifndef MHLO_DISC_OPS
#define MHLO_DISC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_common.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_utils.td"
include "mlir/disc/IR/hlo_disc_enums.td"

def HLODISC_Dialect : Dialect {
  let name = "mhlo_disc";
  let cppNamespace = "::mlir::mhlo_disc";
}

class HLODISC_Op<string mnemonic, list<Trait> traits> :
    Op<HLODISC_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}

class HLODISC_ShapedInterfaceOp<string mnemonic, list<Trait> traits> :
    HLODISC_Op<mnemonic, traits # [DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
    ["reifyReturnTypeShapes"]>]> {
}

def HLODISC_H2DOp: HLODISC_ShapedInterfaceOp<"h2d", [Pure, SameOperandsAndResultType]> {
  let summary = "H2D operator";
  let description = [{
    Copy `operand` from host to device.
  }];
  let arguments = (ins HLO_Tensor);
  let results = (outs HLO_Tensor);
}

def HLODISC_D2HOp: HLODISC_ShapedInterfaceOp<"d2h", [Pure, SameOperandsAndResultType]> {
  let summary = "D2H operator";
  let description = [{
    Copy `operand` from device to host.
  }];
  let arguments = (ins HLO_Tensor);
  let results = (outs HLO_Tensor);
}

def HLO_TensorOrIndexTensor : AnyTypeOf<[HLO_Tensor, IndexTensor]>;

def HLO_I32Tensor : TensorOf<[SignlessIntOfWidths<[32]>]>;

def HLO_I64Tensor : TensorOf<[SignlessIntOfWidths<[64]>]>;

def HLO_I32OrI64Tensor : AnyTypeOf<[HLO_I32Tensor, HLO_I64Tensor]>;

// TODO(zk): How does 'has_side_effect' actully work?
def HLODISC_CustomCallOp: HLODISC_ShapedInterfaceOp<"custom_call", []> {
  let summary = "CustomCall operator";
  let description = [{
    A custom call invokes code external to XLA. The `args` are passed to the
    external code, and the external code is expected to produce a result of the
    given type. The exact mechanism is backend-specific. For example, in the CPU
    backend, a call instruction is emitted which targets a symbol with the name
    `call_target_name`.

    `call_target_name` and `backend_config` can be arbitrary strings, but
    `call_target_name` should be short as it may be used in labels.
    `backend_config` can encode arbitrarily large amounts of information.

    See https://www.tensorflow.org/xla/operation_semantics#customcall.
  }];
  let arguments = (ins
    Variadic<HLO_TensorOrIndexTensor>:$args,
    StrAttr:$call_target_name,
    DefaultValuedAttr<BoolAttr, "false">:$has_side_effect,
    DefaultValuedAttr<AnyAttr, "">:$backend_config
  );
  let results = (outs Variadic<HLO_TensorOrIndexTensor>);
}

// Models the behavior of the second version BladeDISC custom call.
// TODO(disc): Merge with the old version once the refactor is done.
def HLODISC_CustomCallV2Op: HLODISC_Op<"custom_call_v2", []> {
  let summary = "CustomCallV2 operator";
  let description = [{
    A custom call invokes code external to BladeDISC. The `args` are passed to the
    external code, and the external code is expected to produce a result of the
    given type. The exact mechanism is backend-specific. For example, in the CPU
    backend, a call instruction is emitted which targets a symbol with the name
    `call_target_name`.
  }];
  let arguments = (ins
    Variadic<HLO_TensorOrIndexTensor>:$args,
    StrAttr:$call_target_name,
    DictionaryAttr:$custom_attrs,
    DefaultValuedAttr<BoolAttr, "false">:$has_side_effect,
    DefaultValuedAttr<StrAttr, "">:$device,
    DefaultValuedAttr<StrAttr, "">:$input_placements,
    DefaultValuedAttr<StrAttr, "">:$output_placements,
    DefaultValuedAttr<StrAttr, "">:$input_layouts,
    DefaultValuedAttr<StrAttr, "">:$output_layouts,
    DefaultValuedAttr<StrAttr, "">:$expected_input_layouts,
    DefaultValuedAttr<StrAttr, "">:$expected_output_layouts
  );
  let results = (outs Variadic<HLO_TensorOrIndexTensor>);
  let extraClassDeclaration = [{
    SmallVector<std::string> parseStringRef(StringRef layouts,
                                        SmallVector<int64_t> ranks) {
      // Only two things will be done:
      // 1. parse the layout setting string.
      // 2. replace * with default layouts according to the corresponding rank.
      // Strings except * (such as **) will not be processed, and the string
      // corresponding to the invalid operand will not be processed
      const std::string defaultLayouts = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
      SmallVector<std::string> parsedStrings;
      SmallVector<StringRef> parsedStringRefs;
      layouts.split(parsedStringRefs, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
      int64_t num_operands = ranks.size();
      for (int64_t i = 0; i < parsedStringRefs.size(); i++) {
        std::string layoutString = parsedStringRefs[i].str();
        if (layoutString == std::string("*") && i < num_operands) {
          int64_t rank = ranks[i];
          if (rank <= defaultLayouts.length()) {
            layoutString = defaultLayouts.substr(0, rank);
          }
        }
        parsedStrings.push_back(layoutString);
      }
      return parsedStrings;
    }
    SmallVector<std::string> parseInputLayouts() {
      SmallVector<int64_t> ranks;
      for (const auto& operands : this->getOperands()) {
        ranks.push_back(operands.getType().dyn_cast<RankedTensorType>().getRank());
      }
      SmallVector<std::string> inputLayouts;
      inputLayouts = parseStringRef(this->getInputLayouts(), ranks);
      return inputLayouts;
    }

    SmallVector<std::string> parseExpectedInputLayouts() {
      SmallVector<int64_t> ranks;
      for (const auto& operands: this->getOperands()) {
        ranks.push_back(operands.getType().dyn_cast<RankedTensorType>().getRank());
      }
      SmallVector<std::string> expectedInputLayouts;
      expectedInputLayouts = parseStringRef(this->getExpectedInputLayouts(), ranks);
      return expectedInputLayouts;
    }

    SmallVector<std::string> parseOutputLayouts() {
      SmallVector<int64_t> ranks;
      for (const auto& result: this->getResults()) {
        ranks.push_back(result.getType().dyn_cast<RankedTensorType>().getRank());
      }
      SmallVector<std::string> outputLayouts;
      outputLayouts = parseStringRef(this->getOutputLayouts(), ranks);
      return outputLayouts;
    }

    SmallVector<std::string> parseExpectedOutputLayouts() {
      SmallVector<int64_t> ranks;
      for (const auto& result: this->getResults()) {
        ranks.push_back(result.getType().dyn_cast<RankedTensorType>().getRank());
      }
      SmallVector<std::string> expectedOutputLayouts;
      expectedOutputLayouts = parseStringRef(this->getExpectedOutputLayouts(), ranks);
      return expectedOutputLayouts;
    }
  }];
}

def HLO_FakeQuantOp: HLODISC_Op<"fake_quant", [Pure]> {
  let summary = "FakeQuant operator";
  let description = [{
    An fake quant op is logically equal to a quant op + a dequant op.
  }];
  let arguments = (ins
    HLO_FpTensor:$input,
    HLO_FpTensor:$scale,
    HLO_I32Tensor:$zero_point,
    DefaultValuedAttr<BoolAttr, "true">:$use_signed,
    DefaultValuedAttr<BoolAttr, "true">:$use_symmetric,
    DefaultValuedAttr<I64ElementsAttr, "ArrayRef<int64_t>{}">:$axis,
    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<I64Attr, "-128">:$quant_min,
    DefaultValuedAttr<I64Attr, "127">:$quant_max,
    DefaultValuedAttr<BoolAttr, "false">:$use_dynamic,
    DefaultValuedAttr<HLO_RoundModeEnum, "RoundModeEnum::RoundHalfAwayFromZero">:$round_mode
  );
  let results = (outs HLO_FpTensor:$result);
}

def HLO_QuantizeOp: HLODISC_Op<"quantize", [Pure]> {
  let summary = "Quantize operator";
  let description = [{
    Quantize a float tensor to int tensor.
  }];
  let arguments = (ins
    HLO_FpTensor:$input,
    HLO_FpTensor:$scale,
    HLO_I32Tensor:$zero_point,
    DefaultValuedAttr<BoolAttr, "true">:$use_symmetric,
    DefaultValuedAttr<I64ElementsAttr, "ArrayRef<int64_t>{}">:$axis,
    DefaultValuedAttr<I64Attr, "-128">:$quant_min,
    DefaultValuedAttr<I64Attr, "127">:$quant_max,
    DefaultValuedAttr<BoolAttr, "false">:$use_dynamic,
    DefaultValuedAttr<HLO_RoundModeEnum, "RoundModeEnum::RoundHalfAwayFromZero">:$round_mode
  );
  let results = (outs HLO_IntTensor:$result);
  let hasFolder = 1;
}

def HLO_DequantizeOp: HLODISC_Op<"dequantize", [Pure]> {
  let summary = "Dequantize operator";
  let description = [{
    Dequantize a int tensor to float tensor.
  }];
  let arguments = (ins
    HLO_IntTensor:$input,
    HLO_FpTensor:$scale,
    HLO_I32Tensor:$zero_point,
    DefaultValuedAttr<BoolAttr, "true">:$use_symmetric,
    DefaultValuedAttr<I64ElementsAttr, "ArrayRef<int64_t>{}">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$use_dynamic,
    DefaultValuedAttr<HLO_RoundModeEnum, "RoundModeEnum::RoundHalfAwayFromZero">:$round_mode
  );
  let results = (outs HLO_FpTensor:$result);
}

def HLO_QuantizedDotGeneralOp: HLODISC_ShapedInterfaceOp<"quantized_dot_general", [Pure]> {
  let summary = "quantized version of dot gerneal operator";
  let description = [{
    Compute the dot product using result quantized inputs.
  }];
  let arguments = (ins
    HLO_IntTensor:$input,
    HLO_IntTensor:$weight,
    HLO_FpTensor:$input_scale,
    HLO_I32Tensor:$input_zero_point,
    HLO_FpTensor:$weight_scale,
    HLO_I32Tensor:$weight_zero_point,
    HLO_FpTensor:$result_scale,
    HLO_I32Tensor:$result_zero_point,
    DotDimensionNumbers:$dot_dimension_numbers,
    DefaultValuedAttr<BoolAttr, "true">:$use_symmetric,
    DefaultValuedAttr<I64ElementsAttr, "ArrayRef<int64_t>{}">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$use_dynamic
  );
  let results = (outs HLO_Tensor:$result);
}

def HLO_QuantizedDynamicConvOp: HLODISC_ShapedInterfaceOp<"quantized_dynamic_conv", [Pure]> {
  let summary = "quantized version of dynamic_conv operator";
  let description = [{
    Compute the convolution result using quantized inputs.
  }];
  let arguments = (ins
    HLO_IntTensor:$input,
    HLO_IntTensor:$weight,
    HLO_IntTensor:$d_padding,
    HLO_FpTensor:$input_scale,
    HLO_I32Tensor:$input_zero_point,
    HLO_FpTensor:$weight_scale,
    HLO_I32Tensor:$weight_zero_point,
    HLO_FpTensor:$result_scale,
    HLO_I32Tensor:$result_zero_point,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    // Default value: two zeros for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$padding,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$lhs_dilation,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$rhs_dilation,
    // Default value: false for each of the spatial dimension.
    OptionalAttr<BoolElementsAttr>:$window_reversal,
    ConvDimensionNumbers:$dimension_numbers,
    I64Attr:$feature_group_count,
    I64Attr:$batch_group_count,
    DefaultValuedAttr<BoolAttr, "true">:$use_symmetric,
    DefaultValuedAttr<I64ElementsAttr, "ArrayRef<int64_t>{}">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$use_dynamic
  );
  let results = (outs HLO_Tensor:$result);
}

def HLO_SparseReshapeOp : HLODISC_ShapedInterfaceOp<"sparse_reshape", [Pure]> {
  let summary = "One-to-one mapping of TF_SparseReshapeOp";
  let description = [{
    See tf's sparse_reshape operator.
  }];
  let arguments = (ins
    TensorOf<[I64]>:$input_indices,
    TensorOf<[I64]>:$input_shape,
    TensorOf<[I64]>:$new_shape
  );

  let results = (outs
    TensorOf<[I64]>:$output_indices,
    TensorOf<[I64]>:$output_shape
  );
}

// possible types for SparseFillEmptyRows:
// Supported integral types: int64/int32/uint16/int16/uint8/int8
// Supported float types: half/bfloat16/float/double
def HLO_SparseFillEmptyRowsOp : HLODISC_ShapedInterfaceOp<"sparse_fill_empty_rows", [Pure]> {
  let summary = "One-to-one mapping of TF_SparseFillEmptyRowsOp";
  let description = [{
    See tf's sparse_fill_empty_rows operator.
  }];
  let arguments = (ins
    TensorOf<[I64]>:$indices,
    HLO_Tensor:$values,
    TensorOf<[I64]>:$dense_shape,
    HLO_Tensor:$default_value
  );

  let results = (outs
    TensorOf<[I64]>:$output_indices,
    HLO_Tensor:$output_values,
    TensorOf<[I1]>:$empty_row_indicator,
    TensorOf<[I64]>:$reverse_index_map,
    HLO_Tensor:$output_elements
  );
}

def HLO_SparseSegmentReductionOp: HLODISC_ShapedInterfaceOp<"sparse_segment_reduction", [Pure]> {
  let summary = "One-to-one mapping of TF_SparseSegmentMeanOp or TF_SparseSegmentSumOp";
  let description = [{
    See tf's sparse_segment_mean or sparse_segment_sum operator.
  }];
  let arguments = (ins
    HLO_FpTensor:$data,
    HLO_I32OrI64Tensor:$indices,
    HLO_I32OrI64Tensor:$segment_ids,
    DefaultValuedAttr<HLO_ReductionModeEnum, "ReductionModeEnum::Mean">:$reduction_mode
  );

  let results = (outs
    HLO_FpTensor:$output
  );
}

def HLO_SparseSegmentReductionWithEmptyRowsOp : HLODISC_ShapedInterfaceOp<"sparse_segment_reduction_with_empty_rows", [Pure]> {
  let summary = "This is combination of sparse_fill_empty_rows + sparse_segment_mean/sum";
  let description = [{
    This is specialized op only used for inference case.
    In traning scenario, these 2 ops should not be fused, since `reverse_index_map`
    is needed for backward computation.
  }];
  let arguments = (ins
    HLO_FpTensor:$data,
    HLO_I32OrI64Tensor:$indices,
    HLO_I32OrI64Tensor:$unfilled_segment_ids,
    TensorOf<[I64]>:$dense_shape, // needed for output shape computation
    DefaultValuedAttr<HLO_ReductionModeEnum, "ReductionModeEnum::Mean">:$reduction_mode
  );

  let results = (outs
    HLO_FpTensor:$output,
    TensorOf<[I1]>:$empty_row_indicator
  );
}

def HLO_WhereOp : HLODISC_ShapedInterfaceOp<"where", [Pure]> {
  let summary = "One-to-one mapping of TF_WhereOp";
  let description = [{
    See tf's where operator.
  }];
  let arguments = (ins
    HLO_Tensor:$input
  );

  let results = (outs
    HLO_I64Tensor:$index,
    HLO_I64Tensor:$num_output_elements
  );
}

#endif // MHLO_DISC_OPS
