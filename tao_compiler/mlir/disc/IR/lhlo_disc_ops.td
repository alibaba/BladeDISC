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

// This is the operation definition file for LMHLO DISC ops.

#ifndef LMHLO_DISC_OPS
#define LMHLO_DISC_OPS

include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "lhlo/IR/lhlo_ops_base.td"
include "lhlo/IR/lhlo_ops_structs.td"
include "lhlo/IR/lhlo_structured_interface.td"
include "mlir/disc/IR/lhlo_disc_enums.td"

def LHLODISC_Dialect : Dialect {
  let name = "lmhlo_disc";
  let cppNamespace = "::mlir::lmhlo_disc";
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

def LHLO_I32Tensor : MemRefOf<[SignlessIntOfWidths<[32]>]>;

class LHLODISC_Op<string mnemonic, list<Trait> traits> :
  Op<LHLODISC_Dialect, mnemonic,
    !listconcat([MemoryEffects<[MemRead, MemWrite]>,
    LmhloStructuredInterface], traits)>;

def LHLODISC_H2DOp: LHLODISC_Op<"h2d", []> {
  let summary = "H2D operator";
  let description = [{
    Copy `operand` from host to device.
  }];
  let arguments = (ins Arg<LHLO_Buffer, "", [MemRead]>:$input,
                       Arg<LHLO_Buffer, "", [MemWrite]>:$output);
}

def LHLODISC_D2HOp: LHLODISC_Op<"d2h", []> {
  let summary = "D2H operator";
  let description = [{
    Copy `operand` from device to host.
  }];
  let arguments = (ins Arg<LHLO_Buffer, "", [MemRead]>:$input,
                       Arg<LHLO_Buffer, "", [MemWrite]>:$output);
}

def IndexBuffer : MemRefOf<[Index]>;
def LHLO_BufferOrIndexBuffer : AnyTypeOf<[LHLO_Buffer, IndexBuffer]>;

def LHLO_CustomCallOp : LHLODISC_Op<"custom_call", [AttrSizedOperandSegments]> {
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
    Arg<Variadic<LHLO_BufferOrIndexBuffer>, "", [MemRead]>:$args,
    Arg<Variadic<LHLO_BufferOrIndexBuffer>, "", [MemWrite]>:$output,
    StrAttr:$call_target_name,
    DefaultValuedAttr<BoolAttr, "false">:$has_side_effect,
    DefaultValuedAttr<AnyAttr, "{}">:$backend_config,
    OptionalAttr<CustomCallTargetArgMappingAttr>:$target_arg_mapping
  );
  let hasVerifier = 1;
}

// Models the behavior of the second version BladeDISC custom call.
// TODO(wyzero): Merge with the old version once the refactor is done.
def LHLODISC_CustomCallV2Op: Op<LHLODISC_Dialect, "custom_call_v2", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "CustomCallV2 operator";
  let description = [{
    A custom call invokes code external to BladeDISC. The `args` are passed to the
    external code, and the external code is expected to produce a result of the
    given type. The exact mechanism is backend-specific. For example, in the CPU
    backend, a call instruction is emitted which targets a symbol with the name
    `call_target_name`.
  }];
  let arguments = (ins
    Arg<Variadic<LHLO_BufferOrIndexBuffer>, "", [MemRead]>:$args,
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
  let results = (outs Variadic<LHLO_BufferOrIndexBuffer>);
}

// TODO(feiwen.zfw): support print float/half/double and memref
def LHLODISC_PrintfOp: LHLODISC_Op<"printf", []> {
  let summary = "Printf Operator";
  let description = [{
    Printf Index, Pred and Int. lower to printf function.
  }];
  let arguments = (ins Arg<Variadic<LHLO_DimensionValue>, "", [MemRead]>:$args,
                       StrAttr:$format);
}

def LHLODISC_QuantizedDotGeneralOp: LHLODISC_Op<"quantized_dot_general", []> {
  let summary = "quantized version of dot gerneal operator";
  let description = [{
    Compute the dot product using result quantized inputs.
  }];
  let arguments = (ins
    Arg<LHLO_IntBuffer, "", [MemRead]>:$input,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$weight,
    Arg<LHLO_FpBuffer, "", [MemRead]>:$input_scale,
    Arg<LHLO_I32Tensor, "", [MemRead]>:$input_zero_point,
    Arg<LHLO_FpBuffer, "", [MemRead]>:$weight_scale,
    Arg<LHLO_I32Tensor, "", [MemRead]>:$weight_zero_point,
    Arg<LHLO_FpBuffer, "", [MemRead]>:$result_scale,
    Arg<LHLO_I32Tensor, "", [MemRead]>:$result_zero_point,
    Arg<LHLO_IntBuffer, "", [MemWrite]>:$result,
    MHLO_DotDimensionNumbers:$dot_dimension_numbers,
    DefaultValuedAttr<BoolAttr, "true">:$use_symmetric,
    DefaultValuedAttr<I64ElementsAttr, "ArrayRef<int64_t>{}">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$use_dynamic
  );
}

def LHLODISC_QuantizedDynamicConvOp: LHLODISC_Op<"quantized_dynamic_conv", []> {
  let summary = "quantized version of dynamic_conv operator";
  let description = [{
    Compute the convolution result using quantized inputs.
  }];
  let arguments = (ins
    Arg<LHLO_IntBuffer, "", [MemRead]>:$input,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$weight,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$d_padding,
    Arg<LHLO_FpBuffer, "", [MemRead]>:$input_scale,
    Arg<LHLO_I32Tensor, "", [MemRead]>:$input_zero_point,
    Arg<LHLO_FpBuffer, "", [MemRead]>:$weight_scale,
    Arg<LHLO_I32Tensor, "", [MemRead]>:$weight_zero_point,
    Arg<LHLO_FpBuffer, "", [MemRead]>:$result_scale,
    Arg<LHLO_I32Tensor, "", [MemRead]>:$result_zero_point,
    Arg<LHLO_IntBuffer, "", [MemWrite]>:$result,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    // Default value: two zeros for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$padding,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$lhs_dilation,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$rhs_dilation,
    // Default value: false for each of the spatial dimension.
    OptionalAttr<MHLO_BoolElementsAttr>:$window_reversal,
    MHLO_ConvDimensionNumbers:$dimension_numbers,
    I64Attr:$feature_group_count,
    I64Attr:$batch_group_count,
    DefaultValuedAttr<BoolAttr, "true">:$use_symmetric,
    DefaultValuedAttr<I64ElementsAttr, "ArrayRef<int64_t>{}">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$use_dynamic
  );
}

def LHLODISC_SparseReshapeOp : LHLODISC_Op<"sparse_reshape", []> {
  let summary = "One-to-one mapping of TF_SparseReshapeOp";
  let description = [{
    See tf's sparse_reshape operator.
  }];
  let arguments = (ins
    Arg<LHLO_IntBuffer, "", [MemRead]>:$input_indices,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$input_shape,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$new_shape,
    Arg<LHLO_IntBuffer, "", [MemWrite]>:$output_indices,
    Arg<LHLO_IntBuffer, "", [MemWrite]>:$output_shape
  );
}

def LHLO_SparseFillEmptyRowsOp : LHLODISC_Op<"sparse_fill_empty_rows", []> {
  let summary = "One-to-one mapping of TF_SparseFillEmptyRowsOp";
  let description = [{
    See tf's sparse_fill_empty_rows operator.
  }];
  let arguments = (ins
    Arg<LHLO_IntBuffer, "", [MemRead]>:$indices,
    Arg<LHLO_Buffer, "", [MemRead]>:$values,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$dense_shape,
    Arg<LHLO_Buffer, "", [MemRead]>:$default_value,
    Arg<LHLO_IntBuffer, "", [MemWrite]>:$output_indices,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output_values,
    Arg<LHLO_PredBuffer, "", [MemWrite]>:$empty_row_indicator,
    Arg<LHLO_IntBuffer, "", [MemWrite]>:$reverse_index_map,
    Arg<LHLO_IntBuffer, "", [MemWrite]>:$output_elements
  );
}

def LHLO_SparseSegmentReductionOp : LHLODISC_Op<"sparse_segment_reduction", []> {
  let summary = "One-to-one mapping of TF_SparseSegmentMeanOp or TF_SparseSegmentSumOp";
  let description = [{
    See tf's sparse_segment_mean or sparse_segment_sum operator.
  }];
  let arguments = (ins
    Arg<LHLO_FpBuffer, "", [MemRead]>:$data,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$indices,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$segment_ids,
    DefaultValuedAttr<LHLO_ReductionModeEnum, "ReductionModeEnum::Mean">:$reduction_mode,
    Arg<LHLO_FpBuffer, "", [MemWrite]>:$output
  );
}

def LHLO_SparseSegmentReductionWithEmptyRowsOp : LHLODISC_Op<"sparse_segment_reduction_with_empty_rows", []> {
  let summary = "This is combination of sparse_fill_empty_rows + sparse_segment_mean/sum";
  let description = [{
    Since fuse 2 thses ops is difficult when these 2 both contains many loops after legalize-to-roots pass, however we can greatly simplify the computation in current impl.
  }];
  let arguments = (ins
    Arg<LHLO_FpBuffer, "", [MemRead]>:$data,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$indices,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$unfilled_segment_ids,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$dense_shape,
    DefaultValuedAttr<LHLO_ReductionModeEnum, "ReductionModeEnum::Mean">:$reduction_mode,
    Arg<LHLO_FpBuffer, "", [MemWrite]>:$output,
    Arg<LHLO_PredBuffer, "", [MemWrite]>:$empty_row_indicator
  );
}

def LHLO_WhereOp : LHLODISC_Op<"where", []> {
  let summary = "One-to-one mapping of TF_WhereOp";
  let description = [{
    See tf's where operator.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$input,
    Arg<LHLO_IntBuffer, "", [MemWrite]>:$index,
    Arg<LHLO_IntBuffer, "", [MemWrite]>:$num_output_elements
  );
}

def LHLODISC_SourceCodeOp: LHLODISC_Op<"source_code", [AttrSizedOperandSegments]> {
  let summary = "Source code func operator";
  let description = [{
    It maintains the source code and the corresponding operands/results of the
    code. The order of {args, output} are the same with that of the source code.
  }];
  let arguments = (ins
    Arg<Variadic<LHLO_BufferOrIndexBuffer>, "", [MemRead]>:$args,
    Arg<Variadic<LHLO_BufferOrIndexBuffer>, "", [MemWrite]>:$output,
    StrAttr:$code,
    StrAttr:$call_target_name
  );
}

def LHLODISC_ConcatenateOp : LHLODISC_Op<"concatenate", []> {
  let summary = "DISC's concatenate op";
  let description = [{
     Concatenates a set of tensors along the specified dimension,
     please note, this operator only supports fixed shape in the current codebase.
   }];
   let arguments = (ins
     Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$val,
     Arg<LHLO_Buffer, "", [MemRead]>:$input_ptrs,
     Arg<LHLO_Buffer, "", [MemWrite]>:$output,
     I64Attr:$dimension
   );
}

#endif // LMHLO_DISC_OPS
