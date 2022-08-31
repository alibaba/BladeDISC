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

def HLODISC_H2DOp: HLODISC_ShapedInterfaceOp<"h2d", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "H2D operator";
  let description = [{
    Copy `operand` from host to device.
  }];
  let arguments = (ins HLO_Tensor);
  let results = (outs HLO_Tensor);
}

def HLODISC_D2HOp: HLODISC_ShapedInterfaceOp<"d2h", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "D2H operator";
  let description = [{
    Copy `operand` from device to host.
  }];
  let arguments = (ins HLO_Tensor);
  let results = (outs HLO_Tensor);
}

def HLO_TensorOrIndexTensor : AnyTypeOf<[HLO_Tensor, IndexTensor]>;

def HLO_I32Tensor : TensorOf<[SignlessIntOfWidths<[32]>]>;

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

def HLO_FakeQuantOp: HLODISC_Op<"fake_quant", [NoSideEffect]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$use_dynamic
  );
  let results = (outs HLO_FpTensor:$result);
}

def HLO_QuantizeOp: HLODISC_Op<"quantize", [NoSideEffect]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$use_dynamic
  );
  let results = (outs HLO_IntTensor:$result);
}

def HLO_DequantizeOp: HLODISC_Op<"dequantize", [NoSideEffect]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$use_dynamic
  );
  let results = (outs HLO_FpTensor:$result);
}

def HLO_QuantizedDotGeneralOp: HLODISC_Op<"quantized_dot_general", [NoSideEffect]> {
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

def HLO_QuantizedDynamicConvOp: HLODISC_Op<"quantized_dynamic_conv", [NoSideEffect]> {
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

def HLO_SparseReshapeOp : HLODISC_ShapedInterfaceOp<"sparse_reshape", [NoSideEffect]> {
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

#endif // MHLO_DISC_OPS
