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


#ifndef DISC_TF_ADDITIONAL_OPS
#define DISC_TF_ADDITIONAL_OPS

//include "tf_op_base.td"
//include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"
include "external/org_tensorflow/tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"



def TF_DiscFakeQuantOp : TF_Op<"DiscFakeQuant", [Pure, TF_NoConstantFold]> {

  let summary = "FakeQuantOp defined by DISC to carry quantization info.";

  let description = [{
This op carry quantization info from tensorflow model to BladeDISC.
  }];

  let arguments = (ins
    TF_Float32Tensor:$input,
    TF_Float32Tensor:$scale,
    TF_Int32Tensor:$zero_point,
    DefaultValuedAttr<I64Attr, "-128">:$quant_min,
    DefaultValuedAttr<I64Attr, "127">:$quant_max,
    DefaultValuedAttr<I64Attr, "8">:$num_bits,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$axis,
    DefaultValuedAttr<BoolAttr, "true">:$use_signed,
    DefaultValuedAttr<BoolAttr, "true">:$use_symmetric,
    DefaultValuedAttr<BoolAttr, "false">:$use_dynamic
  );

  let hasVerifier = 1;

  let results = (outs
    TF_Float32Tensor:$output
  );

  TF_DerivedOperandTypeAttr Tfloat = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tint = TF_DerivedOperandTypeAttr<1>;
}

#endif
