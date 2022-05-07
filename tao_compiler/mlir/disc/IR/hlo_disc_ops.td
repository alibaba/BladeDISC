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
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.td"
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
    DefaultValuedAttr<StrAttr, "">:$backend_config
  );
  let results = (outs Variadic<HLO_TensorOrIndexTensor>);
}

#endif // MHLO_DISC_OPS
