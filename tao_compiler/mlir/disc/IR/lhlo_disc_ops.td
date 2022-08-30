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
include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops_base.td"
include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops_structs.td"
include "mlir-hlo/Dialect/lhlo/IR/lhlo_structured_interface.td"

def LHLODISC_Dialect : Dialect {
  let name = "lmhlo_disc";
  let cppNamespace = "::mlir::lmhlo_disc";
}

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

// TODO(feiwen.zfw): support print float/half/double and memref
def LHLODISC_PrintfOp: LHLODISC_Op<"printf", []> {
  let summary = "Printf Operator";
  let description = [{
    Printf Index, Pred and Int. lower to printf function.
  }];
  let arguments = (ins Arg<Variadic<LHLO_DimensionValue>, "", [MemRead]>:$args,
                       Arg<LHLO_DimensionValue, "", [MemWrite]>:$output,
                       StrAttr:$format);
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

#endif // LMHLO_DISC_OPS
