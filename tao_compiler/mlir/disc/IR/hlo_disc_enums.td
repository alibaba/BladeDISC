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

#ifndef MHLO_DISC_ENUMS
#define MHLO_DISC_ENUMS

include "mlir/IR/EnumAttr.td"


// The round mode enum for MHLO_FakeQuantOp, MHLO_QuantizeOp and MHLO_DequantizeOp
def MHLO_RoundHalfAwayFromZero: I64EnumAttrCase<"RoundHalfAwayFromZero", 0>; // tf default mode
def MHLO_RoundHalfToEven: I64EnumAttrCase<"RoundHalfToEven", 1>; // PyTorch default mode
def MHLO_RoundModeEnum: I64EnumAttr<"RoundModeEnum",
    "Round mode in quantization related op",
    [
        MHLO_RoundHalfAwayFromZero,
        MHLO_RoundHalfToEven
    ]> {
  let summary = "Round mode in quantization related op";
  let cppNamespace = "::mlir::mhlo_disc";
}

// The reduction mode enum used for HLO_DISC_SparseSegmentReduction
def HLO_ReductionModeMean : I64EnumAttrCase<"Mean", 0>;
def HLO_ReductionModeSum : I64EnumAttrCase<"Sum", 1>;
def HLO_ReductionModeSqrtn : I64EnumAttrCase<"Sqrtn", 2>;
def HLO_ReductionModeEnum: I64EnumAttr<"ReductionModeEnum",
    "Reduction model enum used for sparse reduction now",
    [
        HLO_ReductionModeMean,
        HLO_ReductionModeSum,
        HLO_ReductionModeSqrtn
    ]> {
  let summary = "Reduction mode in sparse segment redcution ops";
  let cppNamespace = "::mlir::mhlo_disc";
}

#endif // MHLO_DISC_ENUMS
