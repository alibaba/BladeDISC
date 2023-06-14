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

#ifndef LMHLO_DISC_ENUMS
#define LMHLO_DISC_ENUMS

include "mlir/IR/EnumAttr.td"


// The reduction mode enum used for LHLO_DISC_SparseSegmentReduction
def LHLO_ReductionModeMean : I64EnumAttrCase<"Mean", 0>;
def LHLO_ReductionModeSum : I64EnumAttrCase<"Sum", 1>;
def LHLO_ReductionModeSqrtn : I64EnumAttrCase<"Sqrtn", 2>;
def LHLO_ReductionModeEnum: I64EnumAttr<"ReductionModeEnum",
    "Reduction model enum used for sparse reduction now",
    [
        LHLO_ReductionModeMean,
        LHLO_ReductionModeSum,
        LHLO_ReductionModeSqrtn
    ]> {
  let summary = "Reduction mode in sparse segment redcution ops";
  let cppNamespace = "::mlir::lmhlo_disc";
}

#endif // MHLO_DISC_ENUMS
