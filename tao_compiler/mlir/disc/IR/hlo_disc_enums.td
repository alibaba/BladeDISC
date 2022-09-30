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


// The round mode enum for HLO_FakeQuantOp, HLO_QuantizeOp and HLO_DequantizeOp
def HLO_RoundHalfAwayFromZero: I32EnumAttrCase<"RoundHalfAwayFromZero", 0>; // tf default mode
def HLO_RoundHalfToEven: I32EnumAttrCase<"RoundHalfToEven", 1>; // PyTorch default mode
def HLO_RoundModeEnum: I32EnumAttr<"RoundModeEnum",
    "Round mode in quantization related op",
    [
        HLO_RoundHalfAwayFromZero,
        HLO_RoundHalfToEven
    ]> {
  let summary = "Round mode in quantization related op";
  let cppNamespace = "::mlir::mhlo_disc";
}


#endif // MHLO_DISC_ENUMS
