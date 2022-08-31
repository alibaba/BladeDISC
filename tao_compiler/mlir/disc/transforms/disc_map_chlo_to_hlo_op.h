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
// This file provides mapping from chlo ops to mhlo ops, should be merged into
//     mlir-hlo/Dialect/mhlo/transforms/map_chlo_to_hlo_op.h

#ifndef MLIR_DISC_TRANSFORMS_DISC_MAP_CHLO_TO_HLO_OP_H_
#define MLIR_DISC_TRANSFORMS_DISC_MAP_CHLO_TO_HLO_OP_H_

#include <type_traits>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace mhlo_disc {

template <typename ChloOpTy>
struct ChloToHloOpImpl {
  using Type = std::false_type;
};
template <typename ChloOpTy>
using ChloToHloOp = typename ChloToHloOpImpl<ChloOpTy>::Type;

#define MAP_CHLO_TO_HLO(ChloOpName, MhloOpName) \
  template <>                                   \
  struct ChloToHloOpImpl<chlo::ChloOpName> {    \
    using Type = mhlo::MhloOpName;              \
  }

MAP_CHLO_TO_HLO(BroadcastAddOp, AddOp);
MAP_CHLO_TO_HLO(BroadcastAndOp, AndOp);
MAP_CHLO_TO_HLO(BroadcastAtan2Op, Atan2Op);
MAP_CHLO_TO_HLO(BroadcastCompareOp, CompareOp);
MAP_CHLO_TO_HLO(BroadcastComplexOp, ComplexOp);
MAP_CHLO_TO_HLO(BroadcastDivOp, DivOp);
MAP_CHLO_TO_HLO(BroadcastMaxOp, MaxOp);
MAP_CHLO_TO_HLO(BroadcastMinOp, MinOp);
MAP_CHLO_TO_HLO(BroadcastMulOp, MulOp);
MAP_CHLO_TO_HLO(BroadcastOrOp, OrOp);
MAP_CHLO_TO_HLO(BroadcastPowOp, PowOp);
MAP_CHLO_TO_HLO(BroadcastRemOp, RemOp);
MAP_CHLO_TO_HLO(BroadcastShiftLeftOp, ShiftLeftOp);
MAP_CHLO_TO_HLO(BroadcastShiftRightArithmeticOp, ShiftRightArithmeticOp);
MAP_CHLO_TO_HLO(BroadcastShiftRightLogicalOp, ShiftRightLogicalOp);
MAP_CHLO_TO_HLO(BroadcastSubOp, SubtractOp);
MAP_CHLO_TO_HLO(BroadcastXorOp, XorOp);

#undef MAP_CHLO_TO_HLO

}  // namespace mhlo_disc
}  // namespace mlir

#endif  // MLIR_DISC_TRANSFORMS_DISC_MAP_CHLO_TO_HLO_OP_H_
