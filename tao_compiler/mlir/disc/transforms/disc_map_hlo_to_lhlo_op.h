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

#ifndef MLIR_DISC_TRANSFORMS_DISC_MAP_HLO_TO_LHLO_OP_H_
#define MLIR_DISC_TRANSFORMS_DISC_MAP_HLO_TO_LHLO_OP_H_

#include <type_traits>

#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"

namespace mlir {
namespace mhlo_disc {

template <typename HloOpTy>
struct HloToLhloOpImpl {
  using Type = std::false_type;
};
template <typename HloOpTy>
using HloToLhloOp = typename HloToLhloOpImpl<HloOpTy>::Type;

#define MAP_HLO_TO_LHLO(OpName)               \
  template <>                                 \
  struct HloToLhloOpImpl<mhlo_disc::OpName> { \
    using Type = lmhlo_disc::OpName;          \
  }

MAP_HLO_TO_LHLO(H2DOp);
MAP_HLO_TO_LHLO(D2HOp);
MAP_HLO_TO_LHLO(QuantizedDotGeneralOp);
MAP_HLO_TO_LHLO(QuantizedDynamicConvOp);
MAP_HLO_TO_LHLO(SparseReshapeOp);
MAP_HLO_TO_LHLO(SparseFillEmptyRowsOp);
MAP_HLO_TO_LHLO(SparseSegmentReductionOp);
MAP_HLO_TO_LHLO(SparseSegmentReductionWithEmptyRowsOp);
MAP_HLO_TO_LHLO(WhereOp);

#undef MAP_HLO_TO_LHLO

}  // namespace mhlo_disc
}  // namespace mlir

#endif  // MLIR_DISC_TRANSFORMS_DISC_MAP_HLO_TO_LHLO_OP_H_
