// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <mlir/mhlo/builder/broadcast.h>
#include <mlir/mhlo/builder/mlir_attr_utils.h>
#include <mlir/mhlo/builder/mlir_type_utils.h>
#include <mlir/mhlo/builder/mlir_utils.h>
#include <mlir/mhlo/builder/standard.h>

#include "tensorflow/compiler/mlir/disc/transforms/disc_map_chlo_to_hlo_op.h"

namespace mlir {
namespace mhlo {

static constexpr const char kCompare_EQ[] = "EQ";
static constexpr const char kCompare_NE[] = "NE";
static constexpr const char kCompare_GE[] = "GE";
static constexpr const char kCompare_GT[] = "GT";
static constexpr const char kCompare_LE[] = "LE";
static constexpr const char kCompare_LT[] = "LT";

chlo::ComparisonDirection inline getChloComparisonDirectionFromString(
    const std::string& cmp) {
  if (cmp == kCompare_EQ) {
    return chlo::ComparisonDirection::EQ;
  } else if (cmp == kCompare_NE) {
    return chlo::ComparisonDirection::NE;
  } else if (cmp == kCompare_GE) {
    return chlo::ComparisonDirection::GE;
  } else if (cmp == kCompare_GT) {
    return chlo::ComparisonDirection::GT;
  } else if (cmp == kCompare_LE) {
    return chlo::ComparisonDirection::LE;
  } else if (cmp == kCompare_LT) {
    return chlo::ComparisonDirection::LT;
  } else {
    MHLO_CHECK(false, "Unhandled comparison direction.");
  }
}

mhlo::ComparisonDirection inline getMhloComparisonDirectionFromString(
    const std::string& cmp) {
  if (cmp == kCompare_EQ) {
    return mhlo::ComparisonDirection::EQ;
  } else if (cmp == kCompare_NE) {
    return mhlo::ComparisonDirection::NE;
  } else if (cmp == kCompare_GE) {
    return mhlo::ComparisonDirection::GE;
  } else if (cmp == kCompare_GT) {
    return mhlo::ComparisonDirection::GT;
  } else if (cmp == kCompare_LE) {
    return mhlo::ComparisonDirection::LE;
  } else if (cmp == kCompare_LT) {
    return mhlo::ComparisonDirection::LT;
  } else {
    MHLO_CHECK(false, "Unhandled comparison direction.");
  }
}

// NB: BuildMlirOp functions are a group of tools that were
// given inputs/output information to build up an MLIR subgraph,
// which represents specific math calculation on MLIR values.
template <class MLIR_BINARY_OP, const char* DIRECTION>
struct ChloBinaryOpBuilder {
  MLIR_BINARY_OP createOp(mlir::OpBuilder& builder, const mlir::Location& loc,
                          const mlir::Value& input_lhs,
                          const mlir::Value& input_rhs,
                          mlir::DenseIntElementsAttr broadcast_attr) {
    auto compare_direction =
        getChloComparisonDirectionFromString(std::string(DIRECTION));
    return builder.create<MLIR_BINARY_OP>(loc, input_lhs, input_rhs,
                                          /*broadcast_dims=*/broadcast_attr,
                                          compare_direction);
  }
};

template <class MLIR_BINARY_OP>
struct ChloBinaryOpBuilder<MLIR_BINARY_OP, nullptr> {
  MLIR_BINARY_OP createOp(mlir::OpBuilder& builder, const mlir::Location& loc,
                          const mlir::Value& input_lhs,
                          const mlir::Value& input_rhs,
                          mlir::DenseIntElementsAttr broadcast_attr) {
    return builder.create<MLIR_BINARY_OP>(loc, input_lhs, input_rhs,
                                          /*broadcast_dims=*/broadcast_attr);
  }
};

template <class MLIR_BINARY_OP, const char* DIRECTION>
struct HloBinaryOpBuilder {
  MLIR_BINARY_OP createOp(mlir::OpBuilder& builder, const mlir::Location& loc,
                          const mlir::Value& input_lhs,
                          const mlir::Value& input_rhs) {
    auto compare_direction =
        getMhloComparisonDirectionFromString(std::string(DIRECTION));
    return builder.create<MLIR_BINARY_OP>(loc, input_lhs, input_rhs,
                                          compare_direction);
  }
};

template <class MLIR_BINARY_OP>
struct HloBinaryOpBuilder<MLIR_BINARY_OP, nullptr> {
  MLIR_BINARY_OP createOp(mlir::OpBuilder& builder, const mlir::Location& loc,
                          const mlir::Value& input_lhs,
                          const mlir::Value& input_rhs) {
    return builder.create<MLIR_BINARY_OP>(loc, input_lhs, input_rhs);
  }
};

// NB: BuildMlirOp functions are a group of tools that were
// given inputs/output information to build up an MLIR subgraph,
// which represents specific math calculation on MLIR values.
template <class MLIR_BINARY_OP, const char* DIRECTION = nullptr,
          bool rhs_scalar = false>
mlir::Value BuildMlirBinaryOp(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& input_lhs,
                              const mlir::Value& input_rhs,
                              const mlir::Type& broadcast_elem_type,
                              bool no_implicit_broadcast = false) {
  // TODO: XLA/HLO and Numpy/ATen broadcast semantic is defferent,
  // XLA/HLO broadcast must be specified explicitly, but
  // Numpy/ATen broadcast only infers when need.
  //
  // XLA Broadcast:  https://www.tensorflow.org/xla/broadcasting
  // ATen Broadcast:
  // https://pytorch.org/docs/master/notes/broadcasting.html#broadcasting-semantics
  // Numpy Broadcast: https://numpy.org/doc/stable/user/basics.broadcasting.html
  mlir::Value lhs = input_lhs;
  mlir::Value rhs = input_rhs;
  if (rhs_scalar) {
    rhs = BuildStdScalarToHloTensor(builder, loc, rhs);
  }
  // both lhs & rhs are mhlo::tensor, try do type
  // cast(TryBuildElementTypeCast only work on mhlo::tensor)
  lhs = TryBuildElementTypeCast(builder, loc, lhs, broadcast_elem_type);
  rhs = TryBuildElementTypeCast(builder, loc, rhs, broadcast_elem_type);

  mlir_dim_t lhs_rank = GetRankOfMlirValue(lhs);
  mlir_dim_t rhs_rank = GetRankOfMlirValue(rhs);
  if (lhs_rank != rhs_rank) {
    if (lhs_rank == 0) {
      lhs = BuildBroadcastScalarAsTensor(builder, loc, lhs, rhs);
      lhs_rank = rhs_rank;
      no_implicit_broadcast = true;
    } else if (rhs_rank == 0) {
      rhs = BuildBroadcastScalarAsTensor(builder, loc, rhs, lhs);
      rhs_rank = lhs_rank;
      no_implicit_broadcast = true;
    }
  }

  mlir_dim_t max_rank = std::max(lhs_rank, rhs_rank);
  mlir_dim_t min_rank = std::min(lhs_rank, rhs_rank);

  mlir::DenseIntElementsAttr broadcast_attr = nullptr;
  if (max_rank > min_rank) {
    SmallVec4<mlir_dim_t> broadcast_dims =
        RangeIndices(max_rank - min_rank, max_rank);
    broadcast_attr = BuildI64ElementsAttr(builder, broadcast_dims);
  }

  mlir::Value result = nullptr;

  // Use chlo ops for implicit broadcast semantics;
  // Or, directly use the associated hlo op if no_implicit_broadcast
  if (max_rank == min_rank && no_implicit_broadcast) {
    using HloTy = typename mhlo_disc::ChloToHloOp<MLIR_BINARY_OP>;
    HloBinaryOpBuilder<HloTy, DIRECTION> binary_op_builder;
    result = binary_op_builder.createOp(builder, loc, lhs, rhs).getResult();
  } else {
    ChloBinaryOpBuilder<MLIR_BINARY_OP, DIRECTION> binary_op_builder;
    result = binary_op_builder.createOp(builder, loc, lhs, rhs, broadcast_attr)
                 .getResult();
  }

  return result;
}

}  // namespace mhlo
}  // namespace mlir
