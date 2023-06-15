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

// This file implements logic for lowering skeleton ops (reductions, results) to
// ParallelOp loop logics.

#include "lhlo/IR/lhlo_ops.h"
#include "lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/disc/IR/disc_shape_ops.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/codegen_utils.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "mlir/disc/transforms/input_inline_fusion_pattern.h"
#include "mlir/disc/transforms/lhlo_elemental_utils.h"
#include "mlir/disc/transforms/placement_utils.h"
#include "tensorflow/core/util/env_var.h"
#include "utils/placement_utils.h"

namespace mlir {
namespace disc_ral {

namespace {

template <typename LHLO_OpTy>
LogicalResult elemwiseLowerHelper(OpBuilder& b, Location loc, Operation* op,
                                  Value output_linear_index,
                                  const ShapeAnalysis* shape_analysis,
                                  int vector_size, LowerConfig* lower_config) {
  if (!isa<LHLO_OpTy>(op) || !op->hasTrait<mlir::OpTrait::Elementwise>())
    return failure();

  Value result_memref = cast<lmhlo::LmhloOp>(op).getResultBuffer();
  Value memref = result_memref;
  // TODO(disc): we acutally don't need this after shape constraint refactor
  // Remove this part of code after we finish the benchamrk after the refactor.
  if (auto analysis_deprecated =
          dynamic_cast<const ShapeAnalysisDeprecated*>(shape_analysis)) {
    auto parent_fusion = op->getParentOfType<mhlo::FusionOp>();
    Value leader_memref =
        analysis_deprecated->GetLeaderValueWithSameShapeInFusion(parent_fusion,
                                                                 result_memref);
    if (leader_memref != nullptr) {
      memref = leader_memref;
    }
  }
  Value memref_1d = createMemRef1DReinterpretCast(b, loc, result_memref);
  // Assuming alignment is to help vectorization optimization. Vectorization
  // pass in llvm requires an alignment of `data-byte-width * vector-size`.
  if (vector_size > 1) {
    createAlignMemrefWithTile(b, memref_1d, vector_size);
  }
  Value vec_size = b.create<arith::ConstantIndexOp>(loc, vector_size);
  if (vector_size > 1) {
    output_linear_index =
        b.create<arith::MulIOp>(loc, output_linear_index, vec_size);
  }

  // TODO: find a way to unroll a loop and reorder instructions automatically.
  SmallVector<Value, 4> linear_indices(vector_size);
  for (int i = 0; i < vector_size; i++) {
    linear_indices[i] = b.create<arith::AddIOp>(
        loc, output_linear_index, b.create<arith::ConstantIndexOp>(loc, i));
  }
  SmallVector<SmallVector<Value>> multidim_index_vector(vector_size);
  for (int64_t i = 0; i < vector_size; i++) {
    Value linear_index = linear_indices[i];
    auto multidim_index = calcMultiDimIndex(&b, loc, linear_index, memref);
    multidim_index_vector[i] = std::move(multidim_index);
  }
  SmallVector<SmallVector<Value, 4>> operand_values_vector(vector_size);
  for (Value operand_memref : op->getOperands().drop_back()) {
    for (int64_t i = 0; i < vector_size; i++) {
      auto multidim_index = multidim_index_vector[i];
      // TODO: this does not ensure vectorized load.
      Value operand_data = createLoadOrUseCachedValue(
          loc, &b, op, operand_memref, multidim_index, b.saveInsertionPoint(),
          lower_config);
      operand_values_vector[i].push_back(operand_data);
    }
  }
  SmallVector<Value, 4> results(vector_size);

  for (int64_t i = 0; i < vector_size; i++) {
    auto operand_values = operand_values_vector[i];
    auto res = LhloOpToStdScalarOp::map<LHLO_OpTy>(
        llvm::cast<LHLO_OpTy>(op),
        result_memref.getType().cast<MemRefType>().getElementType(),
        operand_values, &b);
    results[i] = res;
  }

  // For vectorize optimization.
  for (int i = 0; i < vector_size; i++) {
    b.create<memref::StoreOp>(loc, results[i], memref_1d,
                              ValueRange{linear_indices[i]});
  }

  return success();
}

template <typename LHLO_OpTy>
LogicalResult miscLowerHelper(OpBuilder& b, Location loc, Operation* opaque_op,
                              Value output_linear_index,
                              const ShapeAnalysis* shape_analysis,
                              int vector_size, LowerConfig* lower_config) {
  LHLO_OpTy op = dyn_cast<LHLO_OpTy>(opaque_op);
  if (!op) return failure();
  Value result_memref = cast<lmhlo::LmhloOp>(&*op).getResultBuffer();
  Value memref = result_memref;
  // TODO(disc): we actually don't need this after shape constraint refactor
  // Remove this part of code after we finish the benchmark after the refactor.
  if (auto analysis_deprecated =
          dynamic_cast<const ShapeAnalysisDeprecated*>(shape_analysis)) {
    lmhlo::FusionOp fusion = opaque_op->getParentOfType<lmhlo::FusionOp>();
    Value leader_memref =
        analysis_deprecated->GetLeaderValueWithSameShapeInFusion(fusion,
                                                                 result_memref);
    if (leader_memref != nullptr) {
      memref = leader_memref;
    }
  }

  Value vec_size = b.create<arith::ConstantIndexOp>(loc, vector_size);
  if (vector_size > 1) {
    output_linear_index =
        b.create<arith::MulIOp>(loc, output_linear_index, vec_size);
  }

  // TODO: find a way to unroll a loop and reorder instructions automatically.
  SmallVector<Value, 4> linear_indices(vector_size);
  for (int i = 0; i < vector_size; i++) {
    linear_indices[i] = b.create<arith::AddIOp>(
        loc, output_linear_index, b.create<arith::ConstantIndexOp>(loc, i));
  }

  SmallVector<SmallVector<Value>> multidim_index_vector(vector_size);
  for (int64_t i = 0; i < vector_size; i++) {
    Value linear_index = linear_indices[i];
    auto multidim_index = calcMultiDimIndex(&b, loc, linear_index, memref);
    multidim_index_vector[i] = std::move(multidim_index);
  }

  SmallVector<Value, 4> operand_datas(vector_size);
  for (int i = 0; i < vector_size; i++) {
    operand_datas[i] = elementalLower(&b, loc, op, multidim_index_vector[i],
                                      /*check_cache=*/true, lower_config);
  }
  if (vector_size == 1) {
    for (int i = 0; i < vector_size; i++) {
      b.create<memref::StoreOp>(loc, operand_datas[i], result_memref,
                                multidim_index_vector[i]);
    }
    return success();
  }

  Value memref_1d = createMemRef1DReinterpretCast(b, loc, result_memref);
  // Assuming alignment is to help vectorization optimization. Vectorization
  // pass in llvm requires an alignment of `data-byte-width * vector-size`.
  if (vector_size > 1) {
    createAlignMemrefWithTile(b, memref_1d, vector_size);
  }
  // For vectorize optimization.
  for (int i = 0; i < vector_size; i++) {
    b.create<memref::StoreOp>(loc, operand_datas[i], memref_1d,
                              ValueRange{linear_indices[i]});
  }

  return success();
}

template <typename First>
LogicalResult elemwiseLowerHelperOr(OpBuilder& b, Location loc, Operation* op,
                                    Value output_linear_index,
                                    const ShapeAnalysis* shape_analysis,
                                    int vector_size,
                                    LowerConfig* lower_config) {
  return elemwiseLowerHelper<First>(b, loc, op, output_linear_index,
                                    shape_analysis, vector_size, lower_config);
}

template <typename First, typename Second, typename... Rest>
LogicalResult elemwiseLowerHelperOr(OpBuilder& b, Location loc, Operation* op,
                                    Value output_linear_index,
                                    const ShapeAnalysis* shape_analysis,
                                    int vector_size,
                                    LowerConfig* lower_config) {
  return success(succeeded(elemwiseLowerHelperOr<First>(
                     b, loc, op, output_linear_index, shape_analysis,
                     vector_size, lower_config)) ||
                 succeeded(elemwiseLowerHelperOr<Second, Rest...>(
                     b, loc, op, output_linear_index, shape_analysis,
                     vector_size, lower_config)));
}

LogicalResult lowerHelper(OpBuilder& b, Location loc, Operation* op,
                          Value output_linear_index,
                          const ShapeAnalysis* shape_analysis,
                          int vector_size = 1,
                          LowerConfig* lower_config = nullptr) {
  if (succeeded(elemwiseLowerHelperOr<
#define GET_SUPPORTED_OP_LIST
#include "mlir/disc/transforms/disc_supported_list.h.inc"
                >(b, loc, op, output_linear_index, shape_analysis, vector_size,
                  lower_config)) ||
      // clang-format off
      // TODO(disc): Upstream is on the way for more Ops
      succeeded(miscLowerHelper<lmhlo::SliceOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::RealDynamicSliceOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::DynamicBroadcastInDimOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::BroadcastInDimOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::BroadcastOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::NotOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::ClampOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::DynamicReshapeOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::TransposeOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::DynamicPadOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::GatherOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::DynamicGatherOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::IsFiniteOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::ConcatenateOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo_disc::ConcatenateOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::CopyOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::DynamicIotaOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::IotaOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::ReduceOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::ReshapeOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo_disc::H2DOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config)) ||
      succeeded(miscLowerHelper<lmhlo::ReverseOp>(
          b, loc, op, output_linear_index, shape_analysis, vector_size, lower_config))
    ) {
    return success();
    // clang-format on
  }
  return failure();
}

// we don't do inbound check for kLoop Schedule
// LoopSplit pass will do this.
//
/* %num_elements = ElementsIn(root_shape)
 * loop.for %idx = 0 to %num_elements step 1 {
 *   %multidim_indices_0..n = getMultidimIndices(%idx);
 *   %operand_0 = load %operand0[]
 *   %operand_1 = load %operand1[]
 *   emit calculation..
 * }
 */
LogicalResult lowerWithScheduleLoop(
    ArrayRef<Operation*> root_ops, Operation* dominant_op,
    Block* parent = nullptr, bool non_fusion = false, bool parallel_loop = true,
    const ShapeAnalysis* shape_analysis = nullptr, int vector_size = 1) {
  const auto loc = dominant_op->getLoc();
  OpBuilder b(root_ops.back());
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value vec_size = b.create<arith::ConstantIndexOp>(loc, vector_size);
  auto num_elements = emitNumElementsComputation(b, loc, dominant_op);
  auto thread_number = b.create<arith::DivUIOp>(loc, num_elements, vec_size);
  Value var;
  if (parallel_loop) {
    SmallVector<Value, 2> vars;
    (void)createParallelAndSetInsPt(b, loc, vars, {zero}, {thread_number},
                                    {one}, {});
    var = vars[0];
  } else {
    (void)createLoopAndSetInsPt(b, loc, var, zero, thread_number, one, {});
  }
  for (Operation* root_op : root_ops) {
    // TODO: vectorize here
    if (failed(lowerHelper(b, loc, root_op, var, shape_analysis, vector_size)))
      return failure();
  }
  // remove the root_op if it has no other users except the memref
  if (non_fusion) {
    for (Operation* root_op : root_ops) root_op->erase();
  } else {
    assert(parent != nullptr && "Parent must be provided for fusion lowering");
    cleanUnusedLhloOps(parent);
  }
  return success();
}

LogicalResult lowerWithScheduleLoopV2(
    ArrayRef<Operation*> root_ops, Operation* dominant_op,
    Block* parent = nullptr, const ShapeAnalysis* shape_analysis = nullptr,
    int vector_size = 1) {
  const auto loc = dominant_op->getLoc();
  OpBuilder b(root_ops.back());
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value vec_size = b.create<arith::ConstantIndexOp>(loc, vector_size);
  auto num_elements = emitNumElementsComputation(b, loc, dominant_op);
  auto thread_number = b.create<arith::DivUIOp>(loc, num_elements, vec_size);
  Value var;
  SmallVector<Value, 2> vars;
  scf::ParallelOp simt_loop = createParallelAndSetInsPt(
      b, loc, vars, {zero}, {thread_number}, {one}, {});
  Value output_linear_index = vars[0];

  if (vector_size > 1) {
    // This loop will be unrolled and interleaved for vectorization.
    scf::ForOp vec_loop =
        b.create<scf::ForOp>(loc, zero, vec_size, one, ValueRange({}));
    Value vec_id = vec_loop.getInductionVar();
    vec_loop.getBody()->clear();
    b.setInsertionPointToStart(vec_loop.getBody());
    output_linear_index = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, output_linear_index, vec_size),
        vec_id);
  }
  for (Operation* root_op : root_ops) {
    if (failed(lowerHelper(b, loc, root_op, output_linear_index, shape_analysis,
                           1))) {
      return failure();
    }
  }
  if (vector_size > 1) {
    b.create<scf::YieldOp>(loc, ValueRange({}));
  }

  // remove the root_op if it has no other users except the memref
  assert(parent != nullptr && "Parent must be provided for fusion lowering");
  cleanUnusedLhloOps(parent);

  return success();
}

Operation* getMapOpInReduceRegion(Operation* op) {
  if (!op || !isa<lmhlo::ReduceOp>(op)) {
    return nullptr;
  }
  auto reduce_op = cast<lmhlo::ReduceOp>(op);
  Operation* map_op = nullptr;
  int num_lhlo_ops = 0;
  reduce_op.getBody().walk([&](Operation* lhlo_op) {
    if (isa<lmhlo::TerminatorOp>(lhlo_op)) {
      return;
    }
    if (lhlo_op->getDialect() == op->getDialect()) {
      ++num_lhlo_ops;
      map_op = lhlo_op;
    }
  });
  assert(num_lhlo_ops == 1 && "not support reduce region");
  return map_op;
}

template <typename SrcTy, typename OpIntTy, typename OpFloatTy>
struct MapOpCreator {
  static Value build(OpBuilder& b, Location loc, Value lhs, Value rhs) {
    Value result;
    if (lhs.getType().dyn_cast<IntegerType>()) {
      result = b.create<OpIntTy>(loc, lhs, rhs);
    } else if (lhs.getType().dyn_cast<FloatType>()) {
      result = b.create<OpFloatTy>(loc, lhs, rhs);
    } else {
      assert(false && "not supported data type");
    }
    return result;
  }
};

template <>
struct MapOpCreator<lmhlo::MaxOp, arith::CmpIOp, arith::CmpFOp> {
  static Value build(OpBuilder& b, Location loc, Value lhs, Value rhs) {
    Value result;
    if (lhs.getType().dyn_cast<IntegerType>()) {
      Value cond =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, lhs, rhs);
      result =
          b.create<mlir::arith::SelectOp>(loc, lhs.getType(), cond, lhs, rhs);
    } else if (lhs.getType().dyn_cast<FloatType>()) {
      Value cond =
          b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, lhs, rhs);
      result =
          b.create<mlir::arith::SelectOp>(loc, lhs.getType(), cond, lhs, rhs);
    } else {
      assert(false && "not supported data type");
    }
    return result;
  }
};

template <>
struct MapOpCreator<lmhlo::MinOp, arith::CmpIOp, arith::CmpFOp> {
  static Value build(OpBuilder& b, Location loc, Value lhs, Value rhs) {
    Value result;
    if (lhs.getType().dyn_cast<IntegerType>()) {
      Value cond =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, lhs, rhs);
      result =
          b.create<mlir::arith::SelectOp>(loc, lhs.getType(), cond, rhs, lhs);
    } else if (lhs.getType().dyn_cast<FloatType>()) {
      Value cond =
          b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, lhs, rhs);
      result =
          b.create<mlir::arith::SelectOp>(loc, lhs.getType(), cond, rhs, lhs);
    } else {
      assert(false && "not supported data type");
    }
    return result;
  }
};

Value emitReduceMapOp(OpBuilder& b, Location loc, Operation* map_op, Value lhs,
                      Value rhs) {
  Value result;
  assert(lhs.getType() == rhs.getType() && "mismatch operands' type");
  if (isa<lmhlo::AddOp>(map_op)) {
    result = MapOpCreator<lmhlo::AddOp, arith::AddIOp, arith::AddFOp>::build(
        b, loc, lhs, rhs);
  } else if (isa<lmhlo::MulOp>(map_op)) {
    result = MapOpCreator<lmhlo::MulOp, arith::MulIOp, arith::MulFOp>::build(
        b, loc, lhs, rhs);
  } else if (isa<lmhlo::MaxOp>(map_op)) {
    result = MapOpCreator<lmhlo::MaxOp, arith::CmpIOp, arith::CmpFOp>::build(
        b, loc, lhs, rhs);
  } else if (isa<lmhlo::MinOp>(map_op)) {
    result = MapOpCreator<lmhlo::MinOp, arith::CmpIOp, arith::CmpFOp>::build(
        b, loc, lhs, rhs);
  } else {
    assert(false && "not support reduction map type");
  }
  return result;
}

void emitNotToVectorReduction(OpBuilder& b, Location loc, Operation* root_op,
                              ValueRange index) {
  assert(isa<lmhlo::ReduceOp>(root_op));
  unsigned int input_rank =
      root_op->getOperand(0).getType().cast<MemRefType>().getRank();
  auto reduce_op = cast<lmhlo::ReduceOp>(root_op);
  if (input_rank <= 2) {
    llvm::dbgs() << "Error: unexpected isReduceToVector "
                 << "ReduceOp in loop schedule";
    assert(false && "unexpected isReduceToVector ReduceOp in loop schedule");
  }
  Type root_element_type = getLhloOpsElementType(root_op);
  Value data = createLoadOrUseCachedValue(loc, &b, root_op,
                                          *root_op->getOperands().begin(),
                                          index, b.saveInsertionPoint());
  SmallVector<Value, 4> output_multidim_index;
  auto dimensions = reduce_op.getDimensions().getValues<int64_t>();
  for (auto idx : llvm::enumerate(index)) {
    if (std::find(dimensions.begin(), dimensions.end(), idx.index()) ==
        dimensions.end()) {
      output_multidim_index.push_back(idx.value());
    }
  }
  b.create<memref::AtomicRMWOp>(loc, root_element_type,
                                getAtomicRMWKind(reduce_op.getBody()), data,
                                root_op->getOperand(2), output_multidim_index);
}

// BLOCK TILE LOOP IMPL: thread_num threads load thread_num * tile_h elements
// each thread reduces tile_h elements from gmem and atomic to gmem
// var_tile_h = 32;
// var_threads = 512;
// num_blocks_col = ceil(var_cols / var_threads);
// num_blocks_row = ceil(var_rows / var_tile_h);
// for (m = 0; m < num_blocks_col * num_blocks_row; ++m) {
//   for (n = 0; n < var_threads; ++n) {
//     local_col_index = n;
//     block_row_index = m / num_blocks_col;
//     block_col_index = m % num_blocks_col;
//     col_index = block_col_index * var_threads + local_col_index;
//     is_col_valid = col_index < var_cols;
//     if (is_col_valid) {
//       accum = init_value;
//       for (int l = 0; l < var_tile_h; ++l) {
//         row_index = block_row_index * var_tile_h + l;
//         is_row_valid = row_index < var_rows;
//         if (is_row_valid) {
//           accum = accum + global[row_index, col_index];
//         } else {
//           accum = accum;
//         }
//       }
//     } else {
//     }
//     if (is_col_valid && is_row_valid) {
//       atomicAdd(&global[col_index], accum);
//     }
//   }
// }
template <int THREAD_NUM, int TILE_H>
LogicalResult lowerWithScheduleColReduction(
    ArrayRef<Operation*> root_ops, Operation* dominant_op, Block* parent,
    const ShapeAnalysis* shape_analysis = nullptr) {
  if (!isRank2ColReduction(dominant_op)) {
    return failure();
  }

  // Create helper Values
  SmallVector<Operation*, 4> col_reduction_roots;
  std::copy_if(
      root_ops.begin(), root_ops.end(), std::back_inserter(col_reduction_roots),
      [](Operation* operation) { return isRank2ColReduction(operation); });

  Value lhs = *dominant_op->getOperands().begin();
  const int kTileH = TILE_H;        // 32
  const int kThreads = THREAD_NUM;  // 512

  Location loc = dominant_op->getLoc();
  OpBuilder b(root_ops.back());
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value var_rows = b.create<memref::DimOp>(loc, lhs, zero);
  Value var_cols = b.create<memref::DimOp>(loc, lhs, one);
  // Start to emit.

  // num_blocks_col = ceil(var_cols / var_threads);
  // num_blocks_row = ceil(var_rows / var_tile_h);
  Value var_threads = b.create<arith::ConstantIndexOp>(loc, kThreads);
  Value var_tile_h = b.create<arith::ConstantIndexOp>(loc, kTileH);
  Value num_blocks_col =
      b.create<arith::CeilDivUIOp>(loc, var_cols, var_threads);
  Value num_blocks_row =
      b.create<arith::CeilDivUIOp>(loc, var_rows, var_tile_h);
  Value var_blocks =
      b.create<arith::MulIOp>(loc, num_blocks_col, num_blocks_row);

  // for (m = 0; m < num_blocks_col * num_blocks_row; ++m) {
  //   for (n = 0; n < var_threads; ++n) {
  SmallVector<Value, 2> vars;
  scf::ParallelOp parallel_op = createParallelAndSetInsPt(
      b, loc, vars, {zero, zero}, {var_blocks, var_threads}, {one, one}, {});
  parallel_op.getBody()->clear();
  b.setInsertionPointToStart(parallel_op.getBody());
  Value var_m = vars[0];
  Value var_n = vars[1];

  // acc: init_values[num_col_reductions]
  SmallVector<Value, 4> init_values(col_reduction_roots.size());
  SmallVector<Type, 4> init_values_types(col_reduction_roots.size());
  for (auto root_pair : llvm::enumerate(col_reduction_roots)) {
    Operation* root_op = root_pair.value();
    int idx = root_pair.index();
    Value init_value = b.create<memref::LoadOp>(
        loc, cast<lmhlo::ReduceOp>(root_op).getInitValues()[0]);
    init_values[idx] = init_value;
    init_values_types[idx] = init_value.getType();
  }

  // local_col_index = n;
  // block_row_index = m / num_blocks_col;
  // block_col_index = m % num_blocks_col;
  // col_index = block_col_index * var_threads + local_col_index;
  Value block_row_index = b.create<arith::DivUIOp>(loc, var_m, num_blocks_col);
  Value block_col_index = b.create<arith::RemUIOp>(loc, var_m, num_blocks_col);
  Value col_index = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, block_col_index, var_threads), var_n);

  // is_col_valid = col_index < var_cols;
  Value is_lt_cols = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                             col_index, var_cols);
  scf::IfOp if_col_valid_op =
      b.create<scf::IfOp>(loc, /*resultTypes*/ init_values_types, is_lt_cols,
                          /*hasElseRegion*/ true);
  if_col_valid_op.getThenRegion().front().clear();
  if_col_valid_op.getElseRegion().front().clear();

  b.setInsertionPointToStart(&if_col_valid_op.getThenRegion().front());
  // if (is_col_valid) {
  //   accum = init_value;
  //   for (int l = 0; l < tile_size; ++l) {
  //     row_index = block_row_index * var_tile_h + l;
  //     is_row_valid = row_index < var_rows;
  //     if (is_row_valid) {
  //       accum = accum + global[row_index, col_index];
  //     }
  //     else {
  //       accum = accum;
  //     }
  //   }
  // }
  scf::ForOp for_op_l =
      b.create<scf::ForOp>(loc, zero, var_tile_h, one, init_values);
  for_op_l.getBody()->clear();
  b.setInsertionPointToStart(for_op_l.getBody());
  Value var_l = for_op_l.getInductionVar();
  Value row_index = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, block_row_index, var_tile_h), var_l);
  Value is_lt_rows = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                             row_index, var_rows);
  scf::IfOp if_row_valid_op =
      b.create<scf::IfOp>(loc, /*resultTypes*/ init_values_types, is_lt_rows,
                          /*hasElseRegion*/ true);
  if_row_valid_op.getThenRegion().front().clear();
  if_row_valid_op.getElseRegion().front().clear();

  SmallVector<Value, 4> yield_values_for_if;

  ValueRange load_index({row_index, col_index});
  b.setInsertionPointToStart(&if_row_valid_op.getThenRegion().front());
  int col_red_root_op_idx = 0;
  for (auto* root_op : root_ops) {
    if (isRank2ColReduction(root_op)) {
      auto lhs = root_op->getOperands().begin();
      Value data = createLoadOrUseCachedValue(
          loc, &b, root_op, *lhs, load_index, b.saveInsertionPoint());
      Operation* map_op = getMapOpInReduceRegion(root_op);
      assert(map_op && "not supported reduce");
      auto acc = emitReduceMapOp(
          b, loc, map_op,
          *(for_op_l.getRegionIterArgs().begin() + col_red_root_op_idx), data);
      yield_values_for_if.push_back(acc);
      col_red_root_op_idx++;
    } else if (isRank2RowReduction(root_op)) {
      assert(false && "unexpected row_reduction");
      return failure();
    } else if (isa<lmhlo::ReduceOp>(root_op)) {
      auto dominant_shape = getShapeValues(&b, dominant_op->getOperand(0));
      Value linear_index = calcLinearIndex(&b, loc, load_index, dominant_shape);
      auto root_shape = getShapeValues(&b, root_op->getOperand(0));
      auto mapped_index = calcMultiDimIndex(&b, loc, linear_index, root_shape);
      emitNotToVectorReduction(b, loc, root_op, mapped_index);
    } else {
      auto dominant_shape = getShapeValues(&b, dominant_op->getOperand(0));
      Value linear_index = calcLinearIndex(&b, loc, load_index, dominant_shape);
      auto root_shape = getShapeValues(&b, root_op->getOperand(0));
      auto mapped_index = calcMultiDimIndex(&b, loc, linear_index, root_shape);
      if (!succeeded(
              lowerHelper(b, loc, root_op, linear_index, shape_analysis))) {
        return failure();
      }
    }
  }
  b.setInsertionPointToEnd(&if_row_valid_op.getThenRegion().front());
  b.create<scf::YieldOp>(loc, yield_values_for_if);

  b.setInsertionPointToEnd(&if_row_valid_op.getElseRegion().front());
  b.create<scf::YieldOp>(loc, for_op_l.getRegionIterArgs());

  b.setInsertionPointToEnd(for_op_l.getBody());
  b.create<scf::YieldOp>(loc, if_row_valid_op.getResults());

  b.setInsertionPointToEnd(&if_col_valid_op.getThenRegion().front());
  b.create<scf::YieldOp>(loc, for_op_l.getResults());

  b.setInsertionPointToStart(&if_col_valid_op.getElseRegion().front());
  b.create<scf::YieldOp>(loc, init_values);

  b.setInsertionPointAfter(if_col_valid_op);

  scf::IfOp if_valid_op =
      b.create<scf::IfOp>(loc, /*resultTypes*/ TypeRange{}, is_lt_cols,
                          /*hasElseRegion*/ false);
  if_valid_op.getThenRegion().front().clear();
  b.setInsertionPointToStart(&if_valid_op.getThenRegion().front());

  for (auto root_pair : llvm::enumerate(col_reduction_roots)) {
    Operation* root_op = root_pair.value();
    int idx = root_pair.index();
    Value acc = if_col_valid_op.getResults()[idx];
    auto root_element_type = getLhloOpsElementType(root_op);
    b.create<memref::AtomicRMWOp>(
        loc, root_element_type,
        getAtomicRMWKind(cast<lmhlo::ReduceOp>(root_op).getBody()), acc,
        root_op->getOperand(2), ValueRange({col_index}));
  }
  b.create<scf::YieldOp>(loc, ValueRange({}));

  b.setInsertionPointToEnd(parallel_op.getBody());
  b.create<scf::YieldOp>(loc, ValueRange({}));  // parallel

  // remove the root_op if it has no other users except the memref
  cleanUnusedLhloOps(parent);
  return success();
}

// we can emit them in one kernel based on the fact that all the fused
// col reduction should be in the same shape.
void emitInitLoops(OpBuilder& b, ArrayRef<Operation*> col_reduction_ops) {
  Operation* first_root = col_reduction_ops[0];
  Location loc = first_root->getLoc();
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value num_elements = emitNumElementsComputation(b, loc, first_root);

  Value var;
  SmallVector<Value, 2> vars;
  createParallelAndSetInsPt(b, loc, vars, {zero}, {num_elements}, {one}, {});
  var = vars[0];

  for (Operation* reduce : col_reduction_ops) {
    Value result_memref = reduce->getOperand(reduce->getNumOperands() - 1);
    auto multidim_index = calcMultiDimIndex(&b, loc, var, result_memref);
    Value init_value = b.create<memref::LoadOp>(
        loc, cast<lmhlo::ReduceOp>(reduce).getInitValues().front());
    b.create<memref::StoreOp>(loc, init_value, result_memref, multidim_index);
  }
  return;
}

void maybeEmitInitLoops(OpBuilder& b,
                        const SmallVectorImpl<Operation*>& root_ops) {
  b.setInsertionPoint(root_ops.back());
  SmallVector<Operation*, 4> col_reduction_ops;
  for (Operation* root_op : root_ops) {
    if (isRank2ColReduction(root_op)) {
      col_reduction_ops.emplace_back(root_op);
    }
  }
  if (col_reduction_ops.size() > 0) {
    emitInitLoops(b, col_reduction_ops);
  }
}

LogicalResult getShuffleElemType(OpBuilder& b, Type elemType,
                                 Type* shuffleElemType) {
  if (auto int_type = elemType.dyn_cast<IntegerType>()) {
    if (int_type.getWidth() < 32) {
      *shuffleElemType = b.getIntegerType(32);
    } else {
      *shuffleElemType = elemType;
    }
  } else if (auto fp_type = elemType.dyn_cast<FloatType>()) {
    std::size_t width = fp_type.getWidth();
    if (width != 16 && width != 32 && width != 64) {
      llvm::dbgs() << "not supported row reduction fp" << width << "\n";
      return failure();
    }
    *shuffleElemType = elemType;
  } else {
    llvm::dbgs() << "not supported row reduction type"
                 << "\n";
    return failure();
  }
  return success();
}

Value emitWidthAdaptShuffle(OpBuilder& b, Location loc, Value value,
                            Type shuffleElemType, Value offset, Value width,
                            gpu::ShuffleMode strAttr) {
  auto bit_width = shuffleElemType.getIntOrFloatBitWidth();
  if (bit_width < 32) {
    // TODO: modify GPU dialect to support fp16 shuffle.
    if (shuffleElemType.isa<FloatType>()) {
      auto f32_ty = b.getF32Type();
      SmallVector<Type, 2> type = {f32_ty, b.getI1Type()};
      Value ext = b.create<arith::ExtFOp>(loc, f32_ty, value);
      auto result =
          b.create<gpu::ShuffleOp>(loc, type, ext, offset, width, strAttr)
              .getResult(0);
      return b.create<arith::TruncFOp>(loc, shuffleElemType, result);
    } else if (shuffleElemType.isa<IntegerType>()) {
      auto i32_ty = b.getIntegerType(32);
      SmallVector<Type, 2> type = {i32_ty, b.getI1Type()};
      Value extend;
      // Special case boolean values, so they get casted to `1` instead of `-1`.
      if (shuffleElemType.isUnsignedInteger() || bit_width == 1) {
        extend = b.create<arith::ExtUIOp>(loc, i32_ty, value);
      } else {
        extend = b.create<arith::ExtSIOp>(loc, i32_ty, value);
      }
      auto result =
          b.create<gpu::ShuffleOp>(loc, type, extend, offset, width, strAttr)
              .getResult(0);
      return b.create<arith::TruncIOp>(loc, shuffleElemType, result);
    }
  } else if (bit_width == 32) {
    SmallVector<Type, 2> type = {shuffleElemType, b.getI1Type()};
    return b.create<gpu::ShuffleOp>(loc, type, value, offset, width, strAttr)
        .getResult(0);
  } else {
    // The following codes used to imitate shuffle of integers has bits larger
    // than 32.
    // ```
    // int bit_width = bit_width(val);
    // int segments = ceil(bit_width/32);
    // auto val_vec_i32 = bitcast(val, vec(segments, i32));
    // for (int i = 0; i < segments; ++i) {
    //   auto insert_elem = extract_element(val_vec_i32, i);
    //   insert_elem = __xhfl_xor(insert_elem, offset);
    //   insert_element(val_vec_i32, insert_elem, i);
    // }
    // val = bitcast(val_vec_i32, integer(bit_width)))
    // ```
    SmallVector<Type, 2> type = {b.getIntegerType(32), b.getI1Type()};
    int segments = llvm::divideCeil(bit_width, 32);
    auto vec_ty = VectorType::get(segments, b.getIntegerType(32));
    // TODO(yancey.yx): when std.bitcast supports casting a value into an array,
    // using std.bitcast instead of LLVM dialect.
    Value x = b.create<mlir::LLVM::BitcastOp>(loc, vec_ty, value);
    for (int64_t sgt = 0; sgt < segments; ++sgt) {
      Value sgt_idx = b.create<arith::ConstantIntOp>(loc, sgt, /*bitwidth=*/32);
      Value insert_val =
          b.create<mlir::LLVM::ExtractElementOp>(loc, x, sgt_idx);
      insert_val = b.create<gpu::ShuffleOp>(loc, type, insert_val, offset,
                                            width, strAttr)
                       .getResult(0);
      x = b.create<mlir::LLVM::InsertElementOp>(loc, vec_ty, x, insert_val,
                                                sgt_idx);
    }
    return b.create<mlir::LLVM::BitcastOp>(loc, shuffleElemType, x);
  }
}

Value createSharedMemory(OpBuilder& b, Location loc, int64_t size,
                         Type elem_type) {
  auto workgroupMemoryAddressSpace = gpu::AddressSpaceAttr::get(
      b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  auto bufferType =
      MemRefType::get(ArrayRef<int64_t>{size}, elem_type,
                      MemRefLayoutAttrInterface{}, workgroupMemoryAddressSpace);
  auto alloc = b.create<memref::AllocOp>(loc, bufferType);
  return alloc.getResult();
}

LogicalResult emitPerElementReductionImpl(
    OpBuilder& b, Location loc, Operation* op,
    mlir::Block::args_iterator acc_iter, ValueRange& input_index,
    SmallVector<Value, 4>& yield_values_for_j) {
  auto reduce_op = dyn_cast_or_null<lmhlo::ReduceOp>(op);
  if (reduce_op == nullptr) {
    return failure();
  }
  auto data = createLoadOrUseCachedValue(loc, &b, op, op->getOperand(0),
                                         input_index, b.saveInsertionPoint());
  AccumulatorFactory accumFactory =
      getFactory(b, op->getLoc(), reduce_op.getBody());
  auto acc = accumFactory(*acc_iter, data);
  yield_values_for_j.push_back(acc);
  return success();
}

// emit in-thread reduction, and other non-dominant nodes including
// elementwise and column reduction
LogicalResult emitInThreadReduction(
    OpBuilder& b, Location loc, const std::vector<Operation*>& root_ops,
    mlir::Block::args_iterator acc_iter, ValueRange& input_index,
    Value col_reduction_index, SmallVector<Value, 4>& yield_values_for_j,
    Operation* dominant_op, const ShapeAnalysis* shape_analysis) {
  int64_t row_reduction_idx = 0;
  for (auto root_op : root_ops) {
    if (isRank2RowReduction(root_op)) {
      if (failed(emitPerElementReductionImpl(
              b, loc, root_op, acc_iter + row_reduction_idx, input_index,
              yield_values_for_j))) {
        return failure();
      }
      row_reduction_idx++;
    } else if (isRank2ColReduction(root_op)) {
      auto root_element_type = getLhloOpsElementType(root_op);
      auto data = createLoadOrUseCachedValue(
          loc, &b, root_op, *root_op->getOperands().begin(), input_index,
          b.saveInsertionPoint());
      b.create<memref::AtomicRMWOp>(
          loc, root_element_type,
          getAtomicRMWKind(cast<lmhlo::ReduceOp>(root_op).getBody()), data,
          root_op->getOperand(2), ValueRange({col_reduction_index}));
    } else if (isa<lmhlo::ReduceOp>(root_op)) {
      auto dominant_shape = getShapeValues(&b, dominant_op->getOperand(0));
      Value linear_index =
          calcLinearIndex(&b, loc, input_index, dominant_shape);
      auto root_shape = getShapeValues(&b, root_op->getOperand(0));
      auto mapped_index = calcMultiDimIndex(&b, loc, linear_index, root_shape);
      emitNotToVectorReduction(b, loc, root_op, mapped_index);
    } else {
      auto dominant_shape = getShapeValues(&b, dominant_op->getOperand(0));
      Value linear_index =
          calcLinearIndex(&b, loc, input_index, dominant_shape);
      if (!succeeded(
              lowerHelper(b, loc, root_op, linear_index, shape_analysis))) {
        return failure();
      }
    }
  }
  return success();
}

template <DiscRowReductionScheduleType schedule>
LogicalResult lowerWithScheduleRowReduction(ArrayRef<Operation*>, Operation*,
                                            Block*,
                                            const ShapeAnalysis* shape_analysis,
                                            int vector_size = 1) {
  return failure();
}

/* Row reduction with 1 round warp shuffle
 *
 * RowPerBlock = threads / warpSize;
 * for (m = 0; m < rows; m += RowPerBlock) {
 *   for (n = 0; n < threads; ++n) {
 *     rowIdx = m + warpIdx;
 *     if (rowIdx < rows) {
 *       // intra-thread reduction
 *       sum = init_value;
 *       for (k = laneIdx; k < cols; k += warpSize) {
 *         sum += inputs[rowIdx][k];
 *       }
 *
 *       // inter-thread reduction via warp shuffle
 *       for (offset = warpSize / 2; offset > 0; offset /= 2) {
 *         sum += __shfl_xor(sum, offset);
 *       }
 *
 *       // write to output
 *       if (laneIdx == 0) {
 *         outputs[rowIdx] = sum
 *       }
 *     }
 *   }
 */
template <>
LogicalResult lowerWithScheduleRowReduction<DISC_WARP_WISE_ROW_REDUCE>(
    ArrayRef<Operation*> root_ops, Operation* dominant_op, Block* parent,
    const ShapeAnalysis* shape_analysis, int vector_size) {
  if (!isRank2RowReduction(dominant_op)) {
    return failure();
  }

  // Create helper Values
  SmallVector<Operation*, 4> row_reduction_roots;
  std::copy_if(
      root_ops.begin(), root_ops.end(), std::back_inserter(row_reduction_roots),
      [](Operation* operation) { return isRank2RowReduction(operation); });

  const int thread_per_block = getThreadPerBlock(dominant_op);
  Location loc = dominant_op->getLoc();
  OpBuilder b(root_ops.back());

  Value lhs = dominant_op->getOperand(0);
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value rows = b.create<memref::DimOp>(loc, lhs, zero);
  Value cols = b.create<memref::DimOp>(loc, lhs, one);
  Value threads = b.create<arith::ConstantIndexOp>(loc, thread_per_block);
  Value warp_size = b.create<arith::ConstantIndexOp>(loc, kWarpSize);
  Value row_per_block = b.create<arith::ConstantIndexOp>(
      loc, thread_per_block / kWarpSize * vector_size);

  // Start to emit.

  // for (m = 0; m < rows; m += RowPerBlock)
  //   for (n = 0; n < threads; ++n)
  SmallVector<Value, 2> vars;
  scf::ParallelOp parallel_op = createParallelAndSetInsPt(
      b, loc, vars, {zero, zero}, {rows, threads}, {row_per_block, one}, {});
  parallel_op.getBody()->clear();
  b.setInsertionPointToStart(parallel_op.getBody());
  Value var_m = vars[0];
  Value var_n = vars[1];

  Value lane_id = b.create<arith::RemUIOp>(loc, var_n, warp_size);
  Value warp_id = b.create<arith::DivUIOp>(loc, var_n, warp_size);

  // rowIdx = m + warpIdx;
  // if (rowIdx < rows)
  Value vec_size = b.create<arith::ConstantIndexOp>(loc, vector_size);
  Value row_base = b.create<arith::AddIOp>(
      loc, var_m, b.create<arith::MulIOp>(loc, warp_id, vec_size));
  SmallVector<Value, 4> row_ids(vector_size);
  for (int i = 0; i < vector_size; i++) {
    row_ids[i] = b.create<arith::AddIOp>(
        loc, row_base, b.create<arith::ConstantIndexOp>(loc, i));
  }
  // Note that we have already checked that it can be vectorized. Thus if one
  // row is valid, all other rows will be valid.
  Value is_valid_row_id =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, row_ids[0], rows);
  scf::IfOp if_is_valid_row_id =
      b.create<scf::IfOp>(loc, /* resultTypes */ ArrayRef<Type>{},
                          is_valid_row_id, /*hasElseRegion*/ false);
  if_is_valid_row_id.getThenRegion().front().clear();
  b.setInsertionPointToStart(&if_is_valid_row_id.getThenRegion().front());
  {
    // sum = init_value;
    // for (k = laneIdx; k < cols; k += warpSize) {
    //   sum += inputs[rowIdx][k];
    // }
    SmallVector<Value, 4> init_values(row_reduction_roots.size() * vector_size);
    for (auto root_pair : llvm::enumerate(row_reduction_roots)) {
      Operation* root_op = root_pair.value();
      int idx = root_pair.index();
      Value init_value = b.create<memref::LoadOp>(
          loc, cast<lmhlo::ReduceOp>(root_op).getInitValues()[0]);
      for (int i = 0; i < vector_size; i++) {
        init_values[i * row_reduction_roots.size() + idx] = init_value;
      }
    }
    Value var_k = nullptr;
    scf::ForOp for_op_k =
        createLoopAndSetInsPt(b, loc, var_k, /* lb */ lane_id,
                              /* ub */ cols, /* step */ warp_size, init_values);
    SmallVector<Value, 4> yield_values;
    for (int i = 0; i < vector_size; i++) {
      SmallVector<Value, 2> multidim_load_index({row_ids[i], var_k});
      ValueRange load_index(multidim_load_index);
      // Emit for all root ops; all root ops's input tensors have the same
      // shape, and the same access index; the access index is calculated by
      // dominant op.
      if (failed(emitInThreadReduction(b, loc, root_ops,
                                       for_op_k.getRegionIterArgs().begin() +
                                           i * row_reduction_roots.size(),
                                       load_index, var_k, yield_values,
                                       dominant_op, shape_analysis))) {
        return failure();
      }
    }
    b.create<scf::YieldOp>(
        loc, yield_values);  // for (k = laneIdx; k < cols; k += warpSize)
    b.setInsertionPointToEnd(&if_is_valid_row_id.getThenRegion().front());

    SmallVector<Value, 4> shuffle_val(row_reduction_roots.size() * vector_size);
    SmallVector<Type, 4> shuffle_type(row_reduction_roots.size());
    SmallVector<AccumulatorFactory, 4> accum_factory(
        row_reduction_roots.size());
    for (auto root_pair : llvm::enumerate(row_reduction_roots)) {
      Operation* root_op = root_pair.value();
      int idx = root_pair.index();
      for (int i = 0; i < vector_size; i++) {
        shuffle_val[i * row_reduction_roots.size() + idx] =
            for_op_k.getResult(i * row_reduction_roots.size() + idx);
      }

      const auto elem_type = getLhloOpsElementType(root_op);
      Type shuffle_elem_type;
      if (failed(getShuffleElemType(b, elem_type, &shuffle_elem_type))) {
        return failure();
      }
      shuffle_type[idx] = shuffle_elem_type;

      accum_factory[idx] = std::move(getFactory(
          b, root_op->getLoc(), cast<lmhlo::ReduceOp>(root_op).getBody()));
    }

    // for (offset = 1; offset < warpSize; offset *= 2) {
    //   sum += __shfl_xor(sum, offset);
    // }
    Value shuffle_width =
        b.create<arith::ConstantIntOp>(loc, kWarpSize, b.getIntegerType(32));
    for (int offset = 1; offset < kWarpSize; offset *= 2) {
      Value offset_val =
          b.create<arith::ConstantIntOp>(loc, offset, b.getIntegerType(32));

      for (auto root_pair : llvm::enumerate(row_reduction_roots)) {
        Operation* root_op = root_pair.value();
        int idx = root_pair.index();
        for (int i = 0; i < vector_size; i++) {
          int val_idx = i * row_reduction_roots.size() + idx;
          auto result = emitWidthAdaptShuffle(
              b, loc, shuffle_val[val_idx], shuffle_type[idx], offset_val,
              shuffle_width, gpu::ShuffleMode::XOR);
          shuffle_val[val_idx] =
              (accum_factory[idx])(shuffle_val[val_idx], result);
        }
      }
    }

    // if (laneIdx == 0) {
    //   outputs[rowIdx] = sum
    // }
    Value lane_id_is_zero =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane_id, zero);
    scf::IfOp if_lane_id_is_zero =
        b.create<scf::IfOp>(loc, /* resultTypes */ ArrayRef<Type>{},
                            lane_id_is_zero, /*hasElseRegion*/ false);
    if_lane_id_is_zero.getThenRegion().front().clear();
    b.setInsertionPointToStart(&if_lane_id_is_zero.getThenRegion().front());

    for (auto root_pair : llvm::enumerate(row_reduction_roots)) {
      Operation* root_op = root_pair.value();
      int idx = root_pair.index();
      auto output_memref = root_op->getOperand(root_op->getNumOperands() - 1);
      if (vector_size > 1) {
        createAlignMemrefWithTile(b, output_memref, vector_size);
      }
      for (int i = 0; i < vector_size; i++) {
        b.create<memref::StoreOp>(
            loc, shuffle_val[i * row_reduction_roots.size() + idx],
            output_memref, row_ids[i]);
      }
    }

    b.setInsertionPointToEnd(&if_lane_id_is_zero.getThenRegion().front());
    b.create<scf::YieldOp>(loc, ValueRange({}));  // if (laneIdx == 0)
  }

  b.setInsertionPointToEnd(&if_is_valid_row_id.getThenRegion().front());
  b.create<scf::YieldOp>(loc, ValueRange({}));  // if (rowIdx < rows)
  b.setInsertionPointToEnd(parallel_op.getBody());
  b.create<scf::YieldOp>(loc, ValueRange({}));  // parallel

  cleanUnusedLhloOps(parent);

  return success();
}

LogicalResult extendShuffleElemType(OpBuilder& b, Location loc, Value value,
                                    Type shuffleElemType,
                                    Value* extended_value) {
  if (auto int_type = shuffleElemType.dyn_cast<IntegerType>()) {
    auto origin_type = value.getType().dyn_cast<IntegerType>();
    assert(origin_type && "mismatch expected type and actual type");
    assert(origin_type.getWidth() < int_type.getWidth());

    Value result;
    if (origin_type.isSignedInteger()) {
      result = b.create<arith::ExtSIOp>(loc, shuffleElemType, value);
    } else {
      result = b.create<arith::ExtUIOp>(loc, shuffleElemType, value);
    }
    *extended_value = result;
  } else {
    llvm::dbgs() << "not supported suffle type"
                 << "\n";
    return failure();
  }
  return success();
}

LogicalResult truncateShuffleElemType(OpBuilder& b, Location loc, Value value,
                                      Type elemType, Value* truncated_value) {
  if (auto int_type = elemType.dyn_cast<IntegerType>()) {
    auto origin_type = value.getType().dyn_cast<IntegerType>();
    assert(origin_type && "mismatch expected type and actual type");
    assert(origin_type.getWidth() > int_type.getWidth());

    // TODO: There is no signed truncateop, figure out if it's always safe
    // to truncate directly.
    *truncated_value = b.create<arith::TruncIOp>(loc, elemType, value);
  } else {
    llvm::dbgs() << "not supported suffle type"
                 << "\n";
    return failure();
  }
  return success();
}

LogicalResult emitFirstRoundShuffle(
    OpBuilder& b, Location loc,
    const SmallVector<Operation*, 4>& row_reduction_ops,
    const SmallVector<std::map<Operation*, Value>>& shared_mem_map_vec,
    mlir::ResultRange::iterator acc_iter, Value lane_id_is_zero, Value warp_id,
    int vector_size) {
  Value warp_size =
      b.create<arith::ConstantIntOp>(loc, kWarpSize, b.getIntegerType(32));
  auto xorAttr = b.getStringAttr("xor");
  for (auto root_op_en : llvm::enumerate(row_reduction_ops)) {
    auto idx = root_op_en.index();
    auto root_op = root_op_en.value();
    SmallVector<Value, 2> sum_vec(vector_size);
    for (int i = 0; i < vector_size; i++) {
      auto iter = acc_iter + i * row_reduction_ops.size();
      sum_vec[i] = *(iter + idx);
    }
    const auto elemType = getLhloOpsElementType(root_op);
    Type shuffleElemType;
    if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
      return failure();
    }
    AccumulatorFactory accumFactory = getFactory(
        b, root_op->getLoc(), cast<lmhlo::ReduceOp>(root_op).getBody());
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      Value offset_val =
          b.create<arith::ConstantIntOp>(loc, offset, b.getIntegerType(32));
      for (int i = 0; i < vector_size; i++) {
        if (elemType != shuffleElemType) {
          if (failed(extendShuffleElemType(b, loc, sum_vec[i], shuffleElemType,
                                           &sum_vec[i]))) {
            return failure();
          }
        }
      }
      SmallVector<Value, 2> shuffle_result_vec(vector_size);
      for (int i = 0; i < vector_size; i++) {
        shuffle_result_vec[i] =
            emitWidthAdaptShuffle(b, loc, sum_vec[i], shuffleElemType,
                                  offset_val, warp_size, gpu::ShuffleMode::XOR);
      }
      for (int i = 0; i < vector_size; i++) {
        if (elemType != shuffleElemType) {
          if (failed(truncateShuffleElemType(b, loc, sum_vec[1], elemType,
                                             &sum_vec[i])) ||
              failed(truncateShuffleElemType(b, loc, shuffle_result_vec[i],
                                             elemType,
                                             &shuffle_result_vec[i]))) {
            return failure();
          }
        }
      }
      for (int i = 0; i < vector_size; i++) {
        sum_vec[i] = accumFactory(sum_vec[i], shuffle_result_vec[i]);
      }
    }
    auto if_lane_id_is_zero =
        b.create<scf::IfOp>(loc, /*resultTypes*/ ArrayRef<Type>{},
                            lane_id_is_zero, /*hasElseRegion*/ false);
    if_lane_id_is_zero.getThenRegion().front().clear();
    b.setInsertionPointToStart(&if_lane_id_is_zero.getThenRegion().front());
    auto shared_idx = warp_id;
    SmallVector<Value, 4> shared_indices;
    shared_indices.push_back(shared_idx);
    SmallVector<Value, 2> shared_mem_vec;
    for (int i = 0; i < vector_size; i++) {
      assert((shared_mem_map_vec[i].find(root_op) !=
              shared_mem_map_vec[i].end()) &&
             "shared_mem_map unexpected");
      shared_mem_vec[i] = shared_mem_map_vec[i].at(root_op);
      if (elemType != shuffleElemType) {
        if (failed(extendShuffleElemType(b, loc, sum_vec[i], shuffleElemType,
                                         &sum_vec[i]))) {
          return failure();
        }
      }
    }
    for (int i = 0; i < vector_size; i++) {
      b.create<memref::StoreOp>(loc, sum_vec[i], shared_mem_vec[i],
                                shared_indices);
    }
    b.create<scf::YieldOp>(loc, ValueRange({}));
    b.setInsertionPointAfter(if_lane_id_is_zero);
  }
  return success();
}

LogicalResult emitSecondRoundShuffle(
    OpBuilder& b, Location loc,
    const SmallVector<Operation*, 4>& row_reduction_ops,
    mlir::ResultRange::iterator acc_iter, Value thread_id_is_zero,
    SmallVector<Value> output_indices_vec, int vector_size, int block_size) {
  assert(block_size % kWarpSize == 0);
  int num_warps = block_size / kWarpSize;
  Value shuffle_size =
      b.create<arith::ConstantIntOp>(loc, num_warps, b.getIntegerType(32));
  auto xorAttr = gpu::ShuffleMode::XOR;
  for (auto root_op_en : llvm::enumerate(row_reduction_ops)) {
    auto idx = root_op_en.index();
    auto root_op = root_op_en.value();
    SmallVector<Value> sum_vec(vector_size);
    for (int i = 0; i < vector_size; i++) {
      auto iter = acc_iter + i * row_reduction_ops.size();
      sum_vec[i] = *(iter + idx);
    }
    const auto elemType = getLhloOpsElementType(root_op);
    Type shuffleElemType;
    if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
      return failure();
    }
    AccumulatorFactory accumFactory = getFactory(
        b, root_op->getLoc(), cast<lmhlo::ReduceOp>(root_op).getBody());
    for (int offset = 1; offset < num_warps; offset <<= 1) {
      Value offset_val =
          b.create<arith::ConstantIntOp>(loc, offset, b.getIntegerType(32));
      for (int i = 0; i < vector_size; i++) {
        if (elemType != shuffleElemType) {
          if (failed(extendShuffleElemType(b, loc, sum_vec[i], shuffleElemType,
                                           &sum_vec[i]))) {
            return failure();
          }
        }
      }
      SmallVector<Value, 4> shuffle_result_vec(vector_size);
      for (int i = 0; i < vector_size; i++) {
        shuffle_result_vec[i] =
            emitWidthAdaptShuffle(b, loc, sum_vec[i], shuffleElemType,
                                  offset_val, shuffle_size, xorAttr);
      }
      for (int i = 0; i < vector_size; i++) {
        if (elemType != shuffleElemType) {
          if (failed(truncateShuffleElemType(b, loc, sum_vec[i], elemType,
                                             &sum_vec[i])) ||
              failed(truncateShuffleElemType(b, loc, shuffle_result_vec[i],
                                             elemType,
                                             &shuffle_result_vec[i]))) {
            return failure();
          }
        }
      }
      for (int i = 0; i < vector_size; i++) {
        sum_vec[i] = accumFactory(sum_vec[i], shuffle_result_vec[i]);
      }
    }
    auto if_thread_id_is_zero =
        b.create<scf::IfOp>(loc, /*resultTypes*/ ArrayRef<Type>{},
                            thread_id_is_zero, /*hasElseRegion*/ false);
    if_thread_id_is_zero.getThenRegion().front().clear();
    b.setInsertionPointToStart(&if_thread_id_is_zero.getThenRegion().front());
    auto res_memref =
        *(root_op->getOperands().begin() + (root_op->getNumOperands() - 1));
    if (vector_size > 1) {
      createAlignMemrefWithTile(b, res_memref, vector_size);
    }
    for (int i = 0; i < vector_size; i++) {
      b.create<memref::StoreOp>(loc, sum_vec[i], res_memref,
                                output_indices_vec[i]);
    }
    b.create<scf::YieldOp>(loc, ValueRange({}));
    b.setInsertionPointAfter(if_thread_id_is_zero);
  }
  return success();
}

/* Row reduction with 2 round warp shuffle
 *
 * the outter 2 nested loops is to be expanded during LoopSplit
 * for (m=0; m < rows; ++m) {
 *   for (int n = 0; n < c_threads_row_reduction; ++n) {
 *     for each row reduction op {
 *       sum = init_value;
 *       for (int j = n; j < shape_w; j += c_threads_row_reduction) {
 *         float t = x[m][j];
 *         sum += t;
 *       }
 *       for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
 *         sum += __shfl_xor(sum, offset);
 *       }
 *       if (lane_id == 0) {
 *         shared_index = warp_id;
 *         shared_data[shared_index] = sum;
 *       }
 *     }
 *     __syncthreads
 *     for each row reduction op {
 *       shared_index = lane_id;
 *       if (lane_id < threads() / warpSize) {
 *         sum = shared_data[shared_index];
 *       } else {
 *         sum = init_value;
 *       }
 *       for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
 *         sum += __shfl_xor(sum, offset);
 *       }
 *       if (thread_id == 0) {
 *         output[m] = sum
 *       }
 *     }
 *   }
 * }
 */
template <>
LogicalResult lowerWithScheduleRowReduction<DISC_BLOCK_WISE_ROW_REDUCE>(
    ArrayRef<Operation*> root_ops, Operation* dominant_op, Block* parent,
    const ShapeAnalysis* shape_analysis, int vector_size) {
  if (!isRank2RowReduction(dominant_op)) {
    return failure();
  }

  Location loc = dominant_op->getLoc();
  OpBuilder b(root_ops.back());

  Value lhs = *dominant_op->getOperands().begin();
  const MemRefType& lhs_type = lhs.getType().template cast<MemRefType>();
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value shape_h = b.create<memref::DimOp>(loc, lhs, zero);
  Value shape_w = b.create<memref::DimOp>(loc, lhs, one);
  Value num_threads =
      b.create<arith::ConstantIndexOp>(loc, getThreadPerBlock(dominant_op));
  std::map<Operation*, Value> init_values_cache;
  SmallVector<Operation*, 4> row_reduction_ops;
  SmallVector<std::map<Operation*, Value>> shared_mem_map_vec(vector_size);
  for (auto root_op : root_ops) {
    if (isRank2RowReduction(root_op)) {
      row_reduction_ops.push_back(root_op);
    }
  }
  Value warp_size_val = b.create<arith::ConstantIndexOp>(loc, kWarpSize);

  // loop over (rows, threads)
  // Note that we know `shape_h` can be divided by `vector_size` exactly.
  Value vec_size = b.create<arith::ConstantIndexOp>(loc, vector_size);
  Value num_blocks = b.create<arith::DivUIOp>(loc, shape_h, vec_size);
  SmallVector<Value, 2> vars;
  scf::ParallelOp parallel_op = createParallelAndSetInsPt(
      b, loc, vars, {zero, zero}, {num_blocks, num_threads}, {one, one}, {});
  parallel_op.getBody()->clear();
  b.setInsertionPointToStart(parallel_op.getBody());
  Value var_m = vars[0];
  Value var_n = vars[1];

  Value lane_id = b.create<arith::RemUIOp>(loc, var_n, warp_size_val);
  Value warp_id = b.create<arith::DivUIOp>(loc, var_n, warp_size_val);
  Value lane_id_is_zero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane_id, zero);
  Value thread_id_is_zero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, var_n, zero);

  SmallVector<Value, 4> init_values(row_reduction_ops.size() * vector_size);
  for (auto root_pair : llvm::enumerate(row_reduction_ops)) {
    Operation* root_op = root_pair.value();
    int idx = root_pair.index();
    auto init_value = b.create<memref::LoadOp>(
        loc, *(cast<lmhlo::ReduceOp>(root_op).getInitValues().begin()));
    init_values_cache[root_op] = init_value;
    for (int i = 0; i < vector_size; i++) {
      init_values[i * row_reduction_ops.size() + idx] = init_value;
    }
    const auto elemType = getLhloOpsElementType(root_op);
    Type shuffleElemType;
    if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
      return failure();
    }
    for (int i = 0; i < vector_size; i++) {
      auto shared_mem = createSharedMemory(b, loc, kWarpSize, shuffleElemType);
      shared_mem_map_vec[i][root_op] = shared_mem;
    }
  }

  Value row_base = b.create<arith::MulIOp>(loc, var_m, vec_size);
  SmallVector<Value> row_ids(vector_size);
  for (int i = 0; i < vector_size; i++) {
    row_ids[i] = b.create<arith::AddIOp>(
        loc, row_base, b.create<arith::ConstantIndexOp>(loc, i));
  }

  Value var_j = nullptr;
  scf::ForOp for_op_j =
      createLoopAndSetInsPt(b, loc, var_j,
                            /*lb*/ var_n, /*ub*/ shape_w,
                            /*step*/ num_threads, init_values);

  SmallVector<Value, 4> yield_values_for_j;
  for (int i = 0; i < vector_size; i++) {
    SmallVector<Value, 4> multidim({row_ids[i], var_j});
    ValueRange input_index(multidim);
    if (failed(emitInThreadReduction(
            b, loc, root_ops,
            for_op_j.getRegionIterArgs().begin() + i * row_reduction_ops.size(),
            input_index, var_j, yield_values_for_j, dominant_op,
            shape_analysis))) {
      return failure();
    }
  }
  b.create<scf::YieldOp>(loc, yield_values_for_j);

  b.setInsertionPointToEnd(parallel_op.getBody());
  if (failed(emitFirstRoundShuffle(
          b, loc, row_reduction_ops, shared_mem_map_vec,
          for_op_j.getResults().begin(),  // + i * row_reduction_ops.size(),
          lane_id_is_zero, warp_id, vector_size))) {
    return failure();
  }
  b.setInsertionPointToEnd(parallel_op.getBody());
  b.create<gpu::BarrierOp>(loc);

  SmallVector<Type, 4> root_elem_types(row_reduction_ops.size() * vector_size);
  for (auto root_op_en : llvm::enumerate(row_reduction_ops)) {
    auto idx = root_op_en.index();
    auto root_op = root_op_en.value();
    auto elem_type = getLhloOpsElementType(root_op);
    for (int i = 0; i < vector_size; i++) {
      root_elem_types[i * row_reduction_ops.size() + idx] = elem_type;
    }
  }

  Value lane_id_inbound = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, lane_id,
      b.create<arith::ConstantIndexOp>(
          loc, getThreadPerBlock(dominant_op) / kWarpSize));
  scf::IfOp if_lane_id_inbound =
      b.create<scf::IfOp>(loc, /*resultTypes*/ root_elem_types, lane_id_inbound,
                          /*hasElseRegion*/ true);
  if_lane_id_inbound.getThenRegion().front().clear();
  if_lane_id_inbound.getElseRegion().front().clear();
  b.setInsertionPointToStart(&if_lane_id_inbound.getThenRegion().front());
  SmallVector<Value, 4> true_yield_values(row_reduction_ops.size() *
                                          vector_size);
  for (auto root_op_en : llvm::enumerate(row_reduction_ops)) {
    auto idx = root_op_en.index();
    auto root_op = root_op_en.value();
    auto shared_idx = lane_id;
    SmallVector<Value, 4> shared_indices;
    shared_indices.push_back(shared_idx);
    for (int i = 0; i < vector_size; i++) {
      Type elemType = root_elem_types[i * row_reduction_ops.size() + idx];
      Type shuffleElemType;
      if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
        return failure();
      }
      assert((shared_mem_map_vec[i].find(root_op) !=
              shared_mem_map_vec[i].end()) &&
             "shared_mem_map unexpected");
      auto shared_mem = shared_mem_map_vec[i][root_op];
      Value shared_mem_value =
          b.create<memref::LoadOp>(loc, shared_mem, shared_indices);
      if (elemType != shuffleElemType) {
        if (failed(truncateShuffleElemType(b, loc, shared_mem_value, elemType,
                                           &shared_mem_value))) {
          return failure();
        }
      }
      true_yield_values[i * row_reduction_ops.size() + idx] = shared_mem_value;
    }
  }
  b.create<scf::YieldOp>(loc, true_yield_values);
  b.setInsertionPointToStart(&if_lane_id_inbound.getElseRegion().front());
  SmallVector<Value, 4> false_yield_values(row_reduction_ops.size() *
                                           vector_size);
  for (auto root_op_en : llvm::enumerate(row_reduction_ops)) {
    auto idx = root_op_en.index();
    auto root_op = root_op_en.value();
    assert((init_values_cache.find(root_op) != init_values_cache.end()) &&
           "unexpected init_values_cache");
    for (int i = 0; i < vector_size; i++) {
      false_yield_values[i * row_reduction_ops.size() + idx] =
          init_values_cache[root_op];
    }
  }
  b.create<scf::YieldOp>(loc, false_yield_values);
  b.setInsertionPointToEnd(parallel_op.getBody());
  auto acc_iter = if_lane_id_inbound.getResults().begin();
  if (failed(emitSecondRoundShuffle(b, loc, row_reduction_ops, acc_iter,
                                    thread_id_is_zero, row_ids, vector_size,
                                    getThreadPerBlock(dominant_op)))) {
    return failure();
  }

  b.setInsertionPointToEnd(parallel_op.getBody());
  b.create<scf::YieldOp>(loc, ValueRange({}));

  // remove the root_op if it has no other users except the memref
  cleanUnusedLhloOps(parent);

  return success();
}

/* BLOCK TILE LOOP IMPL: tile_w * tile_h threads load tile_w * (tile_h * LOOP)
//  elements from gmem to SHM(tile_w vectorizes load), do reduction in SHM and
//  atomic to gmem
//
// var_tile_h = 16;
// var_tile_w = 32;
// var_threads = 512;
// block_limit = 104; // SM_COUNT
// var_threads = var_tile_h * var_tile_w;

// num_block_cols = ceil(var_cols / var_tile_w);
// num_sw_block_rows = ceil(var_rows / var_tile_h);
// num_hw_block_rows = ceil(block_limit / num_block_cols);
// var_loop_iter = ceil(num_sw_block_rows / num_hw_block_rows);

// for (m = 0; m < num_block_cols * num_hw_block_rows; ++m) {
//   for (n = 0; n < var_threads; ++n) {
//     local_row_index = n / var_tile_w;
//     local_col_index = n % var_tile_w;
//     hw_block_row_index = m / num_block_cols;
//     block_col_index = m % num_block_cols;
//     col_index = block_col_index * var_tile_w + local_col_index;
//     sw_block_row_index_base = hw_block_row_index * var_loop_iter;
//     is_col_valid = col_index < var_cols;
//     if (is_col_valid) {
//       loop_acc = init_value;
//       for (l = 0; l < var_loop_iter; ++l) {
//         sw_block_row_index = sw_block_row_index_base + l;
//         row_index = sw_block_row_index * var_tile_h + local_row_index;
//         is_row_valid = row_index < var_rows;
//         if (is_row_valid) {
//           loop_acc = loop_acc + global[row_index, col_index]
//         } else {
//           loop_acc = loop_acc;
//         }
//       }
//       shm[n] = loop_acc;
//     } else {
//     }
//     row_index_shuffle = sw_block_row_index_base * var_tile_h +
local_row_index
//     for (int stride = var_tile_h / 2; stride > 1; stride /= 2) {
//       __syncthreads();
//       if (local_row_index < stride && (row_index_shuffle + stride) <
var_rows) {
//         shm[n] += shm[stride * var_tile_w + n];
//       }
//     }
//     __syncthreads();
//     if (local_row_index == 0) {
//       if (is_valid) {
//         partial_result = shm[n] + shm[var_tile_w + n];
//         atomicAdd(&global[col_index], partial_result);
//       }
//     }
//   }
// }
*/

// TODO: rocm function is temporarily disabled, buggy and need fix
template <int TILE_W, int TILE_H>
LogicalResult lowerWithScheduleColReductionForRocm(
    ArrayRef<Operation*> root_ops, Operation* dominant_op, Block* parent,
    const int LOOP, const int core_count,
    const ShapeAnalysis* shape_analysis = nullptr) {
  if (!isRank2ColReduction(dominant_op)) {
    return failure();
  }

  // Create helper Values
  SmallVector<Operation*, 4> col_reduction_roots;
  std::copy_if(
      root_ops.begin(), root_ops.end(), std::back_inserter(col_reduction_roots),
      [](Operation* operation) { return isRank2ColReduction(operation); });

  Location loc = dominant_op->getLoc();
  OpBuilder b(root_ops.back());
  const int kTileW = TILE_W;  // 16
  const int kTileH = TILE_H;  // 32
  const int kLoopIter = LOOP;
  const int kThreads = kTileW * kTileH;

  Value lhs = dominant_op->getOperand(0);
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value loop_base_var = b.create<arith::ConstantIndexOp>(loc, kLoopIter);
  Value var512 = b.create<arith::ConstantIndexOp>(loc, 512);
  Value var32 = b.create<arith::ConstantIndexOp>(loc, 32);
  Value block_limit = b.create<arith::ConstantIndexOp>(loc, core_count);
  Value var_rows = b.create<memref::DimOp>(loc, lhs, zero);
  Value var_cols = b.create<memref::DimOp>(loc, lhs, one);
  Value var_tile_h = b.create<arith::ConstantIndexOp>(loc, kTileH);
  Value var_tile_w = b.create<arith::ConstantIndexOp>(loc, kTileW);
  Value var_threads = b.create<arith::ConstantIndexOp>(loc, kThreads);

  // var_tile_h = 16;
  // var_tile_w = 32;
  // block_limit = 104; // SM_COUNT
  // var_threads = var_tile_h * var_tile_w;

  // num_block_cols = ceil(var_cols / var_tile_w);
  // num_sw_block_rows = ceil(var_rows / var_tile_h);
  // num_hw_block_rows = ceil(block_limit / num_block_cols);
  // var_loop_iter = ceil(num_sw_block_rows / num_hw_block_rows);

  Value num_block_cols =
      b.create<arith::CeilDivUIOp>(loc, var_cols, var_tile_w);
  Value num_sw_block_rows =
      b.create<arith::CeilDivUIOp>(loc, var_rows, var_tile_h);
  Value num_hw_block_rows =
      b.create<arith::CeilDivUIOp>(loc, block_limit, num_block_cols);
  Value var_loop_iter =
      b.create<arith::CeilDivUIOp>(loc, num_sw_block_rows, num_hw_block_rows);

  // for (m = 0; m < num_block_cols * num_hw_block_rows; ++m) {
  //   for (n = 0; n < var_threads; ++n) {

  Value var_blocks =
      b.create<arith::MulIOp>(loc, num_block_cols, num_hw_block_rows);
  SmallVector<Value, 2> vars;
  scf::ParallelOp parallel_op = createParallelAndSetInsPt(
      b, loc, vars, {zero, zero}, {var_blocks, var_threads}, {one, one}, {});
  parallel_op.getBody()->clear();
  b.setInsertionPointToStart(parallel_op.getBody());
  Value var_m = vars[0];
  Value var_n = vars[1];

  SmallVector<AccumulatorFactory, 4> accum_factory(col_reduction_roots.size());
  // sum = init_value;
  SmallVector<Value, 4> init_values(col_reduction_roots.size());
  SmallVector<Type, 4> init_values_types(col_reduction_roots.size());
  for (auto root_pair : llvm::enumerate(col_reduction_roots)) {
    Operation* root_op = root_pair.value();
    int idx = root_pair.index();
    Value init_value = b.create<memref::LoadOp>(
        loc, cast<lmhlo::ReduceOp>(root_op).getInitValues()[0]);
    init_values[idx] = init_value;
    init_values_types[idx] = init_value.getType();
    accum_factory[idx] = std::move(getFactory(
        b, root_op->getLoc(), cast<lmhlo::ReduceOp>(root_op).getBody()));
  }
  // local_col_index = n % var_tile_w;
  // block_col_index = m % num_block_cols;
  // col_index = block_col_index * var_tile_w + local_col_index;

  // local_row_index = n / var_tile_w;
  // hw_block_row_index = m / num_block_cols;
  // sw_block_row_index_base = hw_block_row_index * var_loop_iter;

  Value local_row_index = b.create<arith::DivUIOp>(loc, var_n, var_tile_w);
  Value local_col_index = b.create<arith::RemUIOp>(loc, var_n, var_tile_w);
  Value block_col_index = b.create<arith::RemUIOp>(loc, var_m, num_block_cols);
  Value col_index = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, block_col_index, var_tile_w),
      local_col_index);
  Value hw_block_row_index =
      b.create<arith::DivUIOp>(loc, var_m, num_block_cols);
  Value sw_block_row_index_base =
      b.create<arith::MulIOp>(loc, hw_block_row_index, var_loop_iter);

  // define SHM
  std::map<Operation*, Value> shared_mem_map;
  for (auto root_pair : llvm::enumerate(col_reduction_roots)) {
    Operation* root_op = root_pair.value();
    int idx = root_pair.index();
    const auto elemType = getLhloOpsElementType(root_op);
    auto shared_mem = createSharedMemory(b, loc, kThreads, elemType);
    shared_mem_map[root_op] = shared_mem;
  }

  // is_col_valid = col_index < var_cols;
  Value is_col_valid = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                               col_index, var_cols);
  scf::IfOp if_col_valid =
      b.create<scf::IfOp>(loc, ArrayRef<Type>{}, is_col_valid, true);
  if_col_valid.getThenRegion().front().clear();
  if_col_valid.getElseRegion().front().clear();
  b.setInsertionPointToStart(&if_col_valid.getThenRegion().front());

  // if (is_col_valid) {
  //   loop_acc = init_value;
  //   for (l = 0; l < var_loop_iter; ++l) {
  //     sw_block_row_index = sw_block_row_index_base + l;
  //     row_index = sw_block_row_index * var_tile_h + local_row_index;
  //     is_row_valid = row_index < var_rows;
  //     if (is_row_valid) {
  //       loop_acc = loop_acc + global[row_index, col_index]
  //     } else {
  //       loop_acc = loop_acc;
  //     }
  //   }
  //   shm[n] = loop_acc;
  // } else {
  //   shm[n] = init_value;
  // }

  Value var_l = nullptr;
  scf::ForOp for_op_l = createLoopAndSetInsPt(b, loc, var_l,
                                              /*lb*/ zero, /*ub*/ var_loop_iter,
                                              /*step*/ one, init_values);

  Value sw_block_row_index =
      b.create<arith::AddIOp>(loc, sw_block_row_index_base, var_l);
  Value row_index = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, sw_block_row_index, var_tile_h),
      local_row_index);

  scf::IfOp if_row_valid = b.create<scf::IfOp>(
      loc, init_values_types,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, row_index,
                              var_rows),
      true);
  if_row_valid.getThenRegion().front().clear();
  if_row_valid.getElseRegion().front().clear();

  ValueRange load_index({row_index, col_index});
  int64_t row_reduction_idx = 0;

  SmallVector<Value, 4> yield_values_for_if;
  SmallVector<Value, 4> yield_values_for_else;

  int col_red_root_op_idx = 0;
  for (auto root_op : root_ops) {
    if (isRank2ColReduction(root_op)) {
      b.setInsertionPointToEnd(&if_row_valid.getThenRegion().front());
      auto data =
          createLoadOrUseCachedValue(loc, &b, root_op, root_op->getOperand(0),
                                     load_index, b.saveInsertionPoint());
      auto acc = (accum_factory[col_red_root_op_idx])(
          *(for_op_l.getRegionIterArgs().begin() + col_red_root_op_idx), data);
      col_red_root_op_idx++;
      yield_values_for_if.push_back(acc);
    } else if (isRank2RowReduction(root_op)) {
      assert(false && "unexpected row_reduction");
      return failure();
    } else if (isa<lmhlo::ReduceOp>(root_op)) {
      // assert(false && "unexpected reduce_reduction");
      // return failure();
      auto dominant_shape = getShapeValues(&b, dominant_op->getOperand(0));
      Value linear_index = calcLinearIndex(&b, loc, load_index, dominant_shape);
      auto root_shape = getShapeValues(&b, root_op->getOperand(0));
      auto mapped_index = calcMultiDimIndex(&b, loc, linear_index, root_shape);
      emitNotToVectorReduction(b, loc, root_op, mapped_index);
    } else {
      auto dominant_shape = getShapeValues(&b, dominant_op->getOperand(0));
      Value linear_index = calcLinearIndex(&b, loc, load_index, dominant_shape);
      if (!succeeded(
              lowerHelper(b, loc, root_op, linear_index, shape_analysis))) {
        return failure();
      }
    }
  }
  b.setInsertionPointToEnd(&if_row_valid.getThenRegion().front());
  b.create<scf::YieldOp>(loc, yield_values_for_if);
  b.setInsertionPointToEnd(&if_row_valid.getElseRegion().front());
  b.create<scf::YieldOp>(loc, for_op_l.getRegionIterArgs());
  b.setInsertionPointToEnd(for_op_l.getBody());
  b.create<scf::YieldOp>(loc, if_row_valid.getResults());
  b.setInsertionPointAfter(for_op_l);

  for (auto root_pair : llvm::enumerate(col_reduction_roots)) {
    Operation* root_op = root_pair.value();
    int idx = root_pair.index();
    b.create<memref::StoreOp>(loc, *(for_op_l.getResults().begin() + idx),
                              shared_mem_map[root_op], var_n);
  }

  b.create<scf::YieldOp>(loc, ValueRange({}));
  b.setInsertionPointToStart(&if_col_valid.getElseRegion().front());
  for (auto root_pair : llvm::enumerate(col_reduction_roots)) {
    Operation* root_op = root_pair.value();
    int idx = root_pair.index();
    b.create<memref::StoreOp>(loc, init_values[idx], shared_mem_map[root_op],
                              var_n);
  }

  b.create<scf::YieldOp>(loc, ValueRange({}));
  b.setInsertionPointAfter(if_col_valid);

  // row_index_shuffle = sw_block_row_index_base * var_tile_h + local_row_index
  // for (int stride = var_tile_h / 2; stride > 1; stride /= 2) {
  //   __syncthreads();
  //   if (local_row_index < stride && (row_index_shuffle + stride) < var_rows)
  //   {
  //     shm[n] += shm[stride * var_tile_w + n];
  //   }
  // }
  Value row_index_shuffle = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, sw_block_row_index_base, var_tile_h),
      local_row_index);
  for (int stride = kTileH / 2; stride > 1; stride /= 2) {
    b.create<gpu::BarrierOp>(loc);
    // if (local_row_index < stride &&
    //    row_index + stride < var_rows) {
    //  shm[n] += shm[stride * var_tile_w + n];
    // }
    Value var_stride = b.create<arith::ConstantIndexOp>(loc, stride);
    Value is_lt_stride = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                 local_row_index, var_stride);
    Value is_row_idx_plus_stride_lt_rows = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult,
        b.create<arith::AddIOp>(loc, row_index_shuffle, var_stride), var_rows);
    Value is_lt_stride_rows = b.create<arith::AndIOp>(
        loc, is_lt_stride, is_row_idx_plus_stride_lt_rows);
    scf::IfOp if_is_lt_stride_rows =
        b.create<scf::IfOp>(loc, /* resultTypes */ ArrayRef<Type>{},
                            is_lt_stride_rows, /*hasElseRegion*/ false);
    if_is_lt_stride_rows.getThenRegion().front().clear();
    b.setInsertionPointToStart(&if_is_lt_stride_rows.getThenRegion().front());
    Value var_shm_offset = b.create<mlir::arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, var_stride, var_tile_w), var_n);
    for (auto root_pair : llvm::enumerate(col_reduction_roots)) {
      Operation* root_op = root_pair.value();
      int idx = root_pair.index();
      Value shm_op0 =
          b.create<memref::LoadOp>(loc, shared_mem_map[root_op], var_n);
      Value shm_op1 = b.create<memref::LoadOp>(loc, shared_mem_map[root_op],
                                               var_shm_offset);
      b.create<memref::StoreOp>(loc, (accum_factory[idx])(shm_op0, shm_op1),
                                shared_mem_map[root_op], var_n);
    }
    b.create<scf::YieldOp>(loc, ValueRange({}));
    b.setInsertionPointAfter(if_is_lt_stride_rows);
  }
  // __syncthreads();
  // if (local_row_index == 0) {
  //   if (is_valid) {
  //     partial_result = shm[n] + shm[var_tile_w + n];
  //     atomicAdd(&global[col_index], partial_result);
  //   }
  // }
  b.create<gpu::BarrierOp>(loc);

  Value is_local_row_index_eq_zero = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, local_row_index, zero);

  Value is_valid = b.create<arith::AndIOp>(
      loc, is_col_valid,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, row_index_shuffle,
                              var_rows));

  Value is_atom_add =
      b.create<arith::AndIOp>(loc, is_local_row_index_eq_zero, is_valid);
  scf::IfOp if_is_atom_add =
      b.create<scf::IfOp>(loc, /*resultTypes*/ TypeRange{}, is_atom_add,
                          /*hasElseRegion*/ false);
  if_is_atom_add.getThenRegion().front().clear();
  b.setInsertionPointToStart(&if_is_atom_add.getThenRegion().front());
  Value shm_offset = b.create<mlir::arith::AddIOp>(loc, var_tile_w, var_n);
  for (auto root_pair : llvm::enumerate(col_reduction_roots)) {
    Operation* root_op = root_pair.value();
    int idx = root_pair.index();
    Value shm_op0 =
        b.create<memref::LoadOp>(loc, shared_mem_map[root_op], var_n);
    Value shm_op1 =
        b.create<memref::LoadOp>(loc, shared_mem_map[root_op], shm_offset);
    Value partial_result = (accum_factory[idx])(shm_op0, shm_op1);
    auto root_element_type = getLhloOpsElementType(root_op);
    b.create<memref::AtomicRMWOp>(
        loc, root_element_type,
        getAtomicRMWKind(cast<lmhlo::ReduceOp>(root_op).getBody()),
        partial_result, root_op->getOperand(2), ValueRange({col_index}));
  }
  b.create<scf::YieldOp>(loc, ValueRange({}));

  b.setInsertionPointToEnd(parallel_op.getBody());
  b.create<scf::YieldOp>(loc, ValueRange({}));  // parallel

  cleanUnusedLhloOps(parent);
  return success();
}

/* BLOCK TILE IMPL: tile_w*tile_h threads load tile_w*tile_h elements
 * from gmem to SHM(tile_w vectorizes load), do reduction in SHM and
 * atomic to gmem
 *
 * var_tile_w = 8
 * var_tile_h = 32
 * num_block_col = ceil( var_cols() / var_tile_w );
 * num_block_row = ceil( var_rows() / var_tile_h);
 * for (m = 0; m < num_block_col * num_block_row; ++m) {
 *   for (n = 0; n < var_threads; ++n) {
 *     local_row_index = n / var_tile_w;
 *     local_col_index = n % var_tile_w;
 *     block_row_index = m / num_block_col;
 *     block_col_index = m % num_block_col;
 *     row_index = block_row_index * var_tile_h + local_row_index;
 *     col_index = block_col_index * var_tile_w + local_col_index;
 *     is_valid = (row_index < var_rows()) && (col_index < var_cols());
 *     if (is_valid) {
 *       shm[n] = sum + global[row_index, col_index];
 *     } else {
 *       shm[n] = sum;
 *     }
 *     for (int stride = var_tile_h / 2; stride > 1; stride /= 2) {
 *       __syncthreads();
 *       if (local_row_index < stride && row_index + stride < var_rows) {
 *         shm[n] += shm[stride * var_tile_w + n];
 *       }
 *     }
 *     __syncthreads();
 *     if (local_row_index == 0) {
 *       if (is_valid) {
 *         partial_result = shm[n] + shm[var_tile_w + n];
 *         atomicAdd(&global[col_index], partial_result);
 *       }
 *     }
 *   }
 * }
 */
// TODO(feiwen): add vectorization support
template <int TILE_W, int TILE_H>
LogicalResult lowerWithScheduleColReductionBlockTileSchedule(
    ArrayRef<Operation*> root_ops, Operation* dominant_op, Block* parent,
    const ShapeAnalysis* shape_analysis = nullptr) {
  if (!isRank2ColReduction(dominant_op)) {
    return failure();
  }

  // Create helper Values
  SmallVector<Operation*, 4> col_reduction_roots;
  std::copy_if(
      root_ops.begin(), root_ops.end(), std::back_inserter(col_reduction_roots),
      [](Operation* operation) { return isRank2ColReduction(operation); });

  Location loc = dominant_op->getLoc();
  OpBuilder b(root_ops.back());
  const int kTileW = TILE_W;  // 8
  const int kTileH = TILE_H;  // 32
  const int kThreads = kTileW * kTileH;

  Value lhs = dominant_op->getOperand(0);
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value var_rows = b.create<memref::DimOp>(loc, lhs, zero);
  Value var_cols = b.create<memref::DimOp>(loc, lhs, one);
  Value var_threads = b.create<arith::ConstantIndexOp>(loc, kThreads);
  // Start to emit.

  // var_tile_w = 8
  // var_tile_h = 32
  // num_block_col = ceil( var_cols() / var_tile_w );
  // num_block_row = ceil( var_rows() / var_tile_h);
  Value var_tile_w = b.create<arith::ConstantIndexOp>(loc, kTileW);
  Value var_tile_h = b.create<arith::ConstantIndexOp>(loc, kTileH);
  Value num_block_col = b.create<arith::CeilDivSIOp>(loc, var_cols, var_tile_w);
  Value num_block_row = b.create<arith::CeilDivSIOp>(loc, var_rows, var_tile_h);
  Value var_blocks = b.create<arith::MulIOp>(loc, num_block_col, num_block_row);

  // for (m = 0; m < num_block_col * num_block_row; ++m) {
  //  for (n = 0; n < var_threads; ++n) {
  SmallVector<Value, 2> vars;
  scf::ParallelOp parallel_op = createParallelAndSetInsPt(
      b, loc, vars, {zero, zero}, {var_blocks, var_threads}, {one, one}, {});
  parallel_op.getBody()->clear();
  b.setInsertionPointToStart(parallel_op.getBody());
  Value var_m = vars[0];
  Value var_n = vars[1];

  SmallVector<AccumulatorFactory, 4> accum_factory(col_reduction_roots.size());
  // sum = init_value;
  SmallVector<Value, 4> init_values(col_reduction_roots.size());
  for (auto root_pair : llvm::enumerate(col_reduction_roots)) {
    Operation* root_op = root_pair.value();
    int idx = root_pair.index();
    Value init_value = b.create<memref::LoadOp>(
        loc, cast<lmhlo::ReduceOp>(root_op).getInitValues()[0]);
    init_values[idx] = init_value;
    accum_factory[idx] = std::move(getFactory(
        b, root_op->getLoc(), cast<lmhlo::ReduceOp>(root_op).getBody()));
  }
  // local_row_index = n / var_tile_w;
  // local_col_index = n % var_tile_w;
  // block_row_index = m / num_block_col;
  // block_col_index = m % num_block_col;
  Value local_row_index = b.create<arith::DivUIOp>(loc, var_n, var_tile_w);
  Value local_col_index = b.create<arith::RemUIOp>(loc, var_n, var_tile_w);
  Value block_row_index = b.create<arith::DivUIOp>(loc, var_m, num_block_col);
  Value block_col_index = b.create<arith::RemUIOp>(loc, var_m, num_block_col);

  // row_index = block_row_index * var_tile_h + local_row_index;
  // col_index = block_col_index * var_tile_w + local_col_index;
  Value row_index = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, block_row_index, var_tile_h),
      local_row_index);
  Value col_index = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, block_col_index, var_tile_w),
      local_col_index);

  // define SHM
  std::map<Operation*, Value> shared_mem_map;
  for (auto root_pair : llvm::enumerate(col_reduction_roots)) {
    Operation* root_op = root_pair.value();
    int idx = root_pair.index();
    const auto elemType = getLhloOpsElementType(root_op);
    auto shared_mem = createSharedMemory(b, loc, kThreads, elemType);
    shared_mem_map[root_op] = shared_mem;
  }

  // is_valid = (row_index < var_rows()) && (col_index < var_cols());
  Value is_lt_rows = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                             row_index, var_rows);
  Value is_lt_cols = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                             col_index, var_cols);
  Value is_valid = b.create<arith::AndIOp>(loc, is_lt_rows, is_lt_cols);
  scf::IfOp if_is_valid =
      b.create<scf::IfOp>(loc, /* resultTypes */ ArrayRef<Type>{}, is_valid,
                          /*hasElseRegion*/ true);
  if_is_valid.getThenRegion().front().clear();
  if_is_valid.getElseRegion().front().clear();
  b.setInsertionPointToStart(&if_is_valid.getThenRegion().front());
  // if (is_valid) {
  //    shm[n] = sum + global[row_index, col_index];
  // }
  SmallVector<Value, 2> multidim_load_index({row_index, col_index});
  ValueRange load_index(multidim_load_index);
  int col_red_root_op_idx = 0;
  for (auto* root_op : root_ops) {
    if (isRank2ColReduction(root_op)) {
      auto data = createLoadOrUseCachedValue(
          loc, &b, root_op, *root_op->getOperands().begin(), load_index,
          b.saveInsertionPoint());
      b.create<memref::StoreOp>(loc,
                                (accum_factory[col_red_root_op_idx])(
                                    data, init_values[col_red_root_op_idx]),
                                shared_mem_map[root_op], var_n);
      col_red_root_op_idx++;
    } else if (isRank2RowReduction(root_op)) {
      assert(false && "unexpected row_reduction");
      return failure();
    } else if (isa<lmhlo::ReduceOp>(root_op)) {
      auto dominant_shape = getShapeValues(&b, dominant_op->getOperand(0));
      Value linear_index = calcLinearIndex(&b, loc, load_index, dominant_shape);
      auto root_shape = getShapeValues(&b, root_op->getOperand(0));
      auto mapped_index = calcMultiDimIndex(&b, loc, linear_index, root_shape);
      emitNotToVectorReduction(b, loc, root_op, mapped_index);
    } else {
      auto dominant_shape = getShapeValues(&b, dominant_op->getOperand(0));
      Value linear_index = calcLinearIndex(&b, loc, load_index, dominant_shape);
      if (!succeeded(
              lowerHelper(b, loc, root_op, linear_index, shape_analysis))) {
        assert(false && "elementwise lowerHelper failure");
        return failure();
      }
    }
  }

  b.create<scf::YieldOp>(loc, ValueRange({}));

  b.setInsertionPointToStart(&if_is_valid.getElseRegion().front());
  // else {
  //    shm[n] = 0;
  // }
  for (auto root_pair : llvm::enumerate(col_reduction_roots)) {
    Operation* root_op = root_pair.value();
    int idx = root_pair.index();
    b.create<memref::StoreOp>(loc, init_values[idx], shared_mem_map[root_op],
                              var_n);
  }
  b.create<scf::YieldOp>(loc, ValueRange({}));

  b.setInsertionPointAfter(if_is_valid);

  // for (int stride = kTileH / 2; stride > 1; stride /= 2) {
  for (int stride = kTileH / 2; stride > 1; stride /= 2) {
    //     __syncthreads();
    b.create<gpu::BarrierOp>(loc);
    // if (local_row_index < stride &&
    //    row_index + stride < var_rows) {
    //  shm[n] += shm[stride * var_tile_w + n];
    // }
    Value var_stride = b.create<arith::ConstantIndexOp>(loc, stride);
    Value is_lt_stride = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                 local_row_index, var_stride);
    Value is_row_idx_plus_stride_lt_rows = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult,
        b.create<arith::AddIOp>(loc, row_index, var_stride), var_rows);
    Value is_lt_stride_rows = b.create<arith::AndIOp>(
        loc, is_lt_stride, is_row_idx_plus_stride_lt_rows);
    scf::IfOp if_is_lt_stride_rows =
        b.create<scf::IfOp>(loc, /* resultTypes */ ArrayRef<Type>{},
                            is_lt_stride_rows, /*hasElseRegion*/ false);
    if_is_lt_stride_rows.getThenRegion().front().clear();
    b.setInsertionPointToStart(&if_is_lt_stride_rows.getThenRegion().front());
    Value var_shm_offset = b.create<mlir::arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, var_stride, var_tile_w), var_n);
    for (auto root_pair : llvm::enumerate(col_reduction_roots)) {
      Operation* root_op = root_pair.value();
      int idx = root_pair.index();
      Value shm_op0 =
          b.create<memref::LoadOp>(loc, shared_mem_map[root_op], var_n);
      Value shm_op1 = b.create<memref::LoadOp>(loc, shared_mem_map[root_op],
                                               var_shm_offset);
      b.create<memref::StoreOp>(loc, (accum_factory[idx])(shm_op0, shm_op1),
                                shared_mem_map[root_op], var_n);
    }
    b.create<scf::YieldOp>(loc, ValueRange({}));
    b.setInsertionPointAfter(if_is_lt_stride_rows);
  }

  // __syncthreads();
  b.create<gpu::BarrierOp>(loc);

  // if (local_row_index == 0) {
  // 		if (is_valid) {
  //			partial_result = shm[n] + shm[var_tile_w + n];
  //			atomicAdd(&global[col_index], partial_result);
  //    }
  // }
  Value is_local_row_index_eq_zero = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, local_row_index, zero);
  Value is_atom_add =
      b.create<arith::AndIOp>(loc, is_local_row_index_eq_zero, is_valid);
  scf::IfOp if_is_atom_add =
      b.create<scf::IfOp>(loc, /*resultTypes*/ TypeRange{}, is_atom_add,
                          /*hasElseRegion*/ false);
  if_is_atom_add.getThenRegion().front().clear();
  b.setInsertionPointToStart(&if_is_atom_add.getThenRegion().front());
  Value shm_offset = b.create<mlir::arith::AddIOp>(loc, var_tile_w, var_n);
  for (auto root_pair : llvm::enumerate(col_reduction_roots)) {
    Operation* root_op = root_pair.value();
    int idx = root_pair.index();
    Value shm_op0 =
        b.create<memref::LoadOp>(loc, shared_mem_map[root_op], var_n);
    Value shm_op1 =
        b.create<memref::LoadOp>(loc, shared_mem_map[root_op], shm_offset);
    Value partial_result = (accum_factory[idx])(shm_op0, shm_op1);
    auto root_element_type = getLhloOpsElementType(root_op);
    b.create<memref::AtomicRMWOp>(
        loc, root_element_type,
        getAtomicRMWKind(cast<lmhlo::ReduceOp>(root_op).getBody()),
        partial_result, root_op->getOperand(2), ValueRange({col_index}));
  }
  b.create<scf::YieldOp>(loc, ValueRange({}));
  b.setInsertionPointToEnd(parallel_op.getBody());
  b.create<scf::YieldOp>(loc, ValueRange({}));  // parallel

  cleanUnusedLhloOps(parent);
  return success();
}

Value createSharedMemoryForOp(OpBuilder& b, Location loc, Operation* op,
                              int64_t size) {
  const auto elemType = getLhloOpsElementType(op);
  Type shuffleElemType;
  if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
    return nullptr;
  }
  return createSharedMemory(b, loc, size, shuffleElemType);
}

LogicalResult emitFirstRoundShuffleStitch(
    OpBuilder& b, Location loc, Operation* op,
    const SmallVectorImpl<Value>& warp_reduce_result_shm,
    mlir::ResultRange::iterator acc_iter, Value warp_id, Value lane_id_is_zero,
    int64_t row_tile, int64_t reduce_threads, Value result_buffer_shm,
    const SmallVectorImpl<Value>& row_ids,
    const SmallVectorImpl<Value>& block_row_offset, bool external_output_only,
    bool is_output) {
  Value warp_size =
      b.create<arith::ConstantIntOp>(loc, kWarpSize, b.getIntegerType(32));
  auto xorAttr = b.getStringAttr("xor");
  SmallVector<Value, 2> sum_vec(row_tile);
  for (int i = 0; i < row_tile; i++) {
    sum_vec[i] = *(acc_iter + i);
  }
  const auto elemType = getLhloOpsElementType(op);
  Type shuffleElemType;
  if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
    return failure();
  }
  AccumulatorFactory accumFactory =
      getFactory(b, op->getLoc(), cast<lmhlo::ReduceOp>(op).getBody());
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    Value offset_val =
        b.create<arith::ConstantIntOp>(loc, offset, b.getIntegerType(32));
    for (int i = 0; i < row_tile; i++) {
      if (elemType != shuffleElemType) {
        if (failed(extendShuffleElemType(b, loc, sum_vec[i], shuffleElemType,
                                         &sum_vec[i]))) {
          return failure();
        }
      }
    }
    SmallVector<Value, 2> shuffle_result_vec(row_tile);
    for (int i = 0; i < row_tile; i++) {
      shuffle_result_vec[i] =
          emitWidthAdaptShuffle(b, loc, sum_vec[i], shuffleElemType, offset_val,
                                warp_size, gpu::ShuffleMode::XOR);
    }
    for (int i = 0; i < row_tile; i++) {
      if (elemType != shuffleElemType) {
        if (failed(truncateShuffleElemType(b, loc, sum_vec[1], elemType,
                                           &sum_vec[i])) ||
            failed(truncateShuffleElemType(b, loc, shuffle_result_vec[i],
                                           elemType, &shuffle_result_vec[i]))) {
          return failure();
        }
      }
    }
    for (int i = 0; i < row_tile; i++) {
      sum_vec[i] = accumFactory(sum_vec[i], shuffle_result_vec[i]);
    }
  }
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  auto if_lane_id_is_zero =
      b.create<scf::IfOp>(loc, /*resultTypes*/ ArrayRef<Type>{},
                          lane_id_is_zero, /*hasElseRegion*/ false);
  if_lane_id_is_zero.getThenRegion().front().clear();
  b.setInsertionPointToStart(&if_lane_id_is_zero.getThenRegion().front());
  auto shared_idx = warp_id;
  SmallVector<Value, 4> shared_indices;
  shared_indices.push_back(shared_idx);
  for (int i = 0; i < row_tile; i++) {
    if (elemType != shuffleElemType) {
      if (failed(extendShuffleElemType(b, loc, sum_vec[i], shuffleElemType,
                                       &sum_vec[i]))) {
        return failure();
      }
    }
  }
  if (reduce_threads == kWarpSize) {
    if (result_buffer_shm != nullptr && !external_output_only) {
      if (row_tile > 1) {
        createAlignMemrefWithTile(b, result_buffer_shm, row_tile);
      }
      for (int i = 0; i < row_tile; i++) {
        // The elements store in result shm buffer are in index order. (Note it
        // is linear index.)
        b.create<memref::StoreOp>(loc, sum_vec[i], result_buffer_shm,
                                  block_row_offset[i]);
      }
    }
    if (is_output) {
      auto res_memref = op->getOperand(2);
      if (row_tile > 1) {
        createAlignMemrefWithTile(b, res_memref, row_tile);
      }
      for (int i = 0; i < row_tile; i++) {
        b.create<memref::StoreOp>(loc, sum_vec[i], res_memref, row_ids[i]);
      }
    }
  } else {
    for (int i = 0; i < row_tile; i++) {
      b.create<memref::StoreOp>(loc, sum_vec[i], warp_reduce_result_shm[i],
                                shared_indices);
    }
  }
  b.create<scf::YieldOp>(loc, ValueRange({}));
  b.setInsertionPointAfter(if_lane_id_is_zero);
  return success();
}

LogicalResult emitSecondRoundShuffleStitch(
    OpBuilder& b, Location loc, Operation* op, Value init_value,
    Value thread_id, Value warp_id, Value lane_id, int row_tile, int block_size,
    const SmallVectorImpl<Value>& warp_reduce_result_shm,
    SmallVector<Value> output_indices, Value result_buffer_shm, bool is_output,
    bool external_output_only) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value thread_id_is_zero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, thread_id, zero);
  SmallVector<Type, 4> root_elem_types(row_tile);
  auto elemType = getLhloOpsElementType(op);
  for (int i = 0; i < row_tile; i++) {
    root_elem_types[i] = elemType;
  }

  // Only the first warp needs to do the second round shuffle reduction.
  Value is_first_warp =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, warp_id,
                              b.create<arith::ConstantIndexOp>(loc, 0));
  scf::IfOp if_is_first_warp =
      b.create<scf::IfOp>(loc, /*resultTypes*/ TypeRange{}, is_first_warp,
                          /*hasElseRegion*/ false);

  b.setInsertionPointToStart(&if_is_first_warp.getThenRegion().front());

  Value lane_id_inbound = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, lane_id,
      b.create<arith::ConstantIndexOp>(loc, block_size / kWarpSize));
  scf::IfOp if_lane_id_inbound =
      b.create<scf::IfOp>(loc, /*resultTypes*/ root_elem_types, lane_id_inbound,
                          /*hasElseRegion*/ true);
  if_lane_id_inbound.getThenRegion().front().clear();
  if_lane_id_inbound.getElseRegion().front().clear();
  b.setInsertionPointToStart(&if_lane_id_inbound.getThenRegion().front());
  auto shared_idx = lane_id;
  SmallVector<Value, 4> shared_indices;
  shared_indices.push_back(shared_idx);
  SmallVector<Value, 4> true_yield_values(row_tile);
  Type shuffleElemType;
  if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
    return failure();
  }
  for (int i = 0; i < row_tile; i++) {
    auto shared_mem = warp_reduce_result_shm[i];
    Value shared_mem_value =
        b.create<memref::LoadOp>(loc, shared_mem, shared_indices);
    if (elemType != shuffleElemType) {
      if (failed(truncateShuffleElemType(b, loc, shared_mem_value, elemType,
                                         &shared_mem_value))) {
        return failure();
      }
    }
    true_yield_values[i] = shared_mem_value;
  }
  b.create<scf::YieldOp>(loc, true_yield_values);
  b.setInsertionPointToStart(&if_lane_id_inbound.getElseRegion().front());
  SmallVector<Value, 4> false_yield_values(row_tile);

  for (int i = 0; i < row_tile; i++) {
    false_yield_values[i] = init_value;
  }
  b.create<scf::YieldOp>(loc, false_yield_values);
  b.setInsertionPointAfter(if_lane_id_inbound);

  assert(block_size % kWarpSize == 0);
  int num_warps = block_size / kWarpSize;
  Value shuffle_size =
      b.create<arith::ConstantIntOp>(loc, num_warps, b.getIntegerType(32));
  auto xorAttr = gpu::ShuffleMode::XOR;
  SmallVector<Value> results(row_tile);
  for (int i = 0; i < row_tile; i++) {
    results[i] = if_lane_id_inbound.getResult(i);
  }
  AccumulatorFactory accumFactory =
      getFactory(b, op->getLoc(), cast<lmhlo::ReduceOp>(op).getBody());
  for (int offset = 1; offset < num_warps; offset <<= 1) {
    Value offset_val =
        b.create<arith::ConstantIntOp>(loc, offset, b.getIntegerType(32));
    for (int i = 0; i < row_tile; i++) {
      if (elemType != shuffleElemType) {
        if (failed(extendShuffleElemType(b, loc, results[i], shuffleElemType,
                                         &results[i]))) {
          return failure();
        }
      }
    }
    SmallVector<Value, 4> shuffle_result_vec(row_tile);
    for (int i = 0; i < row_tile; i++) {
      shuffle_result_vec[i] =
          emitWidthAdaptShuffle(b, loc, results[i], shuffleElemType, offset_val,
                                shuffle_size, xorAttr);
    }

    for (int i = 0; i < row_tile; i++) {
      if (elemType != shuffleElemType) {
        if (failed(truncateShuffleElemType(b, loc, results[i], elemType,
                                           &results[i])) ||
            failed(truncateShuffleElemType(b, loc, shuffle_result_vec[i],
                                           elemType, &shuffle_result_vec[i]))) {
          return failure();
        }
      }
    }
    for (int i = 0; i < row_tile; i++) {
      results[i] = accumFactory(results[i], shuffle_result_vec[i]);
    }
  }

  auto if_thread_id_is_zero =
      b.create<scf::IfOp>(loc, /*resultTypes*/ ArrayRef<Type>{},
                          thread_id_is_zero, /*hasElseRegion*/ false);
  if_thread_id_is_zero.getThenRegion().front().clear();
  b.setInsertionPointToStart(&if_thread_id_is_zero.getThenRegion().front());
  if (result_buffer_shm != nullptr && !external_output_only) {
    if (row_tile > 1) {
      createAlignMemrefWithTile(b, result_buffer_shm, row_tile);
    }
    for (int i = 0; i < row_tile; i++) {
      Value index = b.create<arith::ConstantIndexOp>(loc, i);
      b.create<memref::StoreOp>(loc, results[i], result_buffer_shm, index);
    }
  }
  if (is_output) {
    auto res_memref = *(op->getOperands().begin() + (op->getNumOperands() - 1));
    if (row_tile > 1) {
      createAlignMemrefWithTile(b, res_memref, row_tile);
    }
    for (int i = 0; i < row_tile; i++) {
      b.create<memref::StoreOp>(loc, results[i], res_memref, output_indices[i]);
    }
  }
  b.create<scf::YieldOp>(loc, ValueRange({}));
  // b.setInsertionPointAfter(if_thread_id_is_zero);

  b.setInsertionPointAfter(if_is_first_warp);

  return success();
}

LogicalResult emitRowReduceThreadBlock(
    OpBuilder& b, Location loc, Block* block, Operation* op, Value block_id,
    Value thread_id, int64_t block_size, int row_reduction_schedule,
    int64_t row_tile, Value result_buffer_shm, ShapeAnalysis* shape_analysis,
    bool is_output, bool external_output_only) {
  Value lhs = op->getOperand(0);
  const MemRefType& lhs_type = lhs.getType().template cast<MemRefType>();
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value shape_h = b.create<memref::DimOp>(loc, lhs, zero);
  Value shape_w = b.create<memref::DimOp>(loc, lhs, one);

  Value warp_size_val = b.create<arith::ConstantIndexOp>(loc, kWarpSize);
  Value warp_id = b.create<arith::DivUIOp>(loc, thread_id, warp_size_val);
  Value lane_id = b.create<arith::RemUIOp>(loc, thread_id, warp_size_val);
  Value lane_id_is_zero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane_id, zero);
  Value row_tile_val = b.create<arith::ConstantIndexOp>(loc, row_tile);
  auto init_value = b.create<memref::LoadOp>(
      loc, *(cast<lmhlo::ReduceOp>(op).getInitValues().begin()));

  int reduce_threads = (row_reduction_schedule == DISC_BLOCK_WISE_ROW_REDUCE)
                           ? block_size
                           : kWarpSize;
  assert(block_size % reduce_threads == 0);
  int64_t row_per_block_wave = block_size / reduce_threads;
  int64_t row_per_block = row_per_block_wave * row_tile;
  Value row_per_block_val =
      b.create<arith::ConstantIndexOp>(loc, row_per_block_wave * row_tile);
  Value block_row_base =
      b.create<arith::MulIOp>(loc, block_id, row_per_block_val);
  SmallVector<Value> block_row_offset(row_tile);
  SmallVector<Value> row_ids(row_tile);
  for (int i = 0; i < row_tile; i++) {
    if (row_reduction_schedule == DISC_BLOCK_WISE_ROW_REDUCE) {
      block_row_offset[i] = b.create<arith::ConstantIndexOp>(loc, i);
    } else {
      // Every warp processes adjacent `row_tile` rows.
      block_row_offset[i] = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, warp_id, row_tile_val),
          b.create<arith::ConstantIndexOp>(loc, i));
    }
    row_ids[i] =
        b.create<arith::AddIOp>(loc, block_row_base, block_row_offset[i]);
  }

  scf::IfOp if_row_in_bound;
  if (reduce_threads == kWarpSize) {
    // We only check the largest row-id processed in this thread.
    auto row_in_bound = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                row_ids[row_tile - 1], shape_h);
    if_row_in_bound =
        b.create<scf::IfOp>(loc, /*resultTypes*/ ArrayRef<Type>{}, row_in_bound,
                            /*hasElseRegion*/ false);
    if_row_in_bound.getThenRegion().front().clear();
    b.setInsertionPointToStart(&if_row_in_bound.getThenRegion().front());
  }

  // First, emit in-thread local reduction.
  SmallVector<Value, 4> init_values(row_tile);
  for (int i = 0; i < row_tile; i++) {
    init_values[i] = init_value;
  }
  Value var_j = nullptr;
  Value reduce_threads_val =
      b.create<arith::ConstantIndexOp>(loc, reduce_threads);
  Value lb = (reduce_threads == kWarpSize) ? lane_id : thread_id;
  scf::ForOp for_op_j =
      createLoopAndSetInsPt(b, loc, var_j,
                            /*lb*/ lb, /*ub*/ shape_w,
                            /*step*/ reduce_threads_val, init_values);

  SmallVector<Value, 4> in_thread_reduction;
  for (int i = 0; i < row_tile; i++) {
    SmallVector<Value, 4> multidim({row_ids[i], var_j});
    ValueRange input_index(multidim);
    if (failed(emitPerElementReductionImpl(
            b, loc, op, for_op_j.getRegionIterArgs().begin() + i, input_index,
            in_thread_reduction))) {
      return failure();
    }
  }
  b.create<scf::YieldOp>(loc, in_thread_reduction);

  // Second, emit per-warp reduction.
  b.setInsertionPointAfter(for_op_j);
  SmallVector<Value> warp_reduce_result_shm(row_tile);
  for (int i = 0; i < row_tile; i++) {
    Value shared_mem = createSharedMemoryForOp(b, loc, op, kWarpSize);
    if (shared_mem == nullptr) {
      return failure();
    }
    warp_reduce_result_shm[i] = shared_mem;
  }
  if (failed(emitFirstRoundShuffleStitch(
          b, loc, op, warp_reduce_result_shm, for_op_j.getResults().begin(),
          warp_id, lane_id_is_zero, row_tile, reduce_threads, result_buffer_shm,
          row_ids, block_row_offset, external_output_only, is_output))) {
    return failure();
  }

  // Finally, for block-wise schedule, emit second round reduction.
  if (reduce_threads == kWarpSize) {
    b.create<scf::YieldOp>(loc, ValueRange({}));
    b.setInsertionPointAfter(if_row_in_bound);
  } else {
    b.create<gpu::BarrierOp>(loc);
    if (failed(emitSecondRoundShuffleStitch(
            b, loc, op, init_value, thread_id, warp_id, lane_id, row_tile,
            block_size, warp_reduce_result_shm, row_ids, result_buffer_shm,
            is_output, external_output_only))) {
      return failure();
    }
  }

  b.setInsertionPointToEnd(block);
  return success();
}

LogicalResult initSkeletonGrpsAndCloneOps(
    lmhlo::FusionOp& fusion_op, FusionPattern& fusion_pattern,
    SmallVector<FusionPattern::SkeletonGroup>& skeleton_groups,
    DenseMap<Operation*, Value>& shm_cached_ops_and_view,
    DenseMap<Operation*, SmallVector<Operation*>>& skeleton_group_ops,
    ShapeAnalysis* shape_analysis, LowerConfig& lower_config, bool merge_group,
    int row_per_block, int shmem_limit_bytes) {
  if (!getOrderedSkeletonGroups(fusion_pattern, skeleton_groups)) {
    return failure();
  }
  if (merge_group) {
    if (!mergeSkeletonGroupsInOrder(fusion_pattern, skeleton_groups,
                                    shape_analysis)) {
      return failure();
    }
  }
  auto op_list = fusion_pattern.getOpList();
  auto sub_root_ops = fusion_pattern.getSubRootOps();
  DenseSet<Operation*> sub_roots(sub_root_ops.begin(), sub_root_ops.end());
  DenseSet<Operation*> regular_xroots = fusion_pattern.getRegularXroots();
  DenseSet<Operation*> irregular_xroots = fusion_pattern.getIrregularXroots();

  // Find ops in the group and reorder according to `op_list`.
  shm_cached_ops_and_view.clear();
  int shmem_usage_bits = 0;
  for (auto& skeleton_group : skeleton_groups) {
    auto skeletons = skeleton_group.skeletons;
    for (auto& skeleton : skeletons) {
      auto output_type = getLhloOpsElementType(skeleton);
      auto bit_width = output_type.getIntOrFloatBitWidth();
      shmem_usage_bits += bit_width;
    }
  }
  int shmem_limit_bits = (shmem_limit_bytes == -1) ? -1 : shmem_limit_bytes * 8;
  DenseSet<Operation*> shm_cached_ops;
  // { skeleton, group-ops-inorder }
  DenseMap<Operation*, SmallVector<Operation*>> skeleton_group_ops_old;
  for (auto& skeleton_group : skeleton_groups) {
    auto skeletons = skeleton_group.skeletons;
    // Find ops in the group and reorder according to `op_list`.
    DenseSet<Operation*> group_ops;
    fusion_pattern.findOpsOfSkeletonGroup(
        skeleton_group, group_ops, shm_cached_ops, skeleton_group_ops_old,
        row_per_block, shmem_usage_bits, shmem_limit_bits);
    SmallVector<Operation*>& group_ops_inorder =
        skeleton_group_ops_old[skeletons[0]];
    for (auto op : op_list) {
      if (group_ops.contains(op)) {
        group_ops_inorder.push_back(op);
      }
    }
  }
  auto createViewOfMemref = [&](Value memref) {
    auto op = memref.getDefiningOp();
    OpBuilder b(op);
    b.setInsertionPointAfter(op);
    return createViewLike(b, memref.getLoc(), memref, memref);
  };
  for (auto op : shm_cached_ops) {
    auto result = op->getOperand(op->getNumOperands() - 1);
    Value view = createViewOfMemref(result);
    shm_cached_ops_and_view[op] = view;
  }

  // Clone ops in each group and build written flags in `lower_config`.
  // Note that for irregular-xroot occurs in several skeleton-groups, it
  // will not be cloned in the first occurance in a group, but will be
  // cloned in the following occurances. Meanwhile, the first occured group
  // will write the value of this irregular-xroot back, and others are not
  // responsible for the duty of writing output.
  DenseSet<Operation*> visited_irregular_xroots;
  DenseMap<Value, TileInfo>& tile_plan = fusion_pattern.getTilePlan();
  auto allocClonedValue = [&](Value val) {
    OpBuilder b(fusion_op);
    Location loc = val.getLoc();
    auto ty = val.getType().cast<MemRefType>();
    auto shape = ty.getShape();
    SmallVector<Value> dims;
    for (int d = 0; d < ty.getRank(); ++d) {
      if (shape[d] == ShapedType::kDynamic) {
        dims.push_back(b.create<memref::DimOp>(loc, val, d));
      }
    }
    auto new_val = b.create<memref::AllocOp>(loc, ty, dims);
    auto tile = tile_plan.find(val);
    if (tile != tile_plan.end()) {
      tile_plan[new_val] = tile->second;
    }
    return new_val;
  };
  DenseSet<Operation*> to_erase;
  auto& fused_block = fusion_op.getRegion().front();
  auto& terminator = *fused_block.rbegin();
  OpBuilder builder(&terminator);
  Operation* last_op = nullptr;
  skeleton_group_ops.clear();
  for (auto& skeleton_group : skeleton_groups) {
    auto skeletons = skeleton_group.skeletons;
    SmallVector<Operation*>& group_ops_inorder_old =
        skeleton_group_ops_old[skeletons[0]];
    IRMapping cloning_map;
    for (auto shm_cached_op : shm_cached_ops_and_view) {
      auto op = shm_cached_op.first;
      auto view = shm_cached_op.second;
      // the shm cached op is not in the current group.
      if (llvm::find(group_ops_inorder_old, op) ==
          group_ops_inorder_old.end()) {
        cloning_map.map(op->getOperand(op->getNumOperands() - 1), view);
      }
    }

    SmallVector<Operation*>& group_ops_inorder =
        skeleton_group_ops[skeletons[0]];
    for (int64_t i = 0; i < group_ops_inorder_old.size(); i++) {
      auto op = group_ops_inorder_old[i];
      int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
      auto updated = op;
      // Alloc for, if necessary, and update output operands.
      if (regular_xroots.contains(op) ||
          (irregular_xroots.contains(op) &&
           !visited_irregular_xroots.contains(op))) {
        // Update input operands.
        // Replace input with new allocated ones in previous iterations.
        for (int64_t j = 0; j < num_input_operand; j++) {
          auto operand = updated->getOperand(j);
          if (auto new_operand = cloning_map.lookup(operand)) {
            updated->replaceUsesOfWith(operand, new_operand);
          }
        }
        // For non-skeleton xroot op, it should be write back during lowering.
        if (skeleton_group_ops_old.find(op) == skeleton_group_ops_old.end()) {
          lower_config.setWrittenBack(op);
        }
      } else if (shm_cached_ops.contains(op)) {
        // Update input operands.
        // Replace input with new allocated ones in previous iterations.
        for (int64_t j = 0; j < num_input_operand; j++) {
          auto operand = updated->getOperand(j);
          if (auto new_operand = cloning_map.lookup(operand)) {
            updated->replaceUsesOfWith(operand, new_operand);
          }
        }
      } else {
        // For to-be-cloned ops, alloc for output and then clone the op with new
        // outputs.
        SmallVector<Value> results;
        for (Value v : op->getOperands().drop_front(num_input_operand)) {
          Value new_operand = allocClonedValue(v);
          cloning_map.map(v, new_operand);
          results.push_back(v);
        }
        updated = builder.clone(*op, cloning_map);
        for (auto result : results) {
          fusion_pattern.updateLastWriter(result, updated);
        }
        if (!irregular_xroots.contains(op)) {
          to_erase.insert(op);
        }
      }
      group_ops_inorder.push_back(updated);
      // Make sure the ops are in order.
      if (last_op != nullptr) {
        updated->moveAfter(last_op);
      }
      last_op = updated;
    }
    visited_irregular_xroots.insert(
        skeleton_group.irregular_root_member_set.begin(),
        skeleton_group.irregular_root_member_set.end());
  }
  // Clean been cloned ops. Note xroot ops are not to be erased.
  for (auto op : to_erase) {
    op->erase();
  }

  return success();
}

LogicalResult lowerWithScheduleStitch(lmhlo::FusionOp& fusion_op,
                                      FusionPattern& fusion_pattern,
                                      ShapeAnalysis* shape_analysis,
                                      int64_t row_tile,
                                      LowerConfig& lower_config,
                                      int row_reduction_schedule) {
  auto root_ops = fusion_pattern.getRootOps();
  auto sub_root_ops = fusion_pattern.getSubRootOps();
  auto result_values = fusion_pattern.getResults();
  DenseSet<Operation*> roots(root_ops.begin(), root_ops.end());
  DenseSet<Value> results(result_values.begin(), result_values.end());
  auto op_list = fusion_pattern.getOpList();
  DenseSet<Operation*> op_set(op_list.begin(), op_list.end());
  DenseSet<Operation*> external_only_roots;
  for (auto value : fusion_pattern.getExternalOnlyResults()) {
    external_only_roots.insert(fusion_pattern.findLastWriter(value));
  }
  auto parent = &(fusion_op.getRegion().front());
  auto dominant_op = fusion_pattern.getDominantOp();
  auto tile_plan = fusion_pattern.getTilePlan();

  for (auto op : sub_root_ops) {
    if (!isRank2RowReduction(op)) {
      LLVM_DEBUG(llvm::dbgs() << "False sub-root: " << *op << "\n");
      return failure();
    }
    auto tile_info = tile_plan.find(op->getOperand(0));
    if (tile_info == tile_plan.end() ||
        tile_info->second.tileSizes.count(0) != 0 ||
        tile_info->second.tileSizes.count(1) != 1) {
      LLVM_DEBUG(llvm::dbgs() << "False tile info for: " << *op << "\n");
      return failure();
    }
  }

  const int thread_per_block = getThreadPerBlock(dominant_op);
  int reduce_threads = (row_reduction_schedule == DISC_BLOCK_WISE_ROW_REDUCE)
                           ? thread_per_block
                           : kWarpSize;
  int64_t row_per_block = thread_per_block / reduce_threads * row_tile;

  SmallVector<FusionPattern::SkeletonGroup> skeleton_groups;
  DenseMap<Operation*, Value> shm_cached_ops;
  DenseMap<Operation*, SmallVector<Operation*>> skeleton_group_ops;
  if (failed(initSkeletonGrpsAndCloneOps(
          fusion_op, fusion_pattern, skeleton_groups, shm_cached_ops,
          skeleton_group_ops, shape_analysis, lower_config, false,
          row_per_block, 0))) {
    LLVM_DEBUG(llvm::dbgs() << "Fail to init skeleton groups or clone.\n");
    return failure();
  }

  Location loc = dominant_op->getLoc();

  // Note that we only support row-reduction to be dominant op currently.
  assert(isRank2RowReduction(dominant_op));
  Value lhs = dominant_op->getOperand(0);
  const MemRefType& lhs_type = lhs.getType().template cast<MemRefType>();

  // Shared memory buffer allocation for sub-roots' result. memref.
  DenseMap<Value, Value> shm_mapping;

  // Codegen of skeleton ops (i.e. roots/sub-roots). As for sub-roots, emit
  // their direct users.
  OpBuilder b(root_ops.back());
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value shape_h = b.create<memref::DimOp>(loc, lhs, zero);
  Value num_threads = b.create<arith::ConstantIndexOp>(loc, thread_per_block);

  // loop over (block-number, threads-per-block)
  // Note that we know `shape_h` can be divided by `row_tile` exactly.
  Value row_per_block_val = b.create<arith::ConstantIndexOp>(
      loc, thread_per_block / reduce_threads * row_tile);
  Value num_blocks =
      b.create<arith::CeilDivUIOp>(loc, shape_h, row_per_block_val);

  SmallVector<Value, 2> vars;
  scf::ParallelOp parallel_op = createParallelAndSetInsPt(
      b, loc, vars, {zero, zero}, {num_blocks, num_threads}, {one, one}, {});
  parallel_op.getBody()->clear();
  b.setInsertionPointToStart(parallel_op.getBody());
  Value block_id = vars[0];
  Value thread_id = vars[1];
  Value warp_size_val = b.create<arith::ConstantIndexOp>(loc, kWarpSize);
  Value warp_id = b.create<arith::DivUIOp>(loc, thread_id, warp_size_val);
  Value lane_id = b.create<arith::RemUIOp>(loc, thread_id, warp_size_val);
  Value row_tile_val = b.create<arith::ConstantIndexOp>(loc, row_tile);

  for (auto& skeleton_group : skeleton_groups) {
    auto skeleton = skeleton_group.skeletons[0];
    Location loc = skeleton->getLoc();
    b.setInsertionPointToEnd(parallel_op.getBody());

    if (isRank2RowReduction(skeleton)) {
      Value out_value = cast<lmhlo::LmhloOp>(skeleton).getResultBuffer();
      Value result_shmem;
      result_shmem = createSharedMemoryForOp(b, loc, skeleton, row_per_block);
      if (result_shmem == nullptr) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Create shared memory failed for: " << *skeleton << "\n");
        return failure();
      }
      shm_mapping[out_value] = result_shmem;

      auto tile_info = tile_plan.find(out_value);
      if (tile_info == tile_plan.end() ||
          tile_info->second.tileSizes.size() != 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Tile info error for: " << *skeleton << "\n");
        return failure();
      }
      LowerConfig::SpecificLoader loader(result_shmem, row_per_block);
      lower_config.setSpecificLoader(
          std::make_pair(fusion_op.getOperation(), out_value), loader);

      bool external_only = external_only_roots.contains(skeleton);
      if (failed(emitRowReduceThreadBlock(
              b, loc, parallel_op.getBody(), skeleton, block_id, thread_id,
              thread_per_block, row_reduction_schedule, row_tile, result_shmem,
              shape_analysis, roots.contains(skeleton), external_only))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to emit InBlockRowReduce for: "
                                << *skeleton << "\n");
        return failure();
      }
      // TODO: check whether the xroots in this group are required by other
      // groups. If not, no barrier.
      // TODO: warp-wise reduction does not need block-barrier.
      bool require_barrier =
          !external_only ||
          llvm::any_of(skeleton_group.root_member_list, [&](Operation* op) {
            return !external_only_roots.contains(op);
          });

      if (require_barrier) {
        b.create<gpu::BarrierOp>(loc);
      }
    } else {
      Value out_value = cast<lmhlo::LmhloOp>(skeleton).getResultBuffer();
      auto tile_info = tile_plan.find(out_value);
      if (tile_info == tile_plan.end()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Failed to find tile info: " << *skeleton << "\n");
        return failure();
      }

      // Get row-id according to codegen schedule.
      // TODO: reuse row_ids between sub-roots and external-only-roots.
      Value block_row_base =
          b.create<arith::MulIOp>(loc, block_id, row_per_block_val);
      SmallVector<Value> row_ids(row_tile);
      for (int i = 0; i < row_tile; i++) {
        Value block_row_offset;
        if (row_reduction_schedule == DISC_BLOCK_WISE_ROW_REDUCE) {
          block_row_offset = b.create<arith::ConstantIndexOp>(loc, i);
        } else {
          // Every warp processes adjacent `row_tile` rows.
          block_row_offset = b.create<arith::AddIOp>(
              loc, b.create<arith::MulIOp>(loc, warp_id, row_tile_val),
              b.create<arith::ConstantIndexOp>(loc, i));
        }
        row_ids[i] =
            b.create<arith::AddIOp>(loc, block_row_base, block_row_offset);
      }

      scf::IfOp if_row_in_bound;
      if (row_reduction_schedule == DISC_WARP_WISE_ROW_REDUCE) {
        // We only check the largest row-id processed in this thread.
        auto row_in_bound = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, row_ids[row_tile - 1], shape_h);
        if_row_in_bound =
            b.create<scf::IfOp>(loc, /*resultTypes*/ ArrayRef<Type>{},
                                row_in_bound, /*hasElseRegion*/ false);
        if_row_in_bound.getThenRegion().front().clear();
        b.setInsertionPointToStart(&if_row_in_bound.getThenRegion().front());
      }

      SmallVector<Value> outShapeValues = getShapeValues(&b, out_value);
      // Deal with the case that tiled dims are not the same between result and
      // sub-roots' input.
      Value tiled_linear = nullptr;
      for (auto en : llvm::enumerate(outShapeValues)) {
        if (tile_info->second.tileSizes.count(en.index()) > 0) {
          tiled_linear =
              (tiled_linear == nullptr)
                  ? en.value()
                  : b.create<arith::MulIOp>(loc, tiled_linear, en.value());
        }
      }
      if (tiled_linear == nullptr) {
        tiled_linear = b.create<arith::ConstantIndexOp>(loc, 1);
      }
      Value reduce_threads_val =
          b.create<arith::ConstantIndexOp>(loc, reduce_threads);
      Value var_j = nullptr;
      Value lb = row_reduction_schedule == DISC_BLOCK_WISE_ROW_REDUCE
                     ? thread_id
                     : lane_id;
      scf::ForOp for_op_j =
          createLoopAndSetInsPt(b, loc, var_j,
                                /*lb*/ lb, /*ub*/ tiled_linear,
                                /*step*/ reduce_threads_val, {});
      for (int64_t i = 0; i < row_tile; i++) {
        Value index = b.create<arith::AddIOp>(
            loc, b.create<arith::MulIOp>(loc, row_ids[i], tiled_linear), var_j);
        if (failed(lowerHelper(b, loc, skeleton, index, shape_analysis, 1,
                               &lower_config))) {
          LLVM_DEBUG(llvm::dbgs() << "Failed to lower: " << *skeleton << "\n");
          return failure();
        }
      }

      b.setInsertionPointAfter(for_op_j);
      if (row_reduction_schedule == DISC_WARP_WISE_ROW_REDUCE) {
        b.create<scf::YieldOp>(loc, ValueRange({}));
        b.setInsertionPointAfter(if_row_in_bound);
      }

      // Currently, a non-sub-root skeleton op will always be external only. We
      // many need the following check for non-reduce sub-root someday.
      // if (!external_only_roots.contains(skeleton)) {
      //   // TODO: use __threadfence_block instead.
      //   b.create<gpu::BarrierOp>(loc);
      // }
    }
  }

  b.setInsertionPointToEnd(parallel_op.getBody());
  b.create<scf::YieldOp>(loc, ValueRange({}));

  for (auto& skeleton_group : skeleton_groups) {
    auto skeleton = skeleton_group.skeletons[0];
    skeleton->erase();
  }

  // remove the root_op if it has no other users except the memref
  cleanUnusedLhloOps(parent);

  return success();
}

LogicalResult emitWarpReduce(
    OpBuilder& b, Location loc, SmallVectorImpl<Operation*>& ops,
    SmallVectorImpl<AccumulatorFactory>& accumFactories,
    SmallVectorImpl<Value>& targets, int width) {
  SmallVector<Type> elemTypes;
  SmallVector<Type> shuffleElemTypes;
  for (auto op : ops) {
    Type elemType = getLhloOpsElementType(op);
    Type shuffleElemType;
    if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
      return failure();
    }
    elemTypes.push_back(elemType);
    shuffleElemTypes.push_back(shuffleElemType);
  }
  // const auto elemType = getLhloOpsElementType(op);
  // Type shuffleElemType;
  Value v_width =
      b.create<arith::ConstantIntOp>(loc, width, b.getIntegerType(32));
  for (int i = 1; i < width; i <<= 1) {
    Value offset = b.create<arith::ConstantIntOp>(loc, i, b.getIntegerType(32));
    // Do not merge the following loops. The loops are for ILP optimization.
    for (int64_t j = 0; j < ops.size(); j++) {
      if (elemTypes[j] != shuffleElemTypes[j]) {
        if (failed(extendShuffleElemType(b, loc, targets[j],
                                         shuffleElemTypes[j], &targets[j]))) {
          return failure();
        }
      }
    }
    SmallVector<Value> shuffle_results;
    for (int64_t j = 0; j < ops.size(); j++) {
      Value shuffle_res =
          emitWidthAdaptShuffle(b, loc, targets[j], shuffleElemTypes[j], offset,
                                v_width, gpu::ShuffleMode::XOR);
      shuffle_results.push_back(shuffle_res);
    }
    for (int64_t j = 0; j < ops.size(); j++) {
      if (elemTypes[j] != shuffleElemTypes[j]) {
        if (failed(truncateShuffleElemType(b, loc, targets[j], elemTypes[j],
                                           &targets[j])) ||
            failed(truncateShuffleElemType(b, loc, shuffle_results[j],
                                           elemTypes[j],
                                           &shuffle_results[j]))) {
          return failure();
        }
      }
    }
    for (int64_t j = 0; j < ops.size(); j++) {
      targets[j] = accumFactories[j](targets[j], shuffle_results[j]);
    }
  }
  return success();
}

LogicalResult emitWarpReduce(OpBuilder& b, Location loc, Operation* op,
                             AccumulatorFactory& accumFactory, Value& target,
                             Value width) {
  const auto elemType = getLhloOpsElementType(op);
  Type shuffleElemType;
  if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
    return failure();
  }

  width = b.create<arith::IndexCastOp>(loc, b.getIntegerType(32), width);

  // Build the while loop for the reduce.
  SmallVector<Type> while_types({b.getIntegerType(32), elemType});
  Value one = b.create<arith::ConstantIntOp>(loc, 1, b.getIntegerType(32));
  SmallVector<Value> while_in({one, target});
  scf::WhileOp while_warp_reduce =
      b.create<scf::WhileOp>(loc, while_types, while_in);

  // Build the "before" region, which effectively consists of a
  // conjunction of "i < upper" tests on all induction.
  SmallVector<Location> locs(while_types.size(), loc);
  Block* before =
      b.createBlock(&while_warp_reduce.getBefore(), {}, while_types, locs);
  b.setInsertionPointToStart(before);
  {
    Value reduce_index = before->getArgument(0);
    Value cond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                         reduce_index, width);
    b.create<scf::ConditionOp>(loc, cond, before->getArguments());
  }

  // The computation body of `while` is in the `after` block.
  Block* after =
      b.createBlock(&while_warp_reduce.getAfter(), {}, while_types, locs);
  b.setInsertionPointToStart(after);
  {
    Value shuffle_offset = after->getArgument(0);
    Value input = after->getArgument(1);
    // TODO: move it out of the loop.
    if (elemType != shuffleElemType) {
      if (failed(
              extendShuffleElemType(b, loc, input, shuffleElemType, &input))) {
        return failure();
      }
    }
    Value shuffle_result =
        emitWidthAdaptShuffle(b, loc, input, shuffleElemType, shuffle_offset,
                              width, gpu::ShuffleMode::XOR);
    if (elemType != shuffleElemType) {
      if (failed(truncateShuffleElemType(b, loc, input, elemType, &input)) ||
          failed(truncateShuffleElemType(b, loc, shuffle_result, elemType,
                                         &shuffle_result))) {
        return failure();
      }
    }
    input = accumFactory(input, shuffle_result);

    // Update while condition.
    Value reduce_index_update = b.create<arith::MulIOp>(
        loc, shuffle_offset,
        b.create<arith::ConstantIntOp>(loc, 2, b.getIntegerType(32)));
    b.create<scf::YieldOp>(loc, ValueRange({reduce_index_update, input}));
  }
  b.setInsertionPointAfter(while_warp_reduce);
  target = while_warp_reduce.getResult(1);

  return success();
}

/*
 * The generated kernel for row-reduce is as following. The code is augly and
 * will be removed.
 *
 * template <typename T>
 * __global__ void row_reduce_combine_two(T *input, int rows, int cols,
 *                                        int threads_per_row, T *output) {
 *   const float init_value = 0.0;
 *   const int warp_id = threadIdx.x / 32;
 *   const int lane_id = threadIdx.x % 32;
 *   const int warps_per_block = blockDim.x / 32;

 *   int row_index;
 *   int tid_in_row;
 *   if (threads_per_row != 32) {
 *     row_index = blockIdx.x;
 *     tid_in_row = threadIdx.x;
 *   } else {
 *     row_index = blockIdx.x * warps_per_block + warp_id;
 *     tid_in_row = lane_id;
 *   }

 *   __shared__ float shared_data[32];
 *   if (row_index < rows) {
 *     float reduce_accum = init_value;
 *     for (int col_index = tid_in_row; col_index < cols;
 *          col_index += threads_per_row) {
 *       int index = row_index * cols + col_index;
 *       auto val = input[index];
 *       val = compute(val);
 *       reduce_accum += val;
 *     }

 *     auto mask = __ballot_sync(0xffffffff, tid_in_row < cols);
 *     if (tid_in_row < cols) {
 *       reduce_accum = warp_reduce_sum(mask, reduce_accum, 32);
 *     }

 *     if (lane_id == 0) {
 *       if (threads_per_row != 32) {
 *         shared_data[warp_id] = reduce_accum;
 *       } else {
 *         output[row_index] = reduce_accum;
 *       }
 *     }

 *     if (threads_per_row != 32) {
 *       __syncthreads();
 *       if (threadIdx.x < 32) {
 *         auto reduce_accum = warp_reduce_sum(
 *             unsigned(-1), shared_data[threadIdx.x], warps_per_block);
 *         if (lane_id == 0) {
 *           output[row_index] = reduce_accum;
 *         }
 *       }
 *     }
 *   }
 * }
 */
LogicalResult emitRowReduceThreadBlockV2(
    OpBuilder& b, Location loc, Block* block, SmallVectorImpl<Operation*>& ops,
    SmallVectorImpl<Value>& result_buffer_shms, int64_t block_size,
    int row_reduction_schedule, int64_t ilp_factor,
    ShapeAnalysis* shape_analysis, SmallVectorImpl<bool>& is_output,
    SmallVectorImpl<bool>& external_output_only) {
  Value block_dim =
      b.create<gpu::BlockDimOp>(loc, b.getIndexType(), gpu::Dimension::x);
  Value block_id =
      b.create<gpu::BlockIdOp>(loc, b.getIndexType(), gpu::Dimension::x);
  Value thread_id =
      b.create<gpu::ThreadIdOp>(loc, b.getIndexType(), gpu::Dimension::x);
  Value warp_size = b.create<arith::ConstantIndexOp>(loc, kWarpSize);
  Value warp_id = b.create<arith::DivUIOp>(loc, thread_id, warp_size);
  Value lane_id = b.create<arith::RemUIOp>(loc, thread_id, warp_size);

  SmallVector<lmhlo::ReduceOp> reduce_ops;
  for (auto op : ops) {
    reduce_ops.push_back(dyn_cast_or_null<lmhlo::ReduceOp>(op));
  }
  SmallVector<Value> lhses;
  for (auto op : ops) {
    lhses.push_back(op->getOperand(0));
  }
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value rows = b.create<memref::DimOp>(loc, lhses[0], zero);
  Value cols = b.create<memref::DimOp>(loc, lhses[0], one);

  int reduce_threads = (row_reduction_schedule == DISC_BLOCK_WISE_ROW_REDUCE)
                           ? block_size
                           : kWarpSize;
  assert(block_size % reduce_threads == 0);
  int64_t row_per_block_wave = block_size / reduce_threads;
  int64_t row_per_block = row_per_block_wave * ilp_factor;
  Value row_per_block_val =
      b.create<arith::ConstantIndexOp>(loc, row_per_block);
  Value block_row_base =
      b.create<arith::MulIOp>(loc, block_id, row_per_block_val);
  Value row_index;
  Value tid_in_row;
  if (row_reduction_schedule == DISC_BLOCK_WISE_ROW_REDUCE) {
    row_index = block_row_base;
    tid_in_row = thread_id;
  } else {
    row_index = b.create<arith::AddIOp>(loc, block_row_base, warp_id);
    tid_in_row = lane_id;
  }

  // if (row_index < rows)
  scf::IfOp if_is_row_valid;
  if (reduce_threads == kWarpSize) {
    // We only check the largest row-id processed in this thread.
    auto is_row_valid = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                row_index, rows);
    if_is_row_valid =
        b.create<scf::IfOp>(loc, /*resultTypes*/ ArrayRef<Type>{}, is_row_valid,
                            /*hasElseRegion*/ false);
    if_is_row_valid.getThenRegion().front().clear();
    b.setInsertionPointToStart(&if_is_row_valid.getThenRegion().front());
  }
  {
    // Do not fuse the loops in this block. It relies the loops for ILP of
    // generated code.
    SmallVector<Value> init_values;
    SmallVector<AccumulatorFactory> accumFactories;
    for (auto reduce_op : reduce_ops) {
      init_values.push_back(
          b.create<memref::LoadOp>(loc, *(reduce_op.getInitValues().begin())));
    }
    for (int64_t i = 0; i < ops.size(); i++) {
      accumFactories.push_back(
          getFactory(b, ops[i]->getLoc(), reduce_ops[i].getBody()));
    }

    // 1. Emit in-thread local reduction.
    Value col_index = nullptr;
    Value threads_per_row =
        b.create<arith::ConstantIndexOp>(loc, reduce_threads);
    scf::ForOp for_local_reduce =
        createLoopAndSetInsPt(b, loc, col_index,
                              /*lb*/ tid_in_row, /*ub*/ cols,
                              /*step*/ threads_per_row, init_values);
    {
      SmallVector<Value> local_reduces;
      SmallVector<Value, 4> multidim({row_index, col_index});
      ValueRange input_index(multidim);
      SmallVector<Value> inputs;
      for (int64_t i = 0; i < ops.size(); i++) {
        auto op = ops[i];
        auto lhs = lhses[i];
        auto input = createLoadOrUseCachedValue(loc, &b, op, lhs, input_index,
                                                b.saveInsertionPoint());
        inputs.push_back(input);
      }
      for (int64_t i = 0; i < ops.size(); i++) {
        auto local_reduce = accumFactories[i](
            *(for_local_reduce.getRegionIterArgs().begin() + i), inputs[i]);
        local_reduces.push_back(local_reduce);
      }
      b.create<scf::YieldOp>(loc, local_reduces);
    }
    b.setInsertionPointAfter(for_local_reduce);

    // 2. Emit per-warp reduction.
    SmallVector<Value> warp_reduces;
    for (int64_t i = 0; i < ops.size(); i++) {
      warp_reduces.push_back(for_local_reduce.getResult(i));
    }
    if (failed(emitWarpReduce(b, loc, ops, accumFactories, warp_reduces,
                              kWarpSize))) {
      return failure();
    }

    // 3. Store warp-reduce result.
    // If it is one-block-one-row reduce, the result will be stored in shm for
    // the second round reduce. Otherwise will be written to output.
    SmallVector<Value> warp_reduce_result_shms;
    int64_t warps_per_block = block_size / kWarpSize;
    for (auto op : ops) {
      auto warp_reduce_result_shm =
          createSharedMemoryForOp(b, loc, op, warps_per_block);
      if (warp_reduce_result_shm == nullptr) {
        return failure();
      }
      warp_reduce_result_shms.push_back(warp_reduce_result_shm);
    }
    Value lane_id_is_zero =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane_id, zero);
    scf::IfOp if_lane_id_is_zero = b.create<scf::IfOp>(
        loc, /*resultTypes*/ ArrayRef<Type>{}, lane_id_is_zero,
        /*hasElseRegion*/ false);
    // if (lane_id == 0)
    if_lane_id_is_zero.getThenRegion().front().clear();
    b.setInsertionPointToEnd(&if_lane_id_is_zero.getThenRegion().front());
    {
      if (reduce_threads == kWarpSize) {
        // Write back result to output, including stitch shm and final output.
        // Do not fuse the loops in this block. It relies the loops for ILP of
        // generated code.
        for (int64_t i = 0; i < ops.size(); i++) {
          if (result_buffer_shms[i] != nullptr && !external_output_only[i]) {
            // The elements store in result shm buffer are in index order. (Note
            // it is linear index.)
            b.create<memref::StoreOp>(loc, warp_reduces[i],
                                      result_buffer_shms[i], warp_id);
          }
        }
        for (int64_t i = 0; i < ops.size(); i++) {
          if (is_output[i]) {
            auto res_memref = ops[i]->getOperand(2);
            b.create<memref::StoreOp>(loc, warp_reduces[i], res_memref,
                                      row_index);
          }
        }
      } else {
        for (int64_t i = 0; i < ops.size(); i++) {
          b.create<memref::StoreOp>(loc, warp_reduces[i],
                                    warp_reduce_result_shms[i], warp_id);
        }
      }
      b.create<scf::YieldOp>(loc, ValueRange({}));
    }
    b.setInsertionPointAfter(if_lane_id_is_zero);

    // 4. Second round reduce for one-block-one-row setting.
    if (reduce_threads != kWarpSize) {
      // __syncthreads()
      b.create<gpu::BarrierOp>(loc);
      auto is_first_warp = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, thread_id, warp_size);
      scf::IfOp if_is_first_warp = b.create<scf::IfOp>(
          loc, /*resultTypes*/ ArrayRef<Type>{}, is_first_warp,
          /*hasElseRegion*/ false);
      if_is_first_warp.getThenRegion().front().clear();
      b.setInsertionPointToStart(&if_is_first_warp.getThenRegion().front());
      {
        // Read results of the first round reduce.
        Value lane_id_inbound = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, lane_id,
            b.create<arith::ConstantIndexOp>(loc, block_size / kWarpSize));
        SmallVector<Type> elemTypes;
        for (auto op : ops) {
          elemTypes.push_back(getLhloOpsElementType(op));
        }
        scf::IfOp if_lane_id_inbound =
            b.create<scf::IfOp>(loc, /*resultTypes*/ elemTypes, lane_id_inbound,
                                /*hasElseRegion*/ true);
        if_lane_id_inbound.getThenRegion().front().clear();
        b.setInsertionPointToStart(&if_lane_id_inbound.getThenRegion().front());
        {
          SmallVector<Value> res;
          for (auto warp_reduce_result_shm : warp_reduce_result_shms) {
            Value reduce_value_r1 =
                b.create<memref::LoadOp>(loc, warp_reduce_result_shm, lane_id);
            res.push_back(reduce_value_r1);
          }
          b.create<scf::YieldOp>(loc, res);
        }
        if_lane_id_inbound.getElseRegion().front().clear();
        b.setInsertionPointToStart(&if_lane_id_inbound.getElseRegion().front());
        { b.create<scf::YieldOp>(loc, init_values); }
        b.setInsertionPointAfter(if_lane_id_inbound);

        // Perform the second round warp reduce.
        SmallVector<Value> reduce_r2s;
        for (int64_t i = 0; i < ops.size(); i++) {
          reduce_r2s.push_back(if_lane_id_inbound.getResult(i));
        }
        int num_warps = block_size / kWarpSize;
        if (failed(emitWarpReduce(b, loc, ops, accumFactories, reduce_r2s,
                                  num_warps))) {
          return failure();
        }

        // Finally, write the output to either stitch shm buffer or global
        // memory.
        scf::IfOp if_lane_id_is_zero_2 = b.create<scf::IfOp>(
            loc, /*resultTypes*/ ArrayRef<Type>{}, lane_id_is_zero,
            /*hasElseRegion*/ false);
        // if (lane_id == 0)
        if_lane_id_is_zero_2.getThenRegion().front().clear();
        b.setInsertionPointToStart(
            &if_lane_id_is_zero_2.getThenRegion().front());
        {
          // Do not fuse the loops in this block. It relies the loops for ILP of
          // generated code.
          for (int64_t i = 0; i < ops.size(); i++) {
            if (result_buffer_shms[i] != nullptr && !external_output_only[i]) {
              b.create<memref::StoreOp>(loc, reduce_r2s[i],
                                        result_buffer_shms[i], zero);
            }
          }
          for (int64_t i = 0; i < ops.size(); i++) {
            if (is_output[i]) {
              auto res_memref = ops[i]->getOperand(2);
              b.create<memref::StoreOp>(loc, reduce_r2s[i], res_memref,
                                        row_index);
            }
          }
          b.create<scf::YieldOp>(loc, ValueRange({}));
        }
        b.setInsertionPointAfter(if_lane_id_is_zero_2);
        b.create<scf::YieldOp>(loc, ValueRange({}));
      }
      b.setInsertionPointAfter(if_is_first_warp);
    }
  }
  if (reduce_threads == kWarpSize) {
    b.create<scf::YieldOp>(loc, ValueRange({}));
    b.setInsertionPointAfter(if_is_row_valid);
  }
  b.setInsertionPointToEnd(block);

  return success();
}

LogicalResult lowerWithScheduleStitchV2(lmhlo::FusionOp& fusion_op,
                                        FusionPattern& fusion_pattern,
                                        ShapeAnalysis* shape_analysis,
                                        int64_t ilp_factor,
                                        LowerConfig& lower_config,
                                        int row_reduction_schedule,
                                        int shmem_limit_bytes) {
  auto root_ops = fusion_pattern.getRootOps();
  auto sub_root_ops = fusion_pattern.getSubRootOps();
  auto result_values = fusion_pattern.getResults();
  DenseSet<Operation*> roots(root_ops.begin(), root_ops.end());
  DenseSet<Value> results(result_values.begin(), result_values.end());
  auto op_list = fusion_pattern.getOpList();
  DenseSet<Operation*> op_set(op_list.begin(), op_list.end());
  DenseSet<Operation*> external_only_roots;
  for (auto value : fusion_pattern.getExternalOnlyResults()) {
    external_only_roots.insert(fusion_pattern.findLastWriter(value));
  }
  auto parent = &(fusion_op.getRegion().front());
  auto dominant_op = fusion_pattern.getDominantOp();
  auto tile_plan = fusion_pattern.getTilePlan();

  for (auto op : sub_root_ops) {
    if (!isRank2RowReduction(op)) {
      LLVM_DEBUG(llvm::dbgs() << "False sub-root: " << *op << "\n");
      return failure();
    }
    auto tile_info = tile_plan.find(op->getOperand(0));
    if (tile_info == tile_plan.end() ||
        tile_info->second.tileSizes.count(0) != 0 ||
        tile_info->second.tileSizes.count(1) != 1) {
      LLVM_DEBUG(llvm::dbgs() << "False tile info for: " << *op << "\n");
      return failure();
    }
  }

  const int thread_per_block = getThreadPerBlock(dominant_op);
  int reduce_threads = (row_reduction_schedule == DISC_BLOCK_WISE_ROW_REDUCE)
                           ? thread_per_block
                           : kWarpSize;
  int64_t row_per_block = thread_per_block / reduce_threads * ilp_factor;

  SmallVector<FusionPattern::SkeletonGroup> skeleton_groups;
  DenseMap<Operation*, Value> shm_cached_ops_and_view;
  DenseMap<Operation*, SmallVector<Operation*>> skeleton_group_ops;
  if (failed(initSkeletonGrpsAndCloneOps(
          fusion_op, fusion_pattern, skeleton_groups, shm_cached_ops_and_view,
          skeleton_group_ops, shape_analysis, lower_config, true, row_per_block,
          shmem_limit_bytes))) {
    LLVM_DEBUG(llvm::dbgs() << "Fail to init skeleton groups or clone.\n");
    return failure();
  }

  Location loc = dominant_op->getLoc();

  // Note that we only support row-reduction to be dominant op currently.
  assert(isRank2RowReduction(dominant_op));
  Value lhs = dominant_op->getOperand(0);
  const MemRefType& lhs_type = lhs.getType().template cast<MemRefType>();

  // Codegen of skeleton ops (i.e. roots/sub-roots). As for sub-roots, emit
  // their direct users.
  OpBuilder b(root_ops.back());
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value shape_h = b.create<memref::DimOp>(loc, lhs, zero);

  Value block_size = b.create<arith::ConstantIndexOp>(loc, thread_per_block);
  Value threads_per_row = b.create<arith::ConstantIndexOp>(loc, reduce_threads);
  Value row_per_block_val =
      b.create<arith::ConstantIndexOp>(loc, row_per_block);
  Value block_number =
      b.create<arith::CeilDivUIOp>(loc, shape_h, row_per_block_val);

  // loop over (block-number, threads-per-block)
  auto global_workgroup = b.create<scf::ParallelOp>(
      loc, SmallVector<Value>({zero}), SmallVector<Value>({block_number}),
      SmallVector<Value>({one}), SmallVector<Value>({}),
      /*bodyBuilderFn=*/nullptr);
  b.setInsertionPointToStart(global_workgroup.getBody());
  auto local_workgroup = b.create<scf::ParallelOp>(
      loc, SmallVector<Value>({zero}), SmallVector<Value>({block_size}),
      SmallVector<Value>({one}), SmallVector<Value>({}),
      /*bodyBuilderFn=*/nullptr);
  local_workgroup.getBody()->clear();
  b.setInsertionPointToStart(local_workgroup.getBody());

  Value block_dim =
      b.create<gpu::BlockDimOp>(loc, b.getIndexType(), gpu::Dimension::x);
  Value block_id =
      b.create<gpu::BlockIdOp>(loc, b.getIndexType(), gpu::Dimension::x);
  Value thread_id =
      b.create<gpu::ThreadIdOp>(loc, b.getIndexType(), gpu::Dimension::x);
  Value warp_size = b.create<arith::ConstantIndexOp>(loc, kWarpSize);
  Value warp_id = b.create<arith::DivUIOp>(loc, thread_id, warp_size);
  Value lane_id = b.create<arith::RemUIOp>(loc, thread_id, warp_size);

  Value block_row_base =
      b.create<arith::MulIOp>(loc, block_id, row_per_block_val);
  Value row_index;
  Value row_in_bound;
  Value tid_in_row;
  if (row_reduction_schedule == DISC_BLOCK_WISE_ROW_REDUCE) {
    row_index = block_row_base;
    tid_in_row = thread_id;
    // The row will always in-bound for block-wise reduction.
  } else {
    row_index = b.create<arith::AddIOp>(loc, block_row_base, warp_id);
    row_in_bound = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                           row_index, shape_h);
    tid_in_row = lane_id;
  }

  // Shared memory buffer allocation for sub-roots' and some intermediate
  // element-wise result. memref.
  DenseMap<Value, Value> shm_mapping;
  DenseSet<Operation*> shm_cached_ops;
  for (auto op : shm_cached_ops_and_view) {
    shm_cached_ops.insert(op.first);
  }
  for (auto& skeleton_group : skeleton_groups) {
    if (isRank2RowReduction(skeleton_group.skeletons[0])) {
      for (auto skeleton : skeleton_group.skeletons) {
        shm_cached_ops.insert(skeleton);
      }
    }
  }
  for (auto op : shm_cached_ops) {
    auto output = op->getOperand(op->getNumOperands() - 1);
    if (tile_plan.find(output) == tile_plan.end()) {
      LLVM_DEBUG(llvm::dbgs() << "Tile info error for: " << *op << "\n");
      return failure();
    }

    int collapsed_tile_dim = fusion_pattern.getCollapsedTileDim(output);
    int element_per_block = row_per_block * collapsed_tile_dim;
    Value result_shmem = createSharedMemoryForOp(b, loc, op, element_per_block);
    if (result_shmem == nullptr) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Create shared memory failed for: " << *op << "\n");
      return failure();
    }
    shm_mapping[output] = result_shmem;

    if (shm_cached_ops_and_view.find(op) != shm_cached_ops_and_view.end()) {
      auto view = shm_cached_ops_and_view[op];
      LowerConfig::SpecificLoader loader(result_shmem, element_per_block);
      lower_config.setSpecificLoader(
          std::make_pair(fusion_op.getOperation(), view), loader);

      LowerConfig::SpecificStore store(result_shmem, element_per_block);
      lower_config.setSpecificStore(
          std::make_pair(fusion_op.getOperation(), output), store);

      // The assume-alignment on shm memref is to prevent error-delete due to
      // canonicalizer.
      b.create<memref::AssumeAlignmentOp>(result_shmem.getLoc(), result_shmem,
                                          32);
    } else {
      LowerConfig::SpecificLoader loader(result_shmem, element_per_block);
      lower_config.setSpecificLoader(
          std::make_pair(fusion_op.getOperation(), output), loader);
    }
  }

  for (auto& skeleton_group : skeleton_groups) {
    auto skeletons = skeleton_group.skeletons;
    Location loc = skeletons[0]->getLoc();
    b.setInsertionPointToEnd(local_workgroup.getBody());

    SmallVector<Value> result_shmems;
    SmallVector<bool> is_output;
    SmallVector<bool> external_only;
    if (isRank2RowReduction(skeletons[0])) {
      for (auto skeleton : skeletons) {
        Value out_value = cast<lmhlo::LmhloOp>(skeleton).getResultBuffer();
        Value result_shmem = shm_mapping[out_value];
        result_shmems.push_back(result_shmem);
        is_output.push_back(roots.contains(skeleton));
        external_only.push_back(external_only_roots.contains(skeleton));
      }

      if (failed(emitRowReduceThreadBlockV2(
              b, loc, local_workgroup.getBody(), skeletons, result_shmems,
              thread_per_block, row_reduction_schedule, ilp_factor,
              shape_analysis, is_output, external_only))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to emit InBlockRowReduce for: "
                                << *skeletons[0] << "\n");
        return failure();
      }
      // TODO: check whether the xroots in this group are required by other
      // groups. If not, no barrier.
      // TODO: warp-wise reduction does not need block-barrier.
      bool require_barrier = false;
      for (int64_t i = 0; i < skeletons.size(); i++) {
        require_barrier |= !external_only[i];
      }
      require_barrier |= llvm::any_of(
          skeleton_group.root_member_list,
          [&](Operation* op) { return !external_only_roots.contains(op); });
      require_barrier |=
          llvm::any_of(skeleton_group_ops[skeletons[0]], [&](Operation* op) {
            return shm_cached_ops_and_view.find(op) !=
                   shm_cached_ops_and_view.end();
          });

      if (require_barrier) {
        b.create<gpu::BarrierOp>(loc);
      }
    } else {
      assert(skeletons.size() == 1);
      auto skeleton = skeletons[0];
      Value out_value = cast<lmhlo::LmhloOp>(skeleton).getResultBuffer();
      auto tile_info = tile_plan.find(out_value);
      if (tile_info == tile_plan.end()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Failed to find tile info: " << *skeleton << "\n");
        return failure();
      }

      scf::IfOp if_row_in_bound;
      if (row_reduction_schedule == DISC_WARP_WISE_ROW_REDUCE) {
        if_row_in_bound =
            b.create<scf::IfOp>(loc, /*resultTypes*/ ArrayRef<Type>{},
                                row_in_bound, /*hasElseRegion*/ false);
        if_row_in_bound.getThenRegion().front().clear();
        b.setInsertionPointToStart(&if_row_in_bound.getThenRegion().front());
      }
      {
        SmallVector<Value> outShapeValues = getShapeValues(&b, out_value);
        // Deal with the case that tiled dims are not the same between result
        // and sub-roots' input.
        Value tiled_linear = nullptr;
        for (auto en : llvm::enumerate(outShapeValues)) {
          if (tile_info->second.tileSizes.count(en.index()) > 0) {
            tiled_linear =
                (tiled_linear == nullptr)
                    ? en.value()
                    : b.create<arith::MulIOp>(loc, tiled_linear, en.value());
          }
        }
        if (tiled_linear == nullptr) {
          tiled_linear = b.create<arith::ConstantIndexOp>(loc, 1);
        }
        Value elemwise_index = nullptr;
        scf::ForOp for_elemwise =
            createLoopAndSetInsPt(b, loc, elemwise_index,
                                  /*lb*/ tid_in_row, /*ub*/ tiled_linear,
                                  /*step*/ threads_per_row, {});
        Value index = b.create<arith::AddIOp>(
            loc, b.create<arith::MulIOp>(loc, row_index, tiled_linear),
            elemwise_index);
        if (failed(lowerHelper(b, loc, skeleton, index, shape_analysis, 1,
                               &lower_config))) {
          LLVM_DEBUG(llvm::dbgs() << "Failed to lower: " << *skeleton << "\n");
          return failure();
        }

        b.setInsertionPointAfter(for_elemwise);
      }
      if (row_reduction_schedule == DISC_WARP_WISE_ROW_REDUCE) {
        b.create<scf::YieldOp>(loc, ValueRange({}));
        b.setInsertionPointAfter(if_row_in_bound);
      }
    }
  }

  b.setInsertionPointToEnd(local_workgroup.getBody());
  b.create<scf::YieldOp>(loc, ValueRange({}));
  b.setInsertionPointAfter(global_workgroup);

  for (auto& skeleton_group : skeleton_groups) {
    for (auto skeleton : skeleton_group.skeletons) {
      skeleton->erase();
    }
  }

  // remove the root_op if it has no other users except the memref
  cleanUnusedLhloOps(parent);

  return success();
}

// Print the params of a fusion with tensor shape at runtime. For debugging.
static void createPrintFusionParams(lmhlo::FusionOp fusion,
                                    FusionPattern pattern) {
  auto loc = fusion.getLoc();
  auto fusion_name = getFusionName(fusion).str();
  OpBuilder b(fusion);
  auto operands = pattern.getOperands();
  auto results = pattern.getResults();
  for (auto operand_enum : llvm::enumerate(operands)) {
    auto index = operand_enum.index();
    auto operand = operand_enum.value();
    int64_t rank = operand.getType().cast<MemRefType>().getRank();
    std::string formStr =
        fusion_name + " operand " + std::to_string(index) + " dims: ";
    SmallVector<Value, 4> buffer_args;
    for (int64_t i = 0; i < rank; i++) {
      auto dim_size = b.create<memref::DimOp>(loc, operand, i);
      buffer_args.push_back(dim_size);
      formStr += "%d ";
    }
    formStr += "\n";
    auto lhloOp = b.create<lmhlo_disc::PrintfOp>(loc, llvm::None, buffer_args);
    lhloOp->setAttr("format", b.getStringAttr(formStr));
  }
  for (auto result_enum : llvm::enumerate(results)) {
    auto index = result_enum.index();
    auto result = result_enum.value();
    int64_t rank = result.getType().cast<MemRefType>().getRank();
    std::string formStr =
        fusion_name + " result " + std::to_string(index) + " dims: ";
    SmallVector<Value, 4> buffer_args;
    for (int64_t i = 0; i < rank; i++) {
      auto dim_size = b.create<memref::DimOp>(loc, result, i);
      buffer_args.push_back(dim_size);
      formStr += "%d ";
    }
    formStr += "\n";
    auto lhloOp = b.create<lmhlo_disc::PrintfOp>(loc, llvm::None, buffer_args);
    lhloOp->setAttr("format", b.getStringAttr(formStr));
  }
}

LogicalResult HandleGpuFusionOp(OpBuilder& b, Operation* fusion,
                                ShapeAnalysis* shape_analysis,
                                LowerConfig& lower_config, int core_count,
                                int cc_major, int cc_minor) {
  auto fusion_op = cast<lmhlo::FusionOp>(fusion);
  assert(fusion_op);
  FusionPattern fusion_pattern(fusion_op, shape_analysis);
  if (auto analysis_deprecated =
          dynamic_cast<ShapeAnalysisDeprecated*>(shape_analysis)) {
    // Update in-fusion shape equality information.
    DenseSet<Value> values_in_fusion;
    auto operands = fusion_pattern.getOperands();
    values_in_fusion.insert(operands.begin(), operands.end());
    auto internal_results = fusion_pattern.getInternalResults();
    values_in_fusion.insert(internal_results.begin(), internal_results.end());
    auto results = fusion_pattern.getResults();
    values_in_fusion.insert(results.begin(), results.end());
    analysis_deprecated->buildEqualShapesInFusion(fusion, values_in_fusion);
  }

  // Debug tool to print fusion kernel parameters in the following format:
  //   func_name operand m dims: x, y
  //   func_name result n dims: p, q
  bool print_params_enabled = false;
  (void)tensorflow::ReadBoolFromEnvVar("DISC_DEBUG_PRINT_FUSION_PARAMS",
                                       print_params_enabled,
                                       &print_params_enabled);
  if (print_params_enabled) {
    createPrintFusionParams(fusion_op, fusion_pattern);
  }

  auto root_ops = fusion_pattern.getRootOps();
  auto fused_block = &(fusion_op.getRegion().front());
  SmallVector<Operation*, 4> op_list;
  for (Operation& op : *fused_block) op_list.push_back(&op);

  // No need to do codegen, return directly.
  if (root_ops.empty()) {
    cleanUnusedLhloOps(fused_block);
    return success();
  }

  // Make a loop to write the buffer into init value for each
  // ColReduction root. This will be further lowered to a init_kernel
  maybeEmitInitLoops(b, root_ops);

  // 1, If any reduce op among the 'root_ops', follow the schedule of it;
  //    or else, follow the schedule of kLoop.
  // 2, If there are a mixer of column reductions and row reductions,
  //    follow the schedule of the row reduction, and implement all the
  //    column reduction with the 'pure atomic' way, which has no
  //    requirement on the schedule.
  // TODO(disc): the support of row reduction and 'pure atomic' reduction
  auto fusion_type = fusion_pattern.getFusionType();
  auto dominant_op = fusion_pattern.getDominantOp();
  switch (fusion_type) {
    case FusionType::kRowReduction: {
      const int row_reduction_schedule =
          getRowReductionScheduleHint(dominant_op);
      const int vector_size = getVectorizeOrTileHint(dominant_op);
      LogicalResult r = success();
      if (row_reduction_schedule == DISC_WARP_WISE_ROW_REDUCE) {
        r = lowerWithScheduleRowReduction<DISC_WARP_WISE_ROW_REDUCE>(
            root_ops, dominant_op, fused_block, shape_analysis, vector_size);
      } else {
        r = lowerWithScheduleRowReduction<DISC_BLOCK_WISE_ROW_REDUCE>(
            root_ops, dominant_op, fused_block, shape_analysis, vector_size);
      }
      if (failed(r)) {
        return dominant_op->emitError()
               << "failed to lower row-reduction loops";
      }
    } break;
    case FusionType::kColReduction: {
      bool use_new = false;
      int loop = 8;
      static const char* envloop = getenv("NEW_COL");
      if (envloop != nullptr) {
        loop = envloop[0] - '0';
        use_new = true;
      }
      const int col_reduction_schedule =
          getColReductionScheduleHint(dominant_op);
      auto kname = getFusionFullName(fusion_op);
      llvm::errs() << "kColReduction <" << kname << ">, use_new: " << use_new
                   << " schedule_hint: " << col_reduction_schedule << "\n";
      LogicalResult r = success();
      if (col_reduction_schedule == DISC_TILE_W8_H32) {
        if (use_new) {
          r = lowerWithScheduleColReductionForRocm<16, 32>(
              root_ops, dominant_op, fused_block, loop, core_count);
        } else {
          r = lowerWithScheduleColReductionBlockTileSchedule<8, 32>(
              root_ops, dominant_op, fused_block);
        }
      } else if (col_reduction_schedule == DISC_TILE_W8_H16) {
        if (use_new) {
          r = lowerWithScheduleColReductionForRocm<16, 32>(
              root_ops, dominant_op, fused_block, loop, core_count);
        } else {
          r = lowerWithScheduleColReductionBlockTileSchedule<8, 16>(
              root_ops, dominant_op, fused_block);
        }
        // } else if (col_reduction_schedule == DISC_TILE_LOOP_W64_H8) {
        //   r = lowerWithScheduleColReductionForRocm<64, 8>(
        //       root_ops, dominant_op, fused_block, loop, core_count);
        // } else if (col_reduction_schedule == DISC_TILE_LOOP_W16_H32) {
        //   r = lowerWithScheduleColReductionForRocm<16, 32>(
        //       root_ops, dominant_op, fused_block, loop, core_count);
        // } else if (col_reduction_schedule == DISC_TILE_LOOP_W8_H8) {
        //   r = lowerWithScheduleColReductionForRocm<8, 8>(
        //       root_ops, dominant_op, fused_block, loop, core_count);
      } else {
        r = lowerWithScheduleColReduction<512, 32>(root_ops, dominant_op,
                                                   fused_block);
      }
      if (failed(r)) {
        return dominant_op->emitError()
               << "failed to lower col-reduction loops";
      }

    } break;

    case FusionType::kLoop: {
      const int vector_size = getVectorizeOrTileHint(dominant_op);
      if (isMemIntensiveOptExperimentalEnabled()) {
        if (failed(lowerWithScheduleLoopV2(root_ops, dominant_op, fused_block,
                                           shape_analysis, vector_size))) {
          return dominant_op->emitError() << "failed to lower to loops";
        }
      } else {
        if (failed(lowerWithScheduleLoop(root_ops, dominant_op, fused_block,
                                         /*non_fusion*/ false,
                                         /*parallel_loop*/ true, shape_analysis,
                                         vector_size))) {
          return dominant_op->emitError() << "failed to lower to loops";
        }
      }
    } break;

    case FusionType::kStitch: {
      const int tile_size = getVectorizeOrTileHint(dominant_op);
      const int row_reduction_schedule =
          getRowReductionScheduleHint(dominant_op);
      if (isMemIntensiveOptExperimentalEnabled()) {
        const int shmem_limit_bytes =
            isMemIntensiveOptExperimentalEnabled()
                ? getShmemSizeBytesNotAffectOccupancy(cc_major, cc_minor)
                : -1;
        if (failed(lowerWithScheduleStitchV2(
                fusion_op, fusion_pattern, shape_analysis, tile_size,
                lower_config, row_reduction_schedule, shmem_limit_bytes))) {
          return fusion->emitError() << "failed to lower kStitch fusion V2.";
        }
      } else {
        if (failed(lowerWithScheduleStitch(
                fusion_op, fusion_pattern, shape_analysis, tile_size,
                lower_config, row_reduction_schedule))) {
          return fusion->emitError() << "failed to lower kStitch fusion.";
        }
      }
    } break;

    default: {
      return dominant_op != nullptr
                 ? dominant_op->emitError() << "Unknown fusion type"
                 : failure();
    }
  }
  return success();
}

// For concat op having many operands, we use following schedule:
//
// parallel.for %idx in range(num_operands_of_concat) {
//   if (%idx == 0) {
//     copy operand #0 to output buffer
//   } else if (idx == 1) {
//     copy operand #1 to output buffer
//   } ...
//   ... ...
//   } else {
//     copy the last operand to output buffer
//   }
// }
//
// To reduce the average level of nested if statement, we further
// reorder the structure of the if-else statement to form a binary
// tree. Basic idea is:
//
// // Emits switch statement for range [from, to)
// def emitSwitch(..., int idx, int from, int to) {
//   if (to - from < 1) return;
//   int mid = (from + to) / 2;
//   if (idx == mid) {
//     copy operand #mid to output
//   } else {
//     if (idx < mid) {
//       emitSwitch(..., idx, from, mid);
//     } else {
//       emitSwitch(..., idx, mid+1, to);
//     }
//   }
// }
LogicalResult emitSwitchOperandIdx(OpBuilder& b, Location loc,
                                   lmhlo::ConcatenateOp op,
                                   SmallVector<Value>& concatOffsets,
                                   Value operandIdx, int from, int to) {
  if (to - from < 1) return success();

  int median = (from + to) / 2;
  Value medianValue = b.create<arith::ConstantIndexOp>(loc, median);
  Value predEQ = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                         operandIdx, medianValue);
  auto ifOp = b.create<scf::IfOp>(loc, llvm::None, predEQ, true);
  Block* thenBlock = &ifOp.getThenRegion().getBlocks().front();
  Block* elseBlock = &ifOp.getElseRegion().getBlocks().front();

  // if operandIdx == median
  {
    b.setInsertionPoint(thenBlock, thenBlock->begin());
    Value operand = op->getOperand(median);
    Value result = cast<lmhlo::LmhloOp>(op.getOperation()).getResultBuffer();
    SmallVector<Value> operandShapeVec = getShapeValues(&b, operand);
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    size_t rank = operandShapeVec.size();
    SmallVector<Value> vars;
    SmallVector<Value> starts(rank, zero);
    SmallVector<Value> limits = operandShapeVec;
    SmallVector<Value> steps(rank, one);
    (void)createParallelAndSetInsPt(b, loc, vars, starts, limits, steps, {});
    SmallVector<Value> resultVars = vars;
    int64_t dimension = op.getDimension();
    resultVars[dimension] =
        b.create<arith::AddIOp>(loc, vars[dimension], concatOffsets[median]);
    Value data = b.create<memref::LoadOp>(loc, operand, vars);
    b.create<memref::StoreOp>(loc, data, result, resultVars);
  }

  b.setInsertionPoint(elseBlock, elseBlock->begin());
  // else if operandIdx != median
  {
    Value predLT = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                           operandIdx, medianValue);
    auto ifOp = b.create<scf::IfOp>(loc, llvm::None, predLT, true);
    Block* thenBlock = &ifOp.getThenRegion().getBlocks().front();
    Block* elseBlock = &ifOp.getElseRegion().getBlocks().front();
    b.setInsertionPoint(thenBlock, thenBlock->begin());
    if (failed(emitSwitchOperandIdx(b, loc, op, concatOffsets, operandIdx, from,
                                    median)))
      return failure();
    b.setInsertionPoint(elseBlock, elseBlock->begin());
    if (failed(emitSwitchOperandIdx(b, loc, op, concatOffsets, operandIdx,
                                    median + 1, to)))
      return failure();
  }
  b.setInsertionPointAfter(ifOp);
  return success();
}

LogicalResult lowerWithScheduleLargeConcatCPU(
    ArrayRef<Operation*> root_ops, Operation* dominant_op,
    Block* parent = nullptr, bool non_fusion = false,
    const ShapeAnalysis* shape_analysis = nullptr) {
  assert(root_ops.size() == 1 && isLargeConcatOp(root_ops[0]));

  auto concat = cast<lmhlo::ConcatenateOp>(root_ops[0]);
  int numOperands = concat->getNumOperands() - 1;
  const auto loc = dominant_op->getLoc();
  OpBuilder b(root_ops.back());
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value numOperandsValue = b.create<arith::ConstantIndexOp>(loc, numOperands);
  SmallVector<Value> vars;
  SmallVector<Value> starts{zero};
  SmallVector<Value> limits{numOperandsValue};
  SmallVector<Value> steps{one};

  (void)createParallelAndSetInsPt(b, loc, vars, starts, limits, steps, {});
  int64_t dimension = concat.getDimension();
  SmallVector<Value> concatOffsets{zero};
  for (int i = 0; i < numOperands; ++i) {
    concatOffsets.push_back(b.create<arith::AddIOp>(
        loc, concatOffsets.back(),
        b.create<memref::DimOp>(loc, concat->getOperand(i), dimension)));
  }
  Value operandIdx = vars[0];
  if (failed(emitSwitchOperandIdx(b, loc, concat, concatOffsets, operandIdx, 0,
                                  numOperands)))
    return failure();

  // remove the root_op if it has no other users except the memref
  if (non_fusion) {
    for (Operation* root_op : root_ops) root_op->erase();
  } else {
    assert(parent != nullptr && "Parent must be provided for fusion lowering");
    cleanUnusedLhloOps(parent);
  }
  return success();
}

// SparseReshape implementation:
//
// Loop over input
// for (i = 0; i < input_indices.shape(0), ++i) {
//   // multidim index to linear index
//   linear_index = ...;
//
//   // linear index to multidim index with new shape
//   multidim_index = ...;
// }
// TODO(lanbo.llb): Support SparseReshapeOp on GPU
LogicalResult lowerWithScheduleSparseReshapeOpCPU(
    ArrayRef<Operation*> root_ops, Operation* dominant_op,
    Block* parent = nullptr, bool non_fusion = false,
    const ShapeAnalysis* shape_analysis = nullptr) {
  if (!(root_ops.size() == 1 &&
        isa<lmhlo_disc::SparseReshapeOp>(root_ops[0]))) {
    return dominant_op->emitError()
           << "root_ops[0] is not a lmhlo_disc::SparseReshapeOp";
  }
  auto sparse_reshape_op = cast<lmhlo_disc::SparseReshapeOp>(root_ops[0]);
  if (!sparse_reshape_op) {
    return dominant_op->emitError()
           << "can not cast root_ops[0] to lmhlo_disc::SparseReshapeOp";
  }

  auto loc = sparse_reshape_op.getLoc();
  OpBuilder b(root_ops.back());

  b.setInsertionPoint(sparse_reshape_op.getOperation());

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value num_values =
      b.create<memref::DimOp>(loc, sparse_reshape_op.getInputIndices(), 0);
  Value origin_rank =
      b.create<memref::DimOp>(loc, sparse_reshape_op.getInputShape(), 0);
  Value new_rank =
      b.create<memref::DimOp>(loc, sparse_reshape_op.getNewShape(), 0);

  Value shape_accum;
  // TODO: How to emit a reverse loop?
  // shape_accum[new_rank];
  // accum = new_shape[new_rank-1];
  // reverse_end = new_rank - 1;
  // for (i = 1, i < new_rank; ++i) {
  //   shape_accum[reverse_end - i] = accum;
  //   accum *= new_shape[reverse_end - i];
  // }
  {
    auto alloc = b.create<memref::AllocOp>(
        loc, MemRefType::get({ShapedType::kDynamic}, b.getIntegerType(64)),
        new_rank);
    shape_accum = alloc.getResult();

    Value reverse_end = b.create<arith::SubIOp>(loc, new_rank, one);
    Value init_value = b.create<memref::LoadOp>(
        loc, sparse_reshape_op.getNewShape(), reverse_end);
    SmallVector<Value, 1> init_values({init_value});
    auto for_op = b.create<scf::ForOp>(loc, /* lowerBound */ one,
                                       /* upperBound */ new_rank,
                                       /* step */ one, init_values);
    for_op.getBody()->clear();
    b.setInsertionPointToStart(for_op.getBody());

    Value i = for_op.getInductionVar();
    Value i_reverse = b.create<arith::SubIOp>(loc, reverse_end, i);
    Value acc_init = *(for_op.getRegionIterArgs().begin());

    b.create<memref::StoreOp>(loc, acc_init, shape_accum, i_reverse);
    Value yield_value = b.create<arith::MulIOp>(
        loc, acc_init,
        b.create<memref::LoadOp>(loc, sparse_reshape_op.getNewShape(),
                                 i_reverse));
    b.create<scf::YieldOp>(loc, ValueRange({yield_value}));

    b.setInsertionPointAfter(sparse_reshape_op.getOperation());
  }

  // for (i = 0; i < input_indices.shape(0), ++i) {
  auto for_op_n = b.create<scf::ForOp>(loc, /* lowerBound */ zero,
                                       /* upperBound */ num_values,
                                       /* step */ one);
  for_op_n.getBody()->clear();
  b.setInsertionPointToStart(for_op_n.getBody());

  // i for the outter loop
  Value n = for_op_n.getInductionVar();

  Value linear_index;
  // linear_index = multidim_index[0];
  // for (i = 1, i < origin_rank; ++i) {
  //   linear_index = (linear_index * input_shape[i]) + multidim_index[i]
  // }
  {
    // Sparse input op's input_indices must be a matrix according to tf's
    // SparseReshapeOp
    SmallVector<Value, 2> load_index({n, zero});
    Value init_value = b.create<memref::LoadOp>(
        loc, sparse_reshape_op.getInputIndices(), load_index);

    auto for_op =
        b.create<scf::ForOp>(loc, /* lowerBound */ one,
                             /* upperBound */ origin_rank,
                             /* step */ one, ValueRange({init_value}));
    for_op.getBody()->clear();
    b.setInsertionPointToStart(for_op.getBody());

    Value i = for_op.getInductionVar();
    Value carry_value = *(for_op.getRegionIterArgs().begin());

    // multidim_index[n][i]
    load_index[1] = i;
    Value index_i = b.create<memref::LoadOp>(
        loc, sparse_reshape_op.getInputIndices(), load_index);
    Value shape_i =
        b.create<memref::LoadOp>(loc, sparse_reshape_op.getInputShape(), i);

    Value yield_value = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, carry_value, shape_i), index_i);
    SmallVector<Value, 1> yield_values({yield_value});
    b.create<scf::YieldOp>(loc, yield_values);

    b.setInsertionPointToEnd(for_op_n.getBody());
    linear_index = for_op.getResult(0);
  }

  // for (i = 0; i < new_rank-1; ++i) {
  //   multidim_index[i] = linear_index / shape_accum[i]
  //   linear_index = linear_index % shape_accum[i]
  // }
  // multidim_index[new_rank-1] = linear_index
  {
    Value reverse_end = b.create<arith::SubIOp>(loc, new_rank, one);
    auto for_op =
        b.create<scf::ForOp>(loc, /* lowerBound */ zero,
                             /* upperBound */ reverse_end,
                             /* step */ one, ValueRange({linear_index}));
    for_op.getBody()->clear();
    b.setInsertionPointToStart(for_op.getBody());

    Value i = for_op.getInductionVar();
    Value divisor = b.create<memref::LoadOp>(loc, shape_accum, i);
    Value carry_value = *(for_op.getRegionIterArgs().begin());
    Value multidim_index = b.create<arith::DivUIOp>(loc, carry_value, divisor);
    Value yield_value = b.create<arith::RemUIOp>(loc, carry_value, divisor);

    b.create<memref::StoreOp>(loc, multidim_index,
                              sparse_reshape_op.getOutputIndices(),
                              ValueRange({n, i}));
    b.create<scf::YieldOp>(loc, yield_value);
    b.setInsertionPointToEnd(for_op_n.getBody());

    b.create<memref::StoreOp>(loc, for_op.getResult(0),
                              sparse_reshape_op.getOutputIndices(),
                              ValueRange({n, reverse_end}));
  }

  // Yield for outter loop
  b.create<scf::YieldOp>(loc, ValueRange({}));
  b.setInsertionPointAfter(sparse_reshape_op.getOperation());

  // read new_shape to make output_shape
  // TODO(lanbo.llb): Handle possible -1 in new_shape
  {
    auto for_op_out_shape = b.create<scf::ForOp>(loc, /* lowerBound */ zero,
                                                 /* upperBound */ new_rank,
                                                 /* step */ one);
    for_op_out_shape.getBody()->clear();
    b.setInsertionPointToStart(for_op_out_shape.getBody());

    Value i = for_op_out_shape.getInductionVar();
    auto shape_val =
        b.create<memref::LoadOp>(loc, sparse_reshape_op.getNewShape(), i);
    b.create<memref::StoreOp>(
        loc, shape_val, sparse_reshape_op.getOutputShape(), ValueRange({i}));
    b.create<scf::YieldOp>(loc, ValueRange({}));
    b.setInsertionPointAfter(for_op_n.getOperation());
  }

  // TODO: Support fusion for sparse reshape
  for (Operation* root_op : root_ops) root_op->erase();
  return success();
}

LogicalResult lowerWithScheduleSparseFillEmptyRowsOpCPU(
    ArrayRef<Operation*> root_ops, Operation* dominant_op,
    Block* parent = nullptr, bool non_fusion = false,
    const ShapeAnalysis* shape_analysis = nullptr) {
  llvm::dbgs() << "come in lowerWithScheduleSparseFillEmptyRowsOpCPU\n";
  if (!(root_ops.size() == 1 &&
        isa<lmhlo_disc::SparseFillEmptyRowsOp>(root_ops[0]))) {
    return dominant_op->emitError()
           << "root_ops[0] is not a lmhlo_disc::SparseFillEmptyRowsOp";
  }
  auto sparse_fill_empty_rows_op =
      dyn_cast<lmhlo_disc::SparseFillEmptyRowsOp>(root_ops[0]);
  if (!sparse_fill_empty_rows_op) {
    return dominant_op->emitError()
           << "can not cast root_ops[0] to lmhlo_disc::SparseFillEmptyRowsOp";
  }

  auto loc = sparse_fill_empty_rows_op.getLoc();
  OpBuilder b(root_ops.back());

  b.setInsertionPoint(sparse_fill_empty_rows_op.getOperation());

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value false_value =
      b.create<arith::ConstantOp>(loc, b.getIntegerAttr(b.getI1Type(), false));
  Value true_value =
      b.create<arith::ConstantOp>(loc, b.getIntegerAttr(b.getI1Type(), true));
  Value zero_int = b.create<arith::ConstantOp>(
      loc, b.getIntegerAttr(b.getIntegerType(64), 0));
  Value one_int = b.create<arith::ConstantOp>(
      loc, b.getIntegerAttr(b.getIntegerType(64), 1));

  // input
  Value indices = sparse_fill_empty_rows_op.getIndices();
  Value values = sparse_fill_empty_rows_op.getValues();
  Value dense_shape = sparse_fill_empty_rows_op.getDenseShape();
  // output
  Value output_indices = sparse_fill_empty_rows_op.getOutputIndices();
  Value output_values = sparse_fill_empty_rows_op.getOutputValues();
  Value empty_row_indicator = sparse_fill_empty_rows_op.getEmptyRowIndicator();
  Value reverse_index_map = sparse_fill_empty_rows_op.getReverseIndexMap();

  Value num_rows = b.create<arith::IndexCastOp>(
      loc, b.getIndexType(), b.create<memref::LoadOp>(loc, dense_shape, zero));

  auto create_init_for_loop = [&](Value init_value, Value target_memref) {
    auto for_op = b.create<scf::ForOp>(loc, /* lowerBound */ zero,
                                       /* upperBound */ num_rows,
                                       /* step */ one);
    for_op.getBody()->clear();
    b.setInsertionPointToStart(for_op.getBody());

    Value i = for_op.getInductionVar();
    b.create<memref::StoreOp>(loc, init_value, target_memref, i);
    b.create<scf::YieldOp>(loc, ValueRange({}));

    b.setInsertionPointAfter(for_op);
  };

  // Init empty_row_indicator to true
  create_init_for_loop(true_value, empty_row_indicator);

  Value row_count_memref;
  {
    auto alloc = b.create<memref::AllocOp>(
        loc,
        MemRefType::get({ShapedType::kDynamic}, b.getIntegerType(64),
                        MemRefLayoutAttrInterface(),
                        StringAttr::get(sparse_fill_empty_rows_op->getContext(),
                                        placement_utils::kCpu)),
        num_rows);
    row_count_memref = alloc.getResult();

    // Init all to 0
    create_init_for_loop(zero_int, row_count_memref);
  }

  Value N_full;
  // num_non_empty_row = 0
  // for (i = 0; i < indices.shape(0); ++i) {
  //   row_idx = indices[i, 0]
  //   if (empty_row_indicator[row_idx] == true) {
  //     empty_row_indicator[row_idx] = false
  //     num_non_empty_row++
  //   }
  //   offset = row_idx + 1 - num_non_empty_row
  //   reverse_index_map[i] = i + offset
  // }
  // num_empty_row = num_rows - num_non_empty_row
  // N_full = num_indices + num_empty_row
  {
    Value num_indices = b.create<memref::DimOp>(loc, indices, 0);
    auto for_op = b.create<scf::ForOp>(loc, /* lowerBound */ zero,
                                       /* upperBound */ num_indices,
                                       /* step */ one,
                                       /* iterArgs */ ValueRange{zero_int});
    for_op.getBody()->clear();
    b.setInsertionPointToStart(for_op.getBody());

    Value i = for_op.getInductionVar();
    SmallVector<Value, 2> index{i, zero};
    Value row_idx_int = b.create<memref::LoadOp>(loc, indices, index);
    Value row_idx =
        b.create<arith::IndexCastOp>(loc, b.getIndexType(), row_idx_int);
    // ********************start if-else**********************
    Value is_not_empty_row =
        b.create<memref::LoadOp>(loc, empty_row_indicator, row_idx);
    auto if_op = b.create<scf::IfOp>(
        loc, /* resultTypes */ b.getIntegerType(64),
        /* cond */ is_not_empty_row, /*withElseRegion*/ true);
    if_op.getThenRegion().front().clear();
    if_op.getElseRegion().front().clear();

    // if then
    b.setInsertionPointToStart(&if_op.getThenRegion().front());
    b.create<memref::StoreOp>(loc, false_value, empty_row_indicator, row_idx);
    b.create<scf::YieldOp>(loc, one_int);

    // else
    b.setInsertionPointToStart(&if_op.getElseRegion().front());
    b.create<scf::YieldOp>(loc, zero_int);
    b.setInsertionPointToEnd(for_op.getBody());

    Value num_non_empty_row = *(for_op.getRegionIterArgs().begin());
    Value yield_value =
        b.create<arith::AddIOp>(loc, num_non_empty_row, if_op.getResult(0));
    // ********************end if-else************************

    Value i_casted = b.create<arith::IndexCastOp>(loc, b.getIntegerType(64), i);
    Value i_offset = b.create<arith::SubIOp>(
        loc, b.create<arith::AddIOp>(loc, row_idx_int, one_int), yield_value);
    b.create<memref::StoreOp>(loc,
                              b.create<arith::AddIOp>(loc, i_casted, i_offset),
                              reverse_index_map, i);
    // row_count[i] += 1
    b.create<memref::StoreOp>(
        loc,
        b.create<arith::AddIOp>(
            loc, b.create<memref::LoadOp>(loc, row_count_memref, row_idx),
            one_int),
        row_count_memref, row_idx);
    b.create<scf::YieldOp>(loc, yield_value);
    b.setInsertionPointAfter(for_op);
    // after loop
    Value num_rows = b.create<memref::LoadOp>(loc, dense_shape, zero);
    Value num_empty_row =
        b.create<arith::SubIOp>(loc, num_rows, for_op.getResult(0));
    N_full = b.create<arith::AddIOp>(
        loc, num_indices,
        b.create<arith::IndexCastOp>(loc, b.getIndexType(), num_empty_row));
  }
  // store N_full
  b.create<memref::StoreOp>(
      loc, b.create<arith::IndexCastOp>(loc, b.getIntegerType(64), N_full),
      sparse_fill_empty_rows_op.getOutputElements(), zero);

  // row_start_idx = 0
  // for (i = 0; i < num_rows; ++i) {
  //   if (empty_row_indicator[i] == true) {
  //     output_indices[row_start_idx, 0] = i
  //     output_indices[row_start_idx, 1] = 0
  //     output_values[row_start_idx] = default_value
  //     row_start_idx += 1
  //   } else {
  //     row_start_idx += row_count[i]
  //   }
  // }
  // fill empty rows
  {
    Value num_rows = b.create<arith::IndexCastOp>(
        loc, b.getIndexType(),
        b.create<memref::LoadOp>(loc, dense_shape, zero));
    auto for_op = b.create<scf::ForOp>(
        loc, /* lowerBound */ zero, /* upperBound */ num_rows,
        /* step */ one, /* iterArgs */ ValueRange{zero});
    for_op.getBody()->clear();
    b.setInsertionPointToStart(for_op.getBody());
    Value i = for_op.getInductionVar();
    // iterArgs[0] --> row_start_idx
    Value row_start_idx = *(for_op.getRegionIterArgs().begin());
    Value if_empty_row = b.create<memref::LoadOp>(loc, empty_row_indicator, i);
    auto if_op = b.create<scf::IfOp>(
        loc,
        /* resultTypes */ ArrayRef<Type>({b.getIndexType()}),
        /* cond */ if_empty_row, /*withElseRegion*/ true);

    if_op.getThenRegion().front().clear();
    if_op.getElseRegion().front().clear();
    b.setInsertionPointToStart(&if_op.getThenRegion().front());

    // output_indices
    SmallVector<Value, 2> store_index{row_start_idx, zero};
    b.create<memref::StoreOp>(
        loc, b.create<arith::IndexCastOp>(loc, b.getIntegerType(64), i),
        output_indices, store_index);
    store_index[1] = one;
    b.create<memref::StoreOp>(loc, zero_int, output_indices, store_index);
    // output_values
    auto default_value = b.create<memref::LoadOp>(
        loc, sparse_fill_empty_rows_op.getDefaultValue());
    b.create<memref::StoreOp>(loc, default_value, output_values, row_start_idx);
    b.create<scf::YieldOp>(loc, one);

    b.setInsertionPointToStart(&if_op.getElseRegion().front());
    Value row_count = b.create<arith::IndexCastOp>(
        loc, b.getIndexType(),
        b.create<memref::LoadOp>(loc, row_count_memref, i));
    b.create<scf::YieldOp>(loc, row_count);

    b.setInsertionPointToEnd(for_op.getBody());
    Value yield_value =
        b.create<arith::AddIOp>(loc, row_start_idx, if_op.getResult(0));
    b.create<scf::YieldOp>(loc, yield_value);
    b.setInsertionPointAfter(for_op);
  }

  // for (i = 0; i < indices.shape(0); ++i) {
  //   output_idx = reverse_index_map[i]
  //   output_indices[output_idx, 0] = indices[i, 0]
  //   output_indices[output_idx, 1] = indices[i, 1]
  //   output_values[output_idx] = values[i]
  // }
  // fill non-empty rows
  {
    Value num_indices = b.create<memref::DimOp>(loc, indices, 0);
    auto for_op = b.create<scf::ForOp>(loc, /* lowerBound */ zero,
                                       /* upperBound */ num_indices,
                                       /* step */ one);
    for_op.getBody()->clear();
    b.setInsertionPointToStart(for_op.getBody());
    Value i = for_op.getInductionVar();
    Value output_idx = b.create<arith::IndexCastOp>(
        loc, b.getIndexType(),
        b.create<memref::LoadOp>(loc, reverse_index_map, i));
    SmallVector<Value, 2> store_index{output_idx, zero};
    SmallVector<Value, 2> load_index{i, zero};
    b.create<memref::StoreOp>(
        loc, b.create<memref::LoadOp>(loc, indices, load_index), output_indices,
        store_index);
    store_index[1] = one;
    load_index[1] = one;
    b.create<memref::StoreOp>(
        loc, b.create<memref::LoadOp>(loc, indices, load_index), output_indices,
        store_index);
    b.create<memref::StoreOp>(loc, b.create<memref::LoadOp>(loc, values, i),
                              output_values, output_idx);
    b.create<scf::YieldOp>(loc, ValueRange{});
    b.setInsertionPoint(sparse_fill_empty_rows_op.getOperation());
  }

  // TODO: Support fusion
  for (Operation* root_op : root_ops) root_op->erase();
  return success();
}

LogicalResult lowerWithScheduleSparseSegmentReductionOpCPU(
    ArrayRef<Operation*> root_ops, Operation* dominant_op,
    Block* parent = nullptr, bool non_fusion = false,
    const ShapeAnalysis* shape_analysis = nullptr) {
  if (!(root_ops.size() == 1 &&
        isa<lmhlo_disc::SparseSegmentReductionOp>(root_ops[0]))) {
    return dominant_op->emitError()
           << "root_ops[0] is not a lmhlo_disc::SparseSegmentReductionOp";
  }
  auto sparse_segment_reduction_op =
      dyn_cast<lmhlo_disc::SparseSegmentReductionOp>(root_ops[0]);
  if (!sparse_segment_reduction_op) {
    return dominant_op->emitError() << "can not cast root_ops[0] to "
                                       "lmhlo_disc::SparseSegmentReductionOp";
  }

  auto loc = sparse_segment_reduction_op.getLoc();

  OpBuilder b(root_ops.back());

  auto operation = sparse_segment_reduction_op.getOperation();
  auto context = sparse_segment_reduction_op.getContext();

  // input
  Value data = sparse_segment_reduction_op.getData();
  Value indices = sparse_segment_reduction_op.getIndices();
  Value segment_ids = sparse_segment_reduction_op.getSegmentIds();
  // output
  Value output = sparse_segment_reduction_op.getOutput();

  MemRefType output_type = output.getType().template cast<MemRefType>();
  int64_t output_rank = output_type.getRank();

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);

  Value zero_floating = b.create<arith::ConstantOp>(
      loc, b.getFloatAttr(output_type.getElementType(), 0));
  Value one_floating = b.create<arith::ConstantOp>(
      loc, b.getFloatAttr(output_type.getElementType(), 1));

  Value num_results = b.create<memref::DimOp>(loc, output, 0);

  // memset output to 0
  auto num_elements = emitNumElementsComputation(b, loc, output);
  auto output_shape = getShapeValues(&b, output);
  auto for_op = b.create<scf::ForOp>(loc, /* lowerBound */ zero,
                                     /* upperBound */ num_elements,
                                     /* step */ one);
  for_op.getBody()->clear();
  b.setInsertionPointToStart(for_op.getBody());

  Value i = for_op.getInductionVar();
  auto index = calcMultiDimIndex(&b, loc, i, output_shape);
  b.create<memref::StoreOp>(loc, zero_floating, output, index);
  b.create<scf::YieldOp>(loc, ValueRange({}));
  b.setInsertionPointAfter(for_op);

  if (sparse_segment_reduction_op.getReductionMode() ==
      lmhlo_disc::ReductionModeEnum::Sum) {
    // data/output's dim 1~(rank-1) are the same
    // for (i = 0; i < indices.shape(0), ++i) {
    //   row_idx = indices[i]
    //   segment_idx = segment_ids[i]
    //   for (j = 0; j < data.shape(1); ++j) {
    //     for (k = 0; j < data.shape(2); ++k) {
    //       ...
    //         for (m = 0; j < data.shape(rank-1); ++m) {
    //           output[segment_idx][j]...[m] += data[row_idx][j]...[m]
    //         }
    //       ...
    //     }
    //   }
    // }
    llvm::SmallVector<Value, 2> lower(output_rank, zero), upper,
        step(output_rank, one);
    upper.push_back(b.create<memref::DimOp>(loc, indices, 0));
    for (int i = 1; i < output_rank; i++) {
      upper.push_back(b.create<memref::DimOp>(loc, data, i));
    }
    auto parallel_op =
        b.create<scf::ParallelOp>(loc, /* lowerBound */ lower,
                                  /* upperBound */ upper, /* step */ step);
    b.setInsertionPointToStart(parallel_op.getBody());
    auto index_i = parallel_op.getInductionVars()[0];
    llvm::SmallVector<Value, 4> data_index, output_index;
    data_index.push_back(b.create<arith::IndexCastOp>(
        loc, b.getIndexType(),
        b.create<memref::LoadOp>(loc, indices, index_i)));
    output_index.push_back(b.create<arith::IndexCastOp>(
        loc, b.getIndexType(),
        b.create<memref::LoadOp>(loc, segment_ids, index_i)));
    for (int i = 1; i < output_rank; ++i) {
      auto index = b.create<arith::IndexCastOp>(
          loc, b.getIndexType(), parallel_op.getInductionVars()[i]);
      data_index.push_back(index);
      output_index.push_back(index);
    }
    b.create<memref::StoreOp>(
        loc,
        b.create<arith::AddFOp>(
            loc, b.create<memref::LoadOp>(loc, output, output_index),
            b.create<memref::LoadOp>(loc, data, data_index)),
        output, output_index);
    b.setInsertionPointAfter(parallel_op);
  } else if (sparse_segment_reduction_op.getReductionMode() ==
             lmhlo_disc::ReductionModeEnum::Mean) {
    // segment_count is only needed for mean
    Value segment_count_memref;
    // segment_count[max_segment_ids]
    // for (i = 0; i < segment_ids.shape(0); ++i) {
    //   segment_idx = segment_ids[i]
    //   segment_count[segment_idx]++
    // }
    {
      auto alloc = b.create<memref::AllocOp>(
          loc,
          MemRefType::get({ShapedType::kDynamic}, output_type.getElementType(),
                          MemRefLayoutAttrInterface(),
                          StringAttr::get(context, placement_utils::kCpu)),
          num_results);
      segment_count_memref = alloc.getResult();

      {
        auto for_op =
            b.create<scf::ForOp>(loc, /* lowerBound */ zero,
                                 /* upperBound */ num_results, /* step */ one);
        for_op.getBody()->clear();
        b.setInsertionPointToStart(for_op.getBody());
        Value i = for_op.getInductionVar();
        b.create<memref::StoreOp>(loc, zero_floating, segment_count_memref, i);
        b.create<scf::YieldOp>(loc, ValueRange({}));
        b.setInsertionPointAfter(for_op);
      }
      {
        Value num_segment_ids = b.create<memref::DimOp>(loc, segment_ids, 0);
        auto for_op = b.create<scf::ForOp>(loc, /* lowerBound */ zero,
                                           /* upperBound */ num_segment_ids,
                                           /* step */ one);
        for_op.getBody()->clear();
        b.setInsertionPointToStart(for_op.getBody());
        Value i = for_op.getInductionVar();
        Value segment_idx = b.create<arith::IndexCastOp>(
            loc, b.getIndexType(),
            b.create<memref::LoadOp>(loc, segment_ids, i));
        b.create<memref::StoreOp>(
            loc,
            b.create<arith::AddFOp>(loc,
                                    b.create<memref::LoadOp>(
                                        loc, segment_count_memref, segment_idx),
                                    one_floating),
            segment_count_memref, segment_idx);
        b.create<scf::YieldOp>(loc, ValueRange({}));
        b.setInsertionPointAfter(for_op);
      }
    }

    // for (i = 0; i < indices.shape(0), ++i) {
    //   row_idx = indices[i]
    //   segment_idx = segment_ids[i]
    //   if (segment_count[segment_idx] != 0) {
    //     for (j = 0; j < data.shape(1); ++j) {
    //       for (k = 0; j < data.shape(2); ++k) {
    //         ...
    //           for (m = 0; j < data.shape(rank-1); ++m) {
    //             output[segment_idx][j]...[m] +=
    //               data[row_idx][j]...[m]/segment_count[segment_idx]
    //           }
    //         ...
    //       }
    //     }
    //   }
    // }
    llvm::SmallVector<Value, 2> lower(output_rank, zero), upper,
        step(output_rank, one);
    upper.push_back(b.create<memref::DimOp>(loc, indices, 0));
    for (int i = 1; i < output_rank; i++) {
      upper.push_back(b.create<memref::DimOp>(loc, data, i));
    }
    auto parallel_op =
        b.create<scf::ParallelOp>(loc, /* lowerBound */ lower,
                                  /* upperBound */ upper, /* step */ step);
    b.setInsertionPointToStart(parallel_op.getBody());

    llvm::SmallVector<Value, 4> data_index, output_index;
    auto segment_idx = b.create<memref::LoadOp>(
        loc, segment_ids, parallel_op.getInductionVars()[0]);
    llvm::SmallVector<Value, 1> segment_count_index(
        1, b.create<arith::IndexCastOp>(loc, b.getIndexType(), segment_idx));
    Value segment_count = b.create<memref::LoadOp>(loc, segment_count_memref,
                                                   segment_count_index);
    Value pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE,
                                         segment_count, zero_floating);
    auto if_op = b.create<scf::IfOp>(loc, pred, /*hasElseRegion*/ false);
    if_op.getThenRegion().front().clear();
    b.setInsertionPointToStart(&if_op.getThenRegion().front());

    data_index.push_back(b.create<arith::IndexCastOp>(
        loc, b.getIndexType(),
        b.create<memref::LoadOp>(loc, indices,
                                 parallel_op.getInductionVars()[0])));
    output_index.push_back(
        b.create<arith::IndexCastOp>(loc, b.getIndexType(), segment_idx));
    for (int i = 1; i < output_rank; ++i) {
      auto index = b.create<arith::IndexCastOp>(
          loc, b.getIndexType(), parallel_op.getInductionVars()[i]);
      data_index.push_back(index);
      output_index.push_back(index);
    }
    auto data_div = b.create<arith::DivFOp>(
        loc, b.create<memref::LoadOp>(loc, data, data_index), segment_count);
    b.create<memref::StoreOp>(
        loc,
        b.create<arith::AddFOp>(
            loc, b.create<memref::LoadOp>(loc, output, output_index), data_div),
        output, output_index);

    b.create<scf::YieldOp>(loc, ValueRange({}));
    b.setInsertionPointAfter(parallel_op);
  } else {
    return dominant_op->emitError() << "Reduction mode is not supported for "
                                       "lmhlo_disc.sparse_segment_reduction";
  }
  b.setInsertionPoint(operation);

  // TODO: Support fusion
  for (Operation* root_op : root_ops) root_op->erase();
  return success();
}

LogicalResult lowerWithScheduleSparseSegmentReductionWithEmptyRowsOpCPU(
    ArrayRef<Operation*> root_ops, Operation* dominant_op,
    Block* parent = nullptr, bool non_fusion = false,
    const ShapeAnalysis* shape_analysis = nullptr) {
  if (non_fusion) {
    if (!(root_ops.size() == 1 &&
          isa<lmhlo_disc::SparseSegmentReductionWithEmptyRowsOp>(
              root_ops[0]))) {
      return dominant_op->emitError()
             << "root_ops[0] is not a "
                "lmhlo_disc::SparseSegmentReductionWithEmptyRowsOp";
    }
  }
  auto sparse_segment_reduction_op =
      dyn_cast<lmhlo_disc::SparseSegmentReductionWithEmptyRowsOp>(
          non_fusion ? root_ops[0] : dominant_op);
  if (!sparse_segment_reduction_op) {
    return dominant_op->emitError()
           << "can not cast root_ops[0] or dominant_op to "
              "lmhlo_disc::SparseSegmentReductionWithEmptyRowsOp";
  }

  auto loc = sparse_segment_reduction_op.getLoc();
  OpBuilder b(root_ops.back());

  auto operation = sparse_segment_reduction_op.getOperation();
  auto context = sparse_segment_reduction_op.getContext();

  // input
  Value data = sparse_segment_reduction_op.getData();
  Value indices = sparse_segment_reduction_op.getIndices();
  Value segment_ids = sparse_segment_reduction_op.getUnfilledSegmentIds();
  Value dense_shape = sparse_segment_reduction_op.getDenseShape();
  // output
  Value output = sparse_segment_reduction_op.getOutput();
  Value empty_row_indicator =
      sparse_segment_reduction_op.getEmptyRowIndicator();

  MemRefType output_type = output.getType().template cast<MemRefType>();
  int64_t output_rank = output_type.getRank();

  b.setInsertionPoint(operation);
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);

  Value zero_floating = b.create<arith::ConstantOp>(
      loc, b.getFloatAttr(output_type.getElementType(), 0));
  Value one_floating = b.create<arith::ConstantOp>(
      loc, b.getFloatAttr(output_type.getElementType(), 1));
  Value true_value =
      b.create<arith::ConstantOp>(loc, b.getIntegerAttr(b.getI1Type(), true));
  Value false_value =
      b.create<arith::ConstantOp>(loc, b.getIntegerAttr(b.getI1Type(), false));

  Value data_dim = b.create<memref::DimOp>(loc, data, 1);
  Value indice_size = b.create<memref::DimOp>(loc, indices, 0);
  Value dense_rows = b.create<memref::DimOp>(loc, output, 0);

  // for kSparseReduction fusion, we postpone memset final output buffer
  // to OutputInlineFusion pass
  if (non_fusion) {
    // memset output to 0
    {
      auto num_elements = emitNumElementsComputation(b, loc, output);
      auto output_shape = getShapeValues(&b, output);
      auto for_op = b.create<scf::ForOp>(loc, /* lowerBound */ zero,
                                         /* upperBound */ num_elements,
                                         /* step */ one);
      for_op.getBody()->clear();
      b.setInsertionPointToStart(for_op.getBody());

      Value i = for_op.getInductionVar();
      auto index = calcMultiDimIndex(&b, loc, i, output_shape);
      b.create<memref::StoreOp>(loc, zero_floating, output, index);
      b.create<scf::YieldOp>(loc, ValueRange({}));
      b.setInsertionPointAfter(for_op);
    }
  }

  auto create_init_for_loop = [&](Value init_value, Value target_memref) {
    auto for_op = b.create<scf::ForOp>(loc, /* lowerBound */ zero,
                                       /* upperBound */ dense_rows,
                                       /* step */ one);
    for_op.getBody()->clear();
    b.setInsertionPointToStart(for_op.getBody());

    Value i = for_op.getInductionVar();
    b.create<memref::StoreOp>(loc, init_value, target_memref, i);
    b.create<scf::YieldOp>(loc, ValueRange({}));

    b.setInsertionPointAfter(for_op);
  };

  Value row_value_count;
  // int* row_value_count = new int[dense_rows];
  // memset(row_value_count, 0, dense_rows * sizeof(int));
  auto alloc = b.create<memref::AllocOp>(
      loc,
      MemRefType::get({ShapedType::kDynamic}, output_type.getElementType(),
                      MemRefLayoutAttrInterface(),
                      StringAttr::get(context, placement_utils::kCpu)),
      dense_rows);
  row_value_count = alloc.getResult();
  create_init_for_loop(zero_floating, row_value_count);

  // scf.for to calc row_count
  {
    // indice_size = segment_ids_size
    auto for_op = b.create<scf::ForOp>(loc, /* lowerBound */ zero,
                                       /* upperBound */ indice_size,
                                       /* step */ one);
    for_op.getBody()->clear();
    b.setInsertionPointToStart(for_op.getBody());
    Value i = for_op.getInductionVar();
    llvm::SmallVector<Value, 4> segment_ids_index;
    segment_ids_index.push_back(i);
    segment_ids_index.push_back(zero);
    Value segment_idx = b.create<arith::IndexCastOp>(
        loc, b.getIndexType(),
        b.create<memref::LoadOp>(loc, segment_ids, segment_ids_index));

    b.create<memref::StoreOp>(
        loc,
        b.create<arith::AddFOp>(
            loc, b.create<memref::LoadOp>(loc, row_value_count, segment_idx),
            one_floating),
        row_value_count, segment_idx);

    b.create<scf::YieldOp>(loc, ValueRange({}));
    b.setInsertionPointAfter(for_op);
  }

  // scf.parallel
  {
    llvm::SmallVector<Value, 2> lower(2, zero), upper(2, indice_size),
        step(2, one);
    upper[1] = data_dim;
    auto parallel_op =
        b.create<scf::ParallelOp>(loc, /* lowerBound */ lower,
                                  /* upperBound */ upper, /* step */ step);
    b.setInsertionPointToStart(parallel_op.getBody());

    llvm::SmallVector<Value, 4> indices_index, segment_ids_index;
    indices_index.push_back(parallel_op.getInductionVars()[0]);
    Value id = b.create<arith::IndexCastOp>(
        loc, b.getIndexType(),
        b.create<memref::LoadOp>(loc, indices, indices_index));
    // row = segment_ids[i, 0]
    segment_ids_index.push_back(parallel_op.getInductionVars()[0]);
    segment_ids_index.push_back(zero);
    Value row = b.create<arith::IndexCastOp>(
        loc, b.getIndexType(),
        b.create<memref::LoadOp>(loc, segment_ids, segment_ids_index));
    llvm::SmallVector<Value, 2> row_index(1, row);
    // if row_value_count[row] > 0
    Value pred = b.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE,
        b.create<memref::LoadOp>(loc, row_value_count, row_index),
        zero_floating);
    auto if_op = b.create<scf::IfOp>(loc, pred, /*hasElseRegion*/ true);
    if_op.getThenRegion().front().clear();
    if_op.getElseRegion().front().clear();
    // input/output index
    Value i = parallel_op.getInductionVars()[1];
    llvm::SmallVector<Value, 2> output_index, input_index;
    output_index.push_back(row);
    output_index.push_back(i);
    input_index.push_back(id);
    input_index.push_back(i);

    b.setInsertionPointToStart(&if_op.getThenRegion().front());
    {
      // add
      b.create<memref::StoreOp>(loc, false_value, empty_row_indicator,
                                row_index);
      Value accumulation = b.create<memref::LoadOp>(loc, output, output_index);
      if (sparse_segment_reduction_op.getReductionMode() ==
          lmhlo_disc::ReductionModeEnum::Mean) {
        auto data_div = b.create<arith::DivFOp>(
            loc, b.create<memref::LoadOp>(loc, data, input_index),
            b.create<memref::LoadOp>(loc, row_value_count, row_index));
        b.create<memref::StoreOp>(
            loc, b.create<arith::AddFOp>(loc, accumulation, data_div), output,
            output_index);
      } else if (sparse_segment_reduction_op.getReductionMode() ==
                 lmhlo_disc::ReductionModeEnum::Sum) {
        b.create<memref::StoreOp>(
            loc,
            b.create<arith::AddFOp>(
                loc, accumulation,
                b.create<memref::LoadOp>(loc, data, input_index)),
            output, output_index);
      }
      b.create<scf::YieldOp>(loc, ValueRange({}));
    }
    b.setInsertionPointToStart(&if_op.getElseRegion().front());
    // for empty rows, just memcpy
    {
      b.create<memref::StoreOp>(loc, true_value, empty_row_indicator,
                                row_index);
      b.create<memref::StoreOp>(
          loc, b.create<memref::LoadOp>(loc, data, input_index), output,
          output_index);
      b.create<scf::YieldOp>(loc, ValueRange({}));
    }
    b.setInsertionPointAfter(parallel_op);
  }

  // TODO: support fusion
  if (non_fusion) {
    for (Operation* root_op : root_ops) root_op->erase();
  }
  return success();
}

LogicalResult lowerWithScheduleWhereOpCPU(
    ArrayRef<Operation*> root_ops, Operation* dominant_op,
    Block* parent = nullptr, bool non_fusion = false,
    const ShapeAnalysis* shape_analysis = nullptr) {
  if (!(root_ops.size() == 1 && isa<lmhlo_disc::WhereOp>(root_ops[0]))) {
    return dominant_op->emitError()
           << "root_ops[0] is not a lmhlo_disc::WhereOp";
  }
  auto where = dyn_cast<lmhlo_disc::WhereOp>(root_ops[0]);
  if (!where) {
    return dominant_op->emitError()
           << "can not cast root_ops[0] to lmhlo_disc::WhereOp";
  }

  auto loc = where.getLoc();
  OpBuilder b(root_ops.back());

  Value input = where.getInput();
  Value index = where.getIndex();
  Value num_output_elements = where.getNumOutputElements();

  auto input_type = input.getType().cast<MemRefType>();
  auto input_rank = input_type.getRank();
  auto input_elem_type = input_type.getElementType();
  auto output_elem_type = index.getType().cast<MemRefType>().getElementType();

  Value zero =
      b.create<arith::ConstantOp>(loc, b.getIntegerAttr(b.getIndexType(), 0));
  Value one =
      b.create<arith::ConstantOp>(loc, b.getIntegerAttr(b.getIndexType(), 1));
  Value one_for_acc = b.create<arith::ConstantOp>(
      loc, b.getIntegerAttr(b.getIntegerType(64), 1));
  SmallVector<Value, 1> zero_load_index = {
      b.create<arith::ConstantIndexOp>(loc, 0)};
  // stack temp buffer for num_output_elements to 0
  auto alloc = b.create<memref::AllocaOp>(
      loc, MemRefType::get(
               {}, b.getIntegerType(64), MemRefLayoutAttrInterface(),
               StringAttr::get(where->getContext(), placement_utils::kCpu)));
  auto temp_count = alloc.getResult();
  b.create<memref::StoreOp>(loc,
                            b.create<arith::ConstantOp>(
                                loc, b.getIntegerAttr(b.getIntegerType(64), 0)),
                            temp_count);

  // for (i = 0; i < num_element; ++i)
  llvm::SmallVector<Value, 2> lower, upper, step;
  llvm::SmallVector<Value, 4> multidim_index;
  llvm::SmallVector<Value, 4> multidim_index_casted;
  for (int i = 0; i < input_rank; ++i) {
    lower.push_back(zero);
    upper.push_back(b.create<memref::DimOp>(loc, input, i));
    step.push_back(one);
  }
  auto for_op =
      b.create<scf::ParallelOp>(loc, /* lowerBound */ lower,
                                /* upperBound */ upper, /* step */ step);
  Value output_index_count;
  b.setInsertionPointToStart(for_op.getBody());
  for (int i = 0; i < input_rank; ++i) {
    Value loop_index = for_op.getInductionVars()[i];
    multidim_index.push_back(loop_index);
    multidim_index_casted.push_back(
        b.create<arith::IndexCastOp>(loc, output_elem_type, loop_index));
  }
  Value input_elem = b.create<memref::LoadOp>(loc, input, multidim_index);
  // if (input[i] != 0)
  Value is_zero;
  // NOTE: make sure cmp value is the same type as input
  // otherwise the generated IR will seems ok, however will get segfault
  // in llvm::CmpInst::CmpInst
  if (input_elem_type.isa<IntegerType>()) {
    Value zero_for_cmp =
        b.create<arith::ConstantOp>(loc, b.getIntegerAttr(input_elem_type, 0));
    is_zero = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, input_elem,
                                      zero_for_cmp);
  } else if (input_elem_type.isa<FloatType>()) {
    Value zero_for_cmp =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(input_elem_type, 0));
    is_zero = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE,
                                      input_elem, zero_for_cmp);
  } else {
    return dominant_op->emitError() << "type other than float or int for "
                                       "lmhlo_disc::where is not supported yet";
  }
  auto if_is_zero = b.create<scf::IfOp>(loc, /*resultTypes*/ llvm::None,
                                        /* condition */ is_zero,
                                        /*hasElseRegion*/ false);
  if_is_zero.getThenRegion().front().clear();
  b.setInsertionPointToStart(&if_is_zero.getThenRegion().front());
  output_index_count = b.create<memref::LoadOp>(loc, temp_count);
  Value output_index_casted =
      b.create<arith::IndexCastOp>(loc, b.getIndexType(), output_index_count);
  for (int i = 0; i < input_rank; ++i) {
    llvm::SmallVector<Value, 2> store_index = {
        output_index_casted, b.create<arith::ConstantIndexOp>(loc, i)};
    b.create<memref::StoreOp>(loc, multidim_index_casted[i], index,
                              store_index);
  }
  output_index_count =
      b.create<arith::AddIOp>(loc, output_index_count, one_for_acc);
  b.create<memref::StoreOp>(loc, output_index_count, temp_count);

  b.create<scf::YieldOp>(loc, ValueRange({}));  // if (input[i] != 0)
  b.setInsertionPoint(
      where.getOperation());  // for (i = 0; i < num_element; ++i)

  b.create<memref::StoreOp>(loc, b.create<memref::LoadOp>(loc, temp_count),
                            num_output_elements, zero_load_index);

  if (non_fusion) {
    for (Operation* root_op : root_ops) root_op->erase();
  } else {
    assert(parent != nullptr && "Parent must be provided for fusion lowering");
    cleanUnusedLhloOps(parent);
  }
  return success();
}

// we don't do inbound check for kLoop Schedule
// LoopSplit pass will do this.
//
/*
 * loop.for %idxs... in shape_of(dominant_op) {
 *   %linear_idx = getLinearIdx(%idxs..., %dominant_shape)
 *   %multidim_indices_0..n = getMultidimIndices(%linear_idx);
 *   %operand_0 = load %operand0[]
 *   %operand_1 = load %operand1[]
 *   emit calculation..
 * }
 */
LogicalResult lowerWithScheduleLoopCPU(
    ArrayRef<Operation*> root_ops, Operation* dominant_op,
    Block* parent = nullptr, bool non_fusion = false, bool parallel_loop = true,
    bool multi_dim_loop = false,
    const ShapeAnalysis* shape_analysis = nullptr) {
  llvm::dbgs() << "parallel_loop: " << parallel_loop << "\n";
  Value result = cast<lmhlo::LmhloOp>(dominant_op).getResultBuffer();
  int64_t rank = result.getType().cast<MemRefType>().getRank();
  if (!multi_dim_loop || !rank || !parallel_loop) {
    llvm::dbgs() << "lower with lowerWithScheduleLoop\n";
    return lowerWithScheduleLoop(root_ops, dominant_op, parent, non_fusion,
                                 parallel_loop, shape_analysis);
  }

  if (non_fusion && isLargeConcatOp(dominant_op)) {
    return lowerWithScheduleLargeConcatCPU(root_ops, dominant_op, parent,
                                           non_fusion, shape_analysis);
  }

  if (non_fusion && isa<lmhlo_disc::SparseReshapeOp>(dominant_op)) {
    return lowerWithScheduleSparseReshapeOpCPU(root_ops, dominant_op, parent,
                                               non_fusion, shape_analysis);
  }

  if (non_fusion && isa<lmhlo_disc::SparseFillEmptyRowsOp>(dominant_op)) {
    return lowerWithScheduleSparseFillEmptyRowsOpCPU(
        root_ops, dominant_op, parent, non_fusion, shape_analysis);
  }

  if (non_fusion && isa<lmhlo_disc::SparseSegmentReductionOp>(dominant_op)) {
    return lowerWithScheduleSparseSegmentReductionOpCPU(
        root_ops, dominant_op, parent, non_fusion, shape_analysis);
  }

  if (non_fusion &&
      isa<lmhlo_disc::SparseSegmentReductionWithEmptyRowsOp>(dominant_op)) {
    return lowerWithScheduleSparseSegmentReductionWithEmptyRowsOpCPU(
        root_ops, dominant_op, parent, non_fusion, shape_analysis);
  }

  if (non_fusion && isa<lmhlo_disc::WhereOp>(dominant_op)) {
    return lowerWithScheduleWhereOpCPU(root_ops, dominant_op, parent,
                                       non_fusion, shape_analysis);
  }

  const auto loc = dominant_op->getLoc();
  OpBuilder b(root_ops.back());

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> vars;
  SmallVector<Value> starts(rank, zero);
  SmallVector<Value> limits = getShapeValues(&b, result);
  SmallVector<Value> steps(rank, one);

  (void)createParallelAndSetInsPt(b, loc, vars, starts, limits, steps, {});
  Value linear = calcLinearIndex(&b, loc, vars, limits);
  for (Operation* root_op : root_ops) {
    Value memref = cast<lmhlo::LmhloOp>(root_op).getResultBuffer();
    if (failed(lowerHelper(b, loc, root_op, linear, shape_analysis)))
      return failure();
  }
  // remove the root_op if it has no other users except the memref
  if (non_fusion) {
    for (Operation* root_op : root_ops) root_op->erase();
  } else {
    assert(parent != nullptr && "Parent must be provided for fusion lowering");
    cleanUnusedLhloOps(parent);
  }
  return success();
}

// Emitter for non-row-reduction kInput fusion pattern.
// Take a column reduction `memref<100x1100xf32> -> memref<1100xf32>` as an
// example:
//
// for (int iOutter = 0; iOutter < 1100; iOutter += kReductionTileSizeOnCPU) {
//   int u = std::min(iOutter+kReductionTileSizeOnCPU, 1100);
//   for (int iInner = iOutter; iInner < u; ++iInner) {
//     out[iInner] = init_value;
//   }
//   for (int j = 0; j < 100; ++j) {
//     for (int iInner = iOutter; iInner < u; ++iInner) {
//       out[iInner] = in[j*1100 + iInnerr];
//     }
//   }
// }
LogicalResult lowerWithSchedulekInputCPU(
    ArrayRef<Operation*> root_ops, Operation* dominant_op,
    Block* parent = nullptr, bool non_fusion = false, bool parallel_loop = true,
    bool multi_dim_loop = false,
    const ShapeAnalysis* shape_analysis = nullptr) {
  if (root_ops.size() != 1) {
    // Not supporting multi output for kInput fusion A.T.M.
    return failure();
  }
  auto reduce = cast<lmhlo::ReduceOp>(dominant_op);
  if (isRowReduction(reduce)) {
    // Row reduction should be processed by other handler.
    return failure();
  }

  Value in = reduce->getOperand(0);
  Value init = reduce->getOperand(1);
  Value out = reduce->getOperand(2);
  auto inTy = in.getType().cast<MemRefType>();
  auto outTy = out.getType().cast<MemRefType>();
  int outRank = outTy.getRank();
  assert(outRank > 0);

  const auto loc = dominant_op->getLoc();
  OpBuilder b(root_ops.back());
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> outVars;
  SmallVector<Value> starts(outRank, zero);
  SmallVector<Value> limits = getShapeValues(&b, out);
  SmallVector<Value> steps(outRank, one);
  Value innerMostDimSize = limits.back();
  Value innerMostStep = innerMostDimSize;
  if (getReductionTileSizeOnCPU() > 0) {
    innerMostStep =
        b.create<arith::ConstantIndexOp>(loc, getReductionTileSizeOnCPU());
  }
  steps.back() = innerMostStep;

  (void)createParallelAndSetInsPt(b, loc, outVars, starts, limits, steps, {});
  Value outterIV = outVars.back();
  Value upperBound = b.create<arith::AddIOp>(loc, outterIV, innerMostStep);
  Value pred = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                       upperBound, innerMostDimSize);
  upperBound =
      b.create<arith::SelectOp>(loc, pred, upperBound, innerMostDimSize);

  // init the output buffer;
  Value initValue;
  if (init.getType().cast<MemRefType>().getRank() > 0) {
    initValue = b.create<memref::LoadOp>(loc, init, zero);
  } else {
    initValue = b.create<memref::LoadOp>(loc, init);
  }
  Value innerIV;
  auto forOp =
      createLoopAndSetInsPt(b, loc, innerIV, outterIV, upperBound, one);
  outVars.back() = innerIV;
  b.create<memref::StoreOp>(loc, initValue, out, outVars);
  b.setInsertionPointAfter(forOp);

  // emit reduction loop
  auto dimensions = reduce.getDimensions().getValues<int64_t>();
  Value totalElemsToReduce = one;
  SmallVector<Value, 4> reduceDimSizeVec;
  for (auto dim : dimensions) {
    auto dim_size = b.create<memref::DimOp>(loc, in, dim);
    reduceDimSizeVec.push_back(dim_size);
    totalElemsToReduce =
        b.create<arith::MulIOp>(loc, totalElemsToReduce, dim_size);
  }
  Value reductionVar;
  auto reductionForOp = createLoopAndSetInsPt(b, loc, reductionVar, zero,
                                              totalElemsToReduce, one);
  auto reduceMultiDimIndex =
      calcMultiDimIndex(&b, loc, reductionVar, reduceDimSizeVec);

  forOp = createLoopAndSetInsPt(b, loc, innerIV, outterIV, upperBound, one);
  outVars.back() = innerIV;

  SmallVector<Value, 4> inVars;
  int reduceDimIdx = 0;
  int nonReduceDimIdx = 0;
  for (int dim = 0; dim < inTy.getRank(); ++dim) {
    bool reduceDim = (std::find(dimensions.begin(), dimensions.end(), dim) !=
                      dimensions.end());
    if (reduceDim) {
      inVars.push_back(reduceMultiDimIndex[reduceDimIdx]);
      reduceDimIdx++;
    } else {
      inVars.push_back(outVars[nonReduceDimIdx]);
      nonReduceDimIdx++;
    }
  }

  auto lhs = b.create<memref::LoadOp>(loc, in, inVars);
  auto rhs = b.create<memref::LoadOp>(loc, out, outVars);
  AccumulatorFactory accumFactory = getFactory(b, loc, reduce.getBody());
  auto acc = accumFactory(lhs, rhs);
  b.create<memref::StoreOp>(loc, acc, out, outVars);
  b.setInsertionPointAfter(reductionForOp);

  for (Operation* root_op : root_ops) root_op->erase();
  return success();
}

LogicalResult HandleCpuFusionOp(OpBuilder& b, Operation* fusion,
                                ShapeAnalysis* shape_analysis) {
  LLVM_DEBUG(llvm::dbgs() << "HandleCpuFusionOp: " << *fusion << "\n");
  auto fusion_op = cast<lmhlo::FusionOp>(fusion);
  assert(fusion_op);
  FusionPattern fusion_pattern(fusion_op, shape_analysis);
  auto root_ops = fusion_pattern.getRootOps();
  auto fused_block = &(fusion_op.getRegion().front());
  LLVM_DEBUG(llvm::dbgs() << "# root ops = " << root_ops.size() << "\n");
  for (Operation* root : root_ops)
    LLVM_DEBUG(llvm::dbgs() << "  root: " << *root << "\n");

  // No need to do codegen, return directly.
  if (root_ops.empty()) {
    cleanUnusedLhloOps(fused_block);
    return success();
  }

  // 1, If any reduce op among the 'root_ops', follow the schedule of it;
  //    or else, follow the schedule of kLoop.
  // 2, If there are a mixer of column reductions and row reductions,
  //    follow the schedule of the row reduction, and implement all the
  //    column reduction with the 'pure atomic' way, which has no
  //    requirement on the schedule.
  // TODO(disc): the support of row reduction and 'pure atomic' reduction
  auto fusion_type = fusion_pattern.getFusionType();
  auto dominant_op = fusion_pattern.getDominantOp();
  switch (fusion_type) {
    case FusionType::kColReduction:
    case FusionType::kInput:
      if (failed(lowerWithSchedulekInputCPU(root_ops, dominant_op, fused_block,
                                            /*non_fusion*/ false,
                                            /*parallel_loop*/ true,
                                            /*multi_dim_loop*/ true))) {
        return dominant_op->emitError() << "failed to lower to loops";
      }
      break;
    case FusionType::kRowReduction:
    case FusionType::kLoop:
      if (failed(lowerWithScheduleLoopCPU(root_ops, dominant_op, fused_block,
                                          /*non_fusion*/ false,
                                          /*parallel_loop*/ true,
                                          /*multi_dim_loop*/ true))) {
        return dominant_op->emitError() << "failed to lower to loops";
      }
      break;
    case FusionType::kLargeConcat:
      if (failed(lowerWithScheduleLargeConcatCPU(
              root_ops, dominant_op, fused_block, /*non_fusion*/ false))) {
        return dominant_op->emitError() << "failed to lower to loops";
      }
      break;
    case FusionType::kWhere:
      if (failed(lowerWithScheduleWhereOpCPU(root_ops, dominant_op, fused_block,
                                             /*non_fusion*/ false))) {
        return dominant_op->emitError() << "failed to lower to loops";
      }
      break;
    case FusionType::kSparseReduction:
      if (failed(lowerWithScheduleSparseSegmentReductionWithEmptyRowsOpCPU(
              root_ops, dominant_op, fused_block,
              /*non_fusion*/ false))) {
        return dominant_op->emitError() << "failed to lower to loops";
      }
      break;
    default:
      return dominant_op->emitError() << "Unknown fusion type";
  }
  return success();
}
}  // namespace

// Expand the root ops in a fused func into a parrallel loop or a set of
// nested loops. This pass must be executed after the fusion pass, and works
// together with the InputInlineFusion pass after it for fusion codegen.
//
// TODO(disc): Currently this pass supports lmhlo.FusionOp to have lmhlo ops
// inside, not mhlo. It's mainly because we now do fusion on lmhlo, not mhlo.
// The fusion pass can be moved to mhlo after shape dialect is brought in to
// represent shape calculation on tensor layer, and we would be able to do shape
// calculation lowering for mhlo.FusionOp. Reconsider the fusion representation
// after these are done, a lmhlo.FusionOp with mhlo inside would be more
// friendly to the legacy FusedIrEmitter.
struct DiscLhloLegalizeRootsToParallelLoops
    : public DiscLhloLegalizeRootsToParallelLoopsPassBase<
          DiscLhloLegalizeRootsToParallelLoops> {
  DiscLhloLegalizeRootsToParallelLoops(int core_count, int cc_major,
                                       int cc_minor) {
    core_count_ = core_count;
    cc_major_ = cc_major;
    cc_minor_ = cc_minor;
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    DiscLhloLegalizeRootsToParallelLoopsPassBase<
        DiscLhloLegalizeRootsToParallelLoops>::getDependentDialects(registry);
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // skip kdot fusion func.
    if (func->getAttrOfType<StringAttr>(kFuncCompIntensFusionAttr)) {
      return;
    }

    // skip shape constraint graph
    if (func.getName() == SymbolicDimMgr::getShapeConstraintGraphFunctionName())
      return;

    std::unique_ptr<ShapeAnalysis> shapeAnalysisPtr;
    if (useShapeConstraintIR()) {
      shapeAnalysisPtr.reset(new ShapeConstraintIRAnalysis(func));
    } else {
      shapeAnalysisPtr.reset(new ShapeAnalysisDeprecated{func});
      if (failed(static_cast<ShapeAnalysisDeprecated*>(shapeAnalysisPtr.get())
                     ->run())) {
        func->emitError("failed to do shape analysis");
        signalPassFailure();
        return;
      }
    }
    auto& shape_analysis = *shapeAnalysisPtr;

    OpBuilder b(func);
    SmallVector<Operation*, 4> gpu_non_fusion_worklist;
    SmallVector<Operation*, 4> cpu_non_fusion_worklist;
    SmallVector<Operation*, 4> gpu_fusion_worklist;
    SmallVector<Operation*, 4> cpu_fusion_worklist;
    // TODO: We should put even single nodes into a fusion by fusion pass
    // Revisit this and walk lmhlo::FusionOp only after the revision done.
    func.walk([&](lmhlo::LmhloOp op) {
      // Skip the embedded ops in lmhlo.fusion or lmhlo.reduce/scatter
      lmhlo::LmhloOp parent = op->getParentOfType<lmhlo::LmhloOp>();
      if (parent && !isa<lmhlo::FusionOp>(op)) {
        return;
      }
      if (isFusionType<FusionType::kStitch>(op) && !disc_ral::isOnGpu(op)) {
        return;
      }
      if (isa<lmhlo::FusionOp>(op)) {
        if (disc_ral::isOnGpu(op))
          gpu_fusion_worklist.push_back(op);
        else
          cpu_fusion_worklist.push_back(op);
      } else {
        if (disc_ral::isOnGpu(op))
          gpu_non_fusion_worklist.push_back(op);
        else
          cpu_non_fusion_worklist.push_back(op);
      }
    });

    for (Operation* op : cpu_non_fusion_worklist) {
      // Only for calculating shapes when the backend is gpu. A simple schedule
      // should be sufficient for performance.
      // TODO(disc): Revisit this when the backend is cpu and the calculation is
      // for data.
      llvm::dbgs() << "cpu-no-fusion-op: \n";
      op->dump();
      if (failed(lowerWithScheduleLoopCPU({op}, op, nullptr,
                                          /*non_fusion=*/true,
#ifdef TAO_CPU_ONLY
                                          /*parallel_loop=*/true,
                                          /*multi_dim_loop=*/true
#else
                                          /*parallel_loop=*/false
#endif
                                          ))) {
        op->emitError() << "failed to lower to loops";
        signalPassFailure();
        return;
      }
    }

    for (Operation* op : gpu_non_fusion_worklist) {
      // TODO(disc): single nodes with non kLoop schedule like ReduceOp
      // is not implemented yet. Currently ReduceOp is lowered with loop
      // schedule, which means for poor performance.
      if (failed(lowerWithScheduleLoop({op}, op, nullptr,
                                       /*non_fusion=*/true,
                                       /*parallel_loop=*/true))) {
        op->emitError() << "failed to lower to loops";
        signalPassFailure();
        return;
      }
    }

    LowerConfig lower_config;
    for (Operation* fusion : gpu_fusion_worklist) {
      // Error message should be emitted inside the function.
      if (failed(HandleGpuFusionOp(b, fusion, &shape_analysis, lower_config,
                                   core_count_, cc_major_, cc_minor_))) {
        signalPassFailure();
        return;
      }
    }

    for (Operation* fusion : cpu_fusion_worklist) {
      // Error message should be emitted inside the function.
      if (failed(HandleCpuFusionOp(b, fusion, &shape_analysis))) {
        signalPassFailure();
        return;
      }
    }
    // Input inline for kStitch fusions on GPU.
    {
      auto* context = &this->getContext();
      RewritePatternSet patterns(context);
      patterns.insert<InputInlineFusionPattern>(context, &lower_config);

      // Just apply the patterns greedily.
      // There should always be one scf.ParallelOp in the fusion.
      constexpr unsigned c_MAX_ITERATION = 4096;
      auto config = GreedyRewriteConfig();
      config.maxIterations = c_MAX_ITERATION;
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns),
                                              config))) {
        signalPassFailure();
      }

      // there should be no lmhlo ops after inline fusion,
      // except for the ConstOp of ColReduction, which for now cannot be
      // properly optimized by general DCE pass
      std::vector<Operation*> to_be_removed;
      func.walk([&](lmhlo::FusionOp fusion) {
        auto op = fusion.getOperation();
        if (!isOnGpu(op)) {
          return;
        }
        FusionType fusionType = FusionType::kNone;
        auto fusionTypeAttr =
            op->getAttrOfType<StringAttr>(kDiscFusionTypeAttrName);
        if (fusionTypeAttr) {
          fusionType = fusionTypeFromString(fusionTypeAttr.getValue());
        }
        if (fusionType != FusionType::kStitch) {
          return;
        }
        fusion.getRegion().walk([&](lmhlo::LmhloOp op) {
          if (isa<lmhlo::TerminatorOp>(op)) {
            return;
          }
          if (isa<lmhlo::ConstantOp>(op)) {
            // TODO(disc): Check the ConstOp is from ReduceOp
            to_be_removed.push_back(op);
            return;
          }
          op.emitError("unexpected remaining operation in a FusionOp");
          signalPassFailure();
        });
        fusion.getRegion().walk([&](memref::AssumeAlignmentOp op) {
          auto memref_type = op.getMemref().getType().cast<MemRefType>();
          auto addrSpace =
              memref_type.getMemorySpace().dyn_cast<gpu::AddressSpaceAttr>();
          if (addrSpace && addrSpace.getValue() ==
                               gpu::GPUDialect::getWorkgroupAddressSpace()) {
            to_be_removed.push_back(op);
          }
        });
      });

      for (auto op : to_be_removed) {
        op->erase();
      }
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscLhloLegalizeRootsToParallelLoopsPass(int core_count, int cc_major,
                                               int cc_minor) {
  return std::make_unique<DiscLhloLegalizeRootsToParallelLoops>(
      core_count, cc_major, cc_minor);
}

}  // namespace disc_ral
}  // namespace mlir
