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

#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir-hlo/utils/placement_utils.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/lhlo_elemental_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"

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
  if (shape_analysis) {
    auto fusion = op->getParentOp();
    if (isa<lmhlo::FusionOp>(fusion)) {
      Value leader_memref = shape_analysis->GetLeaderValueWithSameShapeInFusion(
          fusion, result_memref);
      if (leader_memref != nullptr) memref = leader_memref;
    }
  }
  Value memref_1d = createMemRef1DReinterpretCast(b, loc, result_memref);
  auto memref_ty = memref_1d.getType().cast<MemRefType>();
  // Assuming alignment is to help vectorization optimization. Vectorization
  // pass in llvm requires an alignment of `data-byte-width * vector-size`.
  int byte_width = memref_ty.getElementTypeBitWidth() / 8;
  if (byte_width == 0) {
    // Currently, load and store are at least aligned to 1 byte.
    byte_width = 1;
  }
  b.create<memref::AssumeAlignmentOp>(loc, memref_1d, byte_width * vector_size);

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
    auto res = lmhlo::LhloOpToStdScalarOp::map<LHLO_OpTy>(
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
  if (shape_analysis) {
    auto fusion = opaque_op->getParentOp();
    if (isa<lmhlo::FusionOp>(fusion)) {
      Value leader_memref = shape_analysis->GetLeaderValueWithSameShapeInFusion(
          fusion, result_memref);
      if (leader_memref != nullptr) memref = leader_memref;
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
  auto memref_ty = memref_1d.getType().cast<MemRefType>();
  // Assuming alignment is to help vectorization optimization. Vectorization
  // pass in llvm requires an alignment of `data-byte-width * vector-size`.
  int byte_width = memref_ty.getElementTypeBitWidth() / 8;
  if (byte_width == 0) {
    // Currently, load and store are at least aligned to 1 byte.
    byte_width = 1;
  }
  b.create<memref::AssumeAlignmentOp>(loc, memref_1d, byte_width * vector_size);
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
#include "tensorflow/compiler/mlir/disc/transforms/disc_supported_list.h.inc"
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

Operation* getMapOpInReduceRegion(Operation* op) {
  if (!op || !isa<lmhlo::ReduceOp>(op)) {
    return nullptr;
  }
  auto reduce_op = cast<lmhlo::ReduceOp>(op);
  Operation* map_op = nullptr;
  int num_lhlo_ops = 0;
  reduce_op.body().walk([&](Operation* lhlo_op) {
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
      result = b.create<mlir::SelectOp>(loc, lhs.getType(), cond, lhs, rhs);
    } else if (lhs.getType().dyn_cast<FloatType>()) {
      Value cond =
          b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, lhs, rhs);
      result = b.create<mlir::SelectOp>(loc, lhs.getType(), cond, lhs, rhs);
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
      result = b.create<mlir::SelectOp>(loc, lhs.getType(), cond, rhs, lhs);
    } else if (lhs.getType().dyn_cast<FloatType>()) {
      Value cond =
          b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, lhs, rhs);
      result = b.create<mlir::SelectOp>(loc, lhs.getType(), cond, rhs, lhs);
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

Type getLhloOpsElementType(Operation* op) {
  unsigned int num_operands = op->getNumOperands();
  Type result_type = op->getOperand(num_operands - 1)
                         .getType()
                         .cast<MemRefType>()
                         .getElementType();
  return result_type;
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
  auto dimensions = reduce_op.dimensions().getValues<int64_t>();
  for (auto idx : llvm::enumerate(index)) {
    if (std::find(dimensions.begin(), dimensions.end(), idx.index()) ==
        dimensions.end()) {
      output_multidim_index.push_back(idx.value());
    }
  }
  b.create<memref::AtomicRMWOp>(loc, root_element_type,
                                getAtomicRMWKind(reduce_op.body()), data,
                                root_op->getOperand(2), output_multidim_index);
}

/*
 * <h, w, w_i, h_i>
 * loop.for %h_o = 0 to %sh step c_tilesize_h {
 *   %h_tile_inbound = %sh % c_tilesize_h == 0 ||
 *                     %h_o + c_tilesize_h < %sh;
 *   loop.for %w_o = 0 to %sw step c_tilesize_w {
 *     alloc %acc[c_tilesize_w] = init_value;
 *     %w_tile_inbound = %sw % c_tilesize_w == 0 ||
 *                       %w_o + c_tilesize_w < %sw;
 *     if (%w_tile_inbound && %h_tile_inbound) {
 *       loop.for %h_i = 0 to c_tilesize_h step 1 {
 *         // manually unrolled
 *         loop.for %w_i = 0 to c_tilesize_w step 1 {
 *           %val = load %data[%h_o + %h_i, %w_o + %w_i]
 *           %acc[w_i] = std.addf %acc[w_i], %val
 *         }
 *       }
 *     } else {
 *       loop.for %h_i = 0 to c_tilesize_h step 1 {
 *         %h_index = %h_o + %h_i;
 *         %h_inbound = %h_index < %sh;
 *         // manually unrolled
 *         loop.parallel %w_i = 0 to c_tilesize_w step 1 {
 *           %w_index = %w_o + %w_i;
 *           %w_inbound = %w_index < %sw;
 *           if (%w_inbound && %h_inbound) {
 *             %val = load %data[%h_index, %w_index]
 *             %acc[w_i] = %acc[w_i] + %val
 *           }
 *         }
 *       }
 *     }
 *     // manually unrolled
 *     loop.for %w_i = 0 to tile_w step 1 {
 *       std.atomic_add %w + %w_i, %acc[w_i]
 *     }
 *   }
 * }
 */
LogicalResult lowerWithScheduleColReduction(
    ArrayRef<Operation*> root_ops, Operation* dominant_op, Block* parent,
    const ShapeAnalysis* shape_analysis = nullptr) {
  if (!isRank2ColReduction(dominant_op)) {
    return failure();
  }
  Value lhs = *dominant_op->getOperands().begin();
  const MemRefType& lhs_type = lhs.getType().template cast<MemRefType>();
  const Type& element_type = lhs_type.getElementType();
  const int c_tilesize_h = 128;
  const int c_tilesize_w = 2;
  Location loc = dominant_op->getLoc();
  OpBuilder b(root_ops.back());
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value tilesize_h = b.create<arith::ConstantIndexOp>(loc, c_tilesize_h);
  Value tilesize_w = b.create<arith::ConstantIndexOp>(loc, c_tilesize_w);
  Value shape_h = b.create<memref::DimOp>(loc, lhs, zero);
  Value shape_w = b.create<memref::DimOp>(loc, lhs, one);

  SmallVector<Value, 2> vars;
  scf::ParallelOp parallel_op =
      createParallelAndSetInsPt(b, loc, vars, {zero, zero}, {shape_h, shape_w},
                                {tilesize_h, tilesize_w}, {});
  parallel_op.getBody()->clear();
  b.setInsertionPointToStart(parallel_op.getBody());
  Value var_ho = vars[0];
  Value var_wo = vars[1];

  // acc: init_values[num_col_reductions * c_tilesize_w]
  SmallVector<Value, 4> init_values_hi;
  SmallVector<Type, 4> init_values_types;
  for (Operation* root_op : root_ops) {
    if (isRank2ColReduction(root_op)) {
      Value root_lhs = root_op->getOperand(0);
      MemRefType root_lhs_type = root_lhs.getType().template cast<MemRefType>();
      Type root_element_type = root_lhs_type.getElementType();
      ValueRange empty_index;
      Value init_value =
          createLoadOrUseCachedValue(loc, &b, root_op, root_op->getOperand(1),
                                     empty_index, b.saveInsertionPoint());

      for (int w_idx = 0; w_idx < c_tilesize_w; ++w_idx) {
        init_values_hi.push_back(init_value);
        init_values_types.push_back(init_value.getType());
      }
    }
  }

  // if_tile_inbound
  // Note that, without the consideration of backend device, we
  // should emit h_tile_inbound in the begining of for_op_ho.body().
  // But this breaks the logics in loopCoalescingPass, which can only
  // support "perfectNestedLoops".
  Value h_tile_inbound = b.create<arith::OrIOp>(
      loc,
      b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          b.create<arith::RemUIOp>(loc, shape_h, tilesize_h), zero),
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                              b.create<arith::AddIOp>(loc, var_ho, tilesize_h),
                              shape_h));
  Value w_tile_inbound = b.create<arith::OrIOp>(
      loc,
      b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          b.create<arith::RemUIOp>(loc, shape_w, tilesize_w), zero),
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                              b.create<arith::AddIOp>(loc, var_wo, tilesize_w),
                              shape_w));
  Value tile_inbound =
      b.create<arith::AndIOp>(loc, h_tile_inbound, w_tile_inbound);
  scf::IfOp if_tile_inbound_op =
      b.create<scf::IfOp>(loc, /*resultTypes*/ init_values_types, tile_inbound,
                          /*hasElseRegion*/ true);

  if_tile_inbound_op.getThenRegion().front().clear();
  if_tile_inbound_op.getElseRegion().front().clear();
  b.setInsertionPointToStart(&if_tile_inbound_op.getThenRegion().front());

  // h inner in if_tile_inbound.then
  scf::ForOp for_op_hi =
      b.create<scf::ForOp>(loc, zero, tilesize_h, one, init_values_hi);
  for_op_hi.getBody()->clear();
  b.setInsertionPointToStart(for_op_hi.getBody());
  Value var_hi = for_op_hi.getInductionVar();
  Value h_idx = b.create<arith::AddIOp>(loc, var_ho, var_hi);
  SmallVector<Value, 4> yield_values_for_hi;
  // TODO: check the order of root_ops by op.walk() is as expected
  int root_op_idx = 0;
  for (auto* root_op : root_ops) {
    // w inner, which is manually unrolled
    for (int var_wi = 0; var_wi < c_tilesize_w; ++var_wi) {
      // TODO: swap for loops and reuse w_idx
      Value w_idx = b.create<arith::AddIOp>(
          loc, var_wo, b.create<arith::ConstantIndexOp>(loc, var_wi));
      ValueRange input_index({h_idx, w_idx});
      if (isRank2ColReduction(root_op)) {
        auto lhs = root_op->getOperands().begin();
        Value data = createLoadOrUseCachedValue(
            loc, &b, root_op, *lhs, input_index, b.saveInsertionPoint());
        Operation* map_op = getMapOpInReduceRegion(root_op);
        assert(map_op && "not supported reduce");
        Value acc = emitReduceMapOp(b, loc, map_op,
                                    *(for_op_hi.getRegionIterArgs().begin() +
                                      root_op_idx * c_tilesize_w + var_wi),
                                    data);
        assert(acc && "not supported reduce");
        yield_values_for_hi.push_back(acc);
      } else if (isRank2RowReduction(root_op)) {
        assert(false && "unexpected row_reduction");
        return failure();
      } else if (isa<lmhlo::ReduceOp>(root_op)) {
        auto dominant_shape = getShapeValues(&b, dominant_op->getOperand(0));
        Value linear_index =
            calcLinearIndex(&b, loc, input_index, dominant_shape);
        auto root_shape = getShapeValues(&b, root_op->getOperand(0));
        auto mapped_index =
            calcMultiDimIndex(&b, loc, linear_index, root_shape);
        emitNotToVectorReduction(b, loc, root_op, mapped_index);
      } else {
        auto dominant_shape = getShapeValues(&b, dominant_op->getOperand(0));
        Value linear_index =
            calcLinearIndex(&b, loc, input_index, dominant_shape);
        auto root_shape = getShapeValues(&b, root_op->getOperand(0));
        auto mapped_index =
            calcMultiDimIndex(&b, loc, linear_index, root_shape);
        if (!succeeded(
                lowerHelper(b, loc, root_op, linear_index, shape_analysis))) {
          assert(false && "elementwise lowerHelper failure");
          return failure();
        }
      }
    }
    if (isRank2ColReduction(root_op)) {
      root_op_idx++;
    }
  }
  int num_col_reduction_root_ops = root_op_idx;
  b.create<scf::YieldOp>(loc, yield_values_for_hi);

  b.setInsertionPointToEnd(&if_tile_inbound_op.getThenRegion().front());
  b.create<scf::YieldOp>(loc, for_op_hi.getResults());

  b.setInsertionPointToStart(&if_tile_inbound_op.getElseRegion().front());

  // h inner in if_tile_inbound.else
  for_op_hi = b.create<scf::ForOp>(loc, zero, tilesize_h, one, init_values_hi);
  for_op_hi.getBody()->clear();
  b.setInsertionPointToStart(for_op_hi.getBody());
  var_hi = for_op_hi.getInductionVar();
  h_idx = b.create<arith::AddIOp>(loc, var_ho, var_hi);
  Value h_inbound =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, h_idx, shape_h);

  // create c_tilesize_w IfOps, since loop_wo is unrolled
  SmallVector<Value, 4> w_idx(c_tilesize_w);
  SmallVector<Value, 4> w_inbound(c_tilesize_w);
  SmallVector<Value, 4> inbound(c_tilesize_w);
  SmallVector<scf::IfOp, 4> if_inbound_op(c_tilesize_w);
  for (int var_wi = 0; var_wi < c_tilesize_w; ++var_wi) {
    w_idx[var_wi] = b.create<arith::AddIOp>(
        loc, var_wo, b.create<arith::ConstantIndexOp>(loc, var_wi));
    w_inbound[var_wi] = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                w_idx[var_wi], shape_w);
    inbound[var_wi] =
        b.create<arith::AndIOp>(loc, h_inbound, w_inbound[var_wi]);

    SmallVector<Type, 4> if_inbound_init_values_types;
    for (int j = 0; j < num_col_reduction_root_ops; ++j) {
      if_inbound_init_values_types.push_back(
          init_values_types[j * c_tilesize_w + var_wi]);
    }
    if_inbound_op[var_wi] =
        b.create<scf::IfOp>(loc, /*resultTypes*/ if_inbound_init_values_types,
                            inbound[var_wi], /*hasElseRegion*/ true);
    if_inbound_op[var_wi].getThenRegion().front().clear();
    if_inbound_op[var_wi].getElseRegion().front().clear();
  }

  SmallVector<SmallVector<Value, 4>, 4> yield_values_if_inbound_then(
      c_tilesize_w);
  SmallVector<SmallVector<Value, 4>, 4> yield_values_if_inbound_else(
      c_tilesize_w);
  // TODO: check the order of root_ops by op.walk() is as expected
  root_op_idx = 0;
  for (auto* root_op : root_ops) {
    // w inner, which is manually unrolled
    for (int var_wi = 0; var_wi < c_tilesize_w; ++var_wi) {
      // if_inbound.then
      b.setInsertionPointToEnd(&if_inbound_op[var_wi].getThenRegion().front());
      ValueRange input_index({h_idx, w_idx[var_wi]});
      if (isRank2ColReduction(root_op)) {
        auto lhs = root_op->getOperands().begin();
        Value data = createLoadOrUseCachedValue(
            loc, &b, root_op, *lhs, input_index, b.saveInsertionPoint());
        Operation* map_op = getMapOpInReduceRegion(root_op);
        assert(map_op && "not supported reduce");
        Value acc = emitReduceMapOp(b, loc, map_op,
                                    *(for_op_hi.getRegionIterArgs().begin() +
                                      root_op_idx * c_tilesize_w + var_wi),
                                    data);
        assert(acc && "not supported reduce");
        yield_values_if_inbound_then[var_wi].push_back(acc);
      } else if (isRank2RowReduction(root_op)) {
        assert(false && "unexpected row_reduction");
      } else if (isa<mhlo::ReduceOp>(root_op)) {
        // non-IsReductionToVector reduction
        auto dominant_shape = getShapeValues(&b, dominant_op->getOperand(0));
        Value linear_index =
            calcLinearIndex(&b, loc, input_index, dominant_shape);
        auto root_shape = getShapeValues(&b, root_op->getOperand(0));
        auto mapped_index =
            calcMultiDimIndex(&b, loc, linear_index, root_shape);
        emitNotToVectorReduction(b, loc, root_op, mapped_index);
      } else {
        // TODO: We might further use shape_equality check similar as in
        // dhlo_fuse pass
        //       to avoid redundant index calc, when we can tell their shapes
        //       are identical during compile time
        auto dominant_shape = getShapeValues(&b, dominant_op->getOperand(0));
        Value linear_index =
            calcLinearIndex(&b, loc, input_index, dominant_shape);
        if (!succeeded(
                lowerHelper(b, loc, root_op, linear_index, shape_analysis))) {
          assert(false && "elementwise lowerHelper failure");
          return failure();
        }
      }
      b.setInsertionPointToEnd(&if_inbound_op[var_wi].getElseRegion().front());
      if (isRank2ColReduction(root_op)) {
        Value acc_init = *(for_op_hi.getRegionIterArgs().begin() +
                           root_op_idx * c_tilesize_w + var_wi);
        yield_values_if_inbound_else[var_wi].push_back(acc_init);
      }
    }
    if (isRank2ColReduction(root_op)) {
      root_op_idx++;
    }
  }

  for (int var_wi = 0; var_wi < c_tilesize_w; ++var_wi) {
    b.setInsertionPointToEnd(&if_inbound_op[var_wi].getThenRegion().front());
    b.create<scf::YieldOp>(loc, yield_values_if_inbound_then[var_wi]);
    b.setInsertionPointToEnd(&if_inbound_op[var_wi].getElseRegion().front());
    b.create<scf::YieldOp>(loc, yield_values_if_inbound_else[var_wi]);
  }

  b.setInsertionPointToEnd(for_op_hi.getBody());
  root_op_idx = 0;
  for (auto* root_op : root_ops) {
    if (isRank2ColReduction(root_op)) {
      for (int var_wi = 0; var_wi < c_tilesize_w; ++var_wi) {
        yield_values_for_hi[root_op_idx * c_tilesize_w + var_wi] =
            *(if_inbound_op[var_wi].getResults().begin() + root_op_idx);
      }
      root_op_idx++;
    }
  }
  b.create<scf::YieldOp>(loc, yield_values_for_hi);

  b.setInsertionPointToEnd(&if_tile_inbound_op.getElseRegion().front());
  b.create<scf::YieldOp>(loc, for_op_hi.getResults());

  b.setInsertionPointAfter(if_tile_inbound_op);
  root_op_idx = 0;
  for (auto root_op : root_ops) {
    if (isRank2ColReduction(root_op)) {
      for (int var_wi = 0; var_wi < c_tilesize_w; ++var_wi) {
        // TODO: redundant emission of w_idx
        Value w_idx = b.create<arith::AddIOp>(
            loc, var_wo, b.create<arith::ConstantIndexOp>(loc, var_wi));
        Value w_inbound = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, w_idx, shape_w);
        auto if_inbound_op =
            b.create<scf::IfOp>(loc, /*resultTypes*/ TypeRange{}, w_inbound,
                                /*hasElseRegion*/ false);
        if_inbound_op.getThenRegion().front().clear();
        b.setInsertionPointToEnd(&if_inbound_op.getThenRegion().front());

        SmallVector<Value, 1> w_idx_vec = {w_idx};
        b.create<memref::AtomicRMWOp>(
            loc, element_type,
            getAtomicRMWKind(cast<lmhlo::ReduceOp>(root_op).body()),
            *(if_tile_inbound_op.getResults().begin() +
              root_op_idx * c_tilesize_w + var_wi),
            root_op->getOperand(2), ValueRange(w_idx_vec));
        b.create<scf::YieldOp>(loc, ValueRange{});
        b.setInsertionPointAfter(if_inbound_op);
      }
      root_op_idx++;
    }
  }

  b.setInsertionPointToEnd(parallel_op.getBody());
  b.create<scf::YieldOp>(loc, ValueRange({}));

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
        loc, cast<lmhlo::ReduceOp>(reduce).init_values().front());
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
                            StringAttr strAttr) {
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
    // int bit_width = bit_width(val);
    // int segments = ceil(bit_width, 32);
    // auto val = bitcast(val, vec(segments, i32));
    // for (int i = 0; i < segments; ++i) {
    //   auto insert_elem = extract_element(x, i);
    //   insert_elem = __xhfl_xor(insert_elem, offset);
    //   val = insert_element(insert_elem, i);
    //   elem_val = element_extractor(elem, i);
    //   sum += __xhfl_xor(elem, offset);
    // }
    // sum += bitcast(val, integer(bit_width)))
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
  auto bufferType = MemRefType::get(
      {size}, elem_type, {}, gpu::GPUDialect::getWorkgroupAddressSpace());
  auto alloc = b.create<memref::AllocOp>(loc, bufferType);
  return alloc.getResult();
}

LogicalResult emitInThreadRowReduction(
    OpBuilder& b, Location loc, Operation* op,
    mlir::Block::args_iterator acc_iter, ValueRange& input_index,
    SmallVector<Value, 4>& yield_values_for_j) {
  if (!isRank2RowReduction(op)) {
    return failure();
  }
  auto data =
      createLoadOrUseCachedValue(loc, &b, op, *op->getOperands().begin(),
                                 input_index, b.saveInsertionPoint());
  AccumulatorFactory accumFactory =
      getFactory(b, op->getLoc(), cast<lmhlo::ReduceOp>(op).body());
  // TODO: make sure it is accumulating based on the init value of reduce.
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
      if (failed(emitInThreadRowReduction(b, loc, root_op, acc_iter,
                                          input_index, yield_values_for_j))) {
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
          getAtomicRMWKind(cast<lmhlo::ReduceOp>(root_op).body()), data,
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
                                            int row_tile = 1) {
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
LogicalResult lowerWithScheduleRowReduction<DISC_ONE_ROUND_SHUFFLE_ROW_REDUCE>(
    ArrayRef<Operation*> root_ops, Operation* dominant_op, Block* parent,
    const ShapeAnalysis* shape_analysis, int row_tile) {
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
      loc, thread_per_block / kWarpSize * row_tile);

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
  Value vec_size = b.create<arith::ConstantIndexOp>(loc, row_tile);
  Value row_base = b.create<arith::AddIOp>(
      loc, var_m, b.create<arith::MulIOp>(loc, warp_id, vec_size));
  SmallVector<Value, 4> row_ids(row_tile);
  for (int i = 0; i < row_tile; i++) {
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
    SmallVector<Value, 4> init_values(row_reduction_roots.size() * row_tile);
    for (auto root_pair : llvm::enumerate(row_reduction_roots)) {
      Operation* root_op = root_pair.value();
      int idx = root_pair.index();
      Value init_value = b.create<memref::LoadOp>(
          loc, cast<lmhlo::ReduceOp>(root_op).init_values()[0]);
      for (int i = 0; i < row_tile; i++) {
        init_values[i * row_reduction_roots.size() + idx] = init_value;
      }
    }
    Value var_k = nullptr;
    scf::ForOp for_op_k =
        createLoopAndSetInsPt(b, loc, var_k, /* lb */ lane_id,
                              /* ub */ cols, /* step */ warp_size, init_values);
    SmallVector<Value, 4> yield_values;
    for (int i = 0; i < row_tile; i++) {
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

    SmallVector<Value, 4> shuffle_val(row_reduction_roots.size() * row_tile);
    SmallVector<Type, 4> shuffle_type(row_reduction_roots.size());
    SmallVector<AccumulatorFactory, 4> accum_factory(
        row_reduction_roots.size());
    for (auto root_pair : llvm::enumerate(row_reduction_roots)) {
      Operation* root_op = root_pair.value();
      int idx = root_pair.index();
      for (int i = 0; i < row_tile; i++) {
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
          b, root_op->getLoc(), cast<lmhlo::ReduceOp>(root_op).body()));
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
        for (int i = 0; i < row_tile; i++) {
          int val_idx = i * row_reduction_roots.size() + idx;
          auto result = emitWidthAdaptShuffle(
              b, loc, shuffle_val[val_idx], shuffle_type[idx], offset_val,
              shuffle_width, b.getStringAttr("xor"));
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
      auto memref_ty = output_memref.getType().cast<MemRefType>();
      int byte_width = memref_ty.getElementTypeBitWidth() / 8;
      if (byte_width == 0) {
        // Currently, load and store are at least aligned to 1 byte.
        byte_width = 1;
      }
      b.create<memref::AssumeAlignmentOp>(loc, output_memref,
                                          byte_width * row_tile);
      for (int i = 0; i < row_tile; i++) {
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
    int row_tile) {
  Value warp_size =
      b.create<arith::ConstantIntOp>(loc, kWarpSize, b.getIntegerType(32));
  auto xorAttr = b.getStringAttr("xor");
  for (auto root_op_en : llvm::enumerate(row_reduction_ops)) {
    auto idx = root_op_en.index();
    auto root_op = root_op_en.value();
    SmallVector<Value, 2> sum_vec(row_tile);
    for (int i = 0; i < row_tile; i++) {
      auto iter = acc_iter + i * row_reduction_ops.size();
      sum_vec[i] = *(iter + idx);
    }
    const auto elemType = getLhloOpsElementType(root_op);
    Type shuffleElemType;
    if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
      return failure();
    }
    AccumulatorFactory accumFactory =
        getFactory(b, root_op->getLoc(), cast<lmhlo::ReduceOp>(root_op).body());
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
        shuffle_result_vec[i] = emitWidthAdaptShuffle(
            b, loc, sum_vec[i], shuffleElemType, offset_val, warp_size,
            b.getStringAttr("xor"));
      }
      for (int i = 0; i < row_tile; i++) {
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
      for (int i = 0; i < row_tile; i++) {
        sum_vec[i] = accumFactory(sum_vec[i], shuffle_result_vec[i]);
      }
    }
    auto if_lane_id_is_zero =
        b.create<scf::IfOp>(loc, /*resultTypes*/ ArrayRef<Type>{},
                            lane_id_is_zero, /*hasElseRegion*/ false);
    if_lane_id_is_zero.getThenRegion().front().clear();
    b.setInsertionPointToStart(&if_lane_id_is_zero.getThenRegion().front());
    auto shared_idx = warp_id;
    SmallVector<Value, 4> shared_idx_vec;
    shared_idx_vec.push_back(shared_idx);
    SmallVector<Value, 2> shared_mem_vec;
    for (int i = 0; i < row_tile; i++) {
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
    for (int i = 0; i < row_tile; i++) {
      b.create<memref::StoreOp>(loc, sum_vec[i], shared_mem_vec[i],
                                shared_idx_vec);
    }
    b.create<scf::YieldOp>(loc, ValueRange({}));
    b.setInsertionPointAfter(if_lane_id_is_zero);
  }
  return success();
}

LogicalResult emitSecondRoundShuffleStitch(
    OpBuilder& b, Location loc, Operation* op,
    mlir::ResultRange::iterator acc_iter, Value thread_id_is_zero,
    SmallVector<Value> output_idx_vec, int row_tile, int block_size,
    Value smem_buffer, bool is_output, bool external_output_only) {
  assert(block_size % kWarpSize == 0);
  int num_warps = block_size / kWarpSize;
  Value shuffle_size =
      b.create<arith::ConstantIntOp>(loc, num_warps, b.getIntegerType(32));
  auto xorAttr = b.getStringAttr("xor");
  SmallVector<Value> results(row_tile);
  for (int i = 0; i < row_tile; i++) {
    results[i] = *(acc_iter + i);
  }
  const auto elemType = getLhloOpsElementType(op);
  Type shuffleElemType;
  if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
    return failure();
  }
  AccumulatorFactory accumFactory =
      getFactory(b, op->getLoc(), cast<lmhlo::ReduceOp>(op).body());
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
  if (smem_buffer != nullptr && !external_output_only) {
    auto memref_ty = smem_buffer.getType().cast<MemRefType>();
    int byte_width = memref_ty.getElementTypeBitWidth() / 8;
    if (byte_width == 0) {
      // Currently, load and store are at least aligned to 1 byte.
      byte_width = 1;
    }
    b.create<memref::AssumeAlignmentOp>(loc, smem_buffer,
                                        byte_width * row_tile);
    for (int i = 0; i < row_tile; i++) {
      Value index = b.create<arith::ConstantIndexOp>(loc, i);
      b.create<memref::StoreOp>(loc, results[i], smem_buffer, index);
    }
  }
  if (is_output) {
    auto res_memref = *(op->getOperands().begin() + (op->getNumOperands() - 1));
    auto memref_ty = res_memref.getType().cast<MemRefType>();
    int byte_width = memref_ty.getElementTypeBitWidth() / 8;
    if (byte_width == 0) {
      // Currently, load and store are at least aligned to 1 byte.
      byte_width = 1;
    }
    b.create<memref::AssumeAlignmentOp>(loc, res_memref, byte_width * row_tile);
    for (int i = 0; i < row_tile; i++) {
      b.create<memref::StoreOp>(loc, results[i], res_memref, output_idx_vec[i]);
    }
  }
  b.create<scf::YieldOp>(loc, ValueRange({}));
  b.setInsertionPointAfter(if_thread_id_is_zero);

  return success();
}

LogicalResult emitSecondRoundShuffle(
    OpBuilder& b, Location loc,
    const SmallVector<Operation*, 4>& row_reduction_ops,
    mlir::ResultRange::iterator acc_iter, Value thread_id_is_zero,
    SmallVector<Value> output_idx_vec, int row_tile, int block_size) {
  assert(block_size % kWarpSize == 0);
  int num_warps = block_size / kWarpSize;
  Value shuffle_size =
      b.create<arith::ConstantIntOp>(loc, num_warps, b.getIntegerType(32));
  auto xorAttr = b.getStringAttr("xor");
  for (auto root_op_en : llvm::enumerate(row_reduction_ops)) {
    auto idx = root_op_en.index();
    auto root_op = root_op_en.value();
    SmallVector<Value> sum_vec(row_tile);
    for (int i = 0; i < row_tile; i++) {
      auto iter = acc_iter + i * row_reduction_ops.size();
      sum_vec[i] = *(iter + idx);
    }
    const auto elemType = getLhloOpsElementType(root_op);
    Type shuffleElemType;
    if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
      return failure();
    }
    AccumulatorFactory accumFactory =
        getFactory(b, root_op->getLoc(), cast<lmhlo::ReduceOp>(root_op).body());
    for (int offset = 1; offset < num_warps; offset <<= 1) {
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
      SmallVector<Value, 4> shuffle_result_vec(row_tile);
      for (int i = 0; i < row_tile; i++) {
        shuffle_result_vec[i] =
            emitWidthAdaptShuffle(b, loc, sum_vec[i], shuffleElemType,
                                  offset_val, shuffle_size, xorAttr);
      }
      for (int i = 0; i < row_tile; i++) {
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
      for (int i = 0; i < row_tile; i++) {
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
    auto memref_ty = res_memref.getType().cast<MemRefType>();
    int byte_width = memref_ty.getElementTypeBitWidth() / 8;
    if (byte_width == 0) {
      // Currently, load and store are at least aligned to 1 byte.
      byte_width = 1;
    }
    b.create<memref::AssumeAlignmentOp>(loc, res_memref, byte_width * row_tile);
    for (int i = 0; i < row_tile; i++) {
      b.create<memref::StoreOp>(loc, sum_vec[i], res_memref, output_idx_vec[i]);
    }
    b.create<scf::YieldOp>(loc, ValueRange({}));
    b.setInsertionPointAfter(if_thread_id_is_zero);
  }
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
LogicalResult lowerWithScheduleRowReduction<DISC_TWO_ROUND_SHUFFLE_ROW_REDUCE>(
    ArrayRef<Operation*> root_ops, Operation* dominant_op, Block* parent,
    const ShapeAnalysis* shape_analysis, int row_tile) {
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
  SmallVector<std::map<Operation*, Value>> shared_mem_map_vec(row_tile);
  for (auto root_op : root_ops) {
    if (isRank2RowReduction(root_op)) {
      row_reduction_ops.push_back(root_op);
    }
  }
  Value warp_size_val = b.create<arith::ConstantIndexOp>(loc, kWarpSize);

  // loop over (rows, threads)
  // Note that we know `shape_h` can be divided by `row_tile` exactly.
  Value vec_size = b.create<arith::ConstantIndexOp>(loc, row_tile);
  Value block_number = b.create<arith::DivUIOp>(loc, shape_h, vec_size);
  SmallVector<Value, 2> vars;
  scf::ParallelOp parallel_op = createParallelAndSetInsPt(
      b, loc, vars, {zero, zero}, {block_number, num_threads}, {one, one}, {});
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

  SmallVector<Value, 4> init_values(row_reduction_ops.size() * row_tile);
  for (auto root_pair : llvm::enumerate(row_reduction_ops)) {
    Operation* root_op = root_pair.value();
    int idx = root_pair.index();
    auto init_value = b.create<memref::LoadOp>(
        loc, *(cast<lmhlo::ReduceOp>(root_op).init_values().begin()));
    init_values_cache[root_op] = init_value;
    for (int i = 0; i < row_tile; i++) {
      init_values[i * row_reduction_ops.size() + idx] = init_value;
    }
    const auto elemType = getLhloOpsElementType(root_op);
    Type shuffleElemType;
    if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
      return failure();
    }
    for (int i = 0; i < row_tile; i++) {
      auto shared_mem = createSharedMemory(b, loc, kWarpSize, shuffleElemType);
      shared_mem_map_vec[i][root_op] = shared_mem;
    }
  }

  Value row_base = b.create<arith::MulIOp>(loc, var_m, vec_size);
  SmallVector<Value> row_ids(row_tile);
  for (int i = 0; i < row_tile; i++) {
    row_ids[i] = b.create<arith::AddIOp>(
        loc, row_base, b.create<arith::ConstantIndexOp>(loc, i));
  }

  Value var_j = nullptr;
  scf::ForOp for_op_j =
      createLoopAndSetInsPt(b, loc, var_j,
                            /*lb*/ var_n, /*ub*/ shape_w,
                            /*step*/ num_threads, init_values);

  SmallVector<Value, 4> yield_values_for_j;
  for (int i = 0; i < row_tile; i++) {
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
  emitFirstRoundShuffle(
      b, loc, row_reduction_ops, shared_mem_map_vec,
      for_op_j.getResults().begin(),  // + i * row_reduction_ops.size(),
      lane_id_is_zero, warp_id, row_tile);
  b.setInsertionPointToEnd(parallel_op.getBody());
  b.create<gpu::BarrierOp>(loc);

  SmallVector<Type, 4> root_elem_types(row_reduction_ops.size() * row_tile);
  for (auto root_op_en : llvm::enumerate(row_reduction_ops)) {
    auto idx = root_op_en.index();
    auto root_op = root_op_en.value();
    auto elem_type = getLhloOpsElementType(root_op);
    for (int i = 0; i < row_tile; i++) {
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
  SmallVector<Value, 4> true_yield_values(row_reduction_ops.size() * row_tile);
  for (auto root_op_en : llvm::enumerate(row_reduction_ops)) {
    auto idx = root_op_en.index();
    auto root_op = root_op_en.value();
    auto shared_idx = lane_id;
    SmallVector<Value, 4> shared_idx_vec;
    shared_idx_vec.push_back(shared_idx);
    for (int i = 0; i < row_tile; i++) {
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
          b.create<memref::LoadOp>(loc, shared_mem, shared_idx_vec);
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
  SmallVector<Value, 4> false_yield_values(row_reduction_ops.size() * row_tile);
  for (auto root_op_en : llvm::enumerate(row_reduction_ops)) {
    auto idx = root_op_en.index();
    auto root_op = root_op_en.value();
    assert((init_values_cache.find(root_op) != init_values_cache.end()) &&
           "unexpected init_values_cache");
    for (int i = 0; i < row_tile; i++) {
      false_yield_values[i * row_reduction_ops.size() + idx] =
          init_values_cache[root_op];
    }
  }
  b.create<scf::YieldOp>(loc, false_yield_values);
  b.setInsertionPointToEnd(parallel_op.getBody());
  auto acc_iter = if_lane_id_inbound.getResults().begin();
  emitSecondRoundShuffle(b, loc, row_reduction_ops, acc_iter, thread_id_is_zero,
                         row_ids, row_tile, getThreadPerBlock(dominant_op));

  b.setInsertionPointToEnd(parallel_op.getBody());
  b.create<scf::YieldOp>(loc, ValueRange({}));

  // remove the root_op if it has no other users except the memref
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
        loc, cast<lmhlo::ReduceOp>(root_op).init_values()[0]);
    init_values[idx] = init_value;
    accum_factory[idx] = std::move(getFactory(
        b, root_op->getLoc(), cast<lmhlo::ReduceOp>(root_op).body()));
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
        getAtomicRMWKind(cast<lmhlo::ReduceOp>(root_op).body()), partial_result,
        root_op->getOperand(2), ValueRange({col_index}));
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

LogicalResult emitInBlockRowReduce(OpBuilder& b, Location loc, Block* block,
                                   Operation* op, Value block_id,
                                   Value thread_id, int64_t block_size,
                                   int64_t row_tile, Value smem_buffer,
                                   ShapeAnalysis* shape_analysis,
                                   bool is_output, bool external_output_only) {
  Value lhs = op->getOperand(0);
  const MemRefType& lhs_type = lhs.getType().template cast<MemRefType>();
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value shape_h = b.create<memref::DimOp>(loc, lhs, zero);
  Value shape_w = b.create<memref::DimOp>(loc, lhs, one);

  Value warp_size_val = b.create<arith::ConstantIndexOp>(loc, kWarpSize);
  Value lane_id = b.create<arith::RemUIOp>(loc, thread_id, warp_size_val);
  Value warp_id = b.create<arith::DivUIOp>(loc, thread_id, warp_size_val);
  Value lane_id_is_zero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane_id, zero);
  Value thread_id_is_zero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, thread_id, zero);

  Value vec_size = b.create<arith::ConstantIndexOp>(loc, row_tile);
  Value row_base = b.create<arith::MulIOp>(loc, block_id, vec_size);
  SmallVector<Value> row_ids(row_tile);
  for (int i = 0; i < row_tile; i++) {
    row_ids[i] = b.create<arith::AddIOp>(
        loc, row_base, b.create<arith::ConstantIndexOp>(loc, i));
  }

  SmallVector<Value, 4> init_values(row_tile);
  SmallVector<std::map<Operation*, Value>> shared_mem_map_vec(row_tile);
  auto init_value = b.create<memref::LoadOp>(
      loc, *(cast<lmhlo::ReduceOp>(op).init_values().begin()));
  for (int i = 0; i < row_tile; i++) {
    init_values[i] = init_value;
  }
  const auto elemType = getLhloOpsElementType(op);
  Type shuffleElemType;
  if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
    return failure();
  }
  for (int i = 0; i < row_tile; i++) {
    Value shared_mem = createSharedMemoryForOp(b, loc, op, kWarpSize);
    if (shared_mem == nullptr) {
      return failure();
    }
    shared_mem_map_vec[i][op] = shared_mem;
  }

  Value num_threads = b.create<arith::ConstantIndexOp>(loc, block_size);
  Value var_j = nullptr;
  scf::ForOp for_op_j =
      createLoopAndSetInsPt(b, loc, var_j,
                            /*lb*/ thread_id, /*ub*/ shape_w,
                            /*step*/ num_threads, init_values);

  SmallVector<Value, 4> in_thread_reduction;
  for (int i = 0; i < row_tile; i++) {
    SmallVector<Value, 4> multidim({row_ids[i], var_j});
    ValueRange input_index(multidim);
    if (failed(emitInThreadRowReduction(
            b, loc, op, for_op_j.getRegionIterArgs().begin() + i, input_index,
            in_thread_reduction))) {
      return failure();
    }
  }
  b.create<scf::YieldOp>(loc, in_thread_reduction);

  b.setInsertionPointToEnd(block);
  // For reuse of emitFirstRoundShuffle
  SmallVector<Operation*, 4> row_reduction_ops;
  row_reduction_ops.push_back(op);
  emitFirstRoundShuffle(
      b, loc, row_reduction_ops, shared_mem_map_vec,
      for_op_j.getResults().begin(),  // + i * row_reduction_ops.size(),
      lane_id_is_zero, warp_id, row_tile);
  b.setInsertionPointToEnd(block);
  b.create<gpu::BarrierOp>(loc);

  SmallVector<Type, 4> root_elem_types(row_tile);
  auto elem_type = getLhloOpsElementType(op);
  for (int i = 0; i < row_tile; i++) {
    root_elem_types[i] = elem_type;
  }

  Value lane_id_inbound = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, lane_id,
      b.create<arith::ConstantIndexOp>(loc, block_size / kWarpSize));
  scf::IfOp if_lane_id_inbound =
      b.create<scf::IfOp>(loc, /*resultTypes*/ root_elem_types, lane_id_inbound,
                          /*hasElseRegion*/ true);
  if_lane_id_inbound.getThenRegion().front().clear();
  if_lane_id_inbound.getElseRegion().front().clear();
  b.setInsertionPointToStart(&if_lane_id_inbound.getThenRegion().front());
  SmallVector<Value, 4> true_yield_values(row_tile);
  auto shared_idx = lane_id;
  SmallVector<Value, 4> shared_idx_vec;
  shared_idx_vec.push_back(shared_idx);
  for (int i = 0; i < row_tile; i++) {
    Type elemType = root_elem_types[i];
    Type shuffleElemType;
    if (failed(getShuffleElemType(b, elemType, &shuffleElemType))) {
      return failure();
    }
    assert((shared_mem_map_vec[i].find(op) != shared_mem_map_vec[i].end()) &&
           "shared_mem_map unexpected");
    auto shared_mem = shared_mem_map_vec[i][op];
    Value shared_mem_value =
        b.create<memref::LoadOp>(loc, shared_mem, shared_idx_vec);
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
  SmallVector<Value, 4> false_yield_values(row_reduction_ops.size() * row_tile);

  for (int i = 0; i < row_tile; i++) {
    false_yield_values[i] = init_value;
  }
  b.create<scf::YieldOp>(loc, false_yield_values);
  b.setInsertionPointToEnd(block);

  Value is_first_warp =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, warp_id,
                              b.create<arith::ConstantIndexOp>(loc, 0));
  scf::IfOp if_is_first_warp =
      b.create<scf::IfOp>(loc, /*resultTypes*/ TypeRange{}, is_first_warp,
                          /*hasElseRegion*/ false);
  b.setInsertionPointToStart(&if_is_first_warp.getThenRegion().front());
  auto acc_iter = if_lane_id_inbound.getResults().begin();
  if (failed(emitSecondRoundShuffleStitch(
          b, loc, op, acc_iter, thread_id_is_zero, row_ids, row_tile,
          block_size, smem_buffer, is_output, external_output_only))) {
    return failure();
  }
  b.setInsertionPointToEnd(block);
  return success();
}

LogicalResult initSkeletonGrpsAndCloneOps(
    lmhlo::FusionOp& fusion_op, FusionPattern& fusion_pattern,
    SmallVector<FusionPattern::SkeletonGroup>& skeleton_groups,
    LowerConfig& lower_config) {
  if (!fusion_pattern.getOrderedSkeletonGroups(skeleton_groups)) {
    return failure();
  }
  auto op_list = fusion_pattern.getOpList();
  auto sub_root_ops = fusion_pattern.getSubRootOps();
  DenseSet<Operation*> sub_roots(sub_root_ops.begin(), sub_root_ops.end());
  DenseSet<Operation*> regular_xroots = fusion_pattern.getRegularXroots();
  DenseSet<Operation*> irregular_xroots = fusion_pattern.getIrregularXroots();

  // Find ops in the group and reorder according to `op_list`.
  // { skeleton, group-ops-inorder }
  DenseMap<Operation*, SmallVector<Operation*>> skeleton_group_ops;
  for (auto& skeleton_group : skeleton_groups) {
    auto skeleton = skeleton_group.skeleton;
    // Find ops in the group and reorder according to `op_list`.
    DenseSet<Operation*> group_ops;
    fusion_pattern.findOpsOfSkeletonGroup(skeleton_group, group_ops);
    SmallVector<Operation*>& group_ops_inorder = skeleton_group_ops[skeleton];
    for (auto op : op_list) {
      if (group_ops.contains(op)) {
        group_ops_inorder.push_back(op);
      }
    }
  }

  // Clone ops in each group and build written flags in `lower_config`.
  // Note that for irregular-xroot occurs in several skeleton-groups, it
  // will not be cloned in the first occurance in a group, but will be
  // cloned in the following occurances. Meanwhile, the first occured group
  // will write the value of this irregular-xroot back, and others are not
  // responsible for the duty of writing output.
  DenseSet<Operation*> visited_irregular_xroots;
  auto allocClonedValue = [&](Value val) {
    OpBuilder b(fusion_op);
    Location loc = val.getLoc();
    auto ty = val.getType().cast<MemRefType>();
    auto shape = ty.getShape();
    SmallVector<Value> dims;
    for (int d = 0; d < ty.getRank(); ++d) {
      if (shape[d] == ShapedType::kDynamicSize) {
        dims.push_back(b.create<memref::DimOp>(loc, val, d));
      }
    }
    auto new_val = b.create<memref::AllocOp>(loc, ty, dims);
    return new_val;
  };
  DenseSet<Operation*> to_erase;
  auto& fused_block = fusion_op.region().front();
  auto& terminator = *fused_block.rbegin();
  OpBuilder builder(&terminator);
  Operation* last_op = nullptr;
  for (auto& skeleton_group : skeleton_groups) {
    auto skeleton = skeleton_group.skeleton;
    SmallVector<Operation*>& group_ops_inorder = skeleton_group_ops[skeleton];
    BlockAndValueMapping cloning_map;

    for (int64_t i = 0; i < group_ops_inorder.size(); i++) {
      auto op = group_ops_inorder[i];
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
        // For non-sub-root xroot op, it should be write back during lowering.
        if (!sub_roots.contains(op)) {
          lower_config.setWrittenBack(op);
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
                                      LowerConfig& lower_config) {
  auto root_ops = fusion_pattern.getRootOps();
  auto sub_root_ops = fusion_pattern.getSubRootOps();
  auto result_values = fusion_pattern.getResults();
  DenseSet<Operation*> roots(root_ops.begin(), root_ops.end());
  // DenseSet<Operation*> sub_roots(sub_root_ops.begin(), sub_root_ops.end());
  DenseSet<Value> results(result_values.begin(), result_values.end());
  auto op_list = fusion_pattern.getOpList();
  DenseSet<Operation*> op_set(op_list.begin(), op_list.end());
  SmallVector<Operation*> external_only_root_ops;
  for (auto value : fusion_pattern.getExternalOnlyResults()) {
    external_only_root_ops.push_back(fusion_pattern.findLastWriter(value));
  }
  DenseSet<Operation*> external_only_roots(external_only_root_ops.begin(),
                                           external_only_root_ops.end());
  auto parent = &(fusion_op.region().front());
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

  SmallVector<FusionPattern::SkeletonGroup> skeleton_groups;
  if (failed(initSkeletonGrpsAndCloneOps(fusion_op, fusion_pattern,
                                         skeleton_groups, lower_config))) {
    LLVM_DEBUG(llvm::dbgs() << "Fail to init skeleton groups or clone.\n");
    return failure();
  }

  auto load_config_func = [&](OpBuilder& b, Value value, ValueRange indices,
                              Value shm_buffer, int64_t row_tile) {
    auto loc = value.getLoc();
    assert(shm_buffer != nullptr);
    // shm index = linear_index % row_tile;
    auto shape = getShapeValues(&b, value);
    assert(shape.size() == indices.size());
    Value linear_index = calcLinearIndex(&b, loc, indices, shape);
    Value vec_size = b.create<arith::ConstantIndexOp>(loc, row_tile);
    Value smem_index = b.create<arith::RemUIOp>(loc, linear_index, vec_size);
    Value res =
        b.create<memref::LoadOp>(loc, shm_buffer, ValueRange({smem_index}));
    return res;
  };

  Location loc = dominant_op->getLoc();
  const int thread_per_block = getThreadPerBlock(dominant_op);

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

  // loop over (rows, threads)
  // Note that we know `shape_h` can be divided by `row_tile` exactly.
  Value vec_size = b.create<arith::ConstantIndexOp>(loc, row_tile);
  Value block_number = b.create<arith::DivUIOp>(loc, shape_h, vec_size);

  SmallVector<Value, 2> vars;
  scf::ParallelOp parallel_op = createParallelAndSetInsPt(
      b, loc, vars, {zero, zero}, {block_number, num_threads}, {one, one}, {});
  parallel_op.getBody()->clear();
  b.setInsertionPointToStart(parallel_op.getBody());
  Value var_m = vars[0];
  Value var_n = vars[1];

  for (auto& skeleton_group : skeleton_groups) {
    auto skeleton = skeleton_group.skeleton;
    Location loc = skeleton->getLoc();
    b.setInsertionPointToEnd(parallel_op.getBody());

    if (isRank2RowReduction(skeleton)) {
      Value out_value = cast<lmhlo::LmhloOp>(skeleton).getResultBuffer();
      Value result_shmem;
      result_shmem = createSharedMemoryForOp(b, loc, skeleton, row_tile);
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
      LowerConfig::SpecificLoader loader(load_config_func, result_shmem,
                                         row_tile);
      lower_config.setSpecificLoader(
          std::make_pair(fusion_op.getOperation(), out_value), loader);

      bool external_only = external_only_roots.contains(skeleton);
      if (failed(emitInBlockRowReduce(
              b, loc, parallel_op.getBody(), skeleton, var_m, var_n,
              thread_per_block, row_tile, result_shmem, shape_analysis,
              roots.contains(skeleton), external_only))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to emit InBlockRowReduce for: "
                                << *skeleton << "\n");
        return failure();
      }
      // TODO: check whether the xroots in this group are required by other
      // groups. If not, no barrier.
      bool require_barrier =
          !external_only ||
          llvm::any_of(skeleton_group.root_member_list, [&](Operation* op) {
            return !external_only_roots.contains(op);
          });

      if (require_barrier) {
        b.create<gpu::BarrierOp>(loc);
      }
      // TODO: replace load_hook with DynamicUpdateSliceOp.
    } else {
      Value out_value = cast<lmhlo::LmhloOp>(skeleton).getResultBuffer();
      auto tile_info = tile_plan.find(out_value);
      if (tile_info == tile_plan.end()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Failed to find tile info: " << *skeleton << "\n");
        return failure();
      }
      Value row_base = b.create<arith::MulIOp>(loc, var_m, vec_size);
      SmallVector<Value> row_ids(row_tile);
      for (int i = 0; i < row_tile; i++) {
        row_ids[i] = b.create<arith::AddIOp>(
            loc, row_base, b.create<arith::ConstantIndexOp>(loc, i));
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
      Value var_j = nullptr;
      scf::ForOp for_op_j =
          createLoopAndSetInsPt(b, loc, var_j,
                                /*lb*/ var_n, /*ub*/ tiled_linear,
                                /*step*/ num_threads, {});
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
      if (!external_only_roots.contains(skeleton)) {
        // TODO: use __threadfence_block instead.
        b.create<gpu::BarrierOp>(loc);
      }
    }
  }

  b.setInsertionPointToEnd(parallel_op.getBody());
  b.create<scf::YieldOp>(loc, ValueRange({}));

  for (auto& skeleton_group : skeleton_groups) {
    auto skeleton = skeleton_group.skeleton;
    skeleton->erase();
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
                                LowerConfig& lower_config) {
  auto fusion_op = cast<lmhlo::FusionOp>(fusion);
  assert(fusion_op);

  FusionPattern fusion_pattern(fusion_op, shape_analysis);
  {  // Update in-fusio shape equality information.
    DenseSet<Value> values_in_fusion;
    auto operands = fusion_pattern.getOperands();
    values_in_fusion.insert(operands.begin(), operands.end());
    auto internal_results = fusion_pattern.getInternalResults();
    values_in_fusion.insert(internal_results.begin(), internal_results.end());
    auto results = fusion_pattern.getResults();
    values_in_fusion.insert(results.begin(), results.end());
    shape_analysis->buildEqualShapesInFusion(fusion, values_in_fusion);
  }
  if (false) {  // For debugging.
    createPrintFusionParams(fusion_op, fusion_pattern);
  }

  auto root_ops = fusion_pattern.getRootOps();
  auto fused_block = &(fusion_op.region().front());
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
      const int vector_size = getVectorizationHint(dominant_op);
      LogicalResult r = success();
      if (row_reduction_schedule == DISC_ONE_ROUND_SHUFFLE_ROW_REDUCE) {
        r = lowerWithScheduleRowReduction<DISC_ONE_ROUND_SHUFFLE_ROW_REDUCE>(
            root_ops, dominant_op, fused_block, shape_analysis, vector_size);
      } else {
        r = lowerWithScheduleRowReduction<DISC_TWO_ROUND_SHUFFLE_ROW_REDUCE>(
            root_ops, dominant_op, fused_block, shape_analysis, vector_size);
      }
      if (failed(r)) {
        return dominant_op->emitError()
               << "failed to lower row-reduction loops";
      }
    } break;

    case FusionType::kColReduction: {
      const int col_reduction_schedule =
          getColReductionScheduleHint(dominant_op);
      LogicalResult r = success();
      if (col_reduction_schedule == DISC_TILE_W8_H32) {
        r = lowerWithScheduleColReductionBlockTileSchedule<8, 32>(
            root_ops, dominant_op, fused_block);
      } else if (col_reduction_schedule == DISC_TILE_W8_H16) {
        r = lowerWithScheduleColReductionBlockTileSchedule<8, 16>(
            root_ops, dominant_op, fused_block);
      } else {
        r = lowerWithScheduleColReductionBlockTileSchedule<8, 8>(
            root_ops, dominant_op, fused_block);
      }
      if (failed(r)) {
        return dominant_op->emitError()
               << "failed to lower col-reduction loops";
      }
    } break;

    case FusionType::kLoop: {
      const int vector_size = getVectorizationHint(dominant_op);
      if (failed(lowerWithScheduleLoop(root_ops, dominant_op, fused_block,
                                       /*non_fusion*/ false,
                                       /*parallel_loop*/ true, shape_analysis,
                                       vector_size))) {
        return dominant_op->emitError() << "failed to lower to loops";
      }
    } break;

    case FusionType::kStitch: {
      const int vector_size = getVectorizationHint(dominant_op);
      if (failed(lowerWithScheduleStitch(fusion_op, fusion_pattern,
                                         shape_analysis, vector_size,
                                         lower_config))) {
        return dominant_op->emitError() << "failed to lower kStitch fusion.";
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
    int64_t dimension = op.dimension();
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
  int64_t dimension = concat.dimension();
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
  Value result = cast<lmhlo::LmhloOp>(dominant_op).getResultBuffer();
  int64_t rank = result.getType().cast<MemRefType>().getRank();

  if (!multi_dim_loop || !rank || !parallel_loop) {
    return lowerWithScheduleLoop(root_ops, dominant_op, parent, non_fusion,
                                 parallel_loop, shape_analysis);
  }

  if (non_fusion && isLargeConcatOp(dominant_op)) {
    return lowerWithScheduleLargeConcatCPU(root_ops, dominant_op, parent,
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
  upperBound = b.create<SelectOp>(loc, pred, upperBound, innerMostDimSize);

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
  auto dimensions = reduce.dimensions().getValues<int64_t>();
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
  AccumulatorFactory accumFactory = getFactory(b, loc, reduce.body());
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
  auto fused_block = &(fusion_op.region().front());
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
class DiscLhloLegalizeRootsToParallelLoops
    : public DiscLhloLegalizeRootsToParallelLoopsPassBase<
          DiscLhloLegalizeRootsToParallelLoops> {
  void runOnFunction() override {
    auto func = getFunction();

    ShapeAnalysis shape_analysis(func);
    shape_analysis.run();

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
      if (isStitchFusion(op) && !disc_ral::isOnGpu(op)) return;
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
      if (failed(HandleGpuFusionOp(b, fusion, &shape_analysis, lower_config))) {
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
      OwningRewritePatternList patterns(context);
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
        fusion.region().walk([&](lmhlo::LmhloOp op) {
          if (isa<lmhlo::TerminatorOp>(op)) {
            return;
          }
          if (isa<lmhlo::ConstOp>(op)) {
            // TODO(disc): Check the ConstOp is from ReduceOp
            to_be_removed.push_back(op);
            return;
          }
          op.emitError("unexpected remaining operation in a FusionOp");
          signalPassFailure();
        });
      });
      for (auto op : to_be_removed) {
        op->erase();
      }
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>>
createDiscLhloLegalizeRootsToParallelLoopsPass() {
  return std::make_unique<DiscLhloLegalizeRootsToParallelLoops>();
}

}  // namespace disc_ral
}  // namespace mlir
