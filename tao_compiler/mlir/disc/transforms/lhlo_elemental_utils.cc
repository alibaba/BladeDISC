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

// This file provides basic utilities for the elemental lowering of
// each node

#include "tensorflow/compiler/mlir/disc/transforms/lhlo_elemental_utils.h"

#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"

using mlir::memref::DimOp;
using mlir::memref::LoadOp;
using mlir::memref::StoreOp;

namespace mlir {
namespace disc_ral {

Value createLoadOrUseCachedValue(Location loc, OpBuilder* b, Value memref,
                                 ValueRange indices,
                                 OpBuilder::InsertPoint insert_point) {
  // Check if there are any cached value that can be reused,
  // within the current Block. Alternatively we can do this for
  // all the Blocks that dominant this Block, but that will be
  // complicated anyway.
  std::vector<StoreOp> store_ops;
  insert_point.getBlock()->walk(
      insert_point.getBlock()->begin(), insert_point.getPoint(),
      [&](StoreOp store_op) {
        if (store_op.getOperation()->getBlock() != insert_point.getBlock())
          return;
        if ((store_op.getMemRef() == memref) &&
            (store_op.getIndices() == indices))
          store_ops.emplace_back(store_op);
      });
  if (!store_ops.empty()) return store_ops[0].getOperand(0);
  int rank = memref.getType().cast<MemRefType>().getRank();
  return rank > 0 ? b->create<LoadOp>(loc, memref, indices)
                  : b->create<LoadOp>(loc, memref);
}

DenseSet<Operation*> NoLoaderUser(SmallVectorImpl<Operation*>& ops) {
  SmallVector<Operation*, 4> worklist;
  DenseSet<Operation*> has_loader_ops;
  for (Operation* op : ops) {
    Value memref = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    if (memref == nullptr) continue;
    for (Operation* user : getValueUsersInFusionLike(memref, op)) {
      if (isa<memref::LoadOp>(user)) {
        worklist.push_back(op);
        has_loader_ops.insert(op);
      }
    }
  }

  while (!worklist.empty()) {
    Operation* op = worklist.pop_back_val();
    int num_operands = op->getNumOperands();
    for (int i = 0; i < num_operands - 1; ++i) {
      Value memref = op->getOperand(i);
      for (Operation* user : getValueUsersInFusionLike(memref, op)) {
        if ((!isa<lmhlo::LmhloOp>(user)) || has_loader_ops.count(user))
          continue;
        if (isSameUnderlineBuffer(cast<lmhlo::LmhloOp>(user).getResultBuffer(),
                                  memref)) {
          worklist.push_back(user);
          has_loader_ops.insert(user);
        }
      }
    }
  }

  DenseSet<Operation*> no_loader_ops;
  for (Operation* op : ops)
    if (!has_loader_ops.count(op)) no_loader_ops.insert(op);
  return no_loader_ops;
}

void cleanUnusedLhloOps(Block* parent) {
  SmallVector<Operation*, 4> lhlo_ops;
  for (Operation& op : parent->getOperations()) {
    if (op.getDialect() == op.getContext()->getLoadedDialect("lmhlo") &&
        (!isa<lmhlo::TerminatorOp>(op)))
      lhlo_ops.push_back(&op);
  }
  const DenseSet<Operation*>& no_loader_user = NoLoaderUser(lhlo_ops);
  for (auto* lhlo_op : no_loader_user) lhlo_op->erase();
}

// TODO: only support the reduce body in the form of
// one lhlo_instruction and one terminator
Operation* getReduceOperator(Region& body) {
  Operation* calc_op = nullptr;
  int64_t num_calc_ops = 0;
  body.walk([&](Operation* op) {
    if (isa<memref::AllocOp>(op) || isa<lmhlo::CopyOp>(op) ||
        isa<lmhlo::TerminatorOp>(op)) {
      return;
    }
    num_calc_ops++;
    calc_op = op;
  });
  assert(num_calc_ops == 1 && "unexpected reduce body");
  return calc_op;
}

AccumulatorFactory getFactory(OpBuilder& b, Location loc, Region& body) {
  return AccumulatorFactory([&](Value lhs, Value rhs) {
    auto calc_op = getReduceOperator(body);
    SmallVector<Value, 4> operand_values;
    operand_values.push_back(lhs);
    operand_values.push_back(rhs);
    assert(calc_op && "error in calc_op");
    auto num_operands = calc_op->getNumOperands();
    auto result_type = calc_op->getOperand(num_operands - 1).getType();
    auto result_elem_type = result_type.cast<MemRefType>().getElementType();
    if (isa<lmhlo::AddOp>(calc_op)) {
      return lmhlo::HloOpToStdScalarOp::map<lmhlo::AddOp>(
          llvm::cast<lmhlo::AddOp>(calc_op), result_elem_type, operand_values,
          &b);
    } else if (isa<lmhlo::MulOp>(calc_op)) {
      return lmhlo::HloOpToStdScalarOp::map<lmhlo::MulOp>(
          llvm::cast<lmhlo::MulOp>(calc_op), result_elem_type, operand_values,
          &b);
    } else if (isa<lmhlo::MaxOp>(calc_op)) {
      return lmhlo::HloOpToStdScalarOp::map<lmhlo::MaxOp>(
          llvm::cast<lmhlo::MaxOp>(calc_op), result_elem_type, operand_values,
          &b);
    } else if (isa<lmhlo::MinOp>(calc_op)) {
      return lmhlo::HloOpToStdScalarOp::map<lmhlo::MinOp>(
          llvm::cast<lmhlo::MinOp>(calc_op), result_elem_type, operand_values,
          &b);
    } else if (isa<lmhlo::OrOp>(calc_op)) {
      return lmhlo::HloOpToStdScalarOp::map<lmhlo::OrOp>(
          llvm::cast<lmhlo::OrOp>(calc_op), result_elem_type, operand_values,
          &b);
    } else if (isa<lmhlo::AndOp>(calc_op)) {
      return lmhlo::HloOpToStdScalarOp::map<lmhlo::AndOp>(
          llvm::cast<lmhlo::AndOp>(calc_op), result_elem_type, operand_values,
          &b);
    } else {
      assert(false && "unexpected reduce operation");
      return lmhlo::HloOpToStdScalarOp::map<lmhlo::AddOp>(
          llvm::cast<lmhlo::AddOp>(calc_op), result_elem_type, operand_values,
          &b);
    }
  });
}

template <typename LHLO_OpTy>
Value elementalLower(OpBuilder* b, Location loc, LHLO_OpTy op,
                     ValueRange output_index, bool check_cache);

template <>
Value elementalLower<lmhlo::SliceOp>(OpBuilder* b, Location loc,
                                     lmhlo::SliceOp op, ValueRange output_index,
                                     bool check_cache) {
  int rank = output_index.size();

  SmallVector<Value> input_index;
  for (int dim = 0; dim < rank; ++dim) {
    // for each dim, output[..a..] = input[..a * a_stride + a_start..]
    Value start_index = b->create<ConstantIndexOp>(
        loc, op.start_indices().getValue<int64_t>(dim));
    Value stride =
        b->create<ConstantIndexOp>(loc, op.strides().getValue<int64_t>(dim));
    auto input_dim = b->create<AddIOp>(
        loc, b->create<MulIOp>(loc, output_index[dim], stride), start_index);
    input_index.push_back(input_dim);
  }

  Value operand_memref = op->getOperand(0);
  if (!check_cache) return b->create<LoadOp>(loc, operand_memref, input_index);
  return createLoadOrUseCachedValue(loc, b, operand_memref, input_index,
                                    b->saveInsertionPoint());
}

template <>
Value elementalLower<lmhlo::RealDynamicSliceOp>(OpBuilder* b, Location loc,
                                                lmhlo::RealDynamicSliceOp op,
                                                ValueRange output_index,
                                                bool check_cache) {
  Value start_indices_memref = op->getOperand(1);
  Value strides_memref = op->getOperand(3);
  int rank = output_index.size();
  SmallVector<Value, 4> input_index;
  for (int dim = 0; dim < rank; ++dim) {
    SmallVector<Value, 4> dim_index;
    dim_index.push_back(b->create<ConstantIndexOp>(loc, dim));
    auto start_index_load =
        b->create<LoadOp>(loc, start_indices_memref, ValueRange{dim_index});
    auto start_index = mayConvertToIndexType(start_index_load, b, loc);
    auto stride_load =
        b->create<LoadOp>(loc, strides_memref, ValueRange{dim_index});
    auto stride = mayConvertToIndexType(stride_load, b, loc);
    // input_dim = out_dim * stride + start_index
    auto input_dim = b->create<AddIOp>(
        loc, b->create<MulIOp>(loc, output_index[dim], stride), start_index);
    input_index.push_back(input_dim);
  }

  Value operand_memref = *(op->getOperands().begin());

  if (!check_cache) return b->create<LoadOp>(loc, operand_memref, input_index);
  return createLoadOrUseCachedValue(loc, b, operand_memref, input_index,
                                    b->saveInsertionPoint());
}

namespace {

template <typename T>
Value elementalLowerImplForBroadcastInDimOps(OpBuilder* b, Location loc,
                                             T broadcast_in_dim,
                                             ValueRange output_index,
                                             bool check_cache) {
  auto broadcast_dimensions =
      broadcast_in_dim.broadcast_dimensions().template getValues<int64_t>();
  int out_rank = output_index.size();
  Value operand_memref = broadcast_in_dim->getOperand(0);
  Value result_memref =
      cast<lmhlo::LmhloOp>(broadcast_in_dim.getOperation()).getResultBuffer();
  SmallVector<Value, 4> input_index;
  for (int64_t dim = 0; dim < out_rank; ++dim) {
    auto it = std::find(broadcast_dimensions.begin(),
                        broadcast_dimensions.end(), dim);

    bool is_broadcast_dim = (it != broadcast_dimensions.end());
    if (is_broadcast_dim) {
      int input_dim = std::distance(broadcast_dimensions.begin(), it);
      int64_t static_dim_size =
          operand_memref.getType().cast<MemRefType>().getShape()[input_dim];
      if (static_dim_size == 1) {
        // we know this dim is to be broadcasted at compile time
        auto zero = b->create<ConstantOp>(
            loc, b->getIndexType(), b->getIntegerAttr(b->getIndexType(), 0));
        input_index.push_back(zero);
      } else if (static_dim_size == ShapedType::kDynamicSize) {
        // we are not sure if this dim is to be broadcasted at compile time.
        // To enable more optimization opportunities when dim sizes of operand
        // value and output value are the same SSA value.
        auto dim_size = b->create<DimOp>(loc, operand_memref, input_dim);
        auto output_dim_size = b->create<DimOp>(loc, result_memref, dim);
        auto zero = b->create<ConstantOp>(
            loc, b->getIndexType(), b->getIntegerAttr(b->getIndexType(), 0));
        auto dim_size_is_equal = b->create<CmpIOp>(loc, CmpIPredicate::eq,
                                                   dim_size, output_dim_size);
        input_index.push_back(b->create<mlir::SelectOp>(
            loc, dim_size_is_equal, output_index[dim], zero));
      } else {
        // we know this dim is not to be broadcasted at compile time
        input_index.push_back(output_index[dim]);
      }
    }
  }

  if (!check_cache) {
    int rank = operand_memref.getType().dyn_cast<MemRefType>().getRank();
    return (rank > 0) ? b->create<LoadOp>(loc, operand_memref, input_index)
                      : b->create<LoadOp>(loc, operand_memref, ValueRange());
  }
  return createLoadOrUseCachedValue(loc, b, operand_memref, input_index,
                                    b->saveInsertionPoint());
}

}  // namespace

template <>
Value elementalLower<lmhlo::DynamicBroadcastInDimOp>(
    OpBuilder* b, Location loc, lmhlo::DynamicBroadcastInDimOp op,
    ValueRange output_index, bool check_cache) {
  return elementalLowerImplForBroadcastInDimOps(b, loc, op, output_index,
                                                check_cache);
}

template <>
Value elementalLower<lmhlo::BroadcastInDimOp>(OpBuilder* b, Location loc,
                                              lmhlo::BroadcastInDimOp op,
                                              ValueRange output_index,
                                              bool check_cache) {
  return elementalLowerImplForBroadcastInDimOps(b, loc, op, output_index,
                                                check_cache);
}

template <>
Value elementalLower<lmhlo::BroadcastOp>(OpBuilder* b, Location loc,
                                         lmhlo::BroadcastOp op,
                                         ValueRange output_index,
                                         bool check_cache) {
  Value operand_memref = op->getOperand(0);
  auto operand_type = operand_memref.getType().dyn_cast<MemRefType>();
  int operand_rank = operand_type.getRank();

  int result_rank = output_index.size();

  SmallVector<Value> input_index;
  // broadcast op insert dims in the front, output[a,b,A,B] = input[A,B]
  for (int i = result_rank - operand_rank; i < result_rank; ++i) {
    input_index.push_back(output_index[i]);
  }

  if (!check_cache)
    return operand_rank > 0
               ? b->create<LoadOp>(loc, operand_memref, input_index)
               : b->create<LoadOp>(loc, operand_memref, ValueRange());
  return createLoadOrUseCachedValue(loc, b, operand_memref, input_index,
                                    b->saveInsertionPoint());
}

template <>
Value elementalLower<lmhlo::ReshapeOp>(OpBuilder* b, Location loc,
                                       lmhlo::ReshapeOp op,
                                       ValueRange output_index,
                                       bool check_cache) {
  Value operand_memref = op->getOperand(0);
  Value output_memref = op->getOperand(1);

  // calculate offset in output
  auto output_shape = getShapeValues(b, output_memref);
  Value linear_index = calcLinearIndex(b, loc, output_index, output_shape);
  // transform offset to input index
  auto operand_shape = getShapeValues(b, operand_memref);
  auto operand_index = calcMultiDimIndex(b, loc, linear_index, operand_shape);

  if (!check_cache)
    return b->create<LoadOp>(loc, operand_memref, operand_index);
  return createLoadOrUseCachedValue(loc, b, operand_memref, operand_index,
                                    b->saveInsertionPoint());
}

template <>
Value elementalLower<lmhlo::DynamicReshapeOp>(OpBuilder* b, Location loc,
                                              lmhlo::DynamicReshapeOp op,
                                              ValueRange output_index,
                                              bool check_cache) {
  Value operand_memref = op->getOperand(0);
  Value output_memref = op->getOperand(2);

  // We get the shape of output memref explicitly instead of using the shape
  // stored in operand #1. The latter one may prevent linearize/delinerize
  // elimination.
  auto output_shape = getShapeValues(b, output_memref);
  Value linear_index = calcLinearIndex(b, loc, output_index, output_shape);
  // transform offset to input index
  auto operand_shape = getShapeValues(b, operand_memref);
  auto operand_index = calcMultiDimIndex(b, loc, linear_index, operand_shape);

  if (!check_cache)
    return b->create<LoadOp>(loc, operand_memref, operand_index);
  return createLoadOrUseCachedValue(loc, b, operand_memref, operand_index,
                                    b->saveInsertionPoint());
}

// There is no NotOp in std dialect, thus we provide a basic implementation.
template <>
Value elementalLower<lmhlo::NotOp>(OpBuilder* b, Location loc, lmhlo::NotOp op,
                                   ValueRange output_index, bool check_cache) {
  Value operand_memref = op->getOperand(0);
  auto operand_ty = operand_memref.getType().cast<MemRefType>();

  Value operandValue;
  if (!check_cache) {
    operandValue = b->create<LoadOp>(loc, operand_memref, output_index);
  } else {
    operandValue = createLoadOrUseCachedValue(
        loc, b, operand_memref, output_index, b->saveInsertionPoint());
  }

  Value falseValue;
  if (operand_ty.getElementType().isIndex()) {
    falseValue = b->create<ConstantIndexOp>(loc, 0);
  } else {
    falseValue = b->create<ConstantIntOp>(loc, 0, operand_ty.getElementType());
  }

  return b->create<CmpIOp>(loc, CmpIPredicate::eq, operandValue, falseValue);
}

template <>
Value elementalLower<lmhlo::TransposeOp>(OpBuilder* b, Location loc,
                                         lmhlo::TransposeOp op,
                                         ValueRange output_index,
                                         bool check_cache) {
  Value operand_memref = op->getOperand(0);
  SmallVector<int64_t> permutation(op.permutation().getValues<int64_t>());
  int rank = permutation.size();

  SmallVector<Value> operand_index(rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    auto it = std::find(permutation.begin(), permutation.end(), dim);
    assert(it != permutation.end() && "invalid permutation element");
    operand_index[permutation[dim]] = output_index[dim];
  }

  if (!check_cache)
    return b->create<LoadOp>(loc, operand_memref, operand_index);
  return createLoadOrUseCachedValue(loc, b, operand_memref, operand_index,
                                    b->saveInsertionPoint());
}

template <>
Value elementalLower<lmhlo::ReverseOp>(OpBuilder* b, Location loc,
                                       lmhlo::ReverseOp op,
                                       ValueRange output_index,
                                       bool check_cache) {
  // input_shape = op->getOperand(0).shape()
  // for dim = 0 -> rank:
  //   if dim in axis:
  //     operand_index[dim] = input_shape[dim] - output_index[dim] - 1;
  //   else:
  //     operand_index[dim] = output_index[dim]
  Value operand_memref = op->getOperand(0);
  auto axis = op.dimensions().getValues<int64_t>();
  int rank = output_index.size();
  SmallVector<Value> operand_index(rank);
  Value one = b->create<ConstantIndexOp>(loc, 1);
  auto input_shape = getShapeValues(b, operand_memref);
  for (int64_t dim = 0; dim < rank; ++dim) {
    auto it = std::find(axis.begin(), axis.end(), dim);
    if (it != axis.end()) {
      auto shape_value = b->create<SubIOp>(loc, input_shape[dim], one);
      auto output_dim = b->create<SubIOp>(loc, shape_value, output_index[dim]);
      operand_index[dim] = output_dim;
    } else {
      operand_index[dim] = output_index[dim];
    }
  }
  if (!check_cache)
    return b->create<LoadOp>(loc, operand_memref, operand_index);
  return createLoadOrUseCachedValue(loc, b, operand_memref, operand_index,
                                    b->saveInsertionPoint());
}

template <>
Value elementalLower<lmhlo::DynamicPadOp>(OpBuilder* b, Location loc,
                                          lmhlo::DynamicPadOp op,
                                          ValueRange output_index,
                                          bool check_cache) {
  Value operand_memref = *(op->getOperands().begin());
  Value padding_value_memref = *(op->getOperands().begin() + 1);
  Value edge_padding_low_memref = *(op->getOperands().begin() + 2);
  Value interior_padding_memref = *(op->getOperands().begin() + 4);
  int rank = output_index.size();
  SmallVector<Value, 4> input_index;
  Value in_bound = b->create<ConstantIntOp>(loc, 1, 1);
  Value one = b->create<ConstantIndexOp>(loc, 1);
  Value zero = b->create<ConstantIndexOp>(loc, 0);
  // for each dim i:
  //   x = output_dim[i] - edge_padding_low[i]
  //   y = x % (interior_padding[i] + 1)
  //   z = x / (interior_padding[i] + 1)
  //   in_bound = in_bound &&
  //      (x >= 0) &&
  //      (y == 0) &&
  //      (z < input_shape[i])
  for (int dim = 0; dim < rank; ++dim) {
    SmallVector<Value, 4> dim_const;
    dim_const.push_back(b->create<ConstantIndexOp>(loc, dim));
    Value edge_padding_low =
        b->create<LoadOp>(loc, edge_padding_low_memref, dim_const);
    edge_padding_low = mayConvertToIndexType(edge_padding_low, b, loc);
    Value interior_padding =
        b->create<LoadOp>(loc, interior_padding_memref, dim_const);
    interior_padding = mayConvertToIndexType(interior_padding, b, loc);
    auto x = b->create<SubIOp>(loc, output_index[dim], edge_padding_low);
    auto interior_padding_p1 = b->create<AddIOp>(loc, interior_padding, one);
    auto y = b->create<UnsignedRemIOp>(loc, x, interior_padding_p1);
    auto z = b->create<UnsignedDivIOp>(loc, x, interior_padding_p1);
    in_bound = b->create<mlir::AndOp>(
        loc, in_bound, b->create<CmpIOp>(loc, CmpIPredicate::sge, x, zero));
    in_bound = b->create<mlir::AndOp>(
        loc, in_bound, b->create<CmpIOp>(loc, CmpIPredicate::eq, y, zero));

    in_bound = b->create<mlir::AndOp>(
        loc, in_bound,
        b->create<CmpIOp>(loc, CmpIPredicate::slt, z,
                          disc_ral::getDimSizeValue(b, operand_memref, dim)));
    input_index.push_back(z);
  }

  // if (in_bounds) {
  //   ret_value = operand0[index];  // source
  // } else {
  //   ret_value = operand1;        // padding
  // }
  SmallVector<Type, 4> result_types;
  result_types.push_back(
      operand_memref.getType().cast<MemRefType>().getElementType());
  auto if_inbound_op =
      b->create<scf::IfOp>(loc, result_types, in_bound, /*hasElseRegion*/ true);
  if_inbound_op.thenRegion().front().clear();
  if_inbound_op.elseRegion().front().clear();
  b->setInsertionPointToEnd(&if_inbound_op.thenRegion().front());
  auto ret_value =
      check_cache
          ? createLoadOrUseCachedValue(loc, b, operand_memref, input_index,
                                       b->saveInsertionPoint())
          : b->create<LoadOp>(loc, operand_memref, input_index);
  b->create<scf::YieldOp>(loc, ret_value);

  b->setInsertionPointToEnd(&if_inbound_op.elseRegion().front());
  Value padded_value =
      b->create<LoadOp>(loc, padding_value_memref, ValueRange());
  b->create<scf::YieldOp>(loc, padded_value);

  b->setInsertionPointAfter(if_inbound_op);
  return *(if_inbound_op.results().begin());
}

Value lowerGatherOpInternal(OpBuilder* b, Location loc, Operation* op,
                            ValueRange output_index, bool check_cache) {
  auto gather = dyn_cast<lmhlo::GatherOp>(op);
  auto d_gather = dyn_cast<lmhlo::DynamicGatherOp>(op);
  assert((gather || d_gather) && "unexpected opcode");
  auto operand = gather ? gather.operand() : d_gather.operand();
  auto start_indices =
      gather ? gather.start_indices() : d_gather.start_indices();
  auto dimension_numbers =
      gather ? gather.dimension_numbers() : d_gather.dimension_numbers();
  auto operand_ty = operand.getType().dyn_cast<MemRefType>();
  auto start_indices_ty = start_indices.getType().dyn_cast<MemRefType>();
  auto result = op->getOperand(op->getNumOperands() - 1);
  auto result_ty = result.getType().dyn_cast<MemRefType>();
  assert(operand_ty && "unexpected operand type for GatherOp");
  assert(start_indices_ty && "unexpected start_indices type for GatherOp");
  assert(result_ty && "unexpected result type for GatherOp");
  int64_t operand_rank = operand_ty.getRank();
  int64_t start_indices_rank = start_indices_ty.getRank();
  int64_t result_rank = result_ty.getRank();
  SmallVector<int64_t, 4> collapsed_slice_dims(
      dimension_numbers.collapsed_slice_dims().getValues<int64_t>());
  SmallVector<int64_t, 4> offset_dims(
      dimension_numbers.offset_dims().getValues<int64_t>());
  int64_t index_vector_dim =
      dimension_numbers.index_vector_dim().getValue().getSExtValue();
  SmallVector<int64_t, 4> start_index_map(
      dimension_numbers.start_index_map().getValues<int64_t>());

  // get initial operand_index
  SmallVector<Value, 4> operand_index;
  operand_index.reserve(operand_rank);
  SmallVector<int64_t, 4> operand_to_output_dim(operand_rank, -1);
  for (int64_t i = 0, operand_index_dim = 0; i < operand_rank; ++i) {
    bool is_collapsed_dim =
        (std::find(collapsed_slice_dims.begin(), collapsed_slice_dims.end(),
                   i) != collapsed_slice_dims.end());
    if (is_collapsed_dim) {
      operand_index.push_back(b->create<ConstantIndexOp>(loc, 0));
    } else {
      int64_t output_window_dim = offset_dims[operand_index_dim++];
      operand_to_output_dim[i] = output_window_dim;
      operand_index.push_back(output_index[output_window_dim]);
    }
  }

  // get index to 'start_indices'
  SmallVector<Value, 4> gather_index_index;
  gather_index_index.reserve(start_indices_rank);
  for (int64_t i = 0; i < result_rank; ++i) {
    bool is_in_offset_dims = (std::find(offset_dims.begin(), offset_dims.end(),
                                        i) != offset_dims.end());
    if (!is_in_offset_dims) {
      gather_index_index.push_back(output_index[i]);
    }
  }
  if (gather_index_index.size() != start_indices_rank) {
    gather_index_index.insert(gather_index_index.begin() + index_vector_dim,
                              nullptr);
  }

  auto add_to_operand_index = [&](Value index_component, int64_t dim) {
    int64_t operand_dim = start_index_map[dim];
    int64_t output_dim = operand_to_output_dim[operand_dim];
    Value output_dim_size = nullptr;
    if (output_dim == -1) {
      output_dim_size = b->create<ConstantIndexOp>(loc, 1);
    } else {
      output_dim_size = b->create<DimOp>(loc, result, output_dim);
    }
    Value largest_valid_start_index = b->create<IndexCastOp>(
        loc, index_component.getType(),
        b->create<SubIOp>(loc, b->create<DimOp>(loc, operand, operand_dim),
                          output_dim_size));
    auto zero = b->create<ConstantIntOp>(
        loc, 0, index_component.getType().cast<IntegerType>().getWidth());
    auto max_with_zero = b->create<mlir::SelectOp>(
        loc, b->create<CmpIOp>(loc, CmpIPredicate::sge, zero, index_component),
        zero, index_component);
    auto gather_dim_component_extended_inbound = b->create<mlir::SelectOp>(
        loc,
        b->create<CmpIOp>(loc, CmpIPredicate::slt, max_with_zero,
                          largest_valid_start_index),
        max_with_zero, largest_valid_start_index);

    operand_index[operand_dim] = b->create<AddIOp>(
        loc, operand_index[operand_dim],
        mayConvertToIndexType(gather_dim_component_extended_inbound, b, loc));
  };

  // update operand_index
  if (start_indices_rank == index_vector_dim) {
    Value gather_dim_component =
        check_cache ? createLoadOrUseCachedValue(loc, b, start_indices,
                                                 gather_index_index,
                                                 b->saveInsertionPoint())
                    : b->create<LoadOp>(loc, start_indices, gather_index_index);
    add_to_operand_index(gather_dim_component, 0);
  } else {
    int64_t index_vector_size = start_indices_ty.getDimSize(index_vector_dim);
    assert((index_vector_size != ShapedType::kDynamicSize) &&
           "dynamic index_vector_dim size for GatherOp is unexpected");
    for (int64_t i = 0; i < index_vector_size; ++i) {
      gather_index_index[index_vector_dim] = b->create<ConstantIndexOp>(loc, i);
      Value gather_dim_component =
          check_cache
              ? createLoadOrUseCachedValue(loc, b, start_indices,
                                           gather_index_index,
                                           b->saveInsertionPoint())
              : b->create<LoadOp>(loc, start_indices, gather_index_index);
      add_to_operand_index(gather_dim_component, i);
    }
  }
  return (check_cache
              ? createLoadOrUseCachedValue(loc, b, operand, operand_index,
                                           b->saveInsertionPoint())
              : b->create<LoadOp>(loc, operand, operand_index));
}

template <>
Value elementalLower<lmhlo::GatherOp>(OpBuilder* b, Location loc,
                                      lmhlo::GatherOp op,
                                      ValueRange output_index,
                                      bool check_cache) {
  return lowerGatherOpInternal(b, loc, op, output_index, check_cache);
}

template <>
Value elementalLower<lmhlo::DynamicGatherOp>(OpBuilder* b, Location loc,
                                             lmhlo::DynamicGatherOp op,
                                             ValueRange output_index,
                                             bool check_cache) {
  return lowerGatherOpInternal(b, loc, op, output_index, check_cache);
}

template <>
Value elementalLower<lmhlo::ConcatenateOp>(OpBuilder* b, Location loc,
                                           lmhlo::ConcatenateOp op,
                                           ValueRange output_index,
                                           bool check_cache) {
  size_t axis = op.dimension();
  size_t rank = output_index.size();

  auto num_input_operands = op.getNumOperands() - 1;
  auto zero = b->create<ConstantIndexOp>(loc, 0);

  SmallVector<Value> axis_dim_ranges;
  axis_dim_ranges.push_back(zero);
  for (int i = 0; i < num_input_operands; ++i) {
    axis_dim_ranges.push_back(
        b->create<AddIOp>(loc, getDimSizeValue(b, op.getOperand(i), axis),
                          axis_dim_ranges.back()));
  }

  SmallVector<Type, 4> if_result_types;
  auto result_type = op.getOperand(0).getType();
  auto result_elem_type = result_type.cast<MemRefType>().getElementType();
  if_result_types.push_back(result_elem_type);

  SmallVector<Value, 4> input_index(rank);
  for (size_t dim = 0; dim < rank; ++dim) {
    if (dim != axis) {
      input_index[dim] = output_index[dim];
    }
  }

  Value zero_element;
  if (result_elem_type.isF16() || result_elem_type.isF32() ||
      result_elem_type.isF64()) {
    zero_element = b->create<ConstantOp>(loc, result_elem_type,
                                         b->getFloatAttr(result_elem_type, 0));
  } else if (result_elem_type.isSignlessInteger() ||
             result_elem_type.isSignedInteger() ||
             result_elem_type.isUnsignedInteger()) {
    zero_element = b->create<ConstantOp>(
        loc, result_elem_type, b->getIntegerAttr(result_elem_type, 0));
  } else {
    assert(false && "unexpected concatenate element type");
  }

  // Below checks the output index along dimension axis,
  // to identify which input memref should be load
  //
  // We have previously built a ranged list (named axis_dim_ranges)
  // for indexes among axis dimension, e.g.
  //   [0, n1, n2, ...]
  // which represents that:
  //   output index axis dim valued within range [0, n1), should be read from
  //   input 0.
  //   where: 0 stands for lower_bound_0, n1 stands for upper_bound_0
  //   ...
  //
  // Pseudocode:
  //   val = output_index[axis]
  //   T0 = if ((val >= lower_bound_0) && (val < upper_bound_0) {
  //     load input_0 element and store to output
  //   } else {
  //     yield(T1)
  //   }
  //   T1 = if ((val >= lower_bound_1) && (val < upper_bound_1) {
  //     load input_0 element and store to output
  //   } else {
  //     yield(T2)
  //   }
  //   ...
  //   Tn = if ((val >= lower_bound_n) && (val < upper_bound_n) {
  //     load input_n element and store to output
  //   } else {
  //     store 0 to output
  //   }
  //   return T0;

  auto out_idx = output_index[axis];
  SmallVector<scf::IfOp> if_inbound_ops(num_input_operands);
  for (int i = num_input_operands - 1; i >= 0; --i) {
    auto low_bound = axis_dim_ranges[i];
    auto up_bound = axis_dim_ranges[i + 1];
    auto gt_low_bound =
        b->create<CmpIOp>(loc, CmpIPredicate::uge, out_idx, low_bound);
    auto lt_up_bound =
        b->create<CmpIOp>(loc, CmpIPredicate::ult, out_idx, up_bound);
    auto in_bound = b->create<AndOp>(loc, gt_low_bound, lt_up_bound);
    if_inbound_ops[i] = b->create<scf::IfOp>(loc, if_result_types, in_bound,
                                             /*withElseRegion*/ true);
    if_inbound_ops[i].thenRegion().front().clear();
    if_inbound_ops[i].elseRegion().front().clear();

    b->setInsertionPointToEnd(&if_inbound_ops[i].thenRegion().front());
    input_index[axis] = b->create<SubIOp>(loc, out_idx, low_bound);
    auto operand_memref = op.getOperand(i);
    auto ret_value =
        check_cache
            ? createLoadOrUseCachedValue(loc, b, operand_memref, input_index,
                                         b->saveInsertionPoint())
            : b->create<LoadOp>(loc, operand_memref, input_index);
    b->create<scf::YieldOp>(loc, ret_value);

    b->setInsertionPointToEnd(&if_inbound_ops[i].elseRegion().front());
    if (i == num_input_operands - 1) {
      b->create<scf::YieldOp>(loc, zero_element);  // expect never used
    } else {
      b->create<scf::YieldOp>(loc, if_inbound_ops[i + 1].results());
    }
    b->setInsertionPointAfter(if_inbound_ops[i]);
  }
  b->setInsertionPointAfter(if_inbound_ops[0]);
  return *(if_inbound_ops[0].results().begin());
}

// There is no 'identityOp' in std dialect, thus we provide a basic
// implementation here as a workaround. This can be removed once std dialect
// supports CopyOp.
template <>
Value elementalLower<lmhlo::CopyOp>(OpBuilder* b, Location loc,
                                    lmhlo::CopyOp op, ValueRange output_index,
                                    bool check_cache) {
  Value operand_memref = op->getOperand(0);
  auto input_type = operand_memref.getType().dyn_cast<ShapedType>();
  assert(input_type && "expected operand having ShapedType");
  if (!check_cache) {
    return b->create<LoadOp>(loc, operand_memref, output_index);
  }
  return createLoadOrUseCachedValue(loc, b, operand_memref, output_index,
                                    b->saveInsertionPoint());
}

Value elementalLowerIota(OpBuilder* b, const Location& loc, Operation* op,
                         const ValueRange& output_index,
                         const MemRefType& result_ty) {
  auto result_element_ty = result_ty.getElementType();
  if (result_ty.getRank() == 0) {
    if (result_element_ty.dyn_cast<IntegerType>()) {
      return b->create<ConstantOp>(loc, result_element_ty,
                                   b->getIntegerAttr(result_element_ty, 0));
    } else if (result_element_ty.dyn_cast<IndexType>()) {
      return b->create<ConstantOp>(loc, result_element_ty, b->getIndexAttr(0));
    } else if (result_element_ty.dyn_cast<FloatType>()) {
      return b->create<ConstantOp>(loc, result_element_ty,
                                   b->getFloatAttr(result_element_ty, 0));
    } else {
      op->emitError("element_type of Iota/DynamicIotaOp not implemented");
      return Value(nullptr);
    }
  }
  int64_t iota_dimension = 0;
  if (isa<lmhlo::DynamicIotaOp>(op)) {
    auto dynamic_iota = mlir::dyn_cast<lmhlo::DynamicIotaOp>(op);
    iota_dimension = dynamic_iota.iota_dimension();
  } else if (isa<lmhlo::IotaOp>(op)) {
    auto iota = mlir::dyn_cast<lmhlo::IotaOp>(op);
    iota_dimension = iota.iota_dimension();
  }
  assert(iota_dimension < output_index.size() &&
         "iota_dimension exceeds rank of output_index");
  auto elem_index_linear = output_index[iota_dimension];
  Value result = nullptr;
  if (result_element_ty.dyn_cast<IntegerType>()) {
    result =
        b->create<mlir::IndexCastOp>(loc, elem_index_linear, result_element_ty);
  } else if (result_element_ty.dyn_cast<IndexType>()) {
    result = mayConvertToIndexType(elem_index_linear, b, loc);
  } else if (result_element_ty.dyn_cast<FloatType>()) {
    auto idx2int = mayConvertToIntegerType(elem_index_linear, b, loc);
    result = b->create<mlir::UIToFPOp>(loc, idx2int, result_element_ty);
  } else {
    op->emitError("element_type of Iota/DynamicIotaOp not implemented");
    return Value(nullptr);
  }
  return result;
}

template <>
Value elementalLower<lmhlo::DynamicIotaOp>(OpBuilder* b, Location loc,
                                           lmhlo::DynamicIotaOp op,
                                           ValueRange output_index,
                                           bool check_cache) {
  auto result_ty = op->getOperand(1).getType().dyn_cast<MemRefType>();
  assert(result_ty && "unexpected result type of DynamicIotaOp");
  return elementalLowerIota(b, loc, op.getOperation(), output_index, result_ty);
}

template <>
Value elementalLower<lmhlo::IotaOp>(OpBuilder* b, Location loc,
                                    lmhlo::IotaOp op, ValueRange output_index,
                                    bool check_cache) {
  auto result_ty = op->getOperand(0).getType().dyn_cast<MemRefType>();
  assert(result_ty && "unexpected result type of IotaOp");
  return elementalLowerIota(b, loc, op.getOperation(), output_index, result_ty);
}

template <>
Value elementalLower<lmhlo::ReduceOp>(OpBuilder* b, Location loc,
                                      lmhlo::ReduceOp op,
                                      ValueRange output_index,
                                      bool check_cache) {
  auto operand_memref = *(op->getOperands().begin());
  auto init_value_memref = *(op->getOperands().begin() + 1);
  auto init_value = b->create<LoadOp>(loc, init_value_memref);
  auto dimensions = op.dimensions().getValues<int64_t>();
  auto input_rank = operand_memref.getType().cast<MemRefType>().getRank();
  // total elems to reduce
  Value acc_mul = b->create<ConstantIndexOp>(loc, 1);
  SmallVector<Value, 4> reduction_size_vec;
  for (auto dim : dimensions) {
    auto dim_size = b->create<DimOp>(loc, operand_memref, dim);
    reduction_size_vec.push_back(dim_size);
    acc_mul = b->create<MulIOp>(loc, acc_mul, dim_size);
  }
  auto zero = b->create<ConstantIndexOp>(loc, 0);
  auto one = b->create<ConstantIndexOp>(loc, 1);
  auto forOp = b->create<scf::ForOp>(loc, zero, acc_mul, one,
                                     ArrayRef<Value>({init_value}));
  forOp.getBody()->clear();
  b->setInsertionPointToStart(forOp.getBody());
  auto var = forOp.getInductionVar();

  // get input index
  auto reduce_multidim_index =
      calcMultiDimIndex(b, loc, var, reduction_size_vec);
  SmallVector<Value, 4> input_index;
  int reduced_dim_pos = 0;
  int output_dim_pos = 0;
  for (int dim = 0; dim < input_rank; ++dim) {
    bool is_reduction_dim = (std::find(dimensions.begin(), dimensions.end(),
                                       dim) != dimensions.end());
    if (is_reduction_dim) {
      input_index.push_back(reduce_multidim_index[reduced_dim_pos]);
      reduced_dim_pos++;
    } else {
      input_index.push_back(output_index[output_dim_pos]);
      output_dim_pos++;
    }
  }
  auto data = b->create<LoadOp>(loc, operand_memref, input_index);
  AccumulatorFactory accumFactory =
      getFactory(*b, loc, cast<lmhlo::ReduceOp>(op).body());
  auto acc = accumFactory(*(forOp.getRegionIterArgs().begin()), data);
  SmallVector<Value, 4> yield_values;
  yield_values.push_back(acc);
  b->create<scf::YieldOp>(loc, yield_values);
  b->setInsertionPointAfter(forOp);
  return *(forOp.results().begin());
}

scf::ForOp createLoopAndSetInsPt(OpBuilder& b, Location loc, Value& var,
                                 Value lb, Value ub, Value step,
                                 ArrayRef<Value> init_values) {
  auto for_op = b.create<scf::ForOp>(loc, lb, ub, step, init_values);
  b.setInsertionPointToStart(for_op.getBody());
  var = for_op.getInductionVar();
  return for_op;
}

// This is a workaround implementation for lmhlo.is_finite op since
// mlir std dialect does not support this ATM.
template <>
Value elementalLower<lmhlo::IsFiniteOp>(OpBuilder* b, Location loc,
                                        lmhlo::IsFiniteOp op,
                                        ValueRange output_index,
                                        bool check_cache) {
  Value operand_memref = op->getOperand(0);
  auto tp = op->getOperand(0).getType().dyn_cast<ShapedType>();
  auto elem_tp = tp.getElementType();
  int result_rank = output_index.size();

  auto maybe_load_from_cache = [&](Value operand_memref) -> Value {
    if (!check_cache) {
      return b->create<LoadOp>(loc, operand_memref, output_index);
    }
    return createLoadOrUseCachedValue(loc, b, operand_memref, output_index,
                                      b->saveInsertionPoint());
  };

  Value operand = maybe_load_from_cache(operand_memref);
  auto abs_operand = b->create<AbsFOp>(loc, operand);
  auto INF = b->create<ConstantOp>(
      loc, elem_tp, b->getF32FloatAttr(std::numeric_limits<float>::infinity()));
  return b->create<CmpFOp>(loc, CmpFPredicate::ONE, abs_operand, INF);
}

template <>
Value elementalLower<lmhlo::ClampOp>(OpBuilder* b, Location loc,
                                     lmhlo::ClampOp op, ValueRange output_index,
                                     bool check_cache) {
  Value min_memref = op->getOperand(0);
  Value operand_memref = op->getOperand(1);
  Value max_memref = op->getOperand(2);
  auto min_ty = min_memref.getType().dyn_cast<MemRefType>();
  auto max_ty = max_memref.getType().dyn_cast<MemRefType>();
  auto operand_ty = operand_memref.getType().dyn_cast<MemRefType>();
  assert(min_ty && max_ty && operand_ty &&
         "unexpected operand type of ClampOp");
  auto elem_ty = operand_ty.getElementType();
  bool min_is_scalar = (min_ty.getRank() == 0);
  bool max_is_scalar = (max_ty.getRank() == 0);

  auto maybe_load_from_memref = [&](Value memref, bool is_scalar) -> Value {
    if (!check_cache) {
      return b->create<LoadOp>(loc, memref,
                               is_scalar ? ValueRange{} : output_index);
    }
    return createLoadOrUseCachedValue(loc, b, memref,
                                      (is_scalar ? ValueRange{} : output_index),
                                      b->saveInsertionPoint());
  };
  Value min = maybe_load_from_memref(min_memref, min_is_scalar);
  Value max = maybe_load_from_memref(max_memref, max_is_scalar);
  Value operand = maybe_load_from_memref(operand_memref, false);

  Value lb_clipped = lmhlo::impl::MapLhloOpToStdScalarOp<lmhlo::MaxOp>(
      loc, ArrayRef<Type>{elem_ty}, ArrayRef<Type>{elem_ty, elem_ty},
      ArrayRef<Value>{operand, min}, b);
  return lmhlo::impl::MapLhloOpToStdScalarOp<lmhlo::MinOp>(
      loc, ArrayRef<Type>{elem_ty}, ArrayRef<Type>{elem_ty, elem_ty},
      ArrayRef<Value>{lb_clipped, max}, b);
}

memref::ReinterpretCastOp createMemRef1DReinterpretCastWithStaticShape(
    OpBuilder& b, Location loc, Value memref) {
  auto memref_ty = memref.getType().cast<MemRefType>();
  assert(memref_ty.getAffineMaps().empty());
  assert(memref_ty.hasStaticShape());

  int64_t nElems = 1;
  for (int64_t size : memref_ty.getShape()) {
    nElems *= size;
  }

  auto memref_1d_type =
      MemRefType::get({nElems}, memref_ty.getElementType(),
                      memref_ty.getAffineMaps(), memref_ty.getMemorySpace());
  SmallVector<int64_t> sizes{nElems};
  SmallVector<int64_t> strides{1};
  return b.create<memref::ReinterpretCastOp>(loc, memref_1d_type, memref, 0,
                                             sizes, strides);
}

// reinterpret_cast the input memref into 1D
memref::ReinterpretCastOp createMemRef1DReinterpretCast(OpBuilder& b,
                                                        Location loc,
                                                        Value memref) {
  auto memref_ty = memref.getType().cast<MemRefType>();
  assert(memref_ty.getAffineMaps().empty());
  if (memref_ty.hasStaticShape()) {
    return createMemRef1DReinterpretCastWithStaticShape(b, loc, memref);
  }
  Value size = emitNumElementsComputation(b, loc, memref);
  Value stride = b.create<mlir::ConstantOp>(
      loc, b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 1));
  Value zero = b.create<mlir::ConstantOp>(
      loc, b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 0));
  auto memref_1d_type =
      MemRefType::get({MemRefType::kDynamicSize}, memref_ty.getElementType(),
                      memref_ty.getAffineMaps(), memref_ty.getMemorySpace());
  return b.create<memref::ReinterpretCastOp>(
      loc, memref_1d_type, memref, zero, ValueRange{size}, ValueRange{stride});
}

void createOffsetStore(OpBuilder& b, Location loc, Value res, Value memref,
                       Value offset) {
  Value memref_1d = createMemRef1DReinterpretCast(b, loc, memref);
  b.create<memref::StoreOp>(loc, res, memref_1d, ValueRange{offset});
}

LoadOp createOffsetLoad(OpBuilder& b, Location loc, Value memref,
                        Value offset) {
  Value memref_1d = createMemRef1DReinterpretCast(b, loc, memref);
  return b.create<LoadOp>(loc, memref_1d, ValueRange{offset});
}

// TODO: check the definition of "SignlessInteger"
AtomicRMWKind getAtomicRMWKind(Region& body) {
  auto calc_op = getReduceOperator(body);
  auto num_operands = calc_op->getNumOperands();
  auto result_type = calc_op->getOperand(num_operands - 1).getType();
  auto result_elem_type = result_type.cast<MemRefType>().getElementType();
  if (isa<lmhlo::AddOp>(calc_op)) {
    if (result_elem_type.isF16() || result_elem_type.isF32() ||
        result_elem_type.isF64()) {
      return AtomicRMWKind::addf;
    } else if (result_elem_type.isSignlessInteger() ||
               result_elem_type.isSignedInteger() ||
               result_elem_type.isUnsignedInteger()) {
      return AtomicRMWKind::addi;
    } else {
      assert(false && "unexpected atomic reduce operation");
    }
  } else if (isa<lmhlo::MulOp>(calc_op)) {
    if (result_elem_type.isF16() || result_elem_type.isF32() ||
        result_elem_type.isF64()) {
      return AtomicRMWKind::mulf;
    } else if (result_elem_type.isSignlessInteger() ||
               result_elem_type.isSignedInteger() ||
               result_elem_type.isUnsignedInteger()) {
      return AtomicRMWKind::muli;
    } else {
      assert(false && "unexpected atomic reduce operation");
    }
  } else if (isa<lmhlo::MaxOp>(calc_op)) {
    if (result_elem_type.isF16() || result_elem_type.isF32() ||
        result_elem_type.isF64()) {
      return AtomicRMWKind::maxf;
    } else if (result_elem_type.isSignedInteger()) {
      return AtomicRMWKind::maxs;
    } else if (result_elem_type.isUnsignedInteger() ||
               result_elem_type.isSignlessInteger()) {
      return AtomicRMWKind::maxu;
    } else {
      assert(false && "unexpected atomic reduce operation");
    }
  } else if (isa<lmhlo::MinOp>(calc_op)) {
    if (result_elem_type.isF16() || result_elem_type.isF32() ||
        result_elem_type.isF64()) {
      return AtomicRMWKind::minf;
    } else if (result_elem_type.isSignedInteger()) {
      return AtomicRMWKind::mins;
    } else if (result_elem_type.isUnsignedInteger() ||
               result_elem_type.isSignlessInteger()) {
      return AtomicRMWKind::minu;
    } else {
      assert(false && "unexpected atomic reduce operation");
    }
  } else if (isa<lmhlo::OrOp>(calc_op)) {
    // We convert reduce OrOp to reduce MaxOp supposed that
    // the operand having type i1 or unsigned.
    auto int_tp = result_elem_type.dyn_cast<IntegerType>();
    if (result_elem_type.isUnsignedInteger()) {
      return AtomicRMWKind::maxu;
    } else if (int_tp && int_tp.getWidth() == 1) {
      return AtomicRMWKind::maxu;
    } else {
      assert(false && "unexpected atomic reduce operation");
    }
  } else if (isa<lmhlo::AndOp>(calc_op)) {
    // We convert reduce AndOp to reduce MinOp supposed that
    // the operand having type i1 or unsigned.
    auto int_tp = result_elem_type.dyn_cast<IntegerType>();
    if (result_elem_type.isUnsignedInteger()) {
      return AtomicRMWKind::minu;
    } else if (int_tp && int_tp.getWidth() == 1) {
      return AtomicRMWKind::minu;
    } else {
      assert(false && "unexpected atomic reduce operation");
    }
  } else {
    assert(false && "unexpected atomic reduce operation");
  }
  llvm_unreachable("unsupported atomic operation kind");
  return AtomicRMWKind::addf;
}

Value getRootMemRef(Value memref) {
  Value rootMemRef = memref;
  while (Operation* operandOp = rootMemRef.getDefiningOp()) {
    if (!isa<memref::SubViewOp, memref::ViewOp, memref::CastOp,
             memref::ReinterpretCastOp>(operandOp))
      break;
    rootMemRef = operandOp->getOperand(0);
  }
  return rootMemRef;
}

bool isSameUnderlineBuffer(Value lhs, Value rhs) {
  return getRootMemRef(lhs) == getRootMemRef(rhs);
}

// returns the users of the `memref`. The users should be in the same fusion
// like `op`.
DenseSet<Operation*> getValueUsersInFusionLike(Value memref, Operation* op) {
  // rootMemRef is the underline buffer, by passing some memref cast ops.
  Value rootMemRef = getRootMemRef(memref);

  DenseSet<Operation*> ops;
  SmallVector<Value, 4> worklist{rootMemRef};
  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    for (Operation* user : val.getUsers()) {
      if (isa<memref::SubViewOp, memref::ViewOp, memref::CastOp,
              memref::ReinterpretCastOp>(user)) {
        worklist.push_back(user->getResult(0));
        continue;
      }
      // SpecializeWithSpeculation pass may generate multi versions from a
      // fusion op. This fusion family accesses a same set of memrefs.
      if (!inSameFusionOp(user, op)) continue;
      ops.insert(user);
    }
  }

  return ops;
}

}  // namespace disc_ral
}  // namespace mlir
