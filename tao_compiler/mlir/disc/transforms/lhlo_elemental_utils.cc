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

#include "mlir/disc/transforms/lhlo_elemental_utils.h"

#include "lhlo/IR/lhlo_ops.h"
#include "lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "llvm/Support/Debug.h"
#include "mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"  // from @llvm-project
#include "mlir/Conversion/LLVMCommon/Pattern.h"        // from @llvm-project
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/codegen_utils.h"
#include "mlir/disc/transforms/disc_shape_optimization_utils.h"

using mlir::memref::DimOp;
using mlir::memref::LoadOp;
using mlir::memref::StoreOp;

namespace mlir {
namespace disc_ral {

LowerConfig::SpecificLoader* LowerConfig::getSpecificLoader(Operation* op,
                                                            Value operand) {
  auto getParentFusionOp = [&](Operation* op) {
    auto parent = op->getParentOp();
    while (parent != nullptr && !isa<lmhlo::FusionOp>(parent)) {
      parent = parent->getParentOp();
    }
    return dyn_cast_or_null<lmhlo::FusionOp>(parent);
  };
  lmhlo::FusionOp fusion = getParentFusionOp(op);
  if (fusion == nullptr) {
    return nullptr;
  }
  auto specific_loader =
      specific_loaders_.find(std::make_pair(fusion, operand));
  return (specific_loader == specific_loaders_.end())
             ? nullptr
             : &specific_loader->second;
}

LowerConfig::SpecificStore* LowerConfig::getSpecificStore(Operation* op,
                                                          Value operand) {
  auto getParentFusionOp = [&](Operation* op) {
    auto parent = op->getParentOp();
    while (parent != nullptr && !isa<lmhlo::FusionOp>(parent)) {
      parent = parent->getParentOp();
    }
    return dyn_cast_or_null<lmhlo::FusionOp>(parent);
  };
  lmhlo::FusionOp fusion = getParentFusionOp(op);
  if (fusion == nullptr) {
    return nullptr;
  }
  auto specific_store = specific_stores_.find(std::make_pair(fusion, operand));
  return (specific_store == specific_stores_.end()) ? nullptr
                                                    : &specific_store->second;
}

Value createMaySpecificLoad(OpBuilder& b, Location loc, Operation* op,
                            Value memref, ValueRange indices,
                            LowerConfig* lower_config) {
  if (lower_config != nullptr) {
    auto loader = lower_config->getSpecificLoader(op, memref);
    if (loader != nullptr) {
      return (*loader)(b, memref, indices);
    }
  }

  return b.create<memref::LoadOp>(loc, memref, indices);
}

Value createLoadOrUseCachedValue(Location loc, OpBuilder* b, Operation* op,
                                 Value memref, ValueRange indices,
                                 OpBuilder::InsertPoint insert_point,
                                 LowerConfig* lower_config) {
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
  return rank > 0
             ? createMaySpecificLoad(*b, loc, op, memref, indices, lower_config)
             : b->create<LoadOp>(loc, memref);
}

Value mayCreateStore(OpBuilder* b, Location loc, Operation* op, Value value,
                     ValueRange out_indices, LowerConfig* lower_config) {
  if (lower_config != nullptr) {
    Value out_memref = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    if (out_memref == nullptr) {
      return nullptr;
    }
    // Write back to off-chip memory.
    if (lower_config->isWrittenBack(op)) {
      b->create<memref::StoreOp>(loc, value, out_memref, out_indices);
    }
    // Store on on-chip memory.
    auto store = lower_config->getSpecificStore(op, out_memref);
    if (store != nullptr) {
      (*store)(*b, value, out_memref, out_indices);
    }
  }
  return nullptr;
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
      return LhloOpToStdScalarOp::map<lmhlo::AddOp>(
          llvm::cast<lmhlo::AddOp>(calc_op), result_elem_type, operand_values,
          &b);
    } else if (isa<lmhlo::MulOp>(calc_op)) {
      return LhloOpToStdScalarOp::map<lmhlo::MulOp>(
          llvm::cast<lmhlo::MulOp>(calc_op), result_elem_type, operand_values,
          &b);
    } else if (isa<lmhlo::MaxOp>(calc_op)) {
      return LhloOpToStdScalarOp::map<lmhlo::MaxOp>(
          llvm::cast<lmhlo::MaxOp>(calc_op), result_elem_type, operand_values,
          &b);
    } else if (isa<lmhlo::MinOp>(calc_op)) {
      return LhloOpToStdScalarOp::map<lmhlo::MinOp>(
          llvm::cast<lmhlo::MinOp>(calc_op), result_elem_type, operand_values,
          &b);
    } else if (isa<lmhlo::OrOp>(calc_op)) {
      return LhloOpToStdScalarOp::map<lmhlo::OrOp>(
          llvm::cast<lmhlo::OrOp>(calc_op), result_elem_type, operand_values,
          &b);
    } else if (isa<lmhlo::AndOp>(calc_op)) {
      return LhloOpToStdScalarOp::map<lmhlo::AndOp>(
          llvm::cast<lmhlo::AndOp>(calc_op), result_elem_type, operand_values,
          &b);
    } else {
      assert(false && "unexpected reduce operation");
      return LhloOpToStdScalarOp::map<lmhlo::AddOp>(
          llvm::cast<lmhlo::AddOp>(calc_op), result_elem_type, operand_values,
          &b);
    }
  });
}

template <typename LHLO_OpTy>
Value elementalLower(OpBuilder* b, Location loc, LHLO_OpTy op,
                     ValueRange output_index, bool check_cache,
                     LowerConfig* lower_config);

template <>
Value elementalLower<lmhlo::SliceOp>(OpBuilder* b, Location loc,
                                     lmhlo::SliceOp op, ValueRange output_index,
                                     bool check_cache,
                                     LowerConfig* lower_config) {
  int rank = output_index.size();

  SmallVector<Value> input_index;
  for (int dim = 0; dim < rank; ++dim) {
    // for each dim, output[..a..] = input[..a * a_stride + a_start..]
    Value start_index = b->create<arith::ConstantIndexOp>(
        loc, op.getStartIndices().getValues<int64_t>()[dim]);
    Value stride = b->create<arith::ConstantIndexOp>(
        loc, op.getStrides().getValues<int64_t>()[dim]);
    auto input_dim = b->create<arith::AddIOp>(
        loc, b->create<arith::MulIOp>(loc, output_index[dim], stride),
        start_index);
    input_index.push_back(input_dim);
  }

  Value operand_memref = op->getOperand(0);
  Value result;
  if (!check_cache) {
    result = createMaySpecificLoad(*b, loc, op.getOperation(), operand_memref,
                                   input_index, lower_config);
  } else {
    result = createLoadOrUseCachedValue(loc, b, op.getOperation(),
                                        operand_memref, input_index,
                                        b->saveInsertionPoint(), lower_config);
  }
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo::RealDynamicSliceOp>(OpBuilder* b, Location loc,
                                                lmhlo::RealDynamicSliceOp op,
                                                ValueRange output_index,
                                                bool check_cache,
                                                LowerConfig* lower_config) {
  Value start_indices_memref = op->getOperand(1);
  Value strides_memref = op->getOperand(3);
  std::unique_ptr<SliceOpShapeHelper> helper;
  if (useShapeConstraintIR()) {
    helper.reset(new SliceOpShapeHelper(op));
  }
  int rank = output_index.size();
  SmallVector<Value, 4> input_index;
  for (int dim = 0; dim < rank; ++dim) {
    if (useShapeConstraintIR() && helper->isFullySlicedAxis(dim)) {
      input_index.push_back(output_index[dim]);
      continue;
    }

    SmallVector<Value, 4> dim_index;
    dim_index.push_back(b->create<arith::ConstantIndexOp>(loc, dim));
    Value start_index_load;
    if (useShapeConstraintIR() && !helper->isStartIndexUnknown(dim)) {
      start_index_load =
          b->create<arith::ConstantIndexOp>(loc, helper->getStartIndex(dim));
    } else {
      start_index_load = createMaySpecificLoad(
          *b, loc, op.getOperation(), start_indices_memref,
          ValueRange{dim_index}, lower_config);
    }
    auto start_index = mayConvertToIndexType(start_index_load, b, loc);

    Value stride_load;
    if (useShapeConstraintIR() && !helper->isStrideUnknown(dim)) {
      stride_load =
          b->create<arith::ConstantIndexOp>(loc, helper->getStride(dim));
    } else {
      stride_load =
          createMaySpecificLoad(*b, loc, op.getOperation(), strides_memref,
                                ValueRange{dim_index}, lower_config);
    }
    auto stride = mayConvertToIndexType(stride_load, b, loc);
    // input_dim = out_dim * stride + start_index
    auto input_dim = b->create<arith::AddIOp>(
        loc, b->create<arith::MulIOp>(loc, output_index[dim], stride),
        start_index);
    input_index.push_back(input_dim);
  }

  Value operand_memref = *(op->getOperands().begin());

  Value result;
  if (!check_cache) {
    result = createMaySpecificLoad(*b, loc, op.getOperation(), operand_memref,
                                   input_index, lower_config);
  } else {
    result = createLoadOrUseCachedValue(loc, b, op.getOperation(),
                                        operand_memref, input_index,
                                        b->saveInsertionPoint(), lower_config);
  }
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

namespace {

template <typename T>
Value elementalLowerImplForBroadcastInDimOps(OpBuilder* b, Location loc,
                                             T broadcast_in_dim,
                                             ValueRange output_index,
                                             bool check_cache,
                                             LowerConfig* lower_config) {
  auto broadcast_dimensions =
      broadcast_in_dim.getBroadcastDimensions().template getValues<int64_t>();
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
        auto zero = b->create<arith::ConstantIndexOp>(loc, 0);
        input_index.push_back(zero);
      } else if (static_dim_size == ShapedType::kDynamic) {
        // we are not sure if this dim is to be broadcasted at compile time.
        // To enable more optimization opportunities when dim sizes of operand
        // value and output value are the same SSA value.
        auto dim_size = b->create<DimOp>(loc, operand_memref, input_dim);
        auto output_dim_size = b->create<DimOp>(loc, result_memref, dim);
        auto zero = b->create<arith::ConstantIndexOp>(loc, 0);
        auto dim_size_is_equal = b->create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, dim_size, output_dim_size);
        input_index.push_back(b->create<mlir::arith::SelectOp>(
            loc, dim_size_is_equal, output_index[dim], zero));
      } else {
        // we know this dim is not to be broadcasted at compile time
        input_index.push_back(output_index[dim]);
      }
    }
  }

  Value result;
  if (!check_cache) {
    int rank = operand_memref.getType().dyn_cast<MemRefType>().getRank();
    result = (rank > 0) ? createMaySpecificLoad(
                              *b, loc, broadcast_in_dim.getOperation(),
                              operand_memref, input_index, lower_config)
                        : b->create<LoadOp>(loc, operand_memref, ValueRange());
  } else {
    result = createLoadOrUseCachedValue(loc, b, broadcast_in_dim.getOperation(),
                                        operand_memref, input_index,
                                        b->saveInsertionPoint(), lower_config);
  }
  return result;
}

}  // namespace

template <>
Value elementalLower<lmhlo::DynamicBroadcastInDimOp>(
    OpBuilder* b, Location loc, lmhlo::DynamicBroadcastInDimOp op,
    ValueRange output_index, bool check_cache, LowerConfig* lower_config) {
  Value result = elementalLowerImplForBroadcastInDimOps(
      b, loc, op, output_index, check_cache, lower_config);
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo::BroadcastInDimOp>(OpBuilder* b, Location loc,
                                              lmhlo::BroadcastInDimOp op,
                                              ValueRange output_index,
                                              bool check_cache,
                                              LowerConfig* lower_config) {
  Value result = elementalLowerImplForBroadcastInDimOps(
      b, loc, op, output_index, check_cache, lower_config);
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo::BroadcastOp>(OpBuilder* b, Location loc,
                                         lmhlo::BroadcastOp op,
                                         ValueRange output_index,
                                         bool check_cache,
                                         LowerConfig* lower_config) {
  Value operand_memref = op->getOperand(0);
  auto operand_type = operand_memref.getType().dyn_cast<MemRefType>();
  int operand_rank = operand_type.getRank();

  int result_rank = output_index.size();

  SmallVector<Value> input_index;
  // broadcast op insert dims in the front, output[a,b,A,B] = input[A,B]
  for (int i = result_rank - operand_rank; i < result_rank; ++i) {
    input_index.push_back(output_index[i]);
  }

  Value result;
  if (!check_cache) {
    result =
        operand_rank > 0
            ? createMaySpecificLoad(*b, loc, op.getOperation(), operand_memref,
                                    input_index, lower_config)
            : b->create<LoadOp>(loc, operand_memref, ValueRange());
  } else {
    result = createLoadOrUseCachedValue(loc, b, op.getOperation(),
                                        operand_memref, input_index,
                                        b->saveInsertionPoint(), lower_config);
  }
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo::ReshapeOp>(OpBuilder* b, Location loc,
                                       lmhlo::ReshapeOp op,
                                       ValueRange output_index,
                                       bool check_cache,
                                       LowerConfig* lower_config) {
  Value operand_memref = op->getOperand(0);
  Value output_memref = op->getOperand(1);

  // calculate offset in output
  auto output_shape = getShapeValues(b, output_memref);
  Value linear_index = calcLinearIndex(b, loc, output_index, output_shape);
  // transform offset to input index
  auto operand_shape = getShapeValues(b, operand_memref);
  auto operand_index = calcMultiDimIndex(b, loc, linear_index, operand_shape);

  Value result;
  if (!check_cache) {
    result = createMaySpecificLoad(*b, loc, op.getOperation(), operand_memref,
                                   operand_index, lower_config);
  } else {
    result = createLoadOrUseCachedValue(loc, b, op.getOperation(),
                                        operand_memref, operand_index,
                                        b->saveInsertionPoint(), lower_config);
  }
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo::DynamicReshapeOp>(OpBuilder* b, Location loc,
                                              lmhlo::DynamicReshapeOp op,
                                              ValueRange output_index,
                                              bool check_cache,
                                              LowerConfig* lower_config) {
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

  Value result;
  if (!check_cache) {
    result = createMaySpecificLoad(*b, loc, op.getOperation(), operand_memref,
                                   operand_index, lower_config);
  } else {
    result = createLoadOrUseCachedValue(loc, b, op.getOperation(),
                                        operand_memref, operand_index,
                                        b->saveInsertionPoint(), lower_config);
  }
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

// There is no NotOp in std dialect, thus we provide a basic implementation.
template <>
Value elementalLower<lmhlo::NotOp>(OpBuilder* b, Location loc, lmhlo::NotOp op,
                                   ValueRange output_index, bool check_cache,
                                   LowerConfig* lower_config) {
  Value operand_memref = op->getOperand(0);
  auto operand_ty = operand_memref.getType().cast<MemRefType>();

  Value operandValue;
  if (!check_cache) {
    operandValue = createMaySpecificLoad(
        *b, loc, op.getOperation(), operand_memref, output_index, lower_config);
  } else {
    operandValue = createLoadOrUseCachedValue(
        loc, b, op.getOperation(), operand_memref, output_index,
        b->saveInsertionPoint(), lower_config);
  }

  Value falseValue;
  if (operand_ty.getElementType().isIndex()) {
    falseValue = b->create<arith::ConstantIndexOp>(loc, 0);
  } else {
    falseValue =
        b->create<arith::ConstantIntOp>(loc, 0, operand_ty.getElementType());
  }

  Value result = b->create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                          operandValue, falseValue);
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo::TransposeOp>(OpBuilder* b, Location loc,
                                         lmhlo::TransposeOp op,
                                         ValueRange output_index,
                                         bool check_cache,
                                         LowerConfig* lower_config) {
  Value operand_memref = op->getOperand(0);
  SmallVector<int64_t> permutation(op.getPermutation().getValues<int64_t>());
  int rank = permutation.size();

  SmallVector<Value> operand_index(rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    auto it = std::find(permutation.begin(), permutation.end(), dim);
    assert(it != permutation.end() && "invalid permutation element");
    operand_index[permutation[dim]] = output_index[dim];
  }

  Value result;
  if (!check_cache) {
    result = createMaySpecificLoad(*b, loc, op.getOperation(), operand_memref,
                                   operand_index, lower_config);
  } else {
    result = createLoadOrUseCachedValue(loc, b, op.getOperation(),
                                        operand_memref, operand_index,
                                        b->saveInsertionPoint(), lower_config);
  }
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo::ReverseOp>(OpBuilder* b, Location loc,
                                       lmhlo::ReverseOp op,
                                       ValueRange output_index,
                                       bool check_cache,
                                       LowerConfig* lower_config) {
  // input_shape = op->getOperand(0).shape()
  // for dim = 0 -> rank:
  //   if dim in axis:
  //     operand_index[dim] = input_shape[dim] - output_index[dim] - 1;
  //   else:
  //     operand_index[dim] = output_index[dim]
  Value operand_memref = op->getOperand(0);
  auto axis = op.getDimensions().getValues<int64_t>();
  int rank = output_index.size();
  SmallVector<Value> operand_index(rank);
  Value one = b->create<arith::ConstantIndexOp>(loc, 1);
  auto input_shape = getShapeValues(b, operand_memref);
  for (int64_t dim = 0; dim < rank; ++dim) {
    auto it = std::find(axis.begin(), axis.end(), dim);
    if (it != axis.end()) {
      auto shape_value = b->create<arith::SubIOp>(loc, input_shape[dim], one);
      auto output_dim =
          b->create<arith::SubIOp>(loc, shape_value, output_index[dim]);
      operand_index[dim] = output_dim;
    } else {
      operand_index[dim] = output_index[dim];
    }
  }

  Value result;
  if (!check_cache) {
    result = createMaySpecificLoad(*b, loc, op.getOperation(), operand_memref,
                                   operand_index, lower_config);
  } else {
    result = createLoadOrUseCachedValue(loc, b, op.getOperation(),
                                        operand_memref, operand_index,
                                        b->saveInsertionPoint(), lower_config);
  }
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo::DynamicPadOp>(OpBuilder* b, Location loc,
                                          lmhlo::DynamicPadOp op,
                                          ValueRange output_index,
                                          bool check_cache,
                                          LowerConfig* lower_config) {
  Value operand_memref = *(op->getOperands().begin());
  Value padding_value_memref = *(op->getOperands().begin() + 1);
  Value edge_padding_low_memref = *(op->getOperands().begin() + 2);
  Value interior_padding_memref = *(op->getOperands().begin() + 4);
  int rank = output_index.size();
  SmallVector<Value, 4> input_index;
  Value in_bound = b->create<arith::ConstantIntOp>(loc, 1, 1);
  Value one = b->create<arith::ConstantIndexOp>(loc, 1);
  Value zero = b->create<arith::ConstantIndexOp>(loc, 0);
  std::unique_ptr<PadOpShapeHelper> helper;
  if (useShapeConstraintIR()) {
    helper.reset(new PadOpShapeHelper(op));
  }
  // for each dim i:
  //   x = output_dim[i] - edge_padding_low[i]
  //   y = x % (interior_padding[i] + 1)
  //   z = x / (interior_padding[i] + 1)
  //   in_bound = in_bound &&
  //      (x >= 0) &&
  //      (y == 0) &&
  //      (z < input_shape[i])
  for (int dim = 0; dim < rank; ++dim) {
    if (useShapeConstraintIR() && helper->isNotPaddedAxis(dim)) {
      input_index.push_back(output_index[dim]);
      continue;
    }

    SmallVector<Value, 4> dim_const;
    dim_const.push_back(b->create<arith::ConstantIndexOp>(loc, dim));
    Value edge_padding_low;
    if (useShapeConstraintIR() && !helper->isEdgePaddingLowUnknown(dim)) {
      edge_padding_low = b->create<arith::ConstantIndexOp>(
          loc, helper->getEdgePaddingLow(dim));
    } else {
      edge_padding_low = createMaySpecificLoad(*b, loc, op.getOperation(),
                                               edge_padding_low_memref,
                                               dim_const, lower_config);
    }
    edge_padding_low = mayConvertToIndexType(edge_padding_low, b, loc);

    Value interior_padding;
    if (useShapeConstraintIR() && !helper->isInteriorPaddingUnknown(dim)) {
      interior_padding = b->create<arith::ConstantIndexOp>(
          loc, helper->getInteriorPadding(dim));
    } else {
      interior_padding = createMaySpecificLoad(*b, loc, op.getOperation(),
                                               interior_padding_memref,
                                               dim_const, lower_config);
    }
    interior_padding = mayConvertToIndexType(interior_padding, b, loc);
    auto x = b->create<arith::SubIOp>(loc, output_index[dim], edge_padding_low);
    auto interior_padding_p1 =
        b->create<arith::AddIOp>(loc, interior_padding, one);
    auto y = b->create<arith::RemUIOp>(loc, x, interior_padding_p1);
    auto z = b->create<arith::DivUIOp>(loc, x, interior_padding_p1);
    in_bound = b->create<arith::AndIOp>(
        loc, in_bound,
        b->create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, x, zero));
    in_bound = b->create<arith::AndIOp>(
        loc, in_bound,
        b->create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, y, zero));

    in_bound = b->create<arith::AndIOp>(
        loc, in_bound,
        b->create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, z,
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
  if_inbound_op.getThenRegion().front().clear();
  if_inbound_op.getElseRegion().front().clear();
  b->setInsertionPointToEnd(&if_inbound_op.getThenRegion().front());
  auto ret_value =
      check_cache
          ? createLoadOrUseCachedValue(loc, b, op.getOperation(),
                                       operand_memref, input_index,
                                       b->saveInsertionPoint(), lower_config)
          : createMaySpecificLoad(*b, loc, op.getOperation(), operand_memref,
                                  input_index, lower_config);
  b->create<scf::YieldOp>(loc, ret_value);

  b->setInsertionPointToEnd(&if_inbound_op.getElseRegion().front());
  Value padded_value =
      b->create<LoadOp>(loc, padding_value_memref, ValueRange());
  b->create<scf::YieldOp>(loc, padded_value);

  b->setInsertionPointAfter(if_inbound_op);
  Value result = *(if_inbound_op.getResults().begin());
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

Value lowerGatherOpInternal(OpBuilder* b, Location loc, Operation* op,
                            ValueRange output_index, bool check_cache,
                            LowerConfig* lower_config) {
  auto gather = dyn_cast<lmhlo::GatherOp>(op);
  auto d_gather = dyn_cast<lmhlo::DynamicGatherOp>(op);
  assert((gather || d_gather) && "unexpected opcode");
  auto operand = gather ? gather.getOperand() : d_gather.getOperand();
  auto start_indices =
      gather ? gather.getStartIndices() : d_gather.getStartIndices();
  auto dimension_numbers =
      gather ? gather.getDimensionNumbers() : d_gather.getDimensionNumbers();
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
  auto collapsed_slice_dims = dimension_numbers.getCollapsedSliceDims();
  auto offset_dims = dimension_numbers.getOffsetDims();
  int64_t index_vector_dim = dimension_numbers.getIndexVectorDim();
  auto start_index_map = dimension_numbers.getStartIndexMap();

  // get initial operand_index
  SmallVector<Value, 4> operand_index;
  operand_index.reserve(operand_rank);
  SmallVector<int64_t, 4> operand_to_output_dim(operand_rank, -1);
  for (int64_t i = 0, operand_index_dim = 0; i < operand_rank; ++i) {
    bool is_collapsed_dim =
        (std::find(collapsed_slice_dims.begin(), collapsed_slice_dims.end(),
                   i) != collapsed_slice_dims.end());
    if (is_collapsed_dim) {
      operand_index.push_back(b->create<arith::ConstantIndexOp>(loc, 0));
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
      output_dim_size = b->create<arith::ConstantIndexOp>(loc, 1);
    } else {
      output_dim_size = b->create<DimOp>(loc, result, output_dim);
    }
    Value largest_valid_start_index = b->create<arith::IndexCastOp>(
        loc, index_component.getType(),
        b->create<arith::SubIOp>(
            loc, b->create<DimOp>(loc, operand, operand_dim), output_dim_size));
    auto zero = b->create<arith::ConstantIntOp>(
        loc, 0, index_component.getType().cast<IntegerType>().getWidth());
    auto max_with_zero = b->create<mlir::arith::SelectOp>(
        loc,
        b->create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, zero,
                                 index_component),
        zero, index_component);
    auto gather_dim_component_extended_inbound =
        b->create<mlir::arith::SelectOp>(
            loc,
            b->create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                     max_with_zero, largest_valid_start_index),
            max_with_zero, largest_valid_start_index);

    operand_index[operand_dim] = b->create<arith::AddIOp>(
        loc, operand_index[operand_dim],
        mayConvertToIndexType(gather_dim_component_extended_inbound, b, loc));
  };

  // update operand_index
  if (start_indices_rank == index_vector_dim) {
    Value gather_dim_component =
        check_cache ? createLoadOrUseCachedValue(
                          loc, b, op, start_indices, gather_index_index,
                          b->saveInsertionPoint(), lower_config)
                    : createMaySpecificLoad(*b, loc, op, start_indices,
                                            gather_index_index, lower_config);
    add_to_operand_index(gather_dim_component, 0);
  } else {
    int64_t index_vector_size = start_indices_ty.getDimSize(index_vector_dim);
    assert((index_vector_size != ShapedType::kDynamic) &&
           "dynamic index_vector_dim size for GatherOp is unexpected");
    for (int64_t i = 0; i < index_vector_size; ++i) {
      gather_index_index[index_vector_dim] =
          b->create<arith::ConstantIndexOp>(loc, i);
      Value gather_dim_component =
          check_cache ? createLoadOrUseCachedValue(
                            loc, b, op, start_indices, gather_index_index,
                            b->saveInsertionPoint(), lower_config)
                      : createMaySpecificLoad(*b, loc, op, start_indices,
                                              gather_index_index, lower_config);
      add_to_operand_index(gather_dim_component, i);
    }
  }
  return (check_cache ? createLoadOrUseCachedValue(
                            loc, b, op, operand, operand_index,
                            b->saveInsertionPoint(), lower_config)
                      : createMaySpecificLoad(*b, loc, op, operand,
                                              operand_index, lower_config));
}

template <>
Value elementalLower<lmhlo::GatherOp>(OpBuilder* b, Location loc,
                                      lmhlo::GatherOp op,
                                      ValueRange output_index, bool check_cache,
                                      LowerConfig* lower_config) {
  Value result = lowerGatherOpInternal(b, loc, op, output_index, check_cache,
                                       lower_config);
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo::DynamicGatherOp>(OpBuilder* b, Location loc,
                                             lmhlo::DynamicGatherOp op,
                                             ValueRange output_index,
                                             bool check_cache,
                                             LowerConfig* lower_config) {
  Value result = lowerGatherOpInternal(b, loc, op, output_index, check_cache,
                                       lower_config);
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo::ConcatenateOp>(OpBuilder* b, Location loc,
                                           lmhlo::ConcatenateOp op,
                                           ValueRange output_index,
                                           bool check_cache,
                                           LowerConfig* lower_config) {
  size_t axis = op.getDimension();
  size_t rank = output_index.size();

  auto num_input_operands = op.getNumOperands() - 1;
  auto zero = b->create<arith::ConstantIndexOp>(loc, 0);

  SmallVector<Value> axis_dim_ranges;
  axis_dim_ranges.push_back(zero);
  for (int i = 0; i < num_input_operands; ++i) {
    axis_dim_ranges.push_back(b->create<arith::AddIOp>(
        loc, getDimSizeValue(b, op.getOperand(i), axis),
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
    auto float_result_elem_type = result_elem_type.cast<FloatType>();
    zero_element = b->create<arith::ConstantFloatOp>(
        loc, APFloat::getZero(float_result_elem_type.getFloatSemantics()),
        float_result_elem_type);
  } else if (result_elem_type.isSignlessInteger() ||
             result_elem_type.isSignedInteger() ||
             result_elem_type.isUnsignedInteger()) {
    zero_element = b->create<arith::ConstantIntOp>(loc, 0, result_elem_type);
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
    auto gt_low_bound = b->create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                                 out_idx, low_bound);
    auto lt_up_bound = b->create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                out_idx, up_bound);
    auto in_bound = b->create<arith::AndIOp>(loc, gt_low_bound, lt_up_bound);
    if_inbound_ops[i] = b->create<scf::IfOp>(loc, if_result_types, in_bound,
                                             /*withElseRegion*/ true);
    if_inbound_ops[i].getThenRegion().front().clear();
    if_inbound_ops[i].getElseRegion().front().clear();

    b->setInsertionPointToEnd(&if_inbound_ops[i].getThenRegion().front());
    input_index[axis] = b->create<arith::SubIOp>(loc, out_idx, low_bound);
    auto operand_memref = op.getOperand(i);
    auto ret_value =
        check_cache
            ? createLoadOrUseCachedValue(loc, b, op.getOperation(),
                                         operand_memref, input_index,
                                         b->saveInsertionPoint(), lower_config)
            : createMaySpecificLoad(*b, loc, op.getOperation(), operand_memref,
                                    input_index, lower_config);
    b->create<scf::YieldOp>(loc, ret_value);

    b->setInsertionPointToEnd(&if_inbound_ops[i].getElseRegion().front());
    if (i == num_input_operands - 1) {
      b->create<scf::YieldOp>(loc, zero_element);  // expect never used
    } else {
      b->create<scf::YieldOp>(loc, if_inbound_ops[i + 1].getResults());
    }
    b->setInsertionPointAfter(if_inbound_ops[i]);
  }
  b->setInsertionPointAfter(if_inbound_ops[0]);
  Value result = *(if_inbound_ops[0].getResults().begin());
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo_disc::ConcatenateOp>(OpBuilder* b, Location loc,
                                                lmhlo_disc::ConcatenateOp op,
                                                ValueRange output_index,
                                                bool check_cache,
                                                LowerConfig* lower_config) {
  size_t axis = op.getDimension();
  size_t rank = output_index.size();
  MLIRContext* ctx = b->getContext();

  auto num_input_operands = op.getNumOperands() - 2;
  auto zero = b->create<arith::ConstantIndexOp>(loc, 0);
  auto one = b->create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> axis_dim_ranges;
  axis_dim_ranges.push_back(zero);
  for (int i = 0; i < num_input_operands; ++i) {
    axis_dim_ranges.push_back(b->create<arith::AddIOp>(
        loc, getDimSizeValue(b, op.getOperand(i), axis),
        axis_dim_ranges.back()));
  }

  // {inputs, input_ptr, out}
  auto ptr_array = op.getOperand(op.getNumOperands() - 2);
  auto out = op.getOperand(op.getNumOperands() - 1);

  auto output_shape = getShapeValues(b, out);
  Value linear_index = calcLinearIndex(b, loc, output_index, output_shape);
  auto operand_index = b->create<arith::FloorDivSIOp>(
      loc, b->getIndexType(), output_index[axis],
      getDimSizeValue(b, op.getOperand(0), axis));

  auto int_ptr =
      b->create<memref::LoadOp>(loc, ptr_array, ValueRange{operand_index});
  Type ptr_type = LLVM::LLVMPointerType::get(FloatType::getF32(ctx));
  auto llvm_ptr = b->create<LLVM::IntToPtrOp>(loc, ptr_type, int_ptr);

  SmallVector<Value, 4> input_index;
  std::copy(output_index.begin(), output_index.end(),
            std::back_inserter(input_index));

  input_index[axis] = b->create<arith::SubIOp>(
      loc, output_index[axis],
      b->create<arith::MulIOp>(loc, operand_index,
                               getDimSizeValue(b, op.getOperand(0), axis)));
  Value input_offset =
      calcLinearIndex(b, loc, input_index, getShapeValues(b, op.getOperand(0)));
  input_offset = b->create<arith::IndexCastOp>(loc, IntegerType::get(ctx, 32),
                                               input_offset);
  auto llvm_elem =
      b->create<LLVM::GEPOp>(loc, ptr_type, llvm_ptr, input_offset);
  return b->create<LLVM::LoadOp>(loc, llvm_elem);
}

// There is no 'identityOp' in std dialect, thus we provide a basic
// implementation here as a workaround. This can be removed once std dialect
// supports CopyOp.
template <>
Value elementalLower<lmhlo::CopyOp>(OpBuilder* b, Location loc,
                                    lmhlo::CopyOp op, ValueRange output_index,
                                    bool check_cache,
                                    LowerConfig* lower_config) {
  Value operand_memref = op->getOperand(0);
  auto input_type = operand_memref.getType().dyn_cast<ShapedType>();
  assert(input_type && "expected operand having ShapedType");
  Value result;
  if (!check_cache) {
    result = createMaySpecificLoad(*b, loc, op.getOperation(), operand_memref,
                                   output_index, lower_config);
  } else {
    result = createLoadOrUseCachedValue(loc, b, op.getOperation(),
                                        operand_memref, output_index,
                                        b->saveInsertionPoint(), lower_config);
  }
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo_disc::H2DOp>(OpBuilder* b, Location loc,
                                        lmhlo_disc::H2DOp op,
                                        ValueRange output_index,
                                        bool check_cache,
                                        LowerConfig* lower_config) {
  Value operand_memref = op->getOperand(0);
  int rank = operand_memref.getType().cast<MemRefType>().getRank();
  SmallVector<Value> zeroIndices(rank,
                                 b->create<arith::ConstantIndexOp>(loc, 0));

  Value result;
  if (!check_cache) {
    result = createMaySpecificLoad(*b, loc, op.getOperation(), operand_memref,
                                   zeroIndices, lower_config);
  } else {
    result = createLoadOrUseCachedValue(loc, b, op.getOperation(),
                                        operand_memref, zeroIndices,
                                        b->saveInsertionPoint(), lower_config);
  }
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

Value elementalLowerIota(OpBuilder* b, const Location& loc, Operation* op,
                         const ValueRange& output_index,
                         const MemRefType& result_ty,
                         LowerConfig* lower_config) {
  auto result_element_ty = result_ty.getElementType();
  if (result_ty.getRank() == 0) {
    if (result_element_ty.dyn_cast<IntegerType>()) {
      return b->create<arith::ConstantIntOp>(loc, 0, result_element_ty);
    } else if (result_element_ty.dyn_cast<IndexType>()) {
      return b->create<arith::ConstantIndexOp>(loc, 0);
    } else if (result_element_ty.dyn_cast<FloatType>()) {
      auto float_result_element_ty = result_element_ty.cast<FloatType>();
      return b->create<arith::ConstantFloatOp>(
          loc, APFloat::getZero(float_result_element_ty.getFloatSemantics()),
          float_result_element_ty);
    } else {
      op->emitError("element_type of Iota/DynamicIotaOp not implemented");
      return Value(nullptr);
    }
  }
  int64_t iota_dimension = 0;
  if (isa<lmhlo::DynamicIotaOp>(op)) {
    auto dynamic_iota = mlir::dyn_cast<lmhlo::DynamicIotaOp>(op);
    iota_dimension = dynamic_iota.getIotaDimension();
  } else if (isa<lmhlo::IotaOp>(op)) {
    auto iota = mlir::dyn_cast<lmhlo::IotaOp>(op);
    iota_dimension = iota.getIotaDimension();
  }
  assert(iota_dimension < output_index.size() &&
         "iota_dimension exceeds rank of output_index");
  auto elem_index_linear = output_index[iota_dimension];
  Value result = nullptr;
  if (result_element_ty.dyn_cast<IntegerType>()) {
    result = b->create<arith::IndexCastOp>(loc, result_element_ty,
                                           elem_index_linear);
  } else if (result_element_ty.dyn_cast<IndexType>()) {
    result = mayConvertToIndexType(elem_index_linear, b, loc);
  } else if (result_element_ty.dyn_cast<FloatType>()) {
    auto idx2int = mayConvertToIntegerType(elem_index_linear, b, loc);
    result = b->create<arith::UIToFPOp>(loc, result_element_ty, idx2int);
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
                                           bool check_cache,
                                           LowerConfig* lower_config) {
  auto result_ty = op->getOperand(1).getType().dyn_cast<MemRefType>();
  assert(result_ty && "unexpected result type of DynamicIotaOp");
  Value result = elementalLowerIota(b, loc, op.getOperation(), output_index,
                                    result_ty, lower_config);
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo::IotaOp>(OpBuilder* b, Location loc,
                                    lmhlo::IotaOp op, ValueRange output_index,
                                    bool check_cache,
                                    LowerConfig* lower_config) {
  auto result_ty = op->getOperand(0).getType().dyn_cast<MemRefType>();
  assert(result_ty && "unexpected result type of IotaOp");
  Value result = elementalLowerIota(b, loc, op.getOperation(), output_index,
                                    result_ty, lower_config);
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo::ReduceOp>(OpBuilder* b, Location loc,
                                      lmhlo::ReduceOp op,
                                      ValueRange output_index, bool check_cache,
                                      LowerConfig* lower_config) {
  auto operand_memref = *(op->getOperands().begin());
  auto init_value_memref = *(op->getOperands().begin() + 1);
  auto init_value = b->create<LoadOp>(loc, init_value_memref);
  auto dimensions = op.getDimensions().getValues<int64_t>();
  auto input_rank = operand_memref.getType().cast<MemRefType>().getRank();
  // total elems to reduce
  Value acc_mul = b->create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value, 4> reduction_size_vec;
  for (auto dim : dimensions) {
    auto dim_size = b->create<DimOp>(loc, operand_memref, dim);
    reduction_size_vec.push_back(dim_size);
    acc_mul = b->create<arith::MulIOp>(loc, acc_mul, dim_size);
  }
  auto zero = b->create<arith::ConstantIndexOp>(loc, 0);
  auto one = b->create<arith::ConstantIndexOp>(loc, 1);
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
  auto data = createMaySpecificLoad(*b, loc, op.getOperation(), operand_memref,
                                    input_index, lower_config);
  AccumulatorFactory accumFactory =
      getFactory(*b, loc, cast<lmhlo::ReduceOp>(op).getBody());
  auto acc = accumFactory(*(forOp.getRegionIterArgs().begin()), data);
  SmallVector<Value, 4> yield_values;
  yield_values.push_back(acc);
  b->create<scf::YieldOp>(loc, yield_values);
  b->setInsertionPointAfter(forOp);
  Value result = *(forOp.getResults().begin());
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
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
                                        bool check_cache,
                                        LowerConfig* lower_config) {
  Value operand_memref = op->getOperand(0);
  auto tp = op->getOperand(0).getType().dyn_cast<ShapedType>();
  auto elem_tp = tp.getElementType();
  assert(elem_tp.isa<FloatType>());
  int result_rank = output_index.size();

  auto maybe_load_from_cache = [&](Value operand_memref) -> Value {
    if (!check_cache) {
      return createMaySpecificLoad(*b, loc, op.getOperation(), operand_memref,
                                   output_index, lower_config);
    }
    return createLoadOrUseCachedValue(loc, b, op.getOperation(), operand_memref,
                                      output_index, b->saveInsertionPoint(),
                                      lower_config);
  };

  Value operand = maybe_load_from_cache(operand_memref);
  auto abs_operand = b->create<math::AbsFOp>(loc, operand);
  auto float_elem_tp = elem_tp.cast<FloatType>();
  auto INF = b->create<arith::ConstantFloatOp>(
      loc, APFloat::getInf(float_elem_tp.getFloatSemantics()), float_elem_tp);
  Value result = b->create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE,
                                          abs_operand, INF);
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

template <>
Value elementalLower<lmhlo::ClampOp>(OpBuilder* b, Location loc,
                                     lmhlo::ClampOp op, ValueRange output_index,
                                     bool check_cache,
                                     LowerConfig* lower_config) {
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
      return createMaySpecificLoad(*b, loc, op.getOperation(), memref,
                                   is_scalar ? ValueRange{} : output_index,
                                   lower_config);
    }
    return createLoadOrUseCachedValue(loc, b, op.getOperation(), memref,
                                      (is_scalar ? ValueRange{} : output_index),
                                      b->saveInsertionPoint(), lower_config);
  };
  Value min = maybe_load_from_memref(min_memref, min_is_scalar);
  Value max = maybe_load_from_memref(max_memref, max_is_scalar);
  Value operand = maybe_load_from_memref(operand_memref, false);

  Value lb_clipped =
      mhlo::impl::mapMhloOpToStdScalarOp<lmhlo::LhloToHloOp<lmhlo::MaxOp>>(
          loc, ArrayRef<Type>{elem_ty}, ArrayRef<Type>{elem_ty, elem_ty},
          mhlo::MaxOp::Adaptor(ArrayRef<Value>{operand, min},
                               op->getAttrDictionary()),
          b);
  Value result =
      mhlo::impl::mapMhloOpToStdScalarOp<lmhlo::LhloToHloOp<lmhlo::MinOp>>(
          loc, ArrayRef<Type>{elem_ty}, ArrayRef<Type>{elem_ty, elem_ty},
          mhlo::MinOp::Adaptor(ArrayRef<Value>{lb_clipped, max},
                               op->getAttrDictionary()),
          b);
  mayCreateStore(b, loc, op.getOperation(), result, output_index, lower_config);
  return result;
}

memref::ReinterpretCastOp createMemRef1DReinterpretCastWithStaticShape(
    OpBuilder& b, Location loc, Value memref) {
  auto memref_ty = memref.getType().cast<MemRefType>();
  assert(memref_ty.getLayout().isIdentity());
  assert(memref_ty.hasStaticShape());

  int64_t nElems = 1;
  for (int64_t size : memref_ty.getShape()) {
    nElems *= size;
  }

  auto memref_1d_type =
      MemRefType::get({nElems}, memref_ty.getElementType(),
                      memref_ty.getLayout(), memref_ty.getMemorySpace());
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
  assert(memref_ty.getLayout().isIdentity());
  if (memref_ty.hasStaticShape()) {
    return createMemRef1DReinterpretCastWithStaticShape(b, loc, memref);
  }
  Value size = emitNumElementsComputation(b, loc, memref);
  Value stride = b.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  auto memref_1d_type =
      MemRefType::get({ShapedType::kDynamic}, memref_ty.getElementType(),
                      memref_ty.getLayout(), memref_ty.getMemorySpace());
  auto cast = b.create<memref::ReinterpretCastOp>(
      loc, memref_1d_type, memref, zero, ValueRange{size}, ValueRange{stride});

  // Inherit assume_alignment. If there is assume_alignment operator for
  // `memref`, create the alignment for the castop.
  memref::AssumeAlignmentOp alignment;
  for (auto user : memref.getUsers()) {
    alignment = dyn_cast<memref::AssumeAlignmentOp>(user);
    if (alignment) {
      // We assume there is not conflict alignment setting.
      break;
    }
  }
  if (alignment) {
    b.create<memref::AssumeAlignmentOp>(loc, cast, alignment.getAlignment());
  }

  return cast;
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
arith::AtomicRMWKind getAtomicRMWKind(Region& body) {
  auto calc_op = getReduceOperator(body);
  auto num_operands = calc_op->getNumOperands();
  auto result_type = calc_op->getOperand(num_operands - 1).getType();
  auto result_elem_type = result_type.cast<MemRefType>().getElementType();
  if (isa<lmhlo::AddOp>(calc_op)) {
    if (result_elem_type.isF16() || result_elem_type.isF32() ||
        result_elem_type.isF64()) {
      return arith::AtomicRMWKind::addf;
    } else if (result_elem_type.isSignlessInteger() ||
               result_elem_type.isSignedInteger() ||
               result_elem_type.isUnsignedInteger()) {
      return arith::AtomicRMWKind::addi;
    } else {
      assert(false && "unexpected atomic reduce operation");
    }
  } else if (isa<lmhlo::MulOp>(calc_op)) {
    if (result_elem_type.isF16() || result_elem_type.isF32() ||
        result_elem_type.isF64()) {
      return arith::AtomicRMWKind::mulf;
    } else if (result_elem_type.isSignlessInteger() ||
               result_elem_type.isSignedInteger() ||
               result_elem_type.isUnsignedInteger()) {
      return arith::AtomicRMWKind::muli;
    } else {
      assert(false && "unexpected atomic reduce operation");
    }
  } else if (isa<lmhlo::MaxOp>(calc_op)) {
    if (result_elem_type.isF16() || result_elem_type.isF32() ||
        result_elem_type.isF64()) {
      return arith::AtomicRMWKind::maxf;
    } else if (result_elem_type.isSignedInteger() ||
               result_elem_type.isSignlessInteger()) {
      return arith::AtomicRMWKind::maxs;
    } else if (result_elem_type.isUnsignedInteger()) {
      return arith::AtomicRMWKind::maxu;
    } else {
      assert(false && "unexpected atomic reduce operation");
    }
  } else if (isa<lmhlo::MinOp>(calc_op)) {
    if (result_elem_type.isF16() || result_elem_type.isF32() ||
        result_elem_type.isF64()) {
      return arith::AtomicRMWKind::minf;
    } else if (result_elem_type.isSignedInteger() ||
               result_elem_type.isSignlessInteger()) {
      return arith::AtomicRMWKind::mins;
    } else if (result_elem_type.isUnsignedInteger()) {
      return arith::AtomicRMWKind::minu;
    } else {
      assert(false && "unexpected atomic reduce operation");
    }
  } else if (isa<lmhlo::OrOp>(calc_op)) {
    // We convert reduce OrOp to reduce MaxOp supposed that
    // the operand having type i1 or unsigned.
    auto int_tp = result_elem_type.dyn_cast<IntegerType>();
    if (result_elem_type.isUnsignedInteger()) {
      return arith::AtomicRMWKind::maxu;
    } else if (int_tp && int_tp.getWidth() == 1) {
      return arith::AtomicRMWKind::maxu;
    } else {
      assert(false && "unexpected atomic reduce operation");
    }
  } else if (isa<lmhlo::AndOp>(calc_op)) {
    // We convert reduce AndOp to reduce MinOp supposed that
    // the operand having type i1 or unsigned.
    auto int_tp = result_elem_type.dyn_cast<IntegerType>();
    if (result_elem_type.isUnsignedInteger()) {
      return arith::AtomicRMWKind::minu;
    } else if (int_tp && int_tp.getWidth() == 1) {
      return arith::AtomicRMWKind::minu;
    } else {
      assert(false && "unexpected atomic reduce operation");
    }
  } else {
    assert(false && "unexpected atomic reduce operation");
  }
  llvm_unreachable("unsupported atomic operation kind");
  return arith::AtomicRMWKind::addf;
}

bool isUnsignedIntegerValue(Value val) {
  auto ty = val.getType().cast<MemRefType>().getElementType();
  return ty.isa<IntegerType>() && ty.isUnsignedInteger();
}

Type convertIfIntegerType(Type type) {
  if (auto int_type = type.dyn_cast<IntegerType>())
    return IntegerType::get(int_type.getContext(),
                            int_type.getIntOrFloatBitWidth());
  return type;
}

SmallVector<Value> convertValuesIfIntegerType(Location loc, OpBuilder* b,
                                              ValueRange operands) {
  SmallVector<Value> newOperands;
  for (Value operand : operands) {
    Type oldType = operand.getType();
    Type newType = convertIfIntegerType(oldType);
    Value newOperand = operand;
    if (oldType != newType)
      newOperand = b->create<UnrealizedConversionCastOp>(loc, newType, operand)
                       ->getResult(0);
    newOperands.push_back(newOperand);
  }
  return newOperands;
}

}  // namespace disc_ral
}  // namespace mlir
