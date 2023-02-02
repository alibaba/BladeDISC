// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "disc-shape-utils"

#include "mlir/disc/transforms/shape_utils.h"

#include <unordered_set>
#include <utility>

#include "lhlo/IR/lhlo_ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"
#include "mlir/disc/IR/disc_shape_ops.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/disc_util.h"

namespace mlir {
namespace disc_ral {

// Supports using EquivalenceClasses for Value
bool operator<(const ValueWrapper& lhs, const ValueWrapper& rhs) {
  auto lhs_value = lhs.getValue().getAsOpaquePointer();
  auto rhs_value = rhs.getValue().getAsOpaquePointer();
  return lhs_value < rhs_value;
}

// Merge two symbolDim if they are compatible.
LogicalResult SymbolDim::Merge(SymbolDim* other) {
  if (!isDynamic() && !other->isDynamic() &&
      getDimSize() != other->getDimSize())
    return failure();
  if (isDynamic()) dimSize_ = other->getDimSize();
  return success();
}

Type ShapeAnalysisDeprecated::getRefinedType(Value value) {
  SymbolShape* symbolShape = getShape(value);

  // not-shaped type
  if (!symbolShape) return value.getType();

  auto ty = value.getType().cast<RankedTensorType>();
  auto dims = symbolShape->getDims();
  return RankedTensorType::get(dims, ty.getElementType());
}

LogicalResult ShapeAnalysisDeprecated::run() {
  // Make sure only run once.
  if (initialized) {
    op_->emitError() << "re-initialized shape analysis";
    return failure();
  }
  initialized = true;

  return buildShapeMap();
}

LogicalResult ShapeAnalysisDeprecated::buildShapeMap() {
  // Do not merge the region loops. We want to compute shape-map, dim-value-map
  // and decompose one-by-one.
  for (auto& region : op_->getRegions()) {
    if (failed(buildRegionShapeMap(&region))) {
      return failure();
    }
  }
  for (auto& region : op_->getRegions()) {
    if (failed(buildRegionDimValueMap(&region))) {
      return failure();
    }
  }
  for (auto& region : op_->getRegions()) {
    if (failed(buildRegionMulDecompose(&region))) {
      return failure();
    }
  }
  postProcessingDimValues();

  return success();
}

void ShapeAnalysisDeprecated::postProcessingDimValues() {
  // Update dimValue roots.
  for (auto sym2val : dimSymbol2DimValue_) {
    dimSymbol2DimValue_[sym2val.first] = getRootDimValue(sym2val.second);
  }
  std::unordered_map<DimValue, SmallVector<std::vector<DimValue>>, DimValueHash>
      dimValMulDecomp;
  for (auto delinear : dimValueMulDecompose_) {
    auto key = getRootDimValue(delinear.first);
    std::unordered_set<std::vector<DimValue>, DimValueContainerHash>
        decompDimVals;
    for (auto val_set : delinear.second) {
      std::vector<DimValue> vals;
      for (auto val : val_set) {
        vals.push_back(getRootDimValue(val));
      }
      std::sort(vals.begin(), vals.end());
      decompDimVals.insert(vals);
    }
    dimValMulDecomp[key] = SmallVector<std::vector<DimValue>>(
        decompDimVals.begin(), decompDimVals.end());
  }
  dimValueMulDecompose_ = std::move(dimValMulDecomp);
  // Analyze multi-level decompose:
  //   a -> b, c
  //   b -> d, e
  //      a -> d, e, c
  // Note if `c` is one, we can get: a -> b, c, c, c, ...
  // We add a constraint that every element can be decomposed to as many as 4
  // elements.
  const std::size_t decompose_threshold = 4;
  for (auto delinear : dimValueMulDecompose_) {
    auto key = getRootDimValue(delinear.first);
    std::unordered_set<std::vector<DimValue>, DimValueContainerHash>
        decompDimVals(delinear.second.begin(), delinear.second.end());
    SmallVector<std::vector<DimValue>> worklist = delinear.second;
    while (!worklist.empty()) {
      auto workitem = worklist.pop_back_val();
      for (int64_t i = 0; i < workitem.size(); i++) {
        auto& dimVal = workitem[i];
        auto iter = dimValueMulDecompose_.find(dimVal);
        if (iter != dimValueMulDecompose_.end()) {
          auto deResult = iter->second;
          // For every decomposed set, replace dimVal.
          for (auto& de : deResult) {
            // Size after replacing should not exceed threshold.
            if (de.size() + (workitem.size() - 1) > decompose_threshold) {
              continue;
            }
            std::vector<DimValue> new_decomps;
            // Replace dimVal in workitem with the decomposed values.
            for (int64_t j = 0; j < workitem.size(); j++) {
              if (j != i) {
                new_decomps.emplace_back(workitem[j]);
              } else {
                new_decomps.insert(new_decomps.end(), de.begin(), de.end());
              }
            }
            std::sort(new_decomps.begin(), new_decomps.end());
            if (decompDimVals.find(new_decomps) == decompDimVals.end()) {
              if (new_decomps.size() < decompose_threshold) {
                worklist.emplace_back(new_decomps);
              }
              decompDimVals.emplace(std::move(new_decomps));
            }
          }
        }
      }
    }
    dimValMulDecomp[key] = SmallVector<std::vector<DimValue>>(
        decompDimVals.begin(), decompDimVals.end());
  }
  dimValueMulDecompose_ = std::move(dimValMulDecomp);
}

LogicalResult ShapeAnalysisDeprecated::buildSymbolShape(Value value) {
  if (shapeMap_.find(value) != shapeMap_.end()) {
    return success();
  }
  auto ty = value.getType().dyn_cast<ShapedType>();
  // Skip non-shaped value
  if (!ty) return success();

  // Not support dynamic rank a.t.m.
  if (!ty.hasRank()) return failure();

  SmallVector<SymbolDim*, 4> dims;
  for (int64_t dimSize : ty.getShape()) {
    dimVec_.emplace_back(new SymbolDim(dimSize));
    dims.push_back(dimVec_.back().get());
    dimMap_[dims.back()] = dimVec_.size() - 1;
  }

  shapeMap_[value] = SymbolShape(std::move(dims));
  return success();
}

LogicalResult ShapeAnalysisDeprecated::buildRegionShapeMap(Region* region) {
  // Only SCF is supported a.t.m.
  if (region->getBlocks().size() != 1) {
    return region->getParentOp()->emitError(
        "only single block region is supported");
  }
  for (Block& block : *region) {
    if (failed(buildBlockShapeMap(&block))) return failure();
  }
  return success();
}

LogicalResult ShapeAnalysisDeprecated::buildBlockShapeMap(Block* block) {
  // mapping block arguments
  for (Value value : block->getArguments()) {
    if (failed(buildSymbolShape(value))) {
      return block->getParentOp()->emitError(
          "failed to build shape for block arg");
    }
  }

  // mapping each op inside the block
  WalkResult result = block->walk([&](Operation* op) {
    if (failed(buildOperationShapeMap(op))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }

  return success();
}

LogicalResult ShapeAnalysisDeprecated::buildOperationShapeValueMap(
    Operation* op) {
  auto updateShapeValueEqual = [&](Value lhs, Value rhs) {
    if (!isShapeValueEqual(lhs, rhs)) {
      if (!shapeValueEqual_.isEquivalent(ValueWrapper(lhs),
                                         ValueWrapper(rhs))) {
        shapeValueEqual_.unionSets(ValueWrapper(lhs), ValueWrapper(rhs));
      }
    }
  };
  // Collect tensor shape value.
  if (isa<shape::ShapeOfOp>(op)) {
    // Match value and shape tensor.
    auto iter = value2ShapeValue_.find(op->getOperand(0));
    if (iter == value2ShapeValue_.end()) {
      value2ShapeValue_[op->getOperand(0)] = op->getResult(0);
    } else if (iter->second != op->getResult(0)) {
      updateShapeValueEqual(iter->second, op->getResult(0));
    }
  } else if (isa<shape::BroadcastOp>(op)) {
    if (isShapeValueEqual(op->getOperand(0), op->getOperand(1))) {
      updateShapeValueEqual(op->getOperand(0), op->getResult(0));
      updateShapeValueEqual(op->getOperand(1), op->getResult(0));
    }
  } else if (isa<mhlo::DynamicReshapeOp, mhlo::DynamicBroadcastInDimOp>(op)) {
    // Log shape tensor information and update equivalence.
    auto iter = value2ShapeValue_.find(op->getResult(0));
    if (iter == value2ShapeValue_.end()) {
      value2ShapeValue_[op->getResult(0)] = op->getOperand(1);
    } else if (iter->second != op->getOperand(1)) {
      updateShapeValueEqual(iter->second, op->getOperand(1));
    }
  } else if (isa<arith::IndexCastOp>(op)) {
    // Check whether it is dim-1 tensor. If so, it is a candidate shape tensor.
    auto input = op->getOperand(0);
    auto in_ty = input.getType().dyn_cast_or_null<RankedTensorType>();
    if (in_ty != nullptr && in_ty.getRank() == 1) {
      updateShapeValueEqual(input, op->getResult(0));
    }
  }

  return success();
}

LogicalResult ShapeAnalysisDeprecated::buildOperationShapeMap(Operation* op) {
  // build shapes for the results of op
  for (Value result : op->getResults()) {
    if (failed(buildSymbolShape(result))) {
      return op->emitError("failed to build shape for op's result");
    }
  }

  // collect shape tensor values and map equivalences.
  buildOperationShapeValueMap(op);

  // apply op's shape constraint
  return applyOpConstraint(op);
}

LogicalResult ShapeAnalysisDeprecated::buildRegionDimValueMap(Region* region) {
  // Only SCF is supported a.t.m.
  if (region->getBlocks().size() != 1) {
    return region->getParentOp()->emitError(
        "only single block region is supported");
  }
  for (Block& block : *region) {
    if (failed(buildBlockDimValueMap(&block))) return failure();
  }
  applyDimEqualWithDimValue();
  return success();
}

LogicalResult ShapeAnalysisDeprecated::buildBlockDimValueMap(Block* block) {
  // TODO: deal with memref.load and store directory rather than infer from
  // reshape.
  auto extractLmhloValueOfDim = [&](Value valueOfDims, int64_t index,
                                    DimValue& dimValue) {
    // case 1: dimValues are built with alloc and store.
    //     %0 = memref.alloc() : memref<3xi32, "cpu">
    //     memref.store %1, %0[%c0] : memref<3xi32, "cpu">
    // TODO: there may be more scenarios.
    for (auto user : valueOfDims.getUsers()) {
      if (auto store = dyn_cast_or_null<memref::StoreOp>(user)) {
        auto indices = store.getIndices();
        if (indices.size() != 1) {
          continue;
        }
        int64_t index_val;
        if (getConstIntValue(indices[0].getDefiningOp(), index_val) &&
            index_val == index) {
          dimValue = DimValue(store.getValue());
          return true;
        }
      }
    }
    return false;
  };
  auto mayCreateOrMapConstInt = [&](DimValue& dimVal) {
    DimValue intDimVal;
    if (dimVal.state == DimValue::FOLD) {
      intDimVal = dimVal;
    } else if (dimVal.state == DimValue::UNFOLD) {
      Operation* op = dimVal.unfoldVal.getDefiningOp();
      if (isa<arith::ConstantIntOp, arith::ConstantIndexOp>(op)) {
        int64_t result;
        if (getConstIntValue(op, result) && result > 0) {
          intDimVal = DimValue(result);
        }
      }
    }
    if (intDimVal.isValid()) {
      if (dimVal.state == DimValue::UNFOLD) {
        mapDimValueEqual(dimVal, intDimVal);
      }
    }
  };
  auto isIdentityLayout = [&](Value value) {
    auto memref_type = value.getType().dyn_cast_or_null<MemRefType>();
    if (memref_type) {
      return memref_type.getLayout().isIdentity();
    }
    return false;
  };
  auto getTensorDimValues = [&](Value value) {
    llvm::Optional<SmallVector<DimValue>> result;
    auto op = value.getDefiningOp();
    if (auto from_elements = dyn_cast_or_null<tensor::FromElementsOp>(op)) {
      SmallVector<DimValue> dimValues;
      for (auto operand : from_elements.getOperands()) {
        int64_t intVal;
        if (getConstIntValue(operand.getDefiningOp(), intVal)) {
          dimValues.emplace_back(intVal);
        } else {
          dimValues.emplace_back(operand);
        }
      }
      result = std::move(dimValues);
    } else if (auto cst_shape = dyn_cast_or_null<arith::ConstantOp>(op)) {
      auto cst_attr =
          cst_shape.getValue().dyn_cast_or_null<DenseIntElementsAttr>();
      if (!cst_attr) {
        return result;
      }
      auto elem_ty = cst_attr.getType().cast<ShapedType>().getElementType();
      SmallVector<int64_t, 4> vals;
      if (elem_ty.isInteger(64) || elem_ty.isIndex()) {
        std::copy(cst_attr.getValues<int64_t>().begin(),
                  cst_attr.getValues<int64_t>().end(),
                  std::back_inserter(vals));
      } else if (elem_ty.isInteger(32)) {
        std::copy(cst_attr.getValues<int32_t>().begin(),
                  cst_attr.getValues<int32_t>().end(),
                  std::back_inserter(vals));
      } else {
        return result;
      }
      SmallVector<DimValue> dimValues;
      for (auto val : vals) {
        dimValues.emplace_back(val);
      }
      result = std::move(dimValues);
    }
    // TODO: deal with concat.
    return result;
  };

  WalkResult result = block->walk([&](Operation* op) {
    WalkResult res = WalkResult::advance();
    // TODO: analysis memref::load/store.
    if (auto dynamicReshape = dyn_cast<lmhlo::DynamicReshapeOp>(*op)) {
      Value resultDims = op->getOperand(1);
      Value result = op->getOperand(2);
      if (!isIdentityLayout(result)) {
        return WalkResult::advance();
      }
      int64_t rank = result.getType().cast<MemRefType>().getRank();
      // Analyze shape tensor.
      // TODO: we may not need this any more.
      for (int64_t i = 0; i < rank; i++) {
        DimValue dimValue;
        if (!extractLmhloValueOfDim(resultDims, i, dimValue)) {
          continue;
        }
        if (failed(buildDimValueMap(result, i, dimValue))) {
          continue;
        }
        mayCreateOrMapConstInt(dimValue);
      }
      // Analyze linearize/delinearize between input and result dims.
      // For [a, b, c] -> [d, e], if we know `c` is equal with `e`, we know that
      // `d` can be delinearize to `a` and `b`.
      Value input = op->getOperand(0);
      if (!isIdentityLayout(input)) {
        return WalkResult::advance();
      }
      int64_t in_rank = input.getType().cast<MemRefType>().getRank();
      // Check one-on-one mapping.
      DenseSet<int64_t> in_mapped;
      DenseSet<int64_t> in_non_mapped;
      DenseSet<int64_t> res_non_mapped;
      for (int64_t out_idx = 0; out_idx < rank; out_idx++) {
        bool mapped = false;
        // The first/last dim of out can only be mapped to the first/last dim of
        // input.
        int64_t begin;
        int64_t end;
        if (out_idx == 0) {
          begin = 0;
          end = 1;
        } else if (out_idx == rank - 1) {
          begin = in_rank - 1;
          end = in_rank;
        } else {
          begin = 1;
          end = in_rank - 1;
        }
        if (begin < 0) {
          continue;
        }
        for (int64_t in_idx = begin; in_idx < end; in_idx++) {
          if (in_mapped.contains(in_idx)) {
            continue;
          }
          if (isDimEqual(result, out_idx, input, in_idx)) {
            in_mapped.insert(in_idx);
            mapped = true;
            break;
          }
        }
        if (!mapped) {
          res_non_mapped.insert(out_idx);
        }
      }
      for (int64_t in_idx = 0; in_idx < in_rank; in_idx++) {
        if (in_mapped.contains(in_idx)) {
          continue;
        }
        in_non_mapped.insert(in_idx);
      }
      int64_t key;
      DenseSet<int64_t> vals;
      Value key_from;
      Value val_from;
      if (in_non_mapped.size() == 1) {
        key = *in_non_mapped.begin();
        vals = std::move(res_non_mapped);
        key_from = input;
        val_from = result;
      } else if (res_non_mapped.size() == 1) {
        key = *res_non_mapped.begin();
        vals = std::move(in_non_mapped);
        key_from = result;
        val_from = input;
      } else {
        return WalkResult::advance();
      }
      DimValue keyDimVal = getDimValue(key_from, key);
      if (!keyDimVal.isValid()) {
        return WalkResult::advance();
      }
      std::vector<DimValue> decs;
      for (auto val : vals) {
        auto dimVal = getDimValue(val_from, val);
        if (!dimVal.isValid()) {
          return WalkResult::advance();
        }
        decs.push_back(dimVal);
      }
      auto& decompose = dimValueMulDecompose_[keyDimVal];
      decompose.emplace_back(std::move(decs));
    } else if (isa<lmhlo::DynamicBroadcastInDimOp>(*op)) {
      Value resultDims = op->getOperand(1);
      Value result = op->getOperand(2);
      if (!isIdentityLayout(result)) {
        return WalkResult::advance();
      }
      int64_t rank = result.getType().cast<MemRefType>().getRank();
      // Analyze shape tensor.
      // TODO: we may not need this any more.
      for (int64_t i = 0; i < rank; i++) {
        DimValue dimValue;
        if (!extractLmhloValueOfDim(resultDims, i, dimValue)) {
          continue;
        }
        if (failed(buildDimValueMap(result, i, dimValue))) {
          continue;
        }
        mayCreateOrMapConstInt(dimValue);
      }
    } else if (auto alloc = dyn_cast<memref::AllocOp>(*op)) {
      Value result = op->getResult(0);
      auto type = result.getType().cast<MemRefType>();
      assert(type);
      for (int64_t i = 0; i < type.getRank(); i++) {
        DimValue dimVal;
        if (type.isDynamicDim(i)) {  // Only dynamic dims are operands.
          auto dyn_idx = type.getDynamicDimIndex(i);
          dimVal = DimValue(op->getOperand(dyn_idx));
        } else {
          dimVal = DimValue(type.getDimSize(i));
        }
        if (failed(buildDimValueMap(result, i, dimVal))) {
          continue;
        }
        mayCreateOrMapConstInt(dimVal);
      }
    } else if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(*op)) {
      Value result = reinterpret.getResult();
      auto sizes = reinterpret.sizes();
      for (auto en : llvm::enumerate(sizes)) {
        auto dimVal = DimValue(en.value());
        if (failed(buildDimValueMap(result, en.index(), dimVal))) {
          continue;
        }
        mayCreateOrMapConstInt(dimVal);
      }
    } else if (auto dim = dyn_cast<memref::DimOp>(*op)) {
      Value result = dim.getResult();
      Value value = op->getOperand(0);
      Value index = op->getOperand(1);
      int64_t index_val;
      if (getConstIntValue(index.getDefiningOp(), index_val)) {
        buildDimValueMap(value, index_val, DimValue(result));
      }
    } else if (auto indexCast = dyn_cast<arith::IndexCastOp>(*op)) {
      // Only deal with scalar here.
      auto input = indexCast.getOperand();
      auto in_ty = input.getType().dyn_cast_or_null<RankedTensorType>();
      if (in_ty != nullptr && in_ty.getRank() > 0) {
        return WalkResult::advance();
      }
      DimValue operand(input);
      DimValue result(indexCast.getResult());

      if (failed(mapDimValueEqual(operand, result))) {
        return WalkResult::advance();
      }
      mayCreateOrMapConstInt(operand);
      mayCreateOrMapConstInt(result);
    } else if (isa<mhlo::DynamicReshapeOp, mhlo::DynamicBroadcastInDimOp>(op)) {
      auto dimValues = getTensorDimValues(op->getOperand(1));
      if (!dimValues.has_value()) {
        return WalkResult::advance();
      }
      Value result = op->getResult(0);
      for (auto& val : llvm::enumerate(*dimValues)) {
        auto& dimVal = val.value();
        if (failed(buildDimValueMap(result, val.index(), dimVal))) {
          continue;
        }
        mayCreateOrMapConstInt(dimVal);
      }
    } else if (auto dynSlice = dyn_cast<mhlo::RealDynamicSliceOp>(*op)) {
      auto starts = getTensorDimValues(dynSlice.getStartIndices());
      auto limits = getTensorDimValues(dynSlice.getLimitIndices());
      auto strides = getTensorDimValues(dynSlice.getStrides());
      Value result = op->getResult(0);
      auto out_ty = result.getType().dyn_cast_or_null<RankedTensorType>();
      if (!starts.has_value() || !limits.has_value() || !strides.has_value() ||
          !out_ty) {
        return WalkResult::advance();
      }
      assert(out_ty.getRank() == (*starts).size());
      for (int64_t i = 0; i < out_ty.getRank(); i++) {
        if (!(*starts)[i].isFold() || !(*strides)[i].isFold()) {
          continue;
        }
        int64_t start = (*starts)[i].foldVal;
        int64_t stride = (*strides)[i].foldVal;
        auto limit = (*limits)[i];
        if (limit.isUnfold() && (start == 0) && (stride == 1)) {
          // Start is 0, stride is 1. The limit value is the dim-value of this
          // dimension.
          buildDimValueMap(result, i, limit);
        } else if (limit.isFold()) {
          // size = (limit - start + stride - 1) / stride
          auto size = (limit.foldVal - start + stride - 1) / stride;
          buildDimValueMap(result, i, DimValue(size));
        }
      }
    }
    return res;
  });
  if (result.wasInterrupted()) {
    return failure();
  }

  return success();
}

LogicalResult ShapeAnalysisDeprecated::buildDimValueMap(Value operand,
                                                        int64_t dim,
                                                        DimValue dimValue) {
  auto symbol = getDim(operand, dim);
  if (symbol == nullptr) {
    return failure();
  }

  auto rootDimVal = dimSymbol2DimValue_.find(symbol);
  if (rootDimVal != dimSymbol2DimValue_.end() &&
      rootDimVal->second != dimValue) {
    mapDimValueEqual(rootDimVal->second, dimValue);
  }

  auto root_val = getRootDimValue(dimValue);
  dimSymbol2DimValue_[symbol] = root_val;

  return success();
}

SymbolShape* ShapeAnalysisDeprecated::getShape(Value value) {
  auto it = shapeMap_.find(value);
  if (it == shapeMap_.end()) return nullptr;

  for (int64_t i = 0; i < it->second.rank(); ++i) {
    it->second.setSymbolDim(i, getDim(value, i));
  }
  auto rank = it->second.rank();
  assert(rank >= 0 && rank <= 1000);
  return &it->second;
}

SymbolDim* ShapeAnalysisDeprecated::getRootDim(SymbolDim* symbolDim) {
  assert(symbolDim != nullptr);
  SymbolDim* parentSymbolDim = symbolDim;
  do {
    symbolDim = parentSymbolDim;
    int64_t dimIdx = dimMap_[symbolDim];
    parentSymbolDim = dimVec_[dimIdx].get();
  } while (parentSymbolDim != symbolDim);

  return parentSymbolDim;
}

SymbolDim* ShapeAnalysisDeprecated::getDim(Value value, int64_t dim) {
  auto it = shapeMap_.find(value);
  if (it == shapeMap_.end()) return nullptr;
  if (it->second.rank() <= 0 || dim < 0 || dim > it->second.rank() - 1) {
    return nullptr;
  }
  SymbolDim* symbolDim = it->second.getSymbolDim(dim);
  assert(symbolDim != nullptr);

  symbolDim = getRootDim(symbolDim);
  it->second.setSymbolDim(dim, symbolDim);

  return symbolDim;
}

DimValue ShapeAnalysisDeprecated::getRootDimValue(DimValue dimValue) {
  return dimValueEquivalence_.getOrInsertLeaderValue(dimValue);
}

DimValue ShapeAnalysisDeprecated::getDimValue(SymbolDim* symbolDim) {
  if (symbolDim == nullptr) {
    return DimValue(Value(nullptr));
  }
  auto it = dimSymbol2DimValue_.find(symbolDim);
  if (it == dimSymbol2DimValue_.end()) {
    return DimValue(Value(nullptr));
  }
  // update root of dim-value.
  DimValue rootDimValue = getRootDimValue(it->second);
  if (rootDimValue != it->second) {
    dimSymbol2DimValue_[symbolDim] = rootDimValue;
  }

  return rootDimValue;
}

DimValue ShapeAnalysisDeprecated::getDimValue(Value operand, int64_t dim) {
  auto ty = operand.getType().dyn_cast<ShapedType>();
  if (!ty.hasRank() || ty.getRank() <= 0 || dim >= ty.getRank() || dim < 0) {
    return DimValue(Value(nullptr));
  }

  auto symbolDim = getDim(operand, dim);
  // Check static dim.
  if (ty && !ty.isDynamicDim(dim)) {
    auto dimVal = DimValue(ty.getDimSize(dim));
    // Update symble2dimval mapping.
    if (symbolDim != nullptr) {
      auto it = dimSymbol2DimValue_.find(symbolDim);
      if (it == dimSymbol2DimValue_.end()) {
        dimSymbol2DimValue_[symbolDim] = dimVal;
      } else {
        mapDimValueEqual(dimVal, it->second);
      }
    }
    return dimVal;
  }
  return getDimValue(symbolDim);
}

LogicalResult ShapeAnalysisDeprecated::mapDimEqual(SymbolDim* lhs,
                                                   SymbolDim* rhs) {
  if (lhs == nullptr || rhs == nullptr) {
    return success();
  }
  int64_t lhsIdx = dimMap_[getRootDim(lhs)];
  int64_t rhsIdx = dimMap_[getRootDim(rhs)];

  lhs = dimVec_[lhsIdx].get();
  rhs = dimVec_[rhsIdx].get();

  // let the root with smaller idx to be the root of the merged group.
  if (lhsIdx <= rhsIdx) {
    dimMap_[rhs] = lhsIdx;
    return lhs->Merge(rhs);
  } else {
    dimMap_[lhs] = rhsIdx;
    return rhs->Merge(lhs);
  }
}

LogicalResult ShapeAnalysisDeprecated::mapShapeEqual(SymbolShape* lhs,
                                                     SymbolShape* rhs) {
  if (!lhs || !rhs || lhs->rank() != rhs->rank()) return failure();

  for (auto&& en : llvm::zip(lhs->getSymbolDims(), rhs->getSymbolDims())) {
    if (failed(mapDimEqual(std::get<0>(en), std::get<1>(en)))) return failure();
  }
  return success();
}

LogicalResult ShapeAnalysisDeprecated::mapShapeEqual(Value lhs, Value rhs) {
  SymbolShape* lhsShape = getShape(lhs);
  SymbolShape* rhsShape = getShape(rhs);

  auto mapValueEquivalent = [&](Value lhs, Value rhs,
                                EquivalenceClasses<ValueWrapper>& impl) {
    if (!impl.isEquivalent(ValueWrapper(lhs), ValueWrapper(rhs))) {
      impl.unionSets(ValueWrapper(lhs), ValueWrapper(rhs));
    }
  };

  mapValueEquivalent(lhs, rhs, valueWithEqualShape_);
  mapValueEquivalent(lhs, rhs, valueWithSameElements_);

  // Map or propagate tensor shape equality.
  auto lhs_tensor_shape_iter = value2ShapeValue_.find(lhs);
  auto rhs_tensor_shape_iter = value2ShapeValue_.find(rhs);
  bool lhs_found = lhs_tensor_shape_iter != value2ShapeValue_.end();
  bool rhs_found = rhs_tensor_shape_iter != value2ShapeValue_.end();
  if (lhs_found && rhs_found) {
    // Map tensor shape equality.
    auto lhs_tensor_shape = lhs_tensor_shape_iter->second;
    auto rhs_tensor_shape = rhs_tensor_shape_iter->second;
    mapValueEquivalent(lhs_tensor_shape, rhs_tensor_shape, shapeValueEqual_);
  } else if (lhs_found) {
    // Propagate tensor shape equality .
    auto lhs_tensor_shape = lhs_tensor_shape_iter->second;
    value2ShapeValue_[rhs] = lhs_tensor_shape;
  } else if (rhs_found) {
    // Propagate tensor shape equality .
    auto rhs_tensor_shape = rhs_tensor_shape_iter->second;
    value2ShapeValue_[lhs] = rhs_tensor_shape;
  }

  return mapShapeEqual(lhsShape, rhsShape);
}

LogicalResult ShapeAnalysisDeprecated::mapDimEqual(Value lhs, int64_t lhsIdx,
                                                   Value rhs, int64_t rhsIdx) {
  SymbolDim* lhsDim = getDim(lhs, lhsIdx);
  SymbolDim* rhsDim = getDim(rhs, rhsIdx);
  return mapDimEqual(lhsDim, rhsDim);
}

// TODO: map const and constOp equal.
LogicalResult ShapeAnalysisDeprecated::mapDimValueEqual(DimValue lhs,
                                                        DimValue rhs) {
  if (!dimValueEquivalence_.isEquivalent(lhs, rhs)) {
    dimValueEquivalence_.unionSets(lhs, rhs);
  }
  return success();
}

bool ShapeAnalysisDeprecated::isDimEqual(Value lhs, int64_t lhsDim, Value rhs,
                                         int64_t rhsDim) {
  auto lhs_ty = lhs.getType().dyn_cast<ShapedType>();
  auto rhs_ty = rhs.getType().dyn_cast<ShapedType>();
  if (!lhs_ty.hasRank() || !rhs_ty.hasRank()) {
    return false;
  }
  if (lhsDim >= lhs_ty.getRank() || rhsDim >= rhs_ty.getRank()) {
    return false;
  }

  // Check static dims.
  if (lhs_ty && rhs_ty && !lhs_ty.isDynamicDim(lhsDim) &&
      !rhs_ty.isDynamicDim(rhsDim)) {
    return lhs_ty.getDimSize(lhsDim) == rhs_ty.getDimSize(rhsDim);
  }

  return (getDim(lhs, lhsDim) == getDim(rhs, rhsDim)) ||
         (getDimValue(lhs, lhsDim) == getDimValue(rhs, rhsDim));
}

SmallVector<std::vector<DimValue>> ShapeAnalysisDeprecated::getDimDecompose(
    Value value, int64_t index) {
  auto dim_val = getDimValue(value, index);
  auto iter = dimValueMulDecompose_.find(dim_val);
  return (iter == dimValueMulDecompose_.end())
             ? SmallVector<std::vector<DimValue>>()
             : iter->second;
}

bool ShapeAnalysisDeprecated::isShapeEqual(Value lhs, Value rhs) {
  SymbolShape* lhsShape = getShape(lhs);
  SymbolShape* rhsShape = getShape(rhs);
  if (!lhsShape || !lhsShape) return false;
  if (lhsShape->rank() != rhsShape->rank()) return false;
  for (int i = 0; i < lhsShape->rank(); ++i) {
    if (!isDimEqual(lhs, i, rhs, i)) return false;
  }
  return true;
}

bool ShapeAnalysisDeprecated::isShapeValueEqual(Value lhs, Value rhs) {
  return shapeValueEqual_.isEquivalent(ValueWrapper(lhs), ValueWrapper(rhs));
}

bool ShapeAnalysisDeprecated::isDimValueEqual(DimValue lhs, DimValue rhs) {
  return dimValueEquivalence_.isEquivalent(lhs, rhs);
}

LogicalResult ShapeAnalysisDeprecated::applyOpConstraint(Operation* op) {
  if (op->getDialect() == op->getContext()->getLoadedDialect("tensor")) {
    return applyTensorOpConstraint(op);
  } else if (op->getDialect() == op->getContext()->getLoadedDialect("mhlo") ||
             op->getDialect() ==
                 op->getContext()->getLoadedDialect("mhlo_disc")) {
    return applyMhloOpConstraint(op);
  } else if (op->getDialect() == op->getContext()->getLoadedDialect("lmhlo") ||
             op->getDialect() ==
                 op->getContext()->getLoadedDialect("lmhlo_disc")) {
    return applyLmhloOpConstraint(op);
  }

  return success();
}

LogicalResult ShapeAnalysisDeprecated::applyTensorOpConstraint(Operation* op) {
  if (isa<tensor::CastOp>(op)) {
    return mapShapeEqual(op->getResult(0), op->getOperand(0));
  }
  return success();
}

LogicalResult ShapeAnalysisDeprecated::applyMhloOpConstraint(Operation* op) {
  if (op->hasTrait<mlir::OpTrait::SameOperandsAndResultType>() ||
      op->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>()) {
    Value ref;
    if (op->getNumOperands() > 0) ref = op->getOperands().front();
    if (op->getNumResults() > 0) ref = op->getResults().front();
    if (!ref) return success();
    for (Value operand : op->getOperands()) mapShapeEqual(ref, operand);
    for (Value result : op->getResults()) mapShapeEqual(ref, result);
    return success();
  }

  if (op->hasTrait<mlir::OpTrait::SameTypeOperands>() ||
      op->hasTrait<mlir::OpTrait::SameOperandsShape>() ||
      op->hasTrait<mlir::OpTrait::Elementwise>()) {
    if (op->getNumOperands() == 0) return success();
    Value ref = op->getOperands().front();
    for (Value operand : op->getOperands()) mapShapeEqual(ref, operand);
  }

  if (auto transpose = dyn_cast<mhlo::TransposeOp>(op)) {
    Value operand = op->getOperand(0);
    Value result = op->getResult(0);
    for (auto& en :
         llvm::enumerate(transpose.getPermutation().getValues<int64_t>())) {
      mapDimEqual(operand, en.value(), result, en.index());
    }
  } else if (auto concat = dyn_cast<mhlo::ConcatenateOp>(op)) {
    Value result = op->getResult(0);
    int64_t axis = concat.getDimension();
    int64_t rank = result.getType().cast<RankedTensorType>().getRank();
    for (Value operand : op->getOperands()) {
      for (int64_t i = 0; i < rank; ++i) {
        if (i == axis) continue;
        mapDimEqual(operand, i, result, i);
      }
    }
  } else if (auto reduce = dyn_cast<mhlo::ReduceOp>(op)) {
    Value operand = op->getOperand(0);
    Value result = op->getResult(0);
    int64_t rank = operand.getType().cast<RankedTensorType>().getRank();
    int64_t resultDimIdx = 0;
    for (int64_t i = 0; i < rank; ++i) {
      auto reduceDims = reduce.getDimensions().getValues<int64_t>();
      if (std::find(reduceDims.begin(), reduceDims.end(), i) !=
          reduceDims.end()) {
        continue;
      }
      mapDimEqual(operand, i, result, resultDimIdx++);
    }
  } else if (auto broadcast_in_dim =
                 dyn_cast<mhlo::DynamicBroadcastInDimOp>(op)) {
    auto operand = broadcast_in_dim->getOperand(0);
    Value result = op->getResult(0);
    auto ty = operand.getType().dyn_cast<ShapedType>();
    assert(ty);
    for (auto& dim : llvm::enumerate(
             broadcast_in_dim.getBroadcastDimensions().getValues<int64_t>())) {
      // Deal with non-static & non-one dim.
      if (!ty.isDynamicDim(dim.index()) && ty.getDimSize(dim.index()) != 1) {
        mapDimEqual(operand, dim.index(), result, dim.value());
      }
    }

    // Map shape equality according to tensor shape.
    auto in_tensor_shape = value2ShapeValue_.find(operand);
    // auto result_tensor_shape = value2ShapeValue_.find(result);
    if (in_tensor_shape != value2ShapeValue_.end() &&
        // result_tensor_shape != value2ShapeValue_.end() &&
        isShapeValueEqual(in_tensor_shape->second, op->getOperand(1))) {
      mapShapeEqual(operand, result);
    }
  } else if (auto dot = dyn_cast<mhlo::DotOp>(op)) {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Value result = op->getResult(0);
    // Contracting dimension.
    mapDimEqual(lhs, 1, rhs, 0);
    // M and N dimensions
    mapDimEqual(lhs, 0, result, 0);
    mapDimEqual(rhs, 1, result, 1);
  } else if (auto dot_general = dyn_cast<mhlo::DotGeneralOp>(op)) {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    auto dim_numbers = dot_general.getDotDimensionNumbers();
    DenseSet<int64_t> lhs_contract_batch_dims;
    DenseSet<int64_t> rhs_contract_batch_dims;
    // Contracting dimensions.
    auto lhs_contracting_dims = dim_numbers.getLhsContractingDimensions();
    auto rhs_contracting_dims = dim_numbers.getRhsContractingDimensions();
    assert(lhs_contracting_dims.size() == rhs_contracting_dims.size());
    for (int64_t i = 0; i < lhs_contracting_dims.size(); i++) {
      int64_t lhs_dim = lhs_contracting_dims[i];
      int64_t rhs_dim = rhs_contracting_dims[i];
      mapDimEqual(lhs, lhs_dim, rhs, rhs_dim);
      lhs_contract_batch_dims.insert(lhs_dim);
      rhs_contract_batch_dims.insert(rhs_dim);
    }
    // Batching dimensions.
    auto lhs_batching_dims = dim_numbers.getLhsBatchingDimensions();
    auto rhs_batching_dims = dim_numbers.getRhsBatchingDimensions();
    assert(lhs_batching_dims.size() == rhs_batching_dims.size());
    for (int64_t i = 0; i < lhs_batching_dims.size(); i++) {
      int64_t lhs_dim = lhs_batching_dims[i];
      int64_t rhs_dim = rhs_batching_dims[i];
      mapDimEqual(lhs, lhs_dim, rhs, rhs_dim);
      lhs_contract_batch_dims.insert(lhs_dim);
      rhs_contract_batch_dims.insert(rhs_dim);
    }
    // Resulting dimensions. It follows that the resulting dimension number
    // starts with the batch dimension, then the 'lhs' non-contracting/non-batch
    // dimension, and finally the 'rhs' non-contracting/non-batch dimension.
    Value result = op->getResult(0);
    for (int64_t i = 0; i < lhs_batching_dims.size(); i++) {
      int64_t lhs_dim = lhs_batching_dims[i];
      mapDimEqual(lhs, lhs_dim, result, i);
    }
    SmallVector<std::pair<Value, int64_t>, 4> mn_values;
    for (int64_t i = 0; i < lhs.getType().cast<RankedTensorType>().getRank();
         i++) {
      if (lhs_contract_batch_dims.find(i) == lhs_contract_batch_dims.end()) {
        mn_values.emplace_back(lhs, i);
      }
    }
    for (int64_t i = 0; i < rhs.getType().cast<RankedTensorType>().getRank();
         i++) {
      if (rhs_contract_batch_dims.find(i) == rhs_contract_batch_dims.end()) {
        mn_values.emplace_back(rhs, i);
      }
    }
    for (int64_t i = 0; i < mn_values.size(); i++) {
      mapDimEqual(mn_values[i].first, mn_values[i].second, result,
                  i + lhs_batching_dims.size());
    }
  } else if (auto einsum = dyn_cast<mhlo::EinsumOp>(op)) {
    StringRef equation = einsum.getEinsumConfig();
    llvm::SmallDenseMap<char, llvm::SmallDenseMap<EquationVariable, size_t>>
        all_tokens;
    if (!parseEinsumEquation(equation, all_tokens, nullptr, nullptr, nullptr)) {
      return einsum.emitError("unexpected character in einsum equation");
    }
    for (auto token : all_tokens) {
      SmallVector<std::pair<Value, int64_t>> equalValues;
      for (auto item : token.second) {
        if (item.first == kIsLhs) {
          equalValues.push_back(std::make_pair(einsum.getLhs(), item.second));
        } else if (item.first == kIsRhs) {
          equalValues.push_back(std::make_pair(einsum.getRhs(), item.second));
        } else {
          // kIsResult
          equalValues.push_back(
              std::make_pair(einsum.getResult(), item.second));
        }
      }
      if (equalValues.size() >= 2) {
        mapDimEqual(equalValues[0].first, equalValues[0].second,
                    equalValues[1].first, equalValues[1].second);
      }
      if (equalValues.size() == 3) {
        mapDimEqual(equalValues[0].first, equalValues[0].second,
                    equalValues[2].first, equalValues[2].second);
      }
    }
  } else if (auto clamp = dyn_cast<mhlo::ClampOp>(op)) {
    int64_t min_rank =
        clamp.getMin().getType().cast<RankedTensorType>().getRank();
    int64_t max_rank =
        clamp.getMax().getType().cast<RankedTensorType>().getRank();
    if (min_rank != 0) {
      mapShapeEqual(clamp.getOperand(), clamp.getMin());
    }
    if (max_rank != 0) {
      mapShapeEqual(clamp.getOperand(), clamp.getMax());
    }
    mapShapeEqual(clamp.getOperand(), op->getResult(0));
  } else if (auto select = dyn_cast<mhlo::SelectOp>(op)) {
    int64_t pred_rank =
        select.getPred().getType().cast<RankedTensorType>().getRank();
    int64_t true_rank =
        select.getOnTrue().getType().cast<RankedTensorType>().getRank();
    int64_t false_rank =
        select.getOnFalse().getType().cast<RankedTensorType>().getRank();
    if (pred_rank != 0) {
      mapShapeEqual(select.getPred(), select.getResult());
    }
    if (true_rank != 0) {
      mapShapeEqual(select.getOnTrue(), select.getResult());
    }
    if (false_rank != 0) {
      mapShapeEqual(select.getOnFalse(), select.getResult());
    }
  } else if (isa<mhlo::GatherOp, mhlo::DynamicGatherOp>(op)) {
    auto gather = dyn_cast<mhlo::GatherOp>(op);
    auto d_gather = dyn_cast<mhlo::DynamicGatherOp>(op);
    auto dimension_numbers =
        gather ? gather.getDimensionNumbers() : d_gather.getDimensionNumbers();
    Value in = op->getOperand(0);
    Value result = op->getResult(0);
    auto in_ty = in.getType().dyn_cast<RankedTensorType>();
    if (in_ty == nullptr) {
      return success();
    }

    auto collapsed_slice_dims = dimension_numbers.getCollapsedSliceDims();
    auto offset_dims = dimension_numbers.getOffsetDims();
    // Map index of offset_dims to operand.
    SmallVector<int64_t, 4> remapped_offset_dims;
    DenseSet<int64_t> collapsed_set(collapsed_slice_dims.begin(),
                                    collapsed_slice_dims.end());
    for (int64_t i = 0; i < in_ty.getRank(); i++) {
      if (collapsed_set.contains(i)) {
        continue;
      }
      remapped_offset_dims.push_back(i);
    }
    for (auto offset : llvm::enumerate(offset_dims)) {
      mapDimEqual(in, remapped_offset_dims[offset.index()], result,
                  offset.value());
    }
  }

  return success();
}

LogicalResult ShapeAnalysisDeprecated::applyLmhloOpConstraint(Operation* op) {
  auto mapValueEquivalent = [&](Value lhs, Value rhs,
                                EquivalenceClasses<ValueWrapper>& impl) {
    if (!impl.isEquivalent(ValueWrapper(lhs), ValueWrapper(rhs))) {
      impl.unionSets(ValueWrapper(lhs), ValueWrapper(rhs));
    }
  };

  if (op->hasTrait<mlir::OpTrait::SameTypeOperands>() ||
      op->hasTrait<mlir::OpTrait::SameOperandsShape>() ||
      op->hasTrait<mlir::OpTrait::Elementwise>()) {
    if (op->getNumOperands() == 0) return success();
    Value ref = op->getOperands().front();
    for (Value operand : op->getOperands().drop_front()) {
      mapShapeEqual(ref, operand);
    }
  }

  if (auto transpose = dyn_cast<lmhlo::TransposeOp>(op)) {
    Value operand = op->getOperand(0);
    Value result = op->getOperand(1);
    for (auto& en :
         llvm::enumerate(transpose.getPermutation().getValues<int64_t>())) {
      mapDimEqual(operand, en.value(), result, en.index());
    }
    mapValueEquivalent(operand, result, valueWithSameElements_);
  } else if (auto concat = dyn_cast<lmhlo::ConcatenateOp>(op)) {
    Value result = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    int64_t axis = concat.getDimension();
    int64_t rank = result.getType().cast<MemRefType>().getRank();
    for (Value operand : op->getOperands().drop_back()) {
      for (int64_t i = 0; i < rank; ++i) {
        if (i == axis) continue;
        mapDimEqual(operand, i, result, i);
      }
    }
  } else if (auto reduce = dyn_cast<lmhlo::ReduceOp>(op)) {
    Value operand = op->getOperand(0);
    Value result = op->getOperand(2);
    int64_t rank = operand.getType().cast<MemRefType>().getRank();
    int64_t resultDimIdx = 0;
    for (int64_t i = 0; i < rank; ++i) {
      auto reduceDims = reduce.getDimensions().getValues<int64_t>();
      if (std::find(reduceDims.begin(), reduceDims.end(), i) !=
          reduceDims.end()) {
        continue;
      }
      mapDimEqual(operand, i, result, resultDimIdx++);
    }
  } else if (auto broadcast_in_dim =
                 dyn_cast<lmhlo::DynamicBroadcastInDimOp>(op)) {
    auto operand = broadcast_in_dim->getOperand(0);
    Value result = op->getOperand(2);
    auto ty = operand.getType().dyn_cast<ShapedType>();
    assert(ty);
    for (auto& dim : llvm::enumerate(
             broadcast_in_dim.getBroadcastDimensions().getValues<int64_t>())) {
      // Deal with non-static & non-one dim.
      if (!ty.isDynamicDim(dim.index()) && ty.getDimSize(dim.index()) != 1) {
        mapDimEqual(operand, dim.index(), result, dim.value());
      }
    }
  } else if (auto dot_general = dyn_cast<lmhlo::DotGeneralOp>(op)) {
    // Note that there should be no lmhlo::DotOp as we have already converted
    // DotOp to DotGeneralOp with mhlo Dialect already. Thus we only deal with
    // DotGeneralOp for lmhlo Dialect.
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    auto dim_numbers = dot_general.getDotDimensionNumbers();
    DenseSet<int64_t> lhs_contract_batch_dims;
    DenseSet<int64_t> rhs_contract_batch_dims;
    // Contracting dimensions.
    auto lhs_contracting_dims = dim_numbers.getLhsContractingDimensions();
    auto rhs_contracting_dims = dim_numbers.getRhsContractingDimensions();
    assert(lhs_contracting_dims.size() == rhs_contracting_dims.size());
    for (int64_t i = 0; i < lhs_contracting_dims.size(); i++) {
      int64_t lhs_dim = lhs_contracting_dims[i];
      int64_t rhs_dim = rhs_contracting_dims[i];
      mapDimEqual(lhs, lhs_dim, rhs, rhs_dim);
      lhs_contract_batch_dims.insert(lhs_dim);
      rhs_contract_batch_dims.insert(rhs_dim);
    }
    // Batching dimensions.
    auto lhs_batching_dims = dim_numbers.getLhsBatchingDimensions();
    auto rhs_batching_dims = dim_numbers.getRhsBatchingDimensions();
    assert(lhs_batching_dims.size() == rhs_batching_dims.size());
    for (int64_t i = 0; i < lhs_batching_dims.size(); i++) {
      int64_t lhs_dim = lhs_batching_dims[i];
      int64_t rhs_dim = rhs_batching_dims[i];
      mapDimEqual(lhs, lhs_dim, rhs, rhs_dim);
      lhs_contract_batch_dims.insert(lhs_dim);
      rhs_contract_batch_dims.insert(rhs_dim);
    }
    // Resulting dimensions. It follows that the resulting dimension number
    // starts with the batch dimension, then the 'lhs' non-contracting/non-batch
    // dimension, and finally the 'rhs' non-contracting/non-batch dimension.
    // Value result = op->getResult(0);
    Value result = op->getOperand(2);
    for (int64_t i = 0; i < lhs_batching_dims.size(); i++) {
      int64_t lhs_dim = lhs_batching_dims[i];
      mapDimEqual(lhs, lhs_dim, result, i);
    }
    SmallVector<std::pair<Value, int64_t>, 4> mn_values;
    for (int64_t i = 0; i < lhs.getType().cast<MemRefType>().getRank(); i++) {
      if (lhs_contract_batch_dims.find(i) == lhs_contract_batch_dims.end()) {
        mn_values.emplace_back(lhs, i);
      }
    }
    for (int64_t i = 0; i < rhs.getType().cast<MemRefType>().getRank(); i++) {
      if (rhs_contract_batch_dims.find(i) == rhs_contract_batch_dims.end()) {
        mn_values.emplace_back(rhs, i);
      }
    }
    for (int64_t i = 0; i < mn_values.size(); i++) {
      mapDimEqual(mn_values[i].first, mn_values[i].second, result,
                  i + lhs_batching_dims.size());
    }
  } else if (auto clamp = dyn_cast<lmhlo::ClampOp>(op)) {
    int64_t min_rank = clamp.getMin().getType().cast<MemRefType>().getRank();
    int64_t max_rank = clamp.getMax().getType().cast<MemRefType>().getRank();
    if (min_rank != 0) {
      mapShapeEqual(clamp.getOperand(), clamp.getMin());
    }
    if (max_rank != 0) {
      mapShapeEqual(clamp.getOperand(), clamp.getMax());
    }
    Value result = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    mapShapeEqual(clamp.getOperand(), result);
  } else if (auto select = dyn_cast<lmhlo::SelectOp>(op)) {
    int64_t pred_rank = select.getPred().getType().cast<MemRefType>().getRank();
    int64_t true_rank =
        select.getOnTrue().getType().cast<MemRefType>().getRank();
    int64_t false_rank =
        select.getOnFalse().getType().cast<MemRefType>().getRank();
    Value result = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    if (pred_rank != 0) {
      mapShapeEqual(select.getPred(), result);
    }
    if (true_rank != 0) {
      mapShapeEqual(select.getOnTrue(), result);
    }
    if (false_rank != 0) {
      mapShapeEqual(select.getOnFalse(), result);
    }
  } else if (isa<lmhlo::DynamicReshapeOp, lmhlo::ReshapeOp>(op)) {
    Value input = op->getOperand(0);
    // The last operand is the output memref by design
    int64_t num_operand = op->getNumOperands();
    Value output = op->getOperand(num_operand - 1);
    mapValueEquivalent(input, output, valueWithSameElements_);
  }

  return success();
}

LogicalResult ShapeAnalysisDeprecated::applyDimEqualWithDimValue() {
  std::unordered_map<DimValue, SmallVector<SymbolDim*, 4>, DimValueHash>
      dimValue2DimSymbols;
  for (auto symbol_value : dimSymbol2DimValue_) {
    auto dimSymbol = symbol_value.first;
    DimValue valueRoot = getRootDimValue(symbol_value.second);
    auto& dims = dimValue2DimSymbols[valueRoot];
    dims.push_back(dimSymbol);
  }
  for (auto valueDimsPair : dimValue2DimSymbols) {
    auto dims = valueDimsPair.second;
    auto dim = dims[0];
    for (int64_t i = 1; i < dims.size(); i++) {
      if (failed(mapDimEqual(dim, dims[i]))) {
        return failure();
      }
    }
  }

  return success();
}

LogicalResult ShapeAnalysisDeprecated::buildRegionMulDecompose(Region* region) {
  if (region->getBlocks().size() != 1) {
    return region->getParentOp()->emitError(
        "only single block region is supported");
  }
  for (Block& block : *region) {
    if (failed(buildBlockMulDecompose(&block))) return failure();
  }
  return success();
}

LogicalResult ShapeAnalysisDeprecated::buildBlockMulDecompose(Block* block) {
  // mapping each op inside the block
  WalkResult result = block->walk([&](Operation* op) {
    if (auto muli = dyn_cast_or_null<arith::MulIOp>(op)) {
      DimValue lhs(muli.getOperand(0));
      DimValue rhs(muli.getOperand(1));
      DimValue result(muli.getResult());
      auto& decompose = dimValueMulDecompose_[result];
      decompose.emplace_back(std::vector<DimValue>({lhs, rhs}));
    } else if (isa<lmhlo::DynamicReshapeOp, lmhlo::ReshapeOp>(op)) {
      Value in_value = op->getOperand(0);
      // The last operand is the output memref by design
      int64_t num_operand = op->getNumOperands();
      Value out_value = op->getOperand(num_operand - 1);
      SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> dim_eq;
      int64_t in_rank = in_value.getType().cast<MemRefType>().getRank();
      int64_t out_rank = out_value.getType().cast<MemRefType>().getRank();
      if (!extractContinuousDimEqualInfo(in_value, out_value, dim_eq)) {
        return WalkResult::advance();
      }
      DenseSet<int64_t> in_matched;
      DenseSet<int64_t> out_matched;
      for (auto& in_out : dim_eq) {
        in_matched.insert(in_out.first.begin(), in_out.first.end());
        out_matched.insert(in_out.second.begin(), in_out.second.end());
      }
      DenseSet<int64_t> in_non_matched;
      DenseSet<int64_t> out_non_matched;

      if (in_rank == in_matched.size() || out_rank == out_matched.size()) {
        return WalkResult::advance();
      } else if (in_rank == in_matched.size() + 1 ||
                 out_rank == out_matched.size() + 1) {
        // Only one dim is not matched in either input or output, thus we can
        // build decompose of this dim.
        for (int64_t i = 0; i < in_rank; i++) {
          if (in_matched.contains(i)) {
            continue;
          }
          in_non_matched.insert(i);
        }
        for (int64_t i = 0; i < out_rank; i++) {
          if (out_matched.contains(i)) {
            continue;
          }
          out_non_matched.insert(i);
        }
        if (in_non_matched.size() == 1 && out_non_matched.size() == 1) {
          mapDimEqual(in_value, *(in_non_matched.begin()), out_value,
                      *(out_non_matched.begin()));
        } else {
          int64_t single_dim = (in_non_matched.size() == 1)
                                   ? *(in_non_matched.begin())
                                   : *(out_non_matched.begin());
          Value single_val =
              (in_non_matched.size() == 1) ? in_value : out_value;
          DenseSet<int64_t>* decom_dims =
              (in_non_matched.size() == 1) ? &out_non_matched : &in_non_matched;
          Value decom_val = (in_non_matched.size() == 1) ? out_value : in_value;
          DimValue key = getDimValue(single_val, single_dim);
          if (!key.isValid()) {
            return WalkResult::advance();
          }
          std::vector<DimValue> decom_dimvals;
          for (auto dim : *decom_dims) {
            auto val = getDimValue(decom_val, dim);
            if (!val.isValid()) {
              return WalkResult::advance();
            }
            decom_dimvals.push_back(val);
          }
          auto& decompose = dimValueMulDecompose_[key];
          decompose.emplace_back(std::move(decom_dimvals));
        }
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  } else {
    return success();
  }
}

LogicalResult ShapeAnalysisDeprecated::visitSymbolShapes(
    SymbolShapeVisitor visitor) {
  for (auto it = shapeMap_.begin(), stop = shapeMap_.end(); it != stop; ++it) {
    Value value = it->first;
    SymbolShape& symbolShape = it->second;
    OpBuilder b(op_);
    Location loc = op_->getLoc();
    Operation* definingOp = value.getDefiningOp();
    if (!definingOp) {
      Block* block = value.dyn_cast<BlockArgument>().getOwner();
      loc = block->getParentOp()->getLoc();
      b.setInsertionPoint(block, block->begin());
    } else {
      loc = definingOp->getLoc();
      b.setInsertionPointAfter(definingOp);
    }
    if (failed(visitor(b, loc, value))) return failure();
  }
  return success();
}

LogicalResult ShapeAnalysisDeprecated::buildSymbolDimInstances(
    DenseMap<SymbolDim*, SmallVector<Value>>& symbolDim2Instances,
    DenseMap<Value, SmallVector<Value>>& shapedValue2Dims) {
  // Instantiate each symbol dim and group all instances for each symbol dim.
  SymbolShapeVisitor visitor = [&](OpBuilder& b, Location& loc, Value value) {
    SymbolShape* symbolShape = getShape(value);
    auto& dims = shapedValue2Dims[value];
    for (int64_t i = 0, rank = symbolShape->rank(); i < rank; ++i) {
      Value dimSize = b.create<tensor::DimOp>(loc, value, i);
      SymbolDim* symbolDim = symbolShape->getSymbolDim(i);
      assert(symbolDim);
      auto& instances = symbolDim2Instances[symbolDim];
      instances.push_back(dimSize);
      dims.push_back(dimSize);
    }
    return success();
  };
  return visitSymbolShapes(visitor);
}

LogicalResult ShapeAnalysisDeprecated::buildSymbolDimInstancesDominantMap(
    DenseMap<SymbolDim*, SmallVector<Value>>& instanceMap,
    DenseMap<SymbolDim*, DenseMap<Value, Value>>& dominantMap) {
  DominanceInfo dominanceInfo(op_);
  for (auto& it : instanceMap) {
    auto& symbolDim = it.first;
    auto& instances = it.second;
    auto& dominants = dominantMap[symbolDim];

    // in normal cases, there should be only one root, aka, dominant value
    // of all the values of instances
    SmallVector<Value> roots;
    for (Value v : instances) {
      bool is_root = true;
      for (Value other : instances) {
        if (v == other) continue;
        if (dominanceInfo.dominates(other, v.getDefiningOp())) {
          is_root = false;
          continue;
        }
      }
      if (is_root) {
        roots.push_back(v);
      }
    }
    // we should let as much values as possible to be dominated by a same root
    for (Value root : roots) {
      for (Value v : instances) {
        if (dominants.find(v) == dominants.end() &&
            dominanceInfo.dominates(root, v.getDefiningOp())) {
          dominants[v] = root;
        }
      }
    }
    assert(dominants.size() == instances.size());
  }
  return success();
}

void ShapeAnalysisDeprecated::dumpSymbol2InstancesDominant(
    Symbol2InstancesDominantType symbol2InstancesDominant) {
  llvm::dbgs() << "symbol2InstancesDominant: "
               << symbol2InstancesDominant.size() << "\n";
  for (auto item : symbol2InstancesDominant) {
    llvm::dbgs() << " -- item: \n";
    for (auto val_pair : item.second) {
      llvm::dbgs() << " ---- " << val_pair.first << " map to "
                   << val_pair.second << "\n";
    }
  }
}

// After we do all shape-related simplifications in the tensor level, we build
// explicit connection between all symbolic dims using 'disc_shape.tie_shape'
// right before we go to the buffer world.
//
// The basic idea is shown as below:
// Example IR before processing:
//   func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
//     %0 = mhlo.add(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) ->
//     tensor<?xf32> return %0 : tensor<?xf32>
//   }
// After adding disc.tie_shape ops:
//   func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
//     %c0 = constant 0 : index
//     %arg0_d0 = tensor.dim %arg0, %c0
//     %new_arg0 = disc_shape.tie_shape(%arg0, %arg0_d0)
//     %new_arg1 = disc_shape.tie_shape(%arg1, %arg0_d0)
//     %0 = mhlo.add(%new_arg0, %new_arg1) : (tensor<?xf32>, tensor<?xf32>) ->
//     tensor<?xf32>
//
//     %new_0 = disc_shape.tie_shape(%0, %arg0_d0)
//     return %new_0 : tensor<?xf32>
//   }
// The disc.tie_shape op is translated to a memref.reinterpret_cast op when
// converting to the buffer world. Example IR after bufferizing:
//   func @main(%arg0: memref<?xf32>, %arg1: memref<?xf32>) -> memref<?xf32> {
//     %c0 = constant 0 : index
//     %arg0_d0 = memref.dim %arg0, %c0
//     %new_arg0 = memref.reinterpret_cast %arg0 to offset: [0], sizes:
//     [%arg0_d0], strides: [1] : memref<?xf32> to memref<?xf32>
//
//     %new_arg1 = memref.reinterpret_cast %arg1 to offset: [0], sizes:
//     [%arg0_d0], strides: [1] : memref<?xf32> to memref<?xf32>
//
//     %0 = memref.alloc(%arg0_d0) : memref<?xf32>
//     "lmhlo.add"(%new_arg0, %new_arg1, %0) : (memref<?xf32>, ...
//
//     %new_0 = memref.reinterpret_cast %0 to
//     offset: [0], sizes: [%arg0_d0], strides: [1] : memref<?xf32> to
//     memref<?xf32>
//
//     return %new_0 : tensor<?xf32>
//   }
// After doing AllocOp + reinterpret_cast op canonicalization:
//   func @main(%arg0: memref<?xf32>, %arg1: memref<?xf32>) -> memref<?xf32> {
//     %c0 = constant 0 : index
//     %arg0_d0 = memref.dim %arg0, %c0
//     %new_arg0 = memref.reinterpret_cast %arg0 to offset: [0], sizes:
//     [%arg0_d0], strides: [1] : memref<?xf32> to memref<?xf32>
//
//     %new_arg1 =
//     memref.reinterpret_cast %arg1 to offset: [0], sizes: [%arg0_d0], strides:
//     [1] : memref<?xf32> to memref<?xf32>
//
//     %0 = memref.alloc(%arg0_d0) : memref<?xf32>
//     "lmhlo.add"(%new_arg0, %new_arg1, %0) : (memref<?xf32>, ...
//     return %0 : memref<?xf32>
//   }
// After the above processing, symbolic dims are resolved into the same SSA
// value if they are the same (e.g. the operands for allocOp or reinterpret_cast
// itself if it's external buffer) and following passes can simply reply on
// normal CSE & canonicalization pass to simplify index computation.
LogicalResult ShapeAnalysisDeprecated::buildTieShapeOps() {
  // create a dim op for each dimension of a shaped Value and group all such dim
  // ops by SymbolDim. Examples:
  //   %1 = mhlo.abs(%0) : (tensor<?xf32>) -> tensor<?xf32>
  // After processing:
  //   %0_d0 = tensor.dim %0, c0
  //   %1 = mhlo.abs(%0) : (tensor<?xf32>) -> tensor<?xf32>
  //   %1_d0 = tensor.dim %1, c0
  // symbolDim2Instances:
  //   {symbol0 : {%0_d0, %1_d0}} // %0_d0 and %1_d0 have the same symbolDim
  // shapedValue2Dims:
  //   {%0 : {%0_d0}, %1 : {%1_d0}}
  DenseMap<SymbolDim*, SmallVector<Value>> symbolDim2Instances;
  DenseMap<Value, SmallVector<Value>> shapedValue2Dims;
  if (failed(buildSymbolDimInstances(symbolDim2Instances, shapedValue2Dims)))
    return op_->emitError("failed to buildSymbolDimInstances");

  // map SymbolDim to its instance dominance map.
  // instance dominance map: instance -> dominant instance
  Symbol2InstancesDominantType symbol2InstancesDominant;
  if (failed(buildSymbolDimInstancesDominantMap(symbolDim2Instances,
                                                symbol2InstancesDominant)))
    return op_->emitError("failed to buildSymbolDimInstancesDominantMap");

  LLVM_DEBUG(dumpSymbol2InstancesDominant(symbol2InstancesDominant));

  // create a tie_shape op for each shaped value.
  SymbolShapeVisitor visitor = [&](OpBuilder& b, Location& loc, Value value) {
    // Skip static shaped values
    if (value.getType().cast<RankedTensorType>().hasStaticShape()) {
      return success();
    }

    SmallVector<Value> dominantDims;
    SymbolShape* symbolShape = getShape(value);
    auto& dims = shapedValue2Dims[value];
    DenseSet<Operation*> dimOps;
    for (int64_t i = 0, rank = symbolShape->rank(); i < rank; ++i) {
      SymbolDim* symbolDim = symbolShape->getSymbolDim(i);
      assert(symbolDim);
      auto& dominantInfo = symbol2InstancesDominant[symbolDim];
      dominantDims.push_back(dominantInfo[dims[i]]);
      // if 'value' is not an BlockArgument, we can guarantee
      // 'dominantDims' dominate 'value'
      if (dims[i] == dominantDims.back() || value.isa<BlockArgument>()) {
        b.setInsertionPointAfter(dims[i].getDefiningOp());
      }
      dimOps.insert(dims[i].getDefiningOp());
    }
    Value newValue = b.create<disc_shape::TieShapeOp>(loc, value.getType(),
                                                      value, dominantDims);
    auto users = llvm::to_vector<4>(value.getUsers());
    for (Operation* user : users) {
      // skip those dim ops used to fetch the dim size values of original shaped
      // value.
      if (dimOps.find(user) != dimOps.end()) continue;
      if (user == newValue.getDefiningOp()) continue;
      user->replaceUsesOfWith(value, newValue);
    }
    return success();
  };

  return visitSymbolShapes(visitor);
}

bool ShapeAnalysisDeprecated::isSameNumElements(Value lhs, Value rhs) {
  if (valueWithSameElements_.isEquivalent(ValueWrapper(lhs),
                                          ValueWrapper(rhs))) {
    return true;
  }

  SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> equal;
  if (!extractContinuousDimEqualInfo(lhs, rhs, equal)) {
    return false;
  }
  int64_t lhs_mapped = 0;
  int64_t rhs_mapped = 0;
  for (auto eq : equal) {
    lhs_mapped += eq.first.size();
    lhs_mapped += eq.second.size();
  }
  int64_t lhs_rank = lhs.getType().cast<MemRefType>().getRank();
  int64_t rhs_rank = rhs.getType().cast<MemRefType>().getRank();

  return (lhs_mapped == lhs_rank) && (rhs_mapped == rhs_rank);
}

bool ShapeAnalysisDeprecated::extractContinuousDimEqualInfo(
    Value lhs, Value rhs,
    SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>& equal) {
  equal.clear();
  int64_t lhs_rank = lhs.getType().cast<MemRefType>().getRank();
  DenseSet<int64_t> lhs_ranks_mapped;
  int64_t rhs_rank = rhs.getType().cast<MemRefType>().getRank();
  DenseSet<int64_t> rhs_ranks_mapped;
  if (lhs_rank == 0 || rhs_rank == 0) {
    return true;
  }
  // Check one-on-one mapping.
  for (int64_t out_idx = 0; out_idx < rhs_rank; out_idx++) {
    if (rhs_ranks_mapped.contains(out_idx)) {
      continue;
    }
    // The first/last dim of out can only be mapped to the first/last dim of
    // input.
    int64_t begin;
    int64_t end;
    if (out_idx == 0) {
      begin = 0;
      end = 1;
    } else if (out_idx == rhs_rank - 1) {
      begin = lhs_rank - 1;
      end = lhs_rank;
    } else {
      begin = 1;
      end = lhs_rank - 1;
    }
    if (begin < 0) {
      continue;
    }
    for (int64_t in_idx = begin; in_idx < end; in_idx++) {
      if (lhs_ranks_mapped.contains(in_idx)) {
        continue;
      }
      if (isDimEqual(rhs, out_idx, lhs, in_idx)) {
        equal.emplace_back(SmallVector<int64_t>({in_idx}),
                           SmallVector<int64_t>({out_idx}));
        lhs_ranks_mapped.insert(in_idx);
        rhs_ranks_mapped.insert(out_idx);
        break;
      }
    }
  }
  // Check many-on-one mapping.
  for (int64_t out_idx = 0; out_idx < rhs_rank; out_idx++) {
    if (rhs_ranks_mapped.contains(rhs_rank)) {
      continue;
    }
    const SmallVector<std::vector<DimValue>>& all_decomposes =
        getDimDecompose(rhs, out_idx);
    if (all_decomposes.empty()) {
      continue;
    }
    // Try every item in `decompose` to check whether it can be mapped to input.
    for (const auto& decompose : all_decomposes) {
      // There may be some dims with same value. Thus there may be kinds of dim
      // mappings to `decompose`.
      SmallVector<std::set<int64_t>> possible_mappings;
      for (const auto& val : decompose) {
        // In every iteration, `prev` records the previous mappings, and
        // `possible_mappings` will be cleared and to insert new mappings that
        // are generated by appending to `prev`.
        SmallVector<std::set<int64_t>> prev;
        std::swap(prev, possible_mappings);
        for (int64_t i = 0; i < lhs_rank; i++) {
          if (lhs_ranks_mapped.contains(i)) {
            continue;
          }
          if (isDimValueEqual(val, getDimValue(lhs, i))) {
            // Append the mapped index to each item in `prev`. During which
            // `prev` should not be updated.
            if (prev.empty()) {
              possible_mappings.emplace_back(std::set<int64_t>({i}));
            } else {
              for (int64_t j = 0; j < prev.size(); j++) {
                auto item = prev[j];
                if (item.insert(i).second) {
                  possible_mappings.emplace_back(std::move(item));
                }
              }
            }
          }
        }
        if (possible_mappings.empty()) {
          break;
        }
      }
      if (!possible_mappings.empty()) {  // All values are mapped.
        // Find a mapping group with continuous dims.
        SmallVector<int64_t> in_mapping;
        for (auto& mapping : possible_mappings) {
          // Note std::set is ordered.
          auto min = *mapping.begin();
          auto max = *mapping.rbegin();
          if (max - min == decompose.size() - 1) {
            in_mapping.insert(in_mapping.end(), mapping.begin(), mapping.end());
            break;
          }
        }
        if (!in_mapping.empty()) {
          lhs_ranks_mapped.insert(in_mapping.begin(), in_mapping.end());
          rhs_ranks_mapped.insert(out_idx);
          equal.emplace_back(std::move(in_mapping),
                             SmallVector<int64_t>({out_idx}));
          break;
        }
      }
    }
  }
  // Check one-on-many mapping.
  for (int64_t in_idx = 0; in_idx < lhs_rank; in_idx++) {
    if (lhs_ranks_mapped.contains(lhs_rank)) {
      continue;
    }
    const SmallVector<std::vector<DimValue>>& all_decomposes =
        getDimDecompose(lhs, in_idx);
    if (all_decomposes.empty()) {
      continue;
    }
    // Try every item in `all_decomposes` to check whether it can be mapped to
    // output.
    for (const auto& decompose : all_decomposes) {
      // There may be some dims with same value. Thus there may be kinds of dim
      // mappings to `de`.
      SmallVector<std::set<int64_t>> possible_mappings;
      for (const auto& val : decompose) {
        SmallVector<std::set<int64_t>> prev;
        std::swap(prev, possible_mappings);
        for (int64_t i = 0; i < rhs_rank; i++) {
          if (rhs_ranks_mapped.contains(i)) {
            continue;
          }
          if (isDimValueEqual(val, getDimValue(rhs, i))) {
            // Append the mapped index to each item in `prev`. During which
            // `prev` should not be updated.
            if (prev.empty()) {
              possible_mappings.emplace_back(std::set<int64_t>({i}));
            } else {
              for (int64_t j = 0; j < prev.size(); j++) {
                auto item = prev[j];
                if (item.insert(i).second) {
                  possible_mappings.emplace_back(std::move(item));
                }
              }
            }
          }
        }
        if (possible_mappings.empty()) {
          break;
        }
      }
      if (!possible_mappings.empty()) {  // All values are mapped.
        // Find a mapping group with continuous dims.
        SmallVector<int64_t> out_mapping;
        for (auto& mapping : possible_mappings) {
          auto min = *mapping.begin();
          auto max = *mapping.rbegin();
          if (max - min == decompose.size() - 1) {
            out_mapping.insert(out_mapping.end(), mapping.begin(),
                               mapping.end());
            break;
          }
        }
        if (!out_mapping.empty()) {
          lhs_ranks_mapped.insert(in_idx);
          rhs_ranks_mapped.insert(out_mapping.begin(), out_mapping.end());
          equal.emplace_back(SmallVector<int64_t>({in_idx}),
                             std::move(out_mapping));
          break;
        }
      }
    }
  }
  return true;
}

void ShapeAnalysisDeprecated::buildEqualShapesInFusion(
    const Operation* fusion, const DenseSet<Value>& values_in_fusion) {
  auto& equal_in_fusion = valueWithEqualShapeInFusion_[fusion];
  DenseMap<SymbolShape*, SmallVector<Value>> value_classes;
  for (auto value : values_in_fusion) {
    auto leader = getShape(value);
    if (leader == nullptr) {
      continue;
    }
    auto& value_class = value_classes[leader];
    if (!value_class.empty()) {
      equal_in_fusion.unionSets(ValueWrapper(value_class[0]),
                                ValueWrapper(value));
    }
    // TODO: we actually do not need to use a vector.
    value_class.push_back(value);
  }
}

Value ShapeAnalysisDeprecated::GetLeaderValueWithSameShapeGlobal(
    Value val) const {
  if (valueWithEqualShape_.findLeader(ValueWrapper(val)) ==
      valueWithEqualShape_.member_end()) {
    return nullptr;
  }
  return valueWithEqualShape_.getLeaderValue(ValueWrapper(val)).getValue();
}

Value ShapeAnalysisDeprecated::GetLeaderValueWithSameShapeInFusion(
    const Operation* fusion, Value val) const {
  if (fusion == nullptr) {
    return nullptr;
  }
  auto iter = valueWithEqualShapeInFusion_.find(fusion);
  if (iter != valueWithEqualShapeInFusion_.end()) {
    const auto& equal_in_fusion = iter->second;
    if (equal_in_fusion.findLeader(ValueWrapper(val)) !=
        equal_in_fusion.member_end()) {
      return equal_in_fusion.getLeaderValue(ValueWrapper(val)).getValue();
    }
  }
  return nullptr;
}

// update with view map

bool ShapeAnalysisDeprecated::getConstIntValue(Operation* op, int64_t& result) {
  if (auto constOp = dyn_cast_or_null<arith::ConstantIndexOp>(op)) {
    result = constOp.value();
    return true;
  } else if (auto constOp = dyn_cast_or_null<arith::ConstantIntOp>(op)) {
    result = constOp.value();
    return true;
  }
  return false;
}

// shape equality propagation based on the shape constrains of elementwise ops.
void OpListShapeAnalysis::PropagateEquality(
    const SmallVectorImpl<Operation*>& op_list) {
  bool converged = true;
  do {
    converged = true;
    auto update = [&](Value lhs, Value rhs,
                      EquivalenceClasses<ValueWrapper>& impl) {
      if (!impl.isEquivalent(ValueWrapper(lhs), ValueWrapper(rhs))) {
        converged = false;
        impl.unionSets(ValueWrapper(lhs), ValueWrapper(rhs));
      }
    };
    for (Operation* op : op_list) {
      int num_operand = op->getNumOperands();
      // Propagates same num_elements equality, and shape equality
      if (op->hasTrait<mlir::OpTrait::SameTypeOperands>() ||
          op->hasTrait<mlir::OpTrait::SameOperandsShape>() ||
          op->hasTrait<mlir::OpTrait::Elementwise>()) {
        Value lhs = op->getOperand(0);
        for (Value rhs : op->getOperands().drop_front()) {
          update(lhs, rhs, same_num_elements_impl_);
          update(lhs, rhs, same_shape_impl_);
        }
      }
      // Propagates same num_elements equality, not shape equality
      if (isa<lmhlo::DynamicReshapeOp, lmhlo::ReshapeOp, lmhlo::TransposeOp>(
              op)) {
        Value input = op->getOperand(0);
        // The last operand is the output memref by design
        Value output = op->getOperand(num_operand - 1);
        update(input, output, same_num_elements_impl_);
      }
      if (isa<lmhlo::ClampOp>(op)) {
        Value min = op->getOperand(0);
        Value operand = op->getOperand(1);
        Value max = op->getOperand(2);
        Value output = op->getOperand(3);
        auto min_ty = min.getType().dyn_cast<MemRefType>();
        auto operand_ty = operand.getType().dyn_cast<MemRefType>();
        auto max_ty = max.getType().dyn_cast<MemRefType>();
        if (!min_ty || !max_ty || !operand_ty) {
          continue;
        }
        update(operand, output, same_num_elements_impl_);
        update(operand, output, same_shape_impl_);
        if (min_ty.getRank() == operand_ty.getRank()) {
          update(operand, min, same_num_elements_impl_);
          update(operand, min, same_shape_impl_);
        }
        if (max_ty.getRank() == operand_ty.getRank()) {
          update(operand, max, same_num_elements_impl_);
          update(operand, max, same_shape_impl_);
        }
      }
    }
  } while (!converged);
}

bool ShapeAnalysis::isSameNumElements(Value lhs, Value rhs) {
  if (lhs == rhs) return true;

  auto lhsTy = lhs.getType().dyn_cast<ShapedType>();
  auto rhsTy = rhs.getType().dyn_cast<ShapedType>();

  if (!lhsTy || !rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank()) return false;

  return isProductEqual(lhs, 0, lhsTy.getRank(), rhs, 0, rhsTy.getRank());
}

bool ShapeAnalysis::isProductEqual(Value lhs, int lhsFrom, int lhsTo, Value rhs,
                                   int rhsFrom, int rhsTo) {
  SmallVector<int> lhsDimIdxs, rhsDimIdxs;
  lhsDimIdxs.reserve(lhsTo - lhsFrom);
  rhsDimIdxs.reserve(rhsTo - rhsFrom);
  for (int i = lhsFrom; i < lhsTo; ++i) lhsDimIdxs.push_back(i);
  for (int i = rhsFrom; i < rhsTo; ++i) rhsDimIdxs.push_back(i);

  return isProductEqual(lhs, lhsDimIdxs, rhs, rhsDimIdxs);
}

ShapeConstraintIRAnalysis::ShapeConstraintIRAnalysis(Operation* op)
    : op_(op), mgr_(op->getParentOfType<ModuleOp>()) {
  mgr_.load();
  ModuleOp m = op_->getParentOfType<ModuleOp>();
  op_->walk([&](Operation* op) {
    if (!isa<memref::AllocOp, memref::ReinterpretCastOp>(op)) return;
    Value result = op->getResult(0);
    auto resultTy = result.getType().dyn_cast<MemRefType>();
    // Only support ranked memref types.
    if (!resultTy) return;
    // Ealry return if the memref is static or does not have symbolic dim attr
    auto attrs = op->getAttrOfType<ArrayAttr>(
        disc_shape::SymbolicDimOp::getSymbolicDimAttrName());
    if (!attrs) return;
    auto& symbols = memrefValue2SymDims_[result];
    for (const auto& attr : attrs) {
      auto symOp = mgr_.symbolTable().lookup<disc_shape::SymbolicDimOp>(
          attr.cast<FlatSymbolRefAttr>().getValue());
      assert(symOp && "symbolic op is not found");
      symbols.push_back(symOp);
    }
  });
}

ShapeConstraintIRAnalysis::~ShapeConstraintIRAnalysis() { mgr_.save(); }

bool ShapeConstraintIRAnalysis::isShapeEqual(Value lhs, Value rhs) {
  if (lhs == rhs) return true;

  auto lhsTy = lhs.getType().dyn_cast<ShapedType>();
  auto rhsTy = rhs.getType().dyn_cast<ShapedType>();

  if (!lhsTy || !rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank()) return false;

  if (lhsTy.hasStaticShape() && rhsTy.hasStaticShape()) {
    return lhsTy.getShape() == rhsTy.getShape();
  }

  auto lhsIt = memrefValue2SymDims_.find(lhs);
  auto rhsIt = memrefValue2SymDims_.find(rhs);

  if (lhsIt == memrefValue2SymDims_.end() ||
      rhsIt == memrefValue2SymDims_.end() ||
      lhsIt->second.size() != rhsIt->second.size())
    return false;

  SmallVector<disc_shape::SymbolicDimOp> lhsSyms;
  SmallVector<disc_shape::SymbolicDimOp> rhsSyms;
  for (auto sym : lhsIt->second) {
    lhsSyms.push_back(mgr_.getRootSymbolicDim(sym));
  }
  for (auto sym : rhsIt->second) {
    rhsSyms.push_back(mgr_.getRootSymbolicDim(sym));
  }
  return lhsSyms == rhsSyms;
}

bool ShapeConstraintIRAnalysis::isProductEqual(Value lhs,
                                               ArrayRef<int> lhsDimIdxs,
                                               Value rhs,
                                               ArrayRef<int> rhsDimIdxs) {
  SymbolicDimProduct lhsProd;
  SymbolicDimProduct rhsProd;

  auto buildSymbolicDimProduct = [&](SymbolicDimProduct& prod, Value value,
                                     ArrayRef<int> dimIdxs) {
    auto ty = value.getType().dyn_cast<ShapedType>();
    auto it = memrefValue2SymDims_.find(value);
    if (!ty || !ty.hasRank()) return false;

    for (int idx : dimIdxs) {
      if (ty.getShape()[idx] == ShapedType::kDynamic) {
        if (it == memrefValue2SymDims_.end() || it->second.size() <= idx)
          return false;
        prod.symbols.push_back(it->second[idx]);
      } else {
        prod.factor *= ty.getShape()[idx];
      }
    }
    return true;
  };

  if (!buildSymbolicDimProduct(lhsProd, lhs, lhsDimIdxs) ||
      !buildSymbolicDimProduct(rhsProd, rhs, rhsDimIdxs)) {
    return false;
  }

  return mgr_.isSymbolicDimProductEqual(lhsProd, rhsProd);
}

}  // namespace disc_ral
}  // namespace mlir
