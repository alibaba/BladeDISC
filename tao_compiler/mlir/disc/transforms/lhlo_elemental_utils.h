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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_LHLO_ELEMENTAL_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_LHLO_ELEMENTAL_UTILS_H_

#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"

namespace mlir {

class Value;
class Location;
class Operation;
class ValueRange;
class Region;
class FuncOp;
namespace arith {
enum class AtomicRMWKind : uint64_t;
}

namespace scf {
class ForOp;
class ParallelOp;
}  // namespace scf

namespace memref {
class LoadOp;
class ReinterpretCastOp;
}  // namespace memref

namespace disc_ral {

class LowerConfig {
 public:
  // SpecificLoader is for stitch codegen, which helps to hook the original load
  // with specific shm memref.
  using SpecificLoaderFunc = std::function</*output*/ Value(
      OpBuilder&, /*operand-memref*/ Value, /*indices*/ ValueRange,
      /*shmem-buffer*/ Value, /*row-per-block*/ int64_t)>;
  struct SpecificLoader {
    SpecificLoader() {}
    SpecificLoader(SpecificLoaderFunc f, Value shm, int64_t tile_size)
        : func_(f), target_shm_(shm), row_per_block_(tile_size) {}
    SpecificLoaderFunc func_;
    Value target_shm_;
    int64_t row_per_block_;
    Value operator()(OpBuilder& b, Value memref, ValueRange indices) {
      return func_(b, memref, indices, target_shm_, row_per_block_);
    }
  };

  SpecificLoader* getSpecificLoader(Operation* op, Value operand);

  bool isWrittenBack(Operation* op) { return is_written_back_.contains(op); }

  void setWrittenBack(Operation* op) { is_written_back_.insert(op); }

  void setSpecificLoader(std::pair<Operation*, Value> loader_for,
                         SpecificLoader loader) {
    specific_loaders_[loader_for] = loader;
  }

 private:
  DenseMap<std::pair<Operation*, Value>, SpecificLoader> specific_loaders_;
  DenseSet<Operation*> is_written_back_;
};

Value createMaySpecificLoad(OpBuilder& b, Location loc, Operation* op,
                            Value memref, ValueRange indices,
                            LowerConfig* lower_config = nullptr);

Value mayCreateStore(OpBuilder* b, Location loc, Operation* op, Value value,
                     ValueRange out_indices, LowerConfig* lower_config);

using AccumulatorFactory = std::function<Value(Value, Value)>;

AccumulatorFactory getFactory(OpBuilder& b, Location loc, Region& body);

Value createLoadOrUseCachedValue(Location loc, OpBuilder* b, Operation* op,
                                 Value memref, ValueRange indices,
                                 OpBuilder::InsertPoint insert_point,
                                 LowerConfig* lower_config = nullptr);

template <typename LHLO_OpTy>
Value elementalLower(OpBuilder* b, Location loc, LHLO_OpTy op,
                     ValueRange output_index, bool check_cache = false,
                     LowerConfig* lower_config = nullptr);

scf::ForOp createLoopAndSetInsPt(OpBuilder& b, Location loc, Value& var,
                                 Value lb, Value ub, Value step,
                                 ArrayRef<Value> init_values = {});

memref::ReinterpretCastOp createMemRef1DReinterpretCast(OpBuilder& b,
                                                        Location loc,
                                                        Value memref);

void createOffsetStore(OpBuilder& b, Location loc, Value res, Value memref,
                       Value offset);

memref::LoadOp createOffsetLoad(OpBuilder& b, Location loc, Value memref,
                                Value offset);

arith::AtomicRMWKind getAtomicRMWKind(Region& body);

bool isUnsignedIntegerValue(Value val);

Type convertIfIntegerType(Type type);

bool needUpgradingUnsignedInteger(Operation* op);

struct LhloOpToStdScalarOp {
  template <typename OpTy>
  static Value map(OpTy op, Type resultType, ValueRange operands, OpBuilder* b);
};

template <typename OpTy>
Value LhloOpToStdScalarOp::map(OpTy op, Type resultType, ValueRange operands,
                               OpBuilder* b) {
  if (!needUpgradingUnsignedInteger(op))
    return lmhlo::LhloOpToStdScalarOp::map<OpTy>(cast<OpTy>(op), resultType,
                                                 operands, b);
  Location loc = op->getLoc();
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
  Type newResultType = convertIfIntegerType(resultType);
  Value result = lmhlo::LhloOpToStdScalarOp::map<OpTy>(
      cast<OpTy>(op), resultType, newOperands, b);
  if (newResultType != resultType) {
    result.setType(newResultType);
    result = b->create<UnrealizedConversionCastOp>(loc, resultType, result)
                 ->getResult(0);
  }
  return result;
}

}  // namespace disc_ral
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_LHLO_ELEMENTAL_UTILS_H_
