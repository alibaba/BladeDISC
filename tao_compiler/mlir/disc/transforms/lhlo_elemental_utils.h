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

#include "lhlo/IR/lhlo_ops.h"
#include "lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "llvm/Support/Debug.h"
#include "mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/disc/transforms/codegen_utils.h"

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
  struct SpecificLoader {
    SpecificLoader() {}

    SpecificLoader(Value shm, int64_t element_per_block)
        : shm_buffer_(shm), element_per_block_(element_per_block) {}

    Value operator()(OpBuilder& b, Value memref, ValueRange indices) {
      return loadShmem(b, memref, indices);
    }

    Value loadShmem(OpBuilder& b, Value value, ValueRange indices) {
      auto loc = shm_buffer_.getLoc();
      assert(shm_buffer_ != nullptr);
      // It requires the elements store in shm_buffer are in index order.
      auto shape = getShapeValues(&b, value);
      assert(shape.size() == indices.size());
      Value linear_index = calcLinearIndex(&b, loc, indices, shape);
      Value element_per_block_val =
          b.create<arith::ConstantIndexOp>(loc, element_per_block_);
      Value shmem_index =
          b.create<arith::RemUIOp>(loc, linear_index, element_per_block_val);
      Value res =
          b.create<memref::LoadOp>(loc, shm_buffer_, ValueRange({shmem_index}));
      return res;
    }

    Value shm_buffer_;
    int64_t element_per_block_;
  };

  // SpecificStore is for stitch codegen, which helps to hook the original store
  // with specific shm memref.
  struct SpecificStore {
    SpecificStore() {}

    SpecificStore(Value shm, int64_t element_per_block)
        : shm_buffer_(shm), element_per_block_(element_per_block) {}

    void operator()(OpBuilder& b, Value value, Value memref,
                    ValueRange indices) {
      storeShmem(b, value, memref, indices);
    }

    void storeShmem(OpBuilder& b, Value value, Value memref,
                    ValueRange indices) {
      auto loc = shm_buffer_.getLoc();
      assert(shm_buffer_ != nullptr);
      // It requires the elements store in shm_buffer are in index order.
      auto shape = getShapeValues(&b, memref);
      assert(shape.size() == indices.size());
      Value linear_index = calcLinearIndex(&b, loc, indices, shape);
      Value element_per_block_val =
          b.create<arith::ConstantIndexOp>(loc, element_per_block_);
      Value shmem_index =
          b.create<arith::RemUIOp>(loc, linear_index, element_per_block_val);
      b.create<memref::StoreOp>(loc, value, shm_buffer_,
                                ValueRange({shmem_index}));
    }

    Value shm_buffer_;
    int64_t element_per_block_;
  };

  SpecificLoader* getSpecificLoader(Operation* op, Value operand);
  SpecificStore* getSpecificStore(Operation* op, Value operand);

  bool isWrittenBack(Operation* op) { return is_written_back_.contains(op); }

  void setWrittenBack(Operation* op) { is_written_back_.insert(op); }

  void setSpecificLoader(std::pair<Operation*, Value> loader_for,
                         SpecificLoader loader) {
    specific_loaders_[loader_for] = loader;
  }

  void setSpecificStore(std::pair<Operation*, Value> store_for,
                        SpecificStore store) {
    specific_stores_[store_for] = store;
  }

 private:
  DenseMap<std::pair<Operation*, Value>, SpecificLoader> specific_loaders_;
  DenseMap<std::pair<Operation*, Value>, SpecificStore> specific_stores_;
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

SmallVector<Value> convertValuesIfIntegerType(Location loc, OpBuilder* b,
                                              ValueRange vs);

struct LhloOpToStdScalarOp {
  template <typename OpTy>
  static Value map(OpTy op, Type resultType, ValueRange operands, OpBuilder* b);
};

template <typename OpTy>
Value LhloOpToStdScalarOp::map(OpTy op, Type resultType, ValueRange operands,
                               OpBuilder* b) {
  Location loc = op->getLoc();
  auto newOperands = convertValuesIfIntegerType(loc, b, operands);
  Type newResultType = convertIfIntegerType(resultType);
  Value result;
  if (std::is_same<OpTy, lmhlo::ConvertOp>::value) {
    result = mhlo::impl::mapConvertOpToStdScalarOp(
        loc, resultType, newResultType,
        llvm::to_vector<>(op->getOperandTypes()), newOperands, b);
  } else {
    result = lmhlo::LhloOpToStdScalarOp::map<OpTy>(
        cast<OpTy>(op), newResultType, newOperands, b);
  }
  if (newResultType != resultType) {
    result = b->create<UnrealizedConversionCastOp>(loc, resultType, result)
                 ->getResult(0);
  }
  return result;
}

}  // namespace disc_ral
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_LHLO_ELEMENTAL_UTILS_H_
