/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <openssl/md5.h>

#include "lhlo/IR/lhlo_ops.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"   // TF:llvm-project
#include "mlir/IR/Location.h"     // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/disc/IR/disc_ral_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/placement_utils.h"
#include "mlir/ral/ral_metadata.h"
#include "tensorflow/core/platform/env.h"

namespace mlir {
namespace disc_ral {

namespace {

using lmhlo::ConstantOp;
using StrT = SmallString<128>;
using tao::ral::MetadataFileEmitter;

constexpr static int kMd5DigestLength = 16;

// TODO(disc): refactor this part of code to phase out using HEX format.
void ExtractConstValue(const DenseElementsAttr& valueAttr, MemRefType memref,
                       StrT& data) {
  ArrayRef<char> rawData = valueAttr.getRawData();
  data.append(llvm::toHex(StringRef(rawData.data(), rawData.size())));
}

// unique_name is in the format: {hash_of_literal}_2x3x4xf32
LogicalResult GenerateUniqueNameForConst(MemRefType memref, const StrT& data,
                                         StrT& name) {
  unsigned char md5[kMd5DigestLength];
  MD5(reinterpret_cast<const unsigned char*>(data.data()), data.size(), md5);
  name.append(llvm::toHex(ArrayRef<uint8_t>(md5)));

  Type elemType = memref.getElementType();
  if (auto int_type = elemType.dyn_cast<IntegerType>()) {
    name.append(("_i" + llvm::Twine(int_type.getWidth())).str());
  } else if (auto fp_type = elemType.dyn_cast<FloatType>()) {
    name.append(("_f" + llvm::Twine(fp_type.getWidth())).str());
  } else {
    return failure();
  }

  name.append("_");

  int rank = memref.getRank();
  auto shape = memref.getShape();
  for (int i = 0; i < rank; ++i) {
    if (i == rank - 1) {
      name.append(llvm::Twine(shape[i]).str());
    } else {
      name.append((llvm::Twine(shape[i]) + "x").str());
    }
  }
  return success();
}

class DiscConstToRALPass : public DiscConstToRALPassBase<DiscConstToRALPass> {
 public:
  explicit DiscConstToRALPass(const std::string& metadata_file_path)
      : DiscConstToRALPassBase<DiscConstToRALPass>::DiscConstToRALPassBase() {
    this->metadata_file_path_ = metadata_file_path;
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect, memref::MemRefDialect,
                    disc_ral::RalDialect>();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    MetadataFileEmitter emitter(metadata_file_path_);
    if (!emitter.emitHeader()) {
      m.emitError("failed to emit header of metadata file: " +
                  metadata_file_path_);
      signalPassFailure();
      return;
    }

    SmallVector<ConstantOp, 4> worklist;
    m.walk([&](ConstantOp op) {
      if (op->getParentOfType<lmhlo::FusionOp>()) {
        return;
      }
      if (auto func = op->getParentOfType<func::FuncOp>()) {
        if (func->getAttrOfType<StringAttr>(kFuncCompIntensFusionAttr)) {
          return;
        }
      }
      worklist.push_back(op);
    });
    for (ConstantOp op : worklist) {
      if (failed(convertConstantOp(op, emitter))) {
        m.emitError("convert lmhlo.const to RAL failed");
        signalPassFailure();
        return;
      }
    }

    if (!emitter.emitTailer()) {
      m.emitError("failed to emit tailer of metadata file: " +
                  metadata_file_path_);
      signalPassFailure();
      return;
    }
  }

 private:
  LogicalResult convertConstantOp(ConstantOp const_op,
                                  MetadataFileEmitter& emitter);

  int num_processing_const_ops_ = 0;
  // Map each unique name of const to a unique index. Such indices are used to
  // reduce the overhead of const lookup at runtime.
  std::unordered_map<std::string, int> host_name_idx_map_;
  std::unordered_map<std::string, int> device_name_idx_map_;
};

// llvm.mlir.global internal constant @unique_name("unique_name\00")
// %1 = call @ral_constant_cpu/gpu(%ctx, %stream, %unique_name) : () ->
// memref<...>
LogicalResult DiscConstToRALPass::convertConstantOp(
    ConstantOp const_op, MetadataFileEmitter& emitter) {
  OpBuilder builder(const_op);
  Location loc = const_op.getLoc();
  DenseElementsAttr valueAttr = const_op.getValue().cast<DenseElementsAttr>();
  Type elemType = getElementTypeOrSelf(valueAttr);

  // Convert i1 -> i8
  if (elemType.getIntOrFloatBitWidth() == 1) {
    using func_type = mlir::APInt(const llvm::APInt&);
    valueAttr = valueAttr.mapValues(
        builder.getIntegerType(8),
        llvm::function_ref<func_type>([](const llvm::APInt& intVal) {
          return llvm::APInt(8, intVal.getZExtValue());
        }));
  }

  Value result = const_op.getOperation()->getOperand(0);
  MemRefType memref = result.getType().cast<MemRefType>();
  bool on_host = !placement_utils::isGpuMemRef(result);

  StrT data;
  StrT name;
  ExtractConstValue(valueAttr, memref, data);
  if (failed(GenerateUniqueNameForConst(memref, data, name))) {
    const_op.emitError("fail to general a unique name for const ops");
    return failure();
  }
  name.push_back('\0');
  std::string name_str = std::string(name).substr(0, name.size() - 1);
  std::string data_str = std::string(data);

  // save data
  auto name_idx_map = on_host ? &host_name_idx_map_ : &device_name_idx_map_;
  auto it = name_idx_map->find(name_str);
  if (it == name_idx_map->end()) {
    int next_const_idx;
    if (on_host) {
      next_const_idx = emitter.getNumHostConstantEmitted();
      if (!emitter.emitHostConstant(name_str, data_str)) return failure();
    } else {
      next_const_idx = emitter.getNumDeviceConstantEmitted();
      if (!emitter.emitDeviceConstant(name_str, data_str)) return failure();
    }
    it = name_idx_map->emplace(std::move(name_str), next_const_idx).first;
  }

  std::string symbol_name =
      ("__global_const_" + llvm::Twine(num_processing_const_ops_++)).str();
  Value const_name_global = LLVM::createGlobalString(
      loc, builder, symbol_name, name, LLVM::Linkage::Internal);

  ModuleOp m = getOperation();
  MLIRContext* ctx = m.getContext();
  Type pointer_type = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  Value zero = builder.create<LLVM::ConstantOp>(loc, IntegerType::get(ctx, 32),
                                                builder.getI32IntegerAttr(0));
  Value stream_idx = builder.create<LLVM::IntToPtrOp>(loc, pointer_type, zero);
  Value ral_context = const_op->getParentOfType<func::FuncOp>().getArgument(0);
  Value const_idx_value = builder.create<arith::ConstantIntOp>(
      loc, it->second, builder.getI32Type());

  SmallVector<Value, 12> newOperands{stream_idx, const_name_global,
                                     const_idx_value};
  auto dispatch_op = builder.create<disc_ral::DispatchOp>(
      loc, memref, ral_context, newOperands, "ral_const", false,
      on_host ? "cpu" : "gpu");

  result.replaceAllUsesWith(dispatch_op.getResult(0));
  const_op.erase();
  return success();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscConstToRALPass(
    const std::string& metadata_file_path) {
  return std::make_unique<DiscConstToRALPass>(metadata_file_path);
}

}  // namespace disc_ral
}  // namespace mlir
