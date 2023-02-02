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

//===- KernelOutlining.cpp - Implementation of GPU kernel outlining -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the GPU dialect kernel outlining pass.
//
//===----------------------------------------------------------------------===//

#include <map>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/placement_utils.h"

namespace mlir {
namespace disc_ral {

using mlir::memref::AllocOp;
using mlir::memref::LoadOp;

namespace {

class ValueComparator {
 public:
  bool operator()(const Value& a, const Value& b) const {
    return a.getAsOpaquePointer() < b.getAsOpaquePointer();
  }
};
// please don't revise to unordered_map
typedef std::map<Value, std::map<std::vector<int64_t>, Value>, ValueComparator>
    LoadedValueCache;

std::vector<int64_t> getMultiDimIndex(int64_t linear_index,
                                      ArrayRef<int64_t> shape) {
  // sizes {a, b, c, d}  ->  acc_mul {b*c*d, c*d, d, 1}
  int64_t rank = shape.size();
  int64_t acc_mul = 1;
  for (int64_t i = 1; i < rank; ++i) {
    acc_mul *= shape[i];
  }
  int64_t linear_remain = linear_index;
  std::vector<int64_t> multidim_index(rank);
  for (int64_t i = 0; i < rank; ++i) {
    multidim_index[i] = linear_remain / acc_mul;
    linear_remain = linear_remain % acc_mul;
    if (i != (rank - 1)) {
      acc_mul = acc_mul / shape[i + 1];
    }
  }
  return multidim_index;
}

int64_t getLinearIndex(std::vector<int64_t> multidim_index,
                       ArrayRef<int64_t> shape) {
  assert(multidim_index.size() == shape.size());
  // sizes {a, b, c, d}  ->  acc_mul {b*c*d, c*d, d, 1}
  int64_t rank = shape.size();
  int64_t acc_mul = 1;
  for (int64_t i = 1; i < rank; ++i) {
    acc_mul *= shape[i];
  }
  int64_t linear_index = 0;
  for (int64_t i = 0; i < rank; ++i) {
    linear_index += multidim_index[i] * acc_mul;
    if (i != (rank - 1)) {
      acc_mul = acc_mul / shape[i + 1];
    }
  }
  return linear_index;
}

int64_t createLoadOpsArray(OpBuilder& b, Location loc, gpu::LaunchFuncOp,
                           Value memref, LoadedValueCache& loaded_value_cache) {
  auto memref_type = memref.getType().cast<MemRefType>();
  auto sizes = memref_type.getShape();
  auto rank = sizes.size();
  int64_t acc_mul = 1;
  for (int64_t i = 0; i < rank; ++i) {
    acc_mul = acc_mul * sizes[i];
  }

  for (int64_t linear_index = 0; linear_index < acc_mul; ++linear_index) {
    auto multidim_index = getMultiDimIndex(linear_index, sizes);
    SmallVector<Value, 4> multidim_index_value;
    for (int64_t dim : multidim_index) {
      auto dim_value = b.create<arith::ConstantIndexOp>(loc, dim);
      multidim_index_value.push_back(dim_value);
    }
    auto loaded_value = b.create<LoadOp>(loc, memref, multidim_index_value);
    loaded_value_cache[memref].emplace(multidim_index, loaded_value);
  }
  return acc_mul;
}

// Return the operand index if 'value' is a operand of 'op'
// If there are multiple operands of 'op' that are all 'value', return
// the first index
int64_t getFirstOperandIndex(Operation* op, Value value) {
  for (int64_t i = 0; i < op->getNumOperands(); ++i) {
    auto operand = op->getOperand(i);
    if (operand == value) {
      return i;
    }
  }
  assert(false && "Exception in getFirstOperandIndex, value is not an operand");
  return -1;
}

// Operation *Operation::clone(IRMapping &mapper) {
//   auto *newOp = cloneWithoutRegions(mapper);
//
//   // Clone the regions.
//   for (unsigned i = 0; i != numRegions; ++i)
//     getRegion(i).cloneInto(&newOp->getRegion(i), mapper);
//
//   return newOp;
// }

// This method is revised from Region::cloneinto()
// TODO: any easier ways?
void cloneRegionAndRemapLoad(Region* src, Region* dest, IRMapping& mapper,
                             int64_t memref_idx, Value memref_arg,
                             Block& new_entry_block, bool is_entry) {
  assert(dest && "expected valid region to clone into");
  assert(src != dest && "cannot clone region into itself");
  Region::iterator destPos = dest->end();

  // If the list is empty there is nothing to clone.
  if (src->empty()) return;

  for (Block& block : *src) {
    // entry block has already been mapped outside
    if (is_entry && block.isEntryBlock()) {
      mapper.map(&block, &new_entry_block);
      // Clone and remap the operations within this block.
      for (auto& op : block) {
        // if is a LoadOp from host MemRef
        if (isa<LoadOp>(&op)) {
          auto load_op = cast<LoadOp>(&op);
          if (isSameUnderlineBuffer(load_op.getOperand(0), memref_arg)) {
            auto indices_value = load_op.getIndices();
            std::vector<int64_t> multidim_index;
            for (auto index_value : indices_value) {
              auto index_value_op = index_value.getDefiningOp();
              auto const_op = dyn_cast<arith::ConstantOp>(index_value_op);
              assert(
                  const_op &&
                  "indices is expected to be const when load from host MemRef");
              multidim_index.emplace_back(
                  const_op.getValue().dyn_cast<mlir::IntegerAttr>().getInt());
            }

            auto memref_type =
                load_op.getOperand(0).getType().cast<MemRefType>();
            auto shape = memref_type.getShape();
            auto linear_index = getLinearIndex(multidim_index, shape);
            mapper.map(load_op.getResult(),
                       new_entry_block.getArgument(memref_idx + linear_index));

            // load_op will be erased
            continue;
          }
        }
        if (auto view = dyn_cast<ViewLikeOpInterface>(&op)) {
          if (isSameUnderlineBuffer(view->getResult(0), memref_arg)) {
            continue;
          }
        }
        // new_entry_block.push_back(op.clone(mapper));
        auto* newOp = op.cloneWithoutRegions(mapper);
        for (unsigned i = 0; i != op.getNumRegions(); ++i) {
          cloneRegionAndRemapLoad(&op.getRegion(i), &newOp->getRegion(i),
                                  mapper, memref_idx, memref_arg,
                                  new_entry_block, false);
        }
        new_entry_block.push_back(newOp);
      }

    } else {
      Block* newBlock = new Block();
      mapper.map(&block, newBlock);
      // Clone the block arguments. The user might be deleting arguments to the
      // block by specifying them in the mapper. If so, we don't add the
      // argument to the cloned block.
      for (auto arg : block.getArguments()) {
        if (!mapper.contains(arg))
          mapper.map(arg, newBlock->addArgument(arg.getType(), arg.getLoc()));
      }
      // Clone and remap the operations within this block.
      for (auto& op : block) {
        // if is a LoadOp from host MemRef
        if (isa<LoadOp>(&op)) {
          LoadOp load_op = cast<LoadOp>(&op);
          if (isSameUnderlineBuffer(load_op.getOperand(0), memref_arg)) {
            auto indices_value = load_op.getIndices();
            std::vector<int64_t> multidim_index;
            for (auto index_value : indices_value) {
              auto index_value_op = index_value.getDefiningOp();
              auto const_op = dyn_cast<arith::ConstantOp>(index_value_op);
              assert(
                  const_op &&
                  "indices is expected to be const when load from host MemRef");
              multidim_index.emplace_back(
                  const_op.getValue().dyn_cast<mlir::IntegerAttr>().getInt());
            }

            auto memref_type =
                load_op.getOperand(0).getType().cast<MemRefType>();
            auto shape = memref_type.getShape();
            auto linear_index = getLinearIndex(multidim_index, shape);
            mapper.map(load_op.getResult(),
                       new_entry_block.getArgument(memref_idx + linear_index));

            // load_op will be erased
            continue;
          }
        }
        if (auto view = dyn_cast<ViewLikeOpInterface>(&op)) {
          if (isSameUnderlineBuffer(view->getResult(0), memref_arg)) {
            continue;
          }
        }
        // newBlock->push_back(op.clone(mapper));
        auto* newOp = op.cloneWithoutRegions(mapper);
        for (unsigned i = 0; i != op.getNumRegions(); ++i) {
          cloneRegionAndRemapLoad(&op.getRegion(i), &newOp->getRegion(i),
                                  mapper, memref_idx, memref_arg,
                                  new_entry_block, false);
        }
        newBlock->push_back(newOp);
      }
      dest->getBlocks().insert(destPos, newBlock);
    }
  }

  // Now that each of the blocks have been cloned, go through and remap the
  // operands of each of the operations.
  auto remapOperands = [&](Operation* op) {
    // don't remap operand of the loadOp from host memory
    if (isa<LoadOp>(op)) {
      LoadOp load_op = cast<LoadOp>(op);
      if (isSameUnderlineBuffer(load_op.getOperand(0), memref_arg)) {
        return;
      }
    }
    for (auto& operand : op->getOpOperands())
      if (auto mappedOp = mapper.lookupOrNull(operand.get()))
        operand.set(mappedOp);
    for (auto& succOp : op->getBlockOperands())
      if (auto* mappedOp = mapper.lookupOrNull(succOp.get()))
        succOp.set(mappedOp);
  };

  for (Region::iterator it(mapper.lookup(&(src->front()))); it != destPos; ++it)
    it->walk(remapOperands);
}

gpu::LaunchFuncOp expandMemRef(gpu::LaunchFuncOp launch_func_op, Value memref,
                               LoadedValueCache& loaded_value_cache) {
  OpBuilder b(launch_func_op);
  Location loc = launch_func_op.getLoc();

  auto expanded_num =
      createLoadOpsArray(b, loc, launch_func_op, memref, loaded_value_cache);

  // create a new gpu.FuncOp inside gpu.module
  auto module = launch_func_op->getParentOfType<ModuleOp>();
  auto gpu_module = module.lookupSymbol<gpu::GPUModuleOp>(
      launch_func_op.getKernelModuleName());
  auto gpu_func_op =
      gpu_module.lookupSymbol<gpu::GPUFuncOp>(launch_func_op.getKernelName());

  SmallVector<Type, 4> gpu_func_op_operand_types;
  auto memref_idx = getFirstOperandIndex(launch_func_op.getOperation(), memref);
  memref_idx = memref_idx - gpu::LaunchOp::kNumConfigOperands;
  SmallVector<Value, 4> new_operands;
  for (int i = 0; i < launch_func_op.getNumKernelOperands(); ++i) {
    if (i != memref_idx) {
      new_operands.push_back(launch_func_op.getKernelOperand(i));
      gpu_func_op_operand_types.push_back(
          launch_func_op.getKernelOperand(i).getType());
    } else {
      for (auto index_value_pair : loaded_value_cache.at(memref)) {
        new_operands.push_back(index_value_pair.second);
        gpu_func_op_operand_types.push_back(index_value_pair.second.getType());
      }
    }
  }
  b.setInsertionPoint(gpu_func_op);
  FunctionType func_type = FunctionType::get(launch_func_op.getContext(),
                                             gpu_func_op_operand_types, {});
  auto new_gpu_func_op = b.create<gpu::GPUFuncOp>(
      loc, Twine(gpu_func_op.getName(), "_revised").str(), func_type);
  new_gpu_func_op->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                           b.getUnitAttr());

  // clone the Ops in the body of the gpu.FuncOp
  IRMapping map;
  Region& new_gpu_func_body = new_gpu_func_op.getBody();
  Block& new_gpu_func_entry_block = new_gpu_func_body.front();
  for (auto operand :
       llvm::enumerate(gpu_func_op.getBody().front().getArguments())) {
    if (operand.index() == memref_idx) {
      continue;
    } else if (operand.index() < memref_idx) {
      map.map(operand.value(),
              new_gpu_func_entry_block.getArgument(operand.index()));
    } else {
      map.map(operand.value(), new_gpu_func_entry_block.getArgument(
                                   operand.index() + expanded_num - 1));
    }
  }
  // memref of the entry block
  auto memref_arg = gpu_func_op.getBody().front().getArgument(memref_idx);
  Block& new_entry_block = new_gpu_func_op.getBody().front();
  cloneRegionAndRemapLoad(&gpu_func_op.getBody(), &new_gpu_func_op.getBody(),
                          map, memref_idx, memref_arg, new_entry_block, true);

  // update the FunctionType of the gpu.Launchfunc inside the module
  b.setInsertionPoint(launch_func_op);
  auto new_launch_func_op = b.create<gpu::LaunchFuncOp>(
      loc, new_gpu_func_op, launch_func_op.getGridSizeOperandValues(),
      launch_func_op.getBlockSizeOperandValues(),
      launch_func_op.getDynamicSharedMemorySize(), new_operands);

  launch_func_op.erase();
  gpu_func_op.erase();

  return new_launch_func_op;
}

void convertWorkgroupBuffer(gpu::GPUFuncOp gpu_func_op, AllocOp alloc) {
  auto memref_type = alloc.getResult().getType().cast<MemRefType>();
  auto buffer =
      gpu_func_op.addWorkgroupAttribution(memref_type, alloc.getLoc());
  alloc.replaceAllUsesWith(buffer);
  alloc.erase();
}

/* This pass revises the kernel outlining:
 *
 * 1, For a MemRef resides in host memory, which always means that the MemRef
 * is for shape representation, expand the MemRef into an array of Values. This
 * is due to that the kernel cannot directly gep/load from host addresses.
 *
 * 2, Since we are only to make a 'dynamic shape' compiler, not 'dynamic rank'
 * compiler, the shape of shape should always be static. So here it is assumed
 * that the host MemRef must always be static shaped. If later we found this is
 * not always true, revise the form of the kernel into variant number of
 * arguments. This is a little bit more compilicated but still doable.
 *
 * 3, For device MemRef, just leave the form here. There will be lots of the
 * args after lowering to llvm, though. Currently no side-effects are observed,
 * we may improve it in the pass in future if neccessary.
 */
class ReviseGpuKernelOutliningPass
    : public ReviseGpuKernelOutliningPassBase<ReviseGpuKernelOutliningPass> {
 public:
  void runOnOperation() override {
    auto module = getOperation();
    std::map<Operation*, std::vector<Value>> to_be_processed;
    module.walk([&](gpu::LaunchFuncOp launch_func_op) {
      auto gpu_module = module.lookupSymbol<gpu::GPUModuleOp>(
          launch_func_op.getKernelModuleName());
      assert(gpu_module && "gpu_module is empty");
      auto gpu_func_op = gpu_module.lookupSymbol<gpu::GPUFuncOp>(
          launch_func_op.getKernelName());
      assert(gpu_func_op && "gpu_func_op is empty");
      for (auto memref : llvm::enumerate(launch_func_op.getOperands())) {
        // the associate arg in gpu.FuncOp
        if (memref.index() < gpu::LaunchOp::kNumConfigOperands) {
          continue;
        }
        auto arg_memref = gpu_func_op.getBody().front().getArgument(
            memref.index() - gpu::LaunchOp::kNumConfigOperands);
        if (arg_memref.getType().isa<MemRefType>() &&
            (!placement_utils::isGpuMemRef(arg_memref))) {
          auto memref_type = memref.value().getType().cast<MemRefType>();
          assert(memref_type.hasStaticShape() &&
                 "Unexpected Host MemRef with dynamic shape");
          to_be_processed[launch_func_op.getOperation()].emplace_back(
              memref.value());
        }
      }
    });

    for (auto op_values_pair : to_be_processed) {
      auto launch_func_op = dyn_cast<gpu::LaunchFuncOp>(op_values_pair.first);
      for (auto memref : op_values_pair.second) {
        LoadedValueCache loaded_value_cache;
        launch_func_op =
            expandMemRef(launch_func_op, memref, loaded_value_cache);
      }
    }

    // convert for shared buffer
    module.walk([&](gpu::LaunchFuncOp launch_func_op) {
      auto gpu_module = module.lookupSymbol<gpu::GPUModuleOp>(
          launch_func_op.getKernelModuleName());
      assert(gpu_module && "gpu_module is empty");
      auto gpu_func_op = gpu_module.lookupSymbol<gpu::GPUFuncOp>(
          launch_func_op.getKernelName());
      assert(gpu_func_op && "gpu_func_op is empty");
      gpu_func_op.walk([&](AllocOp alloc) {
        auto memref_type = alloc.getResult().getType().cast<MemRefType>();
        assert(memref_type.getMemorySpace()
                       .dyn_cast<gpu::AddressSpaceAttr>()
                       .getValue() ==
                   gpu::GPUDialect::getWorkgroupAddressSpace() &&
               "unexpected alloc op in gpu_func_op");
        convertWorkgroupBuffer(gpu_func_op, alloc);
      });
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createReviseGpuKernelOutliningPass() {
  return std::make_unique<ReviseGpuKernelOutliningPass>();
}

}  // namespace disc_ral
}  // namespace mlir
