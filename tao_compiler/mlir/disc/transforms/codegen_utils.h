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

#ifndef DISC_TRANSFORMS_CODEGEN_UTILS_H_
#define DISC_TRANSFORMS_CODEGEN_UTILS_H_

#include <map>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace disc_ral {

#if TENSORFLOW_USE_ROCM
constexpr int kWarpSize = 64;
#else
constexpr int kWarpSize = 32;
#endif

constexpr const char* kGpuBinaryAttrName = "gpu.binary";
// In multi_cc_support node:
//   We provide support for {V100: sm_70, T4: sm_75, A100: sm_80}
//   if the driver is able to support them; Plus that, we also
//   provide a backup support of compute_60 in case the driver
//   version is not high enough at AOT compile time.
// clang-format off
const std::map<std::string, std::tuple</*major*/int,
                                       /*minor*/int,
                                       /*IsPTXNotSASS*/bool>> c_MULTI_SM_CONFIG {
  {"compute_60", {6, 0, true}},
  {"sm_70", {7, 0, false}},
  {"sm_75", { 7, 5, false }}
#if CUDA_VERSION >= 11000
  , {"sm_80", { 8, 0, false }}
#if CUDA_VERSION >= 11100
  , {"sm_86", { 8, 6, false }}
#endif
#endif
};
// clang-format on

// Choose which kind of schedule to use when doing codegen for row reduction.
// We currently have two schedules:
//  - schedule 1: using 1 block to process 1 row, two round warp shuffle
//  - schedule 2: using 1 warp to process 1 row, one round warp shuffle
constexpr const char* kRowReductionScheduleHint =
    "disc_row_reduction_schedule_hint";

constexpr const char* kColReductionScheduleHint =
    "disc_col_reduction_schedule_hint";

constexpr const char* kVectorizeOrTileHint = "disc_vectorize_or_tile_hint";

using DiscRowReductionScheduleType = enum : int {
  DISC_BLOCK_WISE_ROW_REDUCE = 1,
  DISC_WARP_WISE_ROW_REDUCE = 2
};

using DiscColReductionScheduleType = enum : int {
  DISC_TILE_W8_H32 = 1,
  DISC_TILE_W8_H16 = 2,
  DISC_TILE_W8_H8 = 3,
  DISC_TILE_LOOP_W64_H8 = 4,
  DISC_TILE_LOOP_W16_H32 = 5,
  DISC_TILE_LOOP_W8_H8 = 6,
};

// number of therads per block when doing codegen on GPU.
constexpr const char* kThreadPerBlockHint = "disc_thread_per_block_hint";

// empirical column size used to choose different row reduce schedule.
constexpr const int kRowReductionScheduleTurningSize = 512;

// default num of threads per block used when doing codegen
#if TENSORFLOW_USE_ROCM
constexpr const int kThreadsRowReduction = 512;
#else
constexpr const int kThreadsRowReduction = 256;
#endif

constexpr const int kVectorizeOrTileSize = 2;

// A tag used to distinguish cpu kernel func from others.
constexpr const char* kCpuKernelFunc = "disc_cpu_kernel_func";

// A tag used to distinguish small cpu kernel (no need for multi-threading).
constexpr const char* kSmallCpuKernel = "disc.cpu.small_kernel";

// A dimension size is small if it's smaller than this value on CPU.
constexpr const int kReductionTileSizeOnCPU = 128;

int getReductionTileSizeOnCPU();

int getRowReductionScheduleHint(Operation* op);

int getVectorizeOrTileHint(Operation* op);

int getThreadPerBlock(Operation* op);

int getColReductionScheduleHint(Operation* op);

SmallVector<Value> getShapeValues(OpBuilder* b, Value memref);

Value calcLinearIndex(OpBuilder* b, Location loc, const ValueRange multi_index,
                      const llvm::ArrayRef<Value> shape);

SmallVector<Value> calcMultiDimIndex(OpBuilder* b, Location loc,
                                     Value linear_index,
                                     const llvm::ArrayRef<Value> shape);

Value getDimSizeValue(OpBuilder* b, Value memref, int dim);

// Convert to index type by emitting std.index_cast if needed
Value mayConvertToIndexType(Value val, OpBuilder* b, Location loc);

// Convert to integer type by emitting std.index_cast if needed
Value mayConvertToIntegerType(Value val, OpBuilder* b, Location loc);

Value emitNumElementsComputation(OpBuilder& b, Location loc, Operation* op);
Value emitNumElementsComputation(OpBuilder& b, Location loc, Value memref);

SmallVector<Value> calcMultiDimIndex(OpBuilder* b, Location loc,
                                     Value linear_index, Value memref);

Value CastMemRefTo(OpBuilder& b, Location loc, Value from, Type toType,
                   ValueRange toShape);

scf::ParallelOp createParallelAndSetInsPt(OpBuilder& b, Location loc,
                                          SmallVectorImpl<Value>& vars,
                                          ArrayRef<Value> lbs,
                                          ArrayRef<Value> ubs,
                                          ArrayRef<Value> steps,
                                          ArrayRef<Value> init_values);

std::pair<scf::ParallelOp, scf::ParallelOp> tileParallelLoop(
    scf::ParallelOp op, ArrayRef<int64_t> tileSizes, bool withInboundCheck);

LogicalResult loopUnrollByFactorAndTryInterleave(
    scf::ForOp forOp, uint64_t unrollFactor,
    function_ref<void(unsigned, Operation*, OpBuilder)> annotateFn = nullptr);

void createAlignMemrefWithTile(OpBuilder& b, Value memref, int64_t tile_size);

}  // namespace disc_ral
}  // namespace mlir

#endif  // DISC_TRANSFORMS_CODEGEN_UTILS_H_
