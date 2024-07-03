// Copyright 2024 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef MLIR_HLO_UTILS_BUFFER_LIVING_RANGE
#define MLIR_HLO_UTILS_BUFFER_LIVING_RANGE

#include <vector>

#include "lhlo/IR/lhlo_ops.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/OpDefinition.h"

class BufferLiveRange {
 public:
  explicit BufferLiveRange(ModuleOp main) : main_(main) {}
  void AllocateBuffer(Value value, Operation* op, int64_t position);
  void Analysis();
  std::vector<Value> getBufferList() { return buffer_list_; }
  LivingBuffers getLivingBuffers(Value value) {
    if (living_buffer_map_.count(value) == 0) {
      return LivingBuffers();
    }
    return living_buffer_map_[value];
  }

 private:
  ModuleOp main_;
  std::vector<Value> buffer_list_;
  std::unordered_map<Value, LivingBuffers, LiveRangeHash> living_buffer_map_;
};
