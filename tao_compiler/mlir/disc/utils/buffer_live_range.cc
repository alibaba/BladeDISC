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

#include "mlir/disc/utils/buffer_live_range.h"
void BufferLiveRange::AllocateBuffer(Value value, Operation* op,
                                     int64_t position) {
  LivingBuffers livingBuffers;
  livingBuffers.startOp = op;
  livingBuffers.start_position = position;
  livingBuffers.buffer = value;
  living_buffer_map_[value] = livingBuffers;
  buffer_list_.push_back(value);
}
void BufferLiveRange::Analysis() {
  buffer_list_.clear();
  living_buffer_map_.clear();
  int64_t position;
  main_.walk([&](Operation* op) {
    // Traverse the function's blocks and operations.
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      auto buffer = allocOp.getResult();
      if (isHostBuffer(allocOp.getResult())) {
        return;
      }
      AllocateBuffer(buffer, op, position);
    } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
      living_buffer_map_[deallocOp.getOperand()].endOp = deallocOp;
    } else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      for (auto operand : returnOp.getOperands()) {
        if (living_buffer_map_.count(operand)) {
          living_buffer_map_[operand].endOp = returnOp;
        }
      }
    } else if (isa<lmhlo_disc::H2DOp>(op) || isa<lmhlo_disc::D2HOp>(op)) {
      return;
    } else {
      for (Value operand : op->getOperands()) {
        if (living_buffer_map_.count(operand)) {
          living_buffer_map_[operand].consumers.push_back(std::make_pair(
              op, position - living_buffer_map_[operand].start_position));
        }
      }
    }
    position++;
  });
}