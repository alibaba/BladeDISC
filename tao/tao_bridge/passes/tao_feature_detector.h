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

#pragma once

#include <atomic>
#include <mutex>

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace tao {
class FeatureDetector {
public:
  // tag: describe where the graph comes from. Will use it as key in dumped json
  // info.
  FeatureDetector(const std::string &tag, const Graph *graph);
  ~FeatureDetector(){};

  std::string Tag() { return tag_; }

  // graph properties query
  bool HasGradientOp() { return has_gradient_op_; }
  bool HasWhaleOp() { return has_whale_op_; }

  // global properties. only works in ODPS environment now.
  static bool IsDistributed();
  static int64 GetWorkerId();
  static int64 GetTotalWorkerNum();
  static bool IsDistributedForceOn();
  static bool IsTaoForceOn();

private:
  std::string tag_;
  bool has_gradient_op_{false};
  bool has_whale_op_{false};

  static std::atomic<int> counter_;
  static std::once_flag init_global_flag_;
  static void initGlobalProperties();
  static bool is_distributed_;
  static bool is_distributed_force_on_;
  static int64 dist_worker_id_;
  static int64 dist_worker_num_;
  static bool is_tao_force_on_;

}; // class FeatureDetector
} // namespace tao
} // namespace tensorflow