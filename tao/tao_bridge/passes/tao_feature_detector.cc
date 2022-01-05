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

#include "tao_bridge/passes/tao_feature_detector.h"

#include <sstream>

#include "tao_bridge/common.h"
#include "tao_bridge/errors.h"
#include "tao_bridge/kernels/tao_compilation_info_collector.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace tao {

std::atomic<int> FeatureDetector::counter_{0};
std::once_flag FeatureDetector::init_global_flag_;
bool FeatureDetector::is_distributed_{false};
bool FeatureDetector::is_distributed_force_on_{false};
int64 FeatureDetector::dist_worker_id_{0};
int64 FeatureDetector::dist_worker_num_{0};
bool FeatureDetector::is_tao_force_on_{false};

FeatureDetector::FeatureDetector(const std::string &tag, const Graph *graph) {
  std::call_once(init_global_flag_, &initGlobalProperties);
  {
    std::stringstream ss;
    ss << tag << "_" << counter_++;
    tag_ = ss.str();
  }

  // detect gradient op
  // TODO(fpf): find a more robust way
  // One possible way is to get graident op list from
  // graph->flib_def().func_grad_. But func_grad_ is private in class
  // FunctionLibraryDefinition. We can only retrieve this info by calling
  // graph->lib_def()->ToProto(). This may be heavy for large graphs.
  for (auto n : graph->nodes()) {
    auto &op_type = n->type_string();
    // ends with Grad
    auto len = op_type.size();
    if (len > 4 && op_type.substr(len - 4) == "Grad") {
      has_gradient_op_ = true;
      break;
    }
  }

  // detect whale op
  for (auto n : graph->nodes()) {
    auto &name = n->name();
    if (name.find("WHALE_MICRO_BATCH") != std::string::npos ||
        name.find("WHALE_PARALLEL_SCOPE") != std::string::npos) {
      has_whale_op_ = true;
      break;
    }
  }

  // record information for dump
  auto &collector = TaoCompInfoCollector::Get();
  std::vector<std::string> keys{"features", "graphs", tag_, ""};
  keys[3] = "has_gradient_op";
  collector.SetCustomValue(keys, has_gradient_op_);
  keys[3] = "has_whale_op";
  collector.SetCustomValue(keys, has_whale_op_);
}

void FeatureDetector::initGlobalProperties() {
  CHECK_OK(ReadInt64FromEnvVar("TAO_DIST_INST_IDX", dist_worker_id_,
                               &dist_worker_id_));
  CHECK_OK(ReadInt64FromEnvVar("TAO_DIST_INST_TOTAL", dist_worker_num_,
                               &dist_worker_num_));

  std::string ev;
  CHECK_OK(ReadStringFromEnvVar("TAO_DIST_JOB_ON", "NOT_SET", &ev));
  is_distributed_ = ev != "NOT_SET";
  is_distributed_force_on_ = ev == "on";

  CHECK_OK(ReadStringFromEnvVar("TAO_ON_MODE", "NOT_SET", &ev));
  is_tao_force_on_ = ev == "true";

  // record information for dump
  auto &collector = TaoCompInfoCollector::Get();
  std::vector<std::string> keys{"features", "global", ""};
  keys[2] = "is_distributed";
  collector.SetCustomValue(keys, is_distributed_);
  keys[2] = "is_distributed_force_on";
  collector.SetCustomValue(keys, is_distributed_force_on_);
  keys[2] = "is_tao_force_on";
  collector.SetCustomValue(keys, is_tao_force_on_);
  keys[2] = "dist_worker_id";
  collector.SetCustomValue(keys, dist_worker_id_);
  keys[2] = "dist_worker_num";
  collector.SetCustomValue(keys, dist_worker_num_);
}

bool FeatureDetector::IsDistributed() {
  std::call_once(init_global_flag_, &initGlobalProperties);
  return is_distributed_;
}

int64 FeatureDetector::GetWorkerId() {
  std::call_once(init_global_flag_, &initGlobalProperties);
  return dist_worker_id_;
}

int64 FeatureDetector::GetTotalWorkerNum() {
  std::call_once(init_global_flag_, &initGlobalProperties);
  return dist_worker_num_;
}

bool FeatureDetector::IsDistributedForceOn() {
  std::call_once(init_global_flag_, &initGlobalProperties);
  return is_distributed_force_on_;
}
bool FeatureDetector::IsTaoForceOn() {
  std::call_once(init_global_flag_, &initGlobalProperties);
  return is_tao_force_on_;
}

} // namespace tao
} // namespace tensorflow