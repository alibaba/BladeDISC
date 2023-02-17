/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TAO_TAO_BRIDGE_PASSES_TAO_OPTIMIZATION_PASS_H_
#define TAO_TAO_BRIDGE_PASSES_TAO_OPTIMIZATION_PASS_H_

#include <type_traits>

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tao_bridge/common.h"
#include "tao_bridge/passes/tao_feature_detector.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {
namespace tao {

struct TaoPassOptions {
  // Whether to override flag `tf_xla_ops_to_cluster` in
  // TaoMarkForCompilationPass.
  absl::optional<std::string> override_tf_xla_ops_to_cluster;
  // min/max cluster size for TaoMarkForCompilationPass.
  // this has higher priority than the value in TF_XLA_FLAGS
  absl::optional<int> min_cluster_size;
  absl::optional<int> max_cluster_size;

  // Whether to build a inner-layer TaoLaunch op.
  bool inner_tao_launch;

  bool cluster_recount = true;

  std::unique_ptr<FeatureDetector> feats;
};

class TaoOptimizationPass : public GraphOptimizationPass {
 public:
  TaoOptimizationPass() : opts_(absl::make_unique<TaoPassOptions>()) {}

  void set_options(std::unique_ptr<TaoPassOptions> opts) {
    opts_ = std::move(opts);
  }

  Status Run(const GraphOptimizationPassOptions& options) override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TaoOptimizationPass);
  std::unique_ptr<TaoPassOptions> opts_;
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_PASSES_TAO_OPTIMIZATION_PASS_H_
