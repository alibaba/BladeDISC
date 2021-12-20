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

#ifndef TENSORFLOW_COMPILER_JIT_PARTIALLY_DECLUSTER_PASS_H_
#define TENSORFLOW_COMPILER_JIT_PARTIALLY_DECLUSTER_PASS_H_

#include "tao_bridge/common.h"
#include "tao_bridge/passes/tao_optimization_pass.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {
namespace tao {

// Clones or moves nodes from within a cluster to outside the cluster if
// profitable.  There are two reasons why we do this:
//
//  - Reducing device-to-host copies.
//  - Reducing the number of XLA recompilations.
class TaoPartiallyDeclusterPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;

  void set_opts(const std::unique_ptr<TaoPassOptions>& opt) {
    if (opt) {
      is_inner_ = opt->inner_tao_launch;
    }
  }

  bool is_inner_{false};
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_PARTIALLY_DECLUSTER_PASS_H_
