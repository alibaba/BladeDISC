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

#ifndef TAO_TAO_BRIDGE_PASSES_TAO_BUILD_TAO_OP_PASS_H_
#define TAO_TAO_BRIDGE_PASSES_TAO_BUILD_TAO_OP_PASS_H_

#include "tao_bridge/common.h"
#include "tao_bridge/passes/tao_optimization_pass.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tao {

class TaoBuildTaoOpPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;

  void set_opts(const std::unique_ptr<TaoPassOptions>& opt) {
    if (opt) {
      inner_tao_launch_ = opt->inner_tao_launch;
    }
  }

 private:
  bool inner_tao_launch_{false};  // inner attribute of TaoLaunch op.
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_BUILD_TAO_OP_PASS_H_
