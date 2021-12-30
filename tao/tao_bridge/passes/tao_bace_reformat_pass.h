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

#ifndef TAO_TAO_BRIDGE_PASSES_TAO_BACE_REFORMAT_FP32_PASS_H_
#define TAO_TAO_BRIDGE_PASSES_TAO_BACE_REFORMAT_FP32_PASS_H_

#include "tao_bridge/common.h"
#include "tao_bridge/passes/tao_optimization_pass.h"

namespace tensorflow {
namespace tao {

class TaoBaCEReformatPass : public GraphOptimizationPass {
 public:
  // Constructor for normal pass run.
  TaoBaCEReformatPass();

  // Test-only constructor.
  TaoBaCEReformatPass(bool enabled, int64 max_dim_bar, int64 min_dim_bar,
                      int64 size_bar)
      : enabled_(enabled),
        max_dim_bar_(max_dim_bar),
        min_dim_bar_(min_dim_bar),
        size_bar_(size_bar) {}

  Status Run(const GraphOptimizationPassOptions& options) override;

 private:
  bool enabled_ = false;    // if enable this pass.
  int64 max_dim_bar_ = -1;  // Upper threashold for size of larger dim.
  int64 min_dim_bar_ = -1;  // Upper threashold for size of smaller dim.
  int64 size_bar_ = -1;     // Upper threashold for size const.
};

}  // namespace tao

}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_PASSES_BACE_REFORMAT_FP32_PASS_H_
