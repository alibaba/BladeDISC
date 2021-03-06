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

// Adopted from tensorflow/compiler/tf2xla/functionalize_control_flow.h

#ifndef TAO_TAO_BRIDGE_PASSES_FUNCTIONALIZE_CONTROL_FLOW_H_
#define TAO_TAO_BRIDGE_PASSES_FUNCTIONALIZE_CONTROL_FLOW_H_

#include "tao_bridge/tf/statusor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace tao {

// Transformation that converts tf.while_loop() loops into functional While
// operators and tf.cond() conditionals into function If operators, suitable for
// XLA compilation. If lookup_library is provided, use it to make the library
// for control flow self-contained.
Status FunctionalizeControlFlow(Graph* graph,
                                FunctionLibraryDefinition* library,
                                FunctionLibraryRuntime* flr = nullptr);
Status FunctionalizeControlFlow(const FunctionLibraryDefinition* lookup_library,
                                Graph* graph,
                                FunctionLibraryDefinition* library,
                                FunctionLibraryRuntime* flr = nullptr);

// This pass looks at the graph and all associated FunctionDefs, and turns
// traditional control flow structure (Switch/Merge/etc.) into functional
// control flow structure (If/While).
class FunctionalizeControlFlowPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_PASSES_FUNCTIONALIZE_CONTROL_FLOW_H_
