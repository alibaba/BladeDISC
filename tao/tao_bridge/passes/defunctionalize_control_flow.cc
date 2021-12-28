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

#include "tao_bridge/passes/defunctionalize_control_flow.h"
#include "tao_bridge/tf/lower_if_op.h"
#include "tao_bridge/tf/lower_while_op.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace tao {

Status DefunctionalizeFactory::defunctionalize(
    Node* n, Graph* g, const FunctionLibraryDefinition& flib) {
  int num_old = g->num_op_nodes();
  auto defunctionalize = defunctionalize_factory_.find(n->type_string());
  if (defunctionalize != defunctionalize_factory_.end()) {
    TF_RETURN_IF_ERROR(defunctionalize->second(n, g, flib));
  }
  accum_defunc_ += (g->num_op_nodes() - num_old);
  return Status::OK();
}

void DefunctionalizeFactory::Initialize() {
  defunctionalize_factory_["While"] = RewriteWhileNode;
  defunctionalize_factory_["If"] = RewriteIfNode;
}

int DefunctionalizeFactory::accum_defunc_ = 0;

}  // namespace tao
}  // namespace tensorflow
