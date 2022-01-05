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

#ifndef TAO_TAO_BRIDGE_PASSES_DEFUNCTIONALIZE_CONTROL_FLOW_H_
#define TAO_TAO_BRIDGE_PASSES_DEFUNCTIONALIZE_CONTROL_FLOW_H_

#include <functional>
#include <map>
#include <string>

#include "tao_bridge/tf/statusor.h"

namespace tensorflow {

class Graph;
class Node;
class FunctionLibraryDefinition;

namespace tao {

class DefunctionalizeFactory {
public:
  DefunctionalizeFactory() { Initialize(); };

  Status defunctionalize(Node *n, Graph *g,
                         const FunctionLibraryDefinition &flib);

  static int get_accum_defunc() { return accum_defunc_; }

private:
  void Initialize();

  std::map<string, std::function<Status(Node *n, Graph *g,
                                        const FunctionLibraryDefinition &flib)>>
      defunctionalize_factory_;

  // Accumulated number of new created nodes after defunctionalize
  static int accum_defunc_;
};

} // namespace tao
} // namespace tensorflow

#endif // TAO_TAO_BRIDGE_PASSES_DEFUNCTIONALIZE_CONTROL_FLOW_H_
