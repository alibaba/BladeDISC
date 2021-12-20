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

  Status defunctionalize(Node* n, Graph* g,
                         const FunctionLibraryDefinition& flib);

  static int get_accum_defunc() { return accum_defunc_; }

 private:
  void Initialize();

  std::map<string, std::function<Status(Node* n, Graph* g,
                                        const FunctionLibraryDefinition& flib)>>
      defunctionalize_factory_;

  // Accumulated number of new created nodes after defunctionalize
  static int accum_defunc_;
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_PASSES_DEFUNCTIONALIZE_CONTROL_FLOW_H_
