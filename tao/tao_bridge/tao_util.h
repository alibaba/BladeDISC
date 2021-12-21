#ifndef TAO_TAO_BRIDGE_TAO_UTIL_H_
#define TAO_TAO_BRIDGE_TAO_UTIL_H_

#include "absl/strings/string_view.h"

#include <memory>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"

namespace tensorflow {

class Graph;

namespace tao {
namespace util {

bool HasOpType(const Graph& g, absl::string_view op_type);

std::unique_ptr<FunctionLibraryDefinition> ReachableDefinitions(
    const FunctionLibraryDefinition& flib, const FunctionDef& func);

}  // namespace util
}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_TAO_UTIL_H_
