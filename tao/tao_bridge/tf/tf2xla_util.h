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

// Adopted from tensorflow/compiler/tf2xla/tf2xla_util.h

#ifndef TAO_TAO_BRIDGE_TF_TF2XLA_UTIL_H_
#define TAO_TAO_BRIDGE_TF_TF2XLA_UTIL_H_

#include <unordered_map>

#include "tao_bridge/tf/statusor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tao {

// Indicates how a FunctionDef is associated with a graph node (e.g. the node is
// a function call, or the node has function attrs).
class AssociatedFunctionInfo {
 public:
  enum AssociatedFunctionType {
    kFunctionAttr = 0,
    kFunctionCallNode = 1,
    kSymbolicGradient = 2,
  };

  // The function is an attr of the node.
  static AssociatedFunctionInfo FunctionAttr(const string& func_name,
                                             const AttrValueMap& attrs,
                                             const string& attr_name) {
    return AssociatedFunctionInfo(kFunctionAttr, func_name, attrs, attr_name);
  }

  // The node is a function call.
  static AssociatedFunctionInfo FunctionCall(const string& func_name,
                                             const AttrValueMap& attrs) {
    // attr_name will not be used in this case.
    return AssociatedFunctionInfo(kFunctionCallNode, func_name, attrs,
                                  /*attr_name=*/"");
  }

  // The node is a SymbolicGradient op.
  static AssociatedFunctionInfo SymbolicGradient(const string& func_name,
                                                 const AttrValueMap& attrs) {
    // attr_name will not be used in this case.
    return AssociatedFunctionInfo(kSymbolicGradient, func_name, attrs,
                                  /*attr_name=*/"");
  }

  AssociatedFunctionType type() const { return type_; }

  const string& func_name() const { return func_name_; }

  const string& attr_name() const { return attr_name_; }

  const AttrValueMap& attrs() const { return attrs_; }

 private:
  AssociatedFunctionInfo(AssociatedFunctionType type, const string& func_name,
                         const AttrValueMap& attrs, const string& attr_name)
      : type_(type),
        func_name_(func_name),
        attrs_(attrs),
        attr_name_(attr_name) {}

  // Available for all instances.
  AssociatedFunctionType type_;
  string func_name_;
  AttrValueMap attrs_;

  // Only available if the function is defined in an attr.
  string attr_name_;
};

// Returns if the NodeDef has associated function.
bool HasAssociatedFunction(const NodeDef& node_def,
                           FunctionLibraryRuntime* flr);

// Gets functions associated with the node. Current cases:
// 1. For function call node, its function name;
// 2. For SymbolicGradient op, returned func_name will be "SymbolicGradient",
//    and returned attrs will be this node's attributes;
// 3. For nodes like XlaWhile/XlaIf, all their function attributes.
std::vector<AssociatedFunctionInfo> GetAssociatedFunctions(
    const Node& node, FunctionLibraryRuntime* flr);

// Changes associated functions for the node. Current cases:
// 1. For function call node, creates a new node with the new function name and
//    remove the old node;
// 2. For SymbolicGradient op, add or replace GradientDef in
//    FunctionLibraryDefinition;
// 3. For nodes like XlaWhile/XlaIf, modify their function attributes.
Status RewriteAssociatedFunction(
    Graph* graph, Node* node, FunctionLibraryDefinition* fld,
    const AssociatedFunctionInfo& associated_function,
    const string& rewritten_function_name);

// Attribute to mark nodes to be executed on host.
extern const char kXlaOutsideCompilationAttrName[];

}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_TF_TF2XLA_UTIL_H_
