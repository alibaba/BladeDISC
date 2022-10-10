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

/*
This file is derived from PyTorch, and the following modifications are made to
meet our requirements:

1. modifications mentioned in `onnx.h`
2. `EliminateDeadCode` in `BlockToONNX` will now delete dead Values in `env`
through a callback function `deleteDeadValues`.
*/

#include "pytorch_blade/compiler/jit/torch/onnx.h"

#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/symbolic.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/utils/pybind.h>
#include <sstream>
#include <unordered_map>

namespace torch {
namespace blade {
using namespace torch::jit;

// Transform PythonOps into Nodes that match ONNX semantics.
std::tuple<std::shared_ptr<Graph>, std::unordered_map<Value*, Value*>> ToONNX(
    std::shared_ptr<Graph>& graph,
    ::torch::onnx::OperatorExportTypes operator_export_type) {
  // todo: find ways to directly pass reference of env from python to this
  // function.
  auto new_graph = std::make_shared<Graph>(graph->current_scope());
  std::unordered_map<Value*, Value*> env;
  BlockToONNX(graph->block(), new_graph->block(), operator_export_type, env);
  std::tuple<std::shared_ptr<Graph>, std::unordered_map<Value*, Value*>> result{
      new_graph, env};
  return result;
}

void BlockToONNX(
    Block* old_block,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    std::unordered_map<Value*, Value*>& env) {
  torch::autograd::SymbolicContext ctx{};
  ctx.block = new_block;
  py::object onnx = py::module::import("torch.onnx");
  py::object onnx_symbolic = py::module::import("torch.onnx.symbolic_helper");
  py::object onnx_registry = py::module::import("torch.onnx.symbolic_registry");

  // Returns a node that n maps to in the new graph
  auto envFn = [&env](Value* n) -> Value* {
    auto it = env.find(n);
    TORCH_CHECK(it != env.end(), "Dangling node reference");
    TORCH_CHECK(it->second, "Unused node was subsequently used");
    return it->second;
  };

  // Initialize context and environment
  for (auto input : old_block->inputs()) {
    auto n = ctx.block->addInput()->copyMetadata(input);
    env[input] = n;
  }
  // Put the new outputs in our environment map, and copy the type from the
  // input graph if they were not set by the symbolic. This is called only
  // with results of symbolic call (not for nodes that are just cloned).
  auto setOutputs = [&](const std::string& op_name,
                        Node* node,
                        const value_list& outputs) {
    auto old_outputs = node->outputs();
    // Count all outputs, excluding Handles
    auto num_old_outputs = old_outputs.size();
    if (outputs.size() != num_old_outputs) {
      std::ostringstream ss;
      ss << "symbolic for " << op_name
         << " produced an incorrect number of outputs (expected ";
      ss << num_old_outputs << ", but got " << outputs.size() << ")";
      throw std::runtime_error(ss.str());
    }
    for (size_t i = 0; i < num_old_outputs; ++i) {
      auto old = old_outputs[i];
      if (outputs[i]) {
        // Allow symbolic() to skip specifying the type of the return node.
        // Unfortunately, they are on the hook for all internal nodes
        // (though in practice, the types are not computed.)
        auto old_tensor_type = old->type()->cast<TensorType>();
        if (old_tensor_type == nullptr ||
            old_tensor_type->scalarType().has_value()) {
          // Check if Tensor has scalartype when overwriting output type
          outputs[i]->setType(old->type());
        }

        // Copy over source location and scope information to all nodes
        // created by the symbolic
        outputs[i]->node()->setSourceRange(node->sourceRange());
        outputs[i]->node()->setScope(node->scope());
        env[old] = outputs[i];
      } else {
        // Null output means that the ONNX op doesn't have outputs corresponding
        // to certain PyTorch outputs
        env[old] = nullptr;
        if (!old->uses().empty()) {
          std::ostringstream ss;
          ss << "symbolic for " << op_name << " returned None for the output "
             << i;
          ss << " (indicating conversion for that particular output is not supported), ";
          ss << "but the network uses this output later";
          // TODO: Say what actually used it
          throw std::runtime_error(ss.str());
        }
      }
    }
  };

  // Clone the node and add it to the new graph
  auto cloneNode = [&](Node* node) {
    auto n_ = ctx.block->appendNode(
        ctx.block->owningGraph()->createClone(node, envFn));
    for (size_t i = 0; i < node->outputs().size(); i++) {
      // n_->outputs()[i]->setType(node->outputs()[i]->type());
      env[node->outputs()[i]] = n_->outputs()[i];
    }
  };

  // Cast output of symbolic() python implementation
  auto processSymbolicOutput = [&](const std::string& op_name,
                                   Node* n,
                                   const py::object& raw_output) {
    if (raw_output.ptr() == Py_None) {
      cloneNode(n);
      return;
    }
    // Cast the outputs back to C++ and put them in the new graph
    std::vector<Value*> outputs;
    try {
      if (py::isinstance<Value>(raw_output)) {
        outputs = value_list{py::cast<Value*>(raw_output)};
      } else {
        outputs = py::cast<std::vector<Value*>>(raw_output);
      }
    } catch (const std::exception& ex) {
      std::ostringstream ss;
      ss << "Error casting results of symbolic for " << op_name
         << ": expected to return list of op nodes, instead received type ''"
         << py::str(raw_output.get_type()) << "': " << py::str(raw_output);
      throw std::runtime_error(ss.str());
    }

    setOutputs(op_name, n, outputs);
  };

  auto callPySymbolicFunction = [&](Node* n) {
    // The idea is delegate as much of the actual argument massaging to
    // Python as possible

    py::tuple py_inputs(n->inputs().size());
    Py_ssize_t input_nr = 0;
    for (auto* input : n->inputs()) {
      py_inputs[input_nr++] = py::cast(envFn(input));
    }

    WithInsertPoint insert_point_guard(ctx.block);
    WithCurrentScope scope_guard(*ctx.block->owningGraph(), n->scope());
    py::object raw_output = onnx.attr("_run_symbolic_function")(
        ctx.block->owningGraph(),
#if PYTORCH_MAJOR_VERSION == 1 && PYTORCH_MINOR_VERSION >= 11
        ctx.block,
#endif
        n,
        py_inputs,
        env,
        operator_export_type);

    // TODO: Assert it's an ATen identifier???
    // (Sometimes it's not...)
    processSymbolicOutput(n->kind().toUnqualString(), n, raw_output);
  };

  auto callPySymbolicMethod = [&](ConcretePythonOp* op) {
    // Test if there is a symbolic function; bail if there is not
    auto pyobj = py::handle(op->pyobj.get());
    auto func = op->autogradFunction();
    if (func) {
      pyobj = func->get();
    }
    if (!py::hasattr(pyobj, "symbolic")) {
      cloneNode(op);
      return;
    }

    // Prepare args for Python. First one is the graph, and is followed
    // by regular args, with Variables replaced by corresponding nodes.
    Py_ssize_t input_nr = 0;
    py::tuple py_symbolic_args(1 + op->cconv.size());
    py_symbolic_args[input_nr++] = py::cast(ctx.block->owningGraph());
    auto inputs = op->inputs();
    auto node_it = inputs.begin();
    auto scalar_it = op->scalar_args.begin();
    for (auto arg_type : op->cconv) {
      py::object obj;
      if (arg_type == 'c') {
        TORCH_CHECK(
            scalar_it != op->scalar_args.end(),
            "expected too many scalar args");
        obj = py::reinterpret_borrow<py::object>(
            py::handle((scalar_it++)->get()));
      } else if (arg_type == 'd') {
        TORCH_CHECK(node_it != inputs.end(), "expected too many inputs");
        obj = py::cast(envFn(*node_it++));
      } else {
        throw std::runtime_error("unexpected calling convention");
      }
      py_symbolic_args[input_nr++] = obj;
    }

    WithInsertPoint insert_point_guard(ctx.block);
    WithCurrentScope scope_guard(*ctx.block->owningGraph(), op->scope());
    // Call the symbolic function
    // Use a little trampoline function so we can give good error messages
    // upon argument mismatch
    py::object opset_version = onnx_symbolic.attr("_export_onnx_opset_version");
    onnx_registry.attr("register_op")(
        op->name(), pyobj.attr("symbolic"), "", opset_version);
    py::object raw_output = onnx.attr("_run_symbolic_method")(
        op->name(), pyobj.attr("symbolic"), py_symbolic_args);

    processSymbolicOutput(op->name(), op, raw_output);
  };

  // Finally, visit all nodes in the graph
  for (auto node : old_block->nodes()) {
    if (node->kind().is_caffe2()) {
      // Pass on Caffe2 operator, since we already preprocess it
      cloneNode(node);
    } else if (node->kind() == prim::PythonOp) {
      callPySymbolicMethod(static_cast<ConcretePythonOp*>(node));
    } else {
      callPySymbolicFunction(node);
    }
  }

  for (auto output : old_block->outputs()) {
    ctx.block->registerOutput(env.at(output));
    env.at(output)->setType(output->type());
  }

  auto deleteDeadValues =
      [&](const std::unordered_set<const Value*>& live_values) {
        for (auto it = env.begin(); it != env.end();) {
          if (!live_values.count(it->second)) {
            env.erase(it++);
          } else {
            ++it;
          }
        }
      };
  EliminateDeadCode(
      ctx.block,
      deleteDeadValues,
      DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}

} // namespace blade
} // namespace torch
