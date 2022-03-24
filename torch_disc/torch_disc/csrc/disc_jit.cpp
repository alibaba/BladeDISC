// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_disc/csrc/disc_jit.h"

#include <ATen/Functions.h>
#include <sys/stat.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include <torch/script.h>
#include <unistd.h>

#include "compiler/jit/fusion.h"
#include "compiler/mlir/converters/mhlo_conversion.h"
#include "lazy_tensor_core/csrc/ts_backend/backend_impl.h"
#include "torch_disc/csrc/backend_impl.h"
#include "torch_disc/csrc/disc_class.h"
#include "torch_disc/csrc/io.h"

namespace torch_disc {
namespace compiler {

using namespace ::torch::jit;
using TSData = torch_lazy_tensors::compiler::TSData;

// TODO(Yancey1989):
// This a very rough implementation to go thought the whole DiscBackend,
// this fake function only cluster one node that is supported on mhlo
// conversion module into a group. We should re-implement this.
std::vector<Node*> FakeCluster(const std::shared_ptr<Graph>& graph) {
  std::vector<Node*> nodes;
  for (auto node : graph->nodes()) {
    if (torch::blade::IsMlirMhloSupported(*node) &&
        node->kind() != prim::Constant) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

// Give:
//  with prim::FusionGroup(
//      %1: Scalar,
//      %2: Scalar):
//    %3 Tensor = aten::add(%1, %2)
//  return %3
//
// Execute: CastToTensorInputs(sub_graph)
//
// After:
//  with prim::FusionGroup(
//      %1.1: Tensor,
//      %2.1: Tensor):
//    %4 = aten::item(%1.1, 1)
//    %5 = aten::item(%2.1, 1)
//    %3 Tensor = aten::add(%4, %5)
//    return %3
void DiscSubgraphInputsCast(Graph* disc_graph, size_t i) {
  auto new_input = disc_graph->insertInput(
      i, c10::string(disc_graph->inputs()[i]->debugName() + ".1"));
  auto orig_input = disc_graph->inputs()[i + 1];
  auto item_node = disc_graph->create(aten::item, {new_input});
  // TODO(Yancey1989): supports more types
  item_node->output()->setType(c10::IntType::get());
  disc_graph->appendNode(item_node);
  orig_input->replaceAllUsesWith(item_node->output());
  item_node->moveBefore(item_node->output()->uses()[0].user);
  disc_graph->eraseInput(i + 1);
}

void CastGraphInputsToTensor(const std::shared_ptr<Graph>& graph,
                             Node* sub_graph, Node* disc_node) {
  auto disc_graph = disc_node->owningGraph();

  size_t inputs = sub_graph->inputs().size();
  for (size_t i = 0; i < inputs; ++i) {
    auto input = sub_graph->inputs()[i];
    if (input->type()->cast<c10::TensorType>() == nullptr) {
      auto cast_tensor = graph->createNumToTensor(input);
      cast_tensor->insertAfter(input->node());
      sub_graph->replaceInput(i, cast_tensor->output());

      // TODO(Yancey1989): cast output
      DiscSubgraphInputsCast(disc_graph, i);
    }
  }
  LOG(WARNING) << "After [CastToTensorInputs]: \n"
               << graph->toString() << std::endl;
}

void FusionDiscNodes(const std::shared_ptr<Graph>& graph) {
  auto nodes = FakeCluster(graph);
  for (auto node : nodes) {
    // create a sub-graph
    auto group = graph->createWithSubgraph(prim::FusionGroup);
    auto sub_graph = group->g(attr::Subgraph);
    group->insertAfter(node);
    auto node_merged = torch::blade::MergeNodeIntoGroup(group, node);

    auto outputs = node->outputs();
    auto new_outputs = node_merged->outputs();

    for (int i = 0; i < outputs.size(); ++i) {
      auto out = outputs[i];
      sub_graph->registerOutput(new_outputs[i]);
      auto group_out = group->addOutput();
      out->replaceAllUsesWith(group_out);
      group_out->setType(out->type());
    }
    node->destroy();
    CastGraphInputsToTensor(graph, group, node_merged);
  }
  torch::jit::EliminateDeadCode(graph);
}

std::string DiscCMD(const std::string& mlir_fname,
                    const std::string& out_fname) {
  std::stringstream ss;
  std::string logf = mlir_fname + ".log";
  ss << "./disc_compiler_main " << mlir_fname << " " << out_fname << " > "
     << logf << " 2>&1 ";
  return ss.str();
}

std::tuple<std::string, std::string, std::string> MhloConversaion(
    const std::shared_ptr<Graph>& graph) {
  std::string parsable_mlir;
  std::string pretty_mlir;
  std::string input_dev_str;
  std::string output_dev_str;
  std::tie(parsable_mlir, pretty_mlir, input_dev_str, output_dev_str) =
      torch::blade::ConvertTorchScriptToMhlo(graph);
  TORCH_CHECK(!parsable_mlir.empty(), "mhlo conversaion failed!");

  std::string dir = GetTempDirectory("/tmp");
  std::string in_fname = dir + "/disc.mlir";
  std::ofstream outfile(in_fname);
  outfile << parsable_mlir << std::endl;
  outfile.flush();
  return std::make_tuple(in_fname, input_dev_str, output_dev_str);
}

std::string CallDiscCompiler(const std::string& mlir_fname) {
  std::string out_fname = mlir_fname + ".out";
  std::string cmd = DiscCMD(mlir_fname, out_fname);
  TORCH_CHECK(std::system(cmd.c_str()) == 0,
              "disc compile failed with cmd: " + cmd);
  return out_fname;
}

std::vector<const torch::jit::Value*> ToConstValue(
    c10::ArrayRef<torch::jit::Value*> values) {
  std::vector<const torch::jit::Value*> const_vals;
  std::copy(values.begin(), values.end(), std::back_inserter(const_vals));
  return const_vals;
}

std::vector<c10::IValue> DiscCompilation(const std::shared_ptr<Graph>& graph) {
  std::vector<c10::IValue> disc_inputs;
  std::vector<torch::jit::Node*> disc_nodes;
  std::copy_if(graph->nodes().begin(), graph->nodes().end(),
               std::back_inserter(disc_nodes), [](torch::jit::Node* node) {
                 return node->kind() == prim::FusionGroup;
               });

  for (auto node : disc_nodes) {
    auto sub_graph = node->g(attr::Subgraph);

    auto option = DiscClass::MakeOption();
    std::string mlir_fname;
    std::tie(mlir_fname, option->input_dev_str, option->output_dev_str) =
        MhloConversaion(sub_graph);
    auto output_fname = CallDiscCompiler(mlir_fname);
    option->executable_prog_bytes = ReadFileBytes(output_fname);
    option->constant_bytes = ReadFileBytes(output_fname + ".pbtxt");
    option->input_type_spec_str =
        torch::blade::ShapeTypeSpec::GetShapeTypeSpec(
            ToConstValue(sub_graph->inputs()), false /**force_concrete**/)
            .Serialize();
    option->output_type_spec_str =
        torch::blade::ShapeTypeSpec::GetShapeTypeSpec(
            ToConstValue(sub_graph->outputs()), false /**force_concret**/)
            .Serialize();

    // add DiscClass object as graph input
    c10::IValue disc_val = torch::make_custom_class<DiscClass>(option);
    auto val = graph->addInput(c10::str("disc_class_p", disc_inputs.size()));
    val->setType(disc_val.type());
    disc_inputs.push_back(disc_val);

    // %3 : Tensor[] = ListConstruct(%1, %2)
    auto list_construct = graph->insertNode(graph->create(prim::ListConstruct));
    for (auto inp : node->inputs()) {
      list_construct->addInput(inp);
    }
    list_construct->output()->setType(
        torch::ListType::create(torch::TensorType::get()));
    list_construct->moveBefore(node);

    // %5 : Tensor[] = prim::CallMethod[name="Run"](%disc_class_p0, %4)
    auto call_method = graph->insertNode(graph->create(
        torch::jit::prim::CallMethod, {val, list_construct->output()}));
    call_method->s_(torch::jit::attr::name, std::move("Run"))
        ->output()
        ->setType(torch::ListType::create(torch::TensorType::get()));
    call_method->moveBefore(node);

    // %6 : Tensor, %7 Tensor = prim::ListUnpack(%5)
    auto list_unpack = graph->insertNode(
        graph->create(prim::ListUnpack, {call_method->output()}));
    list_unpack->eraseOutput(0);
    for (auto out : node->outputs()) {
      auto l_out = list_unpack->addOutput();
      l_out->setType(out->type());
      out->replaceAllUsesWith(l_out);
    }
    list_unpack->moveBefore(node);
    node->destroy();
  }
  return disc_inputs;
}

void InferShapes(const std::shared_ptr<Graph>& graph,
                 c10::ArrayRef<torch::lazy::BackendDataPtr> arguments) {
  std::cout << "InferShape graph: \n" << graph->toString() << std::endl;
  std::cout << "input size: " << graph->inputs().size() << "\t"
            << arguments.size() << std::endl;
  for (size_t i = 0; i < arguments.size(); ++i) {
    auto argument = arguments[i];
    auto input = graph->inputs()[i];
    const auto ts_data = std::static_pointer_cast<TSData>(argument);

    if (ts_data->HasValue()) {
      input->setType(c10::TensorType::create(ts_data->data()));
    }
  }
  torch::jit::PropagateInputShapes(graph);
}

void RegisterDiscClass(const std::shared_ptr<Graph>& graph,
                       const std::string& executable_prog_fname,
                       const std::string& meta_fname) {}

std::vector<c10::IValue> DiscJIT(
    const std::shared_ptr<Graph>& graph,
    c10::ArrayRef<torch::lazy::BackendDataPtr> arguments) {
  // auto graph = computation.graph()->copy();
  // 1. clustering and group DISC nodes
  // 2. conversion from torchscript Graph to mhlo Dialect on DISC nodes
  // 2. register DISC Custom Class
  InferShapes(graph, arguments);
  FusionDiscNodes(graph);
  LOG(WARNING) << "After FusionDiscNodes: \n" << graph->toString() << std::endl;
  auto disc_input = DiscCompilation(graph);
  LOG(WARNING) << "After MhloConversion: \n" << graph->toString() << std::endl;
  return disc_input;
}

}  //  namespace compiler
}  //  namespace torch_disc