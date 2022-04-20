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

#include "torch_disc/csrc/disc_compiler/passes/register_disc_class.h"

#include "compiler/mlir/converters/mhlo_conversion.h"
#include "torch_disc/csrc/disc_compiler/passes/disc_class.h"
#include "torch_disc/csrc/disc_compiler/passes/io.h"

namespace torch_disc {
namespace compiler {
using namespace ::torch::jit;

std::string DiscCMD(const std::string& mlir_fname,
                    const std::string& out_fname) {
  std::stringstream ss;
  std::string logf = mlir_fname + ".log";
  auto binary_path = "./disc_compiler_main";
  ss << binary_path << " " << mlir_fname << " " << out_fname << " > " << logf
     << " 2>&1 ";
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

const std::vector<torch::jit::Value*> ArrayToVector(
    c10::ArrayRef<torch::jit::Value*> values) {
  std::vector<torch::jit::Value*> const_vals;
  std::copy(values.begin(), values.end(), std::back_inserter(const_vals));
  return const_vals;
}

// ReplaceDiscClass replace the disc node with prim::CallMethod
void ReplaceDiscClass(const std::shared_ptr<Graph>& graph,
                      torch::jit::Node* disc_node, std::string& param_name,
                      c10::IValue& disc_obj) {
  // add DiscClass object as graph input
  auto val = graph->addInput(param_name);
  val->setType(disc_obj.type());

  // %3 : Tensor[] = ListConstruct(%1, %2)
  auto list_construct = graph->insertNode(graph->create(prim::ListConstruct));
  for (auto inp : disc_node->inputs()) {
    list_construct->addInput(inp);
  }
  list_construct->output()->setType(
      torch::ListType::create(torch::TensorType::get()));
  list_construct->moveBefore(disc_node);

  // %5 : Tensor[] = prim::CallMethod[name="Run"](%disc_class_p0, %4)
  auto call_method = graph->insertNode(graph->create(
      torch::jit::prim::CallMethod, {val, list_construct->output()}));
  call_method->s_(torch::jit::attr::name, std::move("Run"))
      ->output()
      ->setType(torch::ListType::create(torch::TensorType::get()));
  call_method->moveBefore(disc_node);

  // %6 : Tensor, %7 Tensor = prim::ListUnpack(%5)
  auto list_unpack = graph->insertNode(
      graph->create(prim::ListUnpack, {call_method->output()}));
  list_unpack->eraseOutput(0);
  for (auto out : disc_node->outputs()) {
    auto l_out = list_unpack->addOutput();
    l_out->setType(out->type());
    out->replaceAllUsesWith(l_out);
  }
  list_unpack->moveBefore(disc_node);
  disc_node->destroy();
}

std::vector<c10::IValue> RegisterDiscClass(
    const std::shared_ptr<Graph>& graph) {
  std::vector<c10::IValue> disc_inputs;
  std::vector<torch::jit::Node*> disc_nodes;
  std::copy_if(graph->nodes().begin(), graph->nodes().end(),
               std::back_inserter(disc_nodes), [](torch::jit::Node* node) {
                 return node->kind() == prim::FusionGroup;
               });

  for (auto node : disc_nodes) {
    auto sub_graph = node->g(attr::Subgraph);
    auto state = std::make_shared<torch::blade::backends::EngineState>();
    std::vector<torch::blade::backends::EngineState::TensorType> inputs,
        outputs;

    auto cvt_ret = MhloConversaion(sub_graph);
    auto output_fname =
        CallDiscCompiler(std::get<0>(cvt_ret) /*mlir file name*/);
    state->set_engine_bytes(ReadFileBytes(output_fname));
    state->set_model_proto(ReadFileBytes(output_fname + ".pbtxt"));
    for (auto input : sub_graph->inputs()) {
      inputs.push_back(torch::blade::backends::TensorInfo(*input));
    }
    for (auto output : sub_graph->outputs()) {
      outputs.push_back(torch::blade::backends::TensorInfo(*output));
    }
    state->set_inputs(inputs);
    state->set_outputs(outputs);

    // add DiscClass object as graph input
    auto disc_class = torch::make_custom_class<DiscClass>(state);
    auto input_name = c10::str("disc_class_p", disc_inputs.size());
    disc_inputs.push_back(disc_class);

    ReplaceDiscClass(graph, node, input_name, disc_class);
  }
  return disc_inputs;
}
}  //  namespace compiler
}  //  namespace torch_disc
