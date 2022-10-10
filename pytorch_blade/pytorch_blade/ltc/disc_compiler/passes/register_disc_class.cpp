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

#include "pytorch_blade/ltc/disc_compiler/passes/register_disc_class.h"

#include "pytorch_blade/common_utils/logging.h"
#include "pytorch_blade/compiler/backends/engine_class.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_conversion.h"
#include "pytorch_blade/compiler/mlir/runtime/disc_engine.h"
#include "pytorch_blade/ltc/disc_compiler/passes/io.h"
#include "pytorch_blade/ltc/disc_compiler/replay.h"

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/script.h>

#define _GNU_SOURCE
#include <dlfcn.h>
namespace torch_disc {
namespace compiler {
using namespace ::torch::jit;

std::string CurrentLibLocation() {
  Dl_info dl_info;
  dladdr((void*)CurrentLibLocation, &dl_info);
  auto fname = std::string(dl_info.dli_fname);
  return fname.substr(0, fname.find_last_of("/"));
}

std::string DiscCMD(
    const std::string& mlir_fname,
    const std::string& out_fname) {
  std::stringstream ss;
  std::string logf = mlir_fname + ".log";
  std::string binary_path = CurrentLibLocation() + "/disc_compiler_main";
  if (GRAPH_DEBUG_ENABLED)
    binary_path = "TF_CPP_VMODULE=disc_compiler=1 " + binary_path;
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
  outfile.close();
  return std::make_tuple(in_fname, input_dev_str, output_dev_str);
}

std::tuple<std::string, std::string, int> CallDiscCompiler(
    const std::string& mlir_fname) {
  std::string out_fname = mlir_fname + ".out";
  std::string cmd = DiscCMD(mlir_fname, out_fname);
  auto ret = std::system(cmd.c_str());
  return {cmd, out_fname, ret};
}

const std::vector<torch::jit::Value*> ArrayToVector(
    c10::ArrayRef<torch::jit::Value*> values) {
  std::vector<torch::jit::Value*> const_vals;
  std::copy(values.begin(), values.end(), std::back_inserter(const_vals));
  return const_vals;
}

// ReplaceDiscClass replace the disc node with prim::CallMethod
void ReplaceDiscClass(
    const std::shared_ptr<Graph>& graph,
    torch::jit::Node* disc_node,
    std::string& param_name,
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
  call_method->s_(torch::jit::attr::name, std::move("execute"))
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
  torch::blade::backends::InitTorchBladeEngine();
  torch::blade::disc::InitBladeDiscEngine();
  std::vector<c10::IValue> disc_inputs;
  std::vector<torch::jit::Node*> disc_nodes;
  std::copy_if(
      graph->nodes().begin(),
      graph->nodes().end(),
      std::back_inserter(disc_nodes),
      [](torch::jit::Node* node) { return node->kind() == prim::FusionGroup; });
  for (size_t i = 0; i < disc_nodes.size(); ++i) {
    auto node = disc_nodes[i];
    auto sub_graph = node->g(attr::Subgraph);
    auto state = std::make_shared<torch::blade::backends::EngineState>();
    std::vector<torch::blade::backends::EngineState::TensorType> inputs,
        outputs;
    GRAPH_DUMP("Compile before mhlo conversion \n ", sub_graph);
    auto cvt_ret = MhloConversaion(sub_graph);
    auto ret = CallDiscCompiler(std::get<0>(cvt_ret) /*mlir file name*/);
    auto ret_code = std::get<2>(ret);
    auto cmd = std::get<0>(ret);
    if (ret_code != 0) {
      GRAPH_DEBUG("disc compilation fallback to torchscript, cmd: " + cmd);
      continue;
    }
    auto output_fname = std::get<1>(ret);
    GRAPH_DEBUG(
        "disc compile fusionGroup ",
        node->maybeSchema()->operator_name(),
        " cmd: ",
        cmd);

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
    state->set_backend_name(torch::blade::disc::GetBackendName());
    // using hash string as debug attr
    auto disc_hash =
        torch::lazy::DataHash(sub_graph.get(), sub_graph->toString().size());
    auto disc_hash_str = torch::lazy::HashToString(disc_hash);
    GRAPH_DEBUG("registry disc engine with debug attr: ", disc_hash_str);
    auto fallback_module = ConvertGraphToModule(sub_graph->copy());
    std::ostringstream buf;
    fallback_module.save(buf);

    // add DiscClass object as graph input
    auto disc_class = torch::blade::backends::create_engine(
        *state, disc_hash_str, buf.str(), "");
    auto input_name = c10::str("disc_class_p", disc_inputs.size());
    disc_inputs.push_back(disc_class);

    ReplaceDiscClass(graph, node, input_name, disc_class);
  }
  return disc_inputs;
}
} //  namespace compiler
} //  namespace torch_disc
