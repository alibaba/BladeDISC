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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/script.h>
#include "pytorch_blade/common_utils/utils.h"
#include "pytorch_blade/compiler/jit/torch/shape_analysis.h"

// trim from start (in place)
static inline std::string ltrim(std::string s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
  return s;
}

// trim from end (in place)
static inline std::string rtrim(std::string s) {
  s.erase(
      std::find_if(
          s.rbegin(),
          s.rend(),
          [](unsigned char ch) { return !std::isspace(ch); })
          .base(),
      s.end());
  return s;
}

// trim from both ends (in place)
static inline std::string trim(std::string s) {
  s = rtrim(s);
  s = ltrim(s);
  return s;
}

std::vector<std::shared_ptr<torch::jit::Graph>> parse_graphs(
    std::ifstream& ifs) {
  std::stringstream ss;
  std::vector<std::shared_ptr<torch::jit::Graph>> graphs;
  for (std::string line; std::getline(ifs, line);) {
    std::string ltrim_line = ltrim(line);
    if (ltrim_line.find("//") == 0)
      continue;
    if (ltrim_line.find("graph(") == 0) {
      auto graph_str = trim(ss.str());
      if (!graph_str.empty()) {
        auto g = std::make_shared<torch::jit::Graph>();
        torch::jit::parseIR(graph_str, g.get());
        graphs.push_back(g);
      }
      ss.str("");
      ss.clear();
    }
    ss << line << std::endl;
  }
  auto graph_str = trim(ss.str());
  if (!graph_str.empty()) {
    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::parseIR(graph_str, g.get());
    graphs.push_back(g);
  }
  return graphs;
}

char* getCmdOption(char** begin, char** end, const std::string& option) {
  char** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return nullptr;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
  return std::find(begin, end, option) != end;
}

void print_checks(std::ifstream& ifs) {
  for (std::string line; std::getline(ifs, line);) {
    std::string ltrim_line = ltrim(line);
    if (ltrim_line.find("//") == 0)
      std::cout << ltrim_line << std::endl;
  }
}

int main(int argc, char* argv[]) {
  if (cmdOptionExists(argv, argv + argc, "-h")) {
    std::cout
        << "Usage: ./shape_analysis_tool [-h] [--since x.y.z] <shape_test.graph"
        << std::endl;
    return 0;
  }

  char* filename = getCmdOption(argv, argv + argc, "-f");
  if (filename == nullptr) {
    std::cout
        << "Usage: ./shape_analysis_tool [-h] [--since x.y.z] shape_test.graph"
        << std::endl;
    return 0;
  }
  std::ifstream ifs(filename, std::ifstream::in);
  char* version = getCmdOption(argv, argv + argc, "--since");
  if (version != nullptr) {
    const auto& strs = ::torch::blade::split(version, ".");
    if (strs.size() != 2 && strs.size() != 3) {
      std::cout << "Illegal torch version " << version << std::endl;
      return 0;
    }
    int major = std::stoi(strs[0]);
    int minor = std::stoi(strs[1]);
    bool should_skip_check = (major > PYTORCH_MAJOR_VERSION) ||
        (major == PYTORCH_MAJOR_VERSION && minor > PYTORCH_MINOR_VERSION);
    if (should_skip_check) {
      std::cout << "// (" << major << ", " << minor << ") > ("
                << PYTORCH_MAJOR_VERSION << ", " << PYTORCH_MINOR_VERSION << ")"
                << std::endl;
      print_checks(ifs);
      ifs.close();
      return 0;
    }
  }

  auto graphs = parse_graphs(ifs);
  for (auto g : graphs) {
    torch::blade::PropagateInputShapes(g);
    std::cout << *g;
  }
  ifs.close();
  return 0;
}
