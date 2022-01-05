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

#include "tao_bridge/version.h"

#include "tensorflow/core/graph/graph.h"
#include <iostream>
#include <list>
#include <string>
#include <vector>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define ABI_STR "" STR(_GLIBCXX_USE_CXX11_ABI)

extern "C" void print_tao_build_info() {
  std::cout << "BUILD INFO: " << std::endl
            << "TAO_BUILD_VERSION: " << TAO_BUILD_VERSION << std::endl
            << "TAO_BUILD_GIT_BRANCH: " << TAO_BUILD_GIT_BRANCH << std::endl
            << "TAO_BUILD_GIT_HEAD: " << TAO_BUILD_GIT_HEAD << std::endl
            << "TAO_BUILD_HOST: " << TAO_BUILD_HOST << std::endl
            << "TAO_BUILD_IP: " << TAO_BUILD_IP << std::endl
            << "TAO_BUILD_TIME: " << TAO_BUILD_TIME << std::endl
            << std::endl
            << "ABI INFO: " << std::endl
            << "_GLIBCXX_USE_CXX11_ABI: " << ABI_STR << std::endl
            << "sizeof(std::string): " << sizeof(std::string) << std::endl
            << "sizeof(std::list<int>): " << sizeof(std::list<int>) << std::endl
            << "sizeof(std::unordered_map<int, int>): "
            << sizeof(std::unordered_map<std::string, int>) << std::endl
            << "sizeof(graph): " << sizeof(tensorflow::Graph) << std::endl
            << "sizeof(node): " << sizeof(tensorflow::Node) << std::endl
            << "sizeof(edge): " << sizeof(tensorflow::Edge) << std::endl;
}
