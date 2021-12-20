#include "tao_bridge/version.h"

#include <iostream>
#include <string>
#include <list>
#include <vector>
#include "tensorflow/core/graph/graph.h"

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
            << "sizeof(std::unordered_map<int, int>): " << sizeof(std::unordered_map<std::string, int>) << std::endl
            << "sizeof(graph): " << sizeof(tensorflow::Graph) << std::endl
            << "sizeof(node): " << sizeof(tensorflow::Node) << std::endl
            << "sizeof(edge): " << sizeof(tensorflow::Edge) << std::endl;
}
