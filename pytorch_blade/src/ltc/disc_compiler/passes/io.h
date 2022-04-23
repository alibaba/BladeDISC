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

#include <sys/stat.h>
#include <unistd.h>
#include <sstream>
#include <thread>

namespace torch_disc {
namespace compiler {

// ReadFileBytes reads a file as bytes
inline std::string ReadFileBytes(const std::string& fname) {
  std::ifstream input(fname, std::ios::binary);
  std::vector<char> bytes(
      (std::istreambuf_iterator<char>(input)),
      (std::istreambuf_iterator<char>()));
  return std::string(bytes.begin(), bytes.end());
}

inline std::string GetTempDirectory(std::string dir) {
  auto tid = std::this_thread::get_id();
  uint64_t pid = getpid();
  uint64_t us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
  std::stringstream ss;
  ss << dir << "/" << tid << "-" << pid << "-" << us;
  TORCH_CHECK(
      !mkdir(ss.str().c_str(), 0755), "unable to create dir: " + ss.str());
  return ss.str();
}

} // namespace compiler
} // namespace torch_disc
