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

#include <gtest/gtest.h>

#include <torch/script.h>
#include <fstream>
#include "pytorch_blade/common_utils/utils.h"

TEST(DumpIValues, TestDumpIValues) {
  struct stat f_stat;
  std::string tmp_dir("/tmp/test_replay");
  if (stat(tmp_dir.c_str(), &f_stat) == 0) {
    EXPECT_TRUE(system("rm -rf /tmp/test_replay") == 0);
  }
  EXPECT_TRUE(mkdir(tmp_dir.c_str(), 0755) == 0);

  std::vector<c10::IValue> inputs;
  inputs.emplace_back(torch::randint(0, 9, {10, 20}));
  inputs.emplace_back(torch::randint(0, 9, {5, 5}));
  torch::blade::DumpIValues(inputs, tmp_dir);

  std::ifstream input_stream(
      tmp_dir + "/0.pt", std::ios::in | std::ios::binary);
  std::vector<char> load_input(
      (std::istreambuf_iterator<char>(input_stream)),
      std::istreambuf_iterator<char>());
  auto ivalue = torch::jit::pickle_load(load_input);
  EXPECT_TRUE(ivalue.isTensor());
}
