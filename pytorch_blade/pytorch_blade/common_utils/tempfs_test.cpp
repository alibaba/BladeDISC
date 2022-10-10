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

#include <gtest/gtest.h>

#include <fstream>

#include "pytorch_blade/common_utils/tempfs.h"

using torch::blade::TempFile;

TEST(Tempfs, TestNormal) {
  std::string fname;
  {
    TempFile tmp_file("ThePrefix");
    fname = tmp_file.GetFilename();
    std::cerr << "Fname: " << fname << std::endl;
    std::string payload("hello, I'am the payload.");
    ASSERT_TRUE(tmp_file.WriteBytesToFile(payload));

    auto loaded_bytes = tmp_file.ReadBytesFromFile();
    ASSERT_EQ(payload, loaded_bytes);

    auto loaded_str = tmp_file.ReadStringFromFile();
    ASSERT_EQ(payload, loaded_str);

    ASSERT_FALSE(tmp_file.GetFilename().empty());
    ASSERT_TRUE(
        tmp_file.GetFilename().find("/tmp/ThePrefix") != std::string::npos);

    std::ifstream f(fname);
    ASSERT_TRUE(f.good()); // tempfile still acessable
  }
  std::ifstream f(fname);
  ASSERT_FALSE(f.good()); // tempfile not still acessable
}
