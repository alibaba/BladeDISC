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

#include <common_utils/tempfs.h>

using torch::blade::TempFile;

TEST(Tempfs, TestNormal) {
  TempFile f;
  std::string payload("hello, I'am the payload.");
  ASSERT_TRUE(f.WriteBytesToFile(payload));

  auto loaded_bytes = f.ReadBytesFromFile();
  ASSERT_EQ(payload, loaded_bytes);

  auto loaded_str = f.ReadStringFromFile();
  ASSERT_EQ(payload, loaded_str);

  ASSERT_FALSE(f.GetFilename().empty());
}
