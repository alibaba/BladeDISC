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

#include "mlir/xla/ral/ral_metadata.h"

#include "tensorflow/core/platform/test.h"

namespace tao {
namespace ral {

TEST(MetadataFileAndEmitterTest, BasicTest) {
  MetadataFileEmitter emitter("test.bin");
  ASSERT_TRUE(emitter.emitHeader());
  ASSERT_TRUE(emitter.emitHostConstant("h0", "0000"));
  ASSERT_TRUE(emitter.emitHostConstant("h1", "0001"));
  ASSERT_TRUE(emitter.getNumHostConstantEmitted() == 2);
  ASSERT_TRUE(emitter.emitDeviceConstant("d0", "0000"));
  ASSERT_TRUE(emitter.getNumDeviceConstantEmitted() == 1);
  // dumplicate key
  ASSERT_FALSE(emitter.emitDeviceConstant("d0", "0000"));
  ASSERT_TRUE(emitter.emitTailer());

  ASSERT_TRUE(MetadataFile::loadFromFile("not_exist") == nullptr);
  auto metadata = MetadataFile::loadFromFile("test.bin");
  ASSERT_TRUE(metadata != nullptr);

  const std::string *hstr, *dstr;
  ASSERT_TRUE(metadata->getHostConstant("h0", hstr));
  ASSERT_TRUE(*hstr == "0000");
  ASSERT_TRUE(metadata->getHostConstant("h1", hstr));
  ASSERT_TRUE(*hstr == "0001");
  ASSERT_TRUE(metadata->releaseHostConstant("h0"));
  // not exist after removing
  ASSERT_FALSE(metadata->releaseHostConstant("h0"));
  ASSERT_FALSE(metadata->getHostConstant("h0", hstr));
  ASSERT_TRUE(metadata->getDeviceConstant("d0", dstr));
  ASSERT_TRUE(*dstr == "0000");
  ASSERT_TRUE(metadata->releaseDeviceConstant("d0"));
  // not exist after removing
  ASSERT_FALSE(metadata->getDeviceConstant("d0", dstr));
}

}  // namespace ral
}  // namespace tao
