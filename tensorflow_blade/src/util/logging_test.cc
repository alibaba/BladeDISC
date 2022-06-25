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

#include "src/util/logging.h"

#include <gtest/gtest.h>

namespace tf_blade {
namespace util {

// NB(xiafei.qiuxf): This trivial test is just to ensure we've always a
// test to stop bazel from complaining about no test when `bazel test`.
TEST(LoggingTest, Logging) {
  LOG(INFO) << "INFO";
  LOG(WARNING) << "WARN";
  LOG(ERROR) << "ERROR";
  VLOG(1) << "VLOG";
  VLOG(2) << "VLOG";
  VLOG(3) << "VLOG";
}

}  // namespace util
}  // namespace tf_blade
