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

#include "tao_bridge/common.h"

#include <stdlib.h>

#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace tao {
namespace {

struct EnvItem {
  std::string name;
  std::string new_val;
  std::string old_val;
};

void BackUpAndSetEnv(std::vector<EnvItem>* items) {
  for (auto& item : *items) {
    TF_CHECK_OK(ReadStringFromEnvVar(item.name, "", &item.old_val));
    int ret = setenv(item.name.c_str(), item.new_val.c_str(), 1);
    CHECK(ret == 0);
  }
}

void RestoreEnv(const std::vector<EnvItem>& items) {
  for (auto& item : items) {
    int ret = setenv(item.name.c_str(), item.old_val.c_str(), 1);
    CHECK(ret == 0);
  }
}

void DoAsserts(bool refresh) {
  auto* opt = GetTaoBridgeOptions(refresh);
  EXPECT_TRUE(opt->enable_tao);
  EXPECT_TRUE(opt->dump_pass_output);
  EXPECT_FALSE(opt->dump_with_unique_id);
  EXPECT_TRUE(opt->tao_launch_async_compilation);
  EXPECT_TRUE(opt->tao_launch_enable_check);
  EXPECT_FALSE(opt->tao_launch_enable_fallback);
  EXPECT_TRUE(opt->bace_reformat_enabled);
  EXPECT_EQ(opt->bace_reformat_max_dim_bar, 333);
  EXPECT_EQ(opt->bace_reformat_min_dim_bar, 666);
  EXPECT_EQ(opt->bace_reformat_size_bar, 999);
  EXPECT_EQ(opt->tao_compiler_path, "tao_compiler_main");
  EXPECT_EQ(opt->disc_cache_path, "/tao_dump");
  EXPECT_EQ(opt->profiling_guided_compilation_mode, 1);
  EXPECT_EQ(opt->profiling_guided_compilation_lazy_calls, 4);
  EXPECT_EQ(opt->profiling_guided_compilation_profiling_time_by_min, 6);
  EXPECT_EQ(opt->profiling_guided_compilation_candidates, 999);
  EXPECT_TRUE(opt->verbose_compilation_log);
}

TEST(CommonTest, TestGetTaoBridgeOptions) {
  std::vector<EnvItem> env_items = {
      {"TF_ENABLE_TAO", "false", ""},
      {"BRIDGE_ENABLE_TAO", "true", ""},
      {"TAO_DUMP_PASS_OUTPUT", "true", ""},
      {"TAO_DUMP_WITH_UNIQUE_ID", "false", ""},
      {"TAO_COMPILATION_MODE_ASYNC", "true", ""},
      {"TAO_ENABLE_CHECK", "true", ""},
      {"TAO_ENABLE_FALLBACK", "false", ""},
      {"TAO_BACE_REFORMAT_ENABLED", "true", ""},
      {"TAO_BACE_REFORMAT_MAX_DIM_BAR", "333", ""},
      {"TAO_BACE_REFORMAT_MIN_DIM_BAR", "666", ""},
      {"TAO_BACE_REFORMAT_SIZE_BAR", "999", ""},
      {"TAO_COMPILER_PATH", "tao_compiler_main", ""},
      {"DISC_COMPILATION_CACHE_PATH", "/tao_dump", ""},
      {"TAO_PROFILING_GUIDED_COMPILATION_MODE", "1", ""},
      {"TAO_PROFILING_GUIDED_COMPILATION_LAZY_CALLS", "4", ""},
      {"TAO_PROFILING_GUIDED_COMPILATION_PROFILING_TIME_BY_MIN", "6", ""},
      {"TAO_PROFILING_GUIDED_COMPILATION_CANDIDATES", "999", ""},
      {"TAO_ENFORCE_VERBOSE_COMPILATION_LOG", "true", ""},
  };

  BackUpAndSetEnv(&env_items);
  DoAsserts(true);
  RestoreEnv(env_items);

  // Options won't change once allocated, even though env vars are recovered.
  DoAsserts(false);
}

}  // namespace
}  // namespace tao
}  // namespace tensorflow
