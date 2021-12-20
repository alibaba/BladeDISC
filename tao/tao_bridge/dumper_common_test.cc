#include "tao_bridge/dumper_common.h"

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
  auto* opt = GetTaoDumperOptions(refresh);
  EXPECT_FALSE(opt->remove_after_compile);
  EXPECT_EQ(opt->tao_upload_tool_path, "/abc");
  EXPECT_EQ(opt->graph_dump_path, "/text");
  EXPECT_EQ(opt->max_dump_timestamp_record, 1);
  EXPECT_EQ(opt->graphdef_node_size_min, 2);
  EXPECT_EQ(opt->graphdef_node_size_max, 3);
  EXPECT_EQ(opt->max_dump_func_num, 4);
  EXPECT_EQ(opt->max_dump_shape_per_func, 5);
  EXPECT_EQ(opt->intermediate_upload_interval, 10);
  EXPECT_EQ(opt->max_upload_core_dump, 50);
  EXPECT_EQ(opt->max_upload_other_fail, 70);
  EXPECT_EQ(opt->cap_more_envs.count("ENV1"), 1);
}

TEST(CommonTest, TestGetTaoDumperOptions) {
  std::vector<EnvItem> env_items = {
      {"TAO_ENABLE_ONLINE_DUMPER", "true", ""},
      {"TAO_REMOVE_AFTER_COMPILE", "false", ""},
      {"TAO_UPLOAD_TOOL_PATH", "/abc", ""},
      {"TAO_GRAPH_DUMP_PATH", "/text", ""},
      {"TAO_MAX_DUMP_TIMESTAMP_RECORD", "1", ""},
      {"TAO_ONLINE_DUMPER_NODE_SIZE_MIN", "2", ""},
      {"TAO_ONLINE_DUMPER_NODE_SIZE_MAX", "3", ""},
      {"TAO_MAX_DUMP_FUNC_NUM", "4", ""},
      {"TAO_MAX_DUMP_SHAPE_PER_FUNC", "5", ""},
      {"TAO_INTER_UPLOAD_INTERVAL", "10", ""},
      {"TAO_CAPTURE_MORE_ENVS", "ENV1,ENV2", ""},
      {"TAO_MAX_UPLOAD_CORE_DUMP", "50", ""},
      {"TAO_MAX_UPLOAD_OTHER_FAIL", "70", ""},
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
