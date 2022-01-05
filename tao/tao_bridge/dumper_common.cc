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

#include "tao_bridge/dumper_common.h"

#include <climits>
#include <mutex>

#include "absl/strings/str_split.h"
#include "tao_bridge/kernels/tao_compilation_info_collector.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace tao {

namespace {

static TaoDumperOptions* opts{nullptr};
static std::once_flag opt_init;
static std::mutex opt_mtx_;
static void AllocateTaoDumperFlags() {
  std::lock_guard<std::mutex> lock(opt_mtx_);
  if (!opts) {
    opts = new TaoDumperOptions();
  }

  // TAO_ENABLE_ONLINE_DUMPER=true equals to set dump_level to 2
  // TAO_ENABLE_ONLINE_DUMPER=false equals to set dump_level to 0
  // TAO_ENABLE_ONLINE_DUMPER is processed in higher priority than
  // TAO_DUMP_LEVEL
  // dump_level define:
  // 0: skip dumping anything
  // 1: dump perf stat and json info files
  // 2: + dump graph pb & shape files We need to set
  // TAO_DUMP_LEVEL=1 explicitly in tianji to enable dumping
  {
    const char* tmp = getenv("TAO_ENABLE_ONLINE_DUMPER");
    if (tmp != nullptr) {
      // TAO_ENABLE_ONLINE_DUMPER is set
      bool enable_online_dumper;
      CHECK_OK(ReadBoolFromEnvVar("TAO_ENABLE_ONLINE_DUMPER", false,
                                  &enable_online_dumper));
      if (enable_online_dumper) {
        opts->dump_level = 2;
      } else {
        opts->dump_level = 0;
      }
    } else {
      // TAO_ENABLE_ONLINE_DUMPER is not set
      CHECK_OK(ReadInt64FromEnvVar("TAO_DUMP_LEVEL", 0, &opts->dump_level));
    }
  }

  // dlopen libcuda.so and collect gpu information by calling cuda apis
  // This may have side effects sometimes.
  // we need to set TAO_DUMP_GPU_INFO=true explicitly in tianji to enable
  // dumping by default
  CHECK_OK(
      ReadBoolFromEnvVar("TAO_DUMP_GPU_INFO", false, &opts->dump_gpu_info));

  CHECK_OK(ReadBoolFromEnvVar("TAO_REMOVE_AFTER_COMPILE", true,
                              &opts->remove_after_compile));

  CHECK_OK(ReadStringFromEnvVar("TAO_UPLOAD_TOOL_PATH", "",
                                &opts->tao_upload_tool_path));
  CHECK_OK(ReadStringFromEnvVar("TAO_GRAPH_DUMP_PATH", "/tmp",
                                &opts->graph_dump_path));

  std::string capture_more_envs;
  CHECK_OK(
      ReadStringFromEnvVar("TAO_CAPTURE_MORE_ENVS", "", &capture_more_envs));
  if (!capture_more_envs.empty()) {
    if (capture_more_envs == "__CAP_ALL_ENVS__") {
      opts->cap_all_envs = true;
    } else {
      for (auto s : absl::StrSplit(capture_more_envs, ',')) {
        opts->cap_more_envs.emplace(std::string(s));
      }
    }
  }

  // disable the timestamp collecting by default for now
  CHECK_OK(ReadInt64FromEnvVar("TAO_MAX_DUMP_TIMESTAMP_RECORD", 0,
                               &opts->max_dump_timestamp_record));

  // max sampled shape number in json["calls"]. others are put into missed
  // sampled ones.
  CHECK_OK(ReadInt64FromEnvVar("TAO_MAX_SAMPLED_SHAPE_NUM", 100000,
                               &opts->max_sampled_shape_num));
  CHECK_OK(ReadInt64FromEnvVar("TAO_MAX_MISS_SAMPLED_SHAPE_NUM", 100000,
                               &opts->max_miss_sampled_shape_num));

  // knobs for graph dumping control
  CHECK_OK(ReadInt64FromEnvVar("TAO_ONLINE_DUMPER_NODE_SIZE_MIN", 0,
                               &opts->graphdef_node_size_min));
  CHECK_OK(ReadInt64FromEnvVar("TAO_ONLINE_DUMPER_NODE_SIZE_MAX", LLONG_MAX,
                               &opts->graphdef_node_size_max));
  CHECK_OK(ReadInt64FromEnvVar("TAO_MAX_DUMP_FUNC_NUM", 100,
                               &opts->max_dump_func_num));
  CHECK_OK(ReadInt64FromEnvVar("TAO_MAX_DUMP_SHAPE_PER_FUNC", 10,
                               &opts->max_dump_shape_per_func));

  // how frequent to update sess_<idx>.perf.intermediate onto oss
  // interval seconds = TAO_PROFILE_STAT_PRINT_CYCLE * TAO_INTER_UPLOAD_INTERVAL
  // TAO_PROFILE_STAT_PRINT_CYCLE default value: 600 seconds (10 minutes)
  CHECK_OK(ReadInt64FromEnvVar("TAO_INTER_UPLOAD_INTERVAL", 3,
                               &opts->intermediate_upload_interval));

  // knobs for uploading compile fail cases
  CHECK_OK(ReadInt64FromEnvVar("TAO_MAX_UPLOAD_CORE_DUMP", 10,
                               &opts->max_upload_core_dump));
  CHECK_OK(ReadInt64FromEnvVar("TAO_MAX_UPLOAD_OTHER_FAIL", 10,
                               &opts->max_upload_other_fail));
}

}  // namespace

const TaoDumperOptions* GetTaoDumperOptions(bool force_refresh) {
  std::call_once(opt_init, &AllocateTaoDumperFlags);
  if (force_refresh) {
    AllocateTaoDumperFlags();
    // update captured env
    TaoCompInfoCollector::Get().CaptureEnvVars();
  }
  return opts;
}

}  // namespace tao
}  // namespace tensorflow
