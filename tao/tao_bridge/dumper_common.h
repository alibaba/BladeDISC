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

#ifndef TAO_TAO_BRIDGE_DUMPER_COMMON_H
#define TAO_TAO_BRIDGE_DUMPER_COMMON_H

#include <string>
#include <unordered_set>

#include "tao_bridge/errors.h"

namespace tensorflow {

namespace tao {

struct TaoDumperOptions {
  // Whether use perf triage tool
  // not used in bridge codes any more.
  bool enable_perf_triage_tool;

  // If remove compilation tmp files.
  bool remove_after_compile;

  // If read gpu info and upload. True by default.
  bool dump_gpu_info;

  // min/max number of graphdef's node to dump in online dumper
  int64 graphdef_node_size_min;
  int64 graphdef_node_size_max;

  // the bigger the more info dumped
  // default value 1
  // set to 0 to turn off online dumper entirely
  int64 dump_level;

  // max number of function pbs to dump
  int64 max_dump_func_num;
  // max number of shapes to dump for each function
  int64 max_dump_shape_per_func;
  // max number of function calling timestamp records to dump
  int64 max_dump_timestamp_record;
  // max number of sampled shapes in json file
  int64 max_sampled_shape_num;
  // max number of miss sampled shapes
  // for these miss sampled shapes, we will record the cluster
  // & shape id in memory to distinguish new ones
  int64 max_miss_sampled_shape_num;

  // Files failed to upload to oss sometimes when processes exited abnormally
  // So we upload intermediate results every
  // interval*TAO_PROFILE_STAT_PRINT_CYCLE seconds
  // Disable this when interval is 0
  int64 intermediate_upload_interval;

  // Detect core dump/compiling failures in compiling subprocess
  // Uploads related inputs (not core dump file) to OSS for debug
  int64 max_upload_core_dump;   // failures with core dump
  int64 max_upload_other_fail;  // failures except core dump

  // Path to tao_upload_tool
  std::string tao_upload_tool_path;

  // Path prefix to dump output graph of passes. Contrled by env var
  // `TAO_GRAPH_DUMP_PATH` defaults to '/tmp'.
  std::string graph_dump_path;

  // Capture more environment variables. split by comma. Set it to
  // __CAP_ALL_ENVS__ to capture all visible environment variables.
  bool cap_all_envs{false};
  std::unordered_set<std::string> cap_more_envs{"PATH", "LD_LIBRARY_PATH",
                                                "LD_PRELOAD"};
};

// Get the globally singleton of TaoDumperOptions.
const TaoDumperOptions* GetTaoDumperOptions(bool force_refresh = false);

}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_DUMPER_COMMON_H
