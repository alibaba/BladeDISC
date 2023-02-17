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

#ifndef TAO_TAO_BRIDGE_COMMON_H_
#define TAO_TAO_BRIDGE_COMMON_H_

#include <string>

#include "tao_bridge/errors.h"
#include "tao_bridge/tf_compatible.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

namespace tao {

struct OptionalTensor {
  string name;           // A descriptive name
  bool present = false;  // Is the tensor present?
  Tensor value;          // If present, what is the Tensor's value?
};

struct TaoBridgeOptions {
  // If tao bridge is enabled. Contrled by env var `BRIDGE_ENABLE_TAO` defaults
  // to false.
  bool enable_tao;

  // Path to tao_compiler_main.
  std::string tao_compiler_path;
  // If print compiler log on error.
  bool verbose_compilation_err_log;
  // If enforce print compiler log.
  bool verbose_compilation_log;

  // Compilation cache dump path.
  std::string disc_cache_path;
  // Whether to enable mlir compilation.
  bool enable_mlir;

  // If true, tao launch op will only consider mlir branch instead of both xla
  // and mlir branch. This is mainly used to ease debug.
  bool tao_mlir_branch_only;

  // Whether to dump outout graph of passes. Contrled by env var
  // `TAO_DUMP_PASS_OUTPUT` defaults to false.
  bool dump_pass_output;

  // Whether to dump graph with unique id to avoid file overwriting. Contrled by
  // env var `TAO_DUMP_WITH_UNIQUE_ID` defaults to false.
  bool dump_with_unique_id;

  // Whether TaoLaunchOp use async compilation.
  bool tao_launch_async_compilation;
  // Whether to enable result check in TaoLaunchOp.
  bool tao_launch_enable_check;
  // Whether to fallback to TF execution when error occurs in TaoLaunchOp.
  bool tao_launch_enable_fallback;
  // Whether to enable functionalize control flow pass
  bool tao_enable_control_flow;

  // Enable debug mode of disc if true.
  bool disc_debug_mode;
  // Go fallback path if is true.
  bool disc_force_fallback;

  // Whether to enable bace reformat pass to cast come fp32 const to fp16.
  bool bace_reformat_enabled;
  // Threashold for max dim.
  int64 bace_reformat_max_dim_bar;
  // Threashold for min dim.
  int64 bace_reformat_min_dim_bar;
  // Threashold for tensor size.
  int64 bace_reformat_size_bar;

  // compilation cache capacity
  int64 cache_capacity;

  // tao whale options
  bool seperate_whale_micro_batch;

  // Experimental features
  bool experimental_enable_mlir_whole_graph_compilation;

  // profiling_guided_compilation_mode:
  // 0. disable feature 1. profiling guided 2. lazy compilation
  int64 profiling_guided_compilation_mode;
  // stat cluster perf after called n times, similar to lazy compilation
  int64 profiling_guided_compilation_lazy_calls;
  // set proifling time (minutes)
  int64 profiling_guided_compilation_profiling_time_by_min;
  // set candidate size
  int64 profiling_guided_compilation_candidates;
  // skip subgraph containing train op
  bool skip_training_graph;

  // set tao optimization pass start
  int64 optimization_pass_start;
  // set op type clustring black list
  std::string op_type_clustering_black_list;
  // set op name clustring black list
  std::string op_name_clustering_black_list;

  // set max/min cluster size for training tasks
  int64 train_task_max_cluster_size;
  int64 train_task_min_cluster_size;

  // Support for ops in feature columns are still in-progress
  // Default to false
  bool experimental_enable_cpu_sparse_ops_compilation;
};

// Get the globally singleton of TaoBridgeOptions.
// `force_refresh` is only used for debug. DO NOT update tao envs after
// it's initialized.
const TaoBridgeOptions* GetTaoBridgeOptions(bool force_refresh = false);

}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_COMMON_H_
