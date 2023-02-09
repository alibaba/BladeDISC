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

#include <climits>
#include <mutex>

#include "tao_bridge/kernels/tao_compilation_info_collector.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace tao {

namespace {

static TaoBridgeOptions* opts{nullptr};
static std::once_flag opt_init;
static std::mutex opt_mtx_;
static void AllocateTaoBridgeFlags() {
  std::lock_guard<std::mutex> lock(opt_mtx_);
  if (!opts) {
    opts = new TaoBridgeOptions();
  }
  CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_TAO", false, &opts->enable_tao));
  if (!opts->enable_tao) {
    CHECK_OK(ReadBoolFromEnvVar("BRIDGE_ENABLE_TAO", false, &opts->enable_tao));
  }
  if (!opts->enable_tao) {
    LOG(WARNING) << "TAO compiler disabled, set env var TF_ENABLE_TAO or "
                 << "BRIDGE_ENABLE_TAO to true to enable TAO compiler.";
  }

  CHECK_OK(ReadStringFromEnvVar("TAO_COMPILER_PATH", "tao_compiler_main",
                                &opts->tao_compiler_path));
  CHECK_OK(ReadBoolFromEnvVar("TAO_VERBOSE_COMPILATION_ERR_LOG", false,
                              &opts->verbose_compilation_err_log));
  CHECK_OK(ReadBoolFromEnvVar("TAO_ENFORCE_VERBOSE_COMPILATION_LOG", false,
                              &opts->verbose_compilation_log));
  CHECK_OK(ReadStringFromEnvVar("DISC_COMPILATION_CACHE_PATH", "",
                                &opts->disc_cache_path));
  CHECK_OK(ReadBoolFromEnvVar("TAO_ENABLE_MLIR", false, &opts->enable_mlir));

  CHECK_OK(ReadBoolFromEnvVar("TAO_MLIR_BRANCH_ONLY", true,
                              &opts->tao_mlir_branch_only));

  CHECK_OK(ReadBoolFromEnvVar("TAO_DUMP_PASS_OUTPUT", false,
                              &opts->dump_pass_output));

  CHECK_OK(ReadBoolFromEnvVar("TAO_DUMP_WITH_UNIQUE_ID", true,
                              &opts->dump_with_unique_id));

  CHECK_OK(ReadBoolFromEnvVar("TAO_COMPILATION_MODE_ASYNC", false,
                              &opts->tao_launch_async_compilation));
  CHECK_OK(ReadBoolFromEnvVar("TAO_ENABLE_CHECK", false,
                              &opts->tao_launch_enable_check));
  CHECK_OK(ReadBoolFromEnvVar("TAO_ENABLE_FALLBACK", true,
                              &opts->tao_launch_enable_fallback));
  CHECK_OK(ReadBoolFromEnvVar("TAO_ENABLE_CONTROL_FLOW", false,
                              &opts->tao_enable_control_flow));

  CHECK_OK(ReadBoolFromEnvVar("TAO_BACE_REFORMAT_ENABLED", false,
                              &opts->bace_reformat_enabled));
  CHECK_OK(ReadInt64FromEnvVar("TAO_BACE_REFORMAT_MAX_DIM_BAR", 1024 * 8 - 1,
                               &opts->bace_reformat_max_dim_bar));
  CHECK_OK(ReadInt64FromEnvVar("TAO_BACE_REFORMAT_MIN_DIM_BAR", 512 - 1,
                               &opts->bace_reformat_min_dim_bar));
  CHECK_OK(ReadInt64FromEnvVar("TAO_BACE_REFORMAT_SIZE_BAR", 10000 * 512 - 1,
                               &opts->bace_reformat_size_bar));

  CHECK_OK(ReadInt64FromEnvVar("TAO_PROFILING_GUIDED_COMPILATION_MODE", 0,
                               &opts->profiling_guided_compilation_mode));

  CHECK_OK(ReadInt64FromEnvVar("TAO_PROFILING_GUIDED_COMPILATION_LAZY_CALLS", 3,
                               &opts->profiling_guided_compilation_lazy_calls));

  // Enable debug mode of disc if true.
  CHECK_OK(ReadBoolFromEnvVar("DISC_DEBUG", false, &opts->disc_debug_mode));
  CHECK_OK(ReadBoolFromEnvVar("DISC_FORCE_FALLBACK", false,
                              &opts->disc_force_fallback));

  CHECK_OK(ReadInt64FromEnvVar(
      "TAO_PROFILING_GUIDED_COMPILATION_PROFILING_TIME_BY_MIN", 5,
      &opts->profiling_guided_compilation_profiling_time_by_min));

  CHECK_OK(ReadInt64FromEnvVar("TAO_PROFILING_GUIDED_COMPILATION_CANDIDATES",
                               1000,
                               &opts->profiling_guided_compilation_candidates));

  // TAO WHALE features
  CHECK_OK(ReadBoolFromEnvVar("TAO_WHALE_SEPERATE_MICRO_BATCH", false,
                              &opts->seperate_whale_micro_batch));

  CHECK_OK(ReadBoolFromEnvVar(
      "TAO_EXPERIMENTAL_ENABLE_MLIR_WHOLE_GRAPH_COMPILATION", true,
      &opts->experimental_enable_mlir_whole_graph_compilation));

  CHECK_OK(ReadBoolFromEnvVar("TAO_SKIP_TRAINING_GRAPH", false,
                              &opts->skip_training_graph));
  CHECK_OK(ReadInt64FromEnvVar("TAO_OPTIMIZATION_PASS_START", 0,
                               &opts->optimization_pass_start));

  CHECK_OK(ReadStringFromEnvVar("TAO_OP_TYPE_CLUSTERING_BLACK_LIST", "",
                                &opts->op_type_clustering_black_list));

  CHECK_OK(ReadStringFromEnvVar("TAO_OP_NAME_CLUSTERING_BLACK_LIST", "",
                                &opts->op_name_clustering_black_list));

  CHECK_OK(ReadInt64FromEnvVar("TAO_COMPILATION_CACHE_CAPACITY", 1000,
                               &opts->cache_capacity));

  CHECK_OK(ReadInt64FromEnvVar("TAO_TRAIN_TASK_MAX_CLUSTER_SIZE", -1,
                               &opts->train_task_max_cluster_size));
  CHECK_OK(ReadInt64FromEnvVar("TAO_TRAIN_TASK_MIN_CLUSTER_SIZE", -1,
                               &opts->train_task_min_cluster_size));
  CHECK_OK(ReadBoolFromEnvVar(
      "TAO_EXPERIMENTAL_ENABLE_CPU_SPARSE_OPS_COMPILATION", false,
      &opts->experimental_enable_cpu_sparse_ops_compilation));
}

}  // namespace

const TaoBridgeOptions* GetTaoBridgeOptions(bool force_refresh) {
  std::call_once(opt_init, &AllocateTaoBridgeFlags);
  if (force_refresh) {
    AllocateTaoBridgeFlags();
  }
  return opts;
}

}  // namespace tao
}  // namespace tensorflow
