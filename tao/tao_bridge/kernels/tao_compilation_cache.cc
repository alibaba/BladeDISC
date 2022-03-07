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

#include "tao_bridge/kernels/tao_compilation_cache.h"

#include <dlfcn.h>
#include <sys/wait.h>

#include <atomic>
#include <chrono>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <numeric>
#include <queue>
#include <thread>

#include "absl/strings/str_cat.h"
#include "tao_bridge/common.h"
#include "tao_bridge/dumper_common.h"
#include "tao_bridge/passes/tao_build_tao_op_pass.h"
#include "tao_bridge/tao_util.h"
#include "tao_bridge/tf/dump_graph.h"
#include "tao_bridge/tf/subprocess.h"
#include "tao_bridge/tf_compatible.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

namespace tao {

namespace {

class AsyncCompilationMgr {
 public:
  explicit AsyncCompilationMgr(int64 cache_capacity,
                               int64 compilation_cycle_in_ms,
                               int64 max_parallel_compilation_thread_num,
                               int64 num_group)
      : entries_(cache_capacity),
        compilation_cycle_in_ms_(compilation_cycle_in_ms),
        max_parallel_compilation_thread_num_(
            max_parallel_compilation_thread_num),
        num_groups_(num_group) {
    for (int i = 0; i < max_parallel_compilation_thread_num_; ++i) {
      threads_.push_back(Env::Default()->StartThread(
          ThreadOptions(), "TAO-ASYNC-COMPILATION",
          [this, i]() { HandleEntryLoop(i % this->num_groups_); }));
    }
  }

  ~AsyncCompilationMgr() { NotifyAndWaitToStop(); }

  int AllocateEntry() {
    if (static_cast<size_t>(next_entry_slot_) >= entries_.size()) {
      return -1;
    }
    int slot = next_entry_slot_++;
    VLOG(2) << "AllocateEntry: " << slot;
    return static_cast<size_t>(slot) >= entries_.size() ? -1 : slot;
  }

  struct Entry {
    // Have we tried compiling this entry?
    std::atomic<bool> compiled{false};

    // Did compilation succeed?
    Status compilation_status;

    // The XLA executable compiled from <computation>. May be null if no
    // executable has been built.
    std::unique_ptr<Executable> executable;

    // For info collector:
    // We initialized a shape in collector at first DumpInfo() call
    // Skip calling collector APIs before initializing done
    mutex col_shape_init_mtx_;
    std::string func_id;
    uint64_t shape_hash_sig;
    bool col_shape_initialized{false};
    bool col_shape_compile_started{false};
  };

  class CancellationMgr {
   public:
    using Handle = int;
    using CancellationAction = std::function<void()>;

    constexpr static Handle kInvalidHandle = -1;

    // void set_parent(AsyncCompilationMgr* parent) { parent_ = parent; }
    // AsyncCompilationMgr* parent() { return parent_; }

    std::unique_ptr<mutex_lock> Lock() {
      using mlock = mutex_lock;
      return std::unique_ptr<mutex_lock>(new mlock(mutex_));
    }

    ~CancellationMgr() { NotifyCancellation(); }

    void NotifyCancellation() {
      mutex_lock l(mutex_);
      stop_ = true;
      for (auto& pair : actions_) {
        if (pair.second) {
          VLOG(2) << "early stop compilation process...";
          pair.second();
          pair.second = CancellationAction{};
        }
      }
    }

    Handle RegisterCancellationAction(CancellationAction action) {
      mutex_lock l(mutex_);
      if (stop_) {
        action();
        return kInvalidHandle;
      }
      Handle handle = actions_.size();
      if (!free_handles_.empty()) {
        handle = free_handles_.back();
        free_handles_.pop_back();
      }
      actions_[handle] = action;
      return handle;
    }
    Status RemoveCancellationAction(Handle handle) {
      if (handle == kInvalidHandle) {
        return Status::OK();
      }
      mutex_lock l(mutex_);
      auto iter = actions_.find(handle);
      TF_RET_CHECK(iter != actions_.end());
      free_handles_.push_back(iter->first);
      actions_.erase(iter);
      return Status::OK();
    }

   private:
    mutex mutex_;
    bool stop_ = false;
    std::vector<Handle> free_handles_;
    std::unordered_map<Handle, CancellationAction> actions_;
    AsyncCompilationMgr* parent_ = nullptr;
  };

  Entry& GetEntry(int slot) {
    CHECK_GE(slot, 0);
    CHECK_LT(slot, static_cast<int>(entries_.size()));
    return entries_[slot];
  }

  using Action =
      std::function<Status(std::unique_ptr<Executable>*, CancellationMgr*)>;

  void EnqueueAction(int slot, Action action, int tag = 0) {
    mutex_lock l(mutex_);
    GetActionQueue(tag % this->num_groups_).emplace(slot, action);
  }

  constexpr static int64 kMaxCompilationCycle = 16 * 1000;

  void HandleEntryLoop(int tag) {
    while (!stop_) {
      std::vector<std::pair<int, Action>> actions;
      int64 cur_compilation_cycle = compilation_cycle_in_ms_;
      {
        mutex_lock l(mutex_);
        auto& action_queue = GetActionQueue(tag);
        if (!action_queue.empty()) {
          actions.push_back(action_queue.front());
          action_queue.pop();
        } else {
          cur_compilation_cycle <<= 1;
          if (cur_compilation_cycle > kMaxCompilationCycle) {
            cur_compilation_cycle = kMaxCompilationCycle;
          }
        }
        // std::swap(actions, actions_);
      }
      for (auto& pair : actions) {
        int slot = pair.first;
        auto& entry = GetEntry(slot);
        {
          mutex_lock l(entry.col_shape_init_mtx_);
          entry.col_shape_compile_started = true;
          if (entry.col_shape_initialized) {
            TaoCompInfoCollector::Get().UpdateShapeCompileStatus(
                entry.func_id, entry.shape_hash_sig, STATUS_COMPILE_STARTED);
          }
        }
        entry.compilation_status =
            pair.second(&entry.executable, &cancellation_mgr_);
        {
          mutex_lock l(entry.col_shape_init_mtx_);
          if (entry.col_shape_initialized) {
            if (entry.compilation_status == Status::OK()) {
              TaoCompInfoCollector::Get().UpdateShapeCompileStatus(
                  entry.func_id, entry.shape_hash_sig, STATUS_PASS_ON_COMPILE);
              TaoCompInfoCollector::Get().AddExecutable(
                  entry.executable.get(), entry.func_id, entry.shape_hash_sig);
            } else {
              TaoCompInfoCollector::Get().UpdateShapeCompileStatus(
                  entry.func_id, entry.shape_hash_sig, STATUS_FAIL_ON_COMPILE);
            }
          }
        }
        entry.compiled = true;
        if (stop_) {
          break;
        }
        if (cur_compilation_cycle > 0) {
          std::this_thread::sleep_for(
              std::chrono::milliseconds(cur_compilation_cycle));
        }
      }
      if (cur_compilation_cycle > 0) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(cur_compilation_cycle));
      }
    }
  }

  void NotifyAndWaitToStop() {
    if (!stop_) {
      stop_ = true;
      cancellation_mgr_.NotifyCancellation();
      for (auto thread : threads_) {
        delete thread;
      }
    }
  }

  static int64 GetNumGroup() {
    int64 group_num = 1;
    ::tensorflow::ReadInt64FromEnvVar("TAO_COMPILATION_GROUP_NUM", group_num,
                                      &group_num);
    return group_num;
  }

  static int64 GetCompilationCycle() {
    int64 compilation_cycle_in_ms = 1000;
    ::tensorflow::ReadInt64FromEnvVar("TAO_COMPILATION_CYCLE",
                                      compilation_cycle_in_ms,
                                      &compilation_cycle_in_ms);
    return compilation_cycle_in_ms;
  }

  static int64 GetMaxParallelCompilationThread() {
    int64 max_parallel_compilation_thread_num = 1;
    ::tensorflow::ReadInt64FromEnvVar("TAO_MAX_PARALLEL_COMPILATION_THREAD_NUM",
                                      max_parallel_compilation_thread_num,
                                      &max_parallel_compilation_thread_num);
    return max_parallel_compilation_thread_num;
  }

  static AsyncCompilationMgr& Global() {
    static AsyncCompilationMgr mgr(
        GetTaoBridgeOptions()->cache_capacity, GetCompilationCycle(),
        GetMaxParallelCompilationThread(), GetNumGroup());
    return mgr;
  }

 private:
  using QueueTag = int;
  using ActionQueue = std::queue<std::pair<int, Action>>;
  using QueueMap = std::unordered_map<QueueTag, ActionQueue>;
  // Not Locked
  ActionQueue& GetActionQueue(QueueTag tag = 0) { return queue_map_[tag]; }

 private:
  std::atomic<int> next_entry_slot_{0};
  std::vector<Entry> entries_;

  mutex mutex_;
  QueueMap queue_map_;

  std::atomic<bool> stop_{false};

  std::vector<Thread*> threads_;

  int64 num_groups_;

  int64 compilation_cycle_in_ms_;

  CancellationMgr cancellation_mgr_;

  int64 max_parallel_compilation_thread_num_;

  TF_DISALLOW_COPY_AND_ASSIGN(AsyncCompilationMgr);
};

Status GetDeviceId(OpKernelContext* ctx, int* device_id) {
  TF_RET_CHECK(device_id != nullptr);
  DeviceNameUtils::ParsedName parsed_name;
  bool ok = DeviceNameUtils::ParseFullName(ctx->device()->name(), &parsed_name);
  if (ok) {
    *device_id = parsed_name.id;
  }
  return ok ? Status::OK() : errors::Internal("Could not parse device name.");
}

}  // namespace

Tensor ToCpu(OpKernelContext* ctx, Tensor t, MemoryType mem_type) {
  if (HOST_MEMORY == mem_type || !ctx->op_device_context()) return t;

  AllocatorAttributes alloc_attr;
  auto to_ptr = [](const Tensor& tensor) {
    return const_cast<void*>(
        static_cast<const void*>(tensor.tensor_data().data()));
  };
  auto stream = ctx->op_device_context()->stream();
  Tensor cpu_tensor;
  alloc_attr.set_on_host(true);
  ctx->allocate_temp(t.dtype(), t.shape(), &cpu_tensor, alloc_attr);
  stream->ThenMemcpy(to_ptr(cpu_tensor), se::DeviceMemoryBase(to_ptr(t)),
                     t.TotalBytes());
  stream->BlockHostUntilDone();
  return cpu_tensor;
}

mutex TaoCompilationCache::global_mu_;

TaoCompilationCache::TaoCompilationCache(bool async_compilation)
    : async_compilation_(async_compilation) {
  auto* opts = GetTaoBridgeOptions();
  tao_compiler_path_ = opts->tao_compiler_path;
  tao_cache_dump_path_ = opts->tao_cache_dump_path;
  profile_guide_mode_ = opts->profiling_guided_compilation_mode;
  // Dumper Options
  auto* dumper_opts = GetTaoDumperOptions();
  remove_after_compile_ = dumper_opts->remove_after_compile;
  tao_upload_tool_path_ = dumper_opts->tao_upload_tool_path;
  graphdef_node_size_min_ = dumper_opts->graphdef_node_size_min;
  graphdef_node_size_max_ = dumper_opts->graphdef_node_size_max;

  if (opts->disc_debug_mode) {
    VLOG(0) << "Force remove_after_compile = false due to DISC_DEBUG is on";
    remove_after_compile_ = false;
  }

  LOG(INFO) << "TaoCompilationCache initiate: ";
  LOG(INFO) << "    tao compiler path: " << tao_compiler_path_;
  LOG(INFO) << "      cache dump path: " << tao_cache_dump_path_;
  LOG(INFO) << "     remove tmp files: " << remove_after_compile_;
  LOG(INFO) << "    async compilation: " << async_compilation_;
  LOG(INFO) << "    profiling guide compilation: " << profile_guide_mode_;

#ifdef BLAZE_OPT
  if (async_compilation_ && !tao_cache_dump_path_.empty()) {
#else
  if (!tao_cache_dump_path_.empty()) {
#endif
    Status s = LoadFromFile(tao_cache_dump_path_);
    if (!s.ok()) {
      LOG(ERROR) << "Error when loading compilation: " << s.error_message();
    }
  }
  cache_obj_idx_ = TaoCompInfoCollector::Get().GetOrCreateCacheIdx(this);
}

TaoCompilationCache::~TaoCompilationCache() {
  if (async_compilation_) {
    AsyncCompilationMgr::Global().NotifyAndWaitToStop();
  }

#ifdef BLAZE_OPT
  if (async_compilation_ && !tao_cache_dump_path_.empty()) {
#else
  if (!tao_cache_dump_path_.empty()) {
#endif
    Status s = DumpToFile(tao_cache_dump_path_);
    if (!s.ok()) {
      LOG(ERROR) << "Error when dumping compilation: " << s.error_message();
    }
  }

  if (tao_profile_stat_handle_thread_) {
    stop_ = true;
    delete tao_profile_stat_handle_thread_;
    tao_profile_stat_handle_thread_ = nullptr;
  }
  VLOG(2) << "TaoCompilationCache desctructed";

  TaoCompInfoCollector::Get().DeregisterCache(this);
}

Status TaoCompilationCache::DumpToFile(const std::string& dirname) {
  CompilationCacheResult result;
  VLOG(2) << "Dump compilation cache to: " << dirname;
  {
    mutex_lock lock(compile_cache_mu_);
    Status s = tensorflow::Env::Default()->RecursivelyCreateDir(dirname);
    if (!s.ok() && !errors::IsAlreadyExists(s)) {
      errors::AppendToMessage(&s, "when creating directory ", dirname);
      return s;
    }
    int idx = 0;
    for (auto& pair : cache_) {
      if (!pair.second->compilation_status.ok() || !pair.second->compiled) {
        continue;
      }
      auto entry = result.add_entries();
      entry->set_filename(absl::StrCat("entry_", idx++));
      entry->set_target_device(pair.second->executable->target_device());
      auto& sig_proto = *entry->mutable_sig();
      auto& sig = pair.first;
      sig_proto.set_name(sig.name);
      for (size_t i = 0; i < sig.arg_ranks.size(); ++i) {
        const auto& pair = sig.arg_ranks[i];
        TypeRankPair* pair_proto = sig_proto.add_arg_ranks();
        pair_proto->set_type(static_cast<int>(pair.first));
        pair_proto->set_rank(pair.second);
      }
      for (size_t i = 0; i < sig.arg_types.size(); ++i) {
        const auto& pair = sig.arg_types[i];
        tensorflow::TensorShapeProto shape_proto;
        pair.second.AsProto(&shape_proto);
        TypeShapePair* pair_proto = sig_proto.add_arg_types();
        pair_proto->set_type(static_cast<int>(pair.first));
        shape_proto.AppendToString(pair_proto->mutable_shape());
      }
      for (size_t i = 0; i < sig.arg_values.size(); ++i) {
        tensorflow::TensorProto tensor_proto;
        sig.arg_values[i].AsProtoTensorContent(&tensor_proto);
        tensor_proto.AppendToString(sig_proto.add_arg_values());
      }
      std::string entry_full_path =
          absl::StrCat(dirname, "/", entry->filename());
      pair.second->executable->DumpToFile(entry_full_path);
    }
    VLOG(2) << "total dumped entries: " << idx;
  }
#ifdef BLAZE_OPT
  TF_RETURN_IF_ERROR(WriteTextProto(
#else
  TF_RETURN_IF_ERROR(WriteBinaryProto(
#endif
      tensorflow::Env::Default(), absl::StrCat(dirname, "/tao_cache"), result));
  return Status::OK();
}

Status TaoCompilationCache::LoadFromFile(const std::string& dirname) {
  CompilationCacheResult result;
#ifdef BLAZE_OPT
  if (!ReadTextProto(tensorflow::Env::Default(),
#else
  if (!ReadBinaryProto(tensorflow::Env::Default(),
#endif
                     absl::StrCat(dirname, "/tao_cache"), &result)
           .ok()) {
#ifdef BLAZE_OPT
    return errors::NotFound("Skip loading cache due not load file " + dirname +
                            "/tao_cache failed");
#else
    // Ignore if read file failed.
    LOG(WARNING) << "Skip loading cache due not load file "
                 << absl::StrCat(dirname, "/tao_cache") << " failed";
    return Status::OK();
#endif
  }
  mutex_lock lock(compile_cache_mu_);
  for (int i = 0; i < result.entries_size(); ++i) {
    auto& sig_proto = result.entries(i).sig();
    Signature sig;
    sig.name = sig_proto.name();
    for (int j = 0; j < sig_proto.arg_ranks_size(); ++j) {
      const TypeRankPair& pair_proto = sig_proto.arg_ranks(j);
      sig.arg_ranks.emplace_back(
          static_cast<tensorflow::DataType>(pair_proto.type()),
          pair_proto.rank());
    }
    for (int j = 0; j < sig_proto.arg_types_size(); ++j) {
      const TypeShapePair& pair_proto = sig_proto.arg_types(j);
      tensorflow::TensorShapeProto shape_proto;
      shape_proto.ParseFromString(pair_proto.shape());
      sig.arg_types.emplace_back(
          static_cast<tensorflow::DataType>(pair_proto.type()),
          tensorflow::TensorShape(shape_proto));
    }
    for (int j = 0; j < sig_proto.arg_values_size(); ++j) {
      tensorflow::TensorProto tensor_proto;
      tensor_proto.ParseFromString(sig_proto.arg_values(j));
      sig.arg_values.emplace_back();
      TF_RET_CHECK(sig.arg_values.back().FromProto(tensor_proto));
    }
    std::unique_ptr<Entry>& e = cache_[sig];
    TF_RET_CHECK(e == nullptr);
    e.reset(new Entry);
    auto entry = e.get();
    std::string entry_full_path =
        absl::StrCat(dirname, "/", result.entries(i).filename());
    std::string target_device = result.entries(i).target_device();
    entry->executable = ExecutableFactory::Global().NewExecutable(
        target_device, entry_full_path);
    if (!entry->executable) {
      return errors::Internal("Executable Not registered for DEVICE " +
                              target_device);
    }
    entry->compilation_status = entry->executable->Init();
    entry->compiled = true;
  }
  LOG(INFO) << "Load compilation cache success from " << dirname << " with "
            << result.entries_size() << " entries.";
  return Status::OK();
}

// Compute a string signature which encodes the shapes of the
// arguments in the supplied list.
string TaoCompilationCache::SignatureDebugString(const Signature& sig) {
  string result = sig.name;

  for (const auto& a : sig.arg_ranks) {
    absl::StrAppend(&result, ",", DataTypeString(a.first),
                    ", rank: ", a.second);
  }

  for (const auto& a : sig.arg_types) {
    absl::StrAppend(&result, ", shape: ", DataTypeString(a.first),
                    a.second.DebugString());
  }

  for (const auto& v : sig.host_args) {
    absl::StrAppend(&result, ", host_arg: ", v);
  }

  for (const auto& v : sig.arg_values) {
    absl::StrAppend(&result, ", value: ", v.DebugString());
  }

  return result;
}

bool TaoCompilationCache::Signature::operator==(const Signature& other) const {
  if (name != other.name) return false;
  if (arg_ranks != other.arg_ranks) return false;
  if (arg_types != other.arg_types) return false;
  if (host_args != other.host_args) return false;

  if (arg_values.size() != other.arg_values.size()) return false;
  for (size_t i = 0; i < arg_values.size(); ++i) {
    if (arg_values[i].tensor_data() != other.arg_values[i].tensor_data()) {
      return false;
    }
  }

  return true;
}

uint64 TaoCompilationCache::Signature::Hash::operator()(
    const TaoCompilationCache::Signature& signature) const {
  uint64 h = std::hash<string>()(signature.name);
  for (const auto& arg : signature.arg_ranks) {
    h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.first)));
    h = Hash64Combine(h, std::hash<int>()(arg.second));
  }
  for (const auto& arg : signature.arg_types) {
    h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.first)));
    h = Hash64Combine(h, std::hash<int>()(arg.second.dims()));
    for (int dim : arg.second.dim_sizes()) {
      h = Hash64Combine(h, std::hash<int>()(dim));
    }
  }
  for (const auto& arg : signature.arg_values) {
    h = Hash64Combine(
        h, Hash64(arg.tensor_data().data(), arg.tensor_data().size()));
  }
  for (const int& host_arg_idx : signature.host_args) {
    h = Hash64Combine(h, std::hash<int>()(host_arg_idx));
  }
  return h;
}

Status TaoCompilationCache::BuildSignature(
    const NameAttrList& function, const std::map<int, Tensor>& constant_args,
    const std::set<int>& fixed_shape_args, const std::set<int>& host_args,
    const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
    Signature* signature, bool is_mlir) {
  signature->name = Canonicalize(function.name(), AttrSlice(&function.attr()));
  signature->arg_values.reserve(constant_args.size());
  if (is_mlir) {
    signature->arg_types.reserve(fixed_shape_args.size());
    signature->arg_ranks.reserve(ctx->num_inputs() - constant_args.size() -
                                 fixed_shape_args.size());
  } else {
    signature->arg_types.reserve(ctx->num_inputs() - constant_args.size());
  }

  for (int i = 0; i < ctx->num_inputs(); ++i) {
    if (constant_args.count(i) > 0) {
      // Use the values of compile time constants in the signature.
      signature->arg_values.push_back(constant_args.at(i));
    } else if (variable_args.count(i) > 0) {
      const OptionalTensor& variable = variable_args.at(i);
      if (variable.present) {
        signature->arg_types.emplace_back(variable.value.dtype(),
                                          variable.value.shape());
      } else {
        signature->arg_types.emplace_back(tensorflow::DT_INVALID,
                                          TensorShape());
      }
    } else if ((!is_mlir) || (fixed_shape_args.count(i))) {
      signature->arg_types.emplace_back(ctx->input_dtype(i),
                                        ctx->input(i).shape());
    } else {
      // if is_mlir
      signature->arg_ranks.emplace_back(ctx->input_dtype(i),
                                        ctx->input(i).dims());
    }
  }

  signature->host_args.insert(signature->host_args.end(), host_args.begin(),
                              host_args.end());
  return Status::OK();
}

// Currently, we use a TF function to represent a compilable subgraph. TF
// function can be arbitrarily nested both using explict way (e.g. a call node)
// and implicit way (e.g. functional control flow ops).
//
// In order to compile a function, we need to dump all related functions (the
// closure of the entry function). Ideally we should only dump the functions
// that are called directly or indirectly by the entry function. Thus we prune
// the function library before we dump the input.
Status SerializeFunctionLibrary(TaoCompilerInput* compiler_input,
                                const NameAttrList& function,
                                FunctionLibraryRuntime* flib) {
  auto flib_def = flib->GetFunctionLibraryDefinition();
  auto& options = *(compiler_input->mutable_options());
  auto func = flib_def->Find(function.name());
  TF_RET_CHECK(func != nullptr);

  auto reachable_flib = util::ReachableDefinitions(*flib_def, *func);
  TF_RET_CHECK(reachable_flib != nullptr);

  reachable_flib->ToProto().AppendToString(options.mutable_flib_def());

  return Status::OK();
}

Status PrepareCompilerInput(
    const NameAttrList& function, const std::map<int, Tensor>& constant_args,
    const std::set<int>& fixed_shape_args, const std::set<int>& host_args,
    const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
    TaoCompilerInput* compiler_input, bool is_mlir = false) {
  auto& options = *(compiler_input->mutable_options());
  auto flib_def = ctx->function_library();

  std::chrono::time_point<std::chrono::steady_clock> start;
  if (VLOG_IS_ON(1)) {
    start = std::chrono::steady_clock::now();
  }
  TF_RET_CHECK(SerializeFunctionLibrary(compiler_input, function, flib_def));
  if (VLOG_IS_ON(1)) {
    std::chrono::duration<double> elapsed_sec =
        std::chrono::steady_clock::now() - start;
    VLOG(1) << "Dump FlibDef takes " << std::fixed << elapsed_sec.count()
            << " s";
  }

  function.AppendToString(compiler_input->mutable_function());
  if (VLOG_IS_ON(1)) {
    auto* func_def =
        flib_def->GetFunctionLibraryDefinition()->Find(function.name());
    CHECK(func_def != nullptr)
        << "Function not found in function library: " << function.name();
    VLOG(1) << "cluster size of function " << function.name() << ": "
            << func_def->node_def_size();
  }

  bool dump_arg_value = GetTaoBridgeOptions()->disc_debug_mode;
  std::vector<std::string> value_proto_filenames;
  for (int64 input_num = 0; input_num < ctx->num_inputs(); ++input_num) {
    auto arg = compiler_input->add_args();
    if (constant_args.count(input_num) > 0) {
      // Handles compile-time constants.
      const Tensor& input = constant_args.at(input_num);
      CHECK(input.dtype() != tensorflow::DT_RESOURCE);
      arg->set_kind_v2(ArgumentKind::kConstant);
      arg->set_type(static_cast<int>(input.dtype()));
      tensorflow::TensorShapeProto shape_proto;
      input.shape().AsProto(&shape_proto);
      shape_proto.AppendToString(arg->mutable_shape());
      tensorflow::TensorProto tensor_proto;
      input.AsProtoTensorContent(&tensor_proto);
      tensor_proto.AppendToString(arg->mutable_constant_value());
    } else if (variable_args.count(input_num) == 0) {
      // Handles the non-constant arguments.
      const Tensor& input = ctx->input(input_num);
      CHECK(input.dtype() != tensorflow::DT_RESOURCE);
      if (is_mlir) {
        if (fixed_shape_args.count(input_num) > 0) {
          // only used for Mlir dynamic shape compiler
          arg->set_kind_v2(ArgumentKind::kFixedShaped);
        } else if (host_args.count(input_num)) {
          arg->set_kind_v2(ArgumentKind::kHostArgs);
        } else {
          arg->set_kind_v2(ArgumentKind::kParameter);
        }
      } else {
        if (input.NumElements() > 0) {
          arg->set_kind_v2(ArgumentKind::kParameter);
        } else {
          VLOG(2) << "set empty parameter #" << input_num << " to constant";
          arg->set_kind_v2(ArgumentKind::kConstant);
          tensorflow::TensorProto tensor_proto;
          input.AsProtoTensorContent(&tensor_proto);
          tensor_proto.AppendToString(arg->mutable_constant_value());
        }
      }
      arg->set_type(static_cast<int>(input.dtype()));
      tensorflow::TensorShapeProto shape_proto;
      input.shape().AsProto(&shape_proto);
      shape_proto.AppendToString(arg->mutable_shape());
    } else {
      // Handles resource variables.
      const Tensor& input = ctx->input(input_num);
      CHECK(input.dtype() == tensorflow::DT_RESOURCE);
      const OptionalTensor& variable = variable_args.at(input_num);
      *arg->mutable_name() = variable.name;
      ;
      arg->set_kind_v2(ArgumentKind::kResource);
      arg->set_resource_kind_v2(ArgumentResourceKind::kVariable);
      if (variable.present) {
        const Tensor& value = variable.value;
        arg->set_type(static_cast<int>(value.dtype()));
        tensorflow::TensorShapeProto shape_proto;
        value.shape().AsProto(&shape_proto);
        shape_proto.AppendToString(arg->mutable_shape());
        arg->set_initialized(true);
      } else {
        // The values of uninitialized variables are not passed as inputs, since
        // they are meaningless. However, it is legal to assign to a resource
        // variable for the first time inside the XLA computation, so we do
        // permit uninitialized variables.
        arg->set_initialized(false);
        arg->set_type(static_cast<int>(tensorflow::DT_INVALID));
        tensorflow::TensorShapeProto shape_proto;
        tensorflow::TensorShape().AsProto(&shape_proto);
        shape_proto.AppendToString(arg->mutable_shape());
      }
    }
    if (dump_arg_value) {
      Tensor cpu_tensor =
          ToCpu(ctx, ctx->input(input_num), ctx->input_memory_type(input_num));
      tensorflow::TensorProto tensor_proto;
      cpu_tensor.AsProtoTensorContent(&tensor_proto);
      auto env = tensorflow::Env::Default();
      std::string path;
      if (!env->LocalTempFilename(&path)) {
        return errors::Internal("couldn't get temp file name");
      }
      TF_RETURN_IF_ERROR(
          WriteBinaryProto(tensorflow::Env::Default(), path, tensor_proto));
      auto it = path.find_last_of('/');
      std::string filename = (it == std::string::npos)
                                 ? path
                                 : path.substr(path.find_last_of('/') + 1);
      arg->set_value_proto_file(filename);
      value_proto_filenames.push_back(path);
      VLOG(0) << "arg #" << input_num << " proto filename: " << filename;
    }
  }
  if (dump_arg_value) {
    auto env = tensorflow::Env::Default();
    std::string unique_name;
    if (!env->LocalTempFilename(&unique_name)) {
      return errors::Internal("couldn't get temp file name");
    }
    std::string tar_file_path = unique_name + ".tar";
    std::string cmd = "tar -C `dirname " + tar_file_path + "` -cf `basename " +
                      tar_file_path + "` ";
    for (auto& name : value_proto_filenames) {
      cmd += "`basename " + name + "` ";
    }
    cmd += "; mv `basename " + tar_file_path + "` " + tar_file_path;
    VLOG(0) << "tar the cluster input tensors with protobuf format to: "
            << tar_file_path << "\n\tcmd: " << cmd;
    std::system(cmd.c_str());
  }

  return Status::OK();
}
namespace {

Status CompileFunctionImpl(
    const std::string& func_name, const std::string& tao_compiler_path,
    const std::string& output_file_name, TaoCompilerInput& input,
    bool remove_after_compile,
    AsyncCompilationMgr::CancellationMgr* cancellation_mgr) {
  auto env = tensorflow::Env::Default();
  std::string input_file_name;
  if (!env->LocalTempFilename(&input_file_name)) {
    return errors::Internal("couldn't get temp tao_compiler_input file name");
  }
  input_file_name += ".input";
  auto input_cleaner =
      tensorflow::gtl::MakeCleanup([&input_file_name, remove_after_compile] {
        if (remove_after_compile) {
          tensorflow::Env::Default()->DeleteFile(input_file_name);
        } else {
          VLOG(0) << "tao_compiler_input: " << input_file_name;
        }
      });
  TF_RETURN_IF_ERROR(
      WriteBinaryProto(tensorflow::Env::Default(), input_file_name, input));

  if (VLOG_IS_ON(2)) {
    std::string dbg_input_file_name = input_file_name + ".input_txt";
    VLOG(2) << "Writing TaoCompilerInput to " << dbg_input_file_name;
    TF_RETURN_IF_ERROR(
        WriteTextProto(tensorflow::Env::Default(), dbg_input_file_name, input));
  }

  if (VLOG_IS_ON(0)) {
    std::string dbg_input_file_name = input_file_name + ".input_pb";
    VLOG(0) << "Writing TaoCompilerInput Binary to " << dbg_input_file_name;
    TF_RETURN_IF_ERROR(
        WriteBinaryProto(tensorflow::Env::Default(), dbg_input_file_name, input));
  }


  auto start = std::chrono::steady_clock::now();
  tensorflow::tao::SubProcess tao_compiler;
  VLOG(2) << "compiling function " << func_name << ", input file is "
          << input_file_name << ", output file is " << output_file_name;
  std::vector<string> tao_compiler_args = {tao_compiler_path, input_file_name,
                                           output_file_name};
  tao_compiler.SetProgram(tao_compiler_path, tao_compiler_args);
  tao_compiler.SetChannelAction(tensorflow::tao::CHAN_STDOUT,
                                tensorflow::tao::ACTION_PIPE);
  tao_compiler.SetChannelAction(tensorflow::tao::CHAN_STDERR,
                                tensorflow::tao::ACTION_PIPE);
  {
    // We need this trick because OpenBlas has a known bug
    // which leading to deadlock when using fork in multi-thread environment.
    // https://github.com/xianyi/OpenBLAS/issues/2270
    std::unique_ptr<mutex_lock> l;
    if (cancellation_mgr) {
      l = cancellation_mgr->Lock();
    }

    VLOG(2) << "start tao_compiler";
    if (!tao_compiler.Start()) {
      return errors::Internal("Failed to launch tao_comipler: " +
                              tao_compiler_path);
    }
  }
  string stdout_output;
  string stderr_output;

  AsyncCompilationMgr::CancellationMgr::Handle handle =
      AsyncCompilationMgr::CancellationMgr::kInvalidHandle;
  if (cancellation_mgr) {
    handle = cancellation_mgr->RegisterCancellationAction([&tao_compiler]() {
      tao_compiler.Kill(9);
      VLOG(2) << "try to kill compilation process.";
    });
    if (handle == AsyncCompilationMgr::CancellationMgr::kInvalidHandle) {
      // early stop triggered.
      VLOG(2) << "hit early stop...";
      return Status::OK();
    }
  }
  int exit_status = tao_compiler.Communicate(
      /*stdin_input=*/nullptr, &stdout_output, &stderr_output);
  if (cancellation_mgr) {
    cancellation_mgr->RemoveCancellationAction(handle);
  }
  std::chrono::duration<double> elapsed_sec =
      std::chrono::steady_clock::now() - start;

  std::stringstream ss;
  ss << "tao_compiler exits with errcode " << exit_status << " in "
     << std::fixed << elapsed_sec.count() << " seconds to compile " << func_name
     << ":\n"
     << "============= stdout ===============\n"
     << stdout_output << "\n\n\n"
     << "============= stderr ===============\n"
     << stderr_output << "\n\n"
     << "====================================";

  // output to diting or sls sdk if TAO_UPLOAD_TOOL_PATH is set
  // or output to VLOG(2)
  if (GetTaoBridgeOptions()->verbose_compilation_err_log ||
      GetTaoBridgeOptions()->verbose_compilation_log) {
    if (exit_status != 0 || GetTaoBridgeOptions()->verbose_compilation_log) {
      VLOG(0) << ss.str();
    } else {
      VLOG(1) << ss.str();
    }
  }

  if (exit_status == 0) {
    return Status::OK();
  } else {
    // compile failed
    bool exited{false};
    bool signaled{false};
    bool coredump{false};
    int decoded_exit_code = -1;
    int decoded_signal = -1;
    if ((exit_status != -1) && (exit_status != 1)) {
      if (WIFEXITED(exit_status)) {
        exited = true;
        decoded_exit_code = WEXITSTATUS(exit_status);
      } else if (WIFSIGNALED(exit_status)) {
        signaled = true;
        coredump = WCOREDUMP(exit_status);
        decoded_signal = WTERMSIG(exit_status);
      }
    }

    if (!GetTaoDumperOptions()->tao_upload_tool_path.empty()) {
      // upload input file and logs
      TaoCompInfoCollector::Get().AddCompileFailedCase(
          ss.str(), input_file_name, exited, signaled, coredump,
          exited ? decoded_exit_code : (signaled ? decoded_signal : -1));
    }

    switch (decoded_exit_code) {
      case 2:
        return errors::Internal(
            "tao_compiler failed. CUDA_ERROR_OUT_OF_MEMORY detected!");
      case 3:
        return errors::Internal(
            "tao_compiler failed. DEADLINE_EXCEEDED detected!");
      default:
        break;
    }
    return errors::Internal("tao_compiler failed.");
  }
}

}  // namespace

Status CompileFunction(const std::string& func_name,
                       const std::string& tao_compiler_path,
                       const std::string& output_file_name,
                       TaoCompilerInput& input, bool remove_after_compile) {
  return CompileFunctionImpl(func_name, tao_compiler_path, output_file_name,
                             input, remove_after_compile, nullptr);
}

uint64 TaoCompilationCache::GetSignatureHash(
    const NameAttrList& function, const std::map<int, Tensor>& constant_args,
    const std::set<int>& fixed_shape_args, const std::set<int>& host_args,
    const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
    bool is_mlir) {
  Signature signature;
  BuildSignature(function, constant_args, fixed_shape_args, host_args,
                 variable_args, ctx, &signature, is_mlir);
  Signature::Hash hash;
  auto hash_sig = hash(signature);
  return hash_sig;
}

Status TaoCompilationCache::Compile(
    std::unique_ptr<TaoCompilerInput> input, const NameAttrList& function,
    const std::map<int, Tensor>& constant_args,
    const std::set<int>& fixed_shape_args, const std::set<int>& host_args,
    const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
    Executable** executable, TaoProfileStat** stat, bool is_mlir,
    TaoCompileFuncCallInfo* call_info) {
  if (stat) {
    *stat = &tao_profile_stat_;
  }
  if (async_compilation_) {
    return CompileImplAsync(std::move(input), function, constant_args,
                            fixed_shape_args, host_args, variable_args, ctx,
                            executable, is_mlir, call_info);
  } else {
#ifdef BLAZE_OPT
    mutex_lock global_lock(global_mu_);
    Status status;
    if (cache_.empty() && !tao_cache_dump_path_.empty()) {
      status = LoadFromFile(tao_cache_dump_path_);
    }
    if (!status.ok()) {
      LOG(WARNING) << status.error_message();
      cache_.clear();
    }
    cache_updated_ = false;
    status = CompileImpl(std::move(input), function, constant_args,
                         fixed_shape_args, host_args, variable_args, ctx,
                         executable, is_mlir, call_info);
    if (status.ok() && cache_updated_ && !tao_cache_dump_path_.empty()) {
      auto dump_status = DumpToFile(tao_cache_dump_path_);
      if (!dump_status.ok()) {
        LOG(ERROR) << dump_status.error_message();
      }
    }
    return status;
#else
    return CompileImpl(std::move(input), function, constant_args,
                       fixed_shape_args, host_args, variable_args, ctx,
                       executable, is_mlir, call_info);
#endif
  }
}

void TaoCompilationCache::HandleProfileStatLoop() {
  TaoProfileStat* stat = &tao_profile_stat_;
  int64 new_total_time_elapsed_in_us = 0;
  int64 old_total_time_elapsed_in_us = 0;
  int64 new_total_time_saved_in_us = 0;
  int64 old_total_time_saved_in_us = 0;
  double speed_up_all = 0.0;
  double speed_up_recent = 0.0;

  int64 tao_profile_state_print_cycle_in_sec = 10 * 60;  // 10 minutes
  ::tensorflow::ReadInt64FromEnvVar("TAO_PROFILE_STAT_PRINT_CYCLE",
                                    tao_profile_state_print_cycle_in_sec,
                                    &tao_profile_state_print_cycle_in_sec);
  int64 sleep_cycle_in_sec = (tao_profile_state_print_cycle_in_sec < 10)
                                 ? tao_profile_state_print_cycle_in_sec
                                 : 10;

  auto begin = std::chrono::steady_clock::now();

  auto do_print = [&, this](bool last_call) {
    old_total_time_saved_in_us = new_total_time_saved_in_us;
    new_total_time_saved_in_us = stat->total_time_saved_in_us;
    auto end = std::chrono::steady_clock::now();
    old_total_time_elapsed_in_us = new_total_time_elapsed_in_us;
    new_total_time_elapsed_in_us =
        (std::chrono::duration_cast<std::chrono::microseconds>(end - begin))
            .count();
    VLOG(2) << "\n\tnew total = " << new_total_time_elapsed_in_us

            << "\n\told total = " << old_total_time_elapsed_in_us
            << "\n\tnew saved = " << new_total_time_saved_in_us
            << "\n\told saved = " << old_total_time_saved_in_us;
    speed_up_all = (1.0 / new_total_time_elapsed_in_us) *
                   (new_total_time_saved_in_us + new_total_time_elapsed_in_us);
    speed_up_recent =
        (1.0 / (new_total_time_elapsed_in_us - old_total_time_elapsed_in_us)) *
        (new_total_time_saved_in_us - old_total_time_saved_in_us +
         new_total_time_elapsed_in_us - old_total_time_elapsed_in_us);

    int loglevel = 2;
    if (speed_up_recent > 1.1) {
      loglevel = 0;
    }
    VLOG(loglevel) << "TAO/TF estimated speedup (All): " << speed_up_all
                   << "X, estimated speedup (Recent): " << speed_up_recent
                   << "X.";

    if (!tao_upload_tool_path_.empty()) {
      TaoCompInfoCollector::Get().UpdatePerfStats(
          this, new_total_time_elapsed_in_us, new_total_time_saved_in_us,
          tao_profile_state_print_cycle_in_sec, speed_up_all, speed_up_recent,
          last_call);
    }
  };

  int64 total_sleep_time_in_sec = 0;
  while (!stop_) {
    if (total_sleep_time_in_sec < tao_profile_state_print_cycle_in_sec) {
      std::this_thread::sleep_for(std::chrono::seconds(sleep_cycle_in_sec));
      total_sleep_time_in_sec += sleep_cycle_in_sec;
      continue;
    }
    total_sleep_time_in_sec = 0;

    do_print(false);
  }

  do_print(true);
}

void TaoCompilationCache::StartProfileStatHandleThread() {
  tao_profile_stat_handle_thread_ =
      Env::Default()->StartThread(ThreadOptions(), "TAO-PROFILE-STAT-HANDLE",
                                  [this]() { HandleProfileStatLoop(); });
}

Status TaoCompilationCache::CompileImpl(
    std::unique_ptr<TaoCompilerInput> input_ptr, const NameAttrList& function,
    const std::map<int, Tensor>& constant_args,
    const std::set<int>& fixed_shape_args, const std::set<int>& host_args,
    const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
    Executable** executable, bool is_mlir, TaoCompileFuncCallInfo* call_info) {
  CHECK(constant_args.size() + variable_args.size() <=
        static_cast<size_t>(ctx->num_inputs()));

  auto& input = *input_ptr;
  Signature signature;
  TF_RETURN_IF_ERROR(BuildSignature(function, constant_args, fixed_shape_args,
                                    host_args, variable_args, ctx, &signature,
                                    is_mlir));

  VLOG(2) << "Signature: " << SignatureDebugString(signature);

  // The outer lock protects the existence of the cache entry. It does not
  // protect the contents of the cache entry.
  Entry* entry{nullptr};
  bool new_entry{false};
  {
    mutex_lock lock(compile_cache_mu_);
    if (!tao_profile_stat_handle_thread_) {
      StartProfileStatHandleThread();
    }

    // Find or create a cache entry.
    std::unique_ptr<Entry>& e = cache_[signature];
    if (!e) {
      new_entry = true;
      e.reset(new Entry);
    }
    entry = e.get();
  }

  Signature::Hash hash;
  auto hash_sig = hash(signature);
  input.mutable_options()->set_func_hash(hash_sig);

  auto dump_trigger = tensorflow::gtl::MakeCleanup([&] {
    if (GetTaoDumperOptions()->dump_level == 0) return;
    bool wish_dump = false;
    bool compile_ok =
        entry->compiled && entry->compilation_status == Status::OK();
    DumpInfo(ctx, function, constant_args, hash_sig, compile_ok, call_info);
    if (new_entry) {
      // update compilation status
      auto& collector = TaoCompInfoCollector::Get();
      auto func_id = GenFunctionID(function);
      if (compile_ok) {
        collector.UpdateShapeCompileStatus(func_id, hash_sig,
                                           STATUS_PASS_ON_COMPILE);
        TaoCompInfoCollector::Get().AddExecutable(entry->executable.get(),
                                                  func_id, hash_sig);
      } else {
        collector.UpdateShapeCompileStatus(func_id, hash_sig,
                                           STATUS_FAIL_ON_COMPILE);
      }
    }
  });

  // Acquire the cache entry lock and compile, if necessary.
  // TODO(phawkins): this locking will need to be restructured when we implement
  // cache eviction.
  // Note(fanpf): reverted changes by wenyi in commit 6396e67c
  mutex_lock entry_lock(entry->mu);
  if (!entry->compiled) {
    VLOG(2) << "Compilation cache miss for signature: "
            << SignatureDebugString(signature);

    entry->compiled = true;
    // Do the actual JIT compilation without holding the lock (it can take
    // a long time.)
    std::string output_file_name;
    std::string blade_mode;
    CHECK_OK(ReadStringFromEnvVar("BLADE_CPU_MODE", "", &blade_mode));
    auto output_cleaner =
        tensorflow::gtl::MakeCleanup([&output_file_name, &blade_mode, this] {
          // TODO: figure out why blade cpu does not need a cleaner.
          if (blade_mode == "compile" || blade_mode == "serve") {
            return;
          }
          if (remove_after_compile_) {
            tensorflow::Env::Default()->DeleteFile(output_file_name);
          } else {
            VLOG(0) << "tao_compiler_result file: " << output_file_name;
          }
        });

    if (blade_mode == "compile" || blade_mode == "serve") {
      std::string blade_files;
      CHECK_OK(ReadStringFromEnvVar("BLADE_CPU_FILES", "/tmp", &blade_files));
      if (blade_files == "/tmp") {
        VLOG(2) << "Using /tmp as BLADE_CPU_FILES for BladeCpu "
                   "compile/serve mode!";
      }
      output_file_name =
          blade_files + "/out_proto_tvmcluster_" + std::to_string(hash_sig);
    }

    if (blade_mode != "serve") {
      entry->compilation_status =
          PrepareCompilerInput(function, constant_args, fixed_shape_args,
                               host_args, variable_args, ctx, &input, is_mlir);
      TF_RETURN_IF_ERROR(entry->compilation_status);

      if (blade_mode != "compile") {
        if (!tensorflow::Env::Default()->LocalTempFilename(&output_file_name)) {
          return errors::Internal(
              "couldn't get temp tao_compiler_result file name");
        }
        output_file_name += ".output";
      }
      entry->compilation_status = CompileFunctionImpl(
          function.name(), tao_compiler_path_, output_file_name, input,
          remove_after_compile_, nullptr);
      TF_RETURN_IF_ERROR(entry->compilation_status);
#ifdef BLAZE_OPT
      cache_updated_ = true;
#endif
    }

    if (!std::ifstream(output_file_name).good()) {
      return errors::Internal("couldn't get compiled output proto file name" +
                              output_file_name);
    }
    CHECK_EQ(entry->executable.get(), nullptr);
    entry->executable = ExecutableFactory::Global().NewExecutable(
        input.options().device_type(), output_file_name);
    if (!entry->executable) {
      return errors::Internal("Executable Not registered for DEVICE " +
                              input.options().device_type());
    }
    entry->compilation_status = entry->executable->Init();
  }

  if (entry->compilation_status == Status::OK()) {
    *executable = entry->executable.get();
  }
  return entry->compilation_status;
}

Status TaoCompilationCache::CompileImplAsync(
    std::unique_ptr<TaoCompilerInput> input_ptr, const NameAttrList& function,
    const std::map<int, Tensor>& constant_args,
    const std::set<int>& fixed_shape_args, const std::set<int>& host_args,
    const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
    Executable** executable, bool is_mlir, TaoCompileFuncCallInfo* call_info) {
  VLOG(2) << "TaoCompilationCache::CompileImplAsync is called...";
  CHECK(constant_args.size() + variable_args.size() <=
        static_cast<size_t>(ctx->num_inputs()));

  int device_id = 0;
  TF_RET_CHECK(GetDeviceId(ctx, &device_id));

  auto& input = *input_ptr;
  Signature signature;
  TF_RETURN_IF_ERROR(BuildSignature(function, constant_args, fixed_shape_args,
                                    host_args, variable_args, ctx, &signature,
                                    is_mlir));
  Signature::Hash hash;
  auto hash_sig = hash(signature);
  VLOG(2) << "Signature: " << SignatureDebugString(signature);

  // The outer lock protects the existence of the cache entry. It does not
  // protect the contents of the cache entry.
  Entry* entry{nullptr};
  bool new_entry = false;
  bool compile_ok{false};
  int compile_slot{-1};

  auto dump_trigger = tensorflow::gtl::MakeCleanup([&] {
    if (GetTaoDumperOptions()->dump_level == 0) return;
    DumpInfo(ctx, function, constant_args, hash_sig, compile_ok, call_info);
    if (new_entry) {
      auto& collector = TaoCompInfoCollector::Get();
      if (compile_slot >= 0) {
        auto& centry = AsyncCompilationMgr::Global().GetEntry(compile_slot);
        mutex_lock l(centry.col_shape_init_mtx_);
        auto& func_id = centry.func_id;
        if (centry.compiled) {
          if (centry.compilation_status == Status::OK()) {
            // compile passed
            collector.UpdateShapeCompileStatus(func_id, hash_sig,
                                               STATUS_PASS_ON_COMPILE);
          } else {
            // compile failed
            collector.UpdateShapeCompileStatus(func_id, hash_sig,
                                               STATUS_FAIL_ON_COMPILE);
          }
        } else {
          // in queue or compiling
          if (centry.col_shape_compile_started) {
            collector.UpdateShapeCompileStatus(func_id, hash_sig,
                                               STATUS_COMPILE_STARTED);
          } else {
            collector.UpdateShapeCompileStatus(func_id, hash_sig,
                                               STATUS_ENQUEUED);
          }
        }
        centry.col_shape_initialized = true;
      } else {
        // enque failed
        auto func_id = GenFunctionID(function);
        collector.UpdateShapeCompileStatus(func_id, hash_sig,
                                           STATUS_FAIL_ON_CACHE_FULL);
      }
    }
  });

  {
    mutex_lock lock(compile_cache_mu_);
    if (!tao_profile_stat_handle_thread_) {
      StartProfileStatHandleThread();
    }
    // Find or create a cache entry.
    auto it = cache_.find(signature);
    if (it == cache_.end()) {
      new_entry = true;
      compile_slot = AsyncCompilationMgr::Global().AllocateEntry();
      if (compile_slot >= 0) {
        std::unique_ptr<Entry>& e = cache_[signature];
        e.reset(new Entry);
        e->compilation_slot = compile_slot;
        entry = e.get();
      } else {
        return errors::Internal("cache full");
      }
    } else {
      entry = it->second.get();
      compile_slot = entry->compilation_slot;
    }
  }

  // Acquire the cache entry lock and compile, if necessary.
  // TODO(phawkins): this locking will need to be restructured when we implement
  // cache eviction.
  mutex_lock entry_lock(entry->mu);
  if (!entry->compiled) {
    if (!entry->compilation_slot_initialized) {
      VLOG(2) << "Compilation cache miss for signature: "
              << SignatureDebugString(signature);
      entry->compilation_slot_initialized = true;

      if (entry->compilation_slot >= 0) {
        // TODO(kevin.zwy): NEED SOME REFINE, this function call is still
        // expensive. Try to move some work to the background thread in order to
        // reduce overhead in the critical path.
        entry->compilation_status = PrepareCompilerInput(
            function, constant_args, fixed_shape_args, host_args, variable_args,
            ctx, &input, is_mlir);
        TF_RETURN_IF_ERROR(entry->compilation_status);

        auto raw_input_ptr = input_ptr.release();
        string func_name = function.name();
        auto action =
            [this, raw_input_ptr, func_name](
                std::unique_ptr<Executable>* executable,
                AsyncCompilationMgr::CancellationMgr* cancellation_mgr)
            -> Status {
          std::unique_ptr<TaoCompilerInput> input_ptr(raw_input_ptr);
          std::string output_file_name;
          if (!tensorflow::Env::Default()->LocalTempFilename(
                  &output_file_name)) {
            return errors::Internal(
                "couldn't get temp tao_compiler_result file name");
          }
          output_file_name += ".output";
          auto output_cleaner =
              tensorflow::gtl::MakeCleanup([&output_file_name, this] {
                if (remove_after_compile_) {
                  tensorflow::Env::Default()->DeleteFile(output_file_name);
                } else {
                  VLOG(0) << "tao_compiler_result file: "
                          << remove_after_compile_;
                }
              });

          auto status = CompileFunctionImpl(
              func_name, tao_compiler_path_, output_file_name, *input_ptr,
              remove_after_compile_, cancellation_mgr);
          CHECK_EQ(executable->get(), nullptr);
          *executable = ExecutableFactory::Global().NewExecutable(
              input_ptr->options().device_type(), output_file_name);
          if (executable->get() == nullptr) {
            return errors::Internal("Executable Not registered for DEVICE " +
                                    input_ptr->options().device_type());
          }
          status = executable->get()->Init();
          return status;
        };

        auto& centry =
            AsyncCompilationMgr::Global().GetEntry(entry->compilation_slot);
        centry.func_id = GenFunctionID(function);
        centry.shape_hash_sig = hash_sig;

        AsyncCompilationMgr::Global().EnqueueAction(entry->compilation_slot,
                                                    action, device_id);
      }
    }
    if (entry->compilation_slot >= 0) {
      auto& async_compilation_entry =
          AsyncCompilationMgr::Global().GetEntry(entry->compilation_slot);
      if (async_compilation_entry.compiled) {
        entry->compiled = async_compilation_entry.compiled;
        entry->compilation_status = async_compilation_entry.compilation_status;
        entry->executable = std::move(async_compilation_entry.executable);
      }
    }
  }

  compile_ok =
      entry && entry->compiled && entry->compilation_status == Status::OK();
  if (compile_ok) {
    *executable = entry->executable.get();
  }
  return entry->compilation_status;
}

string TaoCompilationCache::DebugString() {
  return "TAO JIT compilation cache";
}

string TaoCompilationCache::DebugString() const {
  return "TAO JIT compilation cache";
}

std::string TaoCompilationCache::GenFunctionID(const NameAttrList& function,
                                               std::string* attrstr) {
  string function_attr;
  if (!attrstr) {
    attrstr = &function_attr;
  }
  *attrstr = Canonicalize("", AttrSlice(&function.attr()));
  std::stringstream ss;
  ss << cache_obj_idx_ << "_" << function.name() << "_"
     << TaoCompInfoCollector::HashValueToStr(std::hash<string>()(*attrstr));
  return ss.str();
}

std::string TaoCompilationCache::DumpGraphToFile(
    OpKernelContext* ctx, const NameAttrList& function,
    const std::string& function_id) {
  FunctionLibraryRuntime* lib = ctx->function_library();
  if (lib != nullptr) {
    FunctionLibraryRuntime::InstantiateOptions opts;
    FunctionLibraryRuntime::Handle handle;
    auto l_ctx = lib->Instantiate(function.name(), AttrSlice(&function.attr()),
                                  opts, &handle);
    const FunctionBody* fbody = lib->GetFunctionBody(handle);
    if (fbody != nullptr) {
      auto graph = tensorflow::MakeUnique<Graph>(fbody->graph->flib_def());
      FunctionLibraryDefinition global_flib(OpRegistry::Global(), {});
      TF_CHECK_OK(graph.get()->AddFunctionLibrary(global_flib.ToProto()));
      CopyGraph(*fbody->graph, graph.get());
      GraphDef graph_def;
      graph->ToGraphDef(&graph_def);
      if (graph_def.node_size() <= graphdef_node_size_min_) {
        VLOG(2) << "Skip dump for very small sub-graph with #nodes of "
                << graph_def.node_size();
        return "";
      }
      if (graph_def.node_size() >= graphdef_node_size_max_) {
        VLOG(2) << "Skip dump for very large sub-graph with #nodes of "
                << graph_def.node_size();
        return "";
      }
      auto flib_def = &fbody->graph->flib_def();
      if (flib_def) {
        *graph_def.mutable_library() = flib_def->ToProto();
      } else {
        VLOG(2) << "flib_def is NULL. we may get an empty graph. should we "
                   "return here?";
      }
      auto graph_path = dump_graph::DumpGraphDefToFile(
          absl::StrCat("xla_func_", function_id, "_graph"), graph_def);
      VLOG(1) << "Dumped Graph Path: " << graph_path;
      return graph_path;
    }
  }
  return "";
}

std::string TaoCompilationCache::DumpShapeToFile(
    OpKernelContext* ctx, const NameAttrList& function,
    const std::string& function_id, int shape_cnt,
    const std::map<int, Tensor>& constant_args) {
  string file_name =
      absl::StrCat("xla_func_", function_id, "_shape_", shape_cnt);
  // Remove illegal characters from `name`.
  for (size_t i = 0; i < file_name.size(); ++i) {
    char ch = file_name[i];
    if (ch == '/' || ch == '[' || ch == ']' || ch == '*' || ch == '?') {
      file_name[i] = '_';
    }
  }

  string result = Canonicalize(function.name(), AttrSlice(&function.attr()));
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    if (constant_args.count(i) <= 0) {
      absl::StrAppend(&result, ",", DataTypeString(ctx->input_dtype(i)),
                      ctx->input(i).shape().DebugString());
    }
  }
  for (auto const& kv : constant_args) {
    int i = kv.first;
    size_t constant_vals_num = ctx->input(i).NumElements();
    auto fully_constant_debug_string = absl::StrCat(
        "Tensor<type: ", DataTypeString(ctx->input_dtype(i)),
        " shape: ", ctx->input(i).shape().DebugString(),
        " values: ", ctx->input(i).SummarizeValue(constant_vals_num), ">");
    absl::StrAppend(&result, "; ", fully_constant_debug_string);
  }
  auto& dir_name = GetTaoDumperOptions()->graph_dump_path;

  std::string shape_path = absl::StrCat(dir_name, "/", file_name);
  Status status = WriteStringToFile(Env::Default(), shape_path, result);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
    return "";
  }
  VLOG(1) << "Dumped Shape Path: " << shape_path;
  return shape_path;
}

void TaoCompilationCache::DumpInfo(OpKernelContext* ctx,
                                   const NameAttrList& function,
                                   const std::map<int, Tensor>& constant_args,
                                   uint64 hash_sig, bool compile_ok,
                                   TaoCompileFuncCallInfo* call_info) {
  if (GetTaoDumperOptions()->dump_level == 0) return;

  auto& collector = TaoCompInfoCollector::Get();

  string function_attr;
  string function_id = GenFunctionID(function, &function_attr);
  int pre_call_count = collector.GetShapeCallCount(function_id, hash_sig);
  if (call_info) {
    call_info->func_id = function_id;
    call_info->signature = hash_sig;
    call_info->call_idx = pre_call_count;
  }

  int shape_cnt = 0;
  if (!collector.FunctionExists(function_id)) {
    VLOG(2) << "Function Miss: " << function_id << " " << shape_cnt;
    // new function
    if (collector.GetShapeNum() >=
        GetTaoDumperOptions()->max_sampled_shape_num) {
      return;
    }
    collector.AddFunction(function_id, function_attr);
    if (GetTaoDumperOptions()->dump_level > 1 &&
        collector.GetFuncNum() < GetTaoDumperOptions()->max_dump_func_num) {
      // dump graph to file
      std::string graph_path = DumpGraphToFile(ctx, function, function_id);
      collector.SetFunctionGraph(function_id, graph_path);
    }
  } else {
    // existing function
    shape_cnt = collector.GetFuncShapeNum(function_id);
    VLOG(2) << "Function Hit: " << function_id << " " << shape_cnt;
  }

  // skip new shape if we've collected enough shapes
  if (pre_call_count == 0 &&
      collector.GetShapeNum() >= GetTaoDumperOptions()->max_sampled_shape_num) {
    return;
  }

  if (collector.AddShapeCallCount(function_id, hash_sig) == 1) {
    // this signature is new, dump shape file
    std::string shape_path;
    if (GetTaoDumperOptions()->dump_level > 1 &&
        shape_cnt < GetTaoDumperOptions()->max_dump_shape_per_func) {
      shape_path =
          DumpShapeToFile(ctx, function, function_id, shape_cnt, constant_args);
    }
    collector.InitShapeInfo(function_id, hash_sig, shape_path, compile_ok);
  } else {
    // this signature(shape) has been added before
    // in async mode, compile_ok is false before compiling done
    collector.UpdateShapeCallInfo(function_id, hash_sig, compile_ok);
  }
}

}  // namespace tao
}  // namespace tensorflow
