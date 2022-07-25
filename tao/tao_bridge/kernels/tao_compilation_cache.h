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

#ifndef TAO_TAO_BRIDGE_TAO_COMPILATION_CACHE_H_
#define TAO_TAO_BRIDGE_TAO_COMPILATION_CACHE_H_

#include <atomic>
#include <list>
#include <unordered_map>

#include "tao_bridge/common.h"
#include "tao_bridge/executable.h"
#include "tao_bridge/kernels/tao_compilation_info_collector.h"
#include "tao_bridge/tao_compiler_input.pb.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

namespace tao {

struct TaoProfileStat {
  std::atomic<int64> total_time_saved_in_us{0};
};

Tensor ToCpu(OpKernelContext* ctx, Tensor t, MemoryType mem_type);

class TaoCompilationCache : public ResourceBase {
 public:
  TaoCompilationCache(bool async_compilation = false);
  ~TaoCompilationCache() override;

  // We keep two different versions because this function's
  // signature varies with TF version. Same reason for the
  // commented 'override' keyword.
  std::string DebugString() /*override*/;
  std::string DebugString() const /*override*/;

  Status Compile(std::unique_ptr<TaoCompilerInput> input,
                 const NameAttrList& function,
                 const std::map<int, Tensor>& constant_args,
                 const std::set<int>& fixed_shape_args,
                 const std::set<int>& host_args,
                 const std::map<int, OptionalTensor>& variable_args,
                 OpKernelContext* ctx, Executable** executable,
                 TaoProfileStat** stat = nullptr, bool is_mlir = false,
                 TaoCompileFuncCallInfo* call_info = nullptr);

 public:
  std::string GenFunctionID(const NameAttrList& function,
                            std::string* attrstr = nullptr);

  static uint64 GetSignatureHash(
      const NameAttrList& function, const std::map<int, Tensor>& constant_args,
      const std::set<int>& fixed_shape_args, const std::set<int>& host_args,
      const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
      bool is_mlir);

  struct Signature {
    std::string name;

    // List of rank/type pairs, only used with Mlir dynamic shape compiler
    std::vector<std::pair<DataType, int>> arg_ranks;

    std::vector<std::pair<DataType, TensorShape>> arg_types;

    // ordinal args that are expected placed on cpu.
    // For cpu only execution, this will be empty.
    std::vector<int> host_args;

    // List of Tensor values for compile-time constant arguments to the
    // compilation, ordered by argument number. Tensors must be in host
    // memory.
    std::vector<Tensor> arg_values;

    bool operator==(const Signature& other) const;

    struct Hash {
      uint64 operator()(const Signature& signature) const;
    };
  };
  static std::string SignatureDebugString(const Signature& sig);

 private:
  Status CompileImpl(std::unique_ptr<TaoCompilerInput> input,
                     const NameAttrList& function,
                     const std::map<int, Tensor>& constant_args,
                     const std::set<int>& fixed_shape_args,
                     const std::set<int>& host_args,
                     const std::map<int, OptionalTensor>& variable_args,
                     OpKernelContext* ctx, Executable** executable,
                     bool is_mlir = false,
                     TaoCompileFuncCallInfo* call_info = nullptr);
  Status CompileImplAsync(std::unique_ptr<TaoCompilerInput> input,
                          const NameAttrList& function,
                          const std::map<int, Tensor>& constant_args,
                          const std::set<int>& fixed_shape_args,
                          const std::set<int>& host_args,
                          const std::map<int, OptionalTensor>& variable_args,
                          OpKernelContext* ctx, Executable** executable,
                          bool is_mlir = false,
                          TaoCompileFuncCallInfo* call_info = nullptr);

  Status DumpToFile();

  // Builds the signature for a compilation.
  static Status BuildSignature(
      const NameAttrList& function, const std::map<int, Tensor>& constant_args,
      const std::set<int>& fixed_shape_args, const std::set<int>& host_args,
      const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
      Signature* signature, bool is_mlir = false);

  struct Entry {
    mutex mu;

    // Have we tried compiling this entry?
    bool compiled = false;

    // Did compilation succeed?
    Status compilation_status GUARDED_BY(mu);

    // Output of the XlaCompiler.
    // XlaCompiler::CompilationResult compilation_result GUARDED_BY(mu);

    // The XLA executable compiled from <computation>. May be null if no
    // executable has been built.
    std::unique_ptr<Executable> executable GUARDED_BY(mu);

    bool compilation_slot_initialized = false;
    int compilation_slot = -1;
  };

  mutex compile_cache_mu_;
  std::unordered_map<Signature, std::unique_ptr<Entry>, Signature::Hash> cache_
      GUARDED_BY(compile_cache_mu_);

  bool async_compilation_;
  std::string tao_compiler_path_;
  std::string tao_upload_tool_path_;
  int64 profile_guide_mode_;

  // TODO: DEBUG ONLY
  bool remove_after_compile_ = false;

  std::string disc_cache_path_;

  std::atomic<bool> stop_{false};
  TaoProfileStat tao_profile_stat_;
  Thread* tao_profile_stat_handle_thread_ = nullptr;

  void HandleProfileStatLoop();
  void StartProfileStatHandleThread();

  // ONLINE DUMPER
  int cache_obj_idx_;  // we may have multiple TaoCompilationCache objects
                       // during the whole task process life-cycle. use this
                       // index to identify different objects in some cases
                       // like dumping filename
  int64 graphdef_node_size_min_;  // min number of graphdef's node to dump
  int64 graphdef_node_size_max_;  // max number of graphdef's node to dump

  // dump graph and shape to local
  std::string DumpGraphToFile(OpKernelContext* ctx,
                              const NameAttrList& function,
                              const std::string& function_id);
  std::string DumpShapeToFile(OpKernelContext* ctx,
                              const NameAttrList& function,
                              const std::string& function_id, int shape_cnt,
                              const std::map<int, Tensor>& constant_args);
  void DumpInfo(OpKernelContext* ctx, const NameAttrList& function,
                const std::map<int, Tensor>& constant_args, uint64 hash_sig,
                bool compile_ok, TaoCompileFuncCallInfo* call_info);

  TF_DISALLOW_COPY_AND_ASSIGN(TaoCompilationCache);
};

Status PrepareCompilerInput(const NameAttrList& function,
                            const std::map<int, Tensor>& constant_args,
                            const std::map<int, OptionalTensor>& variable_args,
                            OpKernelContext* ctx,
                            TaoCompilerInput* compiler_input);

Status CompileFunction(const std::string& func_name,
                       const std::string& tao_compiler_path,
                       const std::string& output_file_name,
                       TaoCompilerInput& input, bool remove_after_compile);
}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_TAO_COMPILATION_CACHE_H_
