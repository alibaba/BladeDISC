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

#pragma once

#include <sys/stat.h>

#include <atomic>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <list>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tao_bridge/dumper_common.h"
#include "tao_bridge/executable.h"
#include "tao_bridge/kernels/process.h"
#include "tao_bridge/tao_compiler_input.pb.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "third_party/json/single_include/nlohmann/json.hpp"

namespace tensorflow {
namespace tao {

enum CallTimeTag {
  // time entering launch op
  TIME_TAO_LAUNCH_OP_EXEC_BEGIN = 0,
  // time the op work finished, not the time leaving launch op (async executing)
  TIME_TAO_LAUNCH_OP_EXEC_END,
  // time entering the cache compile function
  TIME_COMPILE_CALL_BEGIN,
  // time leaving the cache compile function
  // compilation may be not finished at this time in async compiling mode
  TIME_COMPILE_CALL_END,
  // time executable binary from compilation starts running
  TIME_EXEC_BIN_RUN_BEGIN,
  // time executable binary from compilation finishes running
  TIME_EXEC_BIN_RUN_END,
  // time function from function lib (tf) starts running
  TIME_FUNC_RUN_BEGIN,
  // time function from function lib (tf) stops running
  TIME_FUNC_RUN_END,
  // time when compilation failed
  TIME_ENTER_FALL_BACK,
  // add new timestamps above
  // remember update ts_tag_list_ when adding timestamps
  TIME_TOTAL_NUM
};

// define compile status
// remember also update TaoCompInfoCollector::str(CompileStatus s)
enum CompileStatus {
  STATUS_INIT = 0,
  STATUS_ENQUEUED,
  STATUS_FAIL_ON_CACHE_FULL,
  STATUS_COMPILE_STARTED,
  STATUS_FAIL_ON_COMPILE,
  STATUS_PASS_ON_COMPILE,
  STATUS_PROFILE_STARTED,
  STATUS_FAIL_ON_PERF,
  STATUS_PASS_ON_TAO
};

struct TaoCompileFuncCallInfo {
  std::string func_id;
  uint64 signature{0};
  int call_idx{-1};
  std::atomic<uint64_t> ts[TIME_TOTAL_NUM];
  std::string exception;
  uint64_t tf_profile_time{0};
  uint64_t tao_profile_time{0};
  bool profile_done{false};

  TaoCompileFuncCallInfo() {
    for (int i = 0; i < TIME_TOTAL_NUM; ++i) {
      ts[i].store(0);
    }
  }
  bool initialized() const { return !func_id.empty() && signature != 0; }
};

class TaoCompilationCache;
class Executable;
class TaoCompInfoCollector final {
public:
  static TaoCompInfoCollector &Get(bool delay_init = false);

  void Init();

  // TaoCompilationCache is a tf session-level object
  // use this to track session life-cycle
  int GetOrCreateCacheIdx(TaoCompilationCache *cur);
  void DeregisterCache(TaoCompilationCache *cur);

  // a function corresponds to a subgraph (cluster)
  int64 GetFuncNum();
  int64 GetFuncShapeNum(const std::string &funcid);
  int64 GetShapeNum() { return sampled_shape_count_; }

  bool FunctionExists(const std::string &funcid);
  void AddFunction(const std::string &funcid, const std::string &funcattr);
  void SetFunctionGraph(const std::string &funcid, const std::string &path);

  // XLA only supports static shape. different shapes of the same function have
  // different signatures. a signature correspnds to a function with a fixed
  // shape get the current call counter value
  int GetShapeCallCount(const std::string &funcid, uint64 hash_sig);
  // increase the signature call counter. return the counter value after
  // increasing.
  int AddShapeCallCount(const std::string &funcid, uint64 hash_sig);
  void InitShapeInfo(const std::string &funcid, uint64 hash_sig,
                     const std::string &path, bool compile_ok);
  void UpdateShapeCallInfo(const std::string &funcid, uint64 hash_sig,
                           bool compile_ok);

  // set tag with current timestamp
  void SetCallTimestamp(TaoCompileFuncCallInfo *call_info, CallTimeTag tag);
  // process timestamps in one tao launch op call
  void FlushCallTimestamp(const TaoCompileFuncCallInfo *call_info);

  void UpdateShapeCompileStatus(const std::string &funcid, uint64 hash_sig,
                                CompileStatus s);
  void UpdateShapeCompileStatus(const Executable *exec, CompileStatus s);
  void AddExecutable(const Executable *exec, const std::string &funcid,
                     uint64 hash_sig);

  // update op histogram into op summary when online dumper enabled
  void UpdateOPHistogram(const std::string &op, uint64 count);
  // update unclustered op histogram into op summary when online dumper enabled
  void UpdateUnclusteredOPHistogram(const std::string &op, uint64 count);

  void UploadFile(const std::string &local_file, const std::string &type,
                  int timeout = -1, bool background = false,
                  bool compress = false);
  void EnqueUploadFile(const std::string &local_file, const std::string &type,
                       bool compress);

  void UpdatePerfStats(TaoCompilationCache *cur, int64 total_elapsed_us,
                       int64 total_saved_us, int64 print_cycle_sec,
                       double speed_up_all, double speed_up_recent,
                       bool last_call);

  void AddCompileFailedCase(const std::string &log,
                            const std::string &input_file, bool exited,
                            bool signaled, bool coredump, int code);

  // set json[key] = val
  template <typename T>
  void SetCustomValue(const std::string &key, const T &val) {
    if (opts_.dump_level == 0)
      return;
    mutex_lock l(info_mtx_);
    info_[key] = val;
  }

  // json[keys[0]][keys[1]][...] = val
  template <typename T>
  void SetCustomValue(const std::vector<std::string> &keys, const T &val) {
    if (opts_.dump_level == 0)
      return;
    mutex_lock l(info_mtx_);
    nlohmann::json *p = &info_;
    for (auto &&key : keys) {
      if (!p->contains(key)) {
        (*p)[key] = nlohmann::json{};
      }
      p = &(*p)[key];
    }
    (*p) = val;
  }

  // json[key] += val
  void AddCustomNumValue(const std::string &key, int64 val);
  // json[keys[0]][keys[1]][...] += val
  void AddCustomNumValue(const std::vector<std::string> &keys, int64 val);

  void CaptureEnvVars();

private: // internal functions
  TaoCompInfoCollector();
  ~TaoCompInfoCollector();
  TF_DISALLOW_COPY_AND_ASSIGN(TaoCompInfoCollector);

  // generate FINISH.txt file when process exists
  void _UploadInfoFile(const std::string &filename, const std::string &type,
                       bool background);
  // update shape compile queue/main latency when process exits if anything left
  // in shape_compile_ts_
  void CleanShapeCompileTimestamps();

  // Get an iterator to a shape in info_. Insert a new shape if exist=false &&
  // add_new=true.
  nlohmann::json::iterator GetOrCreateShapeInfo(const std::string &funcid,
                                                uint64 hash_sig, bool &exist,
                                                bool add_new);
  // add a new function item
  void AddFunctionInternal(const std::string &funcid,
                           const std::string &funcattr);
  // insert one new line in csv file whenever a tao launch op has been executed
  void AppendCsvFile(const TaoCompileFuncCallInfo *call_info);
  // accmulate latency counters whenever a tao launch op has been executed
  void SetCallLatency(const TaoCompileFuncCallInfo *call_info,
                      nlohmann::json &value);
  // accumulate related counters for those shapes not in dumping range
  void MissSampledShapeCallAdd(const TaoCompileFuncCallInfo *call_info);
  // update shape compile queue/main latency whenever compile status changes
  void UpdateCompileLatency(const std::string &funcid, uint64 hash_sig,
                            CompileStatus s, nlohmann::json &queue,
                            nlohmann::json &main);

  struct upload_file_info {
    std::string local_path;
    std::string type;
    bool compress;
  };
  void _FlushUploadFiles(const std::list<upload_file_info> &flist);
  // each TaoCompilationCache object corresponds to a new session
  struct TaoSessPerfInfo {
    int idx; // session index
    int64 total_elapsed_us{0};
    int64 total_saved_us{0};
    int64 print_cycle_sec{0};
    double speed_up_all{0};
    int interval_count{0};
    bool finalized{false};
    std::vector<double> estimated_speedups;

    TaoSessPerfInfo(int idx_) : idx(idx_){};
  };
  void _UploadPerfFile(TaoSessPerfInfo &sinfo, bool last_call);
  void _FinalizePerfStats();

public: // static functions
  static std::string HashValueToStr(uint64 hash);
  static uint64 GetCurTimeUs();
  static const char *str(CompileStatus s);
  static CompileStatus GetPrevCompileStatus(CompileStatus s);

private:
  TaoDumperOptions opts_;

  mutex obj_id_mtx_;
  std::unordered_map<TaoCompilationCache *, TaoSessPerfInfo>
      cache_obj_id_ GUARDED_BY(obj_id_mtx_);

  mutex info_mtx_;
  // json scheme refer to
  // https://yuque.antfin-inc.com/pai/dl_compiler_optimization/uuuzot#Ly6ro
  nlohmann::json info_ GUARDED_BY(info_mtx_);

  mutex upload_list_mtx_;
  std::list<upload_file_info> upload_file_list_ GUARDED_BY(upload_list_mtx_);

  // for call_timestamps.csv
  mutex ts_csv_mtx_;
  std::string ts_csv_path_;
  std::vector<std::string> ts_tag_list_;
  std::ofstream ts_csv_file_ GUARDED_BY(ts_csv_mtx_);
  std::atomic<uint64_t> ts_record_count_;

  // for miss sampled shapes tracking
  mutex miss_func_shape_mtx_;
  std::unordered_map<std::string, std::unordered_set<uint64_t>>
      miss_func_shape_ GUARDED_BY(ts_csv_mtx_);

  mutex shape_compile_ts_mtx_;
  std::unordered_map<std::string,
                     std::unordered_map<uint64_t, std::vector<uint64_t>>>
      shape_compile_ts_ GUARDED_BY(shape_compile_ts_mtx_);

  mutex shape_exec_mtx_;
  std::unordered_map<const Executable *, std::pair<std::string, uint64_t>>
      shape_exec_ GUARDED_BY(shape_exec_mtx_);

  std::atomic<uint64_t> sampled_shape_count_{0};
  std::atomic<uint64_t> miss_sampled_shape_count_{0};

  std::shared_ptr<Process> tao_p_;

  std::string PathBaseName(const std::string &path) {
    size_t pos = path.find_last_of("/");
    return pos == std::string::npos ? path : path.substr(pos + 1);
  }
};

} // namespace tao
} // namespace tensorflow
