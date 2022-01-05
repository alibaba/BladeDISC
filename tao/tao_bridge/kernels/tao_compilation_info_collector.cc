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

#include "tao_bridge/kernels/tao_compilation_info_collector.h"

#include <dlfcn.h>
#include <sys/time.h>
#include <sys/wait.h>

#include <chrono>
#include <iomanip>

#include "absl/strings/str_split.h"
#include "tao_bridge/tf/subprocess.h"
#include "tao_bridge/version.h"

#ifdef TAO_ENABLE_UPLOAD_TOOL
#include "tools/tao/uploader/uploader.h"
#endif

extern char **environ;

namespace tensorflow {
namespace tao {

namespace {
template <typename T> T json_num_add(nlohmann::json &value, T val) {
  auto orig = value.get<T>();
  orig += val;
  value = orig;
  return orig;
}

template <typename T>
T json_obj_add(nlohmann::json &value, const std::string &key, T val) {
  if (!value.contains(key)) {
    value[key] = (T)0;
  }
  return json_num_add(value[key], val);
}

typedef int (*cuDeviceGetName_t)(char *name, int len, int dev);
typedef int (*cuInit_t)(unsigned int Flags);
typedef int (*cuDeviceGetCount_t)(int *count);
#define CALL_CUDA(api, args)                                                   \
  {                                                                            \
    auto pfunc = (api##_t)dlsym(cudalib, #api);                                \
    if (!pfunc) {                                                              \
      VLOG(1) << "ERROR: failed to get " #api " symbol: " << dlerror();        \
      return #api "_NOT_FOUND";                                                \
    }                                                                          \
    auto ret = pfunc args;                                                     \
    if (ret != 0) {                                                            \
      VLOG(1) << "ERROR: " #api " failed, error code " << ret;                 \
      return #api "_RUN_ERROR";                                                \
    }                                                                          \
  }
std::string GetGpuInfo(int &gpu_count) {
  void *cudalib =
      dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
  if (!cudalib) {
    VLOG(1) << "ERROR: failed to load libcuda.so: " << dlerror();
    return "LOAD_LIBCUDA_FAIL";
  }
  CALL_CUDA(cuInit, (0));
  CALL_CUDA(cuDeviceGetCount, (&gpu_count));
  if (gpu_count < 1) {
    return "NO_DEVICE_FOUND";
  }
  char buf[256];
  CALL_CUDA(cuDeviceGetName, (buf, 256, 0));
  buf[255] = 0; // in case any error happened
  return buf;
}

void get_binary_version(nlohmann::json &info, const std::string &name,
                        const std::string &exe_path) {
  tensorflow::tao::SubProcess exe;
  exe.SetProgram(exe_path, {exe_path, "-v"});
  exe.SetChannelAction(tensorflow::tao::CHAN_STDOUT,
                       tensorflow::tao::ACTION_PIPE);
  if (!exe.Start()) {
    LOG(WARNING) << "Couldn't invoke " << exe_path << " -v";
    return;
  }

  string out;
  int exit_code = exe.Communicate(/*stdin_input=*/nullptr, &out,
                                  /*stderr_output=*/nullptr);
  if (exit_code != 0) {
    LOG(WARNING) << "Running " << exe_path << " -v returned " << exit_code;
    return;
  }

  auto lines = absl::StrSplit(out, '\n');
  for (auto &&line : lines) {
    std::vector<std::string> kv = absl::StrSplit(line, '=');
    if (kv.size() != 2)
      continue;
    info[name][kv[0]] = kv[1];
  }
}

} // namespace

static const TaoCompInfoCollector *_only_for_init_info_collector_dont_use_ =
    &TaoCompInfoCollector::Get(true);

TaoCompInfoCollector &TaoCompInfoCollector::Get(bool delay_init) {
  static TaoCompInfoCollector _obj;
  static std::once_flag _obj_init;
  if (!delay_init) {
    std::call_once(_obj_init, std::bind(&TaoCompInfoCollector::Init, &_obj));
  }
  return _obj;
}

string TaoCompInfoCollector::HashValueToStr(uint64 hash) {
  std::stringstream ss;
  ss << std::hex << std::setw(16) << std::setfill('0') << hash;
  return ss.str();
}

// benchmark: 10 concurrent threads, 10000000 samples per threads
// all return macro seconds since epo
// results:
// gettimeofday: 0.03633us per sample
// std chrono system_clock: 1.14377us per sample
// std chrono steady_clock: 1.14233us per sample
// clock_gettime monotonic: 0.03441us per sample
// time: 0.0028us per sample
// rdtsc: 0.01us per sample
uint64 TaoCompInfoCollector::GetCurTimeUs() {
  timeval tv;
  gettimeofday(&tv, nullptr);
  return uint64(tv.tv_sec) * 1e6 + tv.tv_usec;
}

const char *TaoCompInfoCollector::str(CompileStatus s) {
  const char *status_[] = {"init",
                           "enqueued",
                           "fail_on_cache_full",
                           "compile_started",
                           "fail_on_compile",
                           "pass_on_compile",
                           "profile_started",
                           "fail_on_perf_check",
                           "pass_on_tao"};
  return status_[s];
}

CompileStatus TaoCompInfoCollector::GetPrevCompileStatus(CompileStatus s) {
  // deduce previous status
  CompileStatus pre = STATUS_INIT;
  switch (s) {
  case STATUS_COMPILE_STARTED:
    pre = STATUS_ENQUEUED;
    break;
  case STATUS_FAIL_ON_COMPILE:
  case STATUS_PASS_ON_COMPILE:
    pre = STATUS_COMPILE_STARTED;
    break;
  case STATUS_PROFILE_STARTED:
    pre = STATUS_PASS_ON_COMPILE;
    break;
  case STATUS_FAIL_ON_PERF:
  case STATUS_PASS_ON_TAO:
    pre = STATUS_PROFILE_STARTED;
    break;
  default:
    break;
  }
  return pre;
}

void TaoCompInfoCollector::CaptureEnvVars() {
  if (opts_.dump_level == 0)
    return;
  mutex_lock l(info_mtx_);
  int i = 1;
  char *s = *environ;
  for (; s; i++) {
    std::string kv(s);
    size_t pos = kv.find("=");
    if (pos != std::string::npos) {
      auto name = kv.substr(0, pos);
      if (opts_.cap_all_envs || opts_.cap_more_envs.count(name) ||
          name.find("TAO_") == 0 || name.find("TF_") == 0) {
        info_["env"][name] = kv.substr(pos + 1);
      }
    }
    s = *(environ + i);
  }
}

void TaoCompInfoCollector::Init() {
  // stats always recorded even if online dumper disabled
  if (opts_.dump_level > 0) {
    // create local output directory in case not exists
    tensorflow::Env::Default()->RecursivelyCreateDir(opts_.graph_dump_path);

    // time sensitive first
    info_["libtao_ops_load_us"] = GetCurTimeUs();
    info_["begin_us"] = tao_p_->StartTimeUS();

    // tao bridge version info
    info_["tao_build_version"] = std::string(TAO_BUILD_VERSION);
    info_["tao_build_commit"] = std::string(TAO_BUILD_GIT_HEAD);
    info_["tao_build_time"] = std::string(TAO_BUILD_TIME);

    // tao compiler version
    get_binary_version(info_, "compiler_version",
                       GetTaoBridgeOptions()->tao_compiler_path);
    // upload tool version
    if (!opts_.tao_upload_tool_path.empty()) {
      get_binary_version(info_, "upload_tool_version",
                         opts_.tao_upload_tool_path);
    }

    // get environment variables
    CaptureEnvVars();

    // init sub structures
    info_["calls"] = nlohmann::json{};
    info_["sessions"] = nlohmann::json{};

    // miss sampled
    info_["miss_sampled"] = nlohmann::json{};
    info_["miss_sampled"]["cluster_count"] = 0;
    info_["miss_sampled"]["shape_count"] = 0;
    info_["miss_sampled"]["call_count"] = 0;
    info_["miss_sampled"]["na_func_id"] = 0;
    info_["miss_sampled"]["na_shape_id"] = 0;
    info_["miss_sampled"]["tao_run_total_latency"] = (uint64_t)0;
    info_["miss_sampled"]["tao_run_count"] = 0;
    info_["miss_sampled"]["tf_run_total_latency"] = (uint64_t)0;
    info_["miss_sampled"]["tf_run_count"] = 0;
    info_["miss_sampled"]["tao_launch_op_total_latency"] = (uint64_t)0;
    info_["miss_sampled"]["get_compile_cache_total_latency"] = (uint64_t)0;
    info_["miss_sampled"]["compile_status_count"] = nlohmann::json{};
    info_["miss_sampled"]["compile_queue_total_latency"] = (uint64_t)0;
    info_["miss_sampled"]["compile_main_total_latency"] = (uint64_t)0;

    // unclustered op info
    info_["op_summary"] = nlohmann::json{};
    info_["op_summary"]["all"] = nlohmann::json{};
    info_["op_summary"]["unclustered_op"] = nlohmann::json{};

    // process info
    std::vector<std::string> cmd;
    info_["command"] = tao_p_->CMD();
    info_["cwd"] = tao_p_->CWD();

    // gpu info
    // this will dlopen libcuda.so. not sure if this has any side effects
    // although we haven't met yet. Just wrap it with an env knob and turn it on
    // by default.
    if (opts_.dump_gpu_info) {
      int gpu_count = 0;
      info_["gpu_model"] = GetGpuInfo(gpu_count);
      info_["gpu_count"] = gpu_count;
    }
    // dump and upload begin.txt
    _UploadInfoFile("begin.txt", "-p", false);
  }

  if (opts_.dump_level > 1) {
    // Q: why not put this into some static global variable
    // A: it may be initialized after the global variable
    // '_only_for_init_info_collector_dont_use_' which triggered creation of
    // TaoCompInfoCollector object. Thus we may refer to uninitialized (or
    // even not created yet) ts_tag_list_ here.
    ts_tag_list_ = {
        "tao_launch_op_begin", "tao_launch_op_end", "cache_compile_begin",
        "cache_compile_end",   "exec_run_begin",    "exec_run_end",
        "func_run_begin",      "func_run_end",      "enter_fallback"};

    // initialize timestamp log
    if (opts_.max_dump_timestamp_record > 0) {
      ts_csv_path_ = opts_.graph_dump_path + "/call_timestamps.csv";
      ts_csv_file_.open(ts_csv_path_);
      ts_record_count_ = 0;
      if (ts_csv_file_.is_open()) {
        ts_csv_file_ << "func_id,signature,call_idx,";
        for (auto &&tag : ts_tag_list_)
          ts_csv_file_ << tag << ",";
        ts_csv_file_ << "other_tags" << std::endl;
      } else {
        VLOG(1) << "ERROR: failed to open " << ts_csv_path_;
      }
    }
  }
}

TaoCompInfoCollector::TaoCompInfoCollector() {
  opts_ = *GetTaoDumperOptions();
  // Init time-sensitive fields only
  // Put other initializations to Init()
  tao_p_.reset(new Process());
  if (opts_.dump_level > 0) {
    info_["libtao_ops_load_us"] = GetCurTimeUs();
    info_["begin_us"] = tao_p_->StartTimeUS();
  }
}

TaoCompInfoCollector::~TaoCompInfoCollector() {
  if (opts_.dump_level == 0)
    return;
  info_["end_us"] = GetCurTimeUs();
  info_["latency_us"] =
      info_["end_us"].get<uint64>() - info_["begin_us"].get<uint64>();

  // finalize perf stats if the thread exists abnormally
  _FinalizePerfStats();
  if (opts_.dump_level > 1) {
    CleanShapeCompileTimestamps();
    if (ts_csv_file_.is_open()) {
      ts_csv_file_.close();
      UploadFile(ts_csv_path_, "-o");
    }
    // flush the uploading queue if not empty
    _FlushUploadFiles(upload_file_list_);
    _UploadInfoFile("FINISH.txt", "-o", false);
  } else {
    _UploadInfoFile("end.txt", "-p", false);
  }
  // upload finish flag to OSS to make it easier to scan daily
  UploadFile("", "--finish");
}

int TaoCompInfoCollector::GetOrCreateCacheIdx(TaoCompilationCache *cur) {
  mutex_lock l(obj_id_mtx_);
  auto it = cache_obj_id_.find(cur);
  int idx;
  if (it == cache_obj_id_.end()) {
    // new cache idx
    idx = cache_obj_id_.size();
    cache_obj_id_.emplace(cur, TaoSessPerfInfo(idx));
    if (opts_.dump_level > 0) {
      mutex_lock l(info_mtx_);
      info_["sessions"].emplace_back(
          nlohmann::json{{"begin_us", GetCurTimeUs()}, {"runs", {}}});
    }
  } else {
    // existing cache idx
    idx = it->second.idx;
  }
  return idx;
}

void TaoCompInfoCollector::DeregisterCache(TaoCompilationCache *cur) {
  if (opts_.dump_level == 0)
    return;
  auto idx = GetOrCreateCacheIdx(cur);
  mutex_lock l(info_mtx_);
  info_["sessions"][idx]["end_us"] = GetCurTimeUs();
}

int64 TaoCompInfoCollector::GetFuncNum() {
  if (opts_.dump_level == 0)
    return 0;
  mutex_lock l(info_mtx_);
  return info_["calls"].size();
}

int64 TaoCompInfoCollector::GetFuncShapeNum(const std::string &funcid) {
  if (opts_.dump_level == 0)
    return 0;
  mutex_lock l(info_mtx_);
  auto iter_func = info_["calls"].find(funcid);
  if (iter_func == info_["calls"].end()) {
    return 0;
  }
  return iter_func.value()["shapes"].size();
}

bool TaoCompInfoCollector::FunctionExists(const std::string &funcid) {
  if (opts_.dump_level == 0)
    return false;
  mutex_lock l(info_mtx_);
  return info_["calls"].contains(funcid);
}

void TaoCompInfoCollector::AddFunctionInternal(const std::string &funcid,
                                               const std::string &funcattr) {
  info_["calls"][funcid]["attr"] = funcattr;
  info_["calls"][funcid]["graph"] = "not_set_yet";
  info_["calls"][funcid]["shapes"] = nlohmann::json{};
}

void TaoCompInfoCollector::AddFunction(const std::string &funcid,
                                       const std::string &funcattr) {
  if (opts_.dump_level == 0)
    return;
  mutex_lock l(info_mtx_);
  AddFunctionInternal(funcid, funcattr);
}

void TaoCompInfoCollector::SetFunctionGraph(const std::string &funcid,
                                            const std::string &path) {
  if (opts_.dump_level == 0)
    return;
  mutex_lock l(info_mtx_);
  auto iter_func = info_["calls"].find(funcid);
  if (iter_func == info_["calls"].end()) {
    // this should never happen
    VLOG(1) << "ERROR: function " << funcid << " not registered yet";
    AddFunctionInternal(funcid, "unknown_attr");
    iter_func = info_["calls"].find(funcid);
  }
  iter_func.value()["graph"] = PathBaseName(path);
  EnqueUploadFile(path, "-o", false);
}

nlohmann::json::iterator TaoCompInfoCollector::GetOrCreateShapeInfo(
    const std::string &funcid, uint64 hash_sig_, bool &exist, bool add_new) {
  exist = true;
  nlohmann::json::iterator iter_func;
  if (funcid.empty()) {
    exist = false;
  } else {
    iter_func = info_["calls"].find(funcid);
    if (iter_func == info_["calls"].end()) {
      exist = false;
    }
  }
  if (!exist) {
    if (!add_new) {
      return iter_func;
    }
    // this should never happen
    VLOG(1) << "ERROR: function " << funcid << " not registered yet";
    AddFunctionInternal(funcid, "unknown_attr");
    iter_func = info_["calls"].find(funcid);
  }

  auto &shapes = iter_func.value()["shapes"];
  std::string hash_sig;
  nlohmann::json::iterator iter_shape;
  if (hash_sig_ == 0) {
    exist = false;
  } else {
    hash_sig = HashValueToStr(hash_sig_);
    iter_shape = shapes.find(hash_sig);
    if (iter_shape == shapes.end()) {
      exist = false;
    }
  }

  if (!exist) {
    if (!add_new) {
      return iter_shape;
    }
    // first time add the shape. init sub structures.
    ++sampled_shape_count_;
    shapes[hash_sig] = nlohmann::json{};
    iter_shape = shapes.find(hash_sig);
    iter_shape.value()["count"] = 0;
    iter_shape.value()["compile_ok"] = false;
    iter_shape.value()["compile_pass_index"] = -1;
    iter_shape.value()["tao_run_total_latency"] = (uint64_t)0;
    iter_shape.value()["tao_run_count"] = 0;
    iter_shape.value()["tf_run_total_latency"] = (uint64_t)0;
    iter_shape.value()["tf_run_count"] = 0;
    iter_shape.value()["tao_launch_op_total_latency"] = (uint64_t)0;
    iter_shape.value()["get_compile_cache_total_latency"] = (uint64_t)0;
    iter_shape.value()["compile_status"] = str(STATUS_INIT);
    iter_shape.value()["compile_queue_latency"] = (uint64_t)0;
    iter_shape.value()["compile_main_latency"] = (uint64_t)0;
  }
  return iter_shape;
}

int TaoCompInfoCollector::GetShapeCallCount(const std::string &funcid,
                                            uint64 hash_sig) {
  if (opts_.dump_level == 0)
    return 0;
  mutex_lock l(info_mtx_);
  bool exist;
  auto iter_shape = GetOrCreateShapeInfo(funcid, hash_sig, exist, false);
  if (!exist) {
    return 0;
  }
  return iter_shape.value()["count"].get<int>();
}

int TaoCompInfoCollector::AddShapeCallCount(const std::string &funcid,
                                            uint64 hash_sig) {
  if (opts_.dump_level == 0)
    return 0;
  mutex_lock l(info_mtx_);
  bool exist;
  auto iter_shape = GetOrCreateShapeInfo(funcid, hash_sig, exist, true);
  return json_num_add<int>(iter_shape.value()["count"], 1);
}

void TaoCompInfoCollector::InitShapeInfo(const std::string &funcid,
                                         uint64 hash_sig,
                                         const std::string &path,
                                         bool compile_ok) {
  if (opts_.dump_level == 0)
    return;
  mutex_lock l(info_mtx_);
  bool exist;
  auto iter_shape = GetOrCreateShapeInfo(funcid, hash_sig, exist, false);
  if (!exist) {
    // this should never happen
    VLOG(1) << "ERROR: shape [" << funcid << ", " << hash_sig
            << " not registered yet";
    return;
  }
  std::string shape_file("NA");
  if (!path.empty()) {
    shape_file = PathBaseName(path);
    EnqueUploadFile(path, "-o", false);
  }
  iter_shape.value()["shape"] = shape_file;
  iter_shape.value()["compile_ok"] = compile_ok;
  iter_shape.value()["compile_pass_index"] = compile_ok ? 0 : -1;
}

void TaoCompInfoCollector::UpdateShapeCallInfo(const std::string &funcid,
                                               uint64 hash_sig,
                                               bool compile_ok) {
  if (opts_.dump_level == 0)
    return;
  mutex_lock l(info_mtx_);
  bool exist;
  auto iter_shape = GetOrCreateShapeInfo(funcid, hash_sig, exist, false);
  if (!exist) {
    // this should never happen
    VLOG(1) << "ERROR: shape [" << funcid << ", " << hash_sig
            << " not registered yet";
    return;
  }
  auto orig_status = iter_shape.value()["compile_ok"].get<bool>();
  iter_shape.value()["compile_ok"] = compile_ok;
  if (!orig_status && compile_ok) {
    // compile_ok: false -> true
    auto count = iter_shape.value()["count"].get<int>();
    iter_shape.value()["compile_pass_index"] = count - 1;
  } else if (orig_status && !compile_ok) {
    // exception: compile_ok true -> ok
    auto count = iter_shape.value()["count"].get<int>();
    iter_shape.value()["compile_pass_index"] = -1;
    iter_shape.value()["compile_pass_to_fail_index"] = count - 1;
  }
}

void TaoCompInfoCollector::SetCallTimestamp(TaoCompileFuncCallInfo *call_info,
                                            CallTimeTag tag) {
  if (opts_.dump_level == 0)
    return;
  auto old = call_info->ts[tag].exchange(GetCurTimeUs());
  if (old != 0) {
    // set the tag twice
    std::stringstream ss;
    ss << ts_tag_list_[tag] << " set again; ";
    call_info->exception += ss.str();
  }
}

void TaoCompInfoCollector::AppendCsvFile(
    const TaoCompileFuncCallInfo *call_info) {
  // append timestamps to csv file
  if (!ts_csv_file_.is_open()) {
    return;
  }
  if (ts_record_count_ >= opts_.max_dump_timestamp_record) {
    return;
  }

  mutex_lock l(ts_csv_mtx_);
  ++ts_record_count_;

  ts_csv_file_ << call_info->func_id << ","
               << HashValueToStr(call_info->signature) << ","
               << call_info->call_idx << ",";
  for (auto &&t : call_info->ts) {
    ts_csv_file_ << t << ",";
  }

  if (call_info->exception.empty()) {
    ts_csv_file_ << "-\n";
  } else {
    ts_csv_file_ << call_info->exception << "\n";
  }
}

void TaoCompInfoCollector::FlushCallTimestamp(
    const TaoCompileFuncCallInfo *call_info) {
  if (opts_.dump_level == 0)
    return;
  if (!call_info->initialized())
    return;

  mutex_lock l(info_mtx_);

  AppendCsvFile(call_info);

  bool exist;
  auto iter_shape = GetOrCreateShapeInfo(call_info->func_id,
                                         call_info->signature, exist, false);
  if (exist) {
    // this shape is in sampling range, update its own counters
    SetCallLatency(call_info, iter_shape.value());
  } else {
    // this shape is not in sampling range, update overall statistic counters
    MissSampledShapeCallAdd(call_info);
  }
}

void TaoCompInfoCollector::UpdateOPHistogram(const std::string &op,
                                             uint64 count) {
  if (opts_.dump_level > 1) {
    mutex_lock l(info_mtx_);
    auto &op_summary = info_["op_summary"];
    json_obj_add<int>(op_summary["all"], op, count);
  }
}

void TaoCompInfoCollector::UpdateUnclusteredOPHistogram(const std::string &op,
                                                        uint64 count) {
  if (opts_.dump_level > 1) {
    mutex_lock l(info_mtx_);
    auto &op_summary = info_["op_summary"];
    json_obj_add<int>(op_summary["unclustered_op"], op, count);
  }
}

void TaoCompInfoCollector::UpdateShapeCompileStatus(const std::string &funcid,
                                                    uint64 hash_sig,
                                                    CompileStatus s) {
  if (opts_.dump_level == 0)
    return;
  mutex_lock l(info_mtx_);
  bool exist;
  auto iter_shape = GetOrCreateShapeInfo(funcid, hash_sig, exist, false);
  if (exist) {
    // this shape is in sampling range, update its own counters
    auto &shape = iter_shape.value();
    shape["compile_status"] = str(s);
    VLOG(2) << "update shape " << funcid << "." << hash_sig
            << " compile_status: " << str(s);
    UpdateCompileLatency(funcid, hash_sig, s, shape["compile_queue_latency"],
                         shape["compile_main_latency"]);
  } else {
    // this shape is not in sampling range, update overall statistic counters
    auto &miss = info_["miss_sampled"];
    // previous status count -1, current status count +1
    // INIT status is not counted
    CompileStatus pre = GetPrevCompileStatus(s);
    VLOG(2) << "update shape " << funcid << "." << hash_sig
            << " compile_status from " << str(pre) << " to " << str(s);
    if (pre != STATUS_INIT) {
      json_obj_add<int>(miss["compile_status_count"], str(pre), -1);
    }
    json_obj_add<int>(miss["compile_status_count"], str(s), 1);
    UpdateCompileLatency(funcid, hash_sig, s,
                         miss["compile_queue_total_latency"],
                         miss["compile_main_total_latency"]);
  }
}

void TaoCompInfoCollector::UpdateShapeCompileStatus(const Executable *exec,
                                                    CompileStatus s) {
  if (opts_.dump_level == 0)
    return;
  mutex_lock l(shape_exec_mtx_);
  auto it = shape_exec_.find(exec);
  if (it != shape_exec_.end()) {
    UpdateShapeCompileStatus(it->second.first, it->second.second, s);
  } else {
    // exec not found. this is a weird case. we should figure out it why.
    VLOG(1) << "WARNING: Executable not registered";
    UpdateShapeCompileStatus("-", 0, s);
  }
}

void TaoCompInfoCollector::AddExecutable(const Executable *exec,
                                         const std::string &funcid,
                                         uint64 hash_sig) {
  if (opts_.dump_level == 0)
    return;
  mutex_lock l(shape_exec_mtx_);
  shape_exec_.emplace(exec, std::make_pair(funcid, hash_sig));
}

void TaoCompInfoCollector::CleanShapeCompileTimestamps() {
  mutex_lock l(info_mtx_);
  mutex_lock l2(shape_compile_ts_mtx_);

  auto now = GetCurTimeUs();
  for (auto &&func : shape_compile_ts_) {
    auto &fid = func.first;
    for (auto &&shape : func.second) {
      auto &sid = shape.first;
      auto &ts = shape.second;
      bool exist;
      auto iter_shape = GetOrCreateShapeInfo(fid, sid, exist, false);
      if (ts[1] == 0) {
        // still in enqueue status, update queue latency
        if (exist) {
          iter_shape.value()["compile_queue_latency"] = now - ts[0];
        } else {
          json_num_add<uint64_t>(
              info_["miss_sampled"]["compile_queue_total_latency"],
              now - ts[0]);
        }
      } else {
        // still in compiling status, update main latency
        if (exist) {
          iter_shape.value()["compile_main_latency"] = now - ts[0];
        } else {
          json_num_add<uint64_t>(
              info_["miss_sampled"]["compile_main_total_latency"], now - ts[0]);
        }
      }
    }
  }
}

void TaoCompInfoCollector::SetCallLatency(
    const TaoCompileFuncCallInfo *call_info, nlohmann::json &value) {
  if (call_info->profile_done) {
    value["tf_profile_time"] = call_info->tf_profile_time;
    value["tao_profile_time"] = call_info->tao_profile_time;
  }
  uint64_t t0 = call_info->ts[TIME_EXEC_BIN_RUN_BEGIN];
  uint64_t t1 = call_info->ts[TIME_EXEC_BIN_RUN_END];
  if (t1 > t0 && t0 > 0) {
    // run tao executable bin
    json_num_add<uint64_t>(value["tao_run_total_latency"], t1 - t0);
    json_num_add<int>(value["tao_run_count"], 1);
  } else {
    // run tf function
    t0 = call_info->ts[TIME_FUNC_RUN_BEGIN];
    t1 = call_info->ts[TIME_FUNC_RUN_END];
    if (t1 > t0 && t0 > 0) {
      json_num_add<uint64_t>(value["tf_run_total_latency"], t1 - t0);
      json_num_add<int>(value["tf_run_count"], 1);
    }
  }
  t0 = call_info->ts[TIME_TAO_LAUNCH_OP_EXEC_BEGIN];
  t1 = call_info->ts[TIME_TAO_LAUNCH_OP_EXEC_END];
  if (t1 > t0 && t0 > 0) {
    json_num_add<uint64_t>(value["tao_launch_op_total_latency"], t1 - t0);
  }
  t0 = call_info->ts[TIME_COMPILE_CALL_BEGIN];
  t1 = call_info->ts[TIME_COMPILE_CALL_END];
  if (t1 > t0 && t0 > 0) {
    json_num_add<uint64_t>(value["get_compile_cache_total_latency"], t1 - t0);
  }
}

void TaoCompInfoCollector::MissSampledShapeCallAdd(
    const TaoCompileFuncCallInfo *call_info) {
  mutex_lock l(miss_func_shape_mtx_);
  auto &miss = info_["miss_sampled"];

  if (miss_sampled_shape_count_ < opts_.max_miss_sampled_shape_num) {
    if (!call_info->func_id.empty()) {
      auto it_func = miss_func_shape_.find(call_info->func_id);
      if (it_func == miss_func_shape_.end()) {
        // new function
        json_num_add<int>(miss["cluster_count"], 1);
        it_func =
            miss_func_shape_
                .emplace(call_info->func_id, std::unordered_set<uint64_t>())
                .first;
      }
      if (call_info->signature != 0) {
        if (it_func->second.insert(call_info->signature).second) {
          // new shape
          json_num_add<int>(miss["shape_count"], 1);
          ++miss_sampled_shape_count_;
        }
      } else {
        json_num_add<int>(miss["na_shape_id"], 1);
      }
    } else {
      json_num_add<int>(miss["na_func_id"], 1);
    }
  }

  json_num_add<int>(miss["call_count"], 1);
  SetCallLatency(call_info, miss);
}

void TaoCompInfoCollector::UpdateCompileLatency(const std::string &funcid,
                                                uint64 hash_sig,
                                                CompileStatus s,
                                                nlohmann::json &queue,
                                                nlohmann::json &main) {
  mutex_lock l(shape_compile_ts_mtx_);
  switch (s) {
  case STATUS_ENQUEUED: {
    // record enque start time
    shape_compile_ts_[funcid][hash_sig].resize(2, 0);
    shape_compile_ts_[funcid][hash_sig][0] = GetCurTimeUs();
  } break;
  case STATUS_COMPILE_STARTED: {
    auto t1 = GetCurTimeUs();
    shape_compile_ts_[funcid][hash_sig].resize(
        2, 0); // in case ENQUE is not triggered
    auto t0 = shape_compile_ts_[funcid][hash_sig][0];
    if (t0 == 0) {
      // This may happen when async compiling started before DumpInfo()
      // finish. Just take t0 = t1 to ignore the latency
      t0 = t1;
      shape_compile_ts_[funcid][hash_sig][0] = t0;
    }
    json_num_add<uint64_t>(queue, t1 - t0);
    shape_compile_ts_[funcid][hash_sig][1] = t1;
  } break;
  case STATUS_PASS_ON_COMPILE:
  case STATUS_FAIL_ON_COMPILE: {
    auto t2 = GetCurTimeUs();
    shape_compile_ts_[funcid][hash_sig].resize(
        2, 0); // in case COMP_STARTED is not triggered
    auto t1 = shape_compile_ts_[funcid][hash_sig][1];
    if (t1 == 0) {
      // This may happen when async compiling finished before DumpInfo()
      // finish Just take t1 = t2
      t1 = t2;
    }
    json_num_add<uint64_t>(main, t2 - t1);
    // remove temp record
    shape_compile_ts_[funcid].erase(hash_sig);
    if (shape_compile_ts_[funcid].empty()) {
      shape_compile_ts_.erase(funcid);
    }
  } break;
  default:
    break;
  }
}

void TaoCompInfoCollector::UploadFile(const std::string &local_file,
                                      const std::string &type, int timeout,
                                      bool background, bool compress) {
#ifdef TAO_ENABLE_UPLOAD_TOOL
  static std::atomic<bool> err_printed{false};

  if (opts_.dump_level == 0)
    return;

  auto &upload_tool = opts_.tao_upload_tool_path;

  // skip uploading if tool is not specified
  if (upload_tool.empty())
    return;

  // skip uploading if we have met uploading error before
  if (err_printed)
    return;

  // use external uploading tool
  std::stringstream cmd;
  string target_file;
  if (compress) {
    target_file = local_file + ".tgz";
    cmd << "tar -czf " << target_file << " " << local_file
        << " >& /dev/null && ";
  } else {
    target_file = local_file;
  }
  string upload_log = local_file + ".upload.log";

  cmd << "LD_LIBRARY_PATH=" << getenv("LD_LIBRARY_PATH") << " " << upload_tool
      << " " << type << " " << target_file << " -k "
      << ::tao::tools::calc_expected_caller_key() << " --ppid " << getpid();
  if (timeout >= 0) {
    cmd << " -t " << timeout;
  }
  // delete local file if upload succeed and user not want to keep it
  if (opts_.remove_after_compile) {
    cmd << " --delete";
  }
  if (background) {
    cmd << " --background ";
  }
  cmd << " >& " << upload_log;

  int ret = std::system(cmd.str().c_str());
  if (ret != 0) {
    if (!err_printed) {
      err_printed = true;
      std::string error_log;
      ReadFileToString(Env::Default(), upload_log, &error_log);
      LOG(ERROR) << "[For PAI developer only, Users can ignore] tao "
                    "upload tool returned non-zero value("
                 << ret << "): " << error_log;
    }
    if (ret == -1) {
      VLOG(2) << "'" << cmd.str() << "' start failed: " << strerror(errno);
    } else if (WIFEXITED(ret)) {
      VLOG(2) << "'" << cmd.str()
              << "' exit with non-zero value: " << WEXITSTATUS(ret);
    } else if (WIFSIGNALED(ret)) {
      VLOG(2) << "'" << cmd.str() << "' terminated by signal: " << WTERMSIG(ret)
              << " " << (WCOREDUMP(ret) ? "with" : "without") << " core dump";
    } else {
      VLOG(2) << "'" << cmd.str()
              << "' system() call returned non-zero value: " << ret;
    }
  }
#endif // TAO_ENABLE_UPLOAD_TOOL
}

void TaoCompInfoCollector::_UploadInfoFile(const std::string &filename,
                                           const std::string &type,
                                           bool background) {
  if (opts_.tao_upload_tool_path.empty())
    return;

  // upload FINISH file in the last to indicate the task is done
  // generate finish file in local
  std::string finish_file = opts_.graph_dump_path + "/" + filename;
  std::ofstream finish_ofh(finish_file);
  if (finish_ofh.is_open()) {
    finish_ofh << info_;
    finish_ofh.close();
    // upload to OSS if tool specified
    UploadFile(finish_file, type, -1, background);
  }
}

void TaoCompInfoCollector::_FlushUploadFiles(
    const std::list<upload_file_info> &flist) {
  for (auto &&finfo : flist) {
    UploadFile(finfo.local_path, finfo.type, -1, false, finfo.compress);
  }
}

void TaoCompInfoCollector::EnqueUploadFile(const std::string &local_file,
                                           const std::string &type,
                                           bool compress) {
  if (opts_.dump_level == 0)
    return;
  mutex_lock l(upload_list_mtx_);
  upload_file_list_.emplace_back(upload_file_info{local_file, type, compress});
}

void TaoCompInfoCollector::UpdatePerfStats(
    TaoCompilationCache *cur, int64 total_elapsed_us, int64 total_saved_us,
    int64 print_cycle_sec, double speed_up_all, double speed_up_recent,
    bool last_call) {
  if (opts_.dump_level == 0)
    return;
  bool upload = false;
  mutex_lock l(obj_id_mtx_);
  auto it = cache_obj_id_.find(cur);
  if (it == cache_obj_id_.end()) {
    return;
  }
  it->second.total_elapsed_us = total_elapsed_us;
  it->second.total_saved_us = total_saved_us;
  it->second.print_cycle_sec = print_cycle_sec;
  it->second.speed_up_all = speed_up_all;
  it->second.estimated_speedups.push_back(speed_up_recent);
  if (opts_.intermediate_upload_interval > 0) {
    ++(it->second.interval_count);
    if (it->second.interval_count >= opts_.intermediate_upload_interval) {
      it->second.interval_count = 0;
      upload = true;
    }
  }
  if (upload || last_call) {
    _UploadPerfFile(it->second, last_call);
  }
}

void TaoCompInfoCollector::_UploadPerfFile(TaoSessPerfInfo &sinfo,
                                           bool last_call) {
  if (last_call && sinfo.finalized) {
    // this session info has been uploaded
    return;
  }

  std::stringstream filename_ss;
  filename_ss << opts_.graph_dump_path << "/sess_" << std::setfill('0')
              << std::setw(2) << sinfo.idx << ".perf";
  if (!last_call) {
    filename_ss << ".intermediate";
  }
  std::vector<std::string> out_files{filename_ss.str()};
  // don't upload again if last intermediate file uploading not finished yet
  if (!last_call && tensorflow::Env::Default()->FilesExist(out_files, NULL)) {
    return;
  }
  std::ofstream out(out_files[0]);
  out << sinfo.total_elapsed_us << "," << sinfo.total_saved_us << ","
      << sinfo.print_cycle_sec << "," << sinfo.speed_up_all << ",";
  for (const auto &v : sinfo.estimated_speedups) {
    out << v << ",";
  }
  out.close();
  if (last_call) {
    UploadFile(out_files[0], "-p");
    sinfo.finalized = true;
  } else {
    // upload intermediate files in the background
    UploadFile(out_files[0], "-p", -1, true);
  }
}

void TaoCompInfoCollector::_FinalizePerfStats() {
  mutex_lock l(obj_id_mtx_);
  for (auto it = cache_obj_id_.begin(); it != cache_obj_id_.end(); ++it) {
    _UploadPerfFile(it->second, true);
  }
}

void TaoCompInfoCollector::AddCompileFailedCase(const std::string &log,
                                                const std::string &input_file,
                                                bool exited, bool signaled,
                                                bool coredump, int code) {
  if (opts_.dump_level == 0)
    return;

  bool upload{false};
  std::stringstream fail_name;
  {
    mutex_lock l(info_mtx_);
    auto filename = PathBaseName(input_file);
    if (coredump) {
      upload = json_obj_add<int>(info_, "compile_fail_coredump_count", 1) <
               opts_.max_upload_core_dump;
      if (upload) {
        info_["compile_fail_coredump_list"].emplace_back(filename);
      }
    } else {
      upload = json_obj_add<int>(info_, "compile_fail_other_count", 1) <
               opts_.max_upload_other_fail;
      if (upload) {
        info_["compile_fail_other_list"].emplace_back(filename);
      }
    }
    fail_name << "compile_fail_detail_";
    if (exited) {
      fail_name << "exit_" << code;
    } else if (signaled) {
      fail_name << "signal_" << code;
      if (coredump) {
        fail_name << "_coredump";
      }
    }
    json_obj_add<int64>(info_, fail_name.str(), 1);
  }

  if (upload) {
    std::string compile_log_file = input_file + "." + fail_name.str() + ".log";
    std::ofstream log_fh(compile_log_file);
    if (log_fh.is_open()) {
      log_fh << log;
      log_fh.close();
      UploadFile(compile_log_file, "-c", -1, true, true);
      std::remove(compile_log_file.c_str());
    }
    UploadFile(input_file, "-c", -1, true, true);
  }
}

void TaoCompInfoCollector::AddCustomNumValue(const std::string &key,
                                             int64 val) {
  if (opts_.dump_level == 0)
    return;
  mutex_lock l(info_mtx_);
  json_obj_add(info_, key, val);
}

void TaoCompInfoCollector::AddCustomNumValue(
    const std::vector<std::string> &keys, int64 val) {
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
  json_num_add(*p, val);
}

} // namespace tao
} // namespace tensorflow
