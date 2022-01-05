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

#include "tao_bridge/kernels/tao_profiling_guided_compilation.h"

#include "tensorflow/core/util/env_var.h"
#include <sys/time.h>

namespace tensorflow {
namespace tao {

bool cmp_candidates(const std::pair<uint64, uint64> &a,
                    const std::pair<uint64, uint64> &b) {
  return a.second > b.second;
}

TAOProfilingGuidedCompilation &TAOProfilingGuidedCompilation::Get() {
  static TAOProfilingGuidedCompilation _obj;
  return _obj;
}

TAOProfilingGuidedCompilation::CompilationMode
TAOProfilingGuidedCompilation::Mode() {
  // VLOG(VLOG_LVL) << "UseTAOProfilingGuidedCompilation " << feature_mode_;
  return feature_mode_;
}

bool TAOProfilingGuidedCompilation::Profiling() { return profiling_; }

// return 0 if exceed kSignatureMapSize
int64 TAOProfilingGuidedCompilation::IncSignatureCalls(uint64 signature) {
  auto iter = signature_calls_.find(signature);
  if (iter == signature_calls_.end()) {
    // ignore if meet too many signature
    if (signature_calls_.size() > kSignatureMapSize) {
      return 0;
    }
    signature_calls_.emplace(signature, 1);
    return 1;
  }
  iter->second++;
  return iter->second;
}

uint64 TAOProfilingGuidedCompilation::GetSignatureCalls(uint64 signature) {
  auto iter = signature_calls_.find(signature);
  if (iter == signature_calls_.end()) {
    return 0;
  } else {
    return iter->second;
  }
}

bool TAOProfilingGuidedCompilation::MeasureTFTimeOrNot(uint64 signature) {
  mutex_lock l(signature_calls_mtx_);
  int64 called = IncSignatureCalls(signature);
  if (called == 0) {
    return false;
  } else {
    if (called == lazy_calls_) {
      VLOG(VLOG_LVL) << "Signature: " << signature << " enable_profiler";
      return true;
    }
  }
  return false;
}

uint64 &TAOProfilingGuidedCompilation::GetSignatureTimeRef(uint64 signature) {
  mutex_lock l(signature_time_mtx_);
  auto iter = signature_time_.find(signature);
  if (iter == signature_time_.end()) {
    signature_time_.emplace(signature, 0);
  }
  return signature_time_[signature];
}

bool TAOProfilingGuidedCompilation::InCandidateList(uint64 signature) {
  mutex_lock l(candidate_signature_mtx_);
  auto iter = candidate_signature_.find(signature);
  if (iter == candidate_signature_.end()) {
    return false;
  }
  return true;
}

uint64 TAOProfilingGuidedCompilation::GetCurTimeUs() {
  timeval tv;
  gettimeofday(&tv, nullptr);
  return uint64(tv.tv_sec) * 1e6 + tv.tv_usec;
}

TAOProfilingGuidedCompilation::TAOProfilingGuidedCompilation() {
  auto *bridge_opt = GetTaoBridgeOptions();
  feature_mode_ = static_cast<CompilationMode>(
      bridge_opt->profiling_guided_compilation_mode);
  lazy_calls_ = bridge_opt->profiling_guided_compilation_lazy_calls;
  if (feature_mode_ == CompilationMode::kProfilingGuided) {
    profiling_time_by_min_ =
        bridge_opt->profiling_guided_compilation_profiling_time_by_min;
    candidate_size_ = bridge_opt->profiling_guided_compilation_candidates;
    profiling_ = true;
    init_timestamp_ = GetCurTimeUs();
  } else {
    profiling_ = false;
  }
}

void TAOProfilingGuidedCompilation::GenCandidatesOrNot() {
  mutex_lock l(candidate_signature_mtx_);
  mutex_lock lc(signature_calls_mtx_);
  mutex_lock lt(signature_time_mtx_);

  uint64 profiled_time_in_mins = ProfiledTimeInMins();
  if (profiled_time_in_mins >= profiling_time_by_min_) {
    std::vector<std::pair<uint64, uint64>> candidates;
    for (auto signature_time : signature_time_) {
      uint64 calls = GetSignatureCalls(signature_time.first);
      // VLOG(VLOG_LVL) << "signature " <<signature_time.first << " " <<
      // signature_time.second << "us. Calls " << calls;
      std::pair<uint64, uint64> signature_time_pair =
          std::make_pair(signature_time.first, signature_time.second * calls);
      candidates.push_back(signature_time_pair);
    }
    if (candidates.size() <= candidate_size_) {
      feature_mode_ = CompilationMode::kNormal;
      VLOG(VLOG_LVL)
          << "CompilationMode fallback to normal as profiled candidates "
          << candidates.size() << " < " << candidate_size_;
      return;
    }
    sort(candidates.begin(), candidates.end(), cmp_candidates);
    for (auto signature_time : candidates) {
      VLOG(VLOG_LVL) << signature_time.first << " " << signature_time.second
                     << " us. target_signature size "
                     << candidate_signature_.size();
      candidate_signature_.insert(signature_time.first);
      if (candidate_signature_.size() > candidate_size_) {
        break;
      }
    }
    profiling_ = false;
  }
}

bool TAOProfilingGuidedCompilation::AllowTAOCompile(uint64 signature) {
  mutex_lock l(signature_calls_mtx_);
  uint64 called = IncSignatureCalls(signature);
  if (called == 0) {
    return false;
  }
  if (called >= lazy_calls_) {
    VLOG(VLOG_LVL) << "Signature " << signature << " called " << called
                   << " times. Allow to compile.";
    return true;
  }
  return false;
} // namespace tao

uint64 TAOProfilingGuidedCompilation::ProfiledTimeInMins() {
  uint64 cur_timestamp = GetCurTimeUs();
  uint64 profiled_time_in_mins =
      (cur_timestamp - init_timestamp_) / 1000000 / 60;
  return profiled_time_in_mins;
}

TAOProfilingGuidedCompilation::~TAOProfilingGuidedCompilation() {}

} // namespace tao
} // namespace tensorflow
