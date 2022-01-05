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

#include <unordered_map>
#include <unordered_set>

#include "tao_bridge/common.h"
#include "tao_bridge/tf_compatible.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace tao {

#define VLOG_LVL 2

class TAOProfilingGuidedCompilation final {
public:
  enum class CompilationMode : int64 {
    kNormal = 0,
    kProfilingGuided = 1,
    kLazyOnly = 2,
  };
  static TAOProfilingGuidedCompilation &Get();

  // kNormal,  kProfilingGuided or kLazyOnly
  CompilationMode Mode();

  // profiling has finished
  bool Profiling();

  // measure tf func time in this iter or not
  bool MeasureTFTimeOrNot(uint64 signature);

  // generate top n cacdidates in this iter or not
  void GenCandidatesOrNot();

  // return ref of signature's time record
  uint64 &GetSignatureTimeRef(uint64 signature);

  // can compile this signature or not
  bool InCandidateList(uint64 signature);

  // allow tao compile or not
  bool AllowTAOCompile(uint64 signature);

private:
  // 0. disable feature 1. profiling guided 2. lazy compilation
  CompilationMode feature_mode_ = CompilationMode::kNormal;
  // profiling has finished
  bool profiling_ = true;
  // profiling start time
  uint64 init_timestamp_ = 0;
  // limit signature_calls_ & signature_time_ size
  const uint64 kSignatureMapSize = 50000;
  // stat cluster perf after called n times, similar to lazy compilation
  int64 lazy_calls_;
  // set proifling time (minutes)
  int64 profiling_time_by_min_;
  // set candidate size
  int64 candidate_size_;

  mutex signature_calls_mtx_;
  // record signature's calls
  std::unordered_map<uint64, uint64>
      signature_calls_ GUARDED_BY(signature_calls_mtx_);
  mutex signature_time_mtx_;
  // record signature's time
  std::unordered_map<uint64, uint64>
      signature_time_ GUARDED_BY(signature_time_mtx_);
  mutex candidate_signature_mtx_;
  // record candidates' signature
  std::unordered_set<uint64>
      candidate_signature_ GUARDED_BY(candidate_signature_mtx_);

  // meet a signature and inc its counter
  // return count. return 0 if exceed kSignatureMapSize.
  int64 IncSignatureCalls(uint64 signature);
  // return calls of signature
  uint64 GetSignatureCalls(uint64 signature);
  // return profiled time in minutes
  uint64 ProfiledTimeInMins();
  // get current time
  static uint64 GetCurTimeUs();

  TAOProfilingGuidedCompilation();
  ~TAOProfilingGuidedCompilation();
  TF_DISALLOW_COPY_AND_ASSIGN(TAOProfilingGuidedCompilation);
};

} // namespace tao
} // namespace tensorflow
