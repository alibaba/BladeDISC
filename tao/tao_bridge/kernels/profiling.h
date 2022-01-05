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

#ifndef TAO_TAO_BRIDGE_KERNELS_PROFILING_H_
#define TAO_TAO_BRIDGE_KERNELS_PROFILING_H_

#include <functional>

#include "tao_bridge/common.h"
#include "tao_bridge/errors.h"
#include "tao_bridge/tf/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/stream.h"

namespace tensorflow {

class OpKernelContext;

namespace tao {

class GpuTFProfiler;
class CpuTFProfiler;

template <typename T = GpuTFProfiler> class TFProfiler {
public:
  using Callback = std::function<void(Status, int64, int64)>;
  TFProfiler(Callback cb) : cb_(std::move(cb)) {}

  Status Start(OpKernelContext *ctx) {
    return static_cast<T *>(this)->Start(ctx);
  }

  Status RecordComputationStart(OpKernelContext *ctx) {
    return static_cast<T *>(this)->RecordComputationStart(ctx);
  }

  Status RecordComputationFinish(OpKernelContext *ctx) {
    return static_cast<T *>(this)->RecordComputationFinish(ctx);
  }

  void Stop(Status status) { static_cast<T *>(this)->Stop(status); }

protected:
  Callback cb_;
};

class GpuTFProfiler : public TFProfiler<GpuTFProfiler> {
public:
  using Callback = std::function<void(Status, int64, int64)>;
  GpuTFProfiler(Callback cb) : TFProfiler<GpuTFProfiler>(cb) {}

  Status Start(OpKernelContext *ctx);

  Status RecordComputationStart(OpKernelContext *ctx);

  Status RecordComputationFinish(OpKernelContext *ctx);

  void Stop(Status status);

private:
  se::Stream *stream_ = nullptr;
  std::unique_ptr<se::Timer> timer_;
  std::unique_ptr<se::Timer> comp_timer_;
};

class CpuTFProfiler : public TFProfiler<CpuTFProfiler> {
public:
  using Callback = std::function<void(Status, int64, int64)>;
  using clock = std::chrono::high_resolution_clock;
  CpuTFProfiler(Callback cb) : TFProfiler<CpuTFProfiler>(cb) {}

  Status Start(OpKernelContext *ctx);

  Status RecordComputationStart(OpKernelContext *ctx);

  Status RecordComputationFinish(OpKernelContext *ctx);

  void Stop(Status status);

private:
  clock::time_point timer_start_;
  clock::time_point comp_timer_start_;
  int64 time_in_us_;
  int64 comp_time_in_us_;
};

} // namespace tao
} // namespace tensorflow

#endif // TAO_TAO_BRIDGE_KERNELS_PROFILING_H_
