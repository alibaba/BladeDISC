#include "tao_bridge/kernels/profiling.h"

#include "tao_bridge/errors.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {
namespace tao {

Status GpuTFProfiler::Start(OpKernelContext* ctx) {
  // TODO: This part of code currently only works for gpu device
  TF_RET_CHECK(ctx->op_device_context() != nullptr);
  stream_ = ctx->op_device_context()->stream();
  TF_RET_CHECK(stream_ != nullptr);

  timer_.reset(new se::Timer(stream_->parent()));
  stream_->InitTimer(timer_.get()).ThenStartTimer(timer_.get());
  return Status::OK();
}

Status GpuTFProfiler::RecordComputationStart(OpKernelContext* ctx) {
  // TODO: This part of code currently only works for gpu device
  TF_RET_CHECK(stream_ != nullptr);

  comp_timer_.reset(new se::Timer(stream_->parent()));
  stream_->InitTimer(comp_timer_.get()).ThenStartTimer(comp_timer_.get());
  return Status::OK();
}

Status GpuTFProfiler::RecordComputationFinish(OpKernelContext* ctx) {
  // TODO: This part of code currently only works for gpu device
  TF_RET_CHECK(stream_ != nullptr);
  TF_RET_CHECK(comp_timer_ != nullptr);
  stream_->ThenStopTimer(comp_timer_.get());
  return Status::OK();
}

void GpuTFProfiler::Stop(Status status) {
  stream_->ThenStopTimer(timer_.get());
  stream_->BlockHostUntilDone();
  int64 time_in_us = timer_->Microseconds();
  int64 comp_time_in_us = comp_timer_->Microseconds();
  VLOG(2) << "TF time_in_us: " << time_in_us;
  cb_(status, time_in_us, comp_time_in_us);
}

Status CpuTFProfiler::Start(OpKernelContext* ctx) {
  TF_RET_CHECK(ctx->op_device_context() == nullptr);
  timer_start_ = clock::now();
  return Status::OK();
}

Status CpuTFProfiler::RecordComputationStart(OpKernelContext* ctx) {
  TF_RET_CHECK(ctx->op_device_context() == nullptr);
  comp_timer_start_ = clock::now();
  return Status::OK();
}

Status CpuTFProfiler::RecordComputationFinish(OpKernelContext* ctx) {
  TF_RET_CHECK(ctx->op_device_context() == nullptr);
  comp_time_in_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
                         clock::now() - comp_timer_start_)
                         .count();
  return Status::OK();
}

void CpuTFProfiler::Stop(Status status) {
  time_in_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
                    clock::now() - timer_start_)
                    .count();
  int64 time_in_us = time_in_us_;
  int64 comp_time_in_us = comp_time_in_us_;

  VLOG(2) << "TF time_in_us: " << time_in_us;
  cb_(status, time_in_us, comp_time_in_us);
}

}  // namespace tao
}  // namespace tensorflow