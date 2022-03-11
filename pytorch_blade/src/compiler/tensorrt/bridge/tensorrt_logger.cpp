// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorrt_logger.h"

#include <cstdlib>
#include "common_utils/logging.h"

namespace torch {
namespace blade {
TensorrtLogger::TensorrtLogger() : log_level_(Severity::kERROR) {
  const auto trt_log_lvl_cstr = std::getenv("TORCH_BLADE_TRT_LOG_LEVEL");
  if (trt_log_lvl_cstr == nullptr) {
    return;
  }
  log_enable_ = true;
  std::string trt_log_lvl_str = std::string(trt_log_lvl_cstr);
  std::for_each(trt_log_lvl_str.begin(), trt_log_lvl_str.end(), [](char& c) {
    c = ::toupper(c);
  });
  if (trt_log_lvl_str == "FATAL") {
    log_level_ = Severity::kINTERNAL_ERROR;
  } else if (trt_log_lvl_str == "ERROR") {
    log_level_ = Severity::kERROR;
  } else if (trt_log_lvl_str == "WARNING") {
    log_level_ = Severity::kWARNING;
  } else if (trt_log_lvl_str == "INFO") {
    log_level_ = Severity::kINFO;
  }
}

void TensorrtLogger::log(Severity severity, const char* msg) noexcept {
  if (!log_enable_) {
    return;
  }
  if (severity > log_level_) {
    return;
  }
  switch (severity) {
    case Severity::kINTERNAL_ERROR:
      LOG(FATAL) << "BUG: " << msg;
      break;
    case Severity::kERROR:
      LOG(ERROR) << msg;
      break;
    case Severity::kWARNING:
      LOG(WARNING) << msg;
      break;
    case Severity::kINFO:
      LOG(INFO) << msg;
      break;
    default:
      DLOG(INFO) << "UNKOWN: " << msg;
      break;
  }
}
// Per TensorRT documentation, logger needs to be a singleton.
TensorrtLogger& GetTensorrtLogger() {
  static TensorrtLogger trt_logger;
  return trt_logger;
}

} // namespace blade
} // namespace torch
