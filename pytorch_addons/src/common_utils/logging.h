#ifndef __COMMON_LOGGING_H__
#define __COMMON_LOGGING_H__

#include "c10/util/Logging.h"

#define DLOG(...)                                       \
  if (std::getenv("TORCH_ADDONS_DEBUG_LOG") != nullptr) \
  LOG(__VA_ARGS__)

#define LOG_ASSERT(condition) FATAL_IF(condition)

#endif
