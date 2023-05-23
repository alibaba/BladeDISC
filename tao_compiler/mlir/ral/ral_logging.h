//===- ral_logging.h ----------------------===//
//
// Copyright 2020 The PAI Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef RAL_RAL_LOGGING_H_
#define RAL_RAL_LOGGING_H_

#include <atomic>
#include <cassert>
#include <limits>
#include <memory>
#include <sstream>

namespace tao {
namespace ral {

const int INFO = 0;            // base_logging::INFO;
const int WARNING = 1;         // base_logging::WARNING;
const int ERROR = 2;           // base_logging::ERROR;
const int FATAL = 3;           // base_logging::FATAL;
const int NUM_SEVERITIES = 4;  // base_logging::NUM_SEVERITIES;

namespace internal {

#define TAO_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define TAO_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, int severity);
  ~LogMessage() override;

  // Change the location of the log message.
  LogMessage& AtLocation(const char* fname, int line);

  // Returns the minimum log level for VLOG statements.
  // E.g., if MinVLogLevel() is 2, then VLOG(2) statements will produce output,
  // but VLOG(3) will not. Defaults to 0.
  static int MinVLogLevel();

  // Returns whether VLOG level lvl is activated for the file fname.
  //
  // E.g. if the environment variable TF_CPP_VMODULE contains foo=3 and fname is
  // foo.cc and lvl is <= 3, this will return true. It will also return true if
  // the level is lower or equal to TF_CPP_MIN_VLOG_LEVEL (default zero).
  //
  // It is expected that the result of this query will be cached in the VLOG-ing
  // call site to avoid repeated lookups. This routine performs a hash-map
  // access against the VLOG-ing specification provided by the env var.
  static bool VmoduleActivated(const char* fname, int level);

 protected:
  void GenerateLogMessage();

 private:
  const char* fname_;
  int line_;
  int severity_;
};

// Uses the lower operator & precedence to voidify a LogMessage reference, so
// that the ternary VLOG() implementation is balanced, type wise.
struct Voidifier {
  // clang-format off
  template <typename T>
  void operator&(const T&) const {}
  // clang-format on
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal() override;
};

// LogMessageNull supports the DVLOG macro by simply dropping any log messages.
class LogMessageNull : public std::basic_ostringstream<char> {
 public:
  LogMessageNull() {}
  ~LogMessageNull() override {}
};

}  // namespace internal

#define _TAO_LOG_INFO \
  ::tao::ral::internal::LogMessage(__FILE__, __LINE__, ::tao::ral::INFO)
#define _TAO_LOG_WARNING \
  ::tao::ral::internal::LogMessage(__FILE__, __LINE__, ::tao::ral::WARNING)
#define _TAO_LOG_ERROR \
  ::tao::ral::internal::LogMessage(__FILE__, __LINE__, ::tao::ral::ERROR)
#define _TAO_LOG_FATAL ::tao::ral::internal::LogMessageFatal(__FILE__, __LINE__)

#define TAO_LOG(severity) _TAO_LOG_##severity

#define TAO_VLOG_IS_ON(lvl)                                               \
  (([](int level, const char* fname) {                                    \
    static const bool vmodule_activated =                                 \
        ::tao::ral::internal::LogMessage::VmoduleActivated(fname, level); \
    return vmodule_activated;                                             \
  })(lvl, __FILE__))

#define TAO_VLOG(level)                                        \
  TAO_PREDICT_TRUE(!TAO_VLOG_IS_ON(level))                     \
  ? (void)0                                                    \
  : ::tao::ral::internal::Voidifier() &                        \
          ::tao::ral::internal::LogMessage(__FILE__, __LINE__, \
                                           ::tao::ral::INFO)

#define TAO_CHECK(expr) \
  while (!(expr)) TAO_LOG(FATAL) << "TAO_CHECK failed: "

}  // namespace ral
}  // namespace tao

#endif  // RAL_RAL_LOGGING_H_