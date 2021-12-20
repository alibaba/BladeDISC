#ifndef TENSORFLOW_COMPILER_DECOUPLING_TAO_COMPILER_TRACE_H_
#define TENSORFLOW_COMPILER_DECOUPLING_TAO_COMPILER_TRACE_H_

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace tao {

// Trace events in tao compiler.
class TaoCompilerTrace {
 private:
  TaoCompilerTrace();

 public:
  void OnEvent(std::string&& key, std::string&& value);
  void Shutdown();

  static TaoCompilerTrace* Instance();

 private:
  struct Event {
    tensorflow::uint64 timestamp_us;
    std::string key;
    std::string value;
  };

  std::mutex events_lock_;
  std::vector<Event> events_ TF_GUARDED_BY(
      events_lock_);  // Bookkeeping all the events in memory (for now).
  std::atomic<bool> activated_{false};  // Whether tracing is activated.
  std::string dump_path_ = "";          // File path to dump the events.

  TF_DISALLOW_COPY_AND_ASSIGN(TaoCompilerTrace);
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_DECOUPLING_TAO_COMPILER_TRACE_H_