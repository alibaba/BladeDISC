#ifndef TAO_TAO_BRIDGE_KERNELS_TAO_PROCESS_H_
#define TAO_TAO_BRIDGE_KERNELS_TAO_PROCESS_H_

#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace tensorflow {
namespace tao {

// Process provides helper functions on Linux process
class Process {
 public:
  Process() : pid_(getpid()){};
  Process(pid_t pid) : pid_(pid){};

  // returns the time of last modification
  time_t StartTime();

  // returns the start time of process with timespec struct
  timespec StartTimeTS();

  // returns the start time of process with microseconds
  uint64_t StartTimeUS();

  // returns the command line of process
  std::vector<std::string> CMD();

  // returns the current workspace director of process
  std::string CWD();

  // return the process ID
  inline pid_t Pid() { return pid_; }

 private:
  pid_t pid_;
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_KERNELS_TAO_PROCESS_H_
