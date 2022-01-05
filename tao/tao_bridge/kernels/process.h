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

} // namespace tao
} // namespace tensorflow

#endif // TAO_TAO_BRIDGE_KERNELS_TAO_PROCESS_H_
