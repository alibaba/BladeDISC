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

#ifndef DISC_REPLAY_TAR_HELPER_H_
#define DISC_REPLAY_TAR_HELPER_H_

#include <sstream>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/subprocess.h"

namespace replay {

tensorflow::Status DeCompressTar(const std::string& tar_fname,
                                 const std::string& out_dir) {
  // tar xf cluster_123.tar -C /tmp/
  tensorflow::SubProcess proc;
  std::string stdout;
  std::string stderr;
  std::vector<std::string> args = {"tar", "xf", tar_fname,
                                   "-C " + out_dir + "/"};
  proc.SetProgram("/bin/tar", {"tar", "xf", tar_fname, "-C", out_dir});
  proc.SetChannelAction(tensorflow::CHAN_STDOUT, tensorflow::ACTION_PIPE);
  proc.SetChannelAction(tensorflow::CHAN_STDERR, tensorflow::ACTION_PIPE);

  if (!proc.Start()) {
    return tensorflow::errors::Internal("failed to run tar command");
  }

  if (proc.Communicate(
          /*stdin_input=*/nullptr, &stdout, &stderr) != 0) {
    std::stringstream ss;
    ss << "failed to decompress tar.\nstdout:\n"
       << stdout << "stderr:\n"
       << stderr;
    VLOG(0) << ss.str();
    return tensorflow::errors::Internal(ss.str());
  }
  return tensorflow::Status::OK();
}

tensorflow::Status CompressTar(const std::string& src,
                               const std::string& dest) {
  // tar cf /tmp/cluster_123.tar -C /tmp/ cluster_123
  std::string basename = src.substr(src.find_last_of('/') + 1);
  std::string ch_dir = src.substr(0, src.size() - basename.size());
  tensorflow::SubProcess proc;
  std::string stdout;
  std::string stderr;
  proc.SetProgram("/bin/tar", {"tar", "cf", dest, "-C", ch_dir, basename});
  proc.SetChannelAction(tensorflow::CHAN_STDOUT, tensorflow::ACTION_PIPE);
  proc.SetChannelAction(tensorflow::CHAN_STDERR, tensorflow::ACTION_PIPE);
  if (!proc.Start()) {
    return tensorflow::errors::Internal("failed to launch tar command");
  }
  if (proc.Communicate(
          /*stdin_input=*/nullptr, &stdout, &stderr) != 0) {
    std::stringstream ss;
    ss << "failed to compress tar.\nstdout:\n"
       << stdout << "stderr:\n"
       << stderr;
    VLOG(0) << ss.str();
    return tensorflow::errors::Internal(ss.str());
  }
  return tensorflow::Status::OK();
}

}  // namespace replay

#endif