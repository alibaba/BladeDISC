#ifndef DISC_REPLAY_TAR_HELPER_H_
#define DISC_REPLAY_TAR_HELPER_H_

#include <sstream>
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/framework/types.h"

namespace replay {

tensorflow::Status DeCompressTarGz(const std::string& tar_fname, const std::string& out_dir) {
  //tar xf cluster_123.tar -C /tmp/
  tensorflow::SubProcess proc;
  std::string stdout;
  std::string stderr;
  std::cout << tar_fname << "\t" << out_dir << std::endl;
  std::vector<std::string> args = {"tar", "xf", tar_fname, "-C " + out_dir + "/"};
  proc.SetProgram("/bin/tar", {"tar", "xf", tar_fname, "-C", out_dir});
  proc.SetChannelAction(tensorflow::CHAN_STDOUT, tensorflow::ACTION_PIPE);
  proc.SetChannelAction(tensorflow::CHAN_STDERR, tensorflow::ACTION_PIPE);

  if(!proc.Start()) {
    return tensorflow::errors::Internal("failed to run tar command");
  }

  if (proc.Communicate(
      /*stdin_input=*/nullptr, &stdout, &stderr) != 0) {
    std::stringstream ss;
    ss << "failed to decompress tar.\nstdout:\n" << stdout << "stderr:\n" << stderr;
    VLOG(0) <<  ss.str();
    return tensorflow::errors::Internal(ss.str());
  }
  return tensorflow::Status::OK();
}


tensorflow::Status CompressTarGz(const std::string& src, const std::string& dest) {
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
    ss << "failed to compress tar.\nstdout:\n" << stdout << "stderr:\n" << stderr;
    VLOG(0) << ss.str();
    return tensorflow::errors::Internal(ss.str());
  }
  return tensorflow::Status::OK();
}

}

#endif