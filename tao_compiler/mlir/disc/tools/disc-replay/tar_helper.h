#ifndef DISC_REPLAY_TAR_HELPER_H_
#define DISC_REPLAY_TAR_HELPER_H_

#include <sstream>
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/framework/types.h"

namespace replay {

tensorflow::Status DeCompressTarGz(const std::string& fname, std::string* output_fname) {
  tensorflow::SubProcess proc;
  std::string stdout;
  std::string stderr;
  std::string ext_suffix = ".tar.gz";
  size_t ext_pos = fname.find_last_of(ext_suffix);
  if (ext_pos == std::string::npos) {
    return tensorflow::errors::Internal("input file should be tar.gz");
  }
  *output_fname = fname.substr(0, ext_pos);

  std::vector<std::string> args = {"zxf", fname};
  proc.SetProgram("tar", args);
  proc.SetChannelAction(tensorflow::CHAN_STDOUT, tensorflow::ACTION_PIPE);
  proc.SetChannelAction(tensorflow::CHAN_STDERR, tensorflow::ACTION_PIPE);

  if(!proc.Start()) {
    return tensorflow::errors::Internal("failed to run tar command");
  }

  if (proc.Communicate(
      /*stdin_input=*/nullptr, &stdout, &stderr) != 0) {
    std::stringstream ss;
    ss << "failed to decompress tar.\nstdout:\n" << stdout << "stderr:\n" << stderr;
    return tensorflow::errors::Internal(ss.str());
  }
  return tensorflow::Status::OK();
}


tensorflow::Status CompressTarGz(const std::string& src, const std::string& dest) {
  std::string basename = src.substr(src.find_last_of('/') + 1);
  std::string tar_dir = src.substr(0, src.size() - basename.size());
  tensorflow::SubProcess proc;
  std::string stdout;
  std::string stderr;
  // tar czf /tmp/test.tar /tmp/cluster_123 -C /tmp cluster_123
  std::vector<std::string> args = {"czf", dest, "-C", tar_dir, basename};
  proc.SetProgram("tar", args);
  proc.SetChannelAction(tensorflow::CHAN_STDOUT, tensorflow::ACTION_PIPE);
  proc.SetChannelAction(tensorflow::CHAN_STDERR, tensorflow::ACTION_PIPE);
  if (!proc.Start()) {
    return tensorflow::errors::Internal("failed to launch tar command");
  }
  if (proc.Communicate(
      /*stdin_input=*/nullptr, &stdout, &stderr) == 0) {
    std::stringstream ss;
    ss << "failed to compress tar.\nstdout:\n" << stdout << "stderr:\n" << stderr;
    VLOG(0) << ss.str();
    return tensorflow::errors::Internal(ss.str());
  }
  return tensorflow::Status::OK();
}

}

#endif