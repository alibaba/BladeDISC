#ifndef DISC_REPLAY_RECORD_ARGS_
#define DISC_REPLAY_RECORD_ARGS_

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/subprocess.h"

#include "tensorflow/compiler/decoupling/tao_compiler_input.pb.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-replay/tar_helper.h"

namespace replay {
using tensorflow::TensorProto;

class ReplayRecord {
 public:
  ReplayRecord() {};

  // Initialize ReplayRecord from a record file with tar.gz foramt.
  tensorflow::Status InitFromTarGz(const std::string& fname);

  std::vector<tensorflow::Tensor> Tensors() {return tensors_; };
  std::vector<std::string> Placements() {return placements_; };

 private:
  std::vector<tensorflow::Tensor> tensors_;
  std::vector<std::string> placements_;
  tensorflow::tao::TaoCompilerInput input_;
};

//std::shared_ptr<ReplayRecord> CreateRecordInputFromTar(const std::string& fname);

} //  namespace replay

#endif // DISC_REPLAY_RECORD_ARGS_