#ifndef TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_BASE_H_
#define TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_BASE_H_

#include "tensorflow/compiler/decoupling/tao_compilation_result.pb.h"
#include "tensorflow/compiler/decoupling/tao_compiler_input.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tao {

class CompilerBase {
 public:
  virtual Status Compile(const tensorflow::tao::TaoCompilerInput& input,
                         const string& output_file) = 0;

  using CompilerFactory = std::function<std::unique_ptr<CompilerBase>()>;

  static void RegisterCompilerFactory(DeviceType dt, CompilerFactory factory);

  static StatusOr<CompilerBase*> GetCompilerForDevice(DeviceType dt);
};

}  //  namespace tao
}  //  namespace tensorflow
#endif  //  TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_BASE_H_