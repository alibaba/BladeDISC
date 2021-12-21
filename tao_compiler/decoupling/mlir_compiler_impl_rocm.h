#ifndef TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_IMPL_ROCM_H_
#define TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_IMPL_ROCM_H_

#include <memory>

#include "tensorflow/compiler/decoupling/mlir_compiler.h"

namespace tensorflow {

namespace tao {

class CompilerMLIR_DCU : public CompilerMLIR {
 public:
  CompilerMLIR_DCU();
  ~CompilerMLIR_DCU();

 private:
  Status Init(const TaoCompilerInput& input,
              const string& output_file) override;

  std::string DefaultDevice() override;

  Status FillDeviceInfo(mlir::disc_ral::DISCLoweringOptions& options) override;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_DECOUPLING_DHLO_COMPILER_IMPL_ROCM_H_
