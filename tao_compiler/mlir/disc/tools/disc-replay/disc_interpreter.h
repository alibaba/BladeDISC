#ifndef DISC_REPLAY_DISC_INTERPRETER_H_
#define DISC_REPLAY_DISC_INTERPRETER_H_

#include "tensorflow/compiler/mlir/xla/ral/ral_api.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-replay/record_args.h"
#include "tensorflow/compiler/mlir/xla/ral/context/base/cpu/cpu_context_impl.h"
#include "tensorflow/compiler/mlir/xla/ral/context/base/cuda/cuda_context_impl.h"
#include "tensorflow/compiler/decoupling/mlir_compiler.h"

namespace tensorflow {
class Status;
}  // namespace tensorflow

namespace replay {
// function type for compilation executable file
using func_t = void (*)(void**);

class DiscInterpreter {
 public:
  DiscInterpreter();

  tensorflow::Status Compile(const tensorflow::tao::TaoCompilerInput& input);
  // the entrypoint to replay the specified cluster
  tensorflow::Status Run(const std::vector<tensorflow::Tensor>& tensors,
                         const std::vector<std::string>& placements);

 private:

  tensorflow::Status RunExecutable(std::vector<tensorflow::Tensor> tensors, const std::string& executable_fname);

  std::unique_ptr<tao::ral::gpu::BaseCudaExecutionContext> GetExecCUDAContext(const std::string& executable_fname);
  
  tensorflow::Status GetEntryFunc(const std::string& exectuable_fname, func_t* entry_func);

  void* ral_func_ptr_;  
};

} //  namespace replay

#endif  // DISC_REPLAY_DISC_INTERPRETER_H_
