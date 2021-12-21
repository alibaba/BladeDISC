#ifndef TAO_TAO_BRIDGE_MLIR_EXECUTABLE_H_
#define TAO_TAO_BRIDGE_MLIR_EXECUTABLE_H_

#include "tao_bridge/executable.h"

namespace tensorflow {
namespace tao {

class MlirExecutable : public Executable {
 public:
  MlirExecutable(const string& compiled_result_file,
                 const string& target_device);
  ~MlirExecutable() override;
  void DumpToFile(const std::string& filename) const override;

  std::string target_device() const override;

 private:
  Status InitImpl(const TaoCompilerResult* result) override;
  Status PreRunProcess(const ExecutableRunOptions& options,
                       BufferAllocations& allocations,
                       std::vector<Tensor>& output_tensors) override;
  Status RunImpl(const ExecutableRunOptions& options,
                 BufferAllocations& allocations) override;
  Status PostRunProcess(const ExecutableRunOptions& options,
                        BufferAllocations& allocations,
                        std::vector<Tensor>& output_tensors) override;

  using MLIR_FUNC_T = void (*)(void**);
  std::string target_device_;
  std::string dso_file_;    // path to DSO.
  void* dso_handle_;        // handle to loaded DSO produced by mlir compiler.
  MLIR_FUNC_T entry_func_;  // handle of entry function in DSO.
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_MLIR_EXECUTABLE_H_
