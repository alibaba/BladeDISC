#ifndef TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_H_
#define TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_H_

#include "mlir/IR/BuiltinOps.h"
#include "tensorflow/compiler/decoupling/compiler_base.h"
#include "tensorflow/compiler/mlir/disc/disc_compiler.h"

namespace llvm {
class InitLLVM;
}  // namespace llvm

namespace llvm {}

namespace tensorflow {
namespace tao {

class CompilerMLIR : public tensorflow::tao::CompilerBase {
 public:
  explicit CompilerMLIR();
  virtual ~CompilerMLIR();

  virtual Status Compile(const TaoCompilerInput& input,
                         const string& output_file);

 protected:
  virtual std::string DefaultDevice() = 0;
  virtual Status Init(const TaoCompilerInput& input, const string& output_file);

  virtual Status ConvertToMlir(const TaoCompilerInput& input,
                               const string& output_file);

  virtual Status CompileMlirToExecutable(const TaoCompilerInput& input,
                                         const string& output_file);

  virtual Status FillDeviceInfo(mlir::disc_ral::DISCLoweringOptions& options);

  CompilationResultProto result_proto_;
  std::unique_ptr<mlir::MLIRContext> context_;
  mlir::OwningModuleRef module_;
  std::unique_ptr<llvm::InitLLVM> llvm_init_;
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_H_