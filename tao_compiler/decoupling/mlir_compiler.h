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

#ifndef TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_H_
#define TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_H_

#include "decoupling/compiler_base.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/disc/disc_compiler.h"

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
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::unique_ptr<llvm::InitLLVM> llvm_init_;
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_DECOUPLING_MLIR_COMPILER_H_