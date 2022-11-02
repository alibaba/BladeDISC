#ifndef MLIR_DISC_UTILS_SOURCE_EMITTER
#define MLIR_DISC_UTILS_SOURCE_EMITTER

#include <string>

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace disc_ral {

class SourceEmitterCUDA {
 public:
  using ValueNameBinding = llvm::DenseMap<Value, std::string>;

 public:
  static bool isSupportedOp(Operation* op);

  llvm::Optional<std::string> EmitOp(Operation* op, ValueNameBinding& binding);

  llvm::Optional<std::string> EmitElemWiseUnaryOp(Operation* op,
                                                  ValueNameBinding& binding);

  llvm::Optional<std::string> EmitElemWiseBinaryOp(Operation* op,
                                                   ValueNameBinding& binding);

  llvm::Optional<std::string> EmitElemWiseTernaryOp(Operation* op,
                                                    ValueNameBinding& binding);

  llvm::Optional<std::string> EmitScalarOrSplatConstantOp(
      Operation* op, ValueNameBinding& binding);

  llvm::Optional<std::string> EmitBroadcastOfScalarOrSplatConstantOp(
      Operation* op, ValueNameBinding& binding);

  void bindValueNames(const SmallVectorImpl<Value>& inputs,
                      const SmallVectorImpl<std::string>& names,
                      ValueNameBinding& binding);

 private:
  std::unordered_map<std::string, int32_t> existing_names_;

 private:
  std::string EmitUniqueName(llvm::StringRef op_str);
  llvm::Optional<std::string> EmitScalarOrSplatConstantExpression(
      lmhlo::ConstantOp constant);
};

}  // namespace disc_ral
}  // namespace mlir

#endif  // MLIR_DISC_UTILS_SOURCE_EMITTER