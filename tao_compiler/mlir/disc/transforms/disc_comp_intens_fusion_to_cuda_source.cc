// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/utils/source_emitter.h"

namespace mlir {
namespace disc_ral {
namespace {

constexpr const char* kSpecializedClassName = "SpecializedGemmFusion";
constexpr const char* kSpecializedEpilogue = "SpecializedEpilogue";
constexpr const char* kEpilogueIsHeavy = "EpilogueIsHeavy";

constexpr const char* kGRank = "GRank";
constexpr const char* kElementAType = "ElementAType";
constexpr const char* kElementALayout = "ElementALayout";
constexpr const char* kElementBType = "ElementBType";
constexpr const char* kElementBLayout = "ElementBLayout";
constexpr const char* kElementOutputType = "ElementOutputType";
constexpr const char* kElementOutputLayout = "ElementOutputLayout";
constexpr const char* kElementAccumulatorType = "ElementAccumulatorType";
constexpr const char* kOperatorClassType = "OperatorClassType";
constexpr const char* kSMArch = "SMArch";

constexpr const char* kScaleKind = "EpilogueScaleKind";
constexpr const char* kCountVectorized = "EpilogueCountVectorized";
constexpr const char* kEpilogueType = "EpilogueElementType";

constexpr const char* kGatherA = "IsGatherA";
constexpr const char* kGatherB = "IsGatherB";
constexpr const char* kScatterD = "IsScatterD";
constexpr const char* kPermuteDLayout = "EpiloguePermuteDLayout";

constexpr const char* kParameterPermute = "ParameterPermute";

constexpr const char* kGemmFusionFuncName = "gemmFusionFunc";

int64_t stringReplaceInplace(std::string& subject, const std::string& oldsub,
                             const std::string& newsub, bool replace_all) {
  if (oldsub.empty()) {
    return 0;
  }
  int64_t count = 0;
  size_t pos = 0;
  while ((pos = subject.find(oldsub, pos)) != std::string::npos) {
    subject.replace(pos, oldsub.size(), newsub);
    count++;
    pos += newsub.size();
    if (!replace_all) {
      break;
    }
  }
  return count;
}

struct DiscCompIntensFusionToCUDASourcePass
    : public DiscCompIntensFusionToCUDASourcePassBase<
          DiscCompIntensFusionToCUDASourcePass> {
 public:
  DiscCompIntensFusionToCUDASourcePass() = delete;
  explicit DiscCompIntensFusionToCUDASourcePass(int cc_major, int cc_minor) {
    cc_major_ = cc_major;
    cc_minor_ = cc_minor;
  }
  void runOnOperation() override;

 private:
  int cc_major_;
  int cc_minor_;

 private:
  bool generateCUDASourceForCompIntensFusionFunc(func::FuncOp func);

  bool isHeavyEpilogue(func::FuncOp func);
  bool getCUDATypeString(Type type, std::string& type_str);
  bool getLayoutStrings(const mhlo::DotDimensionNumbersAttr& dimension_numbers,
                        SmallVector<std::string>& layout);
  bool getAccumulatorTypeString(Type output_type,
                                std::string& accumulator_type_str);
  bool getOperatorClassTypeString(std::string& operator_class_type);
  bool getSMArchString(std::string& sm_arch);
  bool getScaleKindString(std::string& scale_kind);
  bool getCountVectorizedString(const std::string& element_output_type,
                                std::string& count_vectorized);
  bool isGatherA(func::FuncOp func);
  bool isGatherB(func::FuncOp func);
  bool isScatterD(func::FuncOp func);
  bool getPermuteDLayoutString(func::FuncOp func,
                               std::string& permute_d_layout);

  void getResults(func::FuncOp func, SmallVector<Value>& results);
  void getEffectiveOperands(func::FuncOp func, SmallVector<Value>& operands);

  void mayConvertAndBindInput(Value input, std::string orig_name,
                              SourceEmitterCUDA::ValueNameBinding& binding);
  bool mayConvertCutlassTypeToCUDAType(
      Value value, std::string old_name, std::string& new_name,
      SmallVectorImpl<std::string>& convert_instructions);
  bool mayConvertCUDATypeToCutlassType(
      Value value, std::string old_name, std::string& new_name,
      SmallVectorImpl<std::string>& convert_instructions);
};

bool DiscCompIntensFusionToCUDASourcePass::isHeavyEpilogue(func::FuncOp func) {
  // TODO: check whether it is heavy or not according to the logic in CUTLASS.
  // return false;
  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::getCUDATypeString(
    Type type, std::string& type_str) {
  if (type.isInteger(1)) {
    type_str = "cutlass::uint1b_t";
  } else if (type.isSignedInteger(8)) {
    type_str = "cutlass::int8b_t";
  } else if (type.isUnsignedInteger(8)) {
    type_str = "cutlass::uint8b_t";
  } else if (type.isSignedInteger(32)) {
    type_str = "cutlass::int32b_t";
  } else if (type.isUnsignedInteger(32)) {
    type_str = "cutlass::uint32b_t";
  } else if (type.isBF16()) {
    type_str = "cutlass::bfloat16_t";
  } else if (type.isF16()) {
    type_str = "cutlass::half_t";
  } else if (type.isF32()) {
    type_str = "float";
  } else if (type.isF64()) {
    type_str = "double";
  } else {
    return false;
  }
  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::getLayoutStrings(
    const mhlo::DotDimensionNumbersAttr& dimension_numbers,
    SmallVector<std::string>& layout) {
  auto lhs_contracting_dims = dimension_numbers.getLhsContractingDimensions();
  auto rhs_contracting_dims = dimension_numbers.getRhsContractingDimensions();
  // Only deal with dot with 1 contracting dim.
  if (lhs_contracting_dims.size() != 1 || rhs_contracting_dims.size() != 1) {
    return false;
  }

  auto lhs_batching_dims = dimension_numbers.getLhsBatchingDimensions();
  auto rhs_batching_dims = dimension_numbers.getRhsBatchingDimensions();
  for (int64_t i = 0; i < lhs_batching_dims.size(); i++) {
    // Only deal with dot whose batch dims are the significant dims.
    if (lhs_batching_dims[i] > lhs_batching_dims.size() ||
        rhs_batching_dims[i] > rhs_batching_dims.size()) {
      return false;
    }
  }
  if (lhs_contracting_dims[0] == lhs_batching_dims.size() + 1) {
    layout.emplace_back("cutlass::layout::RowMajor");
  } else {
    layout.emplace_back("cutlass::layout::ColumnMajor");
  }

  if (rhs_contracting_dims[0] == rhs_batching_dims.size()) {
    layout.emplace_back("cutlass::layout::RowMajor");
  } else {
    layout.emplace_back("cutlass::layout::ColumnMajor");
  }

  // According to https://www.tensorflow.org/xla/operation_semantics#dotgeneral,
  // the result layout follows the order of [batch0, batch1, .., m, n].
  layout.emplace_back("cutlass::layout::RowMajor");

  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::getAccumulatorTypeString(
    Type type, std::string& accumulator_type_str) {
  if (type.isSignedInteger() && type.getIntOrFloatBitWidth() <= 32) {
    accumulator_type_str = "cutlass::int32b_t";
  } else if (type.isUnsignedInteger() && type.getIntOrFloatBitWidth() <= 32) {
    accumulator_type_str = "cutlass::uint32b_t";
  } else if (type.isBF16() || type.isF16() || type.isF32()) {
    accumulator_type_str = "float";
  } else if (type.isF64()) {
    accumulator_type_str = "double";
  } else {
    return false;
  }
  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::getOperatorClassTypeString(
    std::string& operator_class_type) {
  // Currently we only support tensor op.
  operator_class_type = "cutlass::arch::OpClassTensorOp";
  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::getSMArchString(
    std::string& sm_arch) {
  if (cc_major_ == 8) {
    sm_arch = "cutlass::arch::Sm80";
  } else if (cc_major_ == 7) {
    if (cc_minor_ == 5) {
      sm_arch = "cutlass::arch::Sm75";
    } else {
      sm_arch = "cutlass::arch::Sm70";
    }
  } else {
    return false;
  }
  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::getScaleKindString(
    std::string& scale_kind) {
  // TODO: update according to the problem.
  // scale_kind = "cutlass::epilogue::thread::ScaleType::NoBetaScaling";
  scale_kind = "cutlass::epilogue::thread::ScaleType::Nothing";
  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::getCountVectorizedString(
    const std::string& element_output_type, std::string& count_vectorized) {
  count_vectorized =
      "128 / cutlass::sizeof_bits<" + element_output_type + ">::value";
  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::isGatherA(func::FuncOp func) {
  // TODO: update according to the problem.
  return false;
}

bool DiscCompIntensFusionToCUDASourcePass::isGatherB(func::FuncOp func) {
  // TODO: update according to the problem.
  return false;
}

bool DiscCompIntensFusionToCUDASourcePass::isScatterD(func::FuncOp func) {
  // TODO: update according to the problem.
  return false;
}

bool DiscCompIntensFusionToCUDASourcePass::getPermuteDLayoutString(
    func::FuncOp func, std::string& permute_d_layout) {
  // TODO: update according to the problem.
  permute_d_layout = "cutlass::layout::NoPermute";
  return true;
}

void DiscCompIntensFusionToCUDASourcePass::getResults(
    func::FuncOp func, SmallVector<Value>& results) {
  DenseSet<Operation*> op_set;
  auto ops = func.getRegion().getOps();
  for (auto& op : func.getRegion().getOps()) {
    op_set.insert(&op);
  }

  results.clear();
  func.walk([&](Operation* op) {
    int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
    for (Value v : op->getOperands().drop_front(num_input_operand)) {
      bool has_internal_user = false;
      for (Operation* user : getValueUsers(v)) {
        if (op == user) {
          continue;
        }
        has_internal_user |= op_set.contains(user);
      }
      if (!has_internal_user) {
        results.push_back(v);
      }
    }
  });
}

void DiscCompIntensFusionToCUDASourcePass::getEffectiveOperands(
    func::FuncOp func, SmallVector<Value>& operands) {
  DenseSet<Operation*> op_set;
  auto ops = func.getRegion().getOps();
  for (auto& op : func.getRegion().getOps()) {
    op_set.insert(&op);
  }

  operands.clear();
  func.walk([&](Operation* op) {
    SmallVector<Value> ins;
    if (isa<lmhlo::ConstantOp>(op)) {
      // The constant op is splat constant, which should be checked in the
      // fusion pass.
      return;
    } else {
      // The broadcast-in-dim reads constant op, which should be check in the
      // fusion pass.
      int num_input_operand =
          isa<lmhlo::DynamicBroadcastInDimOp>(op)
              ? 1
              : op->getNumOperands() - getNumResultOperands(op);
      for (Value v : op->getOperands().take_front(num_input_operand)) {
        bool has_internal_writter = false;
        for (Operation* user : getValueUsers(v)) {
          if (op == user) {
            continue;
          }
          if (op_set.contains(user) && IsOpWriteValue(user, v)) {
            has_internal_writter = true;
            break;
          }
        }
        if (!has_internal_writter) {
          operands.emplace_back(v);
        }
      }
    }
  });
}

bool DiscCompIntensFusionToCUDASourcePass::mayConvertCutlassTypeToCUDAType(
    Value value, std::string old_name, std::string& new_name,
    SmallVectorImpl<std::string>& convert_instructions) {
  convert_instructions.clear();
  auto type = value.getType().dyn_cast<MemRefType>().getElementType();
  if (type.isF32() || type.isF64()) {
    new_name = old_name;
  } else if (type.isF16()) {
    // CUTLASS type is 'cutlass::half_t'.
    new_name = old_name + "_to_cuda_half";
    convert_instructions.push_back("half " + new_name + " = " + old_name +
                                   ".to_half()");
  } else {
    assert(false && "unsupported type for data conversion");
    return false;
  }

  // TODO: support the following types:
  // cutlass::uint1b_t, cutlass::int8b_t, cutlass::uint8b_t, cutlass::int32b_t,
  // cutlass::uint32b_t, cutlass::bfloat16_t

  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::mayConvertCUDATypeToCutlassType(
    Value value, std::string old_name, std::string& new_name,
    SmallVectorImpl<std::string>& convert_instructions) {
  convert_instructions.clear();
  auto type = value.getType().dyn_cast<MemRefType>().getElementType();
  if (type.isF32() || type.isF64()) {
    new_name = old_name;
  } else if (type.isF16()) {
    // CUTLASS type is 'cutlass::half_t'.
    new_name = old_name + "_to_cutlass_half";
    convert_instructions.push_back("cutlass::half_t " + new_name +
                                   " = cutlass::half_t(" + old_name + ")");
  } else {
    assert(false && "unsupported type for data conversion");
    return false;
  }

  // TODO: support the following types:
  // cutlass::uint1b_t, cutlass::int8b_t, cutlass::uint8b_t, cutlass::int32b_t,
  // cutlass::uint32b_t, cutlass::bfloat16_t

  return true;
}

bool DiscCompIntensFusionToCUDASourcePass::
    generateCUDASourceForCompIntensFusionFunc(func::FuncOp func) {
  std::string cuda_code =
#include "tensorflow/compiler/mlir/disc/utils/gemm_fusion_linear_base_template.hpp"
      ;

  // Values to obtain according to func.
  std::string specialized_class_name;
  std::string specialized_epilogue;
  std::string epilogue_is_heavy;
  std::string gemm_rank;
  std::string element_a_type;
  std::string element_a_layout;
  std::string element_b_type;
  std::string element_b_layout;
  std::string element_output_type;
  std::string element_output_layout;
  std::string element_accumulator_type;
  std::string operator_class_type;
  std::string sm_arch;
  std::string scale_kind;
  std::string count_vectorized;
  std::string epilogue_type;
  std::string gather_a;
  std::string gather_b;
  std::string scatter_d;
  std::string permute_d_layout;
  std::string gemm_fusion_func_name;

  gemm_fusion_func_name = func.getName();
  specialized_class_name = gemm_fusion_func_name + "_specialized";

  specialized_epilogue = "";

  epilogue_is_heavy = isHeavyEpilogue(func) ? "true" : "false";

  SmallVector<Operation*> gemms;
  func->walk([&](lmhlo::DotGeneralOp op) { gemms.push_back(op); });
  // Currently, we only deal with the fusion with a single GEMM.
  assert(gemms.size() == 1 && "GEMM number in kDot fusion is not 1.");
  lmhlo::DotGeneralOp dot = dyn_cast<lmhlo::DotGeneralOp>(gemms[0]);
  Value A = dot->getOperand(0);
  Value B = dot->getOperand(1);
  Value D = dot->getOperand(2);
  auto a_type = A.getType().dyn_cast<MemRefType>();
  auto b_type = B.getType().dyn_cast<MemRefType>();
  auto d_type = D.getType().dyn_cast<MemRefType>();

  gemm_rank = std::to_string(a_type.getRank());

  if (!getCUDATypeString(a_type.getElementType(), element_a_type) ||
      !getCUDATypeString(b_type.getElementType(), element_b_type) ||
      !getCUDATypeString(d_type.getElementType(), element_output_type)) {
    return false;
  }
  SmallVector<std::string> layouts;
  if (!getLayoutStrings(dot.getDotDimensionNumbers(), layouts)) {
    return false;
  }
  element_a_layout = layouts[0];
  element_b_layout = layouts[1];
  element_output_layout = layouts[2];

  if (!getAccumulatorTypeString(d_type.getElementType(),
                                element_accumulator_type) ||
      !getOperatorClassTypeString(operator_class_type) ||
      !getSMArchString(sm_arch) || !getScaleKindString(scale_kind) ||
      !getCountVectorizedString(element_output_type, count_vectorized) ||
      !getPermuteDLayoutString(func, permute_d_layout)) {
    return false;
  }

  // TODO: support more epilogue types.
  epilogue_type = element_output_type;
  gather_a = isGatherA(func) ? "true" : "false";
  gather_b = isGatherB(func) ? "true" : "false";
  scatter_d = isScatterD(func) ? "true" : "false";

  std::string intent = "    ";
  SourceEmitterCUDA source_emitter;
  SourceEmitterCUDA::ValueNameBinding binding;
  for (auto& op : func.getRegion().getOps()) {
    if (isa<lmhlo::DotGeneralOp>(&op)) {
      // Convert CUTLASS datatype to default cuda datatype for computation.
      Value dot_output = op.getOperand(2);
      std::string old_name = "input";
      std::string new_name;
      SmallVector<std::string> convert_instructions;
      if (!mayConvertCutlassTypeToCUDAType(dot_output, old_name, new_name,
                                           convert_instructions)) {
        return false;
      }
      source_emitter.bindValueNames(SmallVector<Value>({dot_output}),
                                    SmallVector<std::string>({new_name}),
                                    binding);
      for (auto& instruction : convert_instructions) {
        specialized_epilogue += intent + instruction + ";\n";
      }
    } else if (!isa<func::ReturnOp>(&op)) {
      assert(source_emitter.isSupportedOp(&op) && "Encounter unsupported op.");
      auto instruction = source_emitter.EmitOp(&op, binding);
      if (!instruction.hasValue()) {
        return false;
      } else {
        specialized_epilogue += intent + instruction.value() + ";\n";
      }
    }
  }
  // Append return instruction.
  SmallVector<Value> results;
  getResults(func, results);
  // Only support one result currently.
  // TODO: support more outputs.
  if (results.size() != 1) {
    return false;
  }
  Value result = results[0];
  std::string old_name = binding[result];
  std::string new_name;
  SmallVector<std::string> convert_result_instructions;
  if (!mayConvertCUDATypeToCutlassType(result, old_name, new_name,
                                       convert_result_instructions)) {
    return false;
  }
  for (auto& instruction : convert_result_instructions) {
    specialized_epilogue += intent + instruction + ";\n";
  }
  specialized_epilogue += intent + "return " + new_name + ";";

  // Create source code op containing the source code string.
  SmallVector<Value> operands;
  getEffectiveOperands(func, operands);

  int param_permute[3];
  for (auto operand : llvm::enumerate(operands)) {
    if (A == operand.value()) {
      param_permute[0] = operand.index();
    } else if (B == operand.value()) {
      param_permute[1] = operand.index();
    }
  }
  // TODO: identify the permutation if there are multiple results.
  param_permute[2] = 2;
  std::string parameter_permute = "{" + std::to_string(param_permute[0]) +
                                  ", " + std::to_string(param_permute[1]) +
                                  ", " + std::to_string(param_permute[2]) + "}";

  // Replace newly generated code in the template.
  stringReplaceInplace(cuda_code, kSpecializedClassName, specialized_class_name,
                       true);
  stringReplaceInplace(cuda_code, kSpecializedEpilogue, specialized_epilogue,
                       true);
  stringReplaceInplace(cuda_code, kEpilogueIsHeavy, epilogue_is_heavy, true);
  stringReplaceInplace(cuda_code, kGRank, gemm_rank, true);
  stringReplaceInplace(cuda_code, kElementAType, element_a_type, true);
  stringReplaceInplace(cuda_code, kElementALayout, element_a_layout, true);
  stringReplaceInplace(cuda_code, kElementBType, element_b_type, true);
  stringReplaceInplace(cuda_code, kElementBLayout, element_b_layout, true);
  stringReplaceInplace(cuda_code, kElementOutputType, element_output_type,
                       true);
  stringReplaceInplace(cuda_code, kElementOutputLayout, element_output_layout,
                       true);
  stringReplaceInplace(cuda_code, kElementAccumulatorType,
                       element_accumulator_type, true);
  stringReplaceInplace(cuda_code, kOperatorClassType, operator_class_type,
                       true);
  stringReplaceInplace(cuda_code, kSMArch, sm_arch, true);
  stringReplaceInplace(cuda_code, kScaleKind, scale_kind, true);
  stringReplaceInplace(cuda_code, kCountVectorized, count_vectorized, true);
  stringReplaceInplace(cuda_code, kEpilogueType, epilogue_type, true);
  stringReplaceInplace(cuda_code, kGatherA, gather_a, true);
  stringReplaceInplace(cuda_code, kGatherB, gather_b, true);
  stringReplaceInplace(cuda_code, kScatterD, scatter_d, true);
  stringReplaceInplace(cuda_code, kPermuteDLayout, permute_d_layout, true);
  stringReplaceInplace(cuda_code, kParameterPermute, parameter_permute, true);
  stringReplaceInplace(cuda_code, kGemmFusionFuncName, gemm_fusion_func_name,
                       true);

  OpBuilder builder(func);
  Location loc = func->getLoc();

  // Replace fun and it's calls with `SourceCodeOp`.
  SmallVector<int32_t> effective_operand_pos(operands.size());
  SmallVector<int32_t> effective_result_pos(results.size());
  for (auto argument : llvm::enumerate(func.getArguments())) {
    auto operand_it = llvm::find(operands, argument.value());
    if (operand_it != operands.end()) {
      effective_operand_pos[operand_it - operands.begin()] = argument.index();
    }
    auto result_it = llvm::find(results, argument.value());
    if (result_it != results.end()) {
      effective_result_pos[result_it - results.begin()] = argument.index();
    }
  }

  auto module_op = func->getParentOfType<ModuleOp>();
  Optional<SymbolTable::UseRange> symbol_uses = func.getSymbolUses(module_op);
  for (SymbolTable::SymbolUse symbol_use : *symbol_uses) {
    Operation* user = symbol_use.getUser();
    auto call = dyn_cast<func::CallOp>(user);
    if (!call) {
      continue;
    }
    SmallVector<Value> operands;
    for (auto idx : effective_operand_pos) {
      operands.push_back(call.getOperand(idx));
    }
    SmallVector<Value> results;
    for (auto idx : effective_result_pos) {
      results.push_back(call.getOperand(idx));
    }
    OpBuilder builder_call(user);
    auto source_code_op = builder_call.create<lmhlo_disc::SourceCodeOp>(
        call->getLoc(), llvm::None, operands, results, cuda_code,
        gemm_fusion_func_name);

    assert(user->getUsers().empty() &&
           "Call of gemm fusion should have no users.");
    user->erase();
  }
  func->erase();

  return true;
}

void DiscCompIntensFusionToCUDASourcePass::runOnOperation() {
  ModuleOp module_op = getOperation();

  SmallVector<func::FuncOp> comp_intens_fusions;
  module_op->walk([&](func::FuncOp func) {
    if (func->getAttrOfType<StringAttr>(kFuncCompIntensFusionAttr)) {
      comp_intens_fusions.push_back(func);
    }
  });

  for (auto func : comp_intens_fusions) {
    if (!generateCUDASourceForCompIntensFusionFunc(func)) {
      signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createDiscCompIntensFusionToCUDASourcePass(int cc_major, int cc_minor) {
  return std::make_unique<DiscCompIntensFusionToCUDASourcePass>(cc_major,
                                                                cc_minor);
}

}  // namespace disc_ral
}  // namespace mlir