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

#include "mlir/disc/utils/source_emitter.h"

#include <sstream>

#include "lhlo/IR/lhlo_ops.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/disc_util.h"

namespace mlir {
namespace disc_ral {

using ValueNameBinding = SourceEmitterCUDA::ValueNameBinding;

namespace {

template <typename Type>
std::string to_string(const Type& data) {
  std::ostringstream os;
  os << data;
  return os.str();
}

template <typename OPType>
std::string CUDAMathFuncName(Type type) {
  assert(false && "unsupported op");
  return "";
}

template <>
std::string CUDAMathFuncName<lmhlo::AbsOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "__habs";
  } else if (type.isF32() || type.isF64() || type.isInteger(32) ||
             type.isInteger(64)) {
    return "abs";
  } else {
    assert(false && "unsupported type for abs op.");
  }
  return "";
}

template <>
std::string CUDAMathFuncName<lmhlo::CeilOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hceil";
  } else if (type.isF32() || type.isF64()) {
    return "ceil";
  } else {
    assert(false && "unsupported type for ceil op.");
  }
  return "";
}

template <>
std::string CUDAMathFuncName<lmhlo::ConvertOp>(Type type) {
  if (type.isBF16()) {
    return "__nv_bfloat162";
  } else if (type.isF16()) {
    return "half";
  } else if (type.isF32()) {
    return "float";
  } else if (type.isF64()) {
    return "double";
  } else if (type.isSignedInteger(32)) {
    return "int";
  } else if (type.isUnsignedInteger(32)) {
    return "(unsigned int)";
  } else if (type.isSignedInteger(64)) {
    return "(long int)";
  } else if (type.isUnsignedInteger(64)) {
    return "(unsigned long int)";
  } else {
    assert(false && "unsupported type for convert op.");
  }
  return "";
}

template <>
std::string CUDAMathFuncName<lmhlo::CosineOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hcos";
  } else if (type.isF32() || type.isF64()) {
    return "cos";
  } else {
    assert(false && "unsupported type for cosine op.");
  }
  return "";
}

template <>
std::string CUDAMathFuncName<lmhlo::ExpOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hexp";
  } else if (type.isF32() || type.isF64()) {
    return "exp";
  } else {
    assert(false && "unsupported type for exp op.");
  }
  return "";
}

template <>
std::string CUDAMathFuncName<lmhlo::FloorOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hfloor";
  } else if (type.isF32() || type.isF64()) {
    return "floor";
  } else {
    assert(false && "unsupported type for floor op.");
  }
  return "";
}

template <>
std::string CUDAMathFuncName<lmhlo::IsFiniteOp>(Type type) {
  if (type.isF32() || type.isF64()) {
    return "isfinite";
  } else {
    assert(false && "unsupported type for isfinite op.");
  }
  return "";
}

template <>
std::string CUDAMathFuncName<lmhlo::LogOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hlog";
  } else if (type.isF32() || type.isF64()) {
    return "log";
  } else {
    assert(false && "unsupported type for log op.");
  }
  return "";
}

template <>
std::string CUDAMathFuncName<lmhlo::Log1pOp>(Type type) {
  if (type.isF32() || type.isF64()) {
    return "log1p";
  } else {
    assert(false && "unsupported type for abs op.");
  }
  return "";
}

template <>
std::string CUDAMathFuncName<lmhlo::RsqrtOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hrsqrt";
  } else if (type.isF32() || type.isF64()) {
    return "rsqrt";
  } else {
    assert(false && "unsupported type for rsqrt op.");
  }
  return "";
}

template <>
std::string CUDAMathFuncName<lmhlo::SqrtOp>(Type type) {
  if (type.isBF16() || type.isF16()) {
    return "hsqrt";
  } else if (type.isF32() || type.isF64()) {
    return "sqrt";
  } else {
    assert(false && "unsupported type for sqrt op.");
  }
  return "";
}

template <>
std::string CUDAMathFuncName<lmhlo::TanhOp>(Type type) {
  if (type.isF32() || type.isF64()) {
    return "tanh";
  } else {
    assert(false && "unsupported type for tanh op.");
  }
  return "";
}

std::string MLIRType2CUDATypeStr(Type type) {
  if (type.isBF16()) {
    return "__nv_bfloat162";
  } else if (type.isF16()) {
    return "half";
  } else if (type.isF32()) {
    return "float";
  } else if (type.isF64()) {
    return "double";
  } else if (type.isSignedInteger(1)) {
    return "bool";
  } else if (type.isSignedInteger(32)) {
    return "int32_t";
  } else if (type.isUnsignedInteger(32)) {
    return "uint32_t";
  } else if (type.isIndex()) {
    return "size_t";
  } else if (type.isSignedInteger(64)) {
    return "int64_t";
  } else if (type.isUnsignedInteger(64)) {
    return "uint64_t";
  } else {
    assert(false && "unsupported type for convert op.");
  }
  return "";
}

llvm::Optional<Operation*> findLastWriterInBlock(Value value, Block* block) {
  DenseMap<Value, Operation*> last_writer;
  for (Operation& op : block->getOperations()) {
    int num_input_operand = op.getNumOperands() - getNumResultOperands(&op);
    for (Value v : op.getOperands().drop_front(num_input_operand)) {
      bool inserted = last_writer.try_emplace(v, &op).second;
      (void)inserted;
      assert(inserted);
    }
  }
  auto it = last_writer.find(value);
  if (it != last_writer.end()) {
    return it->second;
  } else {
    return llvm::None;
  }
}

}  // namespace

// TODO:
// template<lmhlo::SignOp>
// template<lmhlo::LogisticOp>

// Rely on nvcc compiler to use fast-math.
llvm::Optional<std::string> SourceEmitterCUDA::EmitElemWiseUnaryOp(
    Operation* op, ValueNameBinding& binding) {
  auto input = op->getOperand(0);
  if (binding.count(input) == 0) {
    return llvm::None;
  }
  std::string input_str = binding[input];

  Type result_type =
      op->getOperand(1).getType().cast<MemRefType>().getElementType();
  std::string type_str = MLIRType2CUDATypeStr(result_type);

  std::string result_name;
  std::string expression;
  if (isa<lmhlo::AbsOp>(op)) {
    result_name = EmitUniqueName("abs");
    expression =
        CUDAMathFuncName<lmhlo::AbsOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::CeilOp>(op)) {
    result_name = EmitUniqueName("ceil");
    expression =
        CUDAMathFuncName<lmhlo::CeilOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::ConvertOp>(op)) {
    result_name = EmitUniqueName("convert");
    expression =
        CUDAMathFuncName<lmhlo::ConvertOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::CosineOp>(op)) {
    result_name = EmitUniqueName("consine");
    expression =
        CUDAMathFuncName<lmhlo::CosineOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::ExpOp>(op)) {
    result_name = EmitUniqueName("exp");
    expression =
        CUDAMathFuncName<lmhlo::ExpOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::FloorOp>(op)) {
    result_name = EmitUniqueName("floor");
    expression =
        CUDAMathFuncName<lmhlo::FloorOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::IsFiniteOp>(op)) {
    result_name = EmitUniqueName("isfinite");
    expression = CUDAMathFuncName<lmhlo::IsFiniteOp>(result_type) + "(" +
                 input_str + ")";
  } else if (isa<lmhlo::LogOp>(op)) {
    result_name = EmitUniqueName("log");
    expression =
        CUDAMathFuncName<lmhlo::LogOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::Log1pOp>(op)) {
    result_name = EmitUniqueName("log1p");
    expression =
        CUDAMathFuncName<lmhlo::Log1pOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::NegOp>(op)) {
    result_name = EmitUniqueName("neg");
    expression = "(-(" + input_str + "))";
  } else if (isa<lmhlo::NotOp>(op)) {
    result_name = EmitUniqueName("not");
    expression = "(!(" + input_str + "))";
  } else if (isa<lmhlo::RsqrtOp>(op)) {
    result_name = EmitUniqueName("rsqrt");
    expression =
        CUDAMathFuncName<lmhlo::RsqrtOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::SqrtOp>(op)) {
    result_name = EmitUniqueName("sqrt");
    expression =
        CUDAMathFuncName<lmhlo::SqrtOp>(result_type) + "(" + input_str + ")";
  } else if (isa<lmhlo::TanhOp>(op)) {
    result_name = EmitUniqueName("tanh");
    if (result_type.isF16()) {
      input_str = "(float(" + input_str + "))";
    }
    expression =
        CUDAMathFuncName<lmhlo::TanhOp>(result_type) + "(" + input_str + ")";
    if (result_type.isF16()) {
      expression = "(half(" + expression + "))";
    }
  }

  assert(binding.count(op->getOperand(1)) == 0);
  binding[op->getOperand(1)] = result_name;

  return type_str + " " + result_name + " = " + expression;
}

llvm::Optional<std::string> SourceEmitterCUDA::EmitElemWiseBinaryOp(
    Operation* op, ValueNameBinding& binding) {
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  if (binding.count(lhs) == 0 || binding.count(rhs) == 0) {
    return llvm::None;
  }
  std::string lhs_str = binding[lhs];
  std::string rhs_str = binding[rhs];

  Type result_type =
      op->getOperand(2).getType().cast<MemRefType>().getElementType();
  std::string type_str = MLIRType2CUDATypeStr(result_type);

  std::string result_name;
  std::string expression;
  if (isa<lmhlo::AddOp>(op)) {
    result_name = EmitUniqueName("add");
    expression = lhs_str + " + " + rhs_str;
  } else if (isa<lmhlo::SubtractOp>(op)) {
    result_name = EmitUniqueName("subtract");
    expression = lhs_str + " - " + rhs_str;
  } else if (isa<lmhlo::MulOp>(op)) {
    result_name = EmitUniqueName("mul");
    expression = lhs_str + " * " + rhs_str;
  } else if (isa<lmhlo::DivOp>(op)) {
    result_name = EmitUniqueName("div");
    expression = lhs_str + " / " + rhs_str;
  } else if (isa<lmhlo::MaxOp>(op)) {
    result_name = EmitUniqueName("max");
    expression = lhs_str + " > " + rhs_str + " ? " + lhs_str + " : " + rhs_str;
  } else if (isa<lmhlo::MinOp>(op)) {
    result_name = EmitUniqueName("min");
    expression = lhs_str + " < " + rhs_str + " ? " + lhs_str + " : " + rhs_str;
  } else if (auto compare = dyn_cast_or_null<lmhlo::CompareOp>(op)) {
    result_name = EmitUniqueName("compare");
    std::string cmp_str;
    switch (compare.getComparisonDirection()) {
      case mhlo::ComparisonDirection::EQ:
        cmp_str = "==";
        break;
      case mhlo::ComparisonDirection::NE:
        cmp_str = "!=";
        break;
      case mhlo::ComparisonDirection::LT:
        cmp_str = "<";
        break;
      case mhlo::ComparisonDirection::LE:
        cmp_str = "<=";
        break;
      case mhlo::ComparisonDirection::GT:
        cmp_str = ">";
        break;
      case mhlo::ComparisonDirection::GE:
        cmp_str = ">=";
        break;
    }
    expression = lhs_str + " " + cmp_str + " " + rhs_str;
  } else if (isa<lmhlo::AndOp>(op)) {
    result_name = EmitUniqueName("and");
    expression = lhs_str + " & " + rhs_str;
  } else if (isa<lmhlo::OrOp>(op)) {
    result_name = EmitUniqueName("or");
    expression = lhs_str + " | " + rhs_str;
  } else if (isa<lmhlo::RemOp>(op)) {
    result_name = EmitUniqueName("rem");
    expression = lhs_str + " % " + rhs_str;
  } else if (isa<lmhlo::PowOp>(op)) {
    result_name = EmitUniqueName("pow");
    expression = "pow(" + lhs_str + ", " + rhs_str + ")";
  }

  assert(binding.count(op->getOperand(2)) == 0);
  binding[op->getOperand(2)] = result_name;

  return type_str + " " + result_name + " = " + expression;
}

llvm::Optional<std::string> SourceEmitterCUDA::EmitElemWiseTernaryOp(
    Operation* op, ValueNameBinding& binding) {
  auto input0 = op->getOperand(0);
  auto input1 = op->getOperand(1);
  auto input2 = op->getOperand(2);
  if (binding.count(input0) == 0 || binding.count(input1) == 0 ||
      binding.count(input2) == 0) {
    return llvm::None;
  }
  std::string input0_str = binding[input0];
  std::string input1_str = binding[input1];
  std::string input2_str = binding[input2];

  Type result_type =
      op->getOperand(3).getType().cast<MemRefType>().getElementType();
  std::string type_str = MLIRType2CUDATypeStr(result_type);

  std::string result_name;
  std::string expression;
  if (isa<lmhlo::SelectOp>(op)) {
    result_name = EmitUniqueName("select");
    expression = input0_str + " ? " + input1_str + " : " + input2_str;
  } else if (isa<lmhlo::ClampOp>(op)) {
    result_name = EmitUniqueName("clamp");
    expression = input1_str + " < " + input0_str + " ? " + input0_str + " : (" +
                 input1_str + " > " + input2_str + " ? " + input2_str + " : " +
                 input1_str + ")";
  }

  assert(binding.count(op->getOperand(3)) == 0);
  binding[op->getOperand(3)] = result_name;

  return type_str + " " + result_name + " = " + expression;
}

// Only supports scalar and splat constant op.
llvm::Optional<std::string>
SourceEmitterCUDA::EmitScalarOrSplatConstantExpression(
    lmhlo::ConstantOp constant) {
  MemRefType memref_type = constant.getOutput().getType().cast<MemRefType>();
  bool is_splat = constant.getValue().isSplat();
  if (memref_type.getRank() != 0 && !is_splat) {
    return llvm::None;
  }

  auto elem_ty = memref_type.getElementType();
  std::string expression;
  if (elem_ty.isIntOrIndex()) {
    int64_t val =
        is_splat ? constant.getValue().getSplatValue<APInt>().getSExtValue()
                 : constant.getValue().getValues<APInt>()[{}].getSExtValue();
    expression = to_string(val);
  } else if (isa<mlir::FloatType>(elem_ty)) {
    double val =
        is_splat
            ? constant.getValue().getSplatValue<APFloat>().convertToDouble()
            : constant.getValue().getValues<APFloat>()[{}].convertToDouble();
    expression = to_string(val);
  } else {
    return llvm::None;
  }

  return expression;
}

llvm::Optional<std::string> SourceEmitterCUDA::EmitScalarOrSplatConstantOp(
    Operation* op, ValueNameBinding& binding) {
  lmhlo::ConstantOp constant = dyn_cast_or_null<lmhlo::ConstantOp>(op);
  if (!constant) {
    return llvm::None;
  }
  MemRefType memref_type = constant.getOutput().getType().cast<MemRefType>();

  auto expression = EmitScalarOrSplatConstantExpression(constant);
  if (!expression.has_value()) {
    return llvm::None;
  }

  Type result_type = memref_type.getElementType();
  std::string type_str = MLIRType2CUDATypeStr(result_type);

  std::string result_name = EmitUniqueName("constant");

  assert(binding.count(op->getOperand(0)) == 0);
  binding[op->getOperand(0)] = result_name;

  return type_str + " " + result_name + " = " + expression.value();
}

llvm::Optional<std::string>
SourceEmitterCUDA::EmitBroadcastOfScalarOrSplatConstantOp(
    Operation* op, ValueNameBinding& binding) {
  lmhlo::DynamicBroadcastInDimOp bcast =
      dyn_cast_or_null<lmhlo::DynamicBroadcastInDimOp>(op);
  if (!bcast) {
    return llvm::None;
  }

  // Only deal with the case that the last rewriter in the same block is the
  // ConstantOp.
  auto input_value = op->getOperand(0);
  auto input_op = findLastWriterInBlock(input_value, op->getBlock());
  if (!input_op.has_value()) {
    return llvm::None;
  }

  lmhlo::ConstantOp constant =
      dyn_cast_or_null<lmhlo::ConstantOp>(input_op.value());
  if (!input_op) {
    return llvm::None;
  }

  if (binding.count(input_value) == 0) {
    return llvm::None;
  }
  std::string expression = binding[input_value];

  Value result = op->getOperand(2);
  MemRefType memref_type = result.getType().cast<MemRefType>();
  Type result_type = memref_type.getElementType();
  std::string type_str = MLIRType2CUDATypeStr(result_type);

  std::string result_name = EmitUniqueName("dyn_bcast_indim");

  assert(binding.count(result) == 0);
  binding[result] = result_name;

  return type_str + " " + result_name + " = " + expression;
}

llvm::Optional<std::string> SourceEmitterCUDA::EmitDynamicReshapeOp(
    Operation* op, ValueNameBinding& binding) {
  lmhlo::DynamicReshapeOp reshape =
      dyn_cast_or_null<lmhlo::DynamicReshapeOp>(op);
  if (!reshape) {
    return llvm::None;
  }

  Value input_value = op->getOperand(0);
  if (binding.count(input_value) == 0) {
    return llvm::None;
  }
  std::string expression = binding[input_value];

  Value result = op->getOperand(2);
  MemRefType memref_type = result.getType().cast<MemRefType>();
  Type result_type = memref_type.getElementType();
  std::string type_str = MLIRType2CUDATypeStr(result_type);

  std::string result_name = EmitUniqueName("dyn_reshape");

  assert(binding.count(result) == 0);
  binding[result] = result_name;

  return type_str + " " + result_name + " = " + expression;
}

llvm::Optional<std::string> SourceEmitterCUDA::EmitTransposeOp(
    Operation* op, ValueNameBinding& binding) {
  lmhlo::TransposeOp transpose = dyn_cast_or_null<lmhlo::TransposeOp>(op);
  if (!transpose) {
    return llvm::None;
  }

  Value input_value = op->getOperand(0);
  if (binding.count(input_value) == 0) {
    return llvm::None;
  }
  std::string expression = binding[input_value];

  Value result = op->getOperand(1);
  MemRefType memref_type = result.getType().cast<MemRefType>();
  Type result_type = memref_type.getElementType();
  std::string type_str = MLIRType2CUDATypeStr(result_type);

  std::string result_name = EmitUniqueName("transpose");

  assert(binding.count(result) == 0);
  binding[result] = result_name;

  return type_str + " " + result_name + " = " + expression;
}

llvm::Optional<std::string> SourceEmitterCUDA::EmitOp(
    Operation* op, ValueNameBinding& binding) {
  if (isa<lmhlo::AbsOp, lmhlo::CeilOp, lmhlo::ConvertOp, lmhlo::CosineOp,
          lmhlo::ExpOp, lmhlo::FloorOp, lmhlo::IsFiniteOp, lmhlo::LogOp,
          lmhlo::Log1pOp, lmhlo::NegOp, lmhlo::NotOp, lmhlo::RsqrtOp,
          lmhlo::SqrtOp, lmhlo::TanhOp>(op)) {
    return EmitElemWiseUnaryOp(op, binding);
  } else if (isa<lmhlo::AddOp, lmhlo::SubtractOp, lmhlo::MulOp, lmhlo::DivOp,
                 lmhlo::MaxOp, lmhlo::MinOp, lmhlo::CompareOp, lmhlo::AndOp,
                 lmhlo::OrOp, lmhlo::RemOp, lmhlo::PowOp>(op)) {
    return EmitElemWiseBinaryOp(op, binding);
  } else if (isa<lmhlo::SelectOp, lmhlo::ClampOp>(op)) {
    return EmitElemWiseTernaryOp(op, binding);
  } else if (isa<lmhlo::ConstantOp>(op)) {
    return EmitScalarOrSplatConstantOp(op, binding);
  } else if (isa<lmhlo::DynamicBroadcastInDimOp>(op)) {
    return EmitBroadcastOfScalarOrSplatConstantOp(op, binding);
  } else if (isa<lmhlo::DynamicReshapeOp>(op)) {
    return EmitDynamicReshapeOp(op, binding);
  } else if (isa<lmhlo::TransposeOp>(op)) {
    return EmitTransposeOp(op, binding);
  } else {
    return llvm::None;
  }
}

std::string SourceEmitterCUDA::EmitUniqueName(llvm::StringRef op_str) {
  std::string name = "bladedisc_" + op_str.str();
  if (existing_names_.count(op_str.str()) == 0) {
    existing_names_.try_emplace(op_str.str(), 0);
  }
  int32_t count = existing_names_[op_str.str()]++;
  name += "_" + to_string(count);

  return name;
}

void SourceEmitterCUDA::bindValueNames(
    const SmallVectorImpl<Value>& inputs,
    const SmallVectorImpl<std::string>& names, ValueNameBinding& binding) {
  assert(inputs.size() == names.size());
  for (int64_t i = 0; i < inputs.size(); i++) {
    binding[inputs[i]] = names[i];
  }
}

bool SourceEmitterCUDA::isSupportedOp(Operation* op) {
  if (isa<lmhlo::AbsOp, lmhlo::CeilOp, lmhlo::ConvertOp, lmhlo::CosineOp,
          lmhlo::ExpOp, lmhlo::FloorOp, lmhlo::IsFiniteOp, lmhlo::LogOp,
          lmhlo::Log1pOp, lmhlo::NegOp, lmhlo::NotOp, lmhlo::RsqrtOp,
          lmhlo::SqrtOp, lmhlo::TanhOp>(op) ||
      isa<lmhlo::AddOp, lmhlo::SubtractOp, lmhlo::MulOp, lmhlo::DivOp,
          lmhlo::MaxOp, lmhlo::MinOp, lmhlo::CompareOp, lmhlo::AndOp,
          lmhlo::OrOp, lmhlo::RemOp, lmhlo::PowOp>(op) ||
      isa<lmhlo::SelectOp, lmhlo::ClampOp>(op) ||
      isa<lmhlo::DynamicReshapeOp, lmhlo::TransposeOp>(op)) {
    return true;
  } else if (isa<lmhlo::ConstantOp>(op)) {
    lmhlo::ConstantOp constant = dyn_cast<lmhlo::ConstantOp>(op);
    MemRefType memref_type = constant.getOutput().getType().cast<MemRefType>();
    return memref_type.getRank() == 0 || constant.getValue().isSplat();
  } else if (isa<lmhlo::DynamicBroadcastInDimOp>(op)) {
    return isBroadcastOnScalarOrSplatConstant(op);
  } else {
    return false;
  }
}

// Return true if the last rewriter in the same block is scalar or splat
// constant op.
bool SourceEmitterCUDA::isBroadcastOnScalarOrSplatConstant(Operation* op) {
  auto input_op = findLastWriterInBlock(op->getOperand(0), op->getBlock());
  if (!input_op.has_value()) {
    return false;
  }
  lmhlo::ConstantOp constant = dyn_cast<lmhlo::ConstantOp>(input_op.value());
  if (!constant) {
    return false;
  }
  MemRefType memref_type = constant.getOutput().getType().cast<MemRefType>();
  return memref_type.getRank() == 0 || constant.getValue().isSplat();
}

}  // namespace disc_ral
}  // namespace mlir
