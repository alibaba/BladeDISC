/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "mlir/disc/disc_util.h"

#include <numeric>
#include <optional>

#include "lhlo/IR/lhlo_ops.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/transforms/placement_utils.h"
#include "tensorflow/core/util/env_var.h"

namespace mlir {
namespace disc_ral {

bool IsSmallBuffer(Value alloc) {
  constexpr unsigned kMaximumSizeInBytes = 128;
  constexpr unsigned kBitwidthOfIndexType = 64;

  auto type = alloc.getType().dyn_cast<ShapedType>();
  if (!type || !type.hasStaticShape()) return false;

  // For index types, use the provided size, as the type does not know.
  unsigned int bitwidth = type.getElementType().isIndex()
                              ? kBitwidthOfIndexType
                              : type.getElementTypeBitWidth();
  return type.getNumElements() * bitwidth <= kMaximumSizeInBytes * 8;
}

bool IsSmallCpuBuffer(Value alloc) {
  if (placement_utils::isGpuMemRef(alloc)) return false;
  return IsSmallBuffer(alloc);
}

bool IsSmallCpuAlloc(Value alloc) {
  return IsSmallCpuBuffer(alloc) && alloc.getDefiningOp<memref::AllocOp>();
}

bool IsOpWriteValue(Operation* op, Value value) {
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 2> effects;
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  // Suppose that value without `MemoryEffectOpInterface` is readonly.
  if (!interface) return false;

  interface.getEffectsOnValue(value, effects);
  return llvm::any_of(
      effects, [](const mlir::MemoryEffects::EffectInstance& instance) {
        return mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect());
      });
}

bool IsMemRefAliasOp(Operation* op) {
  return dyn_cast<ViewLikeOpInterface>(op) != nullptr;
}

Value getRootMemRef(Value memref) {
  Value rootMemRef = memref;
  while (Operation* operandOp = rootMemRef.getDefiningOp()) {
    if (!isa<memref::SubViewOp, memref::ViewOp, memref::CastOp,
             memref::ReinterpretCastOp>(operandOp))
      break;
    rootMemRef = operandOp->getOperand(0);
  }
  return rootMemRef;
}

bool isSameUnderlineBuffer(Value lhs, Value rhs) {
  return getRootMemRef(lhs) == getRootMemRef(rhs);
}

bool parseEinsumEquation(
    llvm::StringRef equation,
    llvm::SmallDenseMap<char, llvm::SmallDenseMap<EquationVariable, size_t>>&
        tokens,
    SmallVector<char>* lhs_original_tokens,
    SmallVector<char>* rhs_original_tokens,
    SmallVector<char>* result_original_tokens) {
  size_t index = 0;
  size_t sub_index = 0;
  EquationVariable current_variable = kIsLhs;
  SmallVector<char> lhs_original_tokens_internal;
  SmallVector<char> rhs_original_tokens_internal;
  SmallVector<char> result_original_tokens_internal;
  bool explicit_result = false;
  while (index < equation.size()) {
    if (std::isalpha(equation[index])) {
      if (current_variable == kIsLhs) {
        tokens[equation[index]][kIsLhs] = sub_index;
        lhs_original_tokens_internal.push_back(equation[index]);
        sub_index++;
      } else if (current_variable == kIsRhs) {
        tokens[equation[index]][kIsRhs] = sub_index;
        rhs_original_tokens_internal.push_back(equation[index]);
        sub_index++;
      } else {
        tokens[equation[index]][kIsResult] = sub_index;
        result_original_tokens_internal.push_back(equation[index]);
        sub_index++;
      }
    } else if (equation.substr(index, 1).contains(",")) {
      current_variable = kIsRhs;
      sub_index = 0;
    } else if ((index < (equation.size() - 1)) &&
               (equation.substr(index, 2).contains("->"))) {
      current_variable = kIsResult;
      explicit_result = true;
      sub_index = 0;
      index++;
    } else if (equation[index] == ' ') {
      // do nothing but continue
    } else {
      return false;
    }
    index++;
  }

  // If no "->" in the equation, deduce the result tokens
  // TODO: handle when operands contain ellipsis
  if (!explicit_result) {
    sub_index = 0;
    for (char lhs_c : lhs_original_tokens_internal) {
      if (std::find(rhs_original_tokens_internal.begin(),
                    rhs_original_tokens_internal.end(),
                    lhs_c) == rhs_original_tokens_internal.end()) {
        tokens[lhs_c][kIsResult] = sub_index;
        result_original_tokens_internal.push_back(lhs_c);
        sub_index++;
      }
    }
    for (char rhs_c : rhs_original_tokens_internal) {
      if (std::find(lhs_original_tokens_internal.begin(),
                    lhs_original_tokens_internal.end(),
                    rhs_c) == lhs_original_tokens_internal.end()) {
        tokens[rhs_c][kIsResult] = sub_index;
        result_original_tokens_internal.push_back(rhs_c);
        sub_index++;
      }
    }
  }
  if (lhs_original_tokens) {
    lhs_original_tokens->swap(lhs_original_tokens_internal);
  }
  if (rhs_original_tokens) {
    rhs_original_tokens->swap(rhs_original_tokens_internal);
  }
  if (result_original_tokens) {
    result_original_tokens->swap(result_original_tokens_internal);
  }
  return true;
}

llvm::Optional<int32_t> TryMergeNode(GraphCycles* graph_cycles, int32_t a,
                                     int32_t b) {
  if (graph_cycles == nullptr) {
    return llvm::None;
  }
  bool has_edge_inserted_a2b = false;
  if (!graph_cycles->HasEdge(a, b) && !graph_cycles->HasEdge(b, a)) {
    has_edge_inserted_a2b = graph_cycles->InsertEdge(a, b);
    if (!has_edge_inserted_a2b) {
      // Cannot merge a and b as we cannot even insert an edge between a and b.
      return llvm::None;
    }
  }
  int32_t from = graph_cycles->HasEdge(a, b) ? a : b;
  int32_t to = (from == a) ? b : a;
  auto result = graph_cycles->ContractEdge(from, to);
  if (!result.has_value() && has_edge_inserted_a2b) {
    // Restore the graph.
    graph_cycles->RemoveEdge(a, b);
  }
  return result;
}

SmallVector<Value, 4> GetAllPossibleUsedValues(Operation* op) {
  SmallVector<Value, 4> values;
  op->walk([&](Operation* nest_op) {
    for (Value v : nest_op->getOperands()) {
      values.push_back(v);
    }
  });
  return values;
}

bool useShapeConstraintIR() {
  static bool enabled = []() {
    bool enable_shape_constraint_ir = true;
    tensorflow::ReadBoolFromEnvVar("DISC_ENABLE_SHAPE_CONSTRAINT_IR",
                                   enable_shape_constraint_ir,
                                   &enable_shape_constraint_ir);
    return enable_shape_constraint_ir;
  }();
  return enabled;
}

// Returns true if `DISC_ENABLE_HORIZONTAL_FUSION` is true
bool useHorizontalFusion() {
  static bool enabled = []() {
    bool enabled = true;
    tensorflow::ReadBoolFromEnvVar("DISC_ENABLE_HORIZONTAL_FUSION", enabled,
                                   &enabled);
    return enabled;
  }();
  return enabled;
}

// Returns true if `DISC_ENABLE_TRANSFORM_SCHEDULE` is true
bool useTransformSchedule() {
  static bool enabled = []() {
    bool enabled = false;
    tensorflow::ReadBoolFromEnvVar("DISC_ENABLE_TRANSFORM_SCHEDULE", enabled,
                                   &enabled);
    return enabled;
  }();
  return enabled;
}

// Returns true if `DISC_ENABLE_TRANSFORM_GEMM_EPILOGUE_FUSION` is true.
bool useTransformGEMMEpilogueFusionSchedule() {
  static bool enabled = []() {
    bool enabled = true;
    tensorflow::ReadBoolFromEnvVar("DISC_ENABLE_TRANSFORM_GEMM_EPILOGUE_FUSION",
                                   enabled, &enabled);
    return enabled;
  }();
  return enabled;
}

bool lowerFakeQuantToQuantAndDequant() {
  static bool enabled = []() {
    bool enabled = false;
    tensorflow::ReadBoolFromEnvVar("DISC_FAKE_QUANT_TO_QUANT_AND_DEQUANT",
                                   enabled, &enabled);
    return enabled;
  }();
  return enabled;
}

bool isMemIntensiveOptExperimentalEnabled() {
  static bool enabled = []() {
    bool enabled = false;
    tensorflow::ReadBoolFromEnvVar("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL",
                                   enabled, &enabled);
    return enabled;
  }();
  return enabled;
}

bool isStitchEnabled() {
  static bool enabled = []() {
    // Enable stitch by default.
    bool enabled = true;
    tensorflow::ReadBoolFromEnvVar("DISC_ENABLE_STITCH", enabled, &enabled);
    return enabled;
  }();
  return enabled;
}

bool isCompIntensFusionEnabled() {
  static bool enabled = []() {
    bool enabled = false;
    tensorflow::ReadBoolFromEnvVar("DISC_ENABLE_COMPUTE_INTENSIVE_FUSE",
                                   enabled, &enabled);
    return enabled;
  }();
  return enabled;
}

bool enableLoopUnroll() {
  static bool enabled = []() {
    bool enabled = true;
    tensorflow::ReadBoolFromEnvVar(
        "DISC_LLVM_PIPELINE_TUNING_ENABLE_LOOP_UNROLL", enabled, &enabled);
    return enabled;
  }();
  return enabled;
}

bool enableLoopInterleave() {
  static bool enabled = []() {
    bool enabled = true;
    tensorflow::ReadBoolFromEnvVar(
        "DISC_LLVM_PIPELINE_TUNING_ENABLE_LOOP_INTERLEAVE", enabled, &enabled);
    return enabled;
  }();
  return enabled;
}

bool enableLoopVectorize() {
  static bool enabled = []() {
    bool enabled = true;
    tensorflow::ReadBoolFromEnvVar(
        "DISC_LLVM_PIPELINE_TUNING_ENABLE_LOOP_VECTORIZE", enabled, &enabled);
    return enabled;
  }();
  return enabled;
}

bool enableSLPVectorize() {
  static bool enabled = []() {
    bool enabled = true;
    tensorflow::ReadBoolFromEnvVar(
        "DISC_LLVM_PIPELINE_TUNING_ENABLE_SLP_VECTORIZE", enabled, &enabled);
    return enabled;
  }();
  return enabled;
}

// Returns data users of the value and its aliases (e.g. memref.cast).
// Here non-data users means DimOp, DeallocOp and ShapeOfOp.
SmallVector<Operation*, 4> getValueUsers(Value v) {
  Value root = v;
  while (Operation* operandOp = root.getDefiningOp()) {
    auto view = dyn_cast<ViewLikeOpInterface>(operandOp);
    if (!view) break;
    root = operandOp->getOperand(0);
  }

  SmallVector<Operation*, 4> users;
  SmallVector<Value, 4> worklist;
  worklist.push_back(root);
  while (!worklist.empty()) {
    Value curr = worklist.back();
    worklist.pop_back();
    for (Operation* user : curr.getUsers()) {
      // Skip non-data users
      if (isa<memref::DimOp, memref::DeallocOp, shape::ShapeOfOp>(user)) {
        continue;
      }
      // alias value
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        worklist.push_back(view->getResult(0));
      } else {
        users.push_back(user);
      }
    }
  }
  return users;
}

// Returns true if the underlying buffer of this memref is a const buffer.
bool isConstantMemRef(Value value) {
  Value root = getRootMemRef(value);
  for (Operation* user : getValueUsers(root)) {
    if (isa<lmhlo::ConstantOp>(user)) return true;
  }
  return false;
}

// Gets all float values from the given attribute and push them to `values`.
DenseFPElementsAttr GetF32ElementsAttr(Attribute attr, Builder* builder) {
  SmallVector<float, 4> values;
  auto array_attr = attr.dyn_cast<ArrayAttr>();
  CHECK(array_attr) << "Cannot cast arrar_attr to ArrayAttr!";
  values.reserve(array_attr.getValue().size());
  for (Attribute val : array_attr.getValue()) {
    values.push_back(
        static_cast<float>(val.cast<FloatAttr>().getValueAsDouble()));
  }
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder->getF32Type());
  return DenseFPElementsAttr::get(ty, values);
}

// Returns a 1-d i64 elements attribute populated with numbers from start to
// end, excluding.
DenseIntElementsAttr GetI64ElementsAttrForSeq(int start, int end,
                                              Builder* builder) {
  int size = end - start;

  SmallVector<int64_t, 4> vals;
  vals.resize(size);
  std::iota(vals.begin(), vals.end(), start);

  TensorType ty = RankedTensorType::get({size}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, vals);
}

int getNumResultOperands(Operation* op) {
  if (isa<lmhlo_disc::CustomCallOp>(op)) {
    // TODO(disc): add a registration base mechanism to support custom call op
    // with more than one results.
    return 1;
  }

  if (op->getDialect()->getTypeID() != TypeID::get<lmhlo::LmhloDialect>() &&
      op->getDialect()->getTypeID() !=
          TypeID::get<lmhlo_disc::LmhloDiscDialect>()) {
    return 0;
  }
  return llvm::count_if(op->getOperands(),
                        [&](Value v) { return IsOpWriteValue(op, v); });
}

int getShmemSizeBytesNotAffectOccupancy(int cc_major, int cc_minor) {
  return 8192;
}

Type getLhloOpsElementType(Operation* op) {
  unsigned int num_operands = op->getNumOperands();
  Type result_type = op->getOperand(num_operands - 1)
                         .getType()
                         .cast<MemRefType>()
                         .getElementType();
  return result_type;
}

Value CastMemRefTo(OpBuilder& b, Location loc, Value from, Type toType,
                   ValueRange toShape) {
  int64_t rank = toShape.size();
  auto memrefTy = toType.cast<MemRefType>();
  auto getIntAttr = [&](int64_t val) {
    return b.getIntegerAttr(b.getIndexType(), val);
  };

  SmallVector<OpFoldResult> foldSizes;
  for (int i = 0; i < rank; ++i) {
    if (memrefTy.getDimSize(i) == ShapedType::kDynamic) {
      foldSizes.push_back(toShape[i]);
    } else {
      foldSizes.push_back(getIntAttr(memrefTy.getDimSize(i)));
    }
  }

  Value dynamicStride;
  int64_t staticStride = 1;
  SmallVector<OpFoldResult> foldStrides;
  for (int i = rank - 1; i >= 0; --i) {
    if (staticStride != ShapedType::kDynamic) {
      foldStrides.push_back(getIntAttr(staticStride));
      if (memrefTy.getDimSize(i) == ShapedType::kDynamic) {
        dynamicStride = b.create<arith::ConstantIndexOp>(loc, staticStride);
        staticStride = ShapedType::kDynamic;
      } else {
        staticStride *= memrefTy.getDimSize(i);
      }
    } else {
      foldStrides.push_back(dynamicStride);
    }
    if (dynamicStride) {
      dynamicStride = b.create<arith::MulIOp>(loc, dynamicStride, toShape[i]);
    }
  }

  OpFoldResult zero{getIntAttr(0)};
  auto reversedStrides = llvm::to_vector<4>(llvm::reverse(foldStrides));
  auto castOp = b.create<memref::ReinterpretCastOp>(loc, memrefTy, from, zero,
                                                    foldSizes, reversedStrides);
  return castOp;
}

Value createViewLike(OpBuilder& b, Location loc, Value from, Value to) {
  SmallVector<Value> toShape = getShapeValues(&b, to);
  auto toType = to.getType().cast<MemRefType>();
  auto fromType = from.getType().cast<MemRefType>();
  auto targetType =
      MemRefType::get(toType.getShape(), fromType.getElementType(),
                      toType.getLayout(), toType.getMemorySpace());
  return CastMemRefTo(b, loc, from, targetType, toShape);
}

SmallVector<Value> getShapeValues(OpBuilder* b, Value memref) {
  auto shape = memref.getType().dyn_cast<ShapedType>().getShape();
  int64_t rank = shape.size();
  auto loc = memref.getLoc();

  SmallVector<Value> result;
  for (int i = 0; i < rank; ++i) {
    if (shape[i] == ShapedType::kDynamic) {
      result.push_back(b->create<memref::DimOp>(loc, memref, i));
    } else {
      result.push_back(b->create<arith::ConstantIndexOp>(loc, shape[i]));
    }
  }
  return result;
}

// Returns 1D 64-bit dense elements attribute with the given values.

using namespace llvm;

static std::optional<OptimizationLevel> mapToLevel(unsigned optLevel,
                                                   unsigned sizeLevel) {
  switch (optLevel) {
    case 0:
      return OptimizationLevel::O0;

    case 1:
      return OptimizationLevel::O1;

    case 2:
      switch (sizeLevel) {
        case 0:
          return OptimizationLevel::O2;

        case 1:
          return OptimizationLevel::Os;

        case 2:
          return OptimizationLevel::Oz;
      }
      break;
    case 3:
      return OptimizationLevel::O3;
  }
  return std::nullopt;
}
// Create and return a lambda that uses LLVM pass manager builder to set up
// optimizations based on the given level.
std::function<Error(Module*)> makeOptimizingTransformer(
    unsigned optLevel, unsigned sizeLevel, TargetMachine* targetMachine) {
  return [optLevel, sizeLevel, targetMachine](Module* m) -> Error {
    std::optional<OptimizationLevel> ol = mapToLevel(optLevel, sizeLevel);
    if (!ol) {
      return make_error<StringError>(
          formatv("invalid optimization/size level {0}/{1}", optLevel,
                  sizeLevel)
              .str(),
          inconvertibleErrorCode());
    }
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;

    PipelineTuningOptions tuningOptions;
    tuningOptions.LoopUnrolling = enableLoopUnroll();
    tuningOptions.LoopInterleaving = enableLoopInterleave();
    tuningOptions.LoopVectorization = enableLoopVectorize();
    tuningOptions.SLPVectorization = enableSLPVectorize();
    PassBuilder pb(targetMachine, tuningOptions);

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm;
    mpm.addPass(pb.buildPerModuleDefaultPipeline(*ol));
    mpm.run(*m, mam);
    return Error::success();
  };
}
}  // namespace disc_ral
}  // namespace mlir
