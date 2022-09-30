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

#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"

#include <algorithm>
#include <mutex>

#include "mlir-hlo/Dialect/lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir-hlo/utils/placement_utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"  // TF:llvm-project
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/disc_shape_optimization_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/lhlo_elemental_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"

// This file implements some helper functions and classes used to do fusion
// & code generation.

namespace mlir {
namespace disc_ral {

using namespace lmhlo;
using disc_shape::SymbolicDimOp;
using placement_utils::kDiscPlaceAssignment;
using placement_utils::kDiscShapeCalcAttr;

void dumpFusionPattern(FusionPattern& pattern) {
  for (Operation* subOp : pattern.getOpList()) {
    llvm::dbgs() << "  " << *subOp << "\n";
  }
}

DenseSet<Operation*> NoLoaderUser(SmallVectorImpl<Operation*>& ops) {
  SmallVector<Operation*, 4> worklist;
  DenseSet<Operation*> has_loader_ops;
  for (Operation* op : ops) {
    Value memref = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    if (memref == nullptr) continue;
    for (Operation* user : getValueUsersInFusionLike(memref, op)) {
      if (isa<memref::LoadOp>(user)) {
        worklist.push_back(op);
        has_loader_ops.insert(op);
      }
    }
  }

  while (!worklist.empty()) {
    Operation* op = worklist.pop_back_val();
    int num_operands = op->getNumOperands();
    for (int i = 0; i < num_operands - 1; ++i) {
      Value memref = op->getOperand(i);
      for (Operation* user : getValueUsersInFusionLike(memref, op)) {
        if ((!isa<lmhlo::LmhloOp>(user)) || has_loader_ops.count(user))
          continue;
        if (isSameUnderlineBuffer(cast<lmhlo::LmhloOp>(user).getResultBuffer(),
                                  memref)) {
          worklist.push_back(user);
          has_loader_ops.insert(user);
        }
      }
    }
  }

  DenseSet<Operation*> no_loader_ops;
  for (Operation* op : ops)
    if (!has_loader_ops.count(op)) no_loader_ops.insert(op);
  return no_loader_ops;
}

void cleanUnusedLhloOps(Block* parent, PatternRewriter* rewriter) {
  SmallVector<Operation*, 4> lhlo_ops;
  for (Operation& op : parent->getOperations()) {
    if (op.getDialect() == op.getContext()->getLoadedDialect("lmhlo") &&
        (!isa<lmhlo::TerminatorOp>(op)))
      lhlo_ops.push_back(&op);
  }
  const DenseSet<Operation*>& no_loader_user = NoLoaderUser(lhlo_ops);
  for (auto* lhlo_op : no_loader_user) {
    if (rewriter) {
      rewriter->eraseOp(lhlo_op);
    } else {
      lhlo_op->erase();
    }
  }
}

// returns the users of the `memref`. The users should be in the same fusion
// like `op`.
DenseSet<Operation*> getValueUsersInFusionLike(Value memref, Operation* op) {
  // rootMemRef is the underline buffer, by passing some memref cast ops.
  Value rootMemRef = getRootMemRef(memref);

  DenseSet<Operation*> ops;
  SmallVector<Value, 4> worklist{rootMemRef};
  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    for (Operation* user : val.getUsers()) {
      if (isa<memref::SubViewOp, memref::ViewOp, memref::CastOp,
              memref::ReinterpretCastOp>(user)) {
        worklist.push_back(user->getResult(0));
        continue;
      }
      // SpecializeWithSpeculation pass may generate multi versions from a
      // fusion op. This fusion family accesses a same set of memrefs.
      if (!inSameFusionOp(user, op)) continue;
      ops.insert(user);
    }
  }

  return ops;
}

bool isOnGpu(Operation* op) {
  auto attr =
      op->getAttrOfType<StringAttr>(placement_utils::kDiscPlaceAssignment);
  if (attr) {
    return (attr.getValue() == placement_utils::kGpu);
  }

  // Fusion should have been set a placement attribute in lhlo_fusion pass.
  // Leave a default placement here just in case there are fusion ops not
  // generated by the fusion pass (e.g. in the file-check based ut).
  if (isa<lmhlo::FusionOp>(op)) return true;

  assert(isa<lmhlo::LmhloOp>(op) && "Unexpected usage of isOnGpu");
  auto result_memref = cast<lmhlo::LmhloOp>(op).getResultBuffer();
  auto memory_space =
      result_memref.getType().cast<MemRefType>().getMemorySpace();
  return memory_space && memory_space.isa<StringAttr>() &&
         memory_space.cast<StringAttr>().getValue() ==
             mhlo::placement_utils::cGpu;
}

StringRef fusionTypeToString(FusionType ft) {
  switch (ft) {
    case FusionType::kNone:
      return "kNone";
    case FusionType::kLoop:
      return "kLoop";
    case FusionType::kRowReduction:
      return "kRowReduction";
    case FusionType::kColReduction:
      return "kColReduction";
    case FusionType::kInput:
      return "kInput";
    case FusionType::kStitch:
      return "kStitch";
    case FusionType::kLargeConcat:
      return "kLargeConcat";
    default:
      assert(false && "unknown fusion type");
      return "";
  }
}

// Convert to a fusion type from its string representation.
FusionType fusionTypeFromString(StringRef ft) {
  if (ft == "kNone") {
    return FusionType::kNone;
  } else if (ft == "kLoop") {
    return FusionType::kLoop;
  } else if (ft == "kRowReduction") {
    return FusionType::kRowReduction;
  } else if (ft == "kColReduction") {
    return FusionType::kColReduction;
  } else if (ft == "kInput") {
    return FusionType::kInput;
  } else if (ft == "kStitch") {
    return FusionType::kStitch;
  } else if (ft == "kLargeConcat") {
    return FusionType::kLargeConcat;
  }
  assert(false && "unknown fusion type");
  return FusionType::kNone;
}

FusionType getFusionType(Operation* op) {
  if (!isa<lmhlo::FusionOp>(op)) {
    return FusionType::kNone;
  }
  FusionType type = FusionType::kNone;
  auto fusionTypeAttr = op->getAttrOfType<StringAttr>(kDiscFusionTypeAttrName);
  if (fusionTypeAttr) {
    type = fusionTypeFromString(fusionTypeAttr.getValue());
  }
  return type;
}

// Returns the name of the fusion op
StringRef getFusionName(lmhlo::FusionOp op) {
  auto attr = op->getAttrOfType<StringAttr>(kFusionOpNameAttr);
  if (!attr) return "";
  return attr.getValue();
}

// Sets the name of the fusion op
void setFusionName(OpBuilder& b, lmhlo::FusionOp op, StringRef name) {
  op->setAttr(kFusionOpNameAttr, b.getStringAttr(name));
}

// Attaches a new tag to the fusion op.
// Here different tags is mapping to different variants of the fusion op.
void addFusionTag(OpBuilder& b, lmhlo::FusionOp op, StringRef tag) {
  std::string oldTag;
  auto attr = op->getAttrOfType<StringAttr>(kFusionOpTagAttr);
  if (attr && attr.getValue().size()) {
    oldTag = (Twine(attr.getValue()) + "X").str();
  }
  op->setAttr(kFusionOpTagAttr, b.getStringAttr((Twine(oldTag) + tag).str()));
}

// Returns the full name of the fusion op
// Here full name is composed of the name and tag of the fusion op.
std::string getFusionFullName(lmhlo::FusionOp op) {
  auto name_attr = op->getAttrOfType<StringAttr>(kFusionOpNameAttr);
  if (!name_attr) return "";
  auto tag_attr = op->getAttrOfType<StringAttr>(kFusionOpTagAttr);
  if (!tag_attr) return name_attr.getValue().str();
  return (name_attr.getValue() + Twine("___") + tag_attr.getValue()).str();
}

std::string generateSignatureForFusion(FusionPattern& fusionPattern) {
  SmallString<128> sig;

  sig.append(fusionPattern.getFusionTypeStr());
  sig.append("_");

  int64_t num_ops = fusionPattern.getOpList().size();
  for (Operation* op : fusionPattern.getRootOps()) {
    sig.append(op->getName().stripDialect());
    sig.push_back('_');
  }

  sig.append(
      ("_" + Twine(num_ops) + "_" + Twine(fusionPattern.getResults().size()))
          .str());
  return std::string(sig.str());
}

bool inSameFusionOp(Operation* op, Operation* other) {
  FusionOp opFusion = op->getParentOfType<FusionOp>();
  FusionOp otherFusion = other->getParentOfType<FusionOp>();
  if (!opFusion || !otherFusion) {
    return false;
  }
  return opFusion.getOperation() == otherFusion.getOperation();
}

bool inSameFusionFamily(Operation* op, Operation* other) {
  FusionOp opFusion = op->getParentOfType<FusionOp>();
  FusionOp otherFusion = other->getParentOfType<FusionOp>();
  if (!opFusion || !otherFusion) {
    return false;
  }

  StringRef opFusionName = getFusionName(opFusion);
  StringRef otherFusionName = getFusionName(otherFusion);

  return ((opFusionName == otherFusionName) && !opFusionName.empty());
}

// Returns true if the op is an elementwise unary lmhlo op.
// TODO(disc): use fusibility interface
// TODO(disc): Unify with disc_supported_list.h and Elementwise Trait
bool isElementWiseUnary(Operation* op) {
  // clang-format off
  return isa<
    lmhlo::AbsOp,
    lmhlo::CeilOp,
    lmhlo::ConvertOp,
    lmhlo::CopyOp,
    lmhlo::CosineOp,
    lmhlo::ExpOp,
    lmhlo::FloorOp,
    lmhlo::IsFiniteOp,
    lmhlo::LogOp,
    lmhlo::Log1pOp,
    lmhlo::LogisticOp,
    lmhlo::NegOp,
    lmhlo::NotOp,
    lmhlo::RsqrtOp,
    lmhlo::SignOp,
    lmhlo::SineOp,
    lmhlo::SqrtOp,
    lmhlo::TanhOp,
    lmhlo::RoundNearestEvenOp
  >(op);
  // clang-format on
}

// Returns true if the op is an elementwise binary lmhlo op.
// TODO(disc): use fusibility interface
bool isElementWiseBinary(Operation* op) {
  // clang-format off
  return isa<
    lmhlo::AddOp,
    lmhlo::AndOp,
    lmhlo::CompareOp,
    lmhlo::DivOp,
    lmhlo::MaxOp,
    lmhlo::MinOp,
    lmhlo::MulOp,
    lmhlo::OrOp,
    lmhlo::PowOp,
    lmhlo::RemOp,
    lmhlo::SubtractOp
  >(op);
  // clang-format on
}

// Returns true if the op is an elementwise ternary lmhlo op.
bool isElementWiseTernary(Operation* op) {
  // Some ternary lmhlo ops (e.g. select op) suppport a restricted implicit
  // broadcast semantic, that is: if one input has rank zero, then it will be
  // automatically broadcasted if necessary. Thus we can know that there is no
  // broadcast if all operands have the same rank.
  if (isa<lmhlo::SelectOp, lmhlo::ClampOp>(op)) {
    auto ref = op->getOperand(0).getType().cast<MemRefType>();
    for (Value v : op->getOperands().drop_front())
      if (v.getType().cast<MemRefType>().getRank() != ref.getRank())
        return false;
    return true;
  }

  return false;
}

// Returns true if the op is an elementwise lmhlo op.
// TODO(disc): use fusibility interface
bool isElementWise(Operation* op) {
  return isElementWiseUnary(op) || isElementWiseBinary(op) ||
         isElementWiseTernary(op);
}

// Returns true if this op is a rank-2 row reduction.
bool isRank2RowReduction(Operation* op) {
  auto reduce_op = dyn_cast<lmhlo::ReduceOp>(op);
  if (!reduce_op || reduce_op.getDimensions().getNumElements() != 1)
    return false;

  int rank = op->getOperand(0).getType().cast<MemRefType>().getRank();
  auto dimensions = reduce_op.getDimensions().getValues<int64_t>();
  return ((*dimensions.begin() == 1) && (rank == 2));
}

// Returns true if this op is a row reduction.
bool isRowReduction(Operation* op) {
  auto reduce_op = dyn_cast<lmhlo::ReduceOp>(op);
  if (!reduce_op) return false;

  auto dimensions = reduce_op.getDimensions().getValues<int64_t>();
  auto ty = op->getOperand(0).getType().dyn_cast<MemRefType>();
  if (!ty) return false;
  int expected = ty.getRank() - 1;
  for (int64_t dim : llvm::reverse(dimensions)) {
    LLVM_DEBUG(llvm::dbgs()
               << "dim, expected = " << dim << ", " << expected << "\n");
    if (dim != expected--) return false;
  }

  return true;
}

// Returns true if this op is a rank-2 column reduction.
bool isRank2ColReduction(Operation* op) {
  auto reduce_op = dyn_cast<lmhlo::ReduceOp>(op);
  if (!reduce_op || reduce_op.getDimensions().getNumElements() != 1)
    return false;

  int rank = op->getOperand(0).getType().cast<MemRefType>().getRank();
  auto dimensions = reduce_op.getDimensions().getValues<int64_t>();
  return ((*dimensions.begin() == 0) && (rank == 2));
}

// Returns true if the op is supported by the downstreaming fusion codegen
// engine.
bool isFusible(Operation* op) {
  // Only scalar const are supported by the fusion codegen engine a.t.m.
  if (isa<lmhlo::ConstantOp>(op)) {
    auto constant = cast<lmhlo::ConstantOp>(op);
    MemRefType type = constant.getOutput().getType().cast<MemRefType>();
    return (type.getRank() == 0 || constant.getValue().isSplat());
  }

  // All element ops are supported by the fusion codegen engine.
  if (isElementWise(op)) return true;

  // clang-format off
  return isa<
    lmhlo::BroadcastInDimOp,
    lmhlo::BroadcastOp,
    lmhlo::ClampOp,
    lmhlo::ConcatenateOp,
    lmhlo::DynamicBroadcastInDimOp,
    lmhlo::DynamicGatherOp,
    lmhlo::DynamicIotaOp,
    lmhlo::DynamicPadOp,
    lmhlo::DynamicReshapeOp,
    lmhlo::GatherOp,
    lmhlo::RealDynamicSliceOp,
    lmhlo::ReduceOp,
    lmhlo::ReshapeOp,
    lmhlo::ReverseOp,
    lmhlo::SelectOp,
    lmhlo::SliceOp,
    lmhlo::TransposeOp
  >(op);
  // clang-format on
}

// Returns the number of operands that are supposed to be written.
// For some ops (e.g. lmhlo ops), some operands are the output memrefs
// Thus these operands are supposed to be updated.
int getNumResultOperands(Operation* op) {
  if (auto customOp = dyn_cast_or_null<lmhlo_disc::CustomCallOp>(op)) {
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

int64_t getFirstOperandIndex(Operation* op, Value value) {
  for (int64_t i = 0; i < op->getNumOperands(); ++i) {
    auto operand = op->getOperand(i);
    if (operand == value) {
      return i;
    }
  }
  assert(false && "Exception in getFirstOperandIndex, value is not an operand");
  return -1;
}

// Returns a process-level fusion strategy singleton.
FusionStrategy& getFusionStrategy(StringRef device, StringRef strategy);

bool isStitchFusion(Operation* op) {
  return isFusionType<FusionType::kStitch>(op);
}

// Create a new fusion pattern from a single op.
FusionPatternBase::FusionPatternBase(Operation* op) {
  op_list_.push_back(op);
  calculateOperandsAndResults();
}

// Create a new fusion pattern from the ops inside the lmhlo fusion op.
FusionPatternBase::FusionPatternBase(lmhlo::FusionOp op) {
  for (Operation& op : op.getRegion().getBlocks().front()) {
    if (!isa<lmhlo::TerminatorOp>(op)) op_list_.push_back(&op);
  }
  calculateOperandsAndResults();
}

// Create a new fusion pattern from a valid fusion op list.
FusionPatternBase::FusionPatternBase(SmallVectorImpl<Operation*>& op_list)
    : op_list_(op_list.begin(), op_list.end()) {
  calculateOperandsAndResults();
}

// Returns the effective size (e.g. not counting const ops) of the ops this
// fusion pattern contains.
int FusionPatternBase::effectiveSize() {
  return llvm::count_if(
      op_list_, [](Operation* op) { return !matchPattern(op, m_Constant()); });
}

// Sorts the ops inside the fusion pattern according to the keys provided.
void FusionPatternBase::sortFusionOpListBy(
    DenseMap<Operation*, int>& op_to_idx) {
  std::sort(op_list_.begin(), op_list_.end(),
            [&](Operation* lhs, Operation* rhs) {
              return op_to_idx[lhs] < op_to_idx[rhs];
            });
}

void FusionPatternBase::sortFusionOpListWithTopologyOrder() {
  FusionOpList topology;
  // First, build graph and indegree map.
  // {node, {consumers}}
  DenseMap<Operation*, DenseSet<Operation*>> consumer_graph;
  DenseMap<Operation*, int64_t> indegree;
  for (auto& root : op_list_) {
    indegree[root] = 0;
  }
  DenseSet<Operation*> op_set(op_list_.begin(), op_list_.end());
  for (auto& op : op_list_) {
    // Make sure there is a key for `op` in `consumer_graph`.
    auto& consumers = consumer_graph[op];
    int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
    for (Value result : op->getOperands().drop_front(num_input_operand)) {
      for (Operation* user : getValueUsers(result)) {
        if (user == op || !op_set.contains(user)) {
          continue;
        }
        consumers.insert(user);
        indegree[user]++;  // Update indegree of the consumer op.
      }
    }
  }
  // Second, BFS to build topology array.
  SmallVector<Operation*> zero_indegree;
  for (auto& indegree_pair : indegree) {
    if (indegree_pair.second == 0) {
      zero_indegree.push_back(indegree_pair.first);
    }
  }
  while (!zero_indegree.empty()) {
    Operation* op = zero_indegree.pop_back_val();
    topology.push_back(op);
    for (auto consumer : consumer_graph[op]) {
      auto& consumer_indegree = indegree[consumer];
      if (--consumer_indegree == 0) {
        zero_indegree.push_back(consumer);
      }
    }
  }
  assert(op_list_.size() == topology.size());
  op_list_ = std::move(topology);
}

// Calculates the inputs and outputs of the fusion pattern.
void FusionPatternBase::calculateOperandsAndResults() {
  DenseSet<Value> input_set;
  DenseSet<Value> result_set;
  DenseSet<Value> internal_result_set;
  DenseSet<Operation*> op_set(op_list_.begin(), op_list_.end());

  for (Operation* op : op_list_) {
    int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
    for (Value v : op->getOperands().drop_front(num_input_operand)) {
      bool inserted = last_writer_.try_emplace(v, op).second;
      (void)inserted;
      assert(inserted);

      bool has_external_user = false;
      bool has_internal_user = false;
      for (Operation* user : getValueUsers(v)) {
        if (op == user) {
          continue;
        }
        if (!op_set.contains(user) && !inSameFusionFamily(user, op)) {
          has_external_user = true;
        }
        if (op_set.contains(user)) {
          has_internal_user = true;
        }
      }

      if (has_external_user) {
        results_.push_back(v);
        root_ops_.push_back(op);
        if (!has_internal_user) {
          external_only_results_.push_back(v);
        }
      } else {
        internal_results_.push_back(v);
      }
    }
  }

  for (Operation* op : op_list_) {
    int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
    for (Value value : op->getOperands().take_front(num_input_operand)) {
      if (last_writer_.find(value) != last_writer_.end()) {
        // skip if defining op is in the pattern
        continue;
      }
      input_set.insert(value);
    }
  }

  for (Value v : input_set) operands_.push_back(v);
}

// Create a new fusion pattern from a single op.
FusionPattern::FusionPattern(Operation* op) : FusionPatternBase(op) {
  fusion_type_ = FusionType::kNone;
  dominant_op_ = op;
}

// Create a new fusion pattern from the ops inside the lmhlo fusion op.
FusionPattern::FusionPattern(lmhlo::FusionOp op, ShapeAnalysis* shape_analysis)
    : FusionPatternBase(op) {
  if (shape_analysis != nullptr) {
    FusionType fusionType = FusionType::kNone;
    auto deviceAttr = op->getAttrOfType<StringAttr>(kDiscPlaceAssignment);
    auto fusionTypeAttr =
        op->getAttrOfType<StringAttr>(kDiscFusionTypeAttrName);
    if (fusionTypeAttr) {
      fusionType = fusionTypeFromString(fusionTypeAttr.getValue());
    }
    if (!deviceAttr || fusionType == FusionType::kNone) {
      fusion_type_ = FusionType::kNone;
      dominant_op_ = (size() == 0 ? nullptr : getOpList()[0]);
      return;
    }
    StringRef strategyStr = "base";
    if (fusionType == FusionType::kStitch) {
      strategyStr = "stitch";
    }
    FusionStrategy& strategy =
        getFusionStrategy(deviceAttr.getValue(), strategyStr);
    bool status = strategy.initFusionPattern(*shape_analysis, *this);
    assert(status);
    (void)(status);
  }
}

// Create a new fusion pattern from a valid fusion op list.
FusionPattern::FusionPattern(SmallVectorImpl<Operation*>& op_list)
    : FusionPatternBase(op_list) {
  // just set to a initialized state. It's responsibility of the user to
  // reset the state.
  fusion_type_ = FusionType::kNone;
  dominant_op_ = (size() == 0 ? nullptr : getOpList()[0]);
}

// Merges two fusion patterns and returns the merged pattern. The original
// pattern remains unmodified. The new merged pattern is uninitialized.
FusionPattern FusionPattern::mergeWithoutInit(FusionPattern& other) {
  FusionOpList new_op_list = getOpList();
  new_op_list.insert(new_op_list.end(), other.getOpList().begin(),
                     other.getOpList().end());
  FusionPattern new_fusion_pattern{new_op_list};
  return new_fusion_pattern;
}

void FusionPattern::findOpsOfSkeletonGroup(SkeletonGroup group,
                                           DenseSet<Operation*>& ops) {
  ops.clear();
  // All operators dominanted by `group` can be traced back from skeleton op.
  SmallVector<Operation*> skeletons = group.skeletons;
  DenseSet<Operation*> fusion_ops(op_list_.begin(), op_list_.end());
  DenseSet<Operation*> xroot_members(group.root_member_list.begin(),
                                     group.root_member_list.end());
  std::function<void(Operation*, DenseSet<Operation*>&)> findGroupOps;
  findGroupOps = [&](Operation* op, DenseSet<Operation*>& grp_ops) {
    int64_t num_input_operand = op->getNumOperands() - getNumResultOperands(op);
    for (int64_t i = 0; i < num_input_operand; i++) {
      auto operand = op->getOperand(i);
      auto input_op = findLastWriter(operand);
      // Ops in current group:
      //   1. Op is in `op_list_`;
      //   2. If it is regular-xroot, it can only be owned by current group.
      bool not_owned_regular_xroot = regular_xroots_.contains(input_op) &&
                                     !xroot_members.contains(input_op);
      if (!fusion_ops.contains(input_op) || not_owned_regular_xroot) {
        // TODO: to check operand of fusion?
        continue;
      }
      grp_ops.insert(input_op);
      findGroupOps(input_op, grp_ops);
    }
  };
  for (auto skeleton : skeletons) {
    ops.insert(skeleton);
    findGroupOps(skeleton, ops);
  }
}

bool getOrderedSkeletonGroups(
    FusionPattern& pattern, SmallVector<FusionPattern::SkeletonGroup>& groups) {
  const auto& sub_root_ops = pattern.getSubRootOps();
  DenseSet<Operation*> skeletons(sub_root_ops.begin(), sub_root_ops.end());
  for (auto value : pattern.getExternalOnlyResults()) {
    skeletons.insert(pattern.findLastWriter(value));
  }

  // Find the predecessor xroot-op for every xroot.
  // { xroot, { predecessor xroot } }
  DenseMap<Operation*, DenseSet<Operation*>> xroot_predecessors;
  DenseSet<Operation*> xroot_set;
  const DenseSet<Operation*>& regular_xroots = pattern.getRegularXroots();
  const DenseSet<Operation*>& irregular_xroots = pattern.getIrregularXroots();
  xroot_set.insert(regular_xroots.begin(), regular_xroots.end());
  xroot_set.insert(irregular_xroots.begin(), irregular_xroots.end());
  std::function<void(Operation*, Operation*)> buildXrootPredecessors;
  const auto& op_list = pattern.getOpList();
  DenseSet<Operation*> op_set(op_list.begin(), op_list.end());
  buildXrootPredecessors = [&](Operation* xroot, Operation* op) {
    int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
    for (Value result : op->getOperands().drop_front(num_input_operand)) {
      for (Operation* user : getValueUsers(result)) {
        if (user == op || !op_set.contains(user)) {
          continue;
        }
        if (xroot_set.contains(user)) {
          auto& producers_of_user = xroot_predecessors[user];
          producers_of_user.insert(xroot);
        } else {
          // A DFS process to find predecessors.
          buildXrootPredecessors(xroot, user);
        }
      }
    }
  };
  for (auto xroot : xroot_set) {
    // TODO: rm.. Make sure there is a key for `xroot` in `xroot_predecessors`.
    // auto& predecessors = xroot_predecessors[xroot];
    buildXrootPredecessors(xroot, xroot);
  }

  // Build skeleton groups according to op order in `op_list_`. We assume that
  // the op order in `op_list_` is in topology order. Otherwise, we need to call
  // `sortFusionOpListWithTopologyOrder` to sort op-list ahead.
  groups.clear();
  // Regular xroots can not be replicated in multiple groups, while irregular
  // xroots can be replicated in multiple groups.
  DenseSet<Operation*> visited_regular_xroots;
  std::function<void(Operation*, DenseSet<Operation*>&, DenseSet<Operation*>&)>
      findMembersInSkGroup;
  findMembersInSkGroup = [&](Operation* visiting, DenseSet<Operation*>& members,
                             DenseSet<Operation*>& irregular_members) {
    auto predecessors = xroot_predecessors.find(visiting);
    if (predecessors == xroot_predecessors.end()) {
      return;
    }
    for (auto op : predecessors->second) {
      // for (auto op : xroot_predecessors[visiting]) {
      // Skeleton ops and visited regular-xroot ops stops the traverse.
      // Otherwise `op` is a member of current group.
      if (skeletons.contains(op) || visited_regular_xroots.contains(op)) {
        continue;
      }
      members.insert(op);
      if (irregular_xroots.contains(op)) {
        irregular_members.insert(op);
      } else {
        visited_regular_xroots.insert(op);
      }
      findMembersInSkGroup(op, members, irregular_members);
    }
  };
  // `xroot_inorder` is to help sort operators when building skeleton groups.
  SmallVector<Operation*> xroot_inorder;
  for (auto op : op_list) {
    if (xroot_set.contains(op)) {
      xroot_inorder.push_back(op);
    }
  }
  for (auto op : xroot_inorder) {
    if (!skeletons.contains(op)) {
      continue;
    }
    auto& skeleton = op;
    DenseSet<Operation*> members;
    DenseSet<Operation*> irregular_members;
    // The DFS process guarantees that there is no cycle between two groups.
    findMembersInSkGroup(skeleton, members, irregular_members);
    SmallVector<Operation*> root_member_list;
    for (auto op : xroot_inorder) {
      if (members.contains(op)) {
        root_member_list.push_back(op);
      }
    }
    FusionPattern::SkeletonGroup group;
    group.skeletons = SmallVector<Operation*>({skeleton});
    group.root_member_list = std::move(root_member_list);
    group.irregular_root_member_set = std::move(irregular_members);
    groups.emplace_back(std::move(group));
  }

  return true;
}

bool mergeSkeletonGroupsInOrder(
    FusionPattern& pattern, SmallVector<FusionPattern::SkeletonGroup>& groups,
    ShapeAnalysis* shape_analysis) {
  const auto& op_list = pattern.getOpList();
  DenseMap<Operation*, int64_t> op_to_node_id;
  for (int64_t i = 0; i < op_list.size(); i++) {
    auto op = op_list[i];
    op_to_node_id.try_emplace(op, i);
  }
  GraphCycles cycle_detector(op_list.size());
  for (int64_t node_id = 0; node_id < op_list.size(); node_id++) {
    Operation* op = op_list[node_id];
    for (Value operand : GetAllPossibleUsedValues(op)) {
      Operation* operand_op = pattern.findLastWriter(operand);
      if (operand_op == op) {
        continue;
      }
      // Only consider the operand_op inside the target block.
      auto iter = op_to_node_id.find(operand_op);
      if (iter == op_to_node_id.end()) {
        continue;
      }
      cycle_detector.InsertEdge(iter->second, node_id);
    }
  }

  SmallVector<std::pair<int64_t, FusionPattern::SkeletonGroup>> merged_groups;
  for (auto& group : groups) {
    assert(group.skeletons.size() == 1);
    merged_groups.emplace_back(op_to_node_id[group.skeletons[0]], group);
  }

  // Try to merge skeleton-groups.
  for (int64_t i = 0; i < merged_groups.size(); i++) {
    auto& merged = merged_groups[i];
    auto& merged_grp = merged.second;
    // Only deal with reduce ops currently.
    // TODO: deal with non-reduce ops.
    auto merged_grp_reduce =
        dyn_cast_or_null<lmhlo::ReduceOp>(merged_grp.skeletons[0]);
    // The value of `merged.first` is the group index after merging, which is
    // the index of one op in the group. When one group is merged into another,
    // the index is set to `-1`.
    if (merged.first == -1 || !merged_grp_reduce) {
      continue;
    }
    for (int64_t j = i + 1; j < merged_groups.size(); j++) {
      auto& to_merge = merged_groups[j];
      auto& to_merge_grp = to_merge.second;
      auto to_merge_grp_reduce =
          dyn_cast_or_null<lmhlo::ReduceOp>(to_merge_grp.skeletons[0]);
      if (to_merge.first == -1 || !to_merge_grp_reduce) {
        continue;
      }
      if (!shape_analysis->isShapeEqual(
              merged_grp.skeletons[0]->getOperand(0),
              to_merge_grp.skeletons[0]->getOperand(0))) {
        continue;
      }

      // If one is the producer of the other, they cannot be merged.
      if (cycle_detector.IsReachable(merged.first, to_merge.first) ||
          cycle_detector.IsReachable(to_merge.first, merged.first)) {
        continue;
      }

      auto optional_merged_id =
          TryMergeNode(&cycle_detector, merged.first, to_merge.first);
      if (!optional_merged_id.hasValue()) {
        // It forms a cycle.
        continue;
      }

      // Merge `to_merge_grp` into `merged_grp`.
      merged_grp.skeletons.insert(merged_grp.skeletons.end(),
                                  to_merge_grp.skeletons.begin(),
                                  to_merge_grp.skeletons.end());
      merged_grp.irregular_root_member_set.insert(
          to_merge_grp.irregular_root_member_set.begin(),
          to_merge_grp.irregular_root_member_set.end());
      SmallVector<Operation*> root_member_list;
      DenseSet<Operation*> root_member_set;
      root_member_set.insert(merged_grp.root_member_list.begin(),
                             merged_grp.root_member_list.end());
      root_member_set.insert(to_merge_grp.root_member_list.begin(),
                             to_merge_grp.root_member_list.end());
      for (auto op : op_list) {
        if (root_member_set.contains(op)) {
          root_member_list.push_back(op);
        }
      }
      merged_grp.root_member_list = std::move(root_member_list);

      // Set `to_merge` invalid.
      to_merge.first = -1;
      merged.first = *optional_merged_id;
    }
  }

  std::vector<int32_t> ordered_nodes = cycle_detector.AllNodesInPostOrder();
  std::reverse(ordered_nodes.begin(), ordered_nodes.end());
  SmallVector<FusionPattern::SkeletonGroup> result;
  for (auto id : ordered_nodes) {
    for (auto& group : merged_groups) {
      if (group.first == id) {
        result.push_back(group.second);
        break;
      }
    }
  }
  groups = std::move(result);

  return true;
}

////////////////////// FusionStrategy Implemenation //////////////
//////////////////////////////////////////////////////////////////

namespace {
// global fusion options
std::mutex fusionOptionsMu;
FusionOptions fusionOptions;
}  // namespace

// Update global fusion options. It should be updated before any fusion
// strageties is created. It should be called only once or called with
// exactly same options.
void setGlobalFusionOptions(const FusionOptions& options) {
  std::lock_guard<std::mutex> lock(fusionOptionsMu);
  fusionOptions = options;
}

// Returns the global fusion options.
const FusionOptions& getGlobalFusionOptions() {
  std::lock_guard<std::mutex> lock(fusionOptionsMu);
  return fusionOptions;
}

bool FusionStrategy::isFusible(Operation* op) {
  return mlir::disc_ral::isFusible(op);
}

bool FusionStrategy::isFusible(FusionPattern& fusion_pattern) {
  if (!fusion_pattern.isFusible()) return false;
  for (Operation* op : fusion_pattern.getOpList()) {
    if (!isFusible(op)) return false;
  }
  return true;
}

bool FusionStrategy::tryFuseInplace(ShapeAnalysis& shapeAnalysis,
                                    FusionPattern& lhs, FusionPattern& rhs) {
  // both lhs & rhs should be fusible
  if (!isFusible(lhs) || !isFusible(rhs)) {
    return false;
  }
  FusionPattern result = lhs.mergeWithoutInit(rhs);
  if (!tryFuse(shapeAnalysis, lhs, rhs, result)) {
    return false;
  }
  if (!initFusionPattern(shapeAnalysis, result)) {
    return false;
  }
  lhs = result;
  return true;
}

bool FusionStrategy::tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
                             FusionPattern& rhs, FusionPattern& target) {
  auto& op_list = target.getOpList();
  auto& operands = target.getOperands();
  auto& results = target.getResults();

  if (results.size() + operands.size() >
      options_.max_num_arguments_per_kernel) {
    // some backend devices (e.g. GPU) do not support a kernel with
    // too many arguments.
    return false;
  }

  // We currently do not support a constant op as final output of a fusion
  // pattern.
  // TODO(disc): copy small const in case necessary.
  for (Operation* result_op : target.getRootOps()) {
    if (isa<lmhlo::ConstantOp>(result_op)) {
      return false;
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "FusionStrategy::tryFuse success()\n");
  return true;
}

////////////////////// Base FusionStrategy Implemenation /////////
//////////////////////////////////////////////////////////////////
class BaseFusionStrategy : public FusionStrategy {
 public:
  using FusionStrategy::FusionStrategy;

  using FusionStrategy::isFusible;
  bool isFusible(FusionPattern& fusion_pattern) override;
  bool tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
               FusionPattern& rhs, FusionPattern& target) override;
  bool initFusionPattern(ShapeAnalysis& shapeAnalysis,
                         FusionPattern& fusion_pattern) override;
  virtual StringRef getName() override { return "BaseFusionStrategy"; }

 protected:
  virtual Value getEffectiveShape(FusionPattern& target, Value value) = 0;
  virtual bool checkSameShape(FusionPattern& lhs, FusionPattern& rhs,
                              FusionPattern& target) {
    return false;
  }
};

bool BaseFusionStrategy::isFusible(FusionPattern& fusion_pattern) {
  if (fusion_pattern.isStitchFusion()) return false;
  return FusionStrategy::isFusible(fusion_pattern);
}

bool BaseFusionStrategy::initFusionPattern(ShapeAnalysis& shapeAnalysis,
                                           FusionPattern& fusion_pattern) {
  Operation* inferredDominantOp = nullptr;
  FusionType inferredFusionType = FusionType::kNone;
  for (Operation* op : fusion_pattern.getOpList()) {
    if (isRank2RowReduction(op)) {
      inferredFusionType = FusionType::kRowReduction;
      inferredDominantOp = op;
    } else if (isRank2ColReduction(op)) {
      if (inferredFusionType != FusionType::kRowReduction) {
        inferredFusionType = FusionType::kColReduction;
        inferredDominantOp = op;
      }
    } else if (this->isFusible(op)) {
      // Ignore if already a kRowReduction or kColReduction, otherwise update
      // the fusion type to kLoop and dominant op to current op. This supposes
      // that the last op inside the block is a valid candidate dominant op if
      // the fusion pattern is a kLoop.
      if (inferredFusionType == FusionType::kNone ||
          inferredFusionType == FusionType::kLoop) {
        inferredFusionType = FusionType::kLoop;
        inferredDominantOp = op;
      }
    } else if (!isa<lmhlo::TerminatorOp>(op)) {
      // Not a supported fusionOp, early stop.
      inferredFusionType = FusionType::kNone;
      inferredDominantOp = nullptr;
      break;
    }
  }
  fusion_pattern.setDominantOp(inferredDominantOp);
  fusion_pattern.setFusionType(inferredFusionType);
  return (inferredFusionType != FusionType::kNone && inferredDominantOp);
}

bool BaseFusionStrategy::tryFuse(ShapeAnalysis& shapeAnalysis,
                                 FusionPattern& lhs, FusionPattern& rhs,
                                 FusionPattern& target) {
  if (!FusionStrategy::tryFuse(shapeAnalysis, lhs, rhs, target)) {
    return false;
  }

  auto& op_list = target.getOpList();
  auto& operands = target.getOperands();
  auto& results = target.getResults();
  // ReduceOp can not have consumer within the fusion pattern.
  for (Operation* op : op_list) {
    if (!isa<lmhlo::ReduceOp>(op)) continue;
    int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
    for (Value v : op->getOperands().drop_front(num_input_operand)) {
      for (Operation* user : getValueUsers(v)) {
        if (user == op) continue;
        if (std::find(op_list.begin(), op_list.end(), user) != op_list.end()) {
          return false;
        }
      }
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "check reduce no consumer success()\n");

  // All outputs of a fusion pattern should have compatible shape.
  // Here `compatible` means:
  // - if `to` and `from` are both kInput fusion, all output should have same
  // shape.
  // - otherwise, all output should have same number of elements.

  // No outside users, these ops may be eliminated. We fused it here and let
  // latter pass to do such DCE.
  if (results.empty()) {
    return true;
  }

  Value ref_shape = getEffectiveShape(target, results[0]);
  if (!llvm::all_of(results, [&](Value result) {
        Value shape = getEffectiveShape(target, result);
        return checkSameShape(lhs, rhs, target)
                   ? shapeAnalysis.isShapeEqual(ref_shape, shape)
                   : shapeAnalysis.isSameNumElements(ref_shape, shape);
      })) {
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "BaseFusionStrategy::tryFuse success()\n");
  return true;
}

////////////////////// Base CPU FusionStrategy Implemenation /////////
//////////////////////////////////////////////////////////////////
class BaseCpuFusionStrategy : public BaseFusionStrategy {
 public:
  using BaseFusionStrategy::BaseFusionStrategy;

  bool isFusible(Operation* op) override;
  Value getEffectiveShape(FusionPattern& target, Value v) override;
  bool initFusionPattern(ShapeAnalysis& shapeAnalysis,
                         FusionPattern& fusion_pattern) override;
  bool tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
               FusionPattern& rhs, FusionPattern& target) override;
  virtual StringRef getName() override { return "BaseCpuFusionStrategy"; }
};

bool enableEagerTransposeFusion() {
  static const char* env = getenv("DISC_CPU_ENABLE_EAGER_TRANSPOSE_FUSION");
  if (!env) return false;
  std::string envStr = env;
  std::transform(envStr.begin(), envStr.end(), envStr.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return envStr == "true" || envStr == "1";
}

bool BaseCpuFusionStrategy::isFusible(Operation* op) {
  if (isa<lmhlo::ReshapeOp, lmhlo::DynamicReshapeOp>(op)) {
    return useShapeConstraintIR();
  }

  if (!enableEagerTransposeFusion()) {
    if (auto transposeOp = dyn_cast<lmhlo::TransposeOp>(op)) {
      auto permutation = transposeOp.getPermutation().getValues<int64_t>();
      if (*--permutation.end() != permutation.size() - 1) return false;
    }
  }

  // Do not fuse shape computation.
  if (op->getAttr(kDiscShapeCalcAttr) != nullptr) {
    return false;
  }

  return BaseFusionStrategy::isFusible(op);
}

Value BaseCpuFusionStrategy::getEffectiveShape(FusionPattern& target, Value v) {
  return v;
}

bool BaseCpuFusionStrategy::initFusionPattern(ShapeAnalysis& shapeAnalysis,
                                              FusionPattern& fusion_pattern) {
  Operation* inferredDominantOp = nullptr;
  FusionType inferredFusionType = FusionType::kNone;
  for (Operation* op : fusion_pattern.getOpList()) {
    if (isRowReduction(op)) {
      assert(inferredFusionType == FusionType::kNone);
      inferredFusionType = FusionType::kRowReduction;
      inferredDominantOp = op;
    } else if (isa<lmhlo::ReduceOp>(op)) {
      assert(inferredFusionType == FusionType::kNone);
      inferredFusionType = FusionType::kInput;
      inferredDominantOp = op;
    } else if (this->isFusible(op)) {
      // Ignore if already a kRowReduction or kInput, otherwise update
      // the fusion type to kLoop and dominant op to current op. This supposes
      // that the last op inside the block is a valid candidate dominant op if
      // the fusion pattern is a kLoop.
      if (inferredFusionType == FusionType::kNone ||
          inferredFusionType == FusionType::kLoop) {
        inferredFusionType = FusionType::kLoop;
        inferredDominantOp = op;
      }
    } else if (!isa<lmhlo::TerminatorOp>(op)) {
      // Not a supported fusionOp, early stop.
      inferredFusionType = FusionType::kNone;
      inferredDominantOp = nullptr;
      break;
    }
  }
  auto& roots = fusion_pattern.getRootOps();
  if (roots.size() == 1 && isLargeConcatOp(roots[0])) {
    inferredFusionType = FusionType::kLargeConcat;
  }
  fusion_pattern.setDominantOp(inferredDominantOp);
  fusion_pattern.setFusionType(inferredFusionType);
  return (inferredFusionType != FusionType::kNone && inferredDominantOp);
}

static int getLargeConcatNumOperandsLimit() {
  static const char* env = getenv("DISC_CPU_LARGE_CONCAT_NUM_OPERANDS");
  if (!env) return 32;
  return std::atoi(env);
}

bool isLargeConcatOp(Operation* op) {
  return isa<lmhlo::ConcatenateOp>(op) &&
         op->getNumOperands() >= getLargeConcatNumOperandsLimit();
}

bool isExpandLikeReshape(ShapeAnalysis& shapeAnalysis, Operation* op) {
  auto shapeIRAnalysis =
      dynamic_cast<ShapeConstraintIRAnalysis*>(&shapeAnalysis);
  if (!shapeIRAnalysis) return false;

  Value in = op->getOperand(0);
  Value out = cast<lmhlo::LmhloOp>(op).getResultBuffer();
  auto inTy = in.getType().cast<MemRefType>();
  auto outTy = out.getType().cast<MemRefType>();
  if (inTy.hasStaticShape() && outTy.hasStaticShape()) {
    for (int inIdx = 0, outIdx = 0; true; ++inIdx, ++outIdx) {
      while (inIdx < inTy.getRank() && inTy.getShape()[inIdx] == 1) ++inIdx;
      while (outIdx < outTy.getRank() && outTy.getShape()[outIdx] == 1)
        ++outIdx;
      if ((inIdx == inTy.getRank()) != (outIdx == outTy.getRank()))
        return false;
      if ((inIdx == inTy.getRank()) && (outIdx == outTy.getRank())) return true;
      if (inTy.getShape()[inIdx] != outTy.getShape()[outIdx]) return false;
    }
  }

  auto inSyms =
      getMemRefValueSymbolicDims(shapeIRAnalysis->symbolicDimMgr(), in);
  auto outSyms =
      getMemRefValueSymbolicDims(shapeIRAnalysis->symbolicDimMgr(), out);
  if (!inSyms || !outSyms) return false;

  for (size_t inIdx = 0, outIdx = 0; true; ++inIdx, ++outIdx) {
    while (inIdx < (*inSyms).size() && (*inSyms)[inIdx].getDimSize() == 1)
      ++inIdx;
    while (outIdx < (*outSyms).size() && (*outSyms)[outIdx].getDimSize() == 1)
      ++outIdx;
    if ((inIdx == (*inSyms).size()) != (outIdx == (*outSyms).size()))
      return false;
    if ((inIdx == (*inSyms).size()) && (outIdx == (*outSyms).size()))
      return true;
    if (inTy.getShape()[inIdx] != outTy.getShape()[outIdx]) return false;
  }

  return false;
}

bool BaseCpuFusionStrategy::tryFuse(ShapeAnalysis& shapeAnalysis,
                                    FusionPattern& lhs, FusionPattern& rhs,
                                    FusionPattern& target) {
  if (!BaseFusionStrategy::tryFuse(shapeAnalysis, lhs, rhs, target)) {
    return false;
  }

  auto& op_list = target.getOpList();
  auto& operands = target.getOperands();
  auto& results = target.getResults();

  // Only support expand-like reshape a.t.m.
  for (Operation* op : op_list) {
    if (isa<lmhlo::ReshapeOp, lmhlo::DynamicReshapeOp>(op) &&
        !isExpandLikeReshape(shapeAnalysis, op))
      return false;
  }

  bool has_reduce_root = llvm::any_of(target.getRootOps(), [](Operation* op) {
    return isa<lmhlo::ReduceOp>(op);
  });
  // Here 'large' refer to having many operands.
  bool has_large_concat_root =
      llvm::any_of(target.getRootOps(), isLargeConcatOp);

  // Not support multi output fusion if one root op is a reduce or large concat
  // op.
  if (target.getRootOps().size() > 1 &&
      (has_reduce_root || has_large_concat_root)) {
    return false;
  }

  // large concat can not have consumer within the fusion pattern.
  for (Operation* op : op_list) {
    if (!isLargeConcatOp(op)) continue;
    int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
    for (Value v : op->getOperands().drop_front(num_input_operand)) {
      for (Operation* user : getValueUsers(v)) {
        if (user == op) continue;
        if (std::find(op_list.begin(), op_list.end(), user) != op_list.end()) {
          return false;
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "BaseCpuFusionStrategy::tryFuse success()\n");
  return true;
}

////////////////////// Base GPU FusionStrategy Implemenation /////////
//////////////////////////////////////////////////////////////////
class BaseGpuFusionStrategy : public BaseFusionStrategy {
 public:
  using BaseFusionStrategy::BaseFusionStrategy;

  bool isFusible(Operation* op) override;
  bool checkSameShape(FusionPattern& lhs, FusionPattern& rhs,
                      FusionPattern& target) {
    return lhs.isKInputFusion() && rhs.isKInputFusion();
  }

  Value getEffectiveShape(FusionPattern& target, Value v) override;
  virtual StringRef getName() override { return "BaseGpuFusionStrategy"; }
};

bool BaseGpuFusionStrategy::isFusible(Operation* op) {
  // Only rank-2 tensor -> rank-1 tensor reduction are supported now.
  if (isa<lmhlo::ReduceOp>(op) &&
      (!isRank2RowReduction(op) && !isRank2ColReduction(op)))
    return false;
  return BaseFusionStrategy::isFusible(op);
}

Value BaseGpuFusionStrategy::getEffectiveShape(FusionPattern& target, Value v) {
  Operation* result_op = target.findLastWriter(v);
  assert(result_op);
  // effective shape of reduce op is its operand's shape.
  return isa<lmhlo::ReduceOp>(result_op) ? result_op->getOperand(0) : v;
}

////////////////////// Stitch-Base CPU FusionStrategy Implemenation /////
//////////////////////////////////////////////////////////////////

bool isStitchCpuSupported(Operation* op) {
  // TODO(disc): support fuse scalar const op.

  // Do not fuse shape computation.
  if (op->getAttr(kDiscShapeCalcAttr) != nullptr) {
    return false;
  }

  // All element ops are supported by the fusion codegen engine.
  if (isElementWise(op)) return true;

  // Returns false if not a row reduction.
  // TODO(kevin.zwy): support other reduction types.
  if (isa<lmhlo::ReduceOp>(op) && !isRowReduction(op)) {
    return false;
  }

  // Reshape-like ops
  if (useShapeConstraintIR() &&
      isa<lmhlo::ReshapeOp, lmhlo::DynamicReshapeOp>(op)) {
    return true;
  }

  // clang-format off
  return isa<
    lmhlo::BroadcastInDimOp,
    lmhlo::BroadcastOp,
    lmhlo::DynamicBroadcastInDimOp,
    lmhlo::ReduceOp
  >(op);
  // clang-format on
}

class StitchBaseCpuFusionStrategy : public BaseCpuFusionStrategy {
 public:
  using BaseCpuFusionStrategy::BaseCpuFusionStrategy;

  bool tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
               FusionPattern& rhs, FusionPattern& target) override;
  virtual StringRef getName() override { return "StitchBaseCpuFusionStrategy"; }
};

bool StitchBaseCpuFusionStrategy::tryFuse(ShapeAnalysis& shapeAnalysis,
                                          FusionPattern& lhs,
                                          FusionPattern& rhs,
                                          FusionPattern& target) {
  return llvm::all_of(target.getOpList(), isStitchCpuSupported) &&
         BaseCpuFusionStrategy::tryFuse(shapeAnalysis, lhs, rhs, target);
}

////////////////////// Stitch CPU FusionStrategy Implemenation /////
//////////////////////////////////////////////////////////////////
class StitchCpuFusionStrategy : public FusionStrategy {
 public:
  using FusionStrategy::FusionStrategy;

  bool tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
               FusionPattern& rhs, FusionPattern& target) override;
  bool initFusionPattern(ShapeAnalysis& shapeAnalysis,
                         FusionPattern& fusion_pattern) override;
  virtual StringRef getName() override { return "StitchCpuFusionStrategy"; }
};

bool StitchCpuFusionStrategy::tryFuse(ShapeAnalysis& shapeAnalysis,
                                      FusionPattern& lhs, FusionPattern& rhs,
                                      FusionPattern& target) {
  if (!FusionStrategy::tryFuse(shapeAnalysis, lhs, rhs, target)) {
    return false;
  }

  StitchCPUAnalysis stitchAnalysis(target, shapeAnalysis);
  if (!stitchAnalysis.fusibilityAnalysis()) {
    LLVM_DEBUG(llvm::dbgs() << "fusibilityAnalysis failed\n");
    return false;
  }
  return true;
}

bool StitchCpuFusionStrategy::initFusionPattern(ShapeAnalysis& shapeAnalysis,
                                                FusionPattern& fusion_pattern) {
  fusion_pattern.setFusionType(FusionType::kStitch);
  return true;
}

std::unique_ptr<FusionStrategy> makeNewDeviceStrategy(StringRef device,
                                                      StringRef strategy) {
  auto& options = getGlobalFusionOptions();
  if (device == placement_utils::kCpu && strategy == "base") {
    return std::make_unique<BaseCpuFusionStrategy>(options);
  } else if (device == placement_utils::kGpu && strategy == "base" ||
             device == placement_utils::kGpu && strategy == "stitch_base") {
    return std::make_unique<BaseGpuFusionStrategy>(options);
  } else if (device == placement_utils::kCpu && strategy == "stitch") {
    return std::make_unique<StitchCpuFusionStrategy>(options);
  } else if (device == placement_utils::kGpu && strategy == "stitch") {
    return std::make_unique<StitchGpuFusionStrategy>(options);
  } else if (device == placement_utils::kCpu && strategy == "stitch_base") {
    return std::make_unique<StitchBaseCpuFusionStrategy>(options);
  } else {
    assert(false && "not support fusion strategy");
  }
}

// Returns a process-level fusion strategy singleton.
FusionStrategy& getFusionStrategy(StringRef device, StringRef strategy) {
  LLVM_DEBUG(llvm::dbgs() << "Strategy for: device = " << device
                          << ", strategy = " << strategy << "\n");
  static std::mutex mu;
  static DenseMap<StringRef,
                  DenseMap<StringRef, std::unique_ptr<FusionStrategy>>>
      deviceMap;
  std::lock_guard<std::mutex> lock(mu);
  auto& deviceStrategies = deviceMap[device];
  auto it = deviceStrategies.find(strategy);
  if (it == deviceStrategies.end()) {
    it = deviceStrategies
             .insert(std::make_pair(strategy,
                                    makeNewDeviceStrategy(device, strategy)))
             .first;
  }
  return *it->second;
}

using DeviceStrategyMap = DenseMap<StringRef, FusionStrategy*>;

class PlacementAwareFusionStrategy : public FusionStrategy {
 public:
  PlacementAwareFusionStrategy(const FusionOptions& options,
                               StringRef defaultDevice,
                               DeviceStrategyMap deviceStrategyMap)
      : FusionStrategy(options),
        deviceStrategyMap_(std::move(deviceStrategyMap)),
        defaultDevice_(defaultDevice) {}

  bool isFusible(Operation* op) override;
  bool isFusible(FusionPattern& fusion_pattern) override;
  bool tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
               FusionPattern& rhs, FusionPattern& target) override;
  bool initFusionPattern(ShapeAnalysis& shapeAnalysis,
                         FusionPattern& fusion_pattern) override;
  virtual StringRef getName() override {
    return "PlacementAwareFusionStrategy";
  }

 private:
  StringRef getPlacement(Operation* op);
  StringRef getPlacement(FusionPattern& fusion_pattern) {
    return getPlacement(fusion_pattern.getDominantOp());
  }
  FusionStrategy* getStrategy(StringRef placement);
  FusionStrategy* getStrategy(Operation* op) {
    return getStrategy(getPlacement(op));
  }
  FusionStrategy* getStrategy(FusionPattern& fusion_pattern) {
    return getStrategy(getPlacement(fusion_pattern));
  }

  StringRef defaultDevice_;
  DeviceStrategyMap deviceStrategyMap_;
};

StringRef PlacementAwareFusionStrategy::getPlacement(Operation* op) {
  if (!op) return "";
  auto attr = op->getAttrOfType<StringAttr>(kDiscPlaceAssignment);
  if (attr) return attr.getValue();

  if (auto lmhloOp = dyn_cast<lmhlo::LmhloOp>(op)) {
    auto memorySpace =
        lmhloOp.getResultBuffer().getType().cast<MemRefType>().getMemorySpace();
    if (auto strAttr = memorySpace.dyn_cast<StringAttr>())
      return strAttr.getValue();
  }
  return defaultDevice_;
}

FusionStrategy* PlacementAwareFusionStrategy::getStrategy(StringRef placement) {
  auto it = deviceStrategyMap_.find(placement);
  return (it == deviceStrategyMap_.end()) ? nullptr : it->second;
}

bool PlacementAwareFusionStrategy::isFusible(Operation* op) {
  FusionStrategy* strategy = getStrategy(op);
  return strategy && strategy->isFusible(op);
}

bool PlacementAwareFusionStrategy::isFusible(FusionPattern& fusion_pattern) {
  FusionStrategy* strategy = getStrategy(fusion_pattern);
  return strategy && strategy->isFusible(fusion_pattern);
}

bool PlacementAwareFusionStrategy::tryFuse(ShapeAnalysis& shapeAnalysis,
                                           FusionPattern& lhs,
                                           FusionPattern& rhs,
                                           FusionPattern& target) {
  // lhs & rhs should be on the same device.
  StringRef lhsPlacement = getPlacement(lhs);
  StringRef rhsPlacement = getPlacement(rhs);
  if ((lhsPlacement != rhsPlacement)) {
    return false;
  }
  FusionStrategy* strategy = getStrategy(lhsPlacement);
  return strategy && strategy->tryFuse(shapeAnalysis, lhs, rhs, target);
}

bool PlacementAwareFusionStrategy::initFusionPattern(
    ShapeAnalysis& shapeAnalysis, FusionPattern& fusion_pattern) {
  if (fusion_pattern.getOpList().empty()) return true;
  FusionStrategy* strategy = getStrategy(fusion_pattern.getOpList()[0]);
  return strategy && strategy->initFusionPattern(shapeAnalysis, fusion_pattern);
}

std::unique_ptr<FusionStrategy> makeNewPlacementAwareFusionStrategy(
    bool gpu_enabled, StringRef fusion_strategy) {
  DeviceStrategyMap deviceStrategyMap;
  auto& options = getGlobalFusionOptions();
  StringRef defaultDevice =
      gpu_enabled ? placement_utils::kGpu : placement_utils::kCpu;
  deviceStrategyMap[defaultDevice] =
      &getFusionStrategy(defaultDevice, fusion_strategy);
  return std::make_unique<PlacementAwareFusionStrategy>(
      options, defaultDevice, std::move(deviceStrategyMap));
}

void dumpTilePlan(DenseMap<Value, TileInfo>& tilePlan) {
  for (auto& en : tilePlan) {
    llvm::dbgs() << " pair:";
    llvm::dbgs() << "  value: " << en.first << "\n";
    llvm::dbgs() << "  tile: ";
    for (auto& tile : en.second.tileSizes) {
      llvm::dbgs() << tile.first << " : " << tile.second << ", ";
    }
    llvm::dbgs() << "\n";
  }
}

void StitchCPUAnalysis::dumpParallelPlan() {
  for (const auto& en : parallelPlan_) {
    llvm::dbgs() << " pair@" << en.second.size() << ":";
    llvm::dbgs() << "  value: " << en.first << "\n";
    for (const auto& en2 : llvm::enumerate(en.second)) {
      llvm::dbgs() << "  parallel info #" << en2.index() << ": ";
      int test = en2.value();
      auto& info = parallelInfoStore_[test];
      llvm::dbgs() << "id@prevId: " << info.id << "@" << info.producerId
                   << " || ";
      llvm::dbgs() << "consumerIds: ";
      for (int id : info.consumerIds) llvm::dbgs() << id << " ";
      llvm::dbgs() << " || ";
      for (auto& innerEn : info.indices) {
        llvm::dbgs() << innerEn.first << " : "
                     << parallelIndexStore_[innerEn.second].step << ", ";
      }
      llvm::dbgs() << "\n";
    }
  }
}

bool StitchCPUAnalysis::fusibilityAnalysis() {
  auto& target = fusionPattern_;
  // 0, All ops in the pattern should be supported.
  if (!llvm::all_of(target.getOpList(), isStitchCpuSupported)) {
    LLVM_DEBUG(llvm::dbgs() << "found unsupported op.\n");
    return false;
  }

  // Only support expand-like reshape a.t.m.
  if (!llvm::all_of(target.getOpList(), [&](Operation* op) {
        if (!isa<lmhlo::ReshapeOp, lmhlo::DynamicReshapeOp>(op)) return true;
        auto inTy = op->getOperand(0).getType().cast<MemRefType>();
        auto outTy = cast<lmhlo::LmhloOp>(op)
                         .getResultBuffer()
                         .getType()
                         .cast<MemRefType>();
        // TODO(disc): support rank 0 reshape.
        if ((inTy.hasStaticShape() && inTy.getNumElements() == 1) ||
            (outTy.hasStaticShape() && outTy.getNumElements() == 1))
          return false;
        // TODO(disc): support other kinds of reshapes.
        return isExpandLikeReshape(shapeAnalysis_, op);
      })) {
    LLVM_DEBUG(llvm::dbgs() << "found unsupported op.\n");
    return false;
  }

  // 1, roots analysis
  if (!doRootsAnalysis()) {
    LLVM_DEBUG(llvm::dbgs() << "failed to do roots analysis\n");
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << " dominant value: " << dominantValue_
                          << " for fusion pattern\n");
  LLVM_DEBUG(dumpFusionPattern(target));

  // 2, tile analysis
  if (!doTileAnalysis()) {
    LLVM_DEBUG(llvm::dbgs() << "failed to do tile analysis\n");
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << " tilePlan:\n");
  LLVM_DEBUG(dumpTilePlan(tilePlan_));

  // 3, parallel analysis
  if (!doParallelAnalysis()) {
    LLVM_DEBUG(llvm::dbgs() << "failed to do parallel analysis\n");
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << " parallelPlan:\n");
  LLVM_DEBUG(dumpParallelPlan());

  // 4, sub-roots analysis
  if (!doSubRootsAnalysis()) {
    LLVM_DEBUG(llvm::dbgs() << "failed to do sub-roots analysis\n");
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "doSubRootsAnalysis success\n");

  // 5, scratch memory analysis
  // TODO(disc): there is no capacity limit on CPU since we can use alloc op to
  // alloc new scratch memory, thus we skip scratch memory analysis.

  return true;
}

// Builds value dominant graph. Here buffer `a` dominates buffer `b` means
// `a` is larger than or equal to `b`.
// dominantGraph[a][b] = true iff a dominante b
bool StitchCPUAnalysis::buildDominantGraph(ValueGraph& dominantGraph) {
  DenseSet<Value> allValues;
  for (Operation* op : fusionPattern_.getOpList()) {
    assert(isa<lmhlo::LmhloOp>(op));
    for (Value operand : op->getOperands()) allValues.insert(operand);
  }
  // init graph
  for (Value a : allValues)
    for (Value b : allValues) dominantGraph[a][b] = (a == b);

  // init edges
  for (Operation* op : fusionPattern_.getOpList()) {
    if (isa<lmhlo::ConstantOp>(op)) continue;
    if (isElementWise(op)) {
      // all operands are equal
      for (Value a : op->getOperands())
        for (Value b : op->getOperands()) dominantGraph[a][b] = true;
    } else if (isa<lmhlo::ReduceOp>(op)) {
      Value a = op->getOperand(0);
      Value b = op->getOperand(2);
      dominantGraph[a][b] = true;
    } else if (isa<lmhlo::BroadcastInDimOp, lmhlo::BroadcastOp,
                   lmhlo::DynamicBroadcastInDimOp>(op)) {
      Value a = cast<lmhlo::LmhloOp>(op).getResultBuffer();
      Value b = op->getOperand(0);
      dominantGraph[a][b] = true;
    } else if (isa<lmhlo::ReshapeOp, lmhlo::DynamicReshapeOp>(op)) {
      Value a = cast<lmhlo::LmhloOp>(op).getResultBuffer();
      Value b = op->getOperand(0);
      dominantGraph[a][b] = dominantGraph[b][a] = true;
    } else {
      // failure due to unknown ops
      return false;
    }
  }

  // propagate edges.
  bool changed = false;
  do {
    changed = false;
    for (Value a : allValues)
      for (Value b : allValues)
        if (dominantGraph[a][b])
          for (Value c : allValues)
            if (dominantGraph[b][c] && !dominantGraph[a][c]) {
              dominantGraph[a][c] = true;
              changed = true;
            }
  } while (changed);
  return true;
}

// 1, each root is an output of the fusion pattern.
// 2, find a minimum buffer that dominates all root buffer.
// 3, return false if no buffer is qualified to be a dominant.
bool StitchCPUAnalysis::doRootsAnalysis() {
  auto& dominantGraph = dominantGraph_;
  if (!buildDominantGraph(dominantGraph)) {
    LLVM_DEBUG(llvm::dbgs() << "failed to build dominant graph");
    return false;
  }

  Value& dominant = dominantValue_;
  for (auto&& e : dominantGraph) {
    if (llvm::any_of(fusionPattern_.getResults(),
                     [&](Value v) { return !e.second[v]; }))
      continue;
    if (!dominant || dominantGraph[dominant][e.first]) dominant = e.first;
  }

  return (dominant != nullptr);
}

// Returns false if failed to merge.
bool TileInfo::merge(TileInfo& other) {
  for (auto&& e : other.tileSizes) {
    if (!merge(e.first, e.second)) return false;
  }
  return true;
}

// Returns false if failed to merge.
bool TileInfo::merge(int axis, int tileSize) {
  auto it = tileSizes.find(axis);
  if (it == tileSizes.end()) {
    tileSizes[axis] = tileSize;
    return true;
  }

  int minSize = std::min(it->second, tileSize);
  int maxSize = std::max(it->second, tileSize);
  if (minSize == ShapedType::kDynamicSize) {
    it->second = ShapedType::kDynamicSize;
    return true;
  }

  if (minSize == 0 || maxSize % minSize != 0) return false;
  it->second = maxSize;
  return true;
}

// return true if updated.
bool TileInfo::updateIfNotEqual(TileInfo& other) {
  if (other.tileSizes == tileSizes) return false;
  tileSizes = other.tileSizes;
  return true;
}

// Assigns a tile sizes for each buffers.
// Take buffer `a` memref<?x?x?xf32> as an example:
//  Tile(a) = {0 : -1}: means axis 0 is fully selected as tile
//  Tile(a) = {1 : -1, 2 : 4}: means axis 1 is fully selected and
//                             tile size for axis 2 is 4.
bool StitchCPUAnalysis::doTileAnalysis() {
  bool changed = false;
  auto& tilePlan = tilePlan_;
  do {
    changed = false;
    for (Operation* op : fusionPattern_.getOpList()) {
      if (isa<lmhlo::ConstantOp>(op)) continue;
      if (isElementWise(op)) {
        if (!doElemOpTileAnalysis(tilePlan, op, changed)) return false;
      } else if (isa<lmhlo::ReduceOp>(op)) {
        if (!doReduceOpTileAnalysis(tilePlan, op, changed)) return false;
      } else if (isa<lmhlo::BroadcastInDimOp, lmhlo::BroadcastOp,
                     lmhlo::DynamicBroadcastInDimOp>(op)) {
        if (!doBcastOpTileAnalysis(tilePlan, op, changed)) return false;
      } else if (isa<lmhlo::ReshapeOp, lmhlo::DynamicReshapeOp>(op)) {
        if (!doReshapeOpTileAnalysis(tilePlan, op, changed)) return false;
      } else {
        // failure due to unknown ops
        return false;
      }
    }
  } while (changed);
  return true;
}

bool StitchCPUAnalysis::doElemOpTileAnalysis(
    DenseMap<Value, TileInfo>& tilePlan, Operation* op, bool& changed) {
  // all operands are equal
  TileInfo mergedInfo;
  for (Value a : op->getOperands())
    if (!mergedInfo.merge(tilePlan[a])) {
      LLVM_DEBUG(llvm::dbgs() << "failed to merge tile info for op: " << *op);
      return false;
    }
  for (Value a : op->getOperands())
    changed |= tilePlan[a].updateIfNotEqual(mergedInfo);
  return true;
}

bool StitchCPUAnalysis::doReduceOpTileAnalysis(
    DenseMap<Value, TileInfo>& tilePlan, Operation* op, bool& changed) {
  auto reduce = cast<lmhlo::ReduceOp>(op);
  auto dimensions = reduce.getDimensions().getValues<int64_t>();
  // init value should not do parallel computation.
  // Zero-rank init value does not need to assign a tile info, thus skip it.
  if (auto rank = op->getOperand(1).getType().cast<MemRefType>().getRank()) {
    assert(rank == 1);
    auto& init = tilePlan[op->getOperand(1)];
    TileInfo mergedInfo = init;
    if (!mergedInfo.merge(0)) return false;
    changed |= init.updateIfNotEqual(mergedInfo);
  }

  // apply reduce constraint
  TileInfo inMergedInfo = tilePlan[op->getOperand(0)];
  TileInfo outMergedInfo = tilePlan[op->getOperand(2)];
  for (int64_t d : dimensions)
    if (!inMergedInfo.merge(d)) {
      LLVM_DEBUG(llvm::dbgs() << "failed to merge tile info for op: " << *op);
      return false;
    }

  // input <-> output propagation
  int inRank = op->getOperand(0).getType().cast<MemRefType>().getRank();
  int outIdx = 0;
  for (int d = 0; d < inRank; ++d) {
    // skip reduce dims.
    auto reduce_axis_it = std::find(dimensions.begin(), dimensions.end(), d);
    if (reduce_axis_it != dimensions.end()) continue;

    auto in_it = inMergedInfo.tileSizes.find(d);
    // input -> output
    if (in_it != inMergedInfo.tileSizes.end()) {
      if (!outMergedInfo.merge(outIdx, in_it->second)) return false;
    }
    // output -> input
    auto out_it = outMergedInfo.tileSizes.find(outIdx);
    if (out_it != outMergedInfo.tileSizes.end()) {
      if (!inMergedInfo.merge(d, out_it->second)) return false;
    }
    ++outIdx;
  }
  changed |= tilePlan[op->getOperand(0)].updateIfNotEqual(inMergedInfo);
  changed |= tilePlan[op->getOperand(2)].updateIfNotEqual(outMergedInfo);
  return true;
}

bool StitchCPUAnalysis::doBcastOpTileAnalysis(
    DenseMap<Value, TileInfo>& tilePlan, Operation* op, bool& changed) {
  Value inValue = op->getOperand(0);
  Value outValue = cast<lmhlo::LmhloOp>(op).getResultBuffer();
  // bcast_shape should not do parallel computation.
  if (isa<lmhlo::DynamicBroadcastInDimOp>(op)) {
    auto& bcastShape = tilePlan[op->getOperand(1)];
    TileInfo mergedInfo = bcastShape;
    if (!mergedInfo.merge(0)) return false;
    changed |= bcastShape.updateIfNotEqual(mergedInfo);
  }
  auto dimAttr = op->getAttrOfType<DenseElementsAttr>("broadcast_dimensions");
  assert(dimAttr);
  auto dimensions = dimAttr.getValues<int64_t>();

  // apply reduce constraint
  TileInfo inMergedInfo = tilePlan[inValue];
  TileInfo outMergedInfo = tilePlan[outValue];

  // input <-> output propagation
  int inIdx = 0;
  int outRank = outValue.getType().cast<MemRefType>().getRank();
  for (int d = 0; d < outRank; ++d) {
    // skip bcast dims.
    auto axis_it = std::find(dimensions.begin(), dimensions.end(), d);
    if (axis_it == dimensions.end()) continue;

    auto in_it = inMergedInfo.tileSizes.find(inIdx);
    // input -> output
    if (in_it != inMergedInfo.tileSizes.end()) {
      if (!outMergedInfo.merge(d, in_it->second)) return false;
    }
    // output -> input
    auto out_it = outMergedInfo.tileSizes.find(d);
    if (out_it != outMergedInfo.tileSizes.end()) {
      if (!inMergedInfo.merge(inIdx, out_it->second)) return false;
    }
    ++inIdx;
  }
  changed |= tilePlan[inValue].updateIfNotEqual(inMergedInfo);
  changed |= tilePlan[outValue].updateIfNotEqual(outMergedInfo);

  return true;
}

SmallVector<int> getNonSizeOneDims(MemRefType ty) {
  SmallVector<int> dims;
  for (int i = 0; i < ty.getRank(); ++i)
    if (ty.getShape()[i] != 1) dims.push_back(i);
  return dims;
}

bool StitchCPUAnalysis::doReshapeOpTileAnalysis(
    DenseMap<Value, TileInfo>& tilePlan, Operation* op, bool& changed) {
  Value inValue = op->getOperand(0);
  Value outValue = cast<lmhlo::LmhloOp>(op).getResultBuffer();
  // target_shape should not do parallel computation.
  if (isa<lmhlo::DynamicReshapeOp>(op)) {
    auto& targetShape = tilePlan[op->getOperand(1)];
    TileInfo mergedInfo = targetShape;
    if (!mergedInfo.merge(0)) return false;
    changed |= targetShape.updateIfNotEqual(mergedInfo);
  }

  auto propagateTileDims = [&](Value in, TileInfo& inInfo, Value out,
                               TileInfo& outInfo) {
    auto inTy = in.getType().cast<MemRefType>();
    auto inNonSizeOneDims = getNonSizeOneDims(inTy);
    auto outTy = out.getType().cast<MemRefType>();
    auto outNonSizeOneDims = getNonSizeOneDims(outTy);

    if (inNonSizeOneDims.size() != outNonSizeOneDims.size()) return false;
    for (const auto& z : llvm::zip(inNonSizeOneDims, outNonSizeOneDims)) {
      int inIdx, outIdx;
      std::tie(inIdx, outIdx) = z;
      auto outIt = outInfo.tileSizes.find(outIdx);
      if (outIt != outInfo.tileSizes.end())
        if (!inInfo.merge(inIdx, outIt->second)) return false;
    }
    return true;
  };

  TileInfo inMergedInfo = tilePlan[inValue];
  TileInfo outMergedInfo = tilePlan[outValue];

  if (!propagateTileDims(inValue, inMergedInfo, outValue, outMergedInfo) ||
      !propagateTileDims(outValue, outMergedInfo, inValue, inMergedInfo)) {
    return false;
  }

  changed |= tilePlan[inValue].updateIfNotEqual(inMergedInfo);
  changed |= tilePlan[outValue].updateIfNotEqual(outMergedInfo);
  return true;
}

// Analyzes parallel indices of dominant value to roots & all related producer
// buffers.
bool StitchCPUAnalysis::doParallelAnalysis() {
  // 1, init dominant parallel info
  auto& dominantParallelInfo = makeParallelInfo(dominantValue_);
  // Return failure if no parallel indices for dominant value.
  if (dominantParallelInfo.indices.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "failed due to no dominantParallelInfo\n");
    return false;
  }
  dominantParallelInfo.producerId = dominantParallelInfo.id;
  auto& dominantPlan = parallelPlan_[dominantValue_];
  dominantPlan.insert(dominantParallelInfo.id);

  // 2, propagate dominant parallel info to all roots.
  if (!propagateFromDominantToRoots()) {
    LLVM_DEBUG(llvm::dbgs() << "propagateFromDominantToRoots failed\n");
    return false;
  }

  // 3, back-propagation from roots to their operands.
  if (!propagateFromRootsToProducers()) {
    LLVM_DEBUG(llvm::dbgs() << "propagateFromRootsToProducers failed\n");
    return false;
  }

  return true;
}

// Creates a new parallel index.
ParallelIndex& StitchCPUAnalysis::makeParallelIndex(int64_t step,
                                                    int64_t value) {
  // Return early if a existing const parallel index.
  if (value != ShapedType::kDynamicSize) {
    auto it = constParallelIndexStore_.find(value);
    if (it != constParallelIndexStore_.end())
      return parallelIndexStore_[it->second];
  }

  int id = newSymbolId();
  // Cache const parallel index
  if (value != ShapedType::kDynamicSize) constParallelIndexStore_[value] = id;
  return parallelIndexStore_[id] = ParallelIndex{id, step, value};
}

// Creates a new parallel info.
ParallelInfo& StitchCPUAnalysis::makeParallelInfo(Value value, int producerId,
                                                  Operation* op) {
  auto ty = value.getType().cast<MemRefType>();
  int rank = ty.getRank();
  auto& tileInfo = tilePlan_[value];
  ParallelInfo parallelInfo;
  parallelInfo.value = value;
  parallelInfo.producerId = producerId;
  parallelInfo.op = op;
  for (int d = 0; d < rank; ++d) {
    int64_t dimSize = ty.getShape()[d];
    auto it = tileInfo.tileSizes.find(d);
    if (it == tileInfo.tileSizes.end()) {
      bool isAlwaysZeroIndex =
          (dimSize != ShapedType::kDynamicSize && dimSize <= 1);
      // select the whole dimension
      parallelInfo.indices[d] =
          makeParallelIndex(1, isAlwaysZeroIndex ? 0 : ShapedType::kDynamicSize)
              .id;
    } else if (it->second != ShapedType::kDynamicSize) {
      bool isAlwaysZeroIndex =
          (dimSize != ShapedType::kDynamicSize && dimSize <= it->second);
      // partially select the dimension.
      parallelInfo.indices[d] =
          makeParallelIndex(it->second,
                            isAlwaysZeroIndex ? 0 : ShapedType::kDynamicSize)
              .id;
    }
  }
  parallelInfo.inBound = newSymbolId();
  parallelInfo.isOwner = newSymbolId();
  parallelInfo.id = newSymbolId();
  auto producer_it = parallelInfoStore_.find(producerId);
  if (producer_it != parallelInfoStore_.end()) {
    producer_it->second.consumerIds.insert(parallelInfo.id);
  }
  return parallelInfoStore_[parallelInfo.id] = std::move(parallelInfo);
}

// Propagates dominant parallel info to all roots.
bool StitchCPUAnalysis::propagateFromDominantToRoots() {
  auto& ops = fusionPattern_.getOpList();
  auto& results = fusionPattern_.getResults();
  auto isTargetOp = [&](Operation* op) {
    return llvm::find(ops, op) != ops.end();
  };
  auto isRoot = [&](Value v) {
    return llvm::find(results, v) != results.end();
  };

  int leftRoots = results.size() - isRoot(dominantValue_);
  // No remaining roots, early return.
  if (!leftRoots) return true;

  // 1, propagate from dominant to roots
  SmallVector<Value> toProcess{dominantValue_};
  DenseSet<Value> processedSet{dominantValue_};
  auto updateToProcess = [&](Value other, int id) {
    leftRoots -= isRoot(other);
    toProcess.push_back(other);
    parallelPlan_[other].insert(id);
  };
  while (!toProcess.empty()) {
    Value v = toProcess.pop_back_val();
    auto& plan = parallelPlan_[v];
    assert(plan.size() == 1);
    // clone the info since `parallelInfoStore_` maybe modified during following
    // procedure.
    auto info = parallelInfoStore_[*plan.begin()];
    for (Operation* user : getValueUsers(v)) {
      if (!isTargetOp(user)) continue;
      if (isa<lmhlo::ConstantOp>(user)) continue;
      if (isElementWise(user)) {
        for (Value other : user->getOperands()) {
          if (v == other || !processedSet.insert(other).second) continue;
          auto& otherInfo = makeParallelInfo(other, info.id, user);
          otherInfo.indices = info.indices;
          otherInfo.inBound = info.inBound;
          otherInfo.isOwner = info.isOwner;
          updateToProcess(other, otherInfo.id);
        }
      } else if (auto reduce = dyn_cast<lmhlo::ReduceOp>(user)) {
        Value other = user->getOperand(2);
        if (v != user->getOperand(0) || !processedSet.insert(other).second)
          continue;
        auto& otherInfo = makeParallelInfo(other, info.id, user);
        otherInfo.inBound = info.inBound;
        otherInfo.isOwner = info.isOwner;
        otherInfo.indices.clear();
        auto dimensions = reduce.getDimensions().getValues<int64_t>();
        int rank = v.getType().cast<MemRefType>().getRank();
        int outIdx = 0;
        for (int d = 0; d < rank; ++d) {
          if (llvm::find(dimensions, d) != dimensions.end()) continue;
          auto it = info.indices.find(d);
          if (it != info.indices.end()) {
            otherInfo.indices[outIdx] = it->second;
          }
          ++outIdx;
        }
        updateToProcess(other, otherInfo.id);
      } else if (isa<lmhlo::BroadcastInDimOp, lmhlo::BroadcastOp,
                     lmhlo::DynamicBroadcastInDimOp>(user)) {
        Value other = user->getOperand(0);
        Value result = cast<lmhlo::LmhloOp>(user).getResultBuffer();
        if (v != result || !processedSet.insert(other).second) continue;

        auto& otherInfo = makeParallelInfo(other, info.id, user);
        otherInfo.inBound = info.inBound;
        otherInfo.indices.clear();

        int inIdx = 0;
        int rank = result.getType().cast<MemRefType>().getRank();
        auto dimAttr =
            user->getAttrOfType<DenseElementsAttr>("broadcast_dimensions");
        assert(dimAttr);
        auto dimensions = dimAttr.getValues<int64_t>();
        for (int d = 0; d < rank; ++d) {
          if (llvm::find(dimensions, d) == dimensions.end()) continue;
          auto it = info.indices.find(d);
          if (it != info.indices.end()) {
            otherInfo.indices[inIdx] = it->second;
          }
          ++inIdx;
        }
        updateToProcess(other, otherInfo.id);
      } else if (isa<lmhlo::ReshapeOp, lmhlo::DynamicReshapeOp>(user)) {
        Value inValue = user->getOperand(0);
        Value outValue = cast<lmhlo::LmhloOp>(user).getResultBuffer();
        // skip the second operand (target shape) of dynamic reshape op.
        if (v != inValue && v != outValue) continue;
        Value from = inValue;
        Value to = outValue;
        if (v == outValue) std::swap(from, to);
        if (!processedSet.insert(to).second) continue;
        auto& toInfo = makeParallelInfo(to, info.id, user);
        toInfo.inBound = info.inBound;
        toInfo.isOwner = info.isOwner;
        auto fromTy = from.getType().cast<MemRefType>();
        auto fromNonSizeOneDims = getNonSizeOneDims(fromTy);
        auto toTy = to.getType().cast<MemRefType>();
        auto toNonSizeOneDims = getNonSizeOneDims(toTy);
        for (const auto& z : llvm::zip(fromNonSizeOneDims, toNonSizeOneDims)) {
          int fromIdx, toIdx;
          std::tie(fromIdx, toIdx) = z;
          auto it = toInfo.indices.find(toIdx);
          if (it == toInfo.indices.end()) continue;
          toInfo.indices[toIdx] = info.indices[fromIdx];
        }
        updateToProcess(to, toInfo.id);
      } else {
        // failure due to unknown ops
        return false;
      }
      if (!leftRoots) break;
    }
    if (!leftRoots) break;
  }

  // 2, clear the indices that are not directly depended by any root.
  DenseSet<int> dependentIds;
  for (Value v : results) {
    auto& plan = parallelPlan_[v];
    assert(plan.size() == 1);
    auto& info = parallelInfoStore_[*plan.begin()];
    int id = info.id;
    int prevId = parallelInfoStore_[id].producerId;
    dependentIds.insert(id);
    while (id != prevId) {
      id = prevId;
      prevId = parallelInfoStore_[id].producerId;
      dependentIds.insert(id);
    }
  }
  for (auto&& e : parallelPlan_) {
    auto copiedId = e.second;
    for (int id : copiedId)
      if (!dependentIds.count(id)) {
        e.second.erase(id);
        auto& info = parallelInfoStore_[id];
        auto& prevInfo = parallelInfoStore_[info.producerId];
        prevInfo.consumerIds.erase(id);
      }
  }

  LLVM_DEBUG(llvm::dbgs() << " parallelPlan after dominant to roots:\n");
  LLVM_DEBUG(dumpParallelPlan());
  return true;
}

// Back-propagation from roots to their operands.
bool StitchCPUAnalysis::propagateFromRootsToProducers() {
  // Back-propagation from roots to their operands.
  auto& ops = fusionPattern_.getOpList();
  auto& results = fusionPattern_.getResults();
  auto isTargetOp = [&](Operation* op) {
    return llvm::find(ops, op) != ops.end();
  };
  SmallVector<std::pair<Operation*, int>> toProcessIds;
  DenseMap<Operation*, DenseSet<int>> processedIds;
  for (Value root : results) {
    auto& plan = parallelPlan_[root];
    assert(plan.size() == 1);
    auto& info = parallelInfoStore_[*plan.begin()];
    toProcessIds.emplace_back(fusionPattern_.findLastWriter(root), info.id);
  }
  while (!toProcessIds.empty()) {
    Operation* op;
    int id;
    std::tie(op, id) = toProcessIds.pop_back_val();
    if (!processedIds[op].insert(id).second) continue;
    LLVM_DEBUG(llvm::dbgs()
               << "back-root-indices for #" << id << " of op: " << *op << "\n");
    parallelInfoStore_[id].consumedByRoots = true;
    // No need to do propagation for const ops
    if (isa<lmhlo::ConstantOp>(op)) continue;
    Value out = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    auto isExistingInputId = [&](Value in, int& inId) {
      auto& info = parallelInfoStore_[id];
      auto& prevInfo = parallelInfoStore_[info.producerId];
      for (int consumerId : parallelPlan_[in]) {
        auto& nextInfo = parallelInfoStore_[consumerId];
        if (nextInfo.producerId == id) {
          inId = nextInfo.id;
          return true;
        }
      }
      if (prevInfo.value == in) {
        inId = prevInfo.id;
        return true;
      }
      return false;
    };
    if (isElementWise(op)) {
      for (Value in : op->getOperands().drop_back()) {
        int inId;
        if (!isExistingInputId(in, inId)) {
          auto& inInfo = makeParallelInfo(in, id, op);
          auto& info = parallelInfoStore_[id];
          inInfo.indices = info.indices;
          inInfo.inBound = info.inBound;
          inInfo.isOwner = info.isOwner;
          parallelPlan_[in].insert(inInfo.id);
          inId = inInfo.id;
          LLVM_DEBUG(llvm::dbgs()
                     << "elemwise back-root-indices value: " << in << "\n\t"
                     << "from @@" << info.id << "@@"
                     << ": indices.size = " << info.indices.size() << "\n\t"
                     << "to @@" << inId << "@@"
                     << ": indices.size = " << inInfo.indices.size() << "\n");
        }
        auto lastWriter = fusionPattern_.findLastWriter(in);
        if (isTargetOp(lastWriter)) {
          toProcessIds.emplace_back(lastWriter, inId);
        }
      }
    } else if (auto reduce = dyn_cast<lmhlo::ReduceOp>(op)) {
      Value in = op->getOperand(0);
      Value init = op->getOperand(1);
      // propagate to init value
      auto& initInfo = makeParallelInfo(init, id, op);
      auto& info = parallelInfoStore_[id];
      initInfo.inBound = info.inBound;
      parallelPlan_[init].insert(initInfo.id);
      // propagate to data value
      int inId;
      if (!isExistingInputId(in, inId)) {
        auto& inInfo = makeParallelInfo(in, id, op);
        auto& info = parallelInfoStore_[id];
        inInfo.inBound = info.inBound;
        inInfo.isOwner = info.isOwner;
        inInfo.indices.clear();
        auto dimensions = reduce.getDimensions().getValues<int64_t>();
        int rank = in.getType().cast<MemRefType>().getRank();
        int outIdx = 0;
        for (int d = 0; d < rank; ++d) {
          if (llvm::find(dimensions, d) != dimensions.end()) continue;
          auto it = info.indices.find(outIdx);
          if (it != info.indices.end()) {
            inInfo.indices[d] = it->second;
          }
          ++outIdx;
        }
        parallelPlan_[in].insert(inInfo.id);
        inId = inInfo.id;
      }
      auto lastWriter = fusionPattern_.findLastWriter(in);
      if (isTargetOp(lastWriter)) {
        toProcessIds.emplace_back(lastWriter, inId);
      }
    } else if (isa<lmhlo::BroadcastInDimOp, lmhlo::BroadcastOp,
                   lmhlo::DynamicBroadcastInDimOp>(op)) {
      Value in = op->getOperand(0);
      if (isa<lmhlo::DynamicBroadcastInDimOp>(op)) {
        Value shapeOperand = op->getOperand(1);
        auto& shapeInfo = makeParallelInfo(shapeOperand, id, op);
        auto& info = parallelInfoStore_[id];
        shapeInfo.inBound = info.inBound;
        parallelPlan_[shapeOperand].insert(shapeInfo.id);
      }
      int inId;
      if (!isExistingInputId(in, inId)) {
        auto& inInfo = makeParallelInfo(in, id, op);
        auto& info = parallelInfoStore_[id];
        inInfo.inBound = info.inBound;
        inInfo.indices.clear();

        int inIdx = 0;
        int rank = out.getType().cast<MemRefType>().getRank();
        auto dimAttr =
            op->getAttrOfType<DenseElementsAttr>("broadcast_dimensions");
        assert(dimAttr);
        auto dimensions = dimAttr.getValues<int64_t>();
        for (int d = 0; d < rank; ++d) {
          if (llvm::find(dimensions, d) == dimensions.end()) continue;
          auto it = info.indices.find(d);
          if (it != info.indices.end()) {
            inInfo.indices[inIdx] = it->second;
          }
          ++inIdx;
        }
        parallelPlan_[in].insert(inInfo.id);
        inId = inInfo.id;
      }
      auto lastWriter = fusionPattern_.findLastWriter(in);
      if (isTargetOp(lastWriter)) {
        toProcessIds.emplace_back(lastWriter, inId);
      }
    } else if (isa<lmhlo::ReshapeOp, lmhlo::DynamicReshapeOp>(op)) {
      Value in = op->getOperand(0);
      if (isa<lmhlo::DynamicReshapeOp>(op)) {
        Value shapeOperand = op->getOperand(1);
        auto& shapeInfo = makeParallelInfo(shapeOperand, id, op);
        auto& info = parallelInfoStore_[id];
        shapeInfo.inBound = info.inBound;
        parallelPlan_[shapeOperand].insert(shapeInfo.id);
      }
      int inId;
      if (!isExistingInputId(in, inId)) {
        auto& inInfo = makeParallelInfo(in, id, op);
        auto& info = parallelInfoStore_[id];
        inInfo.inBound = info.inBound;
        inInfo.isOwner = info.isOwner;

        auto inTy = in.getType().cast<MemRefType>();
        auto inNonSizeOneDims = getNonSizeOneDims(inTy);
        auto outTy = out.getType().cast<MemRefType>();
        auto outNonSizeOneDims = getNonSizeOneDims(outTy);
        for (const auto& z : llvm::zip(inNonSizeOneDims, outNonSizeOneDims)) {
          int inIdx, outIdx;
          std::tie(inIdx, outIdx) = z;
          auto it = inInfo.indices.find(inIdx);
          if (it == inInfo.indices.end()) continue;
          inInfo.indices[inIdx] = info.indices[outIdx];
        }

        parallelPlan_[in].insert(inInfo.id);
        inId = inInfo.id;
      }
      auto lastWriter = fusionPattern_.findLastWriter(in);
      if (isTargetOp(lastWriter)) {
        toProcessIds.emplace_back(lastWriter, inId);
      }
    } else {
      // failure due to unknown ops
      return false;
    }
  }
  return true;
}

// Returns true if the  parallelInfo id set is consistent.
bool StitchCPUAnalysis::isConsistentParallelInfoSet(DenseSet<int>& idSet) {
  if (idSet.size() <= 1) return true;
  auto& refInfo = parallelInfoStore_[*idSet.begin()];
  for (auto id : idSet) {
    if (id == refInfo.id) continue;
    auto& info = parallelInfoStore_[id];
    if (info.indices != refInfo.indices) return false;
  }
  return true;
}

// Sub-roots analysis. cache some intermediate results to avoid expensive
// re-computation.
bool StitchCPUAnalysis::doSubRootsAnalysis() {
  // TODO(disc): implement sub-roots analysis.
  // Currently only using a very Currently simple strategy.
  subRootsAndRootsSet_.insert(fusionPattern_.getResults().begin(),
                              fusionPattern_.getResults().end());

  DenseSet<Value> candidates;
  candidates.insert(fusionPattern_.getInternalResults().begin(),
                    fusionPattern_.getInternalResults().end());
  candidates.insert(fusionPattern_.getResults().begin(),
                    fusionPattern_.getResults().end());

  for (Operation* op : fusionPattern_.getOpList()) {
    if (isa<lmhlo::ReduceOp>(op)) {
      Value in = op->getOperand(0);
      Value out = op->getOperand(2);
      subRootsAndRootsSet_.insert(out);
      if (candidates.find(in) != candidates.end())
        subRootsAndRootsSet_.insert(in);
    } else if (isa<lmhlo::ExpOp, lmhlo::SqrtOp, lmhlo::RsqrtOp>(op)) {
      Value out = op->getOperand(1);
      subRootsAndRootsSet_.insert(out);
    } else if (isa<lmhlo::BroadcastInDimOp, lmhlo::BroadcastOp,
                   lmhlo::DynamicBroadcastInDimOp>(op)) {
      Value in = op->getOperand(0);
      // TODO(disc): use isSmallCpuBuffer
      auto isSmallTile = [&](Value v) {
        auto& tileInfo = tilePlan_[v];
        auto ty = v.getType().cast<MemRefType>();
        if (tileInfo.tileSizes.empty()) return true;
        int64_t totalSize = 1;
        for (auto& en : tileInfo.tileSizes) {
          int64_t tileSize = en.second;
          if (tileSize == ShapedType::kDynamicSize) {
            tileSize = ty.getShape()[en.first];
          }
          if (tileSize == ShapedType::kDynamicSize) return false;
          totalSize *= tileSize;
        }
        return totalSize == 1;
      };
      bool isSmall = isSmallTile(in);
      if (candidates.find(in) != candidates.end() && isSmall)
        subRootsAndRootsSet_.insert(in);
    }
  }

  for (Value v : subRootsAndRootsSet_) {
    if (!isConsistentParallelInfoSet(parallelPlan_[v])) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Value: " << v << " has inconsistent parallel info.\n");
      LLVM_DEBUG(llvm::dbgs() << " parallelPlan:\n");
      LLVM_DEBUG(dumpParallelPlan());
      return false;
    }
  }
  return true;
}

ParallelInfo& StitchCPUAnalysis::getDominantParallelInfo() {
  Value dominant = getDominantValue();
  assert(dominant);
  auto& plan = parallelPlan_[dominant];
  assert(!plan.empty());

  for (int id : plan) {
    auto& info = parallelInfoStore_[id];
    if (info.producerId == id) return info;
  }
  assert(false && "no parallel info for dominant value");
}

scf::ParallelOp StitchCPUAnalysis::emitTileParallelLoop(OpBuilder& b,
                                                        Location loc) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value dominant = getDominantValue();
  auto& info = getDominantParallelInfo();
  auto& indexStore = getParallelIndexStore();
  int numParallelIndices = info.indices.size();
  SmallVector<Value> lbs(numParallelIndices, zero);
  SmallVector<Value> ubs;
  SmallVector<Value> steps;
  SmallVector<int> parallelAxes;
  for (auto&& e : info.indices) parallelAxes.push_back(e.first);
  llvm::sort(parallelAxes);
  for (int axis : parallelAxes) {
    ubs.push_back(b.create<memref::DimOp>(loc, dominant, axis));
    steps.push_back(b.create<arith::ConstantIndexOp>(
        loc, indexStore[info.indices[axis]].step));
  }
  SmallVector<Value, 2> vars;
  return createParallelAndSetInsPt(b, loc, vars, lbs, ubs, steps, {});
}

bool StitchCPUAnalysis::emitParallelIndices(OpBuilder& b, Location loc,
                                            ValueRange dominantIndex) {
  assert(!dominantIndex.empty());
  auto& info = getDominantParallelInfo();
  // Set parallel indices for dominant value
  info.symbolIndices.insert(info.symbolIndices.end(), dominantIndex.begin(),
                            dominantIndex.end());
  // Set in-bound-check pred & is-owner pred
  Value trueValue = b.create<arith::ConstantIntOp>(loc, 1, 1);
  Value falseValue = b.create<arith::ConstantIntOp>(loc, 0, 1);
  info.symbolInBound = info.symbolIsOwner = trueValue;

  // Propagate dominant indices to other values.
  SmallVector<int> toProcess{info.id};
  DenseSet<int> processedIds{info.id};
  while (!toProcess.empty()) {
    int id = toProcess.pop_back_val();
    auto& from = parallelInfoStore_[id];
    LLVM_DEBUG(llvm::dbgs()
               << "infer parallel indices from: " << id
               << " with #consumers = " << from.consumerIds.size() << "\n");
    for (int toId : from.consumerIds) {
      LLVM_DEBUG(llvm::dbgs() << "  consumer id: " << toId << "\n");
      if (!processedIds.insert(toId).second) continue;
      auto& to = parallelInfoStore_[toId];
      Operation* op = to.op;
      LLVM_DEBUG(llvm::dbgs() << "  to.op = " << *op << "\n");
      assert(op);
      if (isa<lmhlo::ConstantOp>(op)) {
        // Only scalar const is fusible, and it should have no parallel indices.
        assert(to.indices.empty());
        to.symbolInBound = trueValue;
        // Constant op can not the final output of the fusion pattern.
        to.symbolIsOwner = falseValue;
        continue;
      } else if (isElementWise(op)) {
        if (!emitElemOpParallelIndex(b, loc, from, to)) {
          LLVM_DEBUG(llvm::dbgs() << "failed to emitElemOpParallelIndex\n");
          return false;
        }
      } else if (isa<lmhlo::ReduceOp>(op)) {
        if (!emitReduceOpParallelIndex(b, loc, from, to)) {
          LLVM_DEBUG(llvm::dbgs() << "failed to emitReduceOpParallelIndex\n");
          return false;
        }
      } else if (isa<lmhlo::BroadcastInDimOp, lmhlo::BroadcastOp,
                     lmhlo::DynamicBroadcastInDimOp>(op)) {
        if (!emitBcastOpParallelIndex(b, loc, from, to)) {
          LLVM_DEBUG(llvm::dbgs() << "failed to emitBcastOpParallelIndex\n");
          return false;
        }
      } else if (isa<lmhlo::ReshapeOp, lmhlo::DynamicReshapeOp>(op)) {
        if (!emitReshapeOpParallelIndex(b, loc, from, to)) {
          LLVM_DEBUG(llvm::dbgs() << "failed to emitBcastOpParallelIndex\n");
          return false;
        }

      } else {
        // failure due to unknown ops
        LLVM_DEBUG(llvm::dbgs() << "unknown op\n");
        return false;
      }
      toProcess.push_back(to.id);
    }
  }
  return true;
}

bool StitchCPUAnalysis::emitElemOpParallelIndex(OpBuilder& b, Location loc,
                                                ParallelInfo& from,
                                                ParallelInfo& to) {
  to.symbolIndices = from.symbolIndices;
  to.symbolInBound = from.symbolInBound;
  to.symbolIsOwner = from.symbolIsOwner;
  return true;
}

Value allIndicesZeros(OpBuilder& b, Location loc, ValueRange indices) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value trueValue = b.create<arith::ConstantIntOp>(loc, 1, 1);
  Value allParallelIndicesZeros = trueValue;
  for (Value idx : indices) {
    Value indexIsZero =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, idx, zero);
    allParallelIndicesZeros =
        b.create<arith::AndIOp>(loc, allParallelIndicesZeros, indexIsZero);
  }
  return allParallelIndicesZeros;
}

bool StitchCPUAnalysis::emitReduceOpParallelIndex(OpBuilder& b, Location loc,
                                                  ParallelInfo& from,
                                                  ParallelInfo& to) {
  Operation* op = to.op;
  assert(op);
  auto reduce = cast<lmhlo::ReduceOp>(op);
  if (op->getNumOperands() != 3) {
    LLVM_DEBUG(llvm::dbgs() << "multi-inputs reduction is not supported\n");
    return false;
  }
  Value in = op->getOperand(0);
  Value init = op->getOperand(1);
  Value out = op->getOperand(2);

  if (from.value != in && from.value != out) {
    LLVM_DEBUG(llvm::dbgs() << "from.value: " << from.value << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "init input of reduce op should not have parallel indices\n");
    return false;
  }

  if (to.value == init) {
    assert(to.indices.empty());
    Value trueValue = b.create<arith::ConstantIntOp>(loc, 1, 1);
    to.symbolInBound = trueValue;
    to.symbolIsOwner = allIndicesZeros(b, loc, from.symbolIndices);
    return true;
  }
  assert(to.value == in || to.value == out);
  to.symbolIndices = from.symbolIndices;
  to.symbolInBound = from.symbolInBound;
  to.symbolIsOwner = from.symbolIsOwner;
  return true;
}

bool StitchCPUAnalysis::emitBcastOpParallelIndex(OpBuilder& b, Location loc,
                                                 ParallelInfo& from,
                                                 ParallelInfo& to) {
  Operation* op = to.op;
  assert(op);
  Value in = op->getOperand(0);
  Value out = cast<lmhlo::LmhloOp>(op).getResultBuffer();

  if (to.value == out || from.value != out) {
    LLVM_DEBUG(llvm::dbgs()
               << "bcast input -> output index inference is invalid.\n");
    return false;
  } else if (to.value != in) {
    // shape operands
    assert(isa<lmhlo::DynamicBroadcastInDimOp>(op));
    Value trueValue = b.create<arith::ConstantIntOp>(loc, 1, 1);
    Value falseValue = b.create<arith::ConstantIntOp>(loc, 0, 1);
    to.symbolInBound = trueValue;
    // 1, shape operand should not be the output of the fusion pattern.
    to.symbolIsOwner = falseValue;
    return true;
  }

  assert(from.value == out && to.value == in);
  int inIdx = 0;
  int outParallelIdx = 0;
  int rank = out.getType().cast<MemRefType>().getRank();
  auto dimAttr = op->getAttrOfType<DenseElementsAttr>("broadcast_dimensions");
  assert(dimAttr);
  auto dimensions = dimAttr.getValues<int64_t>();
  Value isOwner = from.symbolIsOwner;
  for (int d = 0; d < rank; ++d) {
    auto it = from.indices.find(d);
    if (llvm::find(dimensions, d) == dimensions.end()) {
      outParallelIdx += (it != from.indices.end());
      continue;
    }

    if (it != from.indices.end()) {
      Value inDimValue = b.create<memref::DimOp>(loc, in, inIdx);
      Value outDimValue = b.create<memref::DimOp>(loc, out, d);
      Value isEqual = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              inDimValue, outDimValue);
      Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
      to.symbolIndices.push_back(b.create<mlir::arith::SelectOp>(
          loc, isEqual, from.symbolIndices[outParallelIdx], zero));
      Value isZero =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                  from.symbolIndices[outParallelIdx], zero);
      Value isAxisOwner = b.create<arith::OrIOp>(loc, isEqual, isZero);
      isOwner = b.create<arith::AndIOp>(loc, isOwner, isAxisOwner);
      ++outParallelIdx;
    }
    ++inIdx;
  }

  to.symbolInBound = from.symbolInBound;
  to.symbolIsOwner = isOwner;
  return true;
}

bool StitchCPUAnalysis::emitReshapeOpParallelIndex(OpBuilder& b, Location loc,
                                                   ParallelInfo& from,
                                                   ParallelInfo& to) {
  Operation* op = to.op;
  assert(op);
  Value in = op->getOperand(0);
  Value out = cast<lmhlo::LmhloOp>(op).getResultBuffer();

  if (from.value != in && from.value != out) {
    LLVM_DEBUG(llvm::dbgs() << "from.value: " << from.value << "\n");
    LLVM_DEBUG(
        llvm::dbgs()
        << "shape input of reshape op should not have parallel indices\n");
    return false;
  }

  if (to.value != in && to.value != out) {
    // to is the target shape operand
    assert(to.indices.empty());
    Value trueValue = b.create<arith::ConstantIntOp>(loc, 1, 1);
    to.symbolInBound = trueValue;
    to.symbolIsOwner = allIndicesZeros(b, loc, from.symbolIndices);
    return true;
  }

  to.symbolInBound = from.symbolInBound;
  to.symbolIsOwner = from.symbolIsOwner;

  auto fromTy = from.value.getType().cast<MemRefType>();
  SmallVector<Value> fromNonSizeOneIndices;
  for (const auto& en : llvm::enumerate(from.getSortedParallelAxes())) {
    if (fromTy.getShape()[en.value()] == 1) continue;
    fromNonSizeOneIndices.push_back(from.symbolIndices[en.index()]);
  }

  int numNonSizeOneDim = 0;
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  auto toTy = to.value.getType().cast<MemRefType>();
  for (const auto& en : llvm::enumerate(to.getSortedParallelAxes())) {
    if (toTy.getShape()[en.value()] == 1) {
      to.symbolIndices.push_back(zero);
    } else {
      to.symbolIndices.push_back(fromNonSizeOneIndices[numNonSizeOneDim++]);
    }
  }

  return true;
}

using ValueViewStore = DenseMap<SmallVector<Value>, Value>;
using ViewStore = DenseMap<Value, ValueViewStore>;

bool StitchCPUAnalysis::emitInOutTiles(OpBuilder& b, Location loc,
                                       ViewStore& viewStore) {
  auto getIntAttr = [&](int64_t val) {
    return b.getIntegerAttr(b.getIndexType(), val);
  };
  SmallVector<Value> inOuts;
  inOuts.insert(inOuts.end(), fusionPattern_.getOperands().begin(),
                fusionPattern_.getOperands().end());
  inOuts.insert(inOuts.end(), fusionPattern_.getResults().begin(),
                fusionPattern_.getResults().end());
  SymbolicDimMgr* mgr = nullptr;
  if (auto shapeIRAnalysis =
          dynamic_cast<ShapeConstraintIRAnalysis*>(&shapeAnalysis_)) {
    mgr = &shapeIRAnalysis->symbolicDimMgr();
  }
  for (Value in : inOuts) {
    auto& valueViewStore = viewStore[in];
    auto& tileInfo = tilePlan_[in];
    auto inSymbols = getMemRefValueSymbolicDimRefs(in);
    for (int id : parallelPlan_[in]) {
      auto& parallelInfo = parallelInfoStore_[id];
      auto& indices = parallelInfo.symbolIndices;
      auto it = valueViewStore.find(indices);
      if (it != valueViewStore.end()) continue;
      auto inType = in.getType().cast<MemRefType>();
      int rank = inType.getRank();

      // basic logic is:
      //   %inputView = if (%in_bound) {
      //     1, suppose i, j are the parallel indices
      //     2, original buffer has type: memref<?x?x?x?xf32>
      //     return input[i][j][0:n][0:m] : memref<1x1x?x?xf32>
      //   } else {
      //     return input[0][0][0:n][0:m] : memref<1x1x?x?xf32>
      //   }
      // We will never touch the subview that is out of bound, thus the above
      // logic can be simplified to:
      //   %inputView = input[i][j][0:n][0:m] : memref<1x1x?x?xf32>

      bool staticShape = true;
      SmallVector<OpFoldResult> subViewSizes;
      SmallVector<OpFoldResult> subViewOffsets;
      SmallVector<OpFoldResult> subViewSteps;
      SmallVector<SymbolicDimOp> subViewSymbols;
      int parallelIdx = 0;
      for (int d = 0; d < rank; ++d) {
        subViewSteps.push_back(getIntAttr(1));
        auto parallelIt = parallelInfo.indices.find(d);
        auto tileIt = tileInfo.tileSizes.find(d);
        if (parallelIt != parallelInfo.indices.end()) {
          auto& parallelIndex = parallelIndexStore_[parallelIt->second];
          subViewSizes.push_back(getIntAttr(parallelIndex.step));
          subViewOffsets.push_back(parallelInfo.symbolIndices[parallelIdx]);
          if (inSymbols && mgr) {
            subViewSymbols.push_back(
                mgr->newConstantSymbolicDim(parallelIndex.step));
          }
          ++parallelIdx;
        } else if (tileIt != tileInfo.tileSizes.end()) {
          subViewOffsets.push_back(getIntAttr(0));
          if (tileIt->second == ShapedType::kDynamicSize) {
            if (inType.getShape()[d] == ShapedType::kDynamicSize) {
              Value dimSize = b.create<memref::DimOp>(loc, in, d);
              subViewSizes.push_back(dimSize);
              if (inSymbols && mgr) {
                subViewSymbols.push_back(
                    mgr->symbolTable().lookup<SymbolicDimOp>(
                        (*inSymbols)[d].getValue()));
                staticShape = false;
              }
            } else {
              subViewSizes.push_back(getIntAttr(inType.getShape()[d]));
              if (inSymbols && mgr) {
                subViewSymbols.push_back(
                    mgr->newConstantSymbolicDim(inType.getShape()[d]));
              }
            }
          } else {
            subViewSizes.push_back(getIntAttr(tileIt->second));
            if (inSymbols && mgr) {
              subViewSymbols.push_back(
                  mgr->newConstantSymbolicDim(tileIt->second));
            }
          }
        } else {
          assert(false && "unexpected situation\n");
          return false;
        }
      }
      auto view = b.create<memref::SubViewOp>(loc, in, subViewOffsets,
                                              subViewSizes, subViewSteps);
      if (inSymbols && mgr && !staticShape) {
        attachSymbolicDimOpRefArrayAttrOnOperation(view, subViewSymbols);
      }
      valueViewStore[indices] = view.getResult();
    }
  }
  return true;
}

Value StitchCPUAnalysis::emitTileBuffer(OpBuilder& b, Location loc, Value val) {
  auto& tileInfo = tilePlan_[val];
  auto ty = val.getType().cast<MemRefType>();

  bool smallTile = [&]() {
    if (tileInfo.tileSizes.empty()) return true;
    int64_t totalSize = 1;
    for (auto& en : tileInfo.tileSizes) {
      int64_t tileSize = en.second;
      if (tileSize == ShapedType::kDynamicSize) {
        tileSize = ty.getShape()[en.first];
      }
      if (tileSize == ShapedType::kDynamicSize) return false;
      totalSize *= tileSize;
    }
    return totalSize < 64;
  }();

  bool staticShape = true;
  SymbolicDimMgr* mgr = nullptr;
  if (auto shapeIRAnalysis =
          dynamic_cast<ShapeConstraintIRAnalysis*>(&shapeAnalysis_)) {
    mgr = &shapeIRAnalysis->symbolicDimMgr();
  }
  auto valSymbols = getMemRefValueSymbolicDimRefs(val);
  SmallVector<int64_t> newTyShape;
  SmallVector<Value> dynDims;
  SmallVector<SymbolicDimOp> tileSymbols;
  for (int d = 0; d < ty.getRank(); ++d) {
    auto it = tileInfo.tileSizes.find(d);
    if (it == tileInfo.tileSizes.end()) {
      newTyShape.push_back(1);
      if (mgr && valSymbols) {
        tileSymbols.push_back(mgr->newConstantSymbolicDim(1));
      }
      continue;
    }
    if (it->second != ShapedType::kDynamicSize) {
      newTyShape.push_back(it->second);
      if (mgr && valSymbols) {
        tileSymbols.push_back(mgr->newConstantSymbolicDim(it->second));
      }
      continue;
    }
    if (ty.getShape()[d] == ShapedType::kDynamicSize) {
      dynDims.push_back(b.create<memref::DimOp>(loc, val, d));
      if (mgr && valSymbols) {
        tileSymbols.push_back(mgr->symbolTable().lookup<SymbolicDimOp>(
            (*valSymbols)[d].getValue()));
        staticShape = false;
      }
    } else {
      if (mgr && valSymbols) {
        tileSymbols.push_back(mgr->newConstantSymbolicDim(ty.getShape()[d]));
      }
    }
    newTyShape.push_back(ty.getShape()[d]);
  }
  auto newTy =
      MemRefType::get(newTyShape, ty.getElementType(),
                      MemRefLayoutAttrInterface(), ty.getMemorySpace());
  Value tileBuffer;
  if (smallTile) {
    tileBuffer = b.create<memref::AllocaOp>(loc, newTy, dynDims);
  } else {
    static const char* use_alloca = getenv("DISC_USE_ALLOCA_FOR_TILE_BUFFER");
    if (use_alloca && (std::string(use_alloca) == "true")) {
      tileBuffer = b.create<memref::AllocaOp>(loc, newTy, dynDims);
    } else {
      tileBuffer = b.create<memref::AllocOp>(loc, newTy, dynDims);
    }
    if (mgr && valSymbols && !staticShape) {
      attachSymbolicDimOpRefArrayAttrOnOperation(tileBuffer.getDefiningOp(),
                                                 tileSymbols);
    }
  }

  return tileBuffer;
}

bool StitchCPUAnalysis::emitSubRootTile(OpBuilder& b, Location loc, Value val,
                                        ViewStore& viewStore) {
  LLVM_DEBUG(llvm::dbgs() << "emitSubRootTile for: " << val << "\n");
  Value tileBuffer = emitTileBuffer(b, loc, val);
  LLVM_DEBUG(llvm::dbgs() << "tileBuffer: " << tileBuffer << "\n");
  auto& valueViewStore = viewStore[val];
  for (int id : parallelPlan_[val]) {
    auto& info = parallelInfoStore_[id];
    valueViewStore[info.symbolIndices] = tileBuffer;
  }
  return true;
}

bool StitchCPUAnalysis::emitSubRootCalculation(
    OpBuilder& b, Location loc, ParallelInfo& info, ViewStore& viewStore,
    SmallVectorImpl<Operation*>& clonedLmloOps) {
  Value v = info.value;
  Operation* op = fusionPattern_.findLastWriter(v);
  assert(llvm::find(fusionPattern_.getOpList(), op) !=
         fusionPattern_.getOpList().end());

  auto& valueViewStore = viewStore[v];
  if (valueViewStore.find(info.symbolIndices) == valueViewStore.end()) {
    LLVM_DEBUG(llvm::dbgs() << "buffer not found\n");
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "output buffer: "
                          << viewStore[v][info.symbolIndices] << "\n");

  auto isInput = [&](Value v) {
    auto& pattern = fusionPattern_.getOperands();
    return llvm::find(pattern, v) != pattern.end();
  };
  auto isOutput = [&](Value v) {
    auto& pattern = fusionPattern_.getResults();
    return llvm::find(pattern, v) != pattern.end();
  };

  // emit operands if not ready
  SmallVector<Value> newOperands;
  for (Value operand : op->getOperands().drop_back()) {
    auto& prevInfo = parallelInfoStore_[info.producerId];
    ParallelInfo* operandInfo = nullptr;
    if (prevInfo.value == operand) {
      operandInfo = &prevInfo;
    } else {
      for (int consumerId : info.consumerIds) {
        auto& nextInfo = parallelInfoStore_[consumerId];
        if (nextInfo.value == operand) {
          operandInfo = &nextInfo;
          break;
        }
      }
    }
    if (operandInfo == nullptr) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "could not find operand parallel info for op's parallel info #"
          << info.id << " : " << *op << "\noperand: " << operand << "\n");
      return false;
    }
    auto& valueViewStore = viewStore[operand];
    auto it = valueViewStore.find(operandInfo->symbolIndices);
    if (it != valueViewStore.end()) {
      newOperands.push_back(it->second);
      continue;
    }

    // In following case:
    //  1, operand is the input of the fusion pattern
    //  2, operand has no parallel indices
    // We can simply use the original input buffer
    if (isInput(operand) && operandInfo->symbolIndices.empty()) {
      newOperands.push_back(operand);
      continue;
    }

    // Otherwise we allocate a tile buffer for the operand.
    Value tileBuffer = emitTileBuffer(b, loc, operand);
    newOperands.push_back(tileBuffer);
    valueViewStore[operandInfo->symbolIndices] = tileBuffer;
    if (isInput(operand)) {
      if (!emitInputSlice(b, loc, tileBuffer, *operandInfo, clonedLmloOps))
        return false;
    } else {
      if (!emitSubRootCalculation(b, loc, *operandInfo, viewStore,
                                  clonedLmloOps))
        return false;
    }
  }
  // output buffer
  newOperands.push_back(viewStore[v][info.symbolIndices]);

  // clone op with new operands.
  Operation* cloned = b.clone(*op);
  cloned->setOperands(newOperands);
  clonedLmloOps.push_back(cloned);
  return true;
}

bool StitchCPUAnalysis::emitInputSlice(
    OpBuilder& b, Location loc, Value out, ParallelInfo& parallelInfo,
    SmallVectorImpl<Operation*>& clonedLmloOps) {
  Value in = parallelInfo.value;
  auto& tileInfo = tilePlan_[in];
  auto inType = in.getType().cast<MemRefType>();
  int rank = inType.getRank();
  // rank-0 buffer should be handled before.
  assert(rank > 0);

  auto& valueViewStore = inOutViewStore_[in];
  auto it = valueViewStore.find(parallelInfo.symbolIndices);
  if (it == valueViewStore.end()) {
    LLVM_DEBUG(llvm::dbgs() << "no input view found.\n");
    return false;
  }

  auto copyOp = b.create<lmhlo::CopyOp>(loc, it->second, out);
  clonedLmloOps.push_back(copyOp);
  return true;
}

// Emit calculation for roots/sub-roots.
// The basic logic for non-root sub-root value is:
//   %inBound = or(%inBoundOfParallelInfo0, inBoundOfParallelInfo1, ...)
//   if (%inBound) {
//     lmhlo.fusion() {
//       lmhlo.operad_op(...)
//       ...
//       // output sub root op
//       lmhlo.sub_root_op(...)
//     }
//   }
// The basic logic for root sub-root value is:
//   %inBound = or(%inBoundOfParallelInfo0, inBoundOfParallelInfo1, ...)
//   if (%inBound) {
//     %isOwner = if this ivs is the owner for this tile.
//     if (%isOwner) {
//       lmhlo.fusion() {
//         lmhlo.operad_op(...)
//         ...
//         // output sub root op
//         lmhlo.sub_root_op(...)
//         lmhlo.copy(sub_root_tile, output_view)
//       }
//     } else {
//       lmhlo.fusion() {
//         lmhlo.operad_op(...)
//         ...
//         // output sub root op
//         lmhlo.sub_root_op(...)
//       }
//     }
//   }
bool StitchCPUAnalysis::emitAllSubRootsAndRootsCalculation(OpBuilder& b,
                                                           Location loc) {
  int subRootId = 0;
  for (Operation* op : fusionPattern_.getOpList()) {
    Value out = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    if (llvm::find(subRootsAndRootsSet_, out) == subRootsAndRootsSet_.end())
      continue;

    // emit in-bound check logic
    auto& plan = parallelPlan_[out];
    assert(!plan.empty());
    Value inBound = b.create<arith::ConstantIntOp>(loc, 0, 1);
    for (int id : plan) {
      auto& info = parallelInfoStore_[id];
      inBound = b.create<arith::OrIOp>(loc, inBound, info.symbolInBound);
    }
    auto ifOp = b.create<scf::IfOp>(loc, llvm::None, inBound, false);

    // emit sub-root calculation logic
    auto fusionOp = op->getParentOfType<lmhlo::FusionOp>();
    auto subFusionOp = cast<FusionOp>(b.clone(*fusionOp.getOperation()));
    auto fusionName = getFusionName(fusionOp);
    auto subFusionName =
        (llvm::Twine(fusionName) + "_" + llvm::Twine(subRootId)).str();
    ++subRootId;
    setFusionName(b, subFusionOp, subFusionName);
    subFusionOp->setAttr(
        kDiscFusionTypeAttrName,
        b.getStringAttr(fusionTypeToString(FusionType::kLoop)));
    Block* thenBlock = &ifOp.getThenRegion().getBlocks().front();
    subFusionOp->moveBefore(thenBlock, thenBlock->begin());
    OpBuilder innerBuilder(subFusionOp);
    ViewStore localViewStore = subRootViewStore_;
    SmallVector<Operation*> clonedLmhloOps;
    ParallelInfo* info = nullptr;
    for (int id : plan) {
      auto& candidateInfo = parallelInfoStore_[id];
      if (candidateInfo.consumedByRoots) {
        info = &candidateInfo;
        break;
      }
    }
    if (!info) {
      LLVM_DEBUG(llvm::dbgs()
                 << "failed to find a valid parallel info op " << *op << "\n");
      return false;
    }
    if (!emitSubRootCalculation(innerBuilder, loc, *info, localViewStore,
                                clonedLmhloOps)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "failed to do emitSubRootCalculation for " << *op << "\n");
      return false;
    }
    Block& block = subFusionOp.getRegion().front();
    block.clear();
    for (Operation* clonedOp : llvm::reverse(clonedLmhloOps)) {
      clonedOp->moveBefore(&block, block.begin());
    }
    innerBuilder.setInsertionPointAfter(clonedLmhloOps.back());
    innerBuilder.create<lmhlo::TerminatorOp>(loc);
    innerBuilder.setInsertionPoint(subFusionOp);

    // if the sub-root is also a root, emit the root calculation logic
    if (!isFusionResult(out)) continue;
    // emit the is owner check.
    Value isOwner = innerBuilder.create<arith::ConstantIntOp>(loc, 0, 1);
    for (int id : plan) {
      auto& info = parallelInfoStore_[id];
      isOwner =
          innerBuilder.create<arith::OrIOp>(loc, isOwner, info.symbolIsOwner);
    }
    ifOp = innerBuilder.create<scf::IfOp>(loc, llvm::None, isOwner, true);
    Block* elseBlock = &ifOp.getElseRegion().getBlocks().front();
    subFusionOp->moveBefore(elseBlock, elseBlock->begin());

    // emit the root calculation logic.
    auto clonedSubFusionOp =
        cast<lmhlo::FusionOp>(innerBuilder.clone(*subFusionOp.getOperation()));
    addFusionTag(innerBuilder, clonedSubFusionOp, "root_tile");
    auto it = inOutViewStore_[out].find(info->symbolIndices);
    if (it == inOutViewStore_[out].end()) {
      LLVM_DEBUG(llvm::dbgs() << "no output view found.\n");
      return false;
    }
    Operation& tileResultOp =
        *++llvm::reverse(clonedSubFusionOp.getRegion().front()).begin();
    innerBuilder.setInsertionPointAfter(&tileResultOp);
    innerBuilder.create<lmhlo::CopyOp>(loc, tileResultOp.getOperands().back(),
                                       it->second);
    thenBlock = &ifOp.getThenRegion().getBlocks().front();
    clonedSubFusionOp->moveBefore(thenBlock, thenBlock->begin());
  }
  return true;
}

// Do the first level (tile level) codegen for stitch fusion pattern.
// An example is:
//  Input IR
//  ```
//    lmhlo.fusion() {
//      lmhlo.exp(%in, %exp_out) : (memref<?x?xf32>, memref<?x?xf32>, ...)
//      lmhlo.reduce(%exp_out, %init, %red_out) {axis = 1} : (...)
//      lmhlo.bcast(%red_out, ..., %bcast_out) {axis = 0} : (...)
//      lmhlo.div(%exp_out, %bcast_out, %out) : (...)
//    } {fusion_type = "stitch"}
//  ```
//  Ouput IR
//  ```
//    lmhlo.fusion() {
//      // alloc tile buffers
//      %exp_tile_buffer = alloc(...) : memref<1x?xf32>
//      %reduce_tile_buffer = alloc(...) : memref<1xf32>
//      %div_tile_buffer = alloc(...) : memref<1x?xf32>
//      // tile-level parallelism
//      scf.parallel(%ivs, ...) {
//        // bind ins/outs subview for the tile
//        %in_view = memref.subview(%in, %ivs)
//        %out_view = memref.subview(%out, ivs)
//        if (exp_tile_in_bound) { // if in bound for such `ivs`
//          lmhlo.fusion() {
//            lmhlo.exp(%in_view, %exp_tile_buffer)
//              : (memref<1x?xf32>, memref<1x?xf32>, ...)
//          } {fusion_type = "kLoop"}
//        }
//        if (reduce_tile_buffer_in_bound) { // if in bound for such `ivs`
//          lmhlo.fusion() {
//            lmhlo.reduce(%exp_tile_buffer, %init, %reduce_tile_buffer)
//             : (memref<1x?xf32>, memref<f32>, memref<1xf32>)
//          } {fusion_type = "kInput"}
//        }
//        if (div_tile_buffer_in_bound) { // if in bound for such `ivs`
//          // the owner thread (corresponding to the `ivs`) is responsible to
//          // write the corresponding area of the output buffer.
//          // `in_bound` and `is_owner` are same in most cases. However, in
//          // a case where a tile buffer is re-computed among different threads
//          // `in_bound` can be different from `is_owner`.
//          if (is such `ivs` is the owner for this div_tile_buffer) {
//            %bcast_temp_buffer = alloc(...) : memref<1x?xf32>
//            lmhlo.fusion() {
//              lmhlo.bcast(%reduce_tile_buffer, ..., %bcast_temp_buffer)
//               : (memref<1xf32>, ..., memref<1x?xf32>)
//              lmhlo.div(%bcast_temp_buffer, %div_tile_buffer)
//              lmhlo.copy(%div_tile_buffer, %out_view)
//            } {fusion_type = "kLoop"}
//          } else {
//            %bcast_temp_buffer = alloc(...) : memref<1x?xf32>
//            lmhlo.fusion() {
//              lmhlo.bcast(%reduce_tile_buffer, ..., %bcast_temp_buffer)
//               : (memref<1xf32>, ..., memref<1x?xf32>)
//              lmhlo.div(%bcast_temp_buffer, %div_tile_buffer)
//            } {fusion_type = "kLoop"}
//          }
//        }
//      }
//    } {fusion_type = "stitch"}
//  ```
bool StitchCPUAnalysis::doCodeGeneration(OpBuilder& b, lmhlo::FusionOp fusion) {
  LLVM_DEBUG(llvm::dbgs() << "Try to doCodeGeneration for fusion:\n" << fusion);
  if (!isStitchFusion(fusion)) return false;
  if (!fusibilityAnalysis()) return false;

  // 1, create parallel for loop according to dominant value parallel info.
  b.setInsertionPoint(fusion);
  auto cloned = cast<FusionOp>(b.clone(*fusion.getOperation()));
  Block& block = cloned.getRegion().front();
  SmallVector<Operation*> toRemove;
  for (Operation& op : block) {
    if (!isa<lmhlo::TerminatorOp>(&op)) toRemove.push_back(&op);
  }
  for (Operation* op : toRemove) op->erase();
  b.setInsertionPoint(cloned);
  Location loc = fusion->getLoc();
  scf::ParallelOp parallelOp = emitTileParallelLoop(b, loc);
  if (!parallelOp) {
    LLVM_DEBUG(llvm::dbgs() << "failed to do emitTileParallelLoop\n");
    return false;
  }
  parallelOp->moveBefore(&block, block.begin());

  // 2, infer parallel index & in bound check pred & is owner pred for each
  // buffer
  if (!emitParallelIndices(b, loc, parallelOp.getInductionVars())) {
    LLVM_DEBUG(llvm::dbgs() << "failed to do emitParallelIndices\n");
    return false;
  }

  // 3, create memref view for input/output buffers.
  if (!emitInOutTiles(b, loc, inOutViewStore_)) {
    LLVM_DEBUG(llvm::dbgs() << "emitInOutTiles failed\n");
    return false;
  }

  // 4, create sub-buffers for sub roots.
  OpBuilder outterBuilder(parallelOp);
  for (Value subRoot : subRootsAndRootsSet_) {
    if (!emitSubRootTile(b /*outterBuilder*/, loc, subRoot,
                         subRootViewStore_)) {
      LLVM_DEBUG(llvm::dbgs() << "emitSubRootTile failed\n");
      return false;
    }
  }

  // 5, emit roots/sub-roots calculation.
  if (!emitAllSubRootsAndRootsCalculation(b, loc)) {
    LLVM_DEBUG(llvm::dbgs()
               << "failed to do emitAllSubRootsAndRootsCalculation\n");
    return false;
  }

  // 6, do some clean up.
  fusion->erase();
  return true;
}

}  // namespace disc_ral
}  // namespace mlir
