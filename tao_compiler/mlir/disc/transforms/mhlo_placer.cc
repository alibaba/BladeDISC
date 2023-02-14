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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"            // TF:llvm-project
#include "mlir/Pass/Pass.h"                 // TF:local_config_mlir
#include "mlir/Transforms/Passes.h"         // TF:llvm-project
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/placement_utils.h"

namespace mlir {

using placement_utils::kConst;
using placement_utils::kCpu;
using placement_utils::kDiscPlaceAssignment;
using placement_utils::kDiscShapeCalcAttr;
using placement_utils::kGpu;
using placement_utils::kInputPlacementAttr;
using placement_utils::kOutputPlacementAttr;
using placement_utils::PlacementType;

namespace disc_ral {
namespace {

PlacementType toPlacementType(StringRef s, bool default_is_gpu = false) {
  // h -> host (CPU)
  // d -> device (e.g. GPU)
  // x -> xpu (same as the default device. x will be CPU is default is CPU,
  // otherwise is GPU) s -> shape operand, usually placed on CPU as well.
  if (s == "h" || s == "s") return PlacementType::CPU;
  if (s == "d") return PlacementType::GPU;
  assert(s == "x");
  return default_is_gpu ? PlacementType::GPU : PlacementType::CPU;
};

// This pass explicitly place hlo_ops on cpu side by adding an Attr. Nested
// FuncOps should be taken into consideration. 1, Normally, the type of
// kDiscPlaceAssignment is StringAttr; 2, In case the result type of an hlo op
// is TupleType, for example TupleOp
//    or TopKOp, the type of kDiscPlaceAssignment is an ArrayAttr made of
//    StringAttr.
// Following Ops are placed on CPU:
//  - i64 Scalar output
//  - Shape Op's operands
//  - TODO(disc): PrintOp
//  - ConstOp, SelectOp, IotaOp, DynamicIotaOp if type is i32
//  - mhlo.dynamic_gather and mhlo.gather if operand_0's type is i32
//  - Date operands but type is i32 according to kShapeCalcOperandMap
struct OpsPlacer : public PlaceOpsPassBase<OpsPlacer> {
 public:
  using PlaceOpsPassBase<OpsPlacer>::PlaceOpsPassBase;

  OpsPlacer(bool on_gpu) { this->on_gpu_ = on_gpu; }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mhlo_disc::MhloDiscDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    main_func_ = module.lookupSymbol<mlir::func::FuncOp>("main");
    if (!main_func_) {
      module.emitError("entry function not found.");
      signalPassFailure();
      return;
    }

    // initialize argument placement lookup
    if (failed(initInOutsPlacements())) {
      signalPassFailure();
      return;
    }

    // Place custom call op
    if (failed(placeCustomCallV2Ops())) {
      signalPassFailure();
      return;
    }

    // Place I64 scalar output
    placeI64ReturnedCpuScalarOps();

    // Place any shape ops on CPU
    placeShapeOpOnCpu();

    // Place any mhlo Ops that calculates I32 on CPU
    placeI32Ops();

    // Place rest Ops
    addDefaultPlacements();

    // Insert h2d and d2h OP on cross device edges.
    insertMemcpyNodes();
  };

 private:
  // for rule based placement strategy, the placement of the op in the list
  // is up to the placement of the dominant operand
  const DenseMap<TypeID, /*dominant operand index*/ int> kPlaceRuleMap = {
      {TypeID::get<mhlo::DynamicGatherOp>(), /*operand*/ 0},
      {TypeID::get<mhlo::GatherOp>(), /*operand*/ 0},
      {TypeID::get<mlir::mhlo_disc::QuantizedDynamicConvOp>(), /*operand*/ 0},
      {TypeID::get<mlir::mhlo_disc::QuantizedDotGeneralOp>(), /*operand*/ 0}};

  // Place custom call op
  LogicalResult placeCustomCallV2Ops();
  // Place I64 scalar output
  void placeI64ReturnedCpuScalarOps();
  // Place any mhlo ops that calculates I32 on CPU
  void placeShapeOpOnCpu();
  // Place any mhlo Ops that calculates I32 on CPU
  void placeI32Ops();
  // Insert h2d and d2h OP on cross device edges.
  void insertMemcpyNodes();

  // Place I64 scalar output on CPU.
  void markI64ReturnedCpuScalarOps(llvm::DenseSet<Operation*>& marked_ops);

  // Insert Op's operands into set if Op in set
  void markOperands(mlir::func::FuncOp func,
                    llvm::DenseSet<Operation*>& marked_ops);

  // Get placement vector of func's output.
  SmallVector<PlacementType, 4> getOutputPlacements();

  // Get Op's placement according to its attr. If on_gpu is false, always return
  // kCpu.
  PlacementType getOpPlacement(Operation* op);

  // Get the placement of `input` having index `operand_idx`.
  PlacementType getInputTensorPlacement(Operation* dst, size_t operand_idx);

  // Get the placement of `output` having index `result_idx`.
  PlacementType getOutputTensorPlacement(Operation* dst, size_t result_idx);

  // Get input argument's placment. If on_gpu is false, always return kCpu.
  PlacementType getArgumentPlacement(Value arg);

  // Get argument's index of func
  int64_t getArgumentIndex(mlir::func::FuncOp op, Value value);

  // Enforce output's placement.
  void enforceOutputPlacement(
      Operation* dst, func::FuncOp main_func,
      SmallVector<std::pair<Operation*, size_t>, 4>& d2h_worklist,
      SmallVector<std::pair<Operation*, size_t>, 4>& h2d_worklist);

  // Place mhlo Op without placement attr to default device.
  void addDefaultPlacements();

  // insert H2D or D2H Op.
  void insertMemcpy(Operation* dst, size_t operand_index, bool is_h2d);

  // Initializes placement info of inputs and outputs of entry function
  LogicalResult initInOutsPlacements();

  // input placement lookup table
  SmallVector<PlacementType, 4> input_placements_;

  // output placement lookup table
  SmallVector<PlacementType, 4> output_placements_;

  // main func
  mlir::func::FuncOp main_func_;
};

LogicalResult OpsPlacer::placeCustomCallV2Ops() {
  getOperation().walk([&](mhlo_disc::CustomCallV2Op op) {
    OpBuilder b(op);
    auto placement = toPlacementType(op.getDevice(), on_gpu_);
    if (placement == PlacementType::CPU) {
      op->setAttr(kDiscPlaceAssignment, b.getStringAttr(kCpu));
    } else {
      assert(placement == PlacementType::GPU);
      op->setAttr(kDiscPlaceAssignment, b.getStringAttr(kGpu));
    }
  });
  return success();
}

// Mark i64 Scalar output
void OpsPlacer::markI64ReturnedCpuScalarOps(
    llvm::DenseSet<Operation*>& marked_ops) {
  Builder builder(&getContext());

  auto return_op = main_func_.front().getTerminator();
  if (!isa<mlir::func::ReturnOp>(return_op)) return;

  const auto& output_placements = getOutputPlacements();
  auto returned_ops = return_op->getOperands();
  assert(returned_ops.size() == output_placements.size());
  for (auto output : llvm::enumerate(returned_ops)) {
    auto idx = output.index();
    auto op = output.value().getDefiningOp();
    if (!op) continue;

    if (!placement_utils::isMhloOrStdOnTensor(op)) continue;
    // Custom call v2 op has an explicit attribute for output placements.
    // We should respect that configuration.
    if (isa<mhlo_disc::CustomCallV2Op>(op)) continue;

    if (auto type = op->getResult(0).getType().dyn_cast<RankedTensorType>()) {
      if ((output_placements[idx] == PlacementType::CPU) &&
          type.getElementType().isInteger(64) && (type.getRank() == 0)) {
        marked_ops.insert(op);
      }
    }
  }
}

void OpsPlacer::placeI64ReturnedCpuScalarOps() {
  Builder builder(&getContext());
  llvm::DenseSet<Operation*> cpu_placement_ops;

  markI64ReturnedCpuScalarOps(cpu_placement_ops);
  markOperands(main_func_, cpu_placement_ops);
  for (Operation* op : cpu_placement_ops) {
    // We suppose that mhlo op only has single output, either having tensor
    // type or tuple type.
    if (auto tp = op->getResult(0).getType().dyn_cast<TupleType>()) {
      SmallVector<Attribute, 4> attrs(tp.size(), builder.getStringAttr(kCpu));
      op->setAttr(kDiscPlaceAssignment, ArrayAttr::get(tp.getContext(), attrs));
    } else {
      op->setAttr(kDiscPlaceAssignment, builder.getStringAttr(kCpu));
    }
  }
}

// Place any shape ops on CPU
void OpsPlacer::placeShapeOpOnCpu() {
  ModuleOp module = getOperation();
  Builder builder(&getContext());

  module.walk([&](Operation* op) {
    if (!placement_utils::isMhloOrStdOnTensor(op)) {
      return;
    }

    auto op_shape_calc_attr = op->getAttr(kDiscShapeCalcAttr);
    if (!op_shape_calc_attr) return;

    if (auto tp = op->getResult(0).getType().dyn_cast<TupleType>()) {
      SmallVector<Attribute, 4> placement_attrs;
      auto shape_op_attrs =
          op->getAttr(kDiscShapeCalcAttr).dyn_cast_or_null<ArrayAttr>();
      assert(shape_op_attrs && "disc.shape_op not found in TupleOp.");
      for (auto attr : shape_op_attrs) {
        if (attr == builder.getBoolAttr(true)) {
          placement_attrs.push_back(StringAttr::get(&getContext(), kCpu));
        } else {
          placement_attrs.push_back(StringAttr::get(&getContext(), kGpu));
        }
      }
      op->setAttr(kDiscPlaceAssignment,
                  ArrayAttr::get(&getContext(), placement_attrs));
    } else {
      auto bool_attr =
          op->getAttr(kDiscShapeCalcAttr).dyn_cast_or_null<BoolAttr>();
      if (!bool_attr) return;
      op->setAttr(kDiscPlaceAssignment, builder.getStringAttr(kCpu));
    }
    return;
  });
}

int64_t OpsPlacer::getArgumentIndex(mlir::func::FuncOp op, Value value) {
  BlockArgument arg = value.dyn_cast<BlockArgument>();
  if (!arg || arg.getOwner() != &op.front()) return -1;
  return arg.getArgNumber();
}

void OpsPlacer::enforceOutputPlacement(
    Operation* dst, func::FuncOp main_func,
    SmallVector<std::pair<Operation*, size_t>, 4>& d2h_worklist,
    SmallVector<std::pair<Operation*, size_t>, 4>& h2d_worklist) {
  const auto& output_placements = getOutputPlacements();
  assert(output_placements.size() == dst->getNumOperands() &&
         "output_placements size is not equal to num of outputs");
  for (auto i = 0; i < dst->getNumOperands(); ++i) {
    Value operand = dst->getOperand(i);
    auto operand_op = operand.getDefiningOp();
    while (operand_op && isa<tensor::CastOp>(operand_op)) {
      operand = operand_op->getOperand(0);
      operand_op = operand.getDefiningOp();
    }
    auto src_placement =
        operand_op ? getOutputTensorPlacement(
                         operand_op, operand.cast<OpResult>().getResultNumber())
                   : getArgumentPlacement(operand);
    PlacementType dst_placement = output_placements[i];

    if (dst_placement == PlacementType::CPU &&
        src_placement == PlacementType::GPU) {
      d2h_worklist.push_back(std::make_pair(dst, i));
    } else if (dst_placement == PlacementType::GPU &&
               src_placement == PlacementType::CPU) {
      h2d_worklist.push_back(std::make_pair(dst, i));
    }
  }
}

// Insert potential h2d and d2h for cross device edges
void OpsPlacer::insertMemcpyNodes() {
  ModuleOp module = getOperation();
  Builder builder(&getContext());
  SmallVector<std::pair<Operation*, size_t>, 4> d2h_worklist;
  SmallVector<std::pair<Operation*, size_t>, 4> h2d_worklist;

  module.walk([&](Operation* dst) {
    // Enforce output placement specified by the users using attrs.
    if (isa<mlir::func::ReturnOp>(dst)) {
      auto parent = dst->getParentOp();
      if (!isa<mlir::func::FuncOp>(parent) ||
          (cast<mlir::func::FuncOp>(parent).getName() != "main")) {
        return;
      }
      auto main_func = dyn_cast<mlir::func::FuncOp>(parent);
      enforceOutputPlacement(dst, main_func, d2h_worklist, h2d_worklist);
    }
    if (isa<tensor::ExtractOp>(dst)) {
      auto operand = dst->getOperand(0);
      auto parent = operand.getParentRegion()->getParentOp();
      if (!isa<mlir::func::FuncOp>(parent) ||
          (cast<mlir::func::FuncOp>(parent).getName() != "main")) {
        return;
      }
      auto defining_op = operand.getDefiningOp();
      if (defining_op) return;
      if (getArgumentPlacement(operand) == PlacementType::GPU) {
        d2h_worklist.push_back(std::make_pair(dst, 0));
      }
    }
    if (!placement_utils::isMhloOrStdOnTensor(dst) ||
        (isa<mhlo::GetTupleElementOp, mhlo::ReturnOp>(dst)))
      return;

    for (auto indexed_operand : llvm::enumerate(dst->getOperands())) {
      int index = indexed_operand.index();
      Value operand = indexed_operand.value();
      Operation* operand_op = operand.getDefiningOp();
      // If operand is a Block Argument and the parent is not the main func,
      // insert the potential memcpy outside the parent Op.
      if (!operand_op) {
        auto parent = operand.getParentRegion()->getParentOp();
        if (!isa<mlir::func::FuncOp>(parent) ||
            (cast<mlir::func::FuncOp>(parent).getName() != "main")) {
          continue;
        }
      }

      // placement of `tensor::CastOp` is equal to its operand's placement
      while (operand_op && isa<tensor::CastOp>(operand_op)) {
        operand = operand_op->getOperand(0);
        operand_op = operand.getDefiningOp();
      }
      auto dst_placement = getInputTensorPlacement(dst, index);
      auto src_placement =
          operand_op
              ? getOutputTensorPlacement(
                    operand_op, operand.cast<OpResult>().getResultNumber())
              : getArgumentPlacement(operand);
      if (dst_placement == PlacementType::CPU &&
          src_placement == PlacementType::GPU) {
        d2h_worklist.push_back(std::make_pair(dst, index));
      } else if (dst_placement == PlacementType::GPU &&
                 src_placement == PlacementType::CPU) {
        h2d_worklist.push_back(std::make_pair(dst, index));
      }
    }
  });
  for (auto h2d : h2d_worklist) {
    insertMemcpy(h2d.first, h2d.second, 1);
  }
  for (auto d2h : d2h_worklist) {
    insertMemcpy(d2h.first, d2h.second, 0);
  }
}

LogicalResult OpsPlacer::initInOutsPlacements() {
  if (failed(placement_utils::parseEntryFunctionInputPlacements(
          main_func_, on_gpu_, input_placements_))) {
    main_func_.emitError("failed to parse input placement");
    return failure();
  }

  if (failed(placement_utils::parseEntryFunctionOutputPlacements(
          main_func_, on_gpu_, output_placements_))) {
    main_func_.emitError("failed to parse input placement");
    return failure();
  }

  return success();
}

// Place any mhlo Ops that calculates i32 on CPU. This is an rule based
// optimization that mimicking the behavior of tensorflow
void OpsPlacer::placeI32Ops() {
  ModuleOp module = getOperation();
  Builder builder(&getContext());

  module.walk([&](Operation* op) {
    if (!placement_utils::isMhloOrStdOnTensor(op)) return;
    // Custom call v2 op has an explicit attribute for output placements.
    // We should respect that configuration.
    if (isa<mhlo_disc::CustomCallV2Op>(op)) return;

    if (isa<mhlo::TupleOp, mhlo::GetTupleElementOp, mhlo::WhileOp, mhlo::IfOp,
            mhlo::ReturnOp>(op)) {
      return;
    }
    // Skip the Op that is already placed on CPU
    auto attr = op->getAttrOfType<StringAttr>(kDiscPlaceAssignment);
    if ((attr != nullptr) && (attr.getValue() == kCpu)) return;

    // Ops that only cares about the output element type
    if (isa<mhlo::ConstantOp, mhlo::SelectOp, mhlo::IotaOp,
            mhlo::DynamicIotaOp>(op)) {
      auto result_ty = op->getResult(0).getType().dyn_cast<RankedTensorType>();
      assert(result_ty && "unexpected non ranked type for ConstOp");
      auto elem_type = result_ty.getElementType();
      if (elem_type.isInteger(32)) {
        op->setAttr(kDiscPlaceAssignment, builder.getStringAttr(kCpu));
      }
      return;
    }

    auto op_type_id = op->getRegisteredInfo()->getTypeID();
    bool is_shape_calc_op = false;
    // Follow the rule of kPlaceRuleMap exist, or else follow
    // kShapeCalcOperandMap
    auto it = kPlaceRuleMap.find(op_type_id);
    if (it != kPlaceRuleMap.end()) {
      auto dominant_idx = it->second;
      auto operand_ty =
          op->getOperand(dominant_idx).getType().dyn_cast<RankedTensorType>();
      assert(operand_ty && "unexpected non unranked type of operand");
      if (operand_ty.getElementType().isInteger(32)) {
        is_shape_calc_op = true;
      }
    } else {
      const auto& shape_operand_indices =
          placement_utils::getShapeCalcOperandList(op);

      for (int i = 0, e = op->getNumOperands(); i < e; ++i) {
        if (llvm::find(shape_operand_indices, i) != shape_operand_indices.end())
          continue;
        auto operand_ty =
            op->getOperand(i).getType().dyn_cast<RankedTensorType>();
        if (!operand_ty) continue;
        auto elem_type = operand_ty.getElementType();
        if (elem_type.isInteger(32)) {
          is_shape_calc_op = true;
          break;
        }
      }
    }

    // For these ops, currently we only have limited support.
    if (auto custom_call_op = dyn_cast<mhlo_disc::CustomCallOp>(op)) {
      if (custom_call_op.getCallTargetName() == "topk" ||
          custom_call_op.getCallTargetName() == "rng_uniform") {
        is_shape_calc_op = false;
      }
      // TopK op only have gpu kernel implementation a.t.m. Thus we put its
      // operands to gpu as well to eliminate potential cross-device memory
      // copy.
      if (this->on_gpu_ && custom_call_op.getCallTargetName() == "topk") {
        Value indices = op->getOperand(1);
        Operation* indicesOp = indices.getDefiningOp();
        if (indicesOp) {
          if (isa<mhlo::IotaOp, mhlo::DynamicIotaOp>(indicesOp)) {
            indicesOp->setAttr(kDiscPlaceAssignment,
                               builder.getStringAttr(kGpu));
          } else if (isa<mhlo::BroadcastInDimOp, mhlo::DynamicBroadcastInDimOp>(
                         indicesOp)) {
            Operation* iota = indicesOp->getOperand(0).getDefiningOp();
            if (iota && isa<mhlo::IotaOp, mhlo::DynamicIotaOp>(iota)) {
              iota->setAttr(kDiscPlaceAssignment, builder.getStringAttr(kGpu));
              indicesOp->setAttr(kDiscPlaceAssignment,
                                 builder.getStringAttr(kGpu));
            }
          }
        }
      }
    }

    // Set attr if it is a shape Op
    if (is_shape_calc_op) {
      if (auto tp = op->getResult(0).getType().dyn_cast<TupleType>()) {
        SmallVector<Attribute, 4> attrs(tp.size(), builder.getStringAttr(kCpu));
        op->setAttr(kDiscPlaceAssignment,
                    ArrayAttr::get(tp.getContext(), attrs));
      } else {
        op->setAttr(kDiscPlaceAssignment, builder.getStringAttr(kCpu));
      }
    }
    return;
  });
}

// Place mhlo Op without placement attr to default device.
void OpsPlacer::addDefaultPlacements() {
  ModuleOp module = getOperation();
  Builder builder(&getContext());
  auto default_placement = this->on_gpu_ ? kGpu : kCpu;

  module.walk([&](Operation* op) {
    if (!placement_utils::isMhloDialect(op)) return;
    if (op->getNumResults() == 0) return;
    // Skip the Op that is already placed
    auto attr = op->getAttr(kDiscPlaceAssignment);
    if (attr != nullptr) {
      return;
    }
    if (auto tp = op->getResult(0).getType().dyn_cast<TupleType>()) {
      SmallVector<Attribute, 4> attrs(tp.size(),
                                      builder.getStringAttr(default_placement));
      op->setAttr(kDiscPlaceAssignment, ArrayAttr::get(tp.getContext(), attrs));
    } else {
      op->setAttr(kDiscPlaceAssignment,
                  builder.getStringAttr(default_placement));
    }
    return;
  });
}

void OpsPlacer::markOperands(mlir::func::FuncOp func,
                             llvm::DenseSet<Operation*>& marked_ops) {
  auto& block = func.getBlocks().front();
  for (auto& op : llvm::make_early_inc_range(
           llvm::make_range(block.rbegin(), block.rend()))) {
    // TODO(disc): If the operand of the op is a nested FuncOp, mark the
    // associated producer in the nested FuncOp
    if (!placement_utils::isMhloOrStdOnTensor(&op)) continue;
    // If the op is already in set, insert all of its operands into set
    if (marked_ops.contains(&op)) {
      for (auto operand_value : op.getOperands()) {
        Operation* operand = operand_value.getDefiningOp();
        if (operand == nullptr) continue;
        if (!placement_utils::isMhloOrStdOnTensor(operand)) {
          continue;
        }
        marked_ops.insert(operand);
      }
    }
  };
}

PlacementType OpsPlacer::getArgumentPlacement(Value arg) {
  auto parent = arg.getParentRegion()->getParentOp();
  assert(isa<mlir::func::FuncOp>(parent) &&
         "invalid use of getArgumentPlacement");
  auto main_func = cast<mlir::func::FuncOp>(parent);
  assert(main_func.getName() == "main" &&
         "invalid use of getArgumentPlacement");
  auto arg_index = getArgumentIndex(main_func, arg);
  return input_placements_[arg_index];
}

SmallVector<PlacementType, 4> OpsPlacer::getOutputPlacements() {
  return output_placements_;
}

PlacementType OpsPlacer::getOpPlacement(Operation* op) {
  // TODO(disc): consider mhlo_disc dialect as well
  if (!placement_utils::isMhloDialect(op)) return PlacementType::CPU;
  if (!this->on_gpu_) return PlacementType::CPU;
  if (auto attr = op->getAttrOfType<BoolAttr>(kDiscShapeCalcAttr)) {
    return PlacementType::CPU;
  }
  auto attr = op->getAttrOfType<StringAttr>(kDiscPlaceAssignment);
  if ((attr != nullptr) && (attr.getValue() == kCpu)) {
    return PlacementType::CPU;
  }
  return PlacementType::GPU;
}

PlacementType OpsPlacer::getInputTensorPlacement(Operation* dst,
                                                 size_t operand_idx) {
  if (!this->on_gpu_) return PlacementType::CPU;
  // special case when dst is TupleOp
  if (isa<mhlo::TupleOp>(dst)) {
    auto array_attr = dst->getAttrOfType<ArrayAttr>(kDiscPlaceAssignment);
    assert(array_attr && "kDiscPlaceAssignment on Tuple not found");
    if (array_attr[operand_idx].cast<StringAttr>().getValue() == kCpu) {
      return PlacementType::CPU;
    } else {
      return PlacementType::GPU;
    }
  } else if (auto custom_op = dyn_cast<mhlo_disc::CustomCallV2Op>(dst)) {
    SmallVector<StringRef, 4> input_items;
    custom_op.getInputPlacements().split(input_items, ',', /*MaxSplit=*/-1,
                                         /*KeepEmpty=*/false);
    if (operand_idx < input_items.size()) {
      return toPlacementType(input_items[operand_idx], on_gpu_);
    }
  }

  // when dst op placed on CPU
  if (getOpPlacement(dst) == PlacementType::CPU) return PlacementType::CPU;

  // when dst op placed on GPU
  const auto& shape_operand_indices =
      placement_utils::getShapeCalcOperandList(dst);

  if (std::find(shape_operand_indices.begin(), shape_operand_indices.end(),
                operand_idx) != shape_operand_indices.end())
    return PlacementType::CPU;

  return PlacementType::GPU;
}

PlacementType OpsPlacer::getOutputTensorPlacement(Operation* dst,
                                                  size_t result_idx) {
  if (!this->on_gpu_) return PlacementType::CPU;
  // special case when dst is TupleOp
  if (isa<mhlo::TupleOp>(dst)) {
    auto array_attr = dst->getAttrOfType<ArrayAttr>(kDiscPlaceAssignment);
    assert(array_attr && "kDiscPlaceAssignment on Tuple not found");
    if (array_attr[result_idx].cast<StringAttr>().getValue() == kCpu) {
      return PlacementType::CPU;
    } else {
      return PlacementType::GPU;
    }
  } else if (auto custom_op = dyn_cast<mhlo_disc::CustomCallV2Op>(dst)) {
    SmallVector<StringRef, 4> output_items;
    custom_op.getOutputPlacements().split(output_items, ',', /*MaxSplit=*/-1,
                                          /*KeepEmpty=*/false);
    if (result_idx < output_items.size()) {
      return toPlacementType(output_items[result_idx], on_gpu_);
    }
  }

  // default case: output should have the same placement type as the op.
  return getOpPlacement(dst);
}

void OpsPlacer::insertMemcpy(Operation* dst, size_t operand_index,
                             bool is_h2d) {
  OpBuilder b(dst);
  Location loc = dst->getLoc();
  auto orig_operand = dst->getOperand(operand_index);
  Value copy_result = nullptr;
  if (is_h2d) {
    copy_result = b.create<mhlo_disc::H2DOp>(loc, orig_operand).getResult();
  } else {
    auto new_copy = b.create<mhlo_disc::D2HOp>(loc, orig_operand);
    new_copy->setAttr(kDiscPlaceAssignment, b.getStringAttr(kCpu));
    copy_result = new_copy.getResult();
  }
  dst->setOperand(operand_index, copy_result);
}
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createPlacerPass(bool on_gpu) {
  return std::make_unique<OpsPlacer>(on_gpu);
}

}  // namespace disc_ral
}  // namespace mlir
