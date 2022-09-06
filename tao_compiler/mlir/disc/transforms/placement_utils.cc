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

#include "placement_utils.h"

#include "llvm/ADT/StringMap.h"
#include "mlir-hlo/Dialect/lhlo/transforms/map_hlo_to_lhlo_op.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/transforms/disc_map_hlo_to_lhlo_op.h"

namespace mlir {
namespace placement_utils {

bool isGpuMhlo(Operation* op) {
  if (!op) return false;
  auto attr = op->getAttrOfType<StringAttr>(kDiscPlaceAssignment);

  // TODO(disc): OP without placement attribute are supposed to run on gpu ATM.
  // Change the default to CPU once we merge the cpu branch code.
  return ((!attr) || ((attr != nullptr) && (attr.getValue() == kGpu)));
}

// Return true if the memref is on GPU
bool isGpuMemRef(Value memref) {
  assert(memref.getType().isa<MemRefType>());
  auto memory_space = memref.getType().cast<MemRefType>().getMemorySpace();
  return memory_space && memory_space.isa<StringAttr>() &&
         memory_space.cast<StringAttr>().getValue() ==
             mlir::placement_utils::kGpu;
}

using ShapeOperandListMap = DenseMap<TypeID, ShapeOperandList>;
using CustomCallShapeOperandListMap = llvm::StringMap<ShapeOperandList>;

template <typename HloTy>
void appendShapeOperandListForHloOp(ShapeOperandListMap& m,
                                    ShapeOperandList list) {
  m[TypeID::get<HloTy>()] = list;
  using LhloTy = typename mhlo::HloToLhloOp<HloTy>;
  m[TypeID::get<LhloTy>()] = list;
}

template <typename HloDiscTy>
void appendShapeOperandListForHloDiscOp(ShapeOperandListMap& m,
                                        ShapeOperandList list) {
  m[TypeID::get<HloDiscTy>()] = list;
  using LhloDiscTy = typename mhlo_disc::HloToLhloOp<HloDiscTy>;
  m[TypeID::get<LhloDiscTy>()] = list;
}

ShapeOperandListMap initShapeCalcOperandMap() {
  ShapeOperandListMap m;

  // mhlo dialect ops.
  appendShapeOperandListForHloOp<mhlo::RealDynamicSliceOp>(
      m, {/*start_indices*/ 1, /*limit_indices*/ 2, /*strides*/ 3});
  appendShapeOperandListForHloOp<mhlo::DynamicPadOp>(
      m, {/*edge_padding_low*/ 2, /*edge_padding_high*/ 3,
          /*interior_padding*/ 4});
  appendShapeOperandListForHloOp<mhlo::DynamicReshapeOp>(m, {/*shape*/ 1});
  appendShapeOperandListForHloOp<mhlo::DynamicIotaOp>(m, {/*shape*/ 0});
  appendShapeOperandListForHloOp<mhlo::DynamicBroadcastInDimOp>(
      m, {/*out_shape*/ 1});
  appendShapeOperandListForHloOp<mhlo::DynamicGatherOp>(m, {/*slice_sizes*/ 2});
  appendShapeOperandListForHloOp<mhlo::DynamicConvOp>(m, {/*paddings*/ 2});
  appendShapeOperandListForHloOp<mhlo::IfOp>(m, {/*pred*/ 0});

  // mhlo_disc dialect ops.
  appendShapeOperandListForHloDiscOp<mhlo_disc::QuantizedDynamicConvOp>(
      m, {/*paddings*/ 2});

  return m;
}

const ShapeOperandListMap kShapeCalcOperandMap = initShapeCalcOperandMap();

CustomCallShapeOperandListMap initCustomCallShapeCalcOperandMap() {
  CustomCallShapeOperandListMap m;
  m["topk"] = {/*k*/ 2};
  m["rng_uniform"] = {/*start*/ 0, /*limit*/ 1, /*shape*/ 2};
  return m;
}

const CustomCallShapeOperandListMap kCustomCallShapeOperandListMap =
    initCustomCallShapeCalcOperandMap();

ShapeOperandList getShapeCalcOperandList(TypeID op_type_id) {
  auto iter = kShapeCalcOperandMap.find(op_type_id);
  if (iter != kShapeCalcOperandMap.end()) {
    return iter->second;
  }
  return {};
}

ShapeOperandList getShapeCalcOperandList(Operation* op) {
  if (isa<mhlo_disc::CustomCallOp>(op) || isa<lmhlo_disc::CustomCallOp>(op)) {
    std::string target_name;
    if (isa<mhlo_disc::CustomCallOp>(op)) {
      auto custom_call = cast<mhlo_disc::CustomCallOp>(op);
      target_name = custom_call.call_target_name().str();
    } else {
      auto custom_call = cast<lmhlo_disc::CustomCallOp>(op);
      target_name = custom_call.call_target_name().str();
    }
    auto iter = kCustomCallShapeOperandListMap.find(target_name);
    if (iter != kCustomCallShapeOperandListMap.end()) {
      return iter->second;
    }
    return {};
  }
  return getShapeCalcOperandList(op->getRegisteredInfo()->getTypeID());
}

LogicalResult parsePlacementAttribute(func::FuncOp main, bool default_on_gpu,
                                      StringRef attrName, int numAttribute,
                                      SmallVectorImpl<StringRef>& out) {
  auto dict_attr = main->getAttrOfType<DictionaryAttr>("tf.entry_function");
  if (!dict_attr) {
    main.emitError("entry function must have tf.entry_function attr.");
    return failure();
  }

  StringRef default_placement = default_on_gpu ? kGpu : kCpu;
  auto placements_attr = dict_attr.get(attrName);
  if (!placements_attr) {
    for (int i = 0; i < numAttribute; ++i) {
      out.push_back(default_placement);
    }
    return success();
  }

  auto str_attr = placements_attr.dyn_cast<mlir::StringAttr>();
  if (!str_attr) {
    main.emitError("placement attribute should be string type");
    return failure();
  }

  SmallVector<StringRef, 4> parsed_placements;
  str_attr.getValue().split(parsed_placements, ',', /*MaxSplit=*/-1,
                            /*KeepEmpty=*/false);
  if (parsed_placements.size() != numAttribute) {
    main.emitError("placement size mismatch");
    return failure();
  }

  for (StringRef str : parsed_placements) {
    // Not using the parsed strings since they are destroyed after return.
    if (str == kGpu) {
      out.push_back(kGpu);
    } else if (str == kCpu) {
      out.push_back(kCpu);
    } else if (str == kConst) {
      out.push_back(kConst);
    } else {
      main.emitError("unknown placement type name");
      return failure();
    }
  }
  return success();
}

LogicalResult parseEntryFunctionInputPlacements(
    func::FuncOp main, bool default_on_gpu, SmallVectorImpl<StringRef>& out) {
  return parsePlacementAttribute(main, default_on_gpu, kInputPlacementAttr,
                                 main.getNumArguments(), out);
}

LogicalResult parseEntryFunctionOutputPlacements(
    func::FuncOp main, bool default_on_gpu, SmallVectorImpl<StringRef>& out) {
  return parsePlacementAttribute(main, default_on_gpu, kOutputPlacementAttr,
                                 main.getNumResults(), out);
}

PlacementType PlacementFromString(StringRef s) {
  if (s == kGpu) {
    return PlacementType::GPU;
  } else if (s == kCpu) {
    return PlacementType::CPU;
  }
  assert(s == kConst);
  return PlacementType::Const;
}

StringRef PlacementToString(PlacementType type) {
  if (type == PlacementType::CPU) {
    return kCpu;
  } else if (type == PlacementType::GPU) {
    return kGpu;
  }
  assert(type == PlacementType::Const);
  return kConst;
}

LogicalResult parseEntryFunctionInputPlacements(
    func::FuncOp main, bool default_on_gpu,
    SmallVectorImpl<PlacementType>& out) {
  SmallVector<StringRef, 4> str_out;
  if (failed(parseEntryFunctionInputPlacements(main, default_on_gpu, str_out)))
    return failure();
  llvm::transform(str_out, std::back_inserter(out), PlacementFromString);
  return success();
}

LogicalResult parseEntryFunctionOutputPlacements(
    func::FuncOp main, bool default_on_gpu,
    SmallVectorImpl<PlacementType>& out) {
  SmallVector<StringRef, 4> str_out;
  if (failed(parseEntryFunctionOutputPlacements(main, default_on_gpu, str_out)))
    return failure();
  llvm::transform(str_out, std::back_inserter(out), PlacementFromString);
  return success();
}

}  // namespace placement_utils
}  // namespace mlir
