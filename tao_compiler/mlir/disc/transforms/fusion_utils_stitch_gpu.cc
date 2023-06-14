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

#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/fusion_utils.h"

namespace mlir {
namespace disc_ral {

////////////////////// Stitch GPU FusionStrategy Implemenation /////////
////////////////////////////////////////////////////////////////////////

bool findValidReductionOps(FusionPatternBase& target,
                           SmallVectorImpl<Operation*>& row_reductions,
                           SmallVectorImpl<Operation*>& col_reductions) {
  row_reductions.clear();
  col_reductions.clear();
  auto& op_list = target.getOpList();
  for (Operation* op : op_list) {
    if (!isa<lmhlo::ReduceOp>(op)) continue;
    if (isRank2RowReduction(op)) {
      row_reductions.push_back(op);
    } else if (isRank2ColReduction(op)) {
      // Middle col-reduction is not supported currently. We may support it with
      // AStitch technique in the future.
      int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
      for (Value v : op->getOperands().drop_front(num_input_operand)) {
        for (Operation* user : getValueUsers(v)) {
          if (user == op) continue;
          if (std::find(op_list.begin(), op_list.end(), user) !=
              op_list.end()) {
            return false;
          }
        }
      }
      col_reductions.push_back(op);
    } else {
      // Non supported reduction type.
      return false;
    }
  }
  return true;
}

bool StitchGpuFusionStrategy::isFusible(Operation* op) {
  if (isa<lmhlo::TransposeOp>(op) && isRank2or3Transpose(op)) return false;
  return true;
}

bool StitchGpuFusionStrategy::tryFuse(ShapeAnalysis& shapeAnalysis,
                                      FusionPattern& lhs, FusionPattern& rhs,
                                      FusionPattern& target) {
  // TODO(Yancey): support fusion with different reduction type
  bool has_row_reduction = llvm::any_of(target.getOpList(), [](Operation* op) {
    return isRank2RowReduction(op);
  });
  bool has_col_reduciton = llvm::any_of(target.getOpList(), [](Operation* op) {
    return isRank2ColReduction(op);
  });

  if (has_row_reduction && has_col_reduciton) {
    return false;
  }
  return FusionStrategy::tryFuse(shapeAnalysis, lhs, rhs, target);
}

Value StitchGpuFusionStrategy::getEffectiveShape(FusionPattern& target,
                                                 Value v) {
  Operation* result_op = target.findLastWriter(v);
  assert(result_op);
  // effective shape of reduce op is its operand's shape.
  return isa<lmhlo::ReduceOp>(result_op) ? result_op->getOperand(0) : v;
}

bool StitchGpuFusionStrategy::tileCoverInfoPropagateO2I(
    ShapeAnalysis& shapeAnalysis, DenseMap<Value, TileInfo>& tile_plan,
    Operation* op, SmallVector<std::pair<Value, TileInfo>, 4>& in_info,
    bool& cover) {
  if (isa<lmhlo::ConstantOp>(op)) {
    return true;
  }
  if (isElementWise(op) || isa<lmhlo::ConcatenateOp>(op)) {
    Value out_value = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    auto tile_info = tile_plan[out_value];
    for (Value lhs : op->getOperands().drop_back()) {
      in_info.emplace_back(lhs, tile_info);
    }
  } else if (isa<lmhlo::RealDynamicSliceOp, lmhlo::SliceOp>(op)) {
    Value out_value = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    auto tile_info = tile_plan[out_value];
    Value in_value = op->getOperand(0);
    in_info.emplace_back(in_value, tile_info);
    // The output indices cannot cover all input indices.
    cover = false;
  } else if (isa<lmhlo::BroadcastInDimOp, lmhlo::DynamicBroadcastInDimOp>(op)) {
    Value in_value = op->getOperand(0);
    Value out_value = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    auto& out_tile = tile_plan[out_value];
    TileInfo in_tile;
    auto dimAttr = op->getAttrOfType<DenseElementsAttr>("broadcast_dimensions");
    assert(dimAttr);
    auto dimensions = dimAttr.getValues<int64_t>();
    DenseSet<int64_t> dim_set(dimensions.begin(), dimensions.end());
    DenseMap<int64_t, int64_t> broadcast_dim_o2i;
    for (auto en : llvm::enumerate(dimensions)) {
      broadcast_dim_o2i[en.value()] = en.index();
    }
    int64_t rank = out_value.getType().cast<MemRefType>().getRank();
    for (int64_t i = 0; i < rank; i++) {
      auto may_indim = broadcast_dim_o2i.find(i);
      if (may_indim != broadcast_dim_o2i.end()) {
        auto may_intile = out_tile.tileSizes.find(i);
        if (may_intile != out_tile.tileSizes.end()) {
          in_tile.tileSizes[may_indim->second] = may_intile->second;
        }
      }
    }
    in_info.emplace_back(in_value, in_tile);
  } else if (isa<lmhlo::BroadcastOp>(op)) {
    Value in_value = op->getOperand(0);
    Value out_value = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    auto& out_tile = tile_plan[out_value];
    TileInfo in_tile;
    auto sizesAttr = op->getAttrOfType<DenseElementsAttr>("broadcast_sizes");
    assert(sizesAttr);
    auto sizes = sizesAttr.getValues<int64_t>();
    int64_t length = sizes.end() - sizes.begin();
    for (auto out_tile_pair : out_tile.tileSizes) {
      in_tile.tileSizes[out_tile_pair.first - length] = out_tile_pair.second;
    }
    in_info.emplace_back(in_value, in_tile);
  } else if (isa<lmhlo::DynamicReshapeOp, lmhlo::ReshapeOp>(op)) {
    SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> dim_eq;
    Value in_value = op->getOperand(0);
    Value out_value = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    int64_t in_rank = in_value.getType().cast<MemRefType>().getRank();
    int64_t out_rank = out_value.getType().cast<MemRefType>().getRank();

    if (auto analysis_deprecated =
            dynamic_cast<ShapeAnalysisDeprecated*>(&shapeAnalysis)) {
      auto& shapeAnalysis = *analysis_deprecated;
      if (!shapeAnalysis.extractContinuousDimEqualInfo(in_value, out_value,
                                                       dim_eq)) {
        return false;
      }

      auto& out_tile = tile_plan[out_value];
      TileInfo in_tile;

      int64_t out_tiles_left = out_tile.tileSizes.size();
      int64_t out_non_tiles_left = out_rank - out_tiles_left;
      DenseSet<int64_t> in_ranks_mapped;
      for (auto equal : dim_eq) {
        SmallVector<int64_t> ins = equal.first;
        SmallVector<int64_t> outs = equal.second;
        in_ranks_mapped.insert(ins.begin(), ins.end());
        bool is_tiled = (out_tile.tileSizes.count(outs[0]) != 0);
        // Make sure tile-info of all out dims are consistent.
        for (int64_t i = 1; i < outs.size(); i++) {
          if ((out_tile.tileSizes.count(outs[i]) != 0) != is_tiled) {
            LLVM_DEBUG(llvm::dbgs() << "tile backprop failed: " << *op);
            return false;
          }
        }
        if (is_tiled) {
          for (auto in : ins) {
            in_tile.tileSizes[in] = ShapedType::kDynamic;
          }
          out_tiles_left -= outs.size();
        } else {
          out_non_tiles_left -= outs.size();
        }
      }

      if (out_non_tiles_left == 0) {
        if (out_tiles_left != 0) {
          // Sometimes, we failed to map all equal relationships. Since we know
          // all non-tiled dims in equal-map are processed when reach here, all
          // dims not in equal-map are tiled.
          for (int64_t i = 0; i < in_rank; i++) {
            if (!in_ranks_mapped.contains(i)) {
              in_tile.tileSizes[i] = ShapedType::kDynamic;
            }
          }
        }
      } else if (out_tiles_left != 0) {
        // This means some dims not in o2i-map are tiled. But we cannot figure
        // out who are they...
        LLVM_DEBUG(llvm::dbgs() << "tile backprop failed: " << *op);
        return false;
      }
      in_info.emplace_back(in_value, in_tile);
    } else {
      bool last_is_tiled = false;
      auto& out_tile = tile_plan[out_value];
      // Partition output dims according to tile information to make sure
      // each group is continuous and is either tiled or non-tiled.
      SmallVector<int> out_group_end_dims;
      auto outShape = out_value.getType().cast<MemRefType>().getShape();
      for (int i = 0; i < out_rank; ++i) {
        if (out_tile.tileSizes.count(i) == 0) {
          if (!last_is_tiled) continue;
          // ignore size-1 dim between tiled dims.
          if (outShape[i] == 1) continue;
          last_is_tiled = false;
          out_group_end_dims.push_back(i);
        } else {
          if (last_is_tiled) continue;
          last_is_tiled = true;
          if (i != 0) out_group_end_dims.push_back(i);
        }
      }

      TileInfo in_tile;
      int last_map_input_dim = 0;
      int last_map_output_dim = 0;
      for (int out_group_end_dim : out_group_end_dims) {
        bool matched = false;
        for (int i = last_map_input_dim; i <= in_rank; ++i) {
          if (shapeAnalysis.isProductEqual(out_value, last_map_output_dim,
                                           out_group_end_dim, in_value,
                                           last_map_input_dim, i)) {
            last_map_input_dim = i;
            last_map_output_dim = out_group_end_dim;
            if (out_tile.tileSizes.count(last_map_output_dim) != 0) {
              for (int inIdx = last_map_input_dim; inIdx < i; ++inIdx) {
                in_tile.tileSizes[inIdx] = ShapedType::kDynamic;
              }
            }
            matched = true;
            break;
          }
        }
        if (!matched) return false;
      }

      if (last_is_tiled) {
        for (int inIdx = last_map_input_dim; inIdx < in_rank; ++inIdx) {
          in_tile.tileSizes[inIdx] = ShapedType::kDynamic;
        }
      }
      in_info.emplace_back(in_value, in_tile);
    }
  } else if (isa<lmhlo::TransposeOp>(op)) {
    auto transpose = cast<lmhlo::TransposeOp>(op);
    auto permutation = transpose.getPermutation().getValues<int64_t>();
    Value out_value = transpose.getOutput();
    auto tile_info = tile_plan[out_value];
    int64_t rank = out_value.getType().cast<MemRefType>().getRank();
    TileInfo in_tile;
    for (int64_t i = 0; i < rank; i++) {
      if (tile_info.tileSizes.find(i) != tile_info.tileSizes.end()) {
        in_tile.tileSizes[permutation[i]] = tile_info.tileSizes[i];
      }
    }
    Value in_value = transpose.getOperand();
    in_info.emplace_back(in_value, in_tile);
  } else if (isa<lmhlo::DynamicGatherOp, lmhlo::GatherOp>(op)) {
    Value in_value = op->getOperand(0);
    Value out_value = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    int64_t in_rank = in_value.getType().cast<MemRefType>().getRank();
    int64_t out_rank = out_value.getType().cast<MemRefType>().getRank();
    auto gather = dyn_cast<lmhlo::GatherOp>(op);
    auto d_gather = dyn_cast<lmhlo::DynamicGatherOp>(op);
    auto dimension_numbers =
        gather ? gather.getDimensionNumbers() : d_gather.getDimensionNumbers();

    SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> dim_eq;
    // Non-collapsed dims.
    auto offset_dims = dimension_numbers.getOffsetDims();
    for (auto offset : llvm::enumerate(offset_dims)) {
      dim_eq.emplace_back(SmallVector<int64_t>({offset.index()}),
                          SmallVector<int64_t>({offset.value()}));
    }
    // Collapsed dims.
    auto collapsed_slice_dims = dimension_numbers.getCollapsedSliceDims();
    SmallVector<int64_t> batch_dims;
    DenseSet<int64_t> offset_dim_set(offset_dims.begin(), offset_dims.end());
    for (int64_t i = 0; i < out_rank; ++i) {
      if (!offset_dim_set.contains(i)) {
        batch_dims.push_back(i);
      }
    }
    SmallVector<int64_t> collapsed_slice_dims_vec(collapsed_slice_dims.begin(),
                                                  collapsed_slice_dims.end());
    dim_eq.emplace_back(collapsed_slice_dims_vec, batch_dims);

    // TODO: the logic is the same with reshape. Rewrite as a function.
    auto& out_tile = tile_plan[out_value];
    TileInfo in_tile;
    int64_t out_tiles_left = out_tile.tileSizes.size();
    int64_t out_non_tiles_left = out_rank - out_tiles_left;
    DenseSet<int64_t> in_ranks_mapped;
    for (auto equal : dim_eq) {
      SmallVector<int64_t> ins = equal.first;
      SmallVector<int64_t> outs = equal.second;
      in_ranks_mapped.insert(ins.begin(), ins.end());
      bool is_tiled = (out_tile.tileSizes.count(outs[0]) != 0);
      // Make sure tile-info of all out dims are consistent.
      for (int64_t i = 1; i < outs.size(); i++) {
        if ((out_tile.tileSizes.count(outs[i]) != 0) != is_tiled) {
          LLVM_DEBUG(llvm::dbgs() << "tile backprop failed: " << *op);
          return false;
        }
      }
      if (is_tiled) {
        for (auto in : ins) {
          in_tile.tileSizes[in] = ShapedType::kDynamic;
        }
        out_tiles_left -= outs.size();
      } else {
        out_non_tiles_left -= outs.size();
      }
    }

    // TODO: gather op do not need the following logic?
    if (out_non_tiles_left == 0) {
      if (out_tiles_left != 0) {
        // Sometimes, we failed to map all equal relationships. Since we know
        // all non-tiled dims in equal-map are processed when reach here, all
        // dims not in equal-map are tiled.
        for (int64_t i = 0; i < in_rank; i++) {
          if (!in_ranks_mapped.contains(i)) {
            in_tile.tileSizes[i] = ShapedType::kDynamic;
          }
        }
      }
    } else if (out_tiles_left != 0) {
      // This means some dims not in o2i-map are tiled. But we cannot figure
      // out who are they...
      LLVM_DEBUG(llvm::dbgs() << "tile backprop failed: " << *op);
      return false;
    }
    in_info.emplace_back(in_value, in_tile);
    cover = false;
  } else {
    // TODO: if there is no root op as direct or indirect producer of this op,
    // do not do any propagation.
    LLVM_DEBUG(llvm::dbgs() << "unsupported op in backprop: " << *op);
    return false;
  }
  return true;
}

bool StitchGpuFusionStrategy::findFusionPatternTypeAndSubroot(
    ShapeAnalysis& shapeAnalysis, FusionPattern& fusion_pattern) {
  FusionType fusion_type = FusionType::kNone;
  Operation* dominant_op = nullptr;
  const auto& results = fusion_pattern.getResults();
  assert(!results.empty());

  SmallVector<Operation*, 4> row_reductions;
  SmallVector<Operation*, 4> col_reductions;
  if (!findValidReductionOps(fusion_pattern, row_reductions, col_reductions)) {
    LLVM_DEBUG(llvm::dbgs() << "Check reduction ops failed.");
    return false;
  }

  // Check whether there is 'middle' row reduction. If there is, it is kStitch.
  const auto& op_list = fusion_pattern.getOpList();
  DenseSet<Operation*> op_set(op_list.begin(), op_list.end());
  for (Operation* op : row_reductions) {
    if (fusion_type != FusionType::kNone) {
      break;
    }
    int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
    for (Value v : op->getOperands().drop_front(num_input_operand)) {
      if (fusion_type != FusionType::kNone) {
        break;
      }
      for (Operation* user : getValueUsers(v)) {
        if (user == op) continue;
        if (op_set.find(user) != op_set.end()) {
          fusion_type = FusionType::kStitch;
          dominant_op = row_reductions.back();
          break;
        }
      }
    }
  }

  // Check basic fusion types.
  // Check compatibility for kRowReduction, kColReduction and kLoop:
  // - for kRowReduction, all row-reduces should have the same shape with
  // dominant, and other results should have the same number of elements.
  // Otherwise we set it as kStitch fusion.
  // - for kColReduction, all col-reduces should have the same shape with
  // dominant, and other results should have the same number of elements.
  // - for kLoop, all results should have the same number of elements.
  // Note that kRowReduce that do not meet compatibility constraint are already
  // regarded as kStitch already.
  if (fusion_type != FusionType::kStitch) {
    if (!row_reductions.empty()) {
      fusion_type = FusionType::kRowReduction;
      dominant_op = row_reductions.back();
      // We do not check for patterns with no row-reduce op for kStitch as we
      // cannot determine the tile information.
      Value ref = cast<lmhlo::LmhloOp>(dominant_op).getResultBuffer();
      Value ref_shape = getEffectiveShape(fusion_pattern, ref);
      if (!llvm::all_of(results, [&](Value result) {
            auto op = fusion_pattern.findLastWriter(result);
            if (op == dominant_op) {
              return true;
            }
            Value shape = getEffectiveShape(fusion_pattern, result);
            return isRank2RowReduction(op)
                       ? shapeAnalysis.isShapeEqual(ref_shape, shape)
                       : shapeAnalysis.isSameNumElements(ref_shape, shape);
          })) {
        fusion_type = FusionType::kStitch;
      }
    } else if (!col_reductions.empty()) {
      fusion_type = FusionType::kColReduction;
      dominant_op = col_reductions.back();
      Value ref = cast<lmhlo::LmhloOp>(dominant_op).getResultBuffer();
      Value ref_shape = getEffectiveShape(fusion_pattern, ref);
      if (!llvm::all_of(results, [&](Value result) {
            auto op = fusion_pattern.findLastWriter(result);
            if (op == dominant_op) {
              return true;
            }
            Value shape = getEffectiveShape(fusion_pattern, result);
            return isRank2ColReduction(op)
                       ? shapeAnalysis.isShapeEqual(ref_shape, shape)
                       : shapeAnalysis.isSameNumElements(ref_shape, shape);
          })) {
        return false;
      }
    } else {
      for (Operation* op : fusion_pattern.getOpList()) {
        if (FusionStrategy::isFusible(op)) {
          // Ignore if already a kRowReduction or kColReduction, otherwise
          // update the fusion type to kLoop and dominant op to current op. This
          // supposes that the last op inside the block is a valid candidate
          // dominant op if the fusion pattern is a kLoop.
          if (fusion_type == FusionType::kNone ||
              fusion_type == FusionType::kLoop) {
            fusion_type = FusionType::kLoop;
            dominant_op = op;
          }
        } else if (!isa<lmhlo::TerminatorOp>(op)) {
          // Not a supported fusionOp, early stop.
          fusion_type = FusionType::kNone;
          dominant_op = nullptr;
          break;
        }
      }
      if (fusion_type == FusionType::kLoop) {
        Value ref_shape = getEffectiveShape(fusion_pattern, results[0]);
        if (!llvm::all_of(results, [&](Value result) {
              Value shape = getEffectiveShape(fusion_pattern, result);
              return shapeAnalysis.isSameNumElements(ref_shape, shape);
            })) {
          return false;
        }
      }
    }
  }

  if (fusion_type == FusionType::kNone || dominant_op == nullptr) {
    return false;
  } else if (fusion_type == FusionType::kStitch && !col_reductions.empty()) {
    // Stitch fusion does not support col-reduction now.
    return false;
  }

  fusion_pattern.setDominantOp(dominant_op);
  fusion_pattern.setFusionType(fusion_type);
  if (fusion_type == FusionType::kStitch) {
    fusion_pattern.setSubRootOps(row_reductions);
  }

  return true;
}

bool StitchGpuFusionStrategy::tileXroots(ShapeAnalysis& shapeAnalysis,
                                         FusionPattern& fusion_pattern) {
  DenseMap<Value, TileInfo>& tile_plan = fusion_pattern.getTilePlan();
  tile_plan.clear();

  // Check constraint: all sub-roots have equivalent tiled dims and equivalent
  // non-tiled dims. As all sub-roots are rank-2 row-reductions, the checking is
  // simplified to check shape equality.
  const auto& subroots = fusion_pattern.getSubRootOps();
  const auto& dominant_op = fusion_pattern.getDominantOp();
  if (!llvm::all_of(subroots, [&](Operation* op) {
        if (op == dominant_op) {
          return true;
        } else {
          return shapeAnalysis.isShapeEqual(dominant_op->getOperand(0),
                                            op->getOperand(0));
        }
      })) {
    LLVM_DEBUG(llvm::dbgs() << "Sub-roots do not have the same shape.");
    return false;
  }

  // Tile subroots (row reductions).
  for (Operation* op : subroots) {
    // Tile input dimentions for input.
    auto& in = tile_plan[op->getOperand(0)];
    auto reduce = cast<lmhlo::ReduceOp>(op);
    auto dimensions = reduce.getDimensions().getValues<int64_t>();
    for (auto& en : llvm::enumerate(dimensions)) {
      in.tileSizes[en.value()] = ShapedType::kDynamic;
    }
    // No tile dimention for output.
    tile_plan.try_emplace(op->getOperand(2), std::move(TileInfo()));
  }

  // Tile non-subroot results. Successfully tiled roots are identified as
  // regular xroots, while other roots are identified as irregular xroots.
  auto& regular_xroots = fusion_pattern.getRegularXroots();
  auto& irregular_xroots = fusion_pattern.getIrregularXroots();
  regular_xroots.insert(subroots.begin(), subroots.end());
  DenseSet<Operation*> subroots_set(subroots.begin(), subroots.end());
  const auto& dominant = fusion_pattern.getDominantOp()->getOperand(0);
  auto& dominant_tile = tile_plan[dominant];
  const auto& results = fusion_pattern.getResults();

  SmallVector<int> dominant_non_tiled_dims;
  for (int i = 0; i < dominant.getType().cast<MemRefType>().getRank(); ++i) {
    if (dominant_tile.tileSizes.count(i) == 0) {
      dominant_non_tiled_dims.push_back(i);
    } else {
      // Suppose that non-tiled dims are the continuous most-major dims.
      // TODO(disc): change this once we support sub-roots other than
      // row-reduction.
      break;
    }
  }
  for (auto res : results) {
    bool irregular = false;
    Operation* op = fusion_pattern.findLastWriter(res);
    if (subroots_set.contains(op)) {
      continue;
    }
    if (auto analysis_deprecated =
            dynamic_cast<ShapeAnalysisDeprecated*>(&shapeAnalysis)) {
      auto& shapeAnalysis = *analysis_deprecated;
      // Find equal dims between res and a sub-root.
      SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> equal;
      shapeAnalysis.extractContinuousDimEqualInfo(dominant, res, equal);
      int64_t dominant_non_tiled_dim_matched_n = 0;
      DenseSet<int64_t> res_non_tiled_dims;
      for (auto eq_dim : equal) {
        auto lhs = eq_dim.first;
        bool non_tiled = (dominant_tile.tileSizes.count(lhs[0]) == 0);
        for (int64_t i = 1; i < lhs.size(); i++) {
          if ((dominant_tile.tileSizes.count(lhs[i]) == 0) != non_tiled) {
            // Both tiled and non-tiled dims are mapped to the same dim of res,
            // thus cannot determine the tile plan for res;
            irregular_xroots.insert(op);
            irregular = true;
            break;
          }
        }
        if (irregular) {
          break;
        }
        if (non_tiled) {
          res_non_tiled_dims.insert(eq_dim.second.begin(), eq_dim.second.end());
          dominant_non_tiled_dim_matched_n += lhs.size();
        }
      }
      if (irregular) {
        continue;
      } else if (dominant_non_tiled_dim_matched_n != 1) {
        // Check constraint: the element-number of non-tiled dims are the same
        // between sub-roots and regular xroots. If not every non-tiled dim is
        // matched between sub-root and current op, it is an irregular xroot.
        // Note we know that sub-root is reduction and has only 1 non-tiled dim.
        irregular_xroots.insert(op);
        continue;
      } else if (*std::max_element(res_non_tiled_dims.begin(),
                                   res_non_tiled_dims.end()) !=
                 res_non_tiled_dims.size() - 1) {
        // Check constraint: the tiled dims are all minor dims for all regular
        // xroot ops. This means the non-tiled dims should all be the major
        // dims.
        irregular_xroots.insert(op);
        continue;
      }

      auto& plan = tile_plan[res];
      int64_t rank = res.getType().cast<MemRefType>().getRank();
      for (int64_t i = 0; i < rank; i++) {
        if (res_non_tiled_dims.count(i) == 0) {
          // Note that we only log whether a dimension is tiled or not. We do
          // not care about the tiling size. Thus we set all tiled dims with
          // with the value `kDynamic`.
          plan.tileSizes[i] = ShapedType::kDynamic;
        }
      }
      regular_xroots.insert(op);
    } else {
      int last_non_tied_dim = -1;
      SmallVector<int> res_non_tiled_dims;
      int result_rank = res.getType().cast<MemRefType>().getRank();
      do {
        if (shapeAnalysis.isProductEqual(dominant, dominant_non_tiled_dims, res,
                                         res_non_tiled_dims)) {
          break;
        }
        ++last_non_tied_dim;
        res_non_tiled_dims.push_back(last_non_tied_dim);
      } while (last_non_tied_dim < result_rank);

      if (last_non_tied_dim >= result_rank) {
        irregular = true;
        irregular_xroots.insert(op);
        continue;
      }

      auto& plan = tile_plan[res];
      for (int i = last_non_tied_dim + 1; i < result_rank; ++i) {
        // Note that we only log whether a dimension is tiled or not. We do
        // not care about the tiling size. Thus we set all tiled dims with
        // with the value `kDynamic`.
        plan.tileSizes[i] = ShapedType::kDynamic;
      }
      regular_xroots.insert(op);
    }
  }

  // Check constraint: all the external-only-roots should be regular. This is
  // because all external-only-roots are skeleton ops. It requires all skeleton
  // ops have the same non-tiling element numbers to ease the building of loop
  // loop skeleton of the fusion during code generation.
  for (auto extern_only_res : fusion_pattern.getExternalOnlyResults()) {
    Operation* op = fusion_pattern.findLastWriter(extern_only_res);
    if (irregular_xroots.contains(op)) {
      LLVM_DEBUG(llvm::dbgs() << "Invalid irregular xroot: " << *op);
      return false;
    }
  }

  return true;
}

bool StitchGpuFusionStrategy::backtraceTileAndCover(
    ShapeAnalysis& shapeAnalysis, FusionPattern& fusion_pattern, Value value) {
  const auto& op_list = fusion_pattern.getOpList();
  const auto& subroots = fusion_pattern.getSubRootOps();
  const auto& roots = fusion_pattern.getRootOps();
  DenseMap<Value, TileInfo>& tile_plan = fusion_pattern.getTilePlan();
  DenseSet<Operation*> op_set(op_list.begin(), op_list.end());
  DenseSet<Operation*> subroots_set(subroots.begin(), subroots.end());
  DenseSet<Operation*> roots_set(roots.begin(), roots.end());
  std::function<bool(Value, bool&)> propagateO2I;
  propagateO2I = [&](Value value, bool& cover) {
    auto op = fusion_pattern.findLastWriter(value);
    if (!op_set.contains(op) || subroots_set.contains(op)) {
      // Note we already know both the input and output tile info for subroots.
      return true;
    }
    SmallVector<std::pair<Value, TileInfo>, 4> in_info;
    if (tileCoverInfoPropagateO2I(shapeAnalysis, tile_plan, op, in_info,
                                  cover)) {
      for (auto& info : in_info) {
        // Check constraint: if an xroot is not skeleton, all results of this op
        // should be able to be inferred by skeleton. The simplified logic is to
        // whether a root op's `cover` is false in O2I propagation.
        auto in_op = fusion_pattern.findLastWriter(info.first);
        if (roots_set.contains(in_op) && !cover) {
          LLVM_DEBUG(llvm::dbgs() << "skeleton fails to cover root: " << *op);
          return false;
        }
        auto exist_tile_info = tile_plan.find(info.first);
        if (exist_tile_info != tile_plan.end()) {
          // When we meet a previously propagated tile info, the newly generated
          // should be the same with previous one.
          if (exist_tile_info->second.tileSizes != info.second.tileSizes) {
            LLVM_DEBUG(llvm::dbgs() << "conflict tile info: " << *op);
            return false;
          }
        } else {
          tile_plan[info.first] = info.second;
          if (!propagateO2I(info.first, cover)) {  // backtracing
            return false;                          // early break
          }
        }
      }
      return true;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "tile and cover propagation failed: " << *op);
      return false;
    }
  };
  bool cover = true;
  return propagateO2I(value, cover);
}

bool StitchGpuFusionStrategy::initFusionPattern(ShapeAnalysis& shapeAnalysis,
                                                FusionPattern& fusion_pattern) {
  const auto& results = fusion_pattern.getResults();
  if (results.empty()) {
    return false;
  }

  if (!findFusionPatternTypeAndSubroot(shapeAnalysis, fusion_pattern)) {
    return false;
  } else if (fusion_pattern.getFusionType() != FusionType::kStitch) {
    return true;
  }
  FusionType fusion_type = fusion_pattern.getFusionType();
  Operation* dominant_op = fusion_pattern.getDominantOp();
  const auto& subroots = fusion_pattern.getSubRootOps();

  // Analyze tile information of sub-roots and roots, identify regular/irregular
  // xroots.
  if (!tileXroots(shapeAnalysis, fusion_pattern)) {
    return false;
  }
  DenseMap<Value, TileInfo>& tile_plan = fusion_pattern.getTilePlan();

  // Propagate tile information and data covering status back from skeleton ops.
  DenseSet<Operation*> skeleton_op_set(subroots.begin(), subroots.end());
  const auto& external_only_results = fusion_pattern.getExternalOnlyResults();
  for (Value res : external_only_results) {
    Operation* op = fusion_pattern.findLastWriter(res);
    skeleton_op_set.insert(op);
  }
  DenseSet<Operation*> subroots_set(subroots.begin(), subroots.end());
  for (auto op : skeleton_op_set) {
    Value value;
    if (subroots_set.contains(op)) {
      // Propagate from input for subroot (row-reduction).
      value = op->getOperand(0);
    } else {
      // Propagate from output, because we do not want to iterate on all inputs
      // one by one.
      value = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    }
    if (!backtraceTileAndCover(shapeAnalysis, fusion_pattern, value)) {
      return false;
    }
  }

  // TODO: global memory buffer for intermeidate common operands.
  // TODO: add speculation hint here.

  return true;
}

}  // namespace disc_ral
}  // namespace mlir
