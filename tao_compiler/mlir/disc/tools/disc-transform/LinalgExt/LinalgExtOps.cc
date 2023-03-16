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

#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtDialect.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtEnums.cc.inc"

namespace mlir {
namespace disc_ral {
namespace disc_linalg_ext {

static void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects,
    ValueRange results, ValueRange inputBuffers, ValueRange outputBuffers) {
  for (Value value : results) {
    effects.emplace_back(MemoryEffects::Allocate::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : inputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : outputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), value,
                         SideEffects::DefaultResource::get());
  }
}

Value getDimValue(OpBuilder& builder, Location loc, Value v, int64_t dim) {
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return builder.create<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType t) -> Value {
        return builder.create<memref::DimOp>(loc, v, dim);
      });
}

OpFoldResult getDim(OpBuilder& builder, Location loc, Value v, int64_t dim) {
  auto t = v.getType().cast<ShapedType>();
  if (t.isDynamicDim(dim)) {
    return getDimValue(builder, loc, v, dim);
  }
  return builder.getI64IntegerAttr(t.getDimSize(dim));
}

SmallVector<OpFoldResult> getDims(OpBuilder& builder, Location loc,
                                  Value shapedTypeValue) {
  return llvm::to_vector(llvm::map_range(
      llvm::seq<int64_t>(
          0, shapedTypeValue.getType().cast<ShapedType>().getRank()),
      [&](int64_t dim) { return getDim(builder, loc, shapedTypeValue, dim); }));
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

static int64_t ceilDiv(int64_t x, int64_t y) { return (x + y - 1) / y; }

/// Custom builder methods for pack ops.
void MultiLevelPackOp::build(OpBuilder& builder, OperationState& state,
                             Value source, Value output,
                             ArrayRef<int64_t> tileLevels,
                             ArrayRef<int64_t> tileSizes,
                             ArrayRef<int64_t> permutation,
                             Optional<Value> paddingValue) {
  SmallVector<int64_t> permutationVec;
  int64_t expectedResultRank =
      MultiLevelPackOp::getExpectedResultRank(tileLevels);
  if (expectedResultRank > 0 && permutation.empty()) {
    permutationVec = llvm::to_vector<>(
        llvm::seq(static_cast<int64_t>(0), expectedResultRank));
    permutation = permutationVec;
  }
  ShapedType resultType = getPackedType(source.getType().cast<ShapedType>(),
                                        tileLevels, tileSizes, permutation);
  build(builder, state, resultType, source, output,
        builder.getI64ArrayAttr(tileLevels), builder.getI64ArrayAttr(tileSizes),
        builder.getI64ArrayAttr(permutation),
        (paddingValue ? paddingValue.value() : nullptr));
}

/* static */ ShapedType MultiLevelPackOp::getPackedType(
    ShapedType inputType, ArrayRef<int64_t> tileLevels,
    ArrayRef<int64_t> tileSizes, ArrayRef<int64_t> permutation) {
  int expectedResultRank = MultiLevelPackOp::getExpectedResultRank(tileLevels);
  SmallVector<int64_t> tiledShape(expectedResultRank, ShapedType::kDynamic);
  int tileSizeIdx = 0;
  int tiledDimIdx = 0;
  for (int dimIdx = 0; dimIdx < inputType.getRank(); ++dimIdx) {
    int64_t dimSize = inputType.getShape()[dimIdx];
    int64_t level = tileLevels[dimIdx];
    int lastTileSize = 1;
    for (int localTiledDimIdx = level; localTiledDimIdx > 0;
         --localTiledDimIdx) {
      int64_t tileSize = tileSizes[tileSizeIdx + localTiledDimIdx - 1];
      tiledShape[tiledDimIdx + localTiledDimIdx] =
          ceilDiv(tileSize, lastTileSize);
      lastTileSize = tileSize;
    }
    if (dimSize != ShapedType::kDynamic)
      tiledShape[tiledDimIdx] = ceilDiv(dimSize, lastTileSize);
    tileSizeIdx += level;
    tiledDimIdx += 1 + level;
  }

  if (!permutation.empty()) {
    tiledShape = interchange<int64_t>(tiledShape, permutation, /*offset=*/0);
  }

  return TypeSwitch<ShapedType, ShapedType>(inputType)
      .Case<RankedTensorType>([&](auto shapedType) {
        return RankedTensorType::get(tiledShape, shapedType.getElementType());
      })
      .Case<MemRefType>([&](auto shapedType) {
        return MemRefType::get(tiledShape, shapedType.getElementType());
      })
      .Default([&](Type t) {
        assert(false && "unexpected type");
        return nullptr;
      });
}

LogicalResult MultiLevelPackOp::verify() {
  Operation* op = getOperation();
  int64_t inputRank = getInputRank();
  if (inputRank != getTileLevels().size()) {
    return op->emitError("mismatch input rank and the size of tile_levels ")
           << inputRank << " vs " << getTileLevels().size() << "\n";
  }
  int64_t expectedResultRank = getExpectedResultRank();
  if (expectedResultRank != getPermutation().size()) {
    return op->emitError(
               "mismatch expected output rank and the size of permutation ")
           << expectedResultRank << " vs " << getPermutation().size() << "\n";
  }
  if (expectedResultRank != getOutputRank()) {
    return op->emitError(
               "mismatch expected output rank and the rank of the output "
               "operand ")
           << expectedResultRank << " vs " << getOutputRank() << "\n";
  }

  auto sortedPermutation = getPermutationVec();
  llvm::sort(sortedPermutation);
  if (!sortedPermutation.empty() &&
      (sortedPermutation[0] != 0 ||
       sortedPermutation[expectedResultRank - 1] != expectedResultRank - 1)) {
    return op->emitError("not a valid permutation setting\n");
  }

  auto tileLevels = getTileLevelsVec();
  auto tileSizes = getTileSizesVec();
  auto permutation = getPermutationVec();
  auto expectedType =
      getPackedType(getInputType(), tileLevels, tileSizes, permutation);
  if (!expectedType) {
    return op->emitError("failed to infer the packed type\n");
  }
  if (expectedType != getOutputType()) {
    return op->emitError(
               "mismatch expected output type and actual output type ")
           << expectedType << " vs " << getOutputType() << "\n";
  }

  return success();
}

void MultiLevelPackOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  SmallVector<Value> inputBuffers = getInputBufferOperands();
  SmallVector<Value> outputBuffers = getOutputBufferOperands();
  getEffectsImpl(effects, getOperation()->getResults(), inputBuffers,
                 outputBuffers);
}

SmallVector<OpFoldResult> MultiLevelPackOp::getResultShape(
    OpBuilder& builder, Location loc, ArrayRef<OpFoldResult> sourceDims,
    ArrayRef<int64_t> tileLevels, ArrayRef<int64_t> tileSizes,
    ArrayRef<int64_t> permutation) {
  int expectedResultRank = getExpectedResultRank(tileLevels);
  SmallVector<OpFoldResult> resultDims(expectedResultRank);

  auto const2IndexAttr = [&](int64_t val) {
    return IntegerAttr::get(builder.getIndexType(), val);
  };

  AffineExpr s0, s1;
  bindSymbols(builder.getContext(), s0, s1);
  AffineExpr ceilDivExpr = s0.ceilDiv(s1);

  int tileSizeIdx = 0;
  int resultDimIdx = 0;
  for (int dimIdx = 0; dimIdx < sourceDims.size(); ++dimIdx) {
    int64_t level = tileLevels[dimIdx];
    OpFoldResult dimSize = sourceDims[dimIdx];
    OpFoldResult lastTileSize = const2IndexAttr(1);
    for (int localResultDimIdx = level; localResultDimIdx > 0;
         --localResultDimIdx) {
      OpFoldResult tileSize =
          const2IndexAttr(tileSizes[tileSizeIdx + localResultDimIdx - 1]);
      resultDims[resultDimIdx + localResultDimIdx] =
          makeComposedFoldedAffineApply(builder, loc, ceilDivExpr,
                                        {tileSize, lastTileSize});
      lastTileSize = tileSize;
    }
    resultDims[resultDimIdx] = makeComposedFoldedAffineApply(
        builder, loc, ceilDivExpr, {dimSize, lastTileSize});
    tileSizeIdx += level;
    resultDimIdx += 1 + level;
  }

  if (!permutation.empty()) {
    resultDims =
        interchange<OpFoldResult>(resultDims, permutation, /*offset=*/0);
  }
  return resultDims;
}

SmallVector<OpFoldResult> MultiLevelPackOp::getResultShape(OpBuilder& builder) {
  auto tileLevels = getTileLevelsVec();
  auto tileSizes = getTileSizesVec();
  auto permutation = getPermutationVec();
  return getResultShape(builder, getLoc(),
                        getDims(builder, getLoc(), getInput()), tileLevels,
                        tileSizes, permutation);
}

LogicalResult MultiLevelPackOp::reifyResultShapes(
    OpBuilder& builder, ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(getOperation());
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0] = getValueOrCreateConstantIndexOp(
      builder, getLoc(), getResultShape(builder));
  return success();
}

static SmallVector<uint64_t> delinearize(uint64_t idx,
                                         ArrayRef<int64_t> shape) {
  SmallVector<uint64_t> indices(shape.size());
  for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
    indices[d] = idx % shape[d];
    idx /= shape[d];
  }
  return indices;
}

LogicalResult MultiLevelPackOp::fold(ArrayRef<Attribute> operands,
                                     SmallVectorImpl<OpFoldResult>& results) {
  if (operands.size() < 1 || !operands[0]) return failure();
  auto srcElemAtts = operands[0].dyn_cast<ElementsAttr>();
  if (!srcElemAtts) return failure();

  auto srcTy = srcElemAtts.getType();
  int srcRank = srcTy.getRank();
  auto dstTy = this->getOutputType();
  if (!dstTy.hasStaticShape()) return failure();
  int dstRank = dstTy.getRank();

  auto tileLevelsVec = this->getTileLevelsVec();
  auto tileSizesVec = this->getTileSizesVec();
  auto permutationVec = this->getPermutationVec();
  auto logicalDim2SrcDim =
      this->getOutputLogicalDimToInputDimMapping(tileLevelsVec, tileSizesVec);
  auto logicalDim2TileSize =
      this->getOutputLogicalDimToTileSizeMapping(tileLevelsVec, tileSizesVec);

  SmallVector<Attribute> dstAttrs;
  for (uint64_t idx = 0; idx < dstTy.getNumElements(); ++idx) {
    auto indices = delinearize(idx, dstTy.getShape());
    SmallVector<uint64_t> srcIndices(srcRank, 0);
    for (int dstIdx = 0; dstIdx < dstRank; ++dstIdx) {
      int logicalIdx = permutationVec[dstIdx];
      int srcIdx = logicalDim2SrcDim[logicalIdx];
      srcIndices[srcIdx] += indices[dstIdx] * logicalDim2TileSize[logicalIdx];
    }
    if (srcElemAtts.isValidIndex(srcIndices)) {
      dstAttrs.push_back(srcElemAtts.getValues<Attribute>()[srcIndices]);
    } else {
      if (operands.size() < 3 || !operands[2]) return failure();
      dstAttrs.push_back(operands[2]);
    }
  }

  results.push_back(DenseElementsAttr::get(dstTy, dstAttrs));
  return success();
}

//===----------------------------------------------------------------------===//
// ConditionalGenericOp
//===----------------------------------------------------------------------===//

/// Common parsing used for both named structured ops created by ods-gen and by
/// manually defined C++ ops. Does not handle regions.
static ParseResult parseCommonStructuredOpParts(
    OpAsmParser& parser, OperationState& result,
    SmallVectorImpl<Type>& inputTypes, SmallVectorImpl<Type>& outputTypes,
    bool addOperandSegmentSizes = true) {
  SMLoc inputsOperandsLoc, outputsOperandsLoc;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands,
      outputsOperands;

  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseLParen()) return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    outputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseLParen() || parser.parseOperandList(outputsOperands) ||
        parser.parseColonTypeList(outputTypes) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(outputsOperands, outputTypes, outputsOperandsLoc,
                             result.operands))
    return failure();

  if (addOperandSegmentSizes) {
    result.addAttribute("operand_segment_sizes",
                        parser.getBuilder().getDenseI32ArrayAttr(
                            {static_cast<int32_t>(inputsOperands.size()),
                             static_cast<int32_t>(outputsOperands.size())}));
  }
  return success();
}

static void printCommonStructuredOpParts(OpAsmPrinter& p, ValueRange inputs,
                                         ValueRange outputs) {
  if (!inputs.empty())
    p << " ins(" << inputs << " : " << inputs.getTypes() << ")";
  if (!outputs.empty())
    p << " outs(" << outputs << " : " << outputs.getTypes() << ")";
}

static void printNamedStructuredOpResults(OpAsmPrinter& p,
                                          TypeRange resultTypes) {
  if (resultTypes.empty()) return;
  p.printOptionalArrowTypeList(resultTypes);
}

static ParseResult parseNamedStructuredOpResults(
    OpAsmParser& parser, SmallVectorImpl<Type>& resultTypes) {
  if (parser.parseOptionalArrowTypeList(resultTypes)) return failure();
  return success();
}

void ConditionalGenericOp::getAsmBlockArgumentNames(
    Region& region, OpAsmSetValueNameFn setNameFn) {
  for (Value v : getRegionInputArgs()) setNameFn(v, "in");
  for (Value v : getRegionOutputArgs()) setNameFn(v, "out");
}

void ConditionalGenericOp::build(
    OpBuilder& builder, OperationState& result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayAttr indexingMaps,
    ArrayAttr iteratorTypes, StringAttr doc, StringAttr libraryCall,
    function_ref<void(OpBuilder&, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, resultTensorTypes, inputs, outputs, indexingMaps,
        iteratorTypes, doc, libraryCall);
  result.addAttributes(attributes);
  if (!bodyBuild) return;

  SmallVector<Type, 4> blockArgTypes;
  SmallVector<Location, 4> blockArgLocs;
  for (ValueRange container : {inputs, outputs}) {
    for (Value v : container) {
      blockArgTypes.push_back(getElementTypeOrSelf(v));
      blockArgLocs.push_back(v.getLoc());
    }
  }

  OpBuilder::InsertionGuard guard(builder);
  auto& region = *result.regions.front();
  Block* bodyBlock =
      builder.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
  bodyBuild(builder, result.location, bodyBlock->getArguments());
}

void ConditionalGenericOp::build(
    OpBuilder& builder, OperationState& result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes, StringRef doc,
    StringRef libraryCall,
    function_ref<void(OpBuilder&, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, resultTensorTypes, inputs, outputs,
        builder.getAffineMapArrayAttr(indexingMaps),
        builder.getArrayAttr(llvm::to_vector(llvm::map_range(
            iteratorTypes,
            [&](utils::IteratorType iter) -> mlir::Attribute {
              return linalg::IteratorTypeAttr::get(builder.getContext(), iter);
            }))),
        doc.empty() ? StringAttr() : builder.getStringAttr(doc),
        libraryCall.empty() ? StringAttr() : builder.getStringAttr(libraryCall),
        bodyBuild, attributes);
}

void ConditionalGenericOp::build(
    OpBuilder& builder, OperationState& result, ValueRange inputs,
    ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes, StringRef doc,
    StringRef libraryCall,
    function_ref<void(OpBuilder&, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, TypeRange{}, inputs, outputs, indexingMaps,
        iteratorTypes, doc, libraryCall, bodyBuild, attributes);
}

void ConditionalGenericOp::build(
    OpBuilder& builder, OperationState& result, ValueRange inputs,
    ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<void(OpBuilder&, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, inputs, outputs, indexingMaps, iteratorTypes,
        /*doc=*/"",
        /*libraryCall=*/"", bodyBuild, attributes);
}

void ConditionalGenericOp::build(
    OpBuilder& builder, OperationState& result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<void(OpBuilder&, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, resultTensorTypes, inputs, outputs, indexingMaps,
        iteratorTypes,
        /*doc=*/"",
        /*libraryCall=*/"", bodyBuild, attributes);
}

void ConditionalGenericOp::print(OpAsmPrinter& p) {
  p << " ";

  // Print extra attributes.
  auto genericAttrNames = linalgTraitAttrNames();

  llvm::StringSet<> genericAttrNamesSet;
  genericAttrNamesSet.insert(genericAttrNames.begin(), genericAttrNames.end());
  SmallVector<NamedAttribute, 8> genericAttrs;
  for (auto attr : (*this)->getAttrs()) {
    if (attr.getName() == getIteratorTypesAttrName()) {
      auto iteratorTypes =
          attr.getValue()
              .cast<ArrayAttr>()
              .getAsValueRange<linalg::IteratorTypeAttr, utils::IteratorType>();
      // Convert IteratorType enums into the string representation. This is
      // needed, because tests still use the old format when 'iterator_types'
      // attribute is represented as an array of strings.
      // TODO: Remove this conversion once tests are fixed.
      SmallVector<Attribute> iteratorTypeNames =
          llvm::to_vector(llvm::map_range(
              iteratorTypes, [&](utils::IteratorType t) -> Attribute {
                return StringAttr::get(getContext(), stringifyIteratorType(t));
              }));

      genericAttrs.emplace_back(
          getIteratorTypesAttrName(),
          ArrayAttr::get(getContext(), iteratorTypeNames));
    } else if (genericAttrNamesSet.count(attr.getName().strref()) > 0) {
      genericAttrs.push_back(attr);
    }
  }
  if (!genericAttrs.empty()) {
    auto genericDictAttr = DictionaryAttr::get(getContext(), genericAttrs);
    p << genericDictAttr;
  }

  // Printing is shared with named ops, except for the region and attributes
  printCommonStructuredOpParts(p, SmallVector<Value>(getDpsInputOperands()),
                               SmallVector<Value>(getDpsInitOperands()));

  genericAttrNames.push_back("operand_segment_sizes");
  genericAttrNamesSet.insert(genericAttrNames.back());

  bool hasExtraAttrs = false;
  for (NamedAttribute n : (*this)->getAttrs()) {
    if ((hasExtraAttrs = !genericAttrNamesSet.contains(n.getName().strref())))
      break;
  }
  if (hasExtraAttrs) {
    p << " attrs = ";
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/genericAttrNames);
  }

  // Print region.
  if (!getRegion().empty()) {
    p << ' ';
    p.printRegion(getRegion());
  }

  // Print results.
  printNamedStructuredOpResults(p, getResultTensors().getTypes());
}

ParseResult ConditionalGenericOp::parse(OpAsmParser& parser,
                                        OperationState& result) {
  DictionaryAttr dictAttr;
  // Parse the core linalg traits that must check into a dictAttr.
  // The name is unimportant as we will overwrite result.attributes.
  // The core linalg traits must contain the information necessary to pass the
  // verifier.
  if (parser.parseAttribute(dictAttr, "_", result.attributes)) return failure();
  result.attributes.assign(dictAttr.getValue().begin(),
                           dictAttr.getValue().end());

  // Convert array of string into an array of IteratyType enums. This is needed,
  // because tests still use the old format when 'iterator_types' attribute is
  // represented as an array of strings.
  // TODO: Remove this conversion once tests are fixed.
  ArrayAttr iteratorTypes =
      result.attributes.get(getIteratorTypesAttrName(result.name))
          .cast<ArrayAttr>();

  SmallVector<Attribute> iteratorTypeAttrs;

  for (StringRef s : iteratorTypes.getAsValueRange<StringAttr>()) {
    auto maybeIteratorType = utils::symbolizeIteratorType(s);
    if (!maybeIteratorType.has_value())
      return parser.emitError(parser.getCurrentLocation())
             << "unexpected iterator_type (" << s << ")";

    iteratorTypeAttrs.push_back(linalg::IteratorTypeAttr::get(
        parser.getContext(), maybeIteratorType.value()));
  }
  result.attributes.set(getIteratorTypesAttrName(result.name),
                        parser.getBuilder().getArrayAttr(iteratorTypeAttrs));

  // Parsing is shared with named ops, except for the region.
  SmallVector<Type, 1> inputTypes, outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes))
    return failure();

  // Optional attributes may be added.
  if (succeeded(parser.parseOptionalKeyword("attrs")))
    if (failed(parser.parseEqual()) ||
        failed(parser.parseOptionalAttrDict(result.attributes)))
      return failure();

  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parser.parseRegion(*region, {})) return failure();
  result.addRegion(std::move(region));

  // Generic ops may specify that a subset of its outputs are tensors. Such
  // outputs are specified in the result type.
  // TODO: may need to move output parsing before region parsing.
  // Need to wait for declarative assembly resolution to decide.
  SmallVector<Type, 1> outputTensorsTypes;
  if (parseNamedStructuredOpResults(parser, outputTensorsTypes))
    return failure();
  result.addTypes(outputTensorsTypes);

  return success();
}

static void getGenericEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects,
    ValueRange results, OpOperandVector inputOperands,
    OpOperandVector outputOperands) {
  for (auto* operand : inputOperands) {
    if (!operand->get().getType().isa<MemRefType>()) continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand->get(),
                         SideEffects::DefaultResource::get());
  }
  for (auto* operand : outputOperands) {
    if (!operand->get().getType().isa<MemRefType>()) continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand->get(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), operand->get(),
                         SideEffects::DefaultResource::get());
  }
}

void ConditionalGenericOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(),
                        getDpsInputOperands(), getDpsInitOperands());
}

LogicalResult ConditionalGenericOp::verify() {
  if (getInputs().size() < 1)
    return emitOpError("expected the first input is the pred");
  if (!getInputs().front().getType().isInteger(1))
    return emitOpError("expected the pred having i1 type");
  return success();
}

void ConditionalGenericOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context) {}

//===----------------------------------------------------------------------===//
// Utility methods for implementation of Tiling Interface for Conditional
// Generic Op
//===----------------------------------------------------------------------===//

/// Return the SSA values that represent the data point accessed using a given
/// `indexingMap` for a given point in the iteration space represented by `ivs`.
static SmallVector<Value> getIndicesForAccess(OpBuilder& b, Location loc,
                                              AffineMap indexingMap,
                                              ValueRange ivs) {
  SmallVector<Value> indices;
  indices.reserve(indexingMap.getNumResults());
  for (auto result : indexingMap.getResults()) {
    AffineMap m = AffineMap::get(indexingMap.getNumDims(),
                                 indexingMap.getNumSymbols(), result);
    Value v = b.create<AffineApplyOp>(loc, m, ivs);
    indices.push_back(v);
  }
  return indices;
}

/// Method to inline the payload of a `linalgOp` given the iteration space
/// point and values for the arguments of the payload.
static LogicalResult inlinePayload(OpBuilder& b, LinalgOp linalgOp,
                                   ValueRange ivs, ValueRange argValues) {
  Block* body = linalgOp.getBlock();
  IRMapping map;
  map.map(body->getArguments(), argValues);
  for (auto& op : body->without_terminator()) {
    if (auto indexOp = dyn_cast<IndexOp>(&op)) {
      map.map(indexOp.getResult(), ivs[indexOp.getDim()]);
      continue;
    }
    b.clone(op, map);
  }

  Operation* terminator = body->getTerminator();
  Location loc = terminator->getLoc();
  for (const auto& operand : llvm::enumerate(terminator->getOperands())) {
    Value toStore = map.lookupOrDefault(operand.value());
    OpOperand* storeInto = linalgOp.getDpsInitOperand(operand.index());
    auto indices = getIndicesForAccess(
        b, loc, linalgOp.getMatchingIndexingMap(storeInto), ivs);
    b.create<memref::StoreOp>(
        loc, toStore, linalgOp.getDpsInitOperand(operand.index())->get(),
        indices);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// External Model for implementing `TilingInterface` for `LinalgOp`s.
//===----------------------------------------------------------------------===//

namespace {
/// External model implementation of TilingInterface for LinalgOps. An external
/// model implementation is used for now till the use of `TilingInterface` is
/// on-par with the current Linalg tiling + fusion patterns. Once it is
/// maybe possible to move this into the op-definition (though there are
/// advantages to leaving it as an external model)
template <typename LinalgOpTy>
struct LinalgOpTilingInterface
    : public TilingInterface::ExternalModel<LinalgOpTilingInterface<LinalgOpTy>,
                                            LinalgOpTy> {
  /// Return the loop iterator type.
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation* op) const {
    LinalgOpTy concreteOp = cast<LinalgOpTy>(op);
    return concreteOp.getIteratorTypesArray();
  }

  /// Return the iteration domain range.
  SmallVector<Range> getIterationDomain(Operation* op, OpBuilder& b) const {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);
    Location loc = op->getLoc();
    LinalgOp linalgOp = cast<LinalgOp>(op);
    SmallVector<OpFoldResult> allShapesSizes =
        linalgOp.createFlatListOfOperandDims(b, loc);
    AffineMap map = linalgOp.getShapesToLoopsMap();

    return llvm::to_vector(
        llvm::map_range(map.getResults(), [&](AffineExpr loopExpr) {
          OpFoldResult ofr =
              makeComposedFoldedAffineApply(b, loc, loopExpr, allShapesSizes);
          return Range{b.getIndexAttr(0), ofr, b.getIndexAttr(1)};
        }));
  }

  // Instantiate the tiled implementation of the operation.
  SmallVector<Operation*> getTiledImplementation(
      Operation* op, OpBuilder& b, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    // Leave the `sizeBounds` value empty. That is only needed when the `sizes`
    // specified could lead to out of bounds accesses.
    Location loc = op->getLoc();
    LinalgOp linalgOp = cast<LinalgOp>(op);
    SmallVector<Value> valuesToTile = linalgOp->getOperands();
    SmallVector<Value, 4> tiledOperands = linalg::makeTiledShapes(
        b, loc, linalgOp, valuesToTile, offsets, sizes, {}, true);

    SmallVector<Type> resultTensorTypes =
        getTensorOutputTypes(linalgOp, tiledOperands);

    Operation* tiledOp = mlir::clone(b, op, resultTensorTypes, tiledOperands);
    offsetIndices(b, cast<LinalgOp>(tiledOp), offsets);

    return {tiledOp};
  }

  // Return the details of the output tile generated by the tiled
  // implementation.
  LogicalResult getResultTilePosition(
      Operation* op, OpBuilder& b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVector<OpFoldResult>& resultOffsets,
      SmallVector<OpFoldResult>& resultSizes) const {
    Location loc = op->getLoc();
    LinalgOp linalgOp = cast<LinalgOp>(op);

    AffineExpr d0;
    bindDims(b.getContext(), d0);
    SmallVector<OpFoldResult> subShapeSizes =
        llvm::to_vector(llvm::map_range(sizes, [&](OpFoldResult ofr) {
          return makeComposedFoldedAffineApply(b, loc, d0 - 1, ofr);
        }));

    OpOperand* outOperand = linalgOp.getDpsInitOperand(resultNumber);
    linalg::SliceParameters sliceParams = linalg::computeSliceParameters(
        b, loc, outOperand->get(), sizes,
        linalgOp.getMatchingIndexingMap(outOperand), offsets,
        /*ubs*/ {}, subShapeSizes, true);
    resultOffsets = sliceParams.offsets;
    resultSizes = sliceParams.sizes;
    return success();
  }

  FailureOr<Value> generateResultTileValue(Operation* op, OpBuilder& b,
                                           unsigned resultNumber,
                                           ArrayRef<OpFoldResult> offsets,
                                           ArrayRef<OpFoldResult> sizes) const {
    auto linalgOp = cast<LinalgOp>(op);

    // Check that the indexing map used for the output is a projected
    // permutation. This could be relaxed with a more general approach that can
    // map the offsets and sizes from the result to iteration space tiles
    // (filling in full extent for dimensions not used to access the result).
    AffineMap indexingMap =
        linalgOp.getIndexingMapMatchingResult(op->getResult(resultNumber));
    if (!indexingMap.isProjectedPermutation()) {
      return op->emitOpError(
          "unhandled tiled implementation generation when result is not "
          "accessed using a permuted projection");
    }

    auto numLoops = linalgOp.getNumLoops();
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    SmallVector<OpFoldResult> iterationTileOffsets(numLoops),
        iterationTileSizes(numLoops);
    if (!indexingMap.isPermutation()) {
      SmallVector<Range> iterationDomain =
          tilingInterfaceOp.getIterationDomain(b);
      for (const auto& range : llvm::enumerate(iterationDomain)) {
        iterationTileOffsets[range.index()] = range.value().offset;
        iterationTileSizes[range.index()] = range.value().size;
      }
    }
    for (const auto& resultExpr : llvm::enumerate(indexingMap.getResults())) {
      unsigned dimPosition =
          resultExpr.value().cast<AffineDimExpr>().getPosition();
      iterationTileOffsets[dimPosition] = offsets[resultExpr.index()];
      iterationTileSizes[dimPosition] = sizes[resultExpr.index()];
    }

    SmallVector<Operation*> tiledOp = tilingInterfaceOp.getTiledImplementation(
        b, iterationTileOffsets, iterationTileSizes);
    if (tiledOp.size() != 1)
      return op->emitOpError("failed to generate tiled implementation");

    return tiledOp[0]->getResult(resultNumber);
  }

  LogicalResult generateScalarImplementation(Operation* op, OpBuilder& builder,
                                             Location loc,
                                             ValueRange ivs) const {
    auto linalgOp = cast<LinalgOp>(op);
    if (!linalgOp.hasBufferSemantics())
      return op->emitOpError("expected operation to have buffer semantics");

    SmallVector<Value> indexedValues;
    indexedValues.reserve(linalgOp->getNumOperands());
    Location linalgOpLoc = op->getLoc();
    /// Load the data corresponding to the block arguments that
    /// represent input operands.
    for (OpOperand& operand : linalgOp->getOpOperands()) {
      if (!linalgOp.payloadUsesValueFromOperand(&operand)) {
        indexedValues.push_back(nullptr);
        continue;
      }
      if (linalgOp.isScalar(&operand)) {
        indexedValues.push_back(operand.get());
        continue;
      }
      SmallVector<Value> indices = getIndicesForAccess(
          builder, linalgOpLoc, linalgOp.getMatchingIndexingMap(&operand), ivs);
      Value load =
          builder.create<memref::LoadOp>(linalgOpLoc, operand.get(), indices);
      indexedValues.push_back(load);
    }

    /// Inline the op payload and store the result.
    return inlinePayload(builder, linalgOp, ivs, indexedValues);
  }
};

template <typename OpType>
static void registerOne(MLIRContext* ctx) {
  OpType::template attachInterface<LinalgOpTilingInterface<OpType>>(*ctx);
}

}  // namespace

void registerTilingInterfaceExternalModels(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, DISCLinalgExtDialect* dialect) {
    registerOne<ConditionalGenericOp>(ctx);
  });
}

//===----------------------------------------------------------------------===//
// External Model for implementing `BufferizableOpInterface` for `LinalgOp`s.
//===----------------------------------------------------------------------===//

namespace {

/// Generic conversion for any DestinationStyleOpInterface on tensors.
static LogicalResult bufferizeDestinationStyleOpInterface(
    RewriterBase& rewriter, DestinationStyleOpInterface op,
    const bufferization::BufferizationOptions& options) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Nothing to do. This op is already bufferized.
  if (op.hasBufferSemantics()) return success();

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasTensorSemantics())
    return op->emitError() << "op does not have tensor semantics";

  // New input operands for the cloned op.
  SmallVector<Value> newInputBuffers;
  newInputBuffers.reserve(op.getNumDpsInputs());
  for (OpOperand* opOperand : op.getDpsInputOperands()) {
    if (op.isScalar(opOperand)) {
      newInputBuffers.push_back(opOperand->get());
      continue;
    }
    FailureOr<Value> buffer = getBuffer(rewriter, opOperand->get(), options);
    if (failed(buffer)) return failure();
    newInputBuffers.push_back(*buffer);
  }

  // New output operands for the cloned op.
  SmallVector<Value> newOutputBuffers;
  for (OpResult opResult : op->getOpResults()) {
    OpOperand* opOperand = op.getDpsInitOperand(opResult.getResultNumber());
    FailureOr<Value> resultBuffer =
        getBuffer(rewriter, opOperand->get(), options);
    if (failed(resultBuffer)) return failure();
    newOutputBuffers.push_back(*resultBuffer);
  }

  // Merge input/output operands.
  SmallVector<Value> newOperands = newInputBuffers;
  newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(op);
  // Clone the op, but use the new operands. Move the existing block into the
  // new op. Since the new op does not have any tensor results, it does not
  // return anything.
  assert(op->getNumRegions() == 1 && "expected that op has 1 region");
  auto newOp = cast<DestinationStyleOpInterface>(cloneWithoutRegions(
      rewriter, op, /*resultTypes=*/TypeRange{}, newOperands));
  rewriter.inlineRegionBefore(op->getRegion(0), newOp->getRegion(0),
                              newOp->getRegion(0).begin());

  // Replace the results of the old op with the new output buffers.
  bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);

  return success();
}

/// Bufferization of linalg.generic. Replace with a new linalg.generic that
/// operates entirely on memrefs.
template <typename OpTy>
struct LinalgOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          LinalgOpInterface<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation* op, OpOperand& opOperand,
                              const bufferization::AnalysisState& state) const {
    // Operand is read if it is used in the computation.
    auto genericOp = cast<linalg::LinalgOp>(op);
    return genericOp.payloadUsesValueFromOperand(&opOperand);
  }

  bool bufferizesToMemoryWrite(
      Operation* op, OpOperand& opOperand,
      const bufferization::AnalysisState& state) const {
    // Operand is written to if it has an aliasing OpResult.
    auto bufferizableOp = cast<bufferization::BufferizableOpInterface>(op);
    return !bufferizableOp.getAliasingOpResults(opOperand, state).empty();
  }

  SmallVector<OpOperand*> getAliasingOpOperand(
      Operation* op, OpResult opResult,
      const bufferization::AnalysisState& state) const {
    auto genericOp = cast<DestinationStyleOpInterface>(op);

    // The i-th OpResult may alias with the i-th "out" tensor.
    return {genericOp.getDpsInitOperand(opResult.getResultNumber())};
  }

  SmallVector<OpResult> getAliasingOpResults(
      Operation* op, OpOperand& opOperand,
      const bufferization::AnalysisState& state) const {
    auto genericOp = cast<DestinationStyleOpInterface>(op);

    // The i-th "out" tensor may alias with the i-th OpResult.
    if (genericOp.isDpsInit(&opOperand))
      return {genericOp.getTiedOpResult(&opOperand)};
    return {};
  }

  bufferization::BufferRelation bufferRelation(
      Operation* op, OpResult opResult,
      const bufferization::AnalysisState& state) const {
    return bufferization::BufferRelation::Equivalent;
  }

  LogicalResult bufferize(
      Operation* op, RewriterBase& rewriter,
      const bufferization::BufferizationOptions& options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

/// Helper structure that iterates over all LinalgOps in `OpTys` and registers
/// the `BufferizableOpInterface` with each of them.
template <typename... Ops>
struct LinalgOpInterfaceHelper {
  static void registerOpInterface(MLIRContext* ctx) {
    (Ops::template attachInterface<LinalgOpInterface<Ops>>(*ctx), ...);
  }
};

}  // namespace

void registerBufferizableOpInterfaceExternalModels(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, DISCLinalgExtDialect* dialect) {
    LinalgOpInterfaceHelper<ConditionalGenericOp>::registerOpInterface(ctx);
  });
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

void YieldOp::print(OpAsmPrinter& p) {
  if (getNumOperands() > 0) p << ' ' << getOperands();
  p.printOptionalAttrDict((*this)->getAttrs());
  if (getNumOperands() > 0) p << " : " << getOperandTypes();
}

ParseResult YieldOp::parse(OpAsmParser& parser, OperationState& result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> opInfo;
  SmallVector<Type, 2> types;
  SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, result.operands));
}

// Check the operand number and types must match the element types of the
// LinalgOp interface's shaped operands.
static LogicalResult verifyYield(YieldOp op, LinalgOp linalgOp) {
  if (op.getNumOperands() != linalgOp.getNumDpsInits())
    return op.emitOpError("expected number of yield values (")
           << linalgOp.getNumDpsInits()
           << ") to match the number of operands of the enclosing "
           << "LinalgOp (" << op.getNumOperands() << ")";

  for (OpOperand& opOperand : op->getOpOperands()) {
    OpOperand* outputOperand =
        linalgOp.getDpsInitOperand(opOperand.getOperandNumber());
    Type elementType = getElementTypeOrSelf(outputOperand->get().getType());
    if (opOperand.get().getType() != elementType)
      return op.emitOpError("type of yield operand ")
             << (opOperand.getOperandNumber() + 1) << " ("
             << opOperand.get().getType() << ") doesn't match "
             << "the element type of the enclosing linalg.generic op ("
             << elementType << ")";
  }
  return success();
}

LogicalResult YieldOp::verify() {
  auto* parentOp = (*this)->getParentOp();
  if (parentOp->getNumRegions() != 1 || parentOp->getRegion(0).empty())
    return emitOpError("expected single non-empty parent region");

  if (auto linalgOp = dyn_cast<LinalgOp>(parentOp))
    return verifyYield(*this, linalgOp);

  return emitOpError("expected parent op with LinalgOp interface");
}

//===----------------------------------------------------------------------===//
// IndexOp
//===----------------------------------------------------------------------===//

LogicalResult IndexOp::verify() {
  auto linalgOp = dyn_cast<LinalgOp>((*this)->getParentOp());
  if (!linalgOp)
    return emitOpError("expected parent op with LinalgOp interface");
  if (linalgOp.getNumLoops() <= getDim())
    return emitOpError("expected dim (")
           << getDim() << ") to be lower than the number of loops ("
           << linalgOp.getNumLoops() << ") of the enclosing LinalgOp";
  return success();
}

namespace {
/// This is derived from mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp without any
/// changes.
struct FoldTensorCastOp : public OpInterfaceRewritePattern<LinalgExtOp> {
  using OpInterfaceRewritePattern<LinalgExtOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgExtOp op,
                                PatternRewriter& rewriter) const override {
    // If no operand comes from a tensor::CastOp and can be folded then fail.
    bool hasTensorCastOperand =
        llvm::any_of(op.getInputAndOutputOperands(), [&](OpOperand* opOperand) {
          if (opOperand->get().isa<BlockArgument>()) return false;
          auto castOp = opOperand->get().getDefiningOp<tensor::CastOp>();
          return castOp && canFoldIntoConsumerOp(castOp);
        });
    if (!hasTensorCastOperand) return failure();

    SmallVector<Type, 4> newResultTypes;
    newResultTypes.reserve(op->getNumResults());
    SmallVector<Value, 4> newOperands;
    newOperands.reserve(op->getNumOperands());
    // Inputs may fold.
    for (OpOperand* opOperand : op.getInputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      newOperands.push_back(canFoldIntoConsumerOp(tensorCastOp)
                                ? tensorCastOp.getSource()
                                : opOperand->get());
    }
    // Init tensors may fold, in which case the resultType must also change.
    for (OpOperand* opOperand : op.getOutputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      bool fold = canFoldIntoConsumerOp(tensorCastOp);
      newOperands.push_back(fold ? tensorCastOp.getOperand()
                                 : opOperand->get());
      newResultTypes.push_back(newOperands.back().getType());
    }
    // Add the other operands.
    for (OpOperand* opOperand : op.getNonInputOrOutputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      newOperands.push_back(canFoldIntoConsumerOp(tensorCastOp)
                                ? tensorCastOp.getSource()
                                : opOperand->get());
    }
    // Clone op.
    Operation* newOp =
        mlir::clone(rewriter, op.getOperation(), newResultTypes, newOperands);
    SmallVector<Value, 4> replacements;
    replacements.reserve(newOp->getNumResults());
    for (auto result : llvm::zip(op->getResults(), newOp->getResults())) {
      Value oldResult = std::get<0>(result);
      Value newResult = std::get<1>(result);
      if (newResult.getType() != oldResult.getType()) {
        replacements.push_back(rewriter.create<tensor::CastOp>(
            op->getLoc(), oldResult.getType(), newResult));
      } else {
        replacements.push_back(newResult);
      }
    }
    rewriter.replaceOp(op, replacements);

    return success();
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// LinalgExtDialect
//===----------------------------------------------------------------------===//

void DISCLinalgExtDialect::getCanonicalizationPatterns(
    RewritePatternSet& results) const {
  results.add<FoldTensorCastOp>(getContext());
}

}  // namespace disc_linalg_ext
}  // namespace disc_ral
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.cc.inc"
