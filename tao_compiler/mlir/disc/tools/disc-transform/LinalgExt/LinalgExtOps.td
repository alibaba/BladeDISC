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

#ifndef DISC_LINALGEXT_OPS
#define DISC_LINALGEXT_OPS

include "external/org_tensorflow/tensorflow/compiler/mlir/disc/tools/disc-transform/LinalgExt/LinalgExtBase.td"
include "external/org_tensorflow/tensorflow/compiler/mlir/disc/tools/disc-transform/LinalgExt/LinalgExtEnums.td"
include "external/org_tensorflow/tensorflow/compiler/mlir/disc/tools/disc-transform/LinalgExt/LinalgExtInterfaces.td"

include "mlir/Dialect/Utils/StructuredOpsUtils.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class DISCLinalgExt_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<DISCLinalgExt_Dialect, mnemonic, traits>;

class DISCLinalgExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<DISCLinalgExt_Dialect, mnemonic, !listconcat(traits,
        [AttrSizedOperandSegments,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         DestinationStyleOpInterface, LinalgExtInterface
  ])> {
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  code extraLinalgExtOpClassDeclaration = "";
}

// We intentionally mark this op not folable since folding it may lose the
// specail semantics captured by this op. For example, it maybe be folded to
// a arith.constant op. It's only supposed to be folded in some special
// usecases (e.g. for packed weight).
def DISCLinalgExt_ConstantWrapperOp : Op<DISCLinalgExt_Dialect, "constant_wrapper", [
    Pure, AllTypesMatch<["value", "result"]>]>{
  let summary = "integer or floating point tensor constant";
  let description = [{
    The `constant` operation produces an SSA value equal to some integer or
    floating-point constant specified by an attribute. This is the way MLIR
    forms simple integer and floating point constants.

    Note that this op is not foldable, and is supposed to be used as a placeholder for
    later rewriting for RAL.

    Example:

    ```
    // Integer constant
    %1 = disc_linalg_ext.constant_wrapper dense<42> : tensor<i32>

    // Equivalent generic form
    %1 = "disc_linalg_ext.constant_wrapper"() {value = dense<42> : tensor<i32>} : () -> tensor<i32>
    ```
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $value";
}

def DISCLinalgExt_PaddingValuePlaceholderOp : Op<DISCLinalgExt_Dialect, "padding_value_placeholder", [
    Pure, AllTypesMatch<["value", "result"]>]>{
  let summary = "Used as a placeholder for padding value when vectorize or padding linalg ops";
  let description = [{
    Wrapper the padding value whne vectorize or pad a linalg op.

    These padding value may have different semantics.

    Example #1, we do not care the exact value of padding value. Padding is just used to get a
    static shape value which is suitable for hardware vector/tensor register.
    ```
    %padding_value = ... // padding value
    %0 = vector.transfer_read %arg0[%c0, %c0], %padding_value {in_bounds = [false, true]} : tensor<?x16xf32>, vector<6x16xf32>
    %1 = vector.transfer_read %arg1[...], %padding_value {in_bounds = [true]} : tensor<3072xf32>, vector<16xf32>
    %2 = vector.broadcast %1 : vector<16xf32> to vector<6x16xf32>
    %3 = arith.addf %0, %2 : vector<6x16xf32>
    %4 = arith.maxf %3, %cst_0 : vector<6x16xf32>
    %5 = vector.transfer_write %4, %arg0[%c0, %c0] {in_bounds = [false, true]} : vector<6x16xf32>, tensor<?x16xf32>
    ```

    Example #2: Padding value does matter, otherwise the result is wrong.
    ```
    %0 = vector.transfer_read %arg0[%c0, %c0], %padding_value {in_bounds = [false, true]} : tensor<?x?xf32>, vector<6x1xf32>
    %1 = vector.transfer_read %arg1[...], %padding_value {in_bounds = [true, false]} : tensor<?x?xf32>, vector<1x16xf32>
    %2 = vector.transfer_read %arg2[...], %padding_value {in_bounds = [false, false]} : tensor<?x?xf32>, vector<6x16xf32>
    %3 = vector.contract %0, %1, %2 : vector<6x16xf32>
    %4 = vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [false, false]} : vector<6x16xf32>, tensor<?x?xf32>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value,
                   DefaultValuedAttr<DISCLinalgExt_PaddingValueMode,
                    "::mlir::disc_ral::disc_linalg_ext::PaddingValueModeEnum::kAny">:$mode);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict `padding_mode` `(` $mode `)` `,` `value` `(` $value `)`";
}

def DISCLinalgExt_MultiLevelPackOp : DISCLinalgExt_Op<"multi_level_pack", [
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
]>{
  let summary = "multi-level pack operation";
  let description = [{
    The multi-level pack operation converts an `input` into a tiled and packed layout.
    The level of tiling for each dimension is obtained from `tile_levels`. If the tiling
    level for one dimension is zero, then it means we do not need to tile along that
    dimension. If a dimension is tiled with N levels, then there will be N tiling size in
    `tile_sizes` filed for that dimension. The tiling size for one dimension starts from
    the outter most level (biggest tiling size). This op will also permute the tiled dimensions
    according to the `permutation` fields.

    Example KC_to_KCck:

    ```mlir
    disc_linalg_ext.multi_level_pack %arg0 with tile_levels = [1, 1]
      tile_sizes = [32, 8] permutation = [0, 2, 3, 1]
      into %arg1 : (tensor<128x256xf32> tensor<4x32x8x32xf32>)
    ```

    Example NC_to_NCnc:

    ```mlir
    disc_linalg_ext.multi_level_pack %arg0 with tile_levels = [1, 1]
      tile_sizes = [8, 32] permutation = [0, 2, 1, 3]
      into %arg1 : (tensor<128x256xf32> tensor<16x8x8x32xf32>)
    ```
    Example KC_to_CKkc

    ```mlir
    disc_linalg_ext.multi_level_pack %arg0 with tile_levels = [1, 1]
      tile_sizes = [32, 8] permutation = [2, 0, 1, 3]
      into %arg1 : (tensor<128x256xf32> tensor<32x4x32x8xf32>)
    ```

    Example NC_to_NCnc with padding:

    ```mlir
    disc_linalg_ext.multi_level_pack %arg0 with padding_value(%pad : f32)
      tile_levels = [1, 1] tile_sizes = [8, 2] permutation = [0, 2, 1, 3]
      into %arg1 : (tensor<13x15xf32> tensor<2x8x8x2xf32>)
    ```

    Example KC_to_KCc{1}k{1}k{2}c{2} with padding:

    ```mlir
    disc_linalg_ext.multi_level_pack %arg0 with padding_value(%pad : f32)
      tile_levels = [2, 2] tile_sizes = [512, 1, 256, 16] permutation = [0, 3, 4, 1, 2, 5]
      into %arg1 : (tensor<1023x1023xf32> tensor<2x4x16x512x1x16xf32>)
    ```
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
    Variadic<AnyShaped>:$outputs,
    I64ArrayAttr:$tile_levels,
    I64ArrayAttr:$tile_sizes,
    I64ArrayAttr:$permutation,
    Optional<AnyType>:$padding_value);

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    attr-dict
    $inputs `with`
    (`padding_value` `(` $padding_value^ `:` type($padding_value) `)`)?
    `tile_levels` `=` $tile_levels
    `tile_sizes` `=` $tile_sizes
    `permutation` `=` $permutation
    `into` $outputs `:` `(` type($inputs) type($outputs) `)`
     (`->` type($results)^)?
  }];

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$output,
      "ArrayRef<int64_t>":$tileLevels,
      "ArrayRef<int64_t>":$tileSizes,
      CArg<"ArrayRef<int64_t>", "{}">:$permutation,
      CArg<"Optional<Value>", "llvm::None">:$paddingValue
      )>
  ];

  let hasFolder = 1;

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{

    // Return the output operand.
    Value getOutput() {
      return getOutputOperand(0)->get();
    }

    // Return the input operand.
    Value getInput() {
      return getInputOperand(0)->get();
    }

    // Return the output rank.
    int64_t getOutputRank() {
      return  getOutputType().getRank();
    }

    // Return the output type.
    ShapedType getOutputType() {
      return getOutput().getType();
    }

    // Return the input type.
    ShapedType getInputType() {
      return getInput().getType();
    }

    // Return the output shape.
    ArrayRef<int64_t> getOutputShape() {
      return getOutputType().getShape();
    }

    // Return the input shape.
    ArrayRef<int64_t> getInputShape() {
      return getInputType().getShape();
    }

    // Return the element type.
    Type getElementType() {
      return getInputType().getElementType();
    }

    // Return the rank of the input operand.
    int64_t getInputRank() {
      return getInputType().getRank();
    }

    // Converts I64Attr to smallvector
    static SmallVector<int64_t> convertI64ArrayAttrToVec(ArrayAttr attrs) {
      SmallVector<int64_t> vs;
      for (auto attr : attrs) vs.push_back(attr.cast<IntegerAttr>().getInt());
      return vs;
    }

    // Returns tileLevels as a small vector
    SmallVector<int64_t> getTileLevelsVec() {
      return convertI64ArrayAttrToVec(getTileLevels());
    }

    // Returns tileSizes as a small vector
    SmallVector<int64_t> getTileSizesVec() {
      return convertI64ArrayAttrToVec(getTileSizes());
    }

    // Returns permutation as a small vector
    SmallVector<int64_t> getPermutationVec() {
      return convertI64ArrayAttrToVec(getPermutation());
    }

    static int64_t getExpectedResultRank(ArrayRef<int64_t> vs) {
      int64_t rank = 0;
      for (int64_t v : vs) rank += 1 + v;
      return rank;
    }

    // Returns the expected rank of the tiled result
    int64_t getExpectedResultRank() {
      auto vs = getTileLevelsVec();
      return getExpectedResultRank(vs);
    }

    // Returns the map: logical dim of output -> input dim
    // Here the logical dim is the dim before transpose.
    DenseMap<int, int> getOutputLogicalDimToInputDimMapping(
        ArrayRef<int64_t> tileLevelsVec, ArrayRef<int64_t> tileSizesVec) {
      int logicalIdx = 0;
      DenseMap<int, int> dst2src;
      for (int i = 0; i < tileLevelsVec.size(); ++i) {
        for (int j = 0; j <= tileLevelsVec[i]; ++j) {
          dst2src[logicalIdx++] = i;
        }
      }
      return dst2src;
    }

    // Returns the map: logical dim of output -> tile size for each dim
    // Here the logical dim is the dim before transpose.
    DenseMap<int, int> getOutputLogicalDimToTileSizeMapping(
        ArrayRef<int64_t> tileLevelsVec, ArrayRef<int64_t> tileSizesVec) {
      int logicalIdx = 0;
      int tileSizeIdx = 0;
      DenseMap<int, int> logicalDim2TileSize;
      for (int i = 0; i < tileLevelsVec.size(); ++i) {
        for (int j = 0; j < tileLevelsVec[i]; ++j) {
          logicalDim2TileSize[logicalIdx++] = tileSizesVec[tileSizeIdx++];
        }
        logicalDim2TileSize[logicalIdx++] = 1;
      }
      return logicalDim2TileSize;
    }

    // Returns the inner most dims
    DenseSet<int> getOutputInnerMostDims(
        ArrayRef<int64_t> tileLevelsVec, ArrayRef<int64_t> tileSizesVec) {
      int logicalIdx = 0;
      DenseSet<int> innerMostDims;
      for (int i = 0; i < tileLevelsVec.size(); ++i) {
        logicalIdx += tileLevelsVec[i];
        innerMostDims.insert(logicalIdx++);
      }
      return innerMostDims;
    }

    // Method to get the shape of the result as `SmallVector<OpFoldResult>`.
    // This is a static method to allow getting the shape of the destination
    // expected while creating a `pack` op.
    static SmallVector<OpFoldResult> getResultShape(OpBuilder &builder,
        Location loc, ArrayRef<OpFoldResult> sourceDims,
        ArrayRef<int64_t> tileLevels, ArrayRef<int64_t> tileSizes,
        ArrayRef<int64_t> permutation = {});
    // Method to return the shape of the result as `SmallVector<OpFoldResult>`.
    SmallVector<OpFoldResult> getResultShape(OpBuilder &builder);

    // Method to get the `ShapedType` of the result. This is a static method
    // to allow getting the type of the destination while creating the `pack`
    // op.
    static ShapedType getPackedType(ShapedType sourceType,
        ArrayRef<int64_t> tileLevels, ArrayRef<int64_t> tileSizes,
        ArrayRef<int64_t> permutation = {});

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      std::pair<unsigned, unsigned> outputsIndexAndLength =
        getODSOperandIndexAndLength(1);
      return std::make_pair<int64_t, int64_t>(
          outputsIndexAndLength.first,
          outputsIndexAndLength.first + outputsIndexAndLength.second);
    }
  }];
}

// Copied from linalg dialect
def DISCLinalgExt_YieldOp : DISCLinalgExt_BaseOp<"yield", [Pure, ReturnLike, Terminator]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "disc_linalg_ext yield operation";
  let description = [{
    `disc_linalg_ext.yield` is a special terminator operation for blocks inside regions
    in `disc_linalg_ext` condition_generic ops. It returns values to the immediately enclosing
    `disc_linalg_ext` generic op.

    Example:

    ```mlir
    disc_linalg_ext.yield %f0, %f1 : f32, f32
    ```
  }];
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// Copied from linalg dialect
def DISCLinalgExt_IndexOp : DISCLinalgExt_BaseOp<"index", [Pure]>,
    Arguments<(ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$dim)>,
    Results<(outs Index:$result)> {
  let summary = "disc_linalg_ext index operation";
  let description = [{
    The `disc_linalg_ext.index` operation returns the iteration index of the immediately
    enclosing disc_linalg_ext structured operation for the iteration dimension `dim`. The
    `dim` attribute specifies the position of the accessed dimension in the
    indexing map domain.

    Example:

    ```mlir
    #map = affine_map<(i, j) -> (i, j)>
    disc_linalg_ext.condition_generic {indexing_maps = [#map, #map],
                    iterator_types = ["parallel", "parallel"]}
      outs(%I, %J : memref<?x?xindex>, memref<?x?xindex>) {
      ^bb0(%arg0 : index, %arg1 : index):
      // Access the outer iteration dimension i
      %i = disc_linalg_ext.index 0 : index
      // Access the inner iteration dimension j
      %j = disc_linalg_ext.index 1 : index
      disc_linalg_ext.yield %i, %j : index, index
    }
    ```

    This may lower to IR resembling:

    ```mlir
    %0 = dim %I, %c0 : memref<?x?xindex>
    %1 = dim %I, %c1 : memref<?x?xindex>
    scf.for %i = %c0 to %0 step %c1 {
      scf.for %j = %c0 to %1 step %c1 {
        store %i, %I[%i, %j] : memref<?x?xindex>
        store %j, %J[%i, %j] : memref<?x?xindex>
      }
    }
    ```
  }];

  let assemblyFormat = [{ $dim attr-dict `:` type($result) }];
  let hasVerifier = 1;
}

// Copied from linalg dialect
// Base Tablegen class for Linalg ops.
// Linalg ops that correspond to library calls operate on ShapedType as their
// first operands. These may be optionally followed by non-view operands
// depending on the specific Linalg op.
class DISCLinalgExt_StructuredOp<string mnemonic, list<Trait> props>
  : DISCLinalgExt_BaseOp<mnemonic, !listconcat([
       SingleBlockImplicitTerminator<"YieldOp">,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DestinationStyleOpInterface,
       LinalgStructuredInterface,
       RegionBranchOpInterface,
       ReifyRankedShapedTypeOpInterface], props)> {
  code structuredOpsBaseDecls = [{
    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      return !this->getBody()->getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }

    void getSuccessorRegions(
        Optional<unsigned> index, ArrayRef<Attribute> operands,
        SmallVectorImpl<RegionSuccessor> &regions) {
      // Op has a region, but conceptually the control flow does not enter the
      // region.
    }
  }];
}

def DISCLinalgExt_ConditionalGenericOp : DISCLinalgExt_StructuredOp<"conditional_generic", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    AttrSizedOperandSegments]> {
  let description = [{
    Conditional Generic Linalg op form where the key properties of the computation are
    specified as attributes. In pretty form, a `disc_linalg_ext.conditional_generic` op
    is written as:

      ```mlir
      disc_linalg_ext.conditional_generic #trait_attribute
          ins(%pred, %A, %B : i1, memref<?x?xf32>, memref<?x?xf32>)
          outs(%C : memref<?x?xf32>)
          attrs = {other-optional-attributes}
          {region}
      ```

    Where #trait_attributes is an alias of a dictionary attribute containing:
      - doc [optional]: a documentation string
      - indexing_maps: a list of AffineMapAttr, one AffineMapAttr per each input
        and output view. Such AffineMapAttr specifies the mapping between the
        loops and the indexing within each view.
      - library_call [optional]: a StringAttr containing the name of an
        external library function that the linalg.generic operation maps to.
        The external library is assumed to be dynamically linked and no strong
        compile-time guarantees are provided. In the absence of such a library
        call, linalg.generic will always lower to loops.
      - iterator_types: an ArrayAttr specifying the type of the enclosing loops.
        Each element of the list represents and iterator of one of the following
        types:
          parallel, reduction, window

    Example:
    Defining a elemment-wise epilogue ops
      ```mlir
      disc_linalg_ext.conditional_generic #trait_attribute
        ins(%pred, %A, %B : i1, memref<?x?xf32>, memref<?x?xf32>)
        outs(%C : memref<?x?xf32>)
        {other-optional-attributes} {
        ^bb0(%a: f32, %b: f32, %c: f32) :
          %d = arith.mulf %a, %b: f32
          linalg.yield %d : f32
      }
      ```

    This may lower to:
    ```mlir
    scf.if (%pred) {
      scf.for %m = %c0 to %M step %c1 {
        scf.for %n = %c0 to %N step %c1 {
            %a = load %A[%m, %n] : memref<?x?xf32>
            %b = load %B[%m, %n] : memref<?x?xf32>
            %d = arith.mulf %a, %b: f32
            store %d, %C[%m, %n] : memref<?x?x?xf32>
          }
        }
      }
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       AffineMapArrayAttr:$indexing_maps,
                       IteratorTypeArrayAttr:$iterator_types,
                       OptionalAttr<StrAttr>:$doc,
                       OptionalAttr<StrAttr>:$library_call);
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
  let regions = (region AnyRegion:$region);

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
      "ValueRange":$outputs, "ArrayAttr":$indexingMaps,
      "ArrayAttr":$iteratorTypes, "StringAttr":$doc,
      "StringAttr":$libraryCall,
      "function_ref<void(OpBuilder &, Location, ValueRange)>",
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
      "ValueRange":$outputs, "ArrayRef<AffineMap>":$indexingMaps,
      "ArrayRef<utils::IteratorType>":$iteratorTypes, "StringRef":$doc,
      "StringRef":$libraryCall,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "nullptr">,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputBuffers,
      "ArrayRef<AffineMap>":$indexingMaps, "ArrayRef<utils::IteratorType>":$iteratorTypes,
      "StringRef":$doc, "StringRef":$libraryCall,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "nullptr">,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
      "ValueRange":$outputs, "ArrayRef<AffineMap>":$indexingMaps,
      "ArrayRef<utils::IteratorType>":$iteratorTypes,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "nullptr">,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputBuffers,
      "ArrayRef<AffineMap>":$indexingMaps, "ArrayRef<utils::IteratorType>":$iteratorTypes,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "nullptr">,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    SmallVector<StringRef, 8> linalgTraitAttrNames() {
      return SmallVector<StringRef, 8>{
        getDocAttrName(),
        getIndexingMapsAttrName(), getLibraryCallAttrName(),
        getIteratorTypesAttrName(),
      };
    }
    std::string getLibraryCallName() {
      return getLibraryCall() ?
        getLibraryCall()->str() : "op_has_no_registered_library_name";
    }

    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return nullptr;
    }
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      int64_t getNumOperands = this->getNumOperands();
      return {getNumOperands - getOutputs().size(), getNumOperands};
    }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif  // DISC_LINALGEXT_OPS
