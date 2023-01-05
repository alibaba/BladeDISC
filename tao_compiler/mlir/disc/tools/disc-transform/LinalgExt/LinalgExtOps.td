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

include "tensorflow/compiler/mlir/disc/tools/disc-transform/LinalgExt/LinalgExtBase.td"
include "tensorflow/compiler/mlir/disc/tools/disc-transform/LinalgExt/LinalgExtInterfaces.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
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

#endif  // DISC_LINALGEXT_OPS
