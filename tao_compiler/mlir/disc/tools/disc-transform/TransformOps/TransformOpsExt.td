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

#ifndef DISC_TRANSFORM_OPS_EXT
#define DISC_TRANSFORM_OPS_EXT

include "mlir/Dialect/GPU/TransformOps/GPUDeviceMappingAttr.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Vector/Transforms/VectorTransformsBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/// Some of the following ops are borrowed from IREE project.

def DISCBufferizeOp : Op<Transform_Dialect, "disc.bufferize",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Target the whole module op and call upstream comprehensive bufferize with extra DISC hooks.

    Return modes:
    =============
    This operation calls the upstream one-shot bufferization pass with extra
    registered patterns for DISC.

    The pass is ran on all the ModuleOp nested under the top-level op on which
    the transform dialect interpreter pass is applied.

    If any of the pass on any of the ModuleOp fails, the transformation
    definitely fails. Otherwise the transformation succeeds.

    No handles are consumed or produced.
  }];

  let arguments = (ins PDL_Operation:$target, UnitAttr:$target_gpu);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "attr-dict $target";
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target,
                   CArg<"bool", "false">:$targetGpu)>
  ];
}

def ApplyPatternsOp : Op<Transform_Dialect, "disc.apply_patterns",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Greedily applies patterns as specified by its attributes.

    Must be applied to an op with trait IsolatedFromAbove since the
    GreedyPatternRewriter asserts those.

    Returns the IsolatedFromAbove op whose content it has modified for better
    chaining APIs.

    The following additive attributes can be set, they add patterns in an
    unspecified order:
      - canonicalization: adds all the canonicalization patterns of all
      registered dialects and ops.

    Return modes:
    =============
    This operation applies a number of patterns to rewrite vector IR into
    distributed warp form. To apply these patterns, this operation must target
    an operation that is isolated from above, otherwise the transform definitely
    fails.

    If the pattern application fails, or if the underlying listener fails to
    capture op handles, the transformation definitely fails.

    Otherwise the transformation is successful and no result is returned.
  }];

  let arguments = (ins PDL_Operation:$target,
                       UnitAttr:$canonicalization);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let builders = [
    // TODO: Some bitvector to scale better than n-bools.
    OpBuilder<(ins "Value":$target, "bool":$canonicalization)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def FoldProducerExtractSliceOp : Op<Transform_Dialect, "disc.fold_producer_extract_slice",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Given a tensor.ExtractSliceOp, greedily fold its producers if they are also tensor.ExtractSliceOp.

    Returns the folded new tensor.ExtractSliceOp.

    The following additive attributes can be set:
      - max_repeat_num: fold at most `max_repeat_num` times.

    Return modes:
    =============
    This operation try to fold two tensor.ExtractSliceOp with def-use relationship at most
    `max_repeat_num` times.
  }];

  let arguments = (ins PDL_Operation:$target,
                       I64Attr:$max_repeat_num);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target, "int64_t":$max_repeat_num)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def CacheReadOp : Op<Transform_Dialect, "disc.cache_read",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Given a target `tensor.ExtractSliceOp` named 's0', first tile and pack source of the `s0`,
    and then replace `s0` with a new `tensor.ExtractSliceOp` named `s1`. `s1` will read from
    the packed and tiled source to increase the cache hit ratio. the transformed source tensor
    will be placed right before the `anthor` op.

    Returns the new tensor.ExtractSliceOp.

    The following attributes need to be set:
      - tileLevels: please see the document of `disc_ral::disc_linalg_ext::MultiLevelPackOp`
      - tileSizes: please see the document of `disc_ral::disc_linalg_ext::MultiLevelPackOp`
      - permutation: please see the document of `disc_ral::disc_linalg_ext::MultiLevelPackOp`
      - padded (optional): indicates that the slice op is padded and target tile is the padded
        version.

    Example #0:
     convert from:
     ```
      for (i, j) {
        %0 = tensor.extract_slice %arg0[i, j][32, 32][1, 1] : tensor<256x256xf32> to tensor<32x32xf32>
        use(%0)
      }
     ```
     to:
     ```
      %0 = tensor.empty() : tensor<8x8x32x32xf32>
      %packed = disc_linalg_ext.multi_level_pack %arg0 with
          tile_levels = [1, 1] tile_sizes = [32, 32] permutation = [0, 3, 1, 3] into %0
          (tensor<256x256xf32> tensor<8x8x32x32xf32>) -> tensor<8x8x32x32xf32>
      for (i, j) {
        i', j' = f(i, j) // index mapping
        %0 = tensor.extract_slice %packed[i', j', 0, 0][1, 1, 32, 32][1, 1, 1, 1] : tensor<8x8x32x32xf32> to tensor<32x32xf32>
        use(%0)
      }
     ```

    Example #1:
     convert from:
     ```
      #map = affine_map<(d0)[s0] -> (-d0 + s0, 32)>
      %cst0 = arith.constant 0.000000e+00 : f32
      for (i, j) {
        %s0 = affine.min #map(%i)[%d0]
        %s1 = affine.min #map(%j)[%d1]
        %0 = tensor.extract_slice %arg0[i, j][%s0, %s1][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %1 = tensor.pad %0 low[0, 0] high[32, 32] {
        ^bb0(%arg12: index, %arg13: index):
          tensor.yield %cst0 : f32
        } : tensor<?x?xf32> to tensor<32x32xf32>
        use(%1)
      }
     ```
     to:
     ```
      %0 = tensor.empty() : tensor<?x?x32x32xf32>
      %packed = disc_linalg_ext.multi_level_pack %arg0 with
          %cst0 tile_levels = [1, 1] tile_sizes = [32, 32] permutation = [0, 3, 1, 3] into %0
          (tensor<?x?xf32> tensor<?x?x32x32xf32>) -> tensor<?x?x32x32xf32>
      for (i, j) {
        i', j' = f(i, j) // index mapping
        %0 = tensor.extract_slice %packed[i', j', 0, 0][1, 1, 32, 32][1, 1, 1, 1] : tensor<?x?x32x32xf32> to tensor<32x32xf32>
        use(%0)
      }
     ```
  }];

  let arguments = (ins PDL_Operation:$target,
                       PDL_Operation:$anchor,
                       I64ArrayAttr:$tile_levels,
                       I64ArrayAttr:$tile_sizes,
                       I64ArrayAttr:$permutation,
                       UnitAttr:$padded);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = [{
    attr-dict
    $target `at` $anchor `with`
    `tile_levels` `=` $tile_levels
    `tile_sizes` `=` $tile_sizes
    `permutation` `=` $permutation
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target, "Value":$anchor,
                   "ArrayRef<int64_t>":$tileLevels,
                   "ArrayRef<int64_t>":$tileSizes,
                   CArg<"bool", "false">:$padded,
                   CArg<"ArrayRef<int64_t>", "{}">:$permutation)>
  ];
}

def LowerMultiLevelPackToLoopOp : Op<Transform_Dialect, "disc.lower_multi_level_pack_to_loop",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Replace a MultiLevelPackOp to its loop level equivalent.

    Returns a handle to the outter most loop.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def InlineReductionInitializerOp : Op<Transform_Dialect, "disc.inline_reduction_initializer",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Given a buffer popluated by a reduction loop named `l0`, inline the initializer of the buffer
    into the loop `l0`.

    Returns the new xfer_read op.

    Example:
     convert from:
     ```
      linalg.fill ins(%cst : f32) outs(%0 : memref<?x?xf32>)
      scf.for %iv = %init to %stop step %step {
        %1 = memref.subview %0[...] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: 0>>
        %2 = vector.transfer_read %1[%c0, %c0], %cst : memref<?x?xf32, strided<[?, 1], offset: 0>>, vector<1x16xf32>
        use(%2)
      }
     ```
     to:
     ```
      %0 = memref.alloca() : memref<1x16xf32>
      linalg.fill ins(%cst : f32) outs(%0 : memref<1x16xf32>)
      scf.for %iv = %init to %stop step %step {
        %1 = memref.subview %0[...] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: 0>>
        %first_step = arith.cmpi eq %iv, %init : i1
        %2 = scf.if %first_step -> memref<?x?xf32, strided<[?, 1], offset: ?>> {
          %cast = memref.cast %0 :  memref<1x16xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          scf.yield %cast : memref<?x?xf32, strided<[?, 1], offset: ?>>
        } else {
          %cast = memref.cast %1 :  memref<?x?xf32, strided<[?, 1], offset: 0>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          scf.yield %cast : memref<?x?xf32, strided<[?, 1], offset: ?>>
        }
        %3 = vector.transfer_read %2[%c0, %c0], %cst : memref<?x?xf32, strided<[?, 1], offset: 0>>, vector<1x16xf32>
        use(%3)
      }
     ```
  }];

  let arguments = (ins PDL_Operation:$initializer,
                       PDL_Operation:$loop,
                       PDL_Operation:$reader);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = [{
    attr-dict $initializer `for` `reader` $reader  `into` `loop` $loop
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";
}

def DecomposeVectorsOp : Op<Transform_Dialect, "disc.decompose_vectors",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Decompose vector ops and related vector transform_read/write ops into fine-grained-size vector ops.

    Without this decomposition, llvm backend will mis-allocate vector register for some vector ops
    with specific shape (e.g. 8x12xf32 failed to map to hardware register). This is a workaround for this.

    Examples:
     convert
      ```
       %0 = vector.transfer_read %arg0[..., %c0] : vector<8xf32>
       %1 = vector.transfer_read %arg1[..., %c0] : vector<8xf32>
       %2 = vector.transfer_read %arg2[..., %c0] : vector<8xf32>
       %3 = vector.fma %0, %1, %2 : vector<8xf32>
       vector.transfer_write %3, %arg2[..., %c0] : vector<8xf32>
      ```
     to:
      ```
       %0_0 = vector.transfer_read %arg0[..., %c0] : vector<4xf32>
       %0_1 = vector.transfer_read %arg0[..., %c4] : vector<4xf32>
       %1_0 = vector.transfer_read %arg1[..., %c0] : vector<4xf32>
       %1_1 = vector.transfer_read %arg1[..., %c4] : vector<4xf32>
       %2_0 = vector.transfer_read %arg2[..., %c0] : vector<4xf32>
       %2_1 = vector.transfer_read %arg2[..., %c4] : vector<4xf32>
       %3_0 = vector.fma %0_0, %1_0, %2_0 : vector<4xf32>
       %3_1 = vector.fma %0_1, %1_1, %2_1 : vector<4xf32>
       vector.transfer_write %3_0, %arg2[..., %c0] : vector<4xf32>
       vector.transfer_write %3_1, %arg2[..., %c4] : vector<4xf32>
      ```
  }];

  let arguments = (ins PDL_Operation:$target,
                       I64Attr:$vector_size);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target, "int64_t":$vector_size)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def LinalgFuseOperandOp : Op<Transform_Dialect, "disc.linalg.fuse_operand",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Given a linalg op, try to fuse its specified operand linalg op into it.

    Returns the fused linalg.

    Example use:
     convert
    ```
      %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<?x768xf32>, tensor<?x768xf32>) outs(%2 : tensor<?x768xf32>) attrs =  {disc.device = "cpu", disc.transform.name = "subtract"} {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.subf %in, %in_0 : f32
        linalg.yield %6 : f32
      } -> tensor<?x768xf32>
      %6 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %4 : tensor<?x768xf32>, tensor<?x768xf32>) outs(%5 : tensor<?x768xf32>) attrs =  {disc.device = "cpu", disc.transform.name = "multiply"} {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.mulf %in, %in_0 : f32
        linalg.yield %6 : f32
      } -> tensor<?x768xf32>
    ```
     to
    ```
     %6 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%0, %1, %3 : tensor<?x768xf32>, tensor<?x768xf32>, tensor<?x768xf32>) outs(%5 : tensor<?x768xf32>) attrs =  {disc.device = "cpu", disc.transform.name = "multiply"} {
     ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
       %6 = arith.subf %in, %in_0 : f32
       %7 = arith.mulf %6, %in_1 : f32
       linalg.yield %7 : f32
     } -> tensor<?x768xf32>
    ```

  }];

  let arguments = (ins PDL_Operation:$target,
                       I64Attr:$operand_idx);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target, "int64_t":$operand_idx)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def LinalgFuseProducersOp : Op<Transform_Dialect, "disc.linalg.fuse_producers",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Given a linalg op, try to recursively and greedily fuse its operands if they are
    produced by the given op set.

    Returns the fused linalg.

    Example use. Suppose we have the following payload IR:
    ```
      %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<?x768xf32>, tensor<?x768xf32>) outs(%2 : tensor<?x768xf32>) attrs =  {disc.device = "cpu", disc.transform.name = "subtract"} {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.subf %in, %in_0 : f32
        linalg.yield %6 : f32
      } -> tensor<?x768xf32>
      %6 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %4 : tensor<?x768xf32>, tensor<?x768xf32>) outs(%5 : tensor<?x768xf32>) attrs =  {disc.device = "cpu", disc.transform.name = "multiply"} {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.mulf %in, %in_0 : f32
        linalg.yield %6 : f32
      } -> tensor<?x768xf32>
      %8 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6, %6 : tensor<?x768xf32>, tensor<?x768xf32>) outs(%7 : tensor<?x768xf32>) attrs =  {disc.device = "cpu", disc.transform.name = "add"} {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.addf %in, %in_0 : f32
        linalg.yield %6 : f32
      } -> tensor<?x768xf32>
    ```
     and following transform IR:
    ```
     %0 = transform.structured.match attributes {disc.transform.name = "subtract"} in %arg0
     %1 = transform.structured.match attributes {disc.transform.name = "multiply"} in %arg0
     %2 = transform.structured.match attributes {disc.transform.name = "add"} in %arg0
     %3 = transform.disc.fuse_producers %0, %1 into %2
    ```
     finally we'll get the following transformed IR:
    ```
     %6 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%0, %1, %3 : tensor<?x768xf32>, tensor<?x768xf32>, tensor<?x768xf32>) outs(%5 : tensor<?x768xf32>) attrs =  {disc.device = "cpu", disc.transform.name = "multiply"} {
     ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
       %6 = arith.subf %in, %in_0 : f32
       %7 = arith.mulf %6, %in_1 : f32
       %8 = arith.addf %7, %7 : f32
       linalg.yield %8 : f32
     } -> tensor<?x768xf32>
    ```

  }];

  let arguments = (ins PDL_Operation:$target,
                       Variadic<PDL_Operation>:$producers);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let assemblyFormat = [{
    attr-dict
    $producers `into` $target
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";
}

def ReplaceConstPaddingValueOp : Op<Transform_Dialect, "disc.replace_const_padding_value",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Given a tensor.PadOp, replace its padding content using `padding_value_placeholder`

    Returns the updated tensor.PadOp op.
  }];

  let arguments = (ins PDL_Operation:$target,
                       StrAttr:$mode);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict `mode` `(` $mode `)`";
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target, "StringRef":$mode)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ConvertPaddingPlaceholderToConstOp : Op<Transform_Dialect, "disc.convert_padding_placeholder_to_const",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Given a `padding_value_placeholder` op, convert to arith.const (ignore its padding mode).

    Returns the result const op.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::disc_ral::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def LinalgEagerlyBackwardInitTensorOp : Op<Transform_Dialect, "disc.linalg.eagerly_backward_init_tensor",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Given a linalg op `A`, eagerly backward its init tensor to its producers. The function of such primitive
    is to reuse underline buffer.

    Eamples:
    ```
      %0 = linalg.generic { indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]}
        ins(%arg0, %arg1) outs(%arg2)
      %1 = linalg.generic { indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]}
        ins(%0, %0) outs(%arg3)
    ```
    After conversion:
    ```
      %0 = linalg.generic { indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]}
        ins(%arg0, %arg1) outs(%arg3) // Note here use the final output buffer directly
      %1 = linalg.generic { indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]}
        ins() outs(%0) // Note here: we promote %0 to be an output operand (thus can be read + write).
    ```

    Returns the result const op.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::disc_ral::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// mainly copied from mlir community, adding support to fuse through
// iter_args for normal scf.for ops.

def DISCFuseIntoContainingOp :
    Op<Transform_Dialect, "disc.fuse_into_containing_op",
      [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{Fuse a producer into a containing operation.}];

  let summary = [{
    Fuses the `producer_op` into the `containing_op`.
    Returns a handle to the fused ops.

    The producer is typically a slice of a tileable op (i.e., implements
    TilingInterface). In that case, this transform computes the accessed
    producer slice inside of the containing op ("tile and fuse"). Otherwise,
    the entire producer is cloned inside the containing op ("clone and fuse").

    The containing op handle must be associated with exactly one payload op. The
    producer op handle may be associated with multiple payload ops. This
    transform fuses producers one-by-one, always picking an unspecified producer
    that has at least one use inside the containing op among the
    producers.

    Note: If a producer has multiple uses inside the containing op, it is
    currently tiled and/or cloned multiple times into the containing op.
    TODO: Reuse already fused OpResults instead of tiling/cloning a second time
    when possible. Fuse producers according to a topological sorting to achieve
    the largest amount of reuse.

    #### Return modes

    If at least one producer could not be fused, this operation fails silently.
    This is the case when tiling fails or when no producer op could be found
    among the remaining producers that has at least one use within the
    containing op. I.e., "producers" that are not consumed within the containing
    op are rejected by this operation.

    This operation reads and frees the producer handle.
    This operation reads the containing op handle.
  }];

  let arguments = (ins PDL_Operation:$producer_op,
                       PDL_Operation:$containing_op);
  let results = (outs PDL_Operation:$fused_op);
  let assemblyFormat = "$producer_op `into` $containing_op attr-dict";
  let cppNamespace = "::mlir::disc_ral::transform_dialect";
}

def ReductionOutputFuseOp : Op<Transform_Dialect, "disc.reduction_output_fuse",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Given a linalg op, output fuse it into the reduction loop.

    Returns the fused linalg op and the new loop op.

    Example:
     convert from:
     ```
      %1 = scf.for %arg1 = %c0 to %c768 step %c512 iter_args(%arg0 = %0) -> (tensor<?x3072xf32>) {
        ...
        %t0 = tensor.extract_slice %arg0[0, 0] [...] [1, 1] : tensor<?x3072xf32> to tensor<?x3072xf32>
        %t1 = linalg.matmul {disc.transform.name = "dot_general"} ins(...) outs(%t0 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
        %t2 = tensor.insert_slice %t1 into %arg0[0, 0] [...] [1, 1] : tensor<?x3072xf32> into tensor<?x3072xf32>
        scf.yield %t2 : tensor<?x3072xf32>
      }
      %3 = tensor.extract_slice %2[...] [...] [1, 1] : tensor<?x3072xf32> to tensor<?x3072xf32>
      %4 = linalg.generic {indexing_maps = [#map5, #map6, #map7, #map6], iterator_types = ["parallel", "parallel"]} ins(..., %1, ...) outs(%3 : tensor<?x3072xf32>) attrs =  {disc.device = "cpu", disc.trasform.name = "maximum"} {
      ^bb0(%in: f32, %in_5: f32, %in_6: f32, %out: f32):
        %12 = arith.addf %in_5, %in_6 : f32
        %13 = arith.maxf %in, %12 : f32
        linalg.yield %13 : f32
      } -> tensor<?x3072xf32>
      use(%4)
     ```
     to:
     ```
      %3 = tensor.extract_slice %2[...] [...] [1, 1] : tensor<?x3072xf32> to tensor<?x3072xf32>
      %1, %4 = scf.for %arg1 = %c0 to %c768 step %c512 iter_args(%arg0 = %0, %arg1 = %3) -> (tensor<?x3072xf32>, tensor<?x3072xf32>) {
        ...
        %t0 = tensor.extract_slice %arg0[0, 0] [...] [1, 1] : tensor<?x3072xf32> to tensor<?x3072xf32>
        %t1 = linalg.matmul {disc.transform.name = "dot_general"} ins(...) outs(%t0 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
        %t2 = tensor.insert_slice %t1 into %arg0[0, 0] [...] [1, 1] : tensor<?x3072xf32> into tensor<?x3072xf32>
        %t3 = tensor.extract_slice %arg1[0, 0] [...] [1, 1] : tensor<?x3072xf32> to tensor<?x3072xf32>
        %pred = ... // is last iteration?
        %t4 = disc_linalg_ext.conditional_generic {indexing_maps = [#map5, #map6, #map7, #map6], iterator_types = ["parallel", "parallel"]} ins(%pred, ..., %t2, ...) outs(%t3 : tensor<?x3072xf32>) attrs {disc.device = "cpu", disc.transform.name = "maximum"} {
        ^bb0(%in: f32, %in_5: f32, %in_6: f32, %out: f32):
          %12 = arith.addf %in_5, %in_6 : f32
          %13 = arith.maxf %in, %12 : f32
          linalg.yield %13 : f32
        } -> tensor<?x3072xf32>
        scf.yield %t2, %t4 : tensor<?x3072xf32>, tensor<?x3072xf32>
      }
      use(%4)
     ```
  }];

  let arguments = (ins PDL_Operation:$target,
                       PDL_Operation:$loop);
  let results = (outs PDL_Operation:$tiled_target, PDL_Operation:$fused_loop);

  let assemblyFormat = [{
    attr-dict $target `into` $loop
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";
}

def ReductionInputFuseOp : Op<Transform_Dialect, "disc.reduction_input_fuse",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Given a linalg op, input fuse it into the reduction loop.

    Returns the fused linalg op and the new loop op.

    Example:
     convert from:
     ```
      %0 = linalg.fill(%init)
      %1 = scf.for %arg1 = %c0 to %c768 step %c512 iter_args(%arg0 = %0) -> (tensor<?x3072xf32>) {
        ...
        %t0 = tensor.extract_slice %arg0[0, 0] [...] [1, 1] : tensor<?x3072xf32> to tensor<?x3072xf32>
        %t1 = linalg.matmul {disc.transform.name = "dot_general"} ins(...) outs(%t0 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
        %t2 = tensor.insert_slice %t1 into %arg0[0, 0] [...] [1, 1] : tensor<?x3072xf32> into tensor<?x3072xf32>
        scf.yield %t2 : tensor<?x3072xf32>
      }
      use(%0)
     ```
     to:
     ```
      %1 = scf.for %arg1 = %c0 to %c768 step %c512 iter_args(%arg0 = %0) -> (tensor<?x3072xf32>) {
        ...
        %pred = ... // is first iteration?
        %t0 = disc_linalg_ext.conditional_generic ins(%pred) outs(%arg0)
        %t1 = tensor.extract_slice %t0[0, 0] [...] [1, 1] : tensor<?x3072xf32> to tensor<?x3072xf32>
        %t2 = linalg.matmul {disc.transform.name = "dot_general"} ins(...) outs(%t1 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
        %t3 = tensor.insert_slice %t2 into %arg0[0, 0] [...] [1, 1] : tensor<?x3072xf32> into tensor<?x3072xf32>
        scf.yield %t3 : tensor<?x3072xf32>
      }
      use(%0)
     ```
  }];

  let arguments = (ins PDL_Operation:$target,
                       PDL_Operation:$loop);
  let results = (outs PDL_Operation:$tiled_target, PDL_Operation:$fused_loop);

  let assemblyFormat = [{
    attr-dict $target `into` $loop
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";
}

def VectorizeConditionalGenericOp : Op<Transform_Dialect, "disc.vectorize_conditional_generic",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Vectorize the given conditional generic op and return the generated if op.

    Suppose the conditional generic op has static shape.

    Example:
     convert from:
     ```
       %pred = ... : i1
       %in0 = ... : tensor<8x12xf32>
       disc_linalg_ext.conditional_generic ins(%pred, %in0 : ...) outs(... : tensor<8x12xf32>) {
       ^bb0(%arg0 : i1, %arg1 : f32, %arg2 : f32):
         %t0 = arith.addf %arg1, %arg1 : f32
         disc_linalg_ext.yield %t0 : f32
       }
     ```
     to (replace v0 with v1 and remove v0):
     ```
      %pred = ... : i1
      %in0 = ... : tensor<8x12xf32>
      %v0 = vector.transfer_read %in0 ...
      %out = scf.if %pred {
        %v1 = arith.addf %v0, %v0 : vector<8x12xf32>
        scf.yield %v1 : vector<8x12xf32>
      } else {
        scf.yield %v0 : vector<8x12xf32>
      }
     ```

  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def SplitVectorTransferIntoFullAndPartialOp : Op<Transform_Dialect, "disc.split_vector_transfer_into_full_and_partial",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Split a vector.transfer operation into an in-bounds (i.e., no out-of-bounds masking) fastpath and a slowpath.

    Returns the xfer op if no needs to split, otherwise returns the warpper if op for full version and partial version.

    /// Example (a 2-D vector.transfer_read):
    /// ```
    ///    %1 = vector.transfer_read %0[...], %pad : memref<A...>, vector<...>
    /// ```
    /// is transformed into:
    /// ```
    ///    %1:3 = scf.if (%inBounds) {
    ///      // fastpath, direct cast
    ///      memref.cast %A: memref<A...> to compatibleMemRefType
    ///      scf.yield %view : compatibleMemRefType, index, index
    ///    } else {
    ///      // slowpath, not in-bounds vector.transfer or linalg.copy.
    ///      memref.cast %alloc: memref<B...> to compatibleMemRefType
    ///      scf.yield %4 : compatibleMemRefType, index, index
    //     }
    ///    %0 = vector.transfer_read %1#0[%1#1, %1#2] {in_bounds = [true ... true]}
    /// ```
    /// where `alloc` is a top of the function alloca'ed buffer of one vector.

  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def LowerConditionalGenericOp : Op<Transform_Dialect, "disc.lower_conditional_generic",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Lower the conditional_generic op to a scf.if + linalg.generic op.

    Note the this transform primitive supposes the conditional_generic has been bufferized.

    Returns the generated linalg.generic op.

    /// Example:
    /// ```
    ///    %1 = disc_linalg_ext.conditional_generic ins(%pred, ...), outs(...)
    /// ```
    /// is transformed into:
    /// ```
    ///    scf.if (%pred) {
    ///      linalg.generic ins(...) outs(...)
    ///      scf.yield %view
    ///    }
    /// ```
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// Copied from community as an workaournd for some wired bugs, e.g, it'll not
// converge if the tile of the k is one.
def DISCLowerVectorsOp : Op<Transform_Dialect, "disc.vector.lower_vectors",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Indicates that the vector operations nested under the isolated from above op
    `target` should be lowered to finer-grained vector primitives.

    At this time, the transform is all or nothing.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  // TODO: evolve this to proper enums.
  let arguments = (ins PDL_Operation:$target,
     DefaultValuedAttr<VectorContractLoweringAttr,
       "vector::VectorContractLowering::OuterProduct">:$contraction_lowering,
     DefaultValuedAttr<VectorMultiReductionLoweringAttr,
       "vector::VectorMultiReductionLowering::InnerParallel">:
         $multireduction_lowering,
     DefaultValuedAttr<VectorTransferSplitAttr,
       "vector::VectorTransferSplit::LinalgCopy">:$split_transfers,
     DefaultValuedAttr<VectorTransposeLoweringAttr,
       "vector::VectorTransposeLowering::EltWise">:$transpose_lowering,
     DefaultValuedAttr<BoolAttr, "false">:$transpose_avx2_lowering,
     DefaultValuedAttr<BoolAttr, "true">:$unroll_vector_transfers
  );
  let results = (outs PDL_Operation:$results);

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$target,
      "const vector::LowerVectorsOptions &":$options), [{
        return build($_builder, $_state, resultType, target,
          options.vectorContractLowering,
          options.vectorMultiReductionLowering, options.vectorTransferSplit,
          options.vectorTransposeLowering, options.transposeAVX2Lowering,
          options.unrollVectorTransfers);
      }]
    >
  ];

  let assemblyFormat = [{
    $target
    oilist (
      `contraction_lowering` `=` $contraction_lowering
      | `multireduction_lowering` `=` $multireduction_lowering
      | `split_transfers` `=` $split_transfers
      | `transpose_lowering` `=` $transpose_lowering
    )
    attr-dict
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def DISCForeachThreadToGPUCTAsOp : Op<Transform_Dialect, "disc.foreach_thread_to_gpu_ctas",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    This converts the foreach_threads op with at most three ranks into a
    parallel op for GPU threads mapping of both blocks and threads. the target
    op should have GPU Block mapping.

    Return the generated parallel op of rank 2.

    /// Example:
    ///
    /// from
    /// ```
    /// scf.foreach_thread (%arg3, %arg4) in (%c2, %c2) {
    ///   xxx = some_op(%arg3, %arg4)
    /// }
    /// ```
    /// to
    /// ```
    /// scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c4, %c256)
    ///                               step (%c1, %c1) {
    ///   %0:2 = "disc_shape.delinearize"(%arg3, %c2, %c2) :
    ///                    (index, index, index) -> (index, index)
    ///    xxx = some_op(%0#0, %0#1)
    /// }
    /// ```
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$parallel_op);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}


def DISCForeachThreadToGPUWarpsOp : Op<Transform_Dialect, "disc.foreach_thread_to_gpu_warps",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    This converts the foreach_threads op with at most three ranks into the
    computations of the current warp according to the calculation with gpu
    thread id and numthreads of current op.
    TODO: use gpu.warp instead of gpu.thread for the mapping attribute.

    /// Example:
    ///
    /// from
    /// ```
    /// scf.foreach_thread (%arg3, %arg4) in (%c2, %c2) {
    ///   xxx = some_op(%arg3, %arg4)
    /// } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
    /// ```
    /// to
    /// ```
    /// %warp_linear = gpu.threadX / 32
    /// %warp_idx = disc_shape.delinearize(%warp_linear, %c2, %c2)
    /// xxx = som_op(%warp_idx#0, %warp_idx#1)
    /// ```
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def DISCSplitReductionSerialOp :
  Op<Transform_Dialect, "disc.split_reduction_serial",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Indicates that the given `target` op should be transformed with the
    `tileReduction` transformation with the tile size provided as attribute.

    This transformation tiles the `target` along the reduction dimensions. It
    creates a tensor initialized with the identity value. Then it creates nested
    loops with a parallel version of `target` op inside. The parallel op
    dimensions are less or equal to the tile size passed by user.
    The only difference with `structured.tile_reduction_using_foreach_thread` is
    that, the splitted tiles are executed inplace in serial.
    The initial tensor always uses the tile size dimension. This may overallocate
    if the tile size is greater than the reduction dimension.

    #### Return modes

    This 1 returned handle point to:
      - the splitted op,
      - the parent for op,

    #### Example:

    ```
    %matmul = linalg.matmul ins(%lhs, %rhs : tensor<128x1024xf32>,
                                             tensor<1024x128xf32>)
                            outs(%out : tensor<128x128xf32>) ->
                            tensor<128x128xf32>
    ```

    is transformed into:

    ```
    %1 = scf.for %arg0 = %c0 to %c1024 step %c32
            iter_args(%arg1 = %out) -> tensor<128x128xf32> {
      %extraced_slice_lhs = tensor.extract_slice %lhs[0, %arg0]
            [128, 32] [1, 1] : tensor<128x1024xf32> -> tensor<128x32xf32>
      %extraced_slice_rhs = tensor.extract_slice %rhs[%arg0, 0]
            [32, 128] [1, 1] : tensor<1024x128xf32> -> tensor<32x128xf32>
      %matmul = linalg.matmul ins(%extraced_slice_lhs, %extraced_slice_rhs :
                                  tensor<128x1024xf32>, tensor<1024x128xf32>)
                              outs(%arg1 : tensor<128x128xf32>) ->
                              tensor<128x128xf32>
    }
    ```
  }];

  // TODO: support mixed static-dynamic (see TileToForeachThreadOp).
  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$tile_sizes);
  let results = (outs PDL_Operation:$for_op,
                      PDL_Operation:$splitted_op);

  let assemblyFormat = [{
    $target
    `by` `tile_sizes` `=` $tile_sizes
    attr-dict
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def DISCVectorToMMAConversionOp : Op<Transform_Dialect, "disc.vector.vector_to_mma_conversion",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    This converts slices of operations containing vector.contract op into
    mma operations, targetting warp level tensorcore operations. If the vector
    operations are bigger than the native mma size it will first split up those
    vector operations.

    It uses MMA rather than WMMA.

    #### Return modes

    This transform consumes the target handle and produces a result handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def DISCPromoteDotOperandsOp :
  Op<Transform_Dialect, "disc.promote_dot_operands",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    This op promotes the specified operands of the provided target handle.

    #### Return modes
    This op consume its target handle and returns a new handle to its target handle
    as well as an allocTensorOp for each of the provided valid indices.

    If the promotion of any specified operand fails to occur, the op definitely
    fails.
  }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$indices);
  let results = (outs PDL_Operation:$promoted_dot,
                      PDL_Operation:$lhs_alloc,
                      PDL_Operation:$rhs_alloc);

  let assemblyFormat = [{
    $target
    $indices
    attr-dict
    `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def DISCLowerGmemToSmemOp : Op<Transform_Dialect, "disc.gmem_to_smem",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Converts the linalg.generic that executes the global memory to shared memory
    movement function to memref operations on the GPU. It requires that the gmem
    to smem is expressed as an linalg.generic op.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def DISCEraseDeallocOp : Op<Transform_Dialect, "disc.erase_dealloc",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Erases dealloc ops, for GPU functions.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def DISCTransferWriteZeroToSCFOp : Op<Transform_Dialect, "disc.transfer_write_zero_to_scf",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Convert vector.transfer_write that reads zeros to scf loop containing
    memref.store.

    The input should be the function op.

    Example:

    ```
    %cst = arith.constant dense<0.0> : vector<128x128xf16>
    %vector.transfer_write %cst, %memref[%c0, %c0] {in_bounds = [true, true]} :
        vector<128x128xf16>, memref<128x128xf16>
    ```

    is transformed to:

    ```
    %cst = arith.constant 0.0 : f16
    scf.for %arg0 = %c0 to %c128 step %c1 {
      scf.for %arg0 = %c1 to %c128 step %c1 {
        memref.store %cst, memref[%arg0, %arg1] : memref<128x128xf16>
      }
    }
    ```
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def DISCInlineAndConvertGPUIdsOp : Op<Transform_Dialect, "disc.inline_and_convert_gpu_ids",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Inline the gpu id ops before the parallel op of cta, and convert to the
    calculations according to the induction of parallel op.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::disc_ral::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // DISC_TRANSFORM_OPS_EXT