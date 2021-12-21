
![image.png](http://pai-blade.cn-hangzhou.oss.aliyun-inc.com/disk%2Fdisc_pass_pipeline.png#clientId=u23e5bd68-8133-4&from=paste&height=563&id=u5056a176&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1048&originWidth=1347&originalType=binary&size=188361&status=done&style=none&taskId=ucef62950-c154-4a5c-bf5c-fd0df29f332&width=723.991455078125)

## TF2DHLO

## DHLO2Executable
### Inliner
Inline all functions starting from the entry function `tao_main`. Following passes assume that there is only one computation function, thus we do inlining here to ensure this property.

### UnusedFunctionRemovalPass
A pass that eliminates unused functions. This supposes that any function that isn't called by the `tao_main` function directly or indirectly is unused. After removing all unused functions, it checks if there is only `tao_main` function left. This is because following passes rely on this assumption. This assumption can be ensured by calling `inline` pass.

TODO: re-visit this pass after we upgrade to the latest mlir version since it may be possible to merge inline and remove unused function into single pass in the latest master codebase.

### DhloLoopInvariant
An ordinary loop invariant pass for hlo.WhileOp. This pass not only removes the potential redundant calculations, but also removes the potential redundant alloc/dealloc after the LegalizeLhloControlFlow pass.
Step 1: Remove 'transparent' args of the hlo.WhileOp, an argument of hlo.WhileOp is defined as 'transparent' if it is not used in cond_bb and not updated in body_bb. The pattern is like:
```bash
%a = ...
%external_tuple = tuple(..., %a, ...)
xla_hlo.while(%external_tuple)
  cond_bb(%arg) {
    %b1 = xla_hlo.get_tuple_element(%arg, some_idx)
    ...
  }
  body_bb(%arg) {
    %b2 = xla_hlo.get_tuple_element(%arg, some_idx)
    %body_tuple = xla_hlo.tuple(..., %2, ...);
    return %body_tuple
  }
```
Step 2: Recursively move any ops in the cond/body region out of the While, if all of its operands reside in the dominant block.

TODO: re-order this pass after UnusedFunctionRemovalPass.


### DhloBroadcastSimplifierPass
The main purpose of this pass is to materialize implicit broadcast semantic of element-wise binary ops. To this end, explicit broadcast ops will be inserted to make sure each element-wise binary op numpy-compatible.

After conversion, attribute kDhloNoImplicitBroadcastAttr will be attached to each converted op. If a candidate op already has been tagged with kDhloNoImplicitBroadcastAttr, it will be left untouched.

reference：
There are three types of implicit broadcast:

-  a) Broadcasting a lower-rank array onto a higher-rank array
For example:
|1 2 3| + |7 8 9|
|4 5 6|
equal to:
|1 2 3| + |7 8 9| = |8  10 12|
|4 5 6|   |7 8 9|   |11 13 15|
if we specify the broadcast dimention of `|7 8 9|` to 1.
The broadcast dimention specifies which dimension(s) in the
higher-rank array to match with the lower-rank array.
Scalar operand can omit broadcast dimention. For example:
|1 2 3| + 7 = |8  9  10|
|4 5 6|       |11 12 13| 
-  b) Broadcasting similar-rank arrays with degenerate dimensions
This kind of broadcasting has two operands having the same rank
but different dimension sizes. This is only possible when the
dimensions are compatible. Two arrays are compatible when all their
operands are compatible. Two dimensions are compatible if: 
   1. They are equal, or
   1. One of them is 1


Examples: 

   - (2,1) and (2,3) broadcast to (2,3)
   - (1,2,5) and (7,2,5) broadcast to (7,2,5)

 

-  c) combine of the former two types
|1| + ||5 6|| = ||1 1|| + ||5 6||
|2|             ||2 2|| + ||5 6||
|3|             ||3 3|| + ||5 6||
|4|             ||4 4|| + ||5 6|| 



### DhloBaceGemm
This pass implements some optimizations for dot/gerneal_dot ops. Specifically, following optimizations will be applied:

- folding tranpose ops into their consumer dot/general_dot ops.
- injecting some heuristic logic:  insert padding/slicing logic before and after dot op to make it suitable for (GPU) tensorcore. This heuristic is benifit if the  shape of the dot op is large enough, thus a mhlo conditional op is also inserted to guard this assumption.



### DhloConvRewriterPass
This pass converts conv ops in mhlo dialect to match the format of CUDNN library call (if they are 
placed on GPU).

Currently only convolution forword op is supported. 
The following data format are supported. 
Suppose a convolution forword op having:   output = DConv(input, filter, ...)

- supported input format 
   - NCHW (preferred on GPU?)
   - NHWC
- supported filter (kernel) format 
   - OIHW (preferred on GPU?)
   - OHWI
- supported output format 
   - NCHW (preferred on GPU?)
   - NHWC



The basic logic of this pass is to insert transpose op to ensure the conv op placed on GPU having cudnn-friendly format.


### DhloGpuConvPaddingLegalizationPass
This pass rewriters conv ops' padding value to match the format of CUDNN library call.
cuDNN only supports padding the same amount on the left and right sides, and on the top and bottom sides. So we manually create a new padded input tensor such that we can pass it to cuDNN

### DhloCanonicalizeReductionPass
This pass canonicalizes reduction ops in hlo dialect to match the capacity of codegen backend. Currently our codegen backend only supports 2d row/column reduction ops. Specifically, all the reduce ops can be divided into following four types:


- a) column reduction, only reduce the most significant dimensions.
- b) row reduction, only reduce the least significant dimensions.
- c) reduce to scalar, all dimensions are reduced.
- d) others. (using transpose to canonicalize???)



Currently we do following canonicalization to match the capacity of codegen backend.


- For case a): 

we convert all column reduction to rank-2 column reduction. For example, suppose we have:


```
  func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
    ...
    %2 = "xla_hlo.reduce"(%arg0, ...) ( {...})
      {dimensions = dense<[0]> : tensor<1xi64>} :
      (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
    return %2 : tensor<?x?xf32>
  }
```


After conversion:


```
   func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
     // [a, b, c] -> [a, b*c]
     %1 = xla_hlo.d_reshape(%arg0, ...) : (tensor<?x?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
     %2 = "xla_hlo.reduce"(%1, ...) ( {...})
       {dimensions = dense<[0]> : tensor<1xi64>} :
       (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
     %3 = "xla_hlo.d_reshape"(%2, ...) : (tensor<?xf32>, tensor<1xi64>) -> tensor<?x?f32>
     return %3 : tensor<?x?xf32>
   }
```


- For case b):



we convert all row reduction to rank-2 row reduction. For example, suppose we have:


```
  func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
    ...
    %2 = "xla_hlo.reduce"(%arg0, ...) ( {...})
      {dimensions = dense<[2]> : tensor<1xi64>} :
      (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
    return %2 : tensor<?x?xf32>
  }
```


After conversion:


```
  func @test_rank3_column_reduction(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
    // [a, b, c] -> [a*b, c]
    %1 = xla_hlo.d_reshape(%arg0, ...) : (tensor<?x?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    %2 = "xla_hlo.reduce"(%1, ...) ( {...})
      {dimensions = dense<[1]> : tensor<1xi64>} :
      (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
    %3 = "xla_hlo.d_reshape"(%2, ...) : (tensor<?xf32>, tensor<1xi64>) -> tensor<?x?f32>
    return %3 : tensor<?x?xf32>
  }
```


- For case c):



we convert all reduce-to-scalar to rank-2 column reduction. For example, suppose we have:


```
  func @test(%arg0: tensor<?x?x?xf32>) -> tensor<f32> {
    ...
    %2 = "xla_hlo.reduce"(%arg0, ...) ( {...})
      {dimensions = dense<[0,1,2]> : tensor<3xi64>} :
      (tensor<?x?x?xf32>, tensor<f32>) -> tensor<f32>
    return %2 : tensor<f32>
  }
```


After conversion:


```
   func @test(%arg0: tensor<?x?x?xf32>) -> tensor<f32> {
     // [a, b, c] -> [a*b*c, 1]
     %1 = xla_hlo.d_reshape(%arg0, ...) : (tensor<?x?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
     %2 = "xla_hlo.reduce"(%1, ...) ( {...})
       {dimensions = dense<[0]> : tensor<1xi64>} :
       (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
     %3 = "xla_hlo.reshape"(%2, ...) : (tensor<?xf32>, tensor<1xi64>) -> tensor<f32>
     return %3 : tensor<f32>
   }
```


### DhloSplitLargeOpsPass
Some ops may have too many operands. After lowering, the kernels generated for these ops may exceed the maximum parameter space of a kernel on GPU, which is illegal. Therefore, we explicitly split these ops aforehand.

Examples are:
```
// original op
out = hlo.concat(in0, in1, ... inN)

// after conversion
out0 = hlo.concat(in0, in1, inN1)
out1 = hlo.concat(inN1, ..., inN)
out = hlo.concat(out0, out1)
```


### DhloPlacer
This pass only works for GPU backend, it explicitly place the shape calculating hlo ops on host side and then insert memcpy nodes according the placement result. 

An op is regarded as 'shape calculating op' if one of the two conditions are met:
(1) it is the direct or indirect producer of a shape calculating operand of some hlo op. A shape calculating operand is the operand which affects the shape of the hlo op's result. For example, the shape operand of the DynamicReshapeOp. The rule is similar with the process of BackwardConstAnalysis() in XLA.
(2) its operating data type is I32. This is a rule similar to TensorFlow.

Currently there are no dedicated attributes for placement assignment. kHloPlaceAssignment is currently used for it. Normally, the type of kHloPlaceAssignment is StringAttr; In case the result type of an hlo op is TupleType, for example TupleOp or TopKOp, the type of kHloPlaceAssignment is an ArrayAttr made of StringAttr. For non-tuple type ops, if it is placed on GPU the attributes will be omitted, while for TupleType ops the attibutes cannot be omitted even if all the tuple elements are on GPU.

TODO: add a dedicated attribute in MLIR for placement assignment, since a custom attribute could easily be lost unless all the passes are aware of it.
TODO: Do not omit the attribute even if it is placed on GPU. This is more feasible for CPU backend, or a variation of placement strategy with CPU/GPU mixed.


### DhloElementTypeConverterPass
This pass basically contains two functions:

- eliminates certain element types as the input or output of ops by inserting Convert ops. This allows a backend to support an element type while only actually implementing the Convert op for that element type. This is generally not the fastest approach, but it works. For example, convert i1 type reduction to i32 type reduction.
- A basic automatic mixed precision implmentation. This will convert some ops (e.g. dot, conv) with fp32 to their fp16 formats (with convert op inserted appropriately) accordlingly.



### DHloLegalizeToLhlo
The basic idea of this pass is to convert dhlo ops to their lhlo counterparts. To this end, it will insert allocOp/deallocOp appropriately. If the output shape of a op is unknown, shape inference ir will also be injected. Note that shape inference IRs are supposed to be executed on CPU.

TODO：remove the unnecessary kTempBufferAttr attributes.


### PlacementFixerPass
During legalizing DHLO to LDHLO, new lhlo ops may be inserted (e.g. insert copy op to simplify memref lifetime management for control flow ops). The new ops do not have the placement attribute. This pass assign placement attribute to those ops according to their surrounding ops' placement.


### DhloFuseOnBuffer
This pass has similar functionality of the fusion pass in XLA stack. However, unlike XLA, it targets the fully dynamic shape scenario. Currently, it implements the kLoop and kInput fusion templates.
During conversion, it tries to greedily find kLoop/kInput fusion patterns.

Similar to XLA, this pass supports fusion pattern having multiple outputs if all the shape of outputs are consistent. Following are some examples.

```
          kLoop                          kInput
   +----+  +----+  +----+    +----+    +----+    +----+
   |elem|  |elem|  |elem|    |elem<----+elem+---->elem+----+
   +-+--+  +-+--+  +-+--+    +-+--+    +----+    +-+--+    |
     |       |       |         |                   |       |
     |               |         |                   |       |
   +-v--+    |     +-v--+   +--v---+            +--v---+   |
   |elem+<---+----<+elem|   |reduce|            |reduce|   |
   +-+--+          +-+--+   +--+---+            +--+---+   |
     |               |         |                   |       |
     |               |         |                   |       |
     v               v         v                   v       v
```

To this end, we also add a simple shape constraint analysis phase. For kLoop fusion template, it requires all the outputs of the fused pattern have the same shape. However, we don't know the actual value of the shape at the compile time in the dynamic shape world. Fortunately, we could still infer the relationship among different ops according to their shape constrain traits. Currently, We only consider
shape equality propagation for elementwise ops (assuming that implicit shape broadcast is forbidden). The above process could be built on the shape dialect once it is ready.

The reason that we choose to do fusion on buffer level instead of doing fusion on tensor level is that dim ops of the intermediate output of a fusion pattern are already resolved after bufferization.


### DhloLowerToBufferAlias
After fusion, there still may be some standalone DReshape/DCopy ops. Instead of materialization these ops into concrete kernels, we choose to lower appropriate ops into buffer alias only (e.g. increase the buffer reference count by 1 in the runtime implementation). This optimization is based on the assumption that a buffer can only be written once and is readonly after that. This assumption is generally true for our current implementation.


### DhloLocalizeSmallCpuBuffer
After shape inference, the shape of the output of a hlo op will be stored into a rank-1 buffer.  These buffer are usually relative small.  Thus ,we choose to use stack memory instead of heap memory for them if appropriate (e.g. buffer lifetime is fully controlled by the compiler, not the final output of the executable).


### DhloBufferOptimization
This pass forwards dealloc op as far as possible to free a buffer as soon as it has no users.


### DhloSpecializeFusionWithSpeculation
This pass implements the logic for specializing the fusion kernel with speculation.

Currently we have following two kinds of specializations.

- broadcast simplification with speculation. We generate two versions for a fusion pattern having candidate broadcast ops. The first one is the simplest version, which suppose no implicit broadcast will occur. The second one is the original version with all broadcast ops left untouched.



Here 'candidate broadcast' means input and output of the broadcast op have same rank and all dimensions are matched. Two dimensions are matched if they are both unkonwn or are same.
```
// basic logic is shown as below:

// Here 'if' is implemented using xla_lhlo.conditional op.
%pred = ...
if (%pred) {
  call %simplified_fusion_func(...)
} else {
  call %original_fusion_func(...)
}
```


An example is shown below.
```
// original version
func @fusion(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>, %arg2 : memref<?xf32>, %arg2 : memref<?x?xf32>, %arg3 : memref<?x?xf32>, %arg4 : memref<?x?xf32>) {
	%bcast_shape = ...
  xla_lhlo.d_broadcast(%arg0, %bcast_shape, %arg1) :（memref<?x?xf32>, memref<2xi32>, memref<?x?xf32>） -> memref<?x?xf32>
  xla_lhlo.d_broadcast(%arg2, %bcast_shape, %arg3) :（memref<?xf32>, memref<2xi32>, memref<?x?xf32>） -> memref<?x?xf32>
  xla_lhlo.add(%arg1, %arg3, %arg4) : （memref<?x?xf32>, memref<2xi32>, memref<?x?xf32>） -> memref<?x?xf32>
}

// simplified version
func @fusion(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>, %arg2 : memref<?xf32>, %arg2 : memref<?x?xf32>, %arg3 : memref<?x?xf32>, %arg4 : memref<?x?xf32>) {
	%bcast_shape = ...
  xla_lhlo.d_broadcast(%arg2, %bcast_shape, %arg3) :（memref<?xf32>, memref<2xi32>, memref<?x?xf32>） -> memref<?x?xf32>
  xla_lhlo.add(%arg0, %arg3, %arg4) : （memref<?x?xf32>, memref<2xi32>, memref<?x?xf32>） -> memref<?x?xf32>
}
```


- kInput fusion pattern codegen plan specialization.  The motivation behind this is that we find the best codegen plan for a reduce op depends on its operand's shape size. For example, if the size of the dimension to reduce is small, using one warp to handle one line is appropriate. Otherwise,  we may need to use more warps to handle one line in order to achieve better performace.
```
%reduction_dim_size = ...

// Here 'if' is implemented using xla_lhlo.conditional op.
if (%reduction_dim_size < %threshold0) {
  // version #0
} else if (%reduction_dim_size < %threshold1) {
  // version #1
} else {
  // original (default) version
}

```


### LegalizeLhloControlFlow
This pass lowers the lhlo.WhileOp & lhlo.ConditionalOp into CFG in Standard Dialect.

A variant solution is to directly lower hlo.WhileOp & hlo.ConditionalOp into Standard CFG. The main reason for not doing it this way is: if LegalizeHloControlFlow happens before HLOLegalizeToLHLO, HLOLegalizeToLHLO pass must be able to handle multi blocks in a region. This will bring much complexity in buffer liveness analysis.


### InsertMemoryMarker
For now std.Alloc (or memref.Alloc in the latest version) doesn't have any syntax to indicate if this is a host alloc or device alloc. This pass works with the ConvertMemoryMarker pass as a workaround for this issue.
On buffer world, it analysis the buffer location and insert a void marker call for GPU allocation: 
```bash
call @mcuCuMemMarker2df32_x_x(%213) : (memref<?x?xf32>) -> ()
%214 = alloc(%211) {temp = true} : memref<?xf32>
```
The marker will be processed later in ConvertMemoryMarker pass in LLVM Dialect.
The reason for not using an attribute instead of a marker is to guarantee the consequent passes can properly handle it (not to drop the custom attribute).

TODO: add official syntax on memref.Alloc/Dealloc to indicate the buffer type.

### DhloInjectExecutionContext
DISC is a e2e flow, including both compiler side and runtime side. For runtime side, we have different targeting environments (e.g. tensorflow, pytorch, or sometimes even a standalone binary). In order to simplify the design of the compiler side, we design a Runtime Abstraction Layer (RAL) to sperate the compiler side and runtime side. Thus the compiler side only need to target RAL itself and it is the responsibility of RAL to handle the differences between different targeting environments.

Another function of RAL is to manage stateful resources. To this end, it provides a context object, and hides all stateful operations behind this context, thus the compiler side itself doesn't need to care about the resource initialization. For example, a kernel must be loaded before it can be launched on GPU. However, the loading operation should only be taken once during the whole lifetime of the context in order to achieve the best performance. Based on the initialization-free interfaces (e.g. load_kernel_if_not_and_then_launch_kernel) provided by RAL, compiler side can focus on its core optimization logic and lets the RAL to manage the resource status.

The context mentioned above is passed as a parameter to the entry function and all RAL APIs should always use the context as their first argument. This pass helps to ensure this property. The pass rewrites the entry function and all the fusion functions to make sure their first argument is the context. For entry function, the pass also rewrites its inputs and outputs. To be concrete, all the original inputs and outputs of the entry function are received from and sent to RAL through a sequence of RAL API calls correspondingly. The motivation behind this is to hide the implementation details of memref structure.


Below is an example.

```
// original module
func @tao_main(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>) -> memref<?x?xf32> {
  %tmp = alloc(...)
  %ret = alloc(...)
  call fusion_function(%arg0, %arg1, %tmp, %ret)
  return %ret : memref<?x?xf32>
}

func @fusion_function(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>, %arg2 : memref<?x?xf32>, %arg3 : memref<?x?xf32>) {
	xla_lhlo.add(%arg0, %arg1, %arg2)
  xla_lhlo.mul(%arg0, %arg2, %arg3)
}

// after conversion
func @tao_main(!llvm<"i8*"> %ctx) {
  %arg0 = ral_recv_input(%ctx, 0) // receive the first input
  %arg1 = ral_recv_input(%ctx, 1) // receive the second input
  %tmp = alloc(...)
  %ret = alloc(...)
  call fusion_function(%ctx, %arg0, %arg1, %tmp, %ret)
  ral_send_output(%ctx, 0, %ret) // send the first outpt
}

func @fusion_function(!llvm<"i8*"> %ctx, %arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>, %arg2 : memref<?x?xf32>, %arg3 : memref<?x?xf32>) {
	xla_lhlo.add(%arg0, %arg1, %arg2)
  xla_lhlo.mul(%arg0, %arg2, %arg3)
}
```


### LowerToLibraryCallsV2
Convert the non-codegen ops into Ral library calls. The converted calls are in the form like:
```bash
// In this example "pvoid_pvoid_m2df32_m2df32_m2df32_i1_i1" is the arguments signature;
//   the first pvoid refers to the context handle
//   the second pvoid refers to the reserved stream handle
//   m2df32_m2df32_m2df32 are the types of lhs/rhs and output type of gemm
//   the last two "i1" refers to the transpose indicator of lhs/rhs
//   "void" is the signature of the returned type of the ral call.
call @ral_gemm___pvoid_pvoid_m2df32_m2df32_m2df32_i1_i1___void(%arg0, %7017, %6979, %7018, %7019, %false, %false) : (!llvm<"i8*">, !llvm<"i8*">, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, i1, i1) -> ()

```
The non-codegen ops currently includes:
lhlo.Dot/DotGeneral, lhlo.DConv, lhlo.D2H/H2D/Copy, lhlo.DTopk, lhlo.RngUniform, lhlo.Print 


Please also note that lhlo.DBitcast/Bitcast also needs to be converted to an ral call in order to update the reference counters.

### ConvertConstToTaoRAL
Convert the lhlo.Const ops into Ral library calls. The converted calls are in the form like:
```bash
// in this example "40F7E71261729523BBE798C4347EEA1B" is the hash value of the constant value
// "i32_2" indicates the constant type is <2xi32>
llvm.mlir.global internal constant @"40F7E71261729523BBE798C4347EEA1B_i32_2"("40F7E71261729523BBE798C4347EEA1B_i32_2\00")
 
%741 = llvm.mlir.addressof @"40F7E71261729523BBE798C4347EEA1B_i32_2" : !llvm<"[39 x i8]*">
%742 = llvm.mlir.constant(0 : index) : !llvm.i64
%743 = llvm.getelementptr %741[%742, %742] : (!llvm<"[39 x i8]*">, !llvm.i64, !llvm.i64) -> !llvm<"i8*">
%744 = call @ral_constant_host_40F7E71261729523BBE798C4347EEA1B_i32_2(%arg0, %743) : (!llvm<"i8*">, !llvm<"i8*">) -> memref<2xi32>
```
This pass will write to a seperate pb file with the map from the hash value of each constant to its original hex values. This prevents the lowered binary from being too huge. The Ral runtime will read the pb file at runtime and load the proper constant values.

### LhloLegalizeToLoop
Expand the root ops in a fused func into a set of nested loops.  This pass must be executed after the fusion pass, and works together with the InputInlineFusion pass for fusion codegen. 
![input inline fusion.png](http://pai-blade.cn-hangzhou.oss.aliyun-inc.com/disc/fusion_codegen.png#clientId=uadfaa883-8962-4&from=ui&id=ubfa98258&margin=%5Bobject%20Object%5D&name=input%20inline%20fusion.png&originHeight=523&originWidth=1670&originalType=binary&size=72733&status=done&style=none&taskId=u559be16e-df30-41dc-a685-a569f4136a3)

The expansion is backend-aware and actually decides the schedule of the finally generated kernel. The ops in GPU dialect will also be brought in this pass.

Currently we have 4 schedules for GPU backend:

- RowReductionSchedule1: two round of warp shuffle for row-reduction, suitable for 
- RowReductionSchedule2: one round of warp shuffle for row-reduction
- ColReductionSchedule: reduction implemented with atom operations
- LoopSchedule: schedule of normal loop fusion



It selects a best schedule according to the composition of root ops:

| Row Reduction | Column Reduction | Others | Schedule |
| --- | --- | --- | --- |
| yes | - | - | RowReductionSchedule1 or RowReductionSchedule2 according to the hint |
| no | yes | - | ColReductionSchedule |
| no | no | yes | LoopSchedule |



If both row-reduction and col-reduction exist, the schedule of row-reduction will be chosen and the col-reduction will be implemented with atomic instructions.


If there's any reduction implemented with atomic instructions, a seperate loop will be generated. It will be further lowered into a initializing kernel. 

### InputInlineFusion
This pass works after LhloLegalizeToLoops pass for the fusion codegen. Refer to the figure in LhloLegalizeToLoop.

It iteratively looks for the Lhlo op which is the direct producer of the nested loops, and then inline fuse it if the fusion will not form a loop. 

The inline fusion action can be generalized as:
step 1: replace the producer Lhlo op into associate std op inside the nested loops.
step 2: remove the original Load ops inside the loops and insert new Load ops.

There will be no explicit GeneratedValueCache for LhloLegalizeToLoops and InputInlineFusion. However, similar results will be obtained by the following approach:

- For LhloLegalizeToLoops, if the root_lhlo_a is the direct producer of root_lhlo_b, just use the cached value of root_std_a as operand to emit root_std_b inside the nested loops; If root_lhlo_a is a indirect producer of root_lhlo_b, expand the both in the nested loops and keep root_lhlo_a temporarily. The redundancy will be further processed when the the nodes in between them are fused in InputInlineFusion pass afterwards.
- For InputInlineFusion, if there are multiple LoadOps with the same indices, they will be replaced with the same op. This obtains the similar result as GeneratedValueCache.



### DhloRemoveDeadBuffer
This pass removes dead buffers after fusion (e.g. intermediate buffers of a fusion pattern).


### DeadTempBufferRemoval
A simple pass that removes temporary buffers that are only written to but never read from, or, are to be read from but the read value is never used.

### LoopSplitPass
This pass works after the LoopCoalescingPass for GPU kernels. It splits the external (parallel) loop generated in LoopCoalescingPass logically into a 2 layer nested loops. The size of the internal size is indicated by the kThreadPerBlockHint attribute. The logically 2 layer nested loops is then mapped to gpu blocks and threads and a gpu.LaunchOp will be created.
examples before the LoopSplitPass:
```bash
  %0 = dim %arg11, 0 : memref<?x?xf32>
  %1 = dim %arg11, 1 : memref<?x?xf32>
  %2 = muli %0, %c256 : index
  loop.for %arg12 = %c0 to %2 step %c1 {      // the externel parallel loop
    %3 = remi_signed %arg12, %c256 : index
    %4 = divi_signed %arg12, %c256 : index
    %5 = remi_unsigned %3, %c32 : index
    %6 = divi_unsigned %3, %c32 : index
    %7 = cmpi "eq", %5, %c0 : index
    %8 = cmpi "eq", %3, %c0 : index
    %9 = load %arg5[] : memref<f32>
    %10 = alloc() {temp} : memref<32xf32, 3>
    ...

```
examples after the LoopSplitPass:
```bash
  %3 = muli %c1, %c256_0 : index
  %4 = subi %2, %c0 : index
  %5 = addi %4, %3 : index
  %6 = subi %5, %c1_2 : index
  %7 = divi_unsigned %6, %c256_0 : index
  %c1_3 = constant 1 : index
  %8 = cmpi "ne", %7, %c0_1 : index
  loop.if %8 {
    gpu.launch blocks(%arg12, %arg13, %arg14) in (%arg18 = %7, %arg19 = %c1_3, %arg20 = %c1_3) threads(%arg15, %arg16, %arg17) in (%arg21 = %c256_0, %arg22 = %c1_3, %arg23 = %c1_3) {
      %9 = muli %3, %arg12 : index
      %10 = addi %c0, %9 : index
      %11 = muli %3, %arg18 : index
      loop.for %arg24 = %10 to %2 step %11 {
        %12 = addi %arg24, %3 : index
        %13 = cmpi "slt", %2, %12 : index
        %14 = select %13, %2, %12 : index
        %15 = muli %c1, %arg15 : index
        %16 = addi %arg24, %15 : index
        %17 = muli %c1, %arg21 : index
        loop.for %arg25 = %16 to %14 step %17 {
          %18 = remi_signed %arg25, %c256 : index
          %19 = divi_signed %arg25, %c256 : index
          %20 = remi_unsigned %18, %c32 : index
          %21 = divi_unsigned %18, %c32 : index
          ...
```
TODO: for now we have two unnecessary nested loops inside the gpu.launch. They actually play the role of inbound check. Revise them into a normal inbound check is better for performance.

### ReviseGpuKernelOutliningPass
This pass revises the kernel outlining after the GpuKernelOutliningPass:

- For a MemRef resides in host memory, which always means that the MemRef is for shape representation, expand the MemRef into an array of Values. This is due to that the kernel cannot directly gep/load from host addresses.
- Since we are only to make a 'dynamic shape' compiler, not 'dynamic rank' compiler, the shape of shape should always be static. So here it is assumed that the host MemRef must always be static shaped. If later we found this is not always true, revise the form of the kernel into variant number of arguments. This is a little bit more compilicated but still doable.
- For device MemRef, just leave the form here. There will be lots of the args after lowering to llvm, though. Currently no side-effects are observed, we may improve it in the pass in future if neccessary.



An outlining like:
```bash
  "gpu.launch_func"(%6, %c1, %c1, %c256, %c1, %c1, %2, %arg3, %arg2, %arg1, %1, %arg4) {kernel = "dhlo_fusion_xla_lhlo_reduce_2_0_kernel", kernel_module = @dhlo_fusion_xla_lhlo_reduce_2_0_kernel} : (index, index, index, index, index, index, index, memref<f32>, memref<2xi32>, memref<?xf32>, index, memref<?xf32>) -> ()
  
  ...
  
  gpu.module @dhlo_fusion_xla_lhlo_reduce_2_0_kernel {
    gpu.func @dhlo_fusion_xla_lhlo_reduce_2_0_kernel(%arg0: index, %arg1: memref<f32>, %arg2: memref<2xi32>, %arg3: memref<?xf32>, %arg4: index, %arg5: memref<?xf32>) kernel {
      ...
    }
```
will be converted into:
```bash
  %c0_0 = constant 0 : index
  %8 = load %arg2[%c0_0] : memref<2xi32>
  %c1_1 = constant 1 : index
  %9 = load %arg2[%c1_1] : memref<2xi32>
  "gpu.launch_func"(%6, %c1, %c1, %c256, %c1, %c1, %2, %arg3, %8, %9, %arg1, %1, %arg4) {kernel = "dhlo_fusion_xla_lhlo_reduce_2_0_kernel_revised", kernel_module = @dhlo_fusion_xla_lhlo_reduce_2_0_kernel} : (index, index, index, index, index, index, index, memref<f32>, i32, i32, memref<?xf32>, index, memref<?xf32>) -> ()
  
  ...
  
  gpu.module @dhlo_fusion_xla_lhlo_reduce_2_0_kernel {
    gpu.func @dhlo_fusion_xla_lhlo_reduce_2_0_kernel_revised(%arg0: index, %arg1: memref<f32>, %arg2: i32, %arg3: i32, %arg4: memref<?xf32>, %arg5: index, %arg6: memref<?xf32>) workgroup(%arg7 : memref<32xf32, 3>) kernel {
      ...
    }
```


### ConvertMemoryMarkerV2
For now std.Alloc (or memref.Alloc in the latest version) doesn't have any syntax to indicate if this is a host alloc or device alloc. This pass works with the ConvertMemoryMarker pass as a workaround for this issue.

This pass convert the allocs/deallocs with "memory marker" (refer to InsertMemoryMarker pass) into  tao_ral_alloc/free library calls, which will operate with the assigned device allocator by RAL runtime.


If the environment variable "TAO_REPLACE_CPU_ALLOC" is set, alloc/free in host side will also be converted into tao_ral_cpu_alloc/free correspondingly. By this way the host alloc/free can also be handled with the assigned host allocator at runtime. This feature is by default turned off.


TODO: add official syntax on memref.Alloc/Dealloc to indicate the buffer type.
### GpuLaunchFuncToTaoRALPass
This pass implements the logic to lower a gpu::LaunchFuncOp to a RAL gpu launch API call. The RAL GPU launch is a wrapper of cuda driver launch API and supports initialization-free semantic (e.g. no explicit cubin loading operation).


### RewriteRalApiPass
A RAL API is an ordinary c++ function which satisfies the following requirements.

- The first argument of the function should be a pointer to RAL context.
- RAL supports type bool, int8/16/32/64, half/float/double, MemRef struct and ordinary pointer.
- The types of remaining arguments should be supported by RAL.
- The returning type should be supported by RAL.



For a given RAL API, it has a equivalent interface, namely a type-erased c language interface. The main purpose of this kind of interface is used for cross language binding and linking.

Below is an example to demonstrate the c++ format and correspondding uniform type-erased format.
```c
// A RAL API to do GEMM operation, c++ format
void gemm(RalContext*, gpu_stream_handle*, MemRef<float, 2> lhs, MemRef<float, 2> rhs, MemRef<float, 2> out, bool lhs_transposed, bool, rhs_transposed);

// ----======================================

// Uniform type-erased c format
//   the 1st arg is the ral context by design
//   the 2nd arg is the full name of a ral api
//   the 3rd arg is the pointer which points to the arguments of the ral api
//
//   the full name of a ral api consists of two parts: prefix and encoding of its arugments type and return type
//   For example, the full name of a ral call for gemm maybe:
//     ```ral_gemm___pvoid_pvoid_m2df32_m2df32_m2df32_i1_i1___void```
//   in which `ral_gemm` is the prefix, `pvoid_pvoid_m2df32_m2df32_m2df32_i1_i1` is the encoding for the arguments type,
//   and the final `void` is the return type encoding. 
//
// The dispatcher will dispatch to concret RAL API implementation according to the api_name 
void ral_api_call(void* ctx, const char* api_name, void** args);
```


Essentially, the function of this pass is to convert all RAL API calls from original c++ format to the uniform type-erased format.

Furthermore, we also provide a registry mechanism to automatically convert a RAL API from its c++ format to its uniform format. Below is an example.
```c
// A RAL API to do GEMM operation, c++ format
template <typename T>
void gemm(RalContext*, gpu_stream_handle*, MemRef<T, 2> lhs, MemRef<T, 2> rhs, MemRef<T, 2> out, bool lhs_transposed, bool, rhs_transposed);

// ----====================================

// register a new RAL API using macro `TAO_RAL_API`.
//   Usage: TAO_RAL_API(api_prefix, api_function);
//   - generate a unique name based on the prefix provided and the types of inputs and outputs of the function.
//   - automatically provide a uniform interface wrapper
//   - register the wrapper function using the unique name generated.

// register ral_gemm with float type
TAO_RAL_API("ral_gemm", gemm<float>)
// register ral_gemm with half type 
TAO_RAL_API("ral_gemm", gemm<half>)
```


Thus, based on the mechanism in compiler side (this pass) and the mechanism in c++ side,  users don't need to worry about the details of c interface implementation and can use the c++ level api directly.
