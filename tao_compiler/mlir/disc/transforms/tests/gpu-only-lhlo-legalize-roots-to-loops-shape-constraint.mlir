// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=1 DISC_ENABLE_HORIZONTAL_FUSION=1 disc-opt %s -disc-lhlo-legalize-roots-to-parallel-loops -split-input-file | FileCheck %s
// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=1 DISC_ENABLE_HORIZONTAL_FUSION=1 DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL=true disc-opt \
// RUN:   %s -disc-lhlo-legalize-roots-to-parallel-loops -split-input-file | \
// RUN:   FileCheck %s --check-prefix=MEMOPT

// CHECK-LABEL: @non_fusion_elemwise_gpu
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?x?x?xf32, "gpu">, %[[INPUT2:.*]]: memref<?x?x?xf32, "gpu">, %[[OUT:.*]]: memref<?x?x?xf32, "gpu">) -> memref<?x?x?xf32, "gpu">
func.func @non_fusion_elemwise_gpu(%input1: memref<?x?x?xf32, "gpu">, %input2: memref<?x?x?xf32, "gpu">, %out: memref<?x?x?xf32, "gpu">) -> (memref<?x?x?xf32, "gpu">) {
  // CHECK-NOT: lmhlo
  // CHECK: scf.parallel
  "lmhlo.add"(%input1, %input2, %out) : (memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) -> ()
  // CHECK: return %[[OUT]] : memref<?x?x?xf32, "gpu">
  return %out : memref<?x?x?xf32, "gpu">
}

// -----

// CHECK-LABEL: @non_fusion_elemwise_cpu
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?x?x?xf32>, %[[INPUT2:.*]]: memref<?x?x?xf32>, %[[OUT:.*]]: memref<?x?x?xf32>) -> memref<?x?x?xf32>
func.func @non_fusion_elemwise_cpu(%input1: memref<?x?x?xf32>, %input2: memref<?x?x?xf32>, %out: memref<?x?x?xf32>) -> (memref<?x?x?xf32>) {
  // CHECK-NOT lmhlo
  // CHECK: scf.for
  "lmhlo.add"(%input1, %input2, %out) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
  // CHECK: return %[[OUT]] : memref<?x?x?xf32>
  return %out : memref<?x?x?xf32>
}

// -----

// CHECK-LABEL: @slice
// CHECK-SAME: (%[[INPUT:.*]]: memref<?x?xf32>, %[[OUT:.*]]: memref<?x?xf32>) -> memref<?x?xf32>
func.func @slice(%input: memref<?x?xf32>, %out: memref<?x?xf32>) -> memref<?x?xf32> {
  // CHECK-NOT lmhlo
  // CHECK: scf.for
  "lmhlo.slice"(%input, %out) {
    start_indices = dense<[5,6]> : tensor<2xi64>,
    limit_indices = dense<[-1,-1]> : tensor<2xi64>,
    strides = dense<[7,8]> : tensor<2xi64>
  } : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  return %out : memref<?x?xf32>
}

// -----

// CHECK-LABEL: @broadcast
// CHECK-SAME: (%[[INPUT:.*]]: memref<?xf32>, %[[OUT:.*]]: memref<3x?xf32>) -> memref<3x?xf32>
func.func @broadcast(%input: memref<?xf32>, %out: memref<3x?xf32>)->memref<3x?xf32>{
  // CHECK-NOT lmhlo
  // CHECK: scf.for
  "lmhlo.broadcast"(%input, %out) {
    broadcast_sizes = dense<[3]> : tensor<1xi64>
  } : (memref<?xf32>, memref<3x?xf32>) -> ()
  return %out : memref<3x?xf32>
}

// -----

// CHECK-LABEL: @reshape
// CHECK-SAME: (%[[INPUT:.*]]: memref<?x?xf32>, %[[OUT:.*]]: memref<?x4xf32>) -> memref<?x4xf32>
func.func @reshape(%input: memref<?x?xf32>, %out: memref<?x4xf32>) -> memref<?x4xf32> {
  // CHECK-NOT lmhlo
  // CHECK: scf.for
  "lmhlo.reshape"(%input, %out) {
  } : (memref<?x?xf32>, memref<?x4xf32>) -> ()
  return %out : memref<?x4xf32>
}

// -----

// CHECK-LABEL: @transpose
// CHECK-SAME: (%[[INPUT:.*]]: memref<?x?x?xf32>, %[[OUT:.*]]: memref<?x?x?xf32>) -> memref<?x?x?xf32>
func.func @transpose(%input: memref<?x?x?xf32>, %out: memref<?x?x?xf32>)->memref<?x?x?xf32>{
  // CHECK-NOT lmhlo
  // CHECK: scf.for
  "lmhlo.transpose"(%input, %out) {
    permutation = dense<[2,1,0]> : tensor<3xi64>
  } : (memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
  return %out : memref<?x?x?xf32>
}

// -----

// CHECK-LABEL: @dynamic_pad
func.func @dynamic_pad(%operand: memref<?x?x?xf32>, %padding_value: memref<f32>, %edge_padding_low: memref<3xi32>, %edge_padding_high: memref<3xi32>, %interior_padding: memref<3xi32>, %out: memref<?x?x?xf32>) -> memref<?x?x?xf32> {
  // CHECK-NOT lmhlo
  // CHECK: scf.for
  "lmhlo.dynamic_pad"(%operand, %padding_value, %edge_padding_low, %edge_padding_high, %interior_padding, %out) : (memref<?x?x?xf32>, memref<f32>, memref<3xi32>, memref<3xi32>, memref<3xi32>, memref<?x?x?xf32>) -> ()
  return %out : memref<?x?x?xf32>
}

// -----

// CHECK-LABEL: @is_finite
// CHECK-SAME: (%[[INPUT:.*]]: memref<?x?x?xf32>, %[[OUT:.*]]: memref<?x?x?xi1>) -> memref<?x?x?xi1>
func.func @is_finite(%input: memref<?x?x?xf32>, %out: memref<?x?x?xi1>)->memref<?x?x?xi1>{
  // CHECK-NOT lmhlo
  // CHECK: scf.for
  "lmhlo.is_finite"(%input, %out) {
  } : (memref<?x?x?xf32>, memref<?x?x?xi1>) -> ()
  // CHECK: return %[[OUT]] : memref<?x?x?xi1>
  return %out : memref<?x?x?xi1>
}

// -----

// CHECK-LABEL: @gather
func.func @gather(%operand: memref<3xi32>, %start_indices: memref<2xi32>, %out: memref<2xi32>) -> memref<2xi32> {
  // CHECK-NOT lmhlo
  // CHECK: scf.for
  "lmhlo.gather"(%operand, %start_indices, %out) {dimension_numbers = #mhlo.gather<collapsed_slice_dims = [0], index_vector_dim = 1, offset_dims = [], start_index_map = [0]>, indices_are_sorted = false, slice_sizes = dense<[1]> : tensor<1xi64>} : (memref<3xi32>, memref<2xi32>, memref<2xi32>) -> ()
  return %out : memref<2xi32>
}

// -----

// CHECK-LABEL: @dynamic_gather
func.func @dynamic_gather(%operand: memref<?x?xf32>, %start_indices: memref<?x?xi32>, %slice_sizes: memref<2xi32>, %out: memref<?x?x?xf32>) -> memref<?x?x?xf32> {
  // CHECK-NOT lmhlo
  // CHECK: scf.for
  "lmhlo.dynamic_gather"(%operand, %start_indices, %slice_sizes, %out) {dimension_numbers = #mhlo.gather<collapsed_slice_dims = [0], index_vector_dim = 2, offset_dims = [2], start_index_map = [0]>, indices_are_sorted = false} : (memref<?x?xf32>, memref<?x?xi32>, memref<2xi32>, memref<?x?x?xf32>) -> ()
  return %out : memref<?x?x?xf32>
}

// -----

// CHECK-LABEL: @concatenate
// CHECK-SAME: (%[[INPUT:.*]]: memref<?x?xf32>, %[[INPUT:.*]]: memref<?x?xf32>, %[[INPUT:.*]]: memref<?x?xf32>, %[[OUT:.*]]: memref<?x?xf32>) -> memref<?x?xf32>
func.func @concatenate(%input1: memref<?x?xf32>, %input2: memref<?x?xf32>, %input3: memref<?x?xf32>,%out: memref<?x?xf32>)->memref<?x?xf32>{
  // CHECK-NOT lmhlo
  // CHECK: scf.for
  "lmhlo.concatenate"(%input1, %input2, %input3, %out) {
    dimension = 1 : i64
  } : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return %out : memref<?x?xf32>
}

// -----

// CHECK-LABEL: @copy
func.func @copy(%operand: memref<?x?x?xf32>, %output: memref<?x?x?xf32>) -> memref<?x?x?xf32> {
  // CHECK-NOT lmhlo
  // CHECK: scf.for
  "lmhlo.copy"(%operand, %output) : (memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
  return %output : memref<?x?x?xf32>
}

// -----

// CHECK-LABEL: @naive_reduce
func.func @naive_reduce(%operand: memref<?x?x?xf32>, %init_value: memref<f32>, %output: memref<?x?xf32>) -> memref<?x?xf32> {
  // CHECK-NOT lmhlo
  // CHECK: scf.for
  "lmhlo.reduce"(%operand, %init_value, %output) ( {
    ^bb0(%arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<f32>):    // no predecessors
      %tmp = memref.alloc() {temp = true} : memref<f32>
      "lmhlo.add"(%arg1, %arg2, %tmp) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.copy"(%tmp, %arg3) : (memref<f32>, memref<f32>) -> ()
      memref.dealloc %tmp : memref<f32>
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<?x?x?xf32>, memref<f32>, memref<?x?xf32>) -> ()
  return %output : memref<?x?xf32>
}

// -----

// CHECK-LABEL: @dynamic_iota
func.func @dynamic_iota(%size: memref<2xi32>, %output: memref<?x?xi32>) -> memref<?x?xi32> {
  // CHECK-NOT lmhlo
  // CHECK: scf.for
  "lmhlo.dynamic_iota"(%size, %output) {iota_dimension = 1 : i64} : (memref<2xi32>, memref<?x?xi32>) -> ()
  return %output : memref<?x?xi32>
}

// -----

// CHECK-LABEL: @non_fusion_dynamic_broadcast_in_dim_gpu
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?xf32, "gpu">, %[[INPUT2:.*]]: memref<3xi32>, %[[OUT:.*]]: memref<?x?x?xf32, "gpu">) -> memref<?x?x?xf32, "gpu">
func.func @non_fusion_dynamic_broadcast_in_dim_gpu(%input1: memref<?xf32, "gpu">, %input2: memref<3xi32>, %out: memref<?x?x?xf32, "gpu">) -> (memref<?x?x?xf32, "gpu">) {
  // CHECK-NOT lmhlo
  // CHECK: scf.parallel
  "lmhlo.dynamic_broadcast_in_dim"(%input1, %input2, %out) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (memref<?xf32, "gpu">, memref<3xi32>, memref<?x?x?xf32, "gpu">) -> ()
  // CHECK: return %[[OUT]] : memref<?x?x?xf32, "gpu">
  return %out : memref<?x?x?xf32, "gpu">
}

// -----

// CHECK-LABEL: @basic_loop_fusion_misc_root
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?xf32>, %[[INPUT2:.*]]: memref<?xf32>, %[[INPUT3:.*]]: memref<3xi32>, %[[TMP_BUF:.*]]: memref<?xf32>, %[[OUT:.*]]: memref<?x?x?xf32>) -> memref<?x?x?xf32>
func.func @basic_loop_fusion_misc_root(%input1: memref<?xf32>, %input2: memref<?xf32>, %input3: memref<3xi32>, %tmp: memref<?xf32>, %out: memref<?x?x?xf32>) -> (memref<?x?x?xf32>) {
  // CHECK: "lmhlo.fusion"() ({
  "lmhlo.fusion"() ({
    // CHECK: lmhlo.add
    // CHECK-NOT lmhlo.dynamic_broadcast_in_dim
    // CHECK: scf.parallel
    "lmhlo.add"(%input1, %input2, %tmp) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
    "lmhlo.dynamic_broadcast_in_dim"(%tmp, %input3, %out) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (memref<?xf32>, memref<3xi32>, memref<?x?x?xf32>) -> ()
    // CHECK: "lmhlo.terminator"() : () -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {disc.fusion.name = "test", disc.fusion_type = "kLoop", disc.device = "gpu"} : () -> ()
  // CHECK: return %[[OUT]] : memref<?x?x?xf32>
  return %out : memref<?x?x?xf32>
}

// -----

// CHECK-LABEL: @multioutput_loop_fusion_with_dependency
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?xf32>, %[[INPUT2:.*]]: memref<3xi32>, %[[INPUT3:.*]]: memref<?x?x?xf32>, %[[TMP_BUF:.*]]: memref<?x?x?xf32>, %[[OUT1:.*]]: memref<?x?x?xf32>, %[[OUT2:.*]]: memref<?x?x?xf32>) -> (memref<?x?x?xf32>, memref<?x?x?xf32>)
func.func @multioutput_loop_fusion_with_dependency(%input1: memref<?xf32>, %input2: memref<3xi32>, %input3: memref<?x?x?xf32>, %tmp: memref<?x?x?xf32>, %out_1: memref<?x?x?xf32>, %out_2: memref<?x?x?xf32>) -> (memref<?x?x?xf32>, memref<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = memref.dim %input3, %c0 : memref<?x?x?xf32>
  %1 = memref.dim %input3, %c1 : memref<?x?x?xf32>
  %2 = memref.dim %input3, %c2 : memref<?x?x?xf32>
  %3 = arith.muli %1, %2 : index
  %casted_input3 = memref.reinterpret_cast %input3 to offset: [0], sizes: [%0, %1, %2], strides: [%3, %2, 1] {kDiscSymbolicDimAttr = [@S0, @S1, @S2]} : memref<?x?x?xf32> to memref<?x?x?xf32>
  %casted_tmp = memref.reinterpret_cast %tmp to offset: [0], sizes: [%0, %1, %2], strides: [%3, %2, 1] {kDiscSymbolicDimAttr = [@S0, @S1, @S2]} : memref<?x?x?xf32> to memref<?x?x?xf32>
  %casted_out1 = memref.reinterpret_cast %out_1 to offset: [0], sizes: [%0, %1, %2], strides: [%3, %2, 1] {kDiscSymbolicDimAttr = [@S0, @S1, @S2]} : memref<?x?x?xf32> to memref<?x?x?xf32>
  %casted_out2 = memref.reinterpret_cast %out_2 to offset: [0], sizes: [%0, %1, %2], strides: [%3, %2, 1] {kDiscSymbolicDimAttr = [@S0, @S1, @S2]} : memref<?x?x?xf32> to memref<?x?x?xf32>

  // CHECK: %[[CastedInput3:.*]] = memref.reinterpret_cast %[[INPUT3]]
  // CHECK: %[[CastedTmp:.*]] = memref.reinterpret_cast %[[TMP_BUF]]
  // CHECK: %[[CastedOut1:.*]] = memref.reinterpret_cast %[[OUT1]]
  // CHECK: %[[CastedOut2:.*]] = memref.reinterpret_cast %[[OUT2]]

  // CHECK: "lmhlo.fusion"() ({
  "lmhlo.fusion"() ({
    // CHECK: lmhlo.dynamic_broadcast_in_dim
    // CHECK: lmhlo.add
    // CHECK-NOT: lmhlo.multiply
    // CHECK: scf.parallel
    "lmhlo.dynamic_broadcast_in_dim"(%input1, %input2, %casted_tmp) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (memref<?xf32>, memref<3xi32>, memref<?x?x?xf32>) -> ()
    "lmhlo.add"(%casted_input3, %casted_tmp, %casted_out1) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
    "lmhlo.multiply"(%casted_input3, %casted_out1, %casted_out2) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
    // CHECK: "lmhlo.terminator"() : () -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "test", disc.fusion_type = "kLoop", disc.device = "gpu"} : () -> ()
  // CHECK: return %[[CastedOut1]], %[[CastedOut2]] : memref<?x?x?xf32>, memref<?x?x?xf32>
  return %casted_out1, %casted_out2 : memref<?x?x?xf32>, memref<?x?x?xf32>
}
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -9223372036854775808 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -9223372036854775808 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S2", value = -9223372036854775808 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}

// -----

// CHECK-LABEL: @multioutput_loop_fusion_without_dependency
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?xf32>, %[[INPUT2:.*]]: memref<3xi32>, %[[INPUT3:.*]]: memref<?x?x?xf32>, %[[TMP_BUF:.*]]: memref<?x?x?xf32>, %[[OUT1:.*]]: memref<?x?x?xf32>, %[[OUT2:.*]]: memref<?x?x?xf32>) -> (memref<?x?x?xf32>, memref<?x?x?xf32>)
func.func @multioutput_loop_fusion_without_dependency(%input1: memref<?xf32>, %input2: memref<3xi32>, %input3: memref<?x?x?xf32>, %tmp: memref<?x?x?xf32>, %out_1: memref<?x?x?xf32>, %out_2: memref<?x?x?xf32>) -> (memref<?x?x?xf32>, memref<?x?x?xf32>) {
  // CHECK: "lmhlo.fusion"() ({
  "lmhlo.fusion"() ({
    // CHECK: lmhlo.dynamic_broadcast_in_dim
    // CHECK-NOT: lmhlo.add
    // CHECK-NOT: lmhlo.multiply
    // CHECK: scf.parallel
    "lmhlo.dynamic_broadcast_in_dim"(%input1, %input2, %tmp) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (memref<?xf32>, memref<3xi32>, memref<?x?x?xf32>) -> ()
    "lmhlo.add"(%input3, %tmp, %out_1) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
    "lmhlo.multiply"(%input3, %tmp, %out_2) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
    // CHECK: "lmhlo.terminator"() : () -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "test", disc.fusion_type = "kLoop", disc.device = "gpu"} : () -> ()
  // CHECK: return %[[OUT1]], %[[OUT2]] : memref<?x?x?xf32>, memref<?x?x?xf32>
  return %out_1, %out_2 : memref<?x?x?xf32>, memref<?x?x?xf32>
}

// -----


// CHECK-LABEL: @kinput_col_reduce_schedule_1
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: memref<f32>) -> memref<?xf32>
func.func @kinput_col_reduce_schedule_1(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?xf32>, %arg3: memref<f32>) -> memref<?xf32> {
  // CHECK-NOT: lmhlo.reduce
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C512:.*]] = arith.constant 512 : index
  // CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
  // initializer for column reduction
  // CHECK: %[[OUTSIZE:.*]] = memref.dim %[[ARG2]], {{.*}} : memref<?xf32>
  // CHECK: scf.parallel (%[[INIT_ITER:.*]]) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
  // CHECK:   %[[DELINEARIZE:.*]] = "disc_shape.delinearize"(%[[INIT_ITER]]
  // CHECK:   %[[INIT_VALUE:.*]] = memref.load %[[ARG3]][] : memref<f32>
  // CHECK:   memref.store %[[INIT_VALUE]], %[[ARG2]][%[[DELINEARIZE]]] : memref<?xf32>
  // CHECK:   scf.yield
  // CHECK: }
  // CHECK: %[[ROWS:.*]] = memref.dim %[[ARG1]], %[[C0]] : memref<?x?xf32>
  // CHECK: %[[COLS:.*]] = memref.dim %[[ARG1]], %[[C1]] : memref<?x?xf32>
  // CHECK-DAG: %[[BLKS_PER_COL:.*]] = arith.ceildivui %[[COLS]], %[[C512]] : index
  // CHECK-DAG: %[[BLKS_PER_ROW:.*]] = arith.ceildivui %[[ROWS]], %[[C32]] : index
  // CHECK-DAG: %[[BLKS:.*]] = arith.muli %[[BLKS_PER_COL]], %[[BLKS_PER_ROW]] : index
  // CHECK: scf.parallel (%[[BLOCK_IDX:.*]], %[[THREAD_IDX:.*]]) = (%[[C0]], %[[C0]]) to (%[[BLKS]], %[[C512]]) step (%[[C1]], %[[C1]])
  // CHECK: %[[DATA:.*]] = memref.load %arg3[] : memref<f32>
  // CHECK: memref.atomic_rmw addf %[[TMP:.*]], %[[ARG2]]
  "lmhlo.fusion"() ({
    "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
    "lmhlo.reduce"(%arg1, %arg3, %arg2) ( {
    ^bb0(%arg4: memref<f32>, %arg5: memref<f32>, %arg6: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg4, %arg5, %arg6) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
    // CHECK: "lmhlo.terminator"() : () -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "main_kColReduction_reduce__4_1_0", disc_col_reduction_schedule_hint = 7 : i32, disc.fusion_type = "kColReduction", disc.device = "gpu"} : () -> ()
  // CHECK: return %[[ARG2]] : memref<?xf32>
  return %arg2 : memref<?xf32>
}

// -----

// CHECK-LABEL: @kinput_col_reduce_schedule_2
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: memref<f32>) -> memref<?xf32>
func.func @kinput_col_reduce_schedule_2(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?xf32>, %arg3: memref<f32>) -> memref<?xf32> {
  // CHECK-NOT: lmhlo.reduce
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
  // CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
  // CHECK-DAG: %[[C512:.*]] = arith.constant 512 : index
  // initializer for column reduction
  // CHECK: %[[OUTSIZE:.*]] = memref.dim %[[ARG2]], {{.*}} : memref<?xf32>
  // CHECK: scf.parallel (%[[INIT_ITER:.*]]) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
  // CHECK:   %[[DELINEARIZE:.*]] = "disc_shape.delinearize"(%[[INIT_ITER]]
  // CHECK:   %[[INIT_VALUE:.*]] = memref.load %[[ARG3]][] : memref<f32>
  // CHECK:   memref.store %[[INIT_VALUE]], %[[ARG2]][%[[DELINEARIZE]]] : memref<?xf32>
  // CHECK:   scf.yield
  // CHECK: }
  // CHECK-DAG: %[[ROWS:.*]] = memref.dim %[[ARG1]], %[[C0]] : memref<?x?xf32>
  // CHECK-DAG: %[[COLS:.*]] = memref.dim %[[ARG1]], %[[C1]] : memref<?x?xf32>
  // CHECK-DAG: %[[BLKS_PER_COL:.*]] = arith.ceildivui %[[COLS]], %[[C32]] : index
  // CHECK-DAG: %[[BLKS_PER_ROW:.*]] = arith.ceildivui %[[ROWS]], %[[C512]] : index
  // CHECK-DAG: %[[BLKS:.*]] = arith.muli %[[BLKS_PER_COL]], %[[BLKS_PER_ROW]] : index
  // CHECK: scf.parallel (%[[BLK_IDX:.*]], %[[THRD_IDX:.*]]) = (%[[C0]], %[[C0]]) to (%[[BLKS]], %[[C256]]) step (%[[C1]], %[[C1]]) {
  // CHECK: %[[DATA:.*]] = memref.load %arg3[] : memref<f32>
  // CHECK: gpu.barrier
  // CHECK: gpu.barrier
  // CHECK: gpu.barrier
  // CHECK: memref.atomic_rmw addf %[[TMP:.*]], %[[ARG2]]
  "lmhlo.fusion"() ({
    "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
    "lmhlo.reduce"(%arg1, %arg3, %arg2) ( {
    ^bb0(%arg4: memref<f32>, %arg5: memref<f32>, %arg6: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg4, %arg5, %arg6) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
    // CHECK: "lmhlo.terminator"() : () -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "main_kColReduction_reduce__4_1_0", disc_col_reduction_schedule_hint = 8 : i32, disc.fusion_type = "kColReduction", disc.device = "gpu"} : () -> ()
  // CHECK: return %[[ARG2]] : memref<?xf32>
  return %arg2 : memref<?xf32>
}

// -----

// CHECK-LABEL: @kinput_row_reduce_schedule_2_no_vec
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: memref<f32>) -> memref<?xf32>
func.func @kinput_row_reduce_schedule_2_no_vec(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?xf32>, %arg3: memref<f32>) -> memref<?xf32> {
  // CHECK-NOT: lmhlo.reduce
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[HIGHT:.*]] = memref.dim %[[ARG1]], %[[C0]] : memref<?x?xf32>
  // CHECK-DAG: %[[WIDTH:.*]] = memref.dim %[[ARG1]], %[[C1]] : memref<?x?xf32>
  // CHECK-DAG: %[[BLOCK_SIZE:.*]] = arith.constant 256 : index
  // CHECK-DAG: %[[ROW_PER_BLOCK:.*]] = arith.constant 8 : index
  // CHECK: scf.parallel (%[[H_IDX:.*]], %[[W_IDX:.*]]) = (%[[C0]], %[[C0]]) to (%[[HIGHT]], %[[BLOCK_SIZE]]) step (%[[ROW_PER_BLOCK]], %[[C1]])
  // CHECK: gpu.shuffle
  "lmhlo.fusion"() ({
    "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
    "lmhlo.reduce"(%arg1, %arg3, %arg2) ( {
    ^bb0(%arg4: memref<f32>, %arg5: memref<f32>, %arg6: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg4, %arg5, %arg6) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "kinput_row_reduce_schedule_2", disc_row_reduction_schedule_hint = 2 : i32, disc.fusion_type = "kRowReduction", disc.device = "gpu"} : () -> ()
  // CHECK: "lmhlo.terminator"() : () -> ()
  // CHECK: disc_row_reduction_schedule_hint = 2
  // CHECK: return %[[ARG2]] : memref<?xf32>
  return %arg2 : memref<?xf32>
}

// -----

// CHECK-LABEL: @kinput_row_reduce_schedule_2_vec2
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: memref<f32>) -> memref<?xf32>
func.func @kinput_row_reduce_schedule_2_vec2(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?xf32>, %arg3: memref<f32>) -> memref<?xf32> {
  // CHECK-NOT: lmhlo.reduce
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[HIGHT:.*]] = memref.dim %[[ARG1]], %[[C0]] : memref<?x?xf32>
  // CHECK-DAG: %[[WIDTH:.*]] = memref.dim %[[ARG1]], %[[C1]] : memref<?x?xf32>
  // CHECK-DAG: %[[BLOCK_SIZE:.*]] = arith.constant 256 : index
  // CHECK-DAG: %[[ROW_PER_BLOCK:.*]] = arith.constant 16 : index
  // CHECK: scf.parallel (%[[H_IDX:.*]], %[[W_IDX:.*]]) = (%[[C0]], %[[C0]]) to (%[[HIGHT]], %[[BLOCK_SIZE]]) step (%[[ROW_PER_BLOCK]], %[[C1]])
  // CHECK: gpu.shuffle
  // Adjacent store for vectorization optimization.
  // CHECK: memref.assume_alignment %[[ARG2]], 8 : memref<?xf32>
  // CHECK: memref.store %[[RES1:.*]], %[[ARG2]]
  // CHECK: memref.store %[[RES2:.*]], %[[ARG2]]
  "lmhlo.fusion"() ({
    "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
    "lmhlo.reduce"(%arg1, %arg3, %arg2) ( {
    ^bb0(%arg4: memref<f32>, %arg5: memref<f32>, %arg6: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg4, %arg5, %arg6) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "kinput_row_reduce_schedule_2", disc_row_reduction_schedule_hint = 2 : i32, disc_vectorize_or_tile_hint = 2 : i32, disc.fusion_type = "kRowReduction", disc.device = "gpu"} : () -> ()
  // CHECK: "lmhlo.terminator"() : () -> ()
  // CHECK: disc_row_reduction_schedule_hint = 2
  // CHECK: disc_vectorize_or_tile_hint = 2
  // CHECK: return %[[ARG2]] : memref<?xf32>
  return %arg2 : memref<?xf32>
}

// -----

// CHECK-LABEL: @kinput_row_reduce_schedule_1_no_vec
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: memref<f32>) -> memref<?xf32>
func.func @kinput_row_reduce_schedule_1_no_vec(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?xf32>, %arg3: memref<f32>) -> memref<?xf32> {
  // CHECK-NOT: lmhlo.reduce
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[BLOCK_SIZE:.*]] = arith.constant 256 : index
  // CHECK-DAG: %[[HIGHT:.*]] = memref.dim %[[ARG1]], %[[C0]] : memref<?x?xf32>
  // CHECK-DAG: %[[WIDTH:.*]] = memref.dim %[[ARG1]], %[[C1]] : memref<?x?xf32>
  // CHECK: scf.parallel (%[[H_IDX:.*]], %[[W_IDX:.*]]) = (%[[C0]], %[[C0]]) to (%[[HIGHT]], %[[BLOCK_SIZE]]) step (%[[C1]], %[[C1]])
  // CHECK: %[[SMEM:.*]] = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
  // CHECK: scf.for %[[W_LOCAL_IDX:.*]] = %[[TID:.*]] to %[[WIDTH]] step %[[BLOCK_SIZE]]
  // First round reduce.
  // CHECK: gpu.shuffle
  // CHECK: gpu.barrier
  // CHECK: memref.load %[[SMEM]]
  // Second round reduce.
  // CHECK: gpu.shuffle
  "lmhlo.fusion"() ({
    "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
    "lmhlo.reduce"(%arg1, %arg3, %arg2) ( {
    ^bb0(%arg4: memref<f32>, %arg5: memref<f32>, %arg6: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg4, %arg5, %arg6) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "kinput_row_reduce_schedule_1", disc_row_reduction_schedule_hint = 1 : i32, disc.fusion_type = "kRowReduction", disc.device = "gpu"} : () -> ()
  // CHECK: "lmhlo.terminator"() : () -> ()
  // CHECK: disc_row_reduction_schedule_hint = 1
  // CHECK: return %[[ARG2]] : memref<?xf32>
  return %arg2 : memref<?xf32>
}

// -----

// CHECK-LABEL: @kinput_row_reduce_schedule_1_vec2
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: memref<f32>) -> memref<?xf32>
func.func @kinput_row_reduce_schedule_1_vec2(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?xf32>, %arg3: memref<f32>) -> memref<?xf32> {
  // CHECK-NOT: lmhlo.reduce
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[BLOCK_SIZE:.*]] = arith.constant 256 : index
  // CHECK-DAG: %[[VEC_SIZE:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[HIGHT:.*]] = memref.dim %[[ARG1]], %[[C0]] : memref<?x?xf32>
  // CHECK-DAG: %[[WIDTH:.*]] = memref.dim %[[ARG1]], %[[C1]] : memref<?x?xf32>
  // CHECK: %[[BLOCK_NUMBER:.*]] = arith.divui %[[HIGHT]], %[[VEC_SIZE]] : index
  // CHECK: scf.parallel (%[[H_IDX:.*]], %[[W_IDX:.*]]) = (%[[C0]], %[[C0]]) to (%[[BLOCK_NUMBER]], %[[BLOCK_SIZE]]) step (%[[C1]], %[[C1]])
  // CHECK: %[[SMEM:.*]] = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
  // CHECK: scf.for %[[W_LOCAL_IDX:.*]] = %[[TID:.*]] to %[[WIDTH]] step %[[BLOCK_SIZE]]
  // First round reduce.
  // CHECK: gpu.shuffle
  // CHECK: gpu.barrier
  // CHECK: memref.load %[[SMEM]]
  // Second round reduce.
  // CHECK: gpu.shuffle
  // Adjacent store for vectorization optimization.
  // CHECK: memref.assume_alignment %[[ARG2]], 8 : memref<?xf32>
  // CHECK: memref.store %[[RES1:.*]], %[[ARG2]]
  // CHECK: memref.store %[[RES2:.*]], %[[ARG2]]
  "lmhlo.fusion"() ({
    "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
    "lmhlo.reduce"(%arg1, %arg3, %arg2) ( {
    ^bb0(%arg4: memref<f32>, %arg5: memref<f32>, %arg6: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg4, %arg5, %arg6) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "kinput_row_reduce_schedule_1", disc_row_reduction_schedule_hint = 1 : i32, disc_vectorize_or_tile_hint = 2 : i32, disc.fusion_type = "kRowReduction", disc.device = "gpu"} : () -> ()
  // CHECK: "lmhlo.terminator"() : () -> ()
  // CHECK: disc_row_reduction_schedule_hint = 1
  // CHECK: disc_vectorize_or_tile_hint = 2
  // CHECK: return %[[ARG2]] : memref<?xf32>
  return %arg2 : memref<?xf32>
}

// -----

// CHECK-LABEL: @kloop_dynamic_reshape
// CHECK-SAME: (%[[INPUT1:.*]]: memref<1xf64>, %[[INPUT2:.*]]: memref<2xi32>, %[[OUT1:.*]]: memref<?x?xf64>) -> memref<?x?xf64>
func.func @kloop_dynamic_reshape(
  %arg0: memref<1xf64>,
  %arg1: memref<2xi32>,
  %arg2: memref<?x?xf64>) -> (memref<?x?xf64>) {
      %0 = memref.alloc() : memref<f64>
      // CHECK: "lmhlo.fusion"() ({
      "lmhlo.fusion"() ({
        // CHECK: lmhlo.reshape
        // CHECK-NOT: lmhlo.dynamic_reshape
        // CHECK: scf.parallel
        "lmhlo.reshape"(%arg0, %0) {disc.device = "gpu"} : (memref<1xf64>, memref<f64>) -> ()
        "lmhlo.dynamic_reshape"(%0, %arg1, %arg2) {disc.device = "gpu"} : (memref<f64>, memref<2xi32>, memref<?x?xf64>) -> ()
        "lmhlo.terminator"() : () -> ()
      }) {disc.device = "gpu", disc.fusion.name = "main_kLoop_dynamic_reshape", disc.fusion.tag = "_vectile2", disc.fusion_type = "kLoop", disc_vectorize_or_tile_hint = 2 : i32} : () -> ()
    // CHECK: return %[[OUT1]] :  memref<?x?xf64>
    return  %arg2 :  memref<?x?xf64>
}

// -----

// CHECK-LABEL: @kstitch_small_output
func.func @kstitch_small_output(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?xf32>, %arg3: memref<f32>, %arg4: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %t0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32> to memref<?x?xf32>
  %t1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32> to memref<?x?xf32>
  %t2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%0], strides: [1] {kDiscSymbolicDimAttr = [@S0]} : memref<?xf32> to memref<?xf32>
  %t4 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0], strides: [1] {kDiscSymbolicDimAttr = [@S0]} : memref<?xf32> to memref<?xf32>

  "lmhlo.fusion"() ({
    "lmhlo.abs"(%t0, %t1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
    "lmhlo.reduce"(%t1, %arg3, %t2) ( {
    ^bb0(%arg5: memref<f32>, %arg6: memref<f32>, %arg7: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg5, %arg6, %arg7) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
    "lmhlo.abs"(%t2, %t4) : (memref<?xf32>, memref<?xf32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "kstitch_reduce_abs", disc.fusion_type = "kStitch", disc.device = "gpu"} : () -> ()
  // CHECK: lmhlo.fusion
  // CHECK: scf.parallel

  // CHECK: scf.for

  // It is the default schedule, which is block-wise. Thus it has two rounds of
  // shuffle.

  // round-one
  // CHECK: gpu.shuffle
  // CHECK: memref.store %[[INTER_RES:.*]], %[[SMEM_BUFFER1:.*]][
  // CHECK: gpu.barrier

  // round-two
  // CHECK: gpu.shuffle
  // CHECK: memref.store %[[INTER_RES:.*]], %[[SMEM_BUFFER2:.*]][
  // CHECK: gpu.barrier

  // Local loop for `abs`, which has different shape with the input of reduce.
  // CHECK: scf.for

  // Load from smem buffer.
  // CHECK: memref.load %[[SMEM_BUFFER2]]
  return %t4 : memref<?xf32>
}
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -9223372036854775808 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -9223372036854775808 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}

// -----

// MEMOPT-LABEL: @kloop_tile
func.func @kloop_tile(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) -> memref<?x?xf32> {
  %0 = memref.alloc() : memref<f32>
  "lmhlo.fusion"() ({
    "lmhlo.constant"(%0) {value = dense<1.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    "lmhlo.multiply"(%arg0, %arg1, %arg2) {disc.device = "gpu"} : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    "lmhlo.subtract"(%arg1, %arg2, %arg0) {disc.device = "gpu"} : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    "lmhlo.multiply"(%arg0, %arg2, %arg1) {disc.device = "gpu"} : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "gpu", disc.fusion.name = "main_kLoop", disc.fusion_type = "kLoop", disc_vectorize_or_tile_hint = 4 : i32} : () -> ()
  // MEMOPT-DAG: %[[C0:.*]] = arith.constant 0 : index
  // MEMOPT-DAG: %[[C1:.*]] = arith.constant 1 : index
  // MEMOPT-DAG: %[[C4:.*]] = arith.constant 4 : index
  // MEMOPT: lmhlo.fusion
  // MEMOPT: scf.parallel
  // MEMOPT: scf.for %[[ARG4:.*]] = %[[C0]] to %[[C4]] step %[[C1]]
  return %arg2 : memref<?x?xf32>
}

// -----

// MEMOPT-LABEL: @kstitch_independent_reduce_interleave
func.func @kstitch_independent_reduce_interleave(%arg0: memref<?x?xf32>,
    %arg1: memref<?x?xf32>, %arg2: memref<?xf32>, %arg3: memref<f32>,
    %arg4: memref<?xf32>, %argn1: memref<?x?xf32>, %argn2: memref<?xf32>) ->
    memref<?xf32> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %t0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32> to memref<?x?xf32>
  %t1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32> to memref<?x?xf32>
  %t2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%0], strides: [1] {kDiscSymbolicDimAttr = [@S0]} : memref<?xf32> to memref<?xf32>
  %t4 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0], strides: [1] {kDiscSymbolicDimAttr = [@S0]} : memref<?xf32> to memref<?xf32>
  %tn1 = memref.reinterpret_cast %argn1 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32> to memref<?x?xf32>
  %tn2 = memref.reinterpret_cast %argn2 to offset: [0], sizes: [%0], strides: [1] {kDiscSymbolicDimAttr = [@S0]} : memref<?xf32> to memref<?xf32>

  "lmhlo.fusion"() ({
    "lmhlo.abs"(%t0, %t1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
    "lmhlo.reduce"(%t1, %arg3, %t2) ( {
    ^bb0(%arg5: memref<f32>, %arg6: memref<f32>, %arg7: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg5, %arg6, %arg7) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
    "lmhlo.negate"(%t0, %tn1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
    "lmhlo.reduce"(%tn1, %arg3, %tn2) ( {
    ^bb0(%arg5: memref<f32>, %arg6: memref<f32>, %arg7: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg5, %arg6, %arg7) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
    "lmhlo.add"(%t2, %tn2, %t4) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "kstitch_independent_reduce", disc.fusion_type = "kStitch", disc.device = "gpu"} : () -> ()
  // MEMOPT: lmhlo.fusion
  // MEMOPT-COUNT-2: scf.parallel
  // MEMOPT: gpu.block_id x
  // MEMOPT: gpu.thread_id x

  // Interleaved in-thread reduce.
  // MEMOPT: scf.for
  // MEMOPT-COUNT-2: arith.addf
  // MEMOPT: scf.yield

  // Interleaved first-round reduce.
  // MEMOPT-COUNT-2: gpu.shuffle
  // MEMOPT-COUNT-2: arith.addf

  // MEMOPT: scf.if
  // MEMOPT-COUNT-2: memref.store
  // CHECK: gpu.barrier

  // Interleaved second-round reduce.
  // MEMOPT: scf.if
  // MEMOPT-COUNT-2: gpu.shuffle
  // MEMOPT-COUNT-2: arith.addf
  // CHECK: gpu.barrier

  // Finally, stitch with add op.
  // MEMOPT: scf.for
  // MEMOPT-COUNT-2: memref.load
  // MEMOPT: arith.addf
  // MEMOPT: memref.store

  return %t4 : memref<?xf32>
}
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -9223372036854775808 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -9223372036854775808 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}
