// RUN: disc-opt -pass-pipeline='builtin.func(disc-fusion{gpu-enabled=true fusion-strategy=base})' -split-input-file %s -o - | FileCheck %s --check-prefix=BASE
// RUN: disc-opt -pass-pipeline='builtin.func(disc-fusion{gpu-enabled=true fusion-strategy=stitch})' -split-input-file %s -o - | FileCheck %s --check-prefix=STITCH

// BASE-LABEL: @simple_kloop_fusion
// BASE-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?x?xf32, "gpu">) -> memref<?x?xf32, "gpu">
func @simple_kloop_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">,
                          %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?x?xf32, "gpu">) -> memref<?x?xf32, "gpu"> {
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.add"(%[[ARG1]], %[[ARG2]], %[[ARG3]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: })
  // BASE: return %[[ARG3]] : memref<?x?xf32, "gpu">
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  return %arg3 : memref<?x?xf32, "gpu">
}

// -----

// BASE-LABEL: @simple_multi_output_kloop_fusion
// BASE-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">)
func @simple_multi_output_kloop_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">,
                          %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) {
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.add"(%[[ARG1]], %[[ARG2]], %[[ARG3]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: })
  // BASE: return %[[ARG1]], %[[ARG3]] : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  return %arg1, %arg3 : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
}

// -----

// BASE-LABEL: @simple_multi_output_kloop_fusion_with_reorder
// BASE-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?x?xf32, "gpu">, %[[ARG4:.*]]: memref<2xindex, "cpu">, %[[ARG5:.*]]: memref<?x?xf32, "gpu">)
func @simple_multi_output_kloop_fusion_with_reorder(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">,
                          %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?x?xf32, "gpu">,
                          %arg4: memref<2xindex, "cpu">, %arg5:  memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) {
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.add"(%[[ARG1]], %[[ARG2]], %[[ARG3]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: })
  // BASE: "lmhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[ARG4]], %[[ARG5]])
  // BASE: return %[[ARG1]], %[[ARG3]], %[[ARG5]] : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.dynamic_broadcast_in_dim"(%arg1, %arg4, %arg5) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<?x?xf32, "gpu">, memref<2xindex, "cpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  return %arg1, %arg3, %arg5 : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
}

// -----

// BASE-LABEL: @same_num_elements_multi_output_kloop_fusion
// BASE-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<2xi64>, %[[ARG3:.*]]: memref<?x?x?xf32, "gpu">, %[[ARG4:.*]]: memref<?x?x?xf32, "gpu">, %[[ARG5:.*]]: memref<?x?x?xf32, "gpu">)
func @same_num_elements_multi_output_kloop_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">,
                          %arg2: memref<2xi64>, %arg3: memref<?x?x?xf32, "gpu">,
                          %arg4: memref<?x?x?xf32, "gpu">, %arg5:  memref<?x?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) {
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.dynamic_reshape"(%[[ARG1]], %[[ARG2]], %[[ARG3]])
  // BASE: "lmhlo.add"(%[[ARG3]], %[[ARG4]], %[[ARG5]]) : (memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) -> ()
  // BASE: })
  // BASE: return %[[ARG1]], %[[ARG5]] : memref<?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.dynamic_reshape"(%arg1, %arg2, %arg3) : (memref<?x?xf32, "gpu">, memref<2xi64>, memref<?x?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%arg3, %arg4, %arg5) : (memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) -> ()
  return %arg1, %arg5 : memref<?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">
}

// -----

// BASE-LABEL: @check_not_kloop_fusion
func @check_not_kloop_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) {
  // BASE-NOT: "lmhlo.fusion"
  "lmhlo.add"(%arg0, %arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.subtract"(%arg2, %arg2, %arg3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  return %arg1, %arg3: memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
}

// -----

// BASE-LABEL: @kloop_fusion_with_dealloc
// BASE-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">)
func @kloop_fusion_with_dealloc(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) {
  // BASE: %[[TMP3:.*]] = memref.alloc
  // BASE: %[[TMP5:.*]] = memref.alloc
  // BASE: %[[TMP9:.*]] = memref.alloc
  // BASE: %[[TMP13:.*]] = memref.alloc
  // BASE: %[[TMP16:.*]] = memref.alloc
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.add"(%[[ARG0]], %[[ARG1]], %[[TMP3]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.multiply"(%[[ARG0]], %[[ARG1]], %[[TMP5]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.abs"(%[[TMP3]], %[[TMP9]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.abs"(%[[TMP5]], %[[TMP13]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.multiply"(%[[TMP9]], %[[TMP13]], %[[TMP16]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: })
  // BASE: memref.dealloc %[[TMP3]] : memref<?x?xf32, "gpu">
  // BASE: memref.dealloc %[[TMP5]] : memref<?x?xf32, "gpu">
  // BASE: memref.dealloc %[[TMP13]] : memref<?x?xf32, "gpu">
  // BASE: return %[[TMP9]], %[[TMP16]] : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = shape.shape_of %arg0 : memref<?x?xf32, "gpu"> -> tensor<2xindex, "cpu">
  %1 = tensor.extract %0[%c0] : tensor<2xindex, "cpu">
  %2 = tensor.extract %0[%c1] : tensor<2xindex, "cpu">
  %3 = memref.alloc(%1, %2) : memref<?x?xf32, "gpu">
  "lmhlo.add"(%arg0, %arg1, %3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  %4 = memref.alloc(%1, %2) : memref<?x?xf32, "gpu">
  "lmhlo.multiply"(%arg0, %arg1, %4) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  %5 = shape.shape_of %3 : memref<?x?xf32, "gpu"> -> tensor<2xindex, "cpu">
  %6 = tensor.extract %5[%c0] : tensor<2xindex, "cpu">
  %7 = tensor.extract %5[%c1] : tensor<2xindex, "cpu">
  %8 = memref.alloc(%6, %7) : memref<?x?xf32, "gpu">
  "lmhlo.abs"(%3, %8) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  memref.dealloc %3 : memref<?x?xf32, "gpu">
  %9 = shape.shape_of %4 : memref<?x?xf32, "gpu"> -> tensor<2xindex, "cpu">
  %10 = tensor.extract %9[%c0] : tensor<2xindex, "cpu">
  %11 = tensor.extract %9[%c1] : tensor<2xindex, "cpu">
  %12 = memref.alloc(%10, %11) : memref<?x?xf32, "gpu">
  "lmhlo.abs"(%4, %12) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  memref.dealloc %4 : memref<?x?xf32, "gpu">
  %13 = shape.shape_of %8 : memref<?x?xf32, "gpu"> -> tensor<2xindex, "cpu">
  %14 = tensor.extract %13[%c0] : tensor<2xindex, "cpu">
  %15 = tensor.extract %13[%c1] : tensor<2xindex, "cpu">
  %16 = memref.alloc(%14, %15) : memref<?x?xf32, "gpu">
  "lmhlo.multiply"(%8, %12, %16) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  memref.dealloc %12 : memref<?x?xf32, "gpu">
  return %8, %16 : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
}

// -----

// BASE-LABEL: @simple_kinput
// BASE-SAME: %[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?xf32, "gpu">, %[[ARG3:.*]]: memref<f32, "gpu">
func @simple_kinput(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?xf32, "gpu">, %init: memref<f32, "gpu">) -> memref<?xf32, "gpu"> {
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.reduce"(%[[ARG1]], %[[ARG3]], %[[ARG2]]) ({
  // BASE: })
  // BASE: return %[[ARG2]] : memref<?xf32, "gpu">
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%arg1, %init, %arg2) ({
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[0]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  return %arg2: memref<?xf32, "gpu">
}

// -----

// BASE-LABEL: @multi_output_kinput
// BASE-SAME: %[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?xf32, "gpu">, %[[ARG3:.*]]: memref<f32, "gpu">
func @multi_output_kinput(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?xf32, "gpu">, %init: memref<f32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?xf32, "gpu">) {
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.reduce"(%[[ARG1]], %[[ARG3]], %[[ARG2]]) ({
  // BASE: })
  // BASE: return %[[ARG1]], %[[ARG2]] : memref<?x?xf32, "gpu">, memref<?xf32, "gpu">
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%arg1, %init, %arg2) ({
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[0]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  return %arg1, %arg2: memref<?x?xf32, "gpu">, memref<?xf32, "gpu">
}

// -----

// BASE-LABEL: @row_red_and_row_red_kinput
// BASE-SAME: %[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?xf32, "gpu">, %[[ARG4:.*]]: memref<?xf32, "gpu">, %[[ARG5:.*]]: memref<?x?xf32, "gpu">, %[[ARG6:.*]]: memref<f32, "gpu">
func @row_red_and_row_red_kinput(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?xf32, "gpu">, %arg4: memref<?xf32, "gpu">, %arg5: memref<?x?xf32, "gpu">, %init: memref<f32, "gpu">) -> (memref<?xf32, "gpu">, memref<?xf32, "gpu">) {
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.add"(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.abs"(%[[ARG2]], %[[ARG5]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.reduce"(%[[ARG5]], %[[ARG6]], %[[ARG3]]) ({
  // BASE: "lmhlo.reduce"(%[[ARG2]], %[[ARG6]], %[[ARG4]]) ({
  // BASE: })
  // BASE: return %[[ARG3]], %[[ARG4]] : memref<?xf32, "gpu">, memref<?xf32, "gpu">
  "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.abs"(%arg2, %arg5) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%arg5, %init, %arg3) ({
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[1]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%arg2, %init, %arg4) ({
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[1]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  return %arg3, %arg4: memref<?xf32, "gpu">, memref<?xf32, "gpu">
}

// -----

// BASE-LABEL: @row_red_and_col_red_kinput
// BASE-SAME: %[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?xf32, "gpu">, %[[ARG4:.*]]: memref<?xf32, "gpu">, %[[ARG5:.*]]: memref<?x?xf32, "gpu">, %[[ARG6:.*]]: memref<f32, "gpu">
func @row_red_and_col_red_kinput(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?xf32, "gpu">, %arg4: memref<?xf32, "gpu">, %arg5: memref<?x?xf32, "gpu">, %init: memref<f32, "gpu">) -> (memref<?xf32, "gpu">, memref<?xf32, "gpu">) {
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.add"(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.abs"(%[[ARG2]], %[[ARG5]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.reduce"(%[[ARG5]], %[[ARG6]], %[[ARG3]]) ({
  // BASE: "lmhlo.reduce"(%[[ARG2]], %[[ARG6]], %[[ARG4]]) ({
  // BASE: })
  // BASE: return %[[ARG3]], %[[ARG4]] : memref<?xf32, "gpu">, memref<?xf32, "gpu">
  "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.abs"(%arg2, %arg5) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%arg5, %init, %arg3) ({
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[1]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%arg2, %init, %arg4) ({
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[0]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  return %arg3, %arg4: memref<?xf32, "gpu">, memref<?xf32, "gpu">
}

// -----

// BASE-LABEL: @reduce_should_not_have_consumer_in_the_fusion
// BASE-SAME: %[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">
func @reduce_should_not_have_consumer_in_the_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">)
-> (memref<?x?xf32, "gpu">, memref<?xf32, "gpu">) {
  // BASE: %[[TMP4:.*]] = memref.alloc
  // BASE: %[[TMP7:.*]] = memref.alloc
  // BASE: %[[TMP8:.*]] = memref.alloc
  // BASE: %[[TMP9:.*]] = memref.alloc
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.add"(%[[ARG0]], %[[ARG1]], %[[TMP4]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.subtract"(%[[ARG0]], %[[TMP4]], %[[TMP7]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.constant"(%[[TMP8]]) {value = dense<0.000000e+00> : tensor<f32, "gpu">} : (memref<f32, "gpu">) -> ()
  // BASE: "lmhlo.reduce"(%[[TMP7]], %[[TMP8]], %[[TMP9]]) ({
  // BASE: })
  // BASE: memref.dealloc %[[TMP4]] : memref<?x?xf32, "gpu">
  // BASE: memref.dealloc %[[TMP8]] : memref<f32, "gpu">
  // BASE: %[[TMP12:.*]] = memref.alloc
  // BASE: "lmhlo.add"(%[[TMP9]], %[[TMP9]], %[[TMP12]]) : (memref<?xf32, "gpu">, memref<?xf32, "gpu">, memref<?xf32, "gpu">) -> ()
  // BASE: memref.dealloc %[[TMP9]] : memref<?xf32, "gpu">
  // BASE: return %[[TMP7]], %[[TMP12]] : memref<?x?xf32, "gpu">, memref<?xf32, "gpu">
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = shape.shape_of %arg0 : memref<?x?xf32, "gpu"> -> tensor<2xindex, "cpu">
  %1 = tensor.extract %0[%c0] : tensor<2xindex, "cpu">
  %2 = tensor.extract %0[%c1] : tensor<2xindex, "cpu">
  %3 = memref.alloc(%1, %2) : memref<?x?xf32, "gpu">
  "lmhlo.add"(%arg0, %arg1, %3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  %4 = shape.shape_of %arg0 : memref<?x?xf32, "gpu"> -> tensor<2xindex, "cpu">
  %5 = tensor.extract %4[%c0] : tensor<2xindex, "cpu">
  %6 = tensor.extract %4[%c1] : tensor<2xindex, "cpu">
  %7 = memref.alloc(%5, %6) : memref<?x?xf32, "gpu">
  "lmhlo.subtract"(%arg0, %3, %7) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  memref.dealloc %3 : memref<?x?xf32, "gpu">
  %8 = memref.alloc() : memref<f32, "gpu">
  "lmhlo.constant"(%8) {value = dense<0.000000e+00> : tensor<f32, "gpu">} : (memref<f32, "gpu">) -> ()
  %9 = memref.alloc(%5) : memref<?xf32, "gpu">
  "lmhlo.reduce"(%7, %8, %9) ({
  ^bb0(%arg2: memref<f32, "gpu">, %arg3: memref<f32, "gpu">, %arg4: memref<f32, "gpu">):  // no predecessors
    "lmhlo.add"(%arg2, %arg3, %arg4) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  memref.dealloc %8 : memref<f32, "gpu">
  %10 = shape.shape_of %9 : memref<?xf32, "gpu"> -> tensor<1xindex, "cpu">
  %11 = tensor.extract %10[%c0] : tensor<1xindex, "cpu">
  %12 = memref.alloc(%11) : memref<?xf32, "gpu">
  "lmhlo.add"(%9, %9, %12) : (memref<?xf32, "gpu">, memref<?xf32, "gpu">, memref<?xf32, "gpu">) -> ()
  memref.dealloc %9 : memref<?xf32, "gpu">
  return %7, %12 : memref<?x?xf32, "gpu">, memref<?xf32, "gpu">
}

// -----

// BASE-LABEL: @const_should_not_be_output
func @const_should_not_be_output(%arg0: memref<f32, "gpu">) -> (memref<f32, "gpu">, memref<f32, "gpu">) {
  // BASE-NOT: lmhlo.fusion
  %0 = memref.alloc() : memref<f32, "gpu">
  "lmhlo.constant"(%0) {value = dense<1.000000e+00> : tensor<f32, "gpu">} : (memref<f32, "gpu">) -> ()
  %1 = memref.alloc() : memref<f32, "gpu">
  "lmhlo.add"(%arg0, %0, %1) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
  return %0, %1 : memref<f32, "gpu">, memref<f32, "gpu">
}

// -----

// BASE-LABEL: @fusion_clamp_with_scalar_min_max
func @fusion_clamp_with_scalar_min_max(%arg0: memref<f32, "gpu">, %arg1: memref<f32, "gpu">,
                                       %arg2: memref<f32, "gpu">, %arg3: memref<?x?xf32, "gpu">) ->(memref<f32, "gpu">,  memref<?x?xf32, "gpu">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %shape = shape.shape_of %arg3 : memref<?x?xf32, "gpu"> -> tensor<2xindex, "cpu">
  %dim0 = tensor.extract %shape[%c0] : tensor<2xindex, "cpu">
  %dim1 = tensor.extract %shape[%c1] : tensor<2xindex, "cpu">
  %0 = memref.alloc() : memref<f32, "gpu">
  %1 = memref.alloc() : memref<f32, "gpu">
  %2 = memref.alloc(%dim0, %dim1) : memref<?x?xf32, "gpu">
  // BASE: lmhlo.fusion
  "lmhlo.abs"(%arg1, %0) : (memref<f32, "gpu">, memref<f32, "gpu">) -> ()
  "lmhlo.add"(%arg0, %0, %1) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
  // BASE: }) {
  // BASE-SAME: disc.fusion.name
  // BASE: lmhlo.clamp
  "lmhlo.clamp"(%0, %arg3, %arg2, %2) : (memref<f32, "gpu">, memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  memref.dealloc %0 : memref<f32, "gpu">
  return %1, %2 : memref<f32, "gpu">, memref<?x?xf32, "gpu">
}

// -----

// BASE-LABEL: @fusion_clamp_with_multidim_min_max
func @fusion_clamp_with_multidim_min_max(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">,
                                         %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %shape = shape.shape_of %arg3 : memref<?x?xf32, "gpu"> -> tensor<2xindex, "cpu">
  %dim0 = tensor.extract %shape[%c0] : tensor<2xindex, "cpu">
  %dim1 = tensor.extract %shape[%c1] : tensor<2xindex, "cpu">
  %0 = memref.alloc(%dim0, %dim1) : memref<?x?xf32, "gpu">
  %1 = memref.alloc(%dim0, %dim1) : memref<?x?xf32, "gpu">
  %2 = memref.alloc(%dim0, %dim1) : memref<?x?xf32, "gpu">
  // BASE: lmhlo.fusion
  // BASE: lmhlo.clamp
  "lmhlo.abs"(%arg1, %0) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%arg0, %0, %1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.clamp"(%0, %arg3, %arg2, %2) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: }) {
  // BASE-SAME: disc.fusion.name
  memref.dealloc %0 : memref<?x?xf32, "gpu">
  return %1, %2 : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
}

// -----

// STITCH-LABEL: @kstitch_fusion_mean
func @kstitch_fusion_mean(%arg0: memref<?x?x?xf32, "gpu">) -> memref<?x?xf32, "gpu"> attributes {tf.entry_function = {input_placements = "gpu", inputs = "input0", output_placements = "gpu", outputs = "output0"}} {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.alloc() : memref<f32, "gpu">
  "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32, "gpu">) -> ()
  %1 = memref.dim %arg0, %c2 : memref<?x?x?xf32, "gpu">
  %2 = arith.index_cast %1 : index to i32
  %3 = memref.dim %arg0, %c0 : memref<?x?x?xf32, "gpu">
  %4 = arith.index_cast %3 : index to i32
  %5 = memref.dim %arg0, %c1 : memref<?x?x?xf32, "gpu">
  %6 = arith.index_cast %5 : index to i32
  %7 = arith.muli %4, %6 : i32
  %8 = memref.alloca() : memref<2xi32, "cpu">
  memref.store %7, %8[%c0] : memref<2xi32, "cpu">
  memref.store %2, %8[%c1] : memref<2xi32, "cpu">
  %11 = arith.index_cast %7 : i32 to index
  %16 = memref.alloc(%11, %1) : memref<?x?xf32, "gpu">
  "lmhlo.dynamic_reshape"(%arg0, %8, %16) {disc.device = "gpu"} : (memref<?x?x?xf32, "gpu">, memref<2xi32, "cpu">, memref<?x?xf32, "gpu">) -> ()
  %17 = memref.alloc(%11) : memref<?xf32, "gpu">
  "lmhlo.reduce"(%16, %0, %17) ({
  ^bb0(%arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<f32>):  // no predecessors
    "lmhlo.add"(%arg1, %arg2, %arg3) {disc.device = "gpu"} : (memref<f32>, memref<f32>, memref<f32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {dimensions = dense<1> : tensor<1xi64>, disc.device = "gpu"} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  %18 = memref.alloca() : memref<2xi32, "cpu">
  memref.store %4, %18[%c0] : memref<2xi32, "cpu">
  memref.store %6, %18[%c1] : memref<2xi32, "cpu">
  %26 = memref.alloc(%3, %5) : memref<?x?xf32, "gpu">
  "lmhlo.dynamic_reshape"(%17, %18, %26) {disc.device = "gpu"} : (memref<?xf32, "gpu">, memref<2xi32, "cpu">, memref<?x?xf32, "gpu">) -> ()
  %27 = arith.index_cast %1 : index to i64
  %28 = memref.alloca() : memref<1xi64, "cpu">
  memref.store %27, %28[%c0] : memref<1xi64, "cpu">
  %29 = memref.alloc() : memref<1xi64, "gpu">
  "lmhlo_disc.h2d"(%28, %29) : (memref<1xi64, "cpu">, memref<1xi64, "gpu">) -> ()
  %30 = memref.alloc() : memref<i64, "gpu">
  "lmhlo.reshape"(%29, %30) {disc.device = "gpu"} : (memref<1xi64, "gpu">, memref<i64, "gpu">) -> ()
  %31 = memref.alloc() : memref<f32, "gpu">
  "lmhlo.convert"(%30, %31) {disc.device = "gpu"} : (memref<i64, "gpu">, memref<f32, "gpu">) -> ()
  %32 = memref.alloca() : memref<2xindex, "cpu">
  memref.store %3, %32[%c0] : memref<2xindex, "cpu">
  memref.store %5, %32[%c1] : memref<2xindex, "cpu">
  %33 = memref.alloc(%3, %5) : memref<?x?xf32, "gpu">
  "lmhlo.dynamic_broadcast_in_dim"(%31, %32, %33) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "gpu"} : (memref<f32, "gpu">, memref<2xindex, "cpu">, memref<?x?xf32, "gpu">) -> ()
  %34 = memref.alloc(%3, %5) : memref<?x?xf32, "gpu">
  "lmhlo.divide"(%26, %33, %34) {disc.device = "gpu"} : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // Make sure there is one and only one kStitch fusion.
  // STITCH:      disc.fusion.name
  // STITCH-SAME: disc.fusion_type = "kStitch"
  // STITCH-NOT:  disc.fusion.name
  return %34 : memref<?x?xf32, "gpu">
}