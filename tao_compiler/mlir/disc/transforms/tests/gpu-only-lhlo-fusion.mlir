// RUN: disc-opt --disc-fusion -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: @simple_kloop_fusion
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?x?xf32, "gpu">) -> memref<?x?xf32, "gpu">
func @simple_kloop_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">,
                          %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?x?xf32, "gpu">) -> memref<?x?xf32, "gpu"> {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.add"(%[[ARG1]], %[[ARG2]], %[[ARG3]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: })
  // CHECK: return %[[ARG3]] : memref<?x?xf32, "gpu">
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  return %arg3 : memref<?x?xf32, "gpu">
}

// -----

// CHECK-LABEL: @simple_multi_output_kloop_fusion
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">)
func @simple_multi_output_kloop_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">,
                          %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.add"(%[[ARG1]], %[[ARG2]], %[[ARG3]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: })
  // CHECK: return %[[ARG1]], %[[ARG3]] : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  return %arg1, %arg3 : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
}

// -----

// CHECK-LABEL: @simple_multi_output_kloop_fusion_with_reorder
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?x?xf32, "gpu">, %[[ARG4:.*]]: memref<2xindex, "cpu">, %[[ARG5:.*]]: memref<?x?xf32, "gpu">)
func @simple_multi_output_kloop_fusion_with_reorder(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">,
                          %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?x?xf32, "gpu">,
                          %arg4: memref<2xindex, "cpu">, %arg5:  memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.add"(%[[ARG1]], %[[ARG2]], %[[ARG3]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: })
  // CHECK: "lmhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[ARG4]], %[[ARG5]])
  // CHECK: return %[[ARG1]], %[[ARG3]], %[[ARG5]] : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.dynamic_broadcast_in_dim"(%arg1, %arg4, %arg5) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<?x?xf32, "gpu">, memref<2xindex, "cpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  return %arg1, %arg3, %arg5 : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
}

// -----

// CHECK-LABEL: @same_num_elements_multi_output_kloop_fusion
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<2xi64>, %[[ARG3:.*]]: memref<?x?x?xf32, "gpu">, %[[ARG4:.*]]: memref<?x?x?xf32, "gpu">, %[[ARG5:.*]]: memref<?x?x?xf32, "gpu">)
func @same_num_elements_multi_output_kloop_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">,
                          %arg2: memref<2xi64>, %arg3: memref<?x?x?xf32, "gpu">,
                          %arg4: memref<?x?x?xf32, "gpu">, %arg5:  memref<?x?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.dynamic_reshape"(%[[ARG1]], %[[ARG2]], %[[ARG3]])
  // CHECK: "lmhlo.add"(%[[ARG3]], %[[ARG4]], %[[ARG5]]) : (memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) -> ()
  // CHECK: })
  // CHECK: return %[[ARG1]], %[[ARG5]] : memref<?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.dynamic_reshape"(%arg1, %arg2, %arg3) : (memref<?x?xf32, "gpu">, memref<2xi64>, memref<?x?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%arg3, %arg4, %arg5) : (memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) -> ()
  return %arg1, %arg5 : memref<?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">
}

// -----

// CHECK-LABEL: @check_not_kloop_fusion
func @check_not_kloop_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) {
  // CHECK-NOT: "lmhlo.fusion"
  "lmhlo.add"(%arg0, %arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.subtract"(%arg2, %arg2, %arg3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  return %arg1, %arg3: memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
}

// -----

// CHECK-LABEL: @kloop_fusion_with_dealloc
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">)
func @kloop_fusion_with_dealloc(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) {
  // CHECK: %[[TMP3:.*]] = memref.alloc
  // CHECK: %[[TMP5:.*]] = memref.alloc
  // CHECK: %[[TMP9:.*]] = memref.alloc
  // CHECK: %[[TMP13:.*]] = memref.alloc
  // CHECK: %[[TMP16:.*]] = memref.alloc
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.add"(%[[ARG0]], %[[ARG1]], %[[TMP3]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.multiply"(%[[ARG0]], %[[ARG1]], %[[TMP5]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.abs"(%[[TMP3]], %[[TMP9]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.abs"(%[[TMP5]], %[[TMP13]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.multiply"(%[[TMP9]], %[[TMP13]], %[[TMP16]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: })
  // CHECK: memref.dealloc %[[TMP3]] : memref<?x?xf32, "gpu">
  // CHECK: memref.dealloc %[[TMP5]] : memref<?x?xf32, "gpu">
  // CHECK: memref.dealloc %[[TMP13]] : memref<?x?xf32, "gpu">
  // CHECK: return %[[TMP9]], %[[TMP16]] : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
  %c0 = constant 0 : index
  %c1 = constant 1 : index
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

// CHECK-LABEL: @simple_kinput
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?xf32, "gpu">, %[[ARG3:.*]]: memref<f32, "gpu">
func @simple_kinput(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?xf32, "gpu">, %init: memref<f32, "gpu">) -> memref<?xf32, "gpu"> {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.reduce"(%[[ARG1]], %[[ARG3]], %[[ARG2]]) ( {
  // CHECK: })
  // CHECK: return %[[ARG2]] : memref<?xf32, "gpu">
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%arg1, %init, %arg2) ( {
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[0]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  return %arg2: memref<?xf32, "gpu">
}

// -----

// CHECK-LABEL: @multi_output_kinput
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?xf32, "gpu">, %[[ARG3:.*]]: memref<f32, "gpu">
func @multi_output_kinput(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?xf32, "gpu">, %init: memref<f32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?xf32, "gpu">) {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.reduce"(%[[ARG1]], %[[ARG3]], %[[ARG2]]) ( {
  // CHECK: })
  // CHECK: return %[[ARG1]], %[[ARG2]] : memref<?x?xf32, "gpu">, memref<?xf32, "gpu">
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%arg1, %init, %arg2) ( {
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[0]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  return %arg1, %arg2: memref<?x?xf32, "gpu">, memref<?xf32, "gpu">
}

// -----

// CHECK-LABEL: @row_red_and_row_red_kinput
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?xf32, "gpu">, %[[ARG4:.*]]: memref<?xf32, "gpu">, %[[ARG5:.*]]: memref<?x?xf32, "gpu">, %[[ARG6:.*]]: memref<f32, "gpu">
func @row_red_and_row_red_kinput(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?xf32, "gpu">, %arg4: memref<?xf32, "gpu">, %arg5: memref<?x?xf32, "gpu">, %init: memref<f32, "gpu">) -> (memref<?xf32, "gpu">, memref<?xf32, "gpu">) {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.add"(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.abs"(%[[ARG2]], %[[ARG5]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.reduce"(%[[ARG5]], %[[ARG6]], %[[ARG3]]) ( {
  // CHECK: "lmhlo.reduce"(%[[ARG2]], %[[ARG6]], %[[ARG4]]) ( {
  // CHECK: })
  // CHECK: return %[[ARG3]], %[[ARG4]] : memref<?xf32, "gpu">, memref<?xf32, "gpu">
  "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.abs"(%arg2, %arg5) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%arg5, %init, %arg3) ( {
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[1]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%arg2, %init, %arg4) ( {
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[1]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  return %arg3, %arg4: memref<?xf32, "gpu">, memref<?xf32, "gpu">
}

// -----

// CHECK-LABEL: @row_red_and_col_red_kinput
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?xf32, "gpu">, %[[ARG4:.*]]: memref<?xf32, "gpu">, %[[ARG5:.*]]: memref<?x?xf32, "gpu">, %[[ARG6:.*]]: memref<f32, "gpu">
func @row_red_and_col_red_kinput(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?xf32, "gpu">, %arg4: memref<?xf32, "gpu">, %arg5: memref<?x?xf32, "gpu">, %init: memref<f32, "gpu">) -> (memref<?xf32, "gpu">, memref<?xf32, "gpu">) {
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.add"(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.abs"(%[[ARG2]], %[[ARG5]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.reduce"(%[[ARG5]], %[[ARG6]], %[[ARG3]]) ( {
  // CHECK: "lmhlo.reduce"(%[[ARG2]], %[[ARG6]], %[[ARG4]]) ( {
  // CHECK: })
  // CHECK: return %[[ARG3]], %[[ARG4]] : memref<?xf32, "gpu">, memref<?xf32, "gpu">
  "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.abs"(%arg2, %arg5) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%arg5, %init, %arg3) ( {
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[1]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%arg2, %init, %arg4) ( {
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[0]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  return %arg3, %arg4: memref<?xf32, "gpu">, memref<?xf32, "gpu">
}

// -----

// CHECK-LABEL: @reduce_should_not_have_consumer_in_the_fusion
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">
func @reduce_should_not_have_consumer_in_the_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">)
-> (memref<?x?xf32, "gpu">, memref<?xf32, "gpu">) {
  // CHECK: %[[TMP4:.*]] = memref.alloc
  // CHECK: %[[TMP7:.*]] = memref.alloc
  // CHECK: %[[TMP8:.*]] = memref.alloc
  // CHECK: %[[TMP9:.*]] = memref.alloc
  // CHECK: "lmhlo.fusion"() ( {
  // CHECK: "lmhlo.add"(%[[ARG0]], %[[ARG1]], %[[TMP4]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.subtract"(%[[ARG0]], %[[TMP4]], %[[TMP7]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: "lmhlo.constant"(%[[TMP8]]) {value = dense<0.000000e+00> : tensor<f32, "gpu">} : (memref<f32, "gpu">) -> ()
  // CHECK: "lmhlo.reduce"(%[[TMP7]], %[[TMP8]], %[[TMP9]]) ( {
  // CHECK: })
  // CHECK: memref.dealloc %[[TMP4]] : memref<?x?xf32, "gpu">
  // CHECK: memref.dealloc %[[TMP8]] : memref<f32, "gpu">
  // CHECK: %[[TMP12:.*]] = memref.alloc
  // CHECK: "lmhlo.add"(%[[TMP9]], %[[TMP9]], %[[TMP12]]) : (memref<?xf32, "gpu">, memref<?xf32, "gpu">, memref<?xf32, "gpu">) -> ()
  // CHECK: memref.dealloc %[[TMP9]] : memref<?xf32, "gpu">
  // CHECK: return %[[TMP7]], %[[TMP12]] : memref<?x?xf32, "gpu">, memref<?xf32, "gpu">
  %c1 = constant 1 : index
  %c0 = constant 0 : index
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
  "lmhlo.reduce"(%7, %8, %9) ( {
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

// CHECK-LABEL: @const_should_not_be_output
func @const_should_not_be_output(%arg0: memref<f32, "gpu">) -> (memref<f32, "gpu">, memref<f32, "gpu">) {
  // CHECK-NOT: lmhlo.fusion
  %0 = memref.alloc() : memref<f32, "gpu">
  "lmhlo.constant"(%0) {value = dense<1.000000e+00> : tensor<f32, "gpu">} : (memref<f32, "gpu">) -> ()
  %1 = memref.alloc() : memref<f32, "gpu">
  "lmhlo.add"(%arg0, %0, %1) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
  return %0, %1 : memref<f32, "gpu">, memref<f32, "gpu">
}

// -----

// CHECK-LABEL: @fusion_clamp_with_scalar_min_max
func @fusion_clamp_with_scalar_min_max(%arg0: memref<f32, "gpu">, %arg1: memref<f32, "gpu">,
                                       %arg2: memref<f32, "gpu">, %arg3: memref<?x?xf32, "gpu">) ->(memref<f32, "gpu">,  memref<?x?xf32, "gpu">) {
  %c0 = constant 0 : index
  %c1 = constant 0 : index
  %shape = shape.shape_of %arg3 : memref<?x?xf32, "gpu"> -> tensor<2xindex, "cpu">
  %dim0 = tensor.extract %shape[%c0] : tensor<2xindex, "cpu">
  %dim1 = tensor.extract %shape[%c1] : tensor<2xindex, "cpu">
  %0 = memref.alloc() : memref<f32, "gpu">
  %1 = memref.alloc() : memref<f32, "gpu">
  %2 = memref.alloc(%dim0, %dim1) : memref<?x?xf32, "gpu">
  // CHECK: lmhlo.fusion
  "lmhlo.abs"(%arg1, %0) : (memref<f32, "gpu">, memref<f32, "gpu">) -> ()
  "lmhlo.add"(%arg0, %0, %1) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
  // CHECK: }) {
  // CHECK-SAME: disc.fusion.name
  // CHECK: lmhlo.clamp
  "lmhlo.clamp"(%0, %arg3, %arg2, %2) : (memref<f32, "gpu">, memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  memref.dealloc %0 : memref<f32, "gpu">
  return %1, %2 : memref<f32, "gpu">, memref<?x?xf32, "gpu">
}

// -----

// CHECK-LABEL: @fusion_clamp_with_multidim_min_max
func @fusion_clamp_with_multidim_min_max(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">,
                                         %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) {
  %c0 = constant 0 : index
  %c1 = constant 0 : index
  %shape = shape.shape_of %arg3 : memref<?x?xf32, "gpu"> -> tensor<2xindex, "cpu">
  %dim0 = tensor.extract %shape[%c0] : tensor<2xindex, "cpu">
  %dim1 = tensor.extract %shape[%c1] : tensor<2xindex, "cpu">
  %0 = memref.alloc(%dim0, %dim1) : memref<?x?xf32, "gpu">
  %1 = memref.alloc(%dim0, %dim1) : memref<?x?xf32, "gpu">
  %2 = memref.alloc(%dim0, %dim1) : memref<?x?xf32, "gpu">
  // CHECK: lmhlo.fusion
  // CHECK: lmhlo.clamp
  "lmhlo.abs"(%arg1, %0) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%arg0, %0, %1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.clamp"(%0, %arg3, %arg2, %2) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: }) {
  // CHECK-SAME: disc.fusion.name
  memref.dealloc %0 : memref<?x?xf32, "gpu">
  return %1, %2 : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
}
