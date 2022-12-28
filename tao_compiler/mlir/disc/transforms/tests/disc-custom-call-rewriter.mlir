// RUN: disc-opt --disc-custom-call-rewriter -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: @gemm
// CHECK-SAME: (%[[INPUT:.*]]: tensor<?x?xf32>, %[[WEIGHT:.*]]: tensor<?x?xf32>)
func.func @gemm(%input: tensor<?x?xf32>, %weight : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[T0:.*]] = "mhlo.transpose"
  // CHECK-SAME: %[[WEIGHT]]
  // CHECK-SAME: permutation = dense<[1, 0]>
  // CHECK: %[[T1:.*]] = "mhlo_disc.custom_call_v2"(%[[INPUT]], %[[T0]])
  // CHECK: %[[T2:.*]] = "mhlo.transpose"
  // CHECK-SAME: %[[T1]]
  // CHECK-SAME: permutation = dense<[1, 0]>
  // CHECK: return %[[T2]]
  %output = "mhlo_disc.custom_call_v2"(%input, %weight) {
      call_target_name = "test",
      custom_attrs = {},
      has_side_effect = false,
      device = "d",
      input_placements = "d,h",
      output_placements = "h",
      input_layouts = "AB,AB",
      expected_input_layouts = "AB,BA",
      output_layouts = "BA",
      expected_output_layouts = "AB"
  } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %output : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @conv
// CHECK-SAME: (%[[INPUT:.*]]: tensor<?x?x?x?xf32>, %[[WEIGHT:.*]]: tensor<?x?x?x?xf32>)
func.func @conv(%input: tensor<?x?x?x?xf32>, %weight : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK: %[[T0:.*]] = "mhlo.transpose"
  // CHECK-SAME: %[[INPUT]]
  // CHECK-SAME: permutation = dense<[0, 2, 3, 1]>
  // CHECK: %[[T1:.*]] = "mhlo.transpose"
  // CHECK-SAME: %[[WEIGHT]]
  // CHECK-SAME: permutation = dense<[0, 2, 3, 1]>

  // CHECK: %[[T2:.*]] = "mhlo_disc.custom_call_v2"(%[[T0]], %[[T1]])
  // CHECK: %[[T3:.*]] = "mhlo.transpose"
  // CHECK-SAME: %[[T2]]
  // CHECK-SAME: permutation = dense<[0, 3, 1, 2]>
  // CHECK: return %[[T3]]
  %output = "mhlo_disc.custom_call_v2"(%input, %weight) {
      call_target_name = "test",
      custom_attrs = {},
      has_side_effect = false,
      device = "d",
      input_placements = "d,h",
      output_placements = "h",
      input_layouts = "NCHW,OIHW",
      expected_input_layouts = "NHWC,OHWI",
      output_layouts = "NCHW",
      expected_output_layouts = "NHWC"
  } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %output : tensor<?x?x?x?xf32>
}