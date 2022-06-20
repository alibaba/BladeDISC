// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func @torch.aten.sigmoid(
// CHECK-SAME:          %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[BUILTIN_ARG:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[CST:.*]] = "chlo.constant_like"(%[[BUILTIN_ARG]]) {value = 1.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32
// CHECK:           %[[NEG_ARG:.*]] = mhlo.negate %[[BUILTIN_ARG]] : tensor<?x?xf32>
// CHECK:           %[[EXPONTIAL_NEG:.*]] = mhlo.exponential %[[NEG_ARG]] : tensor<?x?xf32>
// CHECK:           %[[ADD:.*]] = mhlo.add %[[EXPONTIAL_NEG]], %[[CST]] : tensor<?x?xf32>
// CHECK:           %[[DIV:.*]] = mhlo.divide %[[CST]], %[[ADD]] : tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[DIV]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func @torch.aten.sigmoid(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
    %0 = torch.aten.sigmoid %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
    return %0 : !torch.vtensor<[?,?],f32>
}


// CHECK-LABEL: func @torch.aten.relu(
// CHECK-SAME:          %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:             %[[CMP_GT:.*]] = "mhlo.compare"(%{{.*}} {compare_type = #mhlo<"comparison_type NOTYPE">, comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
// CHECK:             %[[SELECT:.*]] = "mhlo.select"(%[[CMP_GT]], %{{.*}}) : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
func @torch.aten.relu(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
    %0 = torch.aten.relu %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
    return %0 : !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL: func @torch.aten.gelu(
// CHECK-SAME:          %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[RSQRT:.*]] = mhlo.rsqrt %{{.*}} : tensor<?x?xf32>
// CHECK:         %[[MUL0:.*]] = mhlo.multiply %{{.*}}, %{{.*}} : tensor<?x?xf32>
// CHECK:         %[[ERF:.*]] = chlo.erf %[[MUL0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         %[[ADD:.*]] = mhlo.add %[[ERF]], %{{.*}} : tensor<?x?xf32>
// CHECK:         %[[MUL1:.*]] = mhlo.multiply %[[ADD]], %{{.*}} : tensor<?x?xf32>
// CHECK:         %[[MUL2:.*]] = mhlo.multiply %{{.*}}, %[[MUL1]] : tensor<?x?xf32>
func @torch.aten.gelu(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
    %str = torch.constant.str "none"
    %0 = torch.aten.gelu %arg0, %str : !torch.vtensor<[?,?],f32>, !torch.str -> !torch.vtensor<[?,?],f32>
    return %0 : !torch.vtensor<[?,?],f32>
}