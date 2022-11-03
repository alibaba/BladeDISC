// RUN: disc-opt -disc-element-type-converter %s | FileCheck %s


// CHECK-LABEL: @rank2_colunm_reduction_i1
func.func @rank2_colunm_reduction_i1(%arg0: tensor<?x?xi1>) -> tensor<?xi1> {
  %0 = mhlo.constant dense<false> : tensor<i1>
  %1 = "mhlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<i1>, %arg2: tensor<i1>):
    %2 = mhlo.add %arg1, %arg2 : tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<?x?xi1>, tensor<i1>) -> tensor<?xi1>
  // CHECK: mhlo.convert({{.*}}) : (tensor<?x?xi1>) -> tensor<?x?xi32>
  // CHECK-NEXT: mhlo.reduce
  // CHECK-NOT: i1
  // CHECK: mhlo.convert({{.*}}) : (tensor<?xi32>) -> tensor<?xi1>
  return %1 : tensor<?xi1>
}

