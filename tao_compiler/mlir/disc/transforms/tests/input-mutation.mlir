// RUN: disc-opt %s -disc-hlo-legalize-to-lhlo -hlo-legalize-to-lhlo -canonicalize -disc-lhlo-rewriter -split-input-file | FileCheck %s

func.func @input_mutation(%arg0: tensor<8x32xf32>, %arg1: tensor<8x32xf32>) -> tensor<8x32xf32> {
  // CHECK: "lmhlo.add"(%arg0, %arg1, %arg0) : (memref<8x32xf32>, memref<8x32xf32>, memref<8x32xf32>) -> ()
  %1 = mhlo.add %arg0, %arg1 : (tensor<8x32xf32>, tensor<8x32xf32>) -> tensor<8x32xf32>
  "mhlo_disc.args_mutation"(%1, %arg0) : (tensor<8x32xf32>, tensor<8x32xf32>) -> ()
  "func.return"(%1) : (tensor<8x32xf32>) -> ()
}