// RUN: disc-opt %s -disc-hlo-legalize-to-lhlo -hlo-legalize-to-lhlo -disc-lhlo-rewriter -split-input-file | FileCheck %s

func.func @input_mutation(%arg0: tensor<8x32xf32>, %arg1: tensor<8x32xf32>) -> tensor<8x32xf32> {
  %1 = chlo.broadcast_add %arg0, %arg1 : (tensor<8x32xf32>, tensor<8x32xf32>) -> tensor<8x32xf32>
  "mhlo_disc.args_mutation"(%1, %arg0) : (tensor<8x32xf32>, tensor<8x32xf32>) -> ()
  "func.return"(%1) : (tensor<8x32xf32>) -> ()
}