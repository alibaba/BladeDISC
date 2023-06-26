func.func @input_mutation(%arg0: tensor<2x8x1x32xf32>, %arg1: tensor<2x8x1x32xf32>):
    %0 = "arith.constant"() {value = dense<[2, 8, 1, 32]> : tensor<4xi32>} : () -> tensor<4xi32>
    %1 = "chlo.broadcast_add"(%arg0, %arg1) : (tensor<2x8x1x32xf32>, tensor<2x8x1x32xf32>) -> tensor<?x?x?x?xf32>
    disc_mhlo.input_mutation(%1, %arg0) : (tensor<2x8x1x32xf32>, tensor<2x8x1x32xf32>) -> ()
    "func.return"(%1) : (tensor<?x?x?x?xf32>) -> ()