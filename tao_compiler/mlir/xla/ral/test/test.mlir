module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 175 : i32}} {
  func.func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {tf.entry_function = {inputs = "arg_x_0_0_0_arg,arg_z_0_2_0_arg", outputs = "add_4:0"}} {
    %0 = xla_hlo.add %arg0, %arg1 : tensor<?x?xf32>
    %1 = xla_hlo.add %0, %0 : tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
}
