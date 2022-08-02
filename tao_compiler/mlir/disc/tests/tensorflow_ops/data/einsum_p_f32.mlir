module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x4x5xf32>, %arg1: tensor<?x5x6xf32>) -> (tensor<?x4x6xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Einsum"(%arg0, %arg1) {equation = "ijk,ikm->ijm"} : (tensor<?x4x5xf32>, tensor<?x5x6xf32>) -> (tensor<?x4x6xf32>)
      tf_executor.fetch %0 : tensor<?x4x6xf32>
    }
    return %graph : tensor<?x4x6xf32>
  }
}