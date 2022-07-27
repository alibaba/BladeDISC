module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<20x?x?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %2:2 = tf_executor.island wraps "tf.Reshape"(%arg0, %arg1) {T = f32, Tshape = i32, device = ""} : (tensor<20x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
      tf_executor.fetch %2 : tensor<?x?xf32>
    }
    return %graph : tensor<?x?xf32>
  }
}