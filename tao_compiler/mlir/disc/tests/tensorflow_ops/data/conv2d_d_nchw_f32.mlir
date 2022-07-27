module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Conv2D"(%arg0, %arg1)
      {
        data_format = "NCHW",
        dilations = [1, 1, 1, 1],
        explicit_paddings = [],
        padding = "VALID",
        strides = [1, 1, 1, 1]
      } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
      tf_executor.fetch %0 : tensor<?x?x?x?xf32>
    }
    return %graph : tensor<?x?x?x?xf32>
  }
}
