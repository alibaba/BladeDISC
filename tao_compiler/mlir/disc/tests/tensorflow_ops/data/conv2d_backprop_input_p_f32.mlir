module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<3x3x1x32xf32>, %arg2: tensor<?x26x26x32xf32>) -> (tensor<?x28x28x1xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Conv2DBackpropInput"(%arg0, %arg1, %arg2)
      {
        data_format = "NHWC",
        dilations = [1, 1, 1, 1],
        explicit_paddings = [],
        padding = "VALID",
        strides = [1, 1, 1, 1]
      } : (tensor<4xi32>, tensor<3x3x1x32xf32>, tensor<?x26x26x32xf32>) -> tensor<?x28x28x1xf32>
      tf_executor.fetch %0 : tensor<?x28x28x1xf32>
    }
    return %graph : tensor<?x28x28x1xf32>
  }
}