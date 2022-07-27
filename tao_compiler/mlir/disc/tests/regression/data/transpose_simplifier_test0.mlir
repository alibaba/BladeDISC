module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %w0:2 = tf_executor.island wraps "tf.Const"() {value = dense<1.0> : tensor<1x1x16x16xf32>} : () -> tensor<1x1x16x16xf32>
      %p0:2 = tf_executor.island wraps "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
      %t0:2 = tf_executor.island wraps "tf.Transpose"(%arg0, %p0) : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
      %0:2 = tf_executor.island wraps "tf.Conv2D"(%t0, %w0)
      {
        data_format = "NHWC",
        dilations = [1, 1, 1, 1],
        padding = "SAME",
        strides = [1, 1, 1, 1]
      } : (tensor<?x?x?x?xf32>, tensor<1x1x16x16xf32>) -> tensor<?x?x?x?xf32>
      %p1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
      %t1:2 = tf_executor.island wraps "tf.Transpose"(%0, %p1) : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
      %1:2 = tf_executor.island wraps "tf.Relu"(%t1) : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

      %w1:2 = tf_executor.island wraps "tf.Const"() {value = dense<1.1> : tensor<1x1x16x16xf32>} : () -> tensor<1x1x16x16xf32>
      %t2:2 = tf_executor.island wraps "tf.Transpose"(%1, %p0) : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
      %2:2 = tf_executor.island wraps "tf.Conv2D"(%t2, %w1)
      {
        data_format = "NHWC",
        dilations = [1, 1, 1, 1],
        padding = "SAME",
        strides = [1, 1, 1, 1]
      } : (tensor<?x?x?x?xf32>, tensor<1x1x16x16xf32>) -> tensor<?x?x?x?xf32>
      %t3:2 = tf_executor.island wraps "tf.Transpose"(%2, %p1) : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>

      %4:2 = tf_executor.island wraps "tf.Add"(%arg0, %1) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

      tf_executor.fetch %4, %t3 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>
    }
    return %graph#0, %graph#1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>
  }
}