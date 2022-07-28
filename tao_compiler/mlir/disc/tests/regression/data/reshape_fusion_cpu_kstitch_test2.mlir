module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?x?xf32>) -> (tensor<?x?x960xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %cst:2 = tf_executor.island wraps "tf.Const"() {value = dense<1.0> : tensor<1x1x560x960xf32>} : () -> tensor<1x1x560x960xf32>
      %cst_0:2 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %cst_1:2 = tf_executor.island wraps "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
      %cst_2:2 = tf_executor.island wraps "tf.Const"() {value = dense<9.99999996E-13> : tensor<f32>} : () -> tensor<f32>
      %cst_3:2 = tf_executor.island wraps "tf.Const"() {value = dense<2.0> : tensor<560xf32>} : () -> tensor<560xf32>
      %cst_4:2 = tf_executor.island wraps "tf.Const"() {value = dense<3.0> : tensor<560xf32>} : () -> tensor<560xf32>
      %0:2 = tf_executor.island wraps "tf.Mean"(%arg0, %cst_1) {keep_dims = true} : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x1xf32>
      %1:2 = tf_executor.island wraps "tf.SquaredDifference"(%arg0, %0) : (tensor<?x?x?xf32>, tensor<?x?x1xf32>) -> tensor<?x?x?xf32>
      %2:2 = tf_executor.island wraps "tf.Mean"(%1, %cst_1) {keep_dims = true} : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x1xf32>
      %3:2 = tf_executor.island wraps "tf.AddV2"(%2, %cst_2) : (tensor<?x?x1xf32>, tensor<f32>) -> tensor<?x?x1xf32>
      %4:2 = tf_executor.island wraps "tf.Rsqrt"(%3) : (tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
      %5:2 = tf_executor.island wraps "tf.Mul"(%4, %cst_3) : (tensor<?x?x1xf32>, tensor<560xf32>) -> tensor<?x?x560xf32>
      %6:2 = tf_executor.island wraps "tf.Mul"(%arg0, %5) : (tensor<?x?x?xf32>, tensor<?x?x560xf32>) -> tensor<?x?x560xf32>
      %7:2 = tf_executor.island wraps "tf.Mul"(%5, %0) : (tensor<?x?x560xf32>, tensor<?x?x1xf32>) -> tensor<?x?x560xf32>
      %8:2 = tf_executor.island wraps "tf.Sub"(%cst_4, %7) : (tensor<560xf32>, tensor<?x?x560xf32>) -> tensor<?x?x560xf32>
      %9:2 = tf_executor.island wraps "tf.AddV2"(%6, %8) : (tensor<?x?x560xf32>, tensor<?x?x560xf32>) -> tensor<?x?x560xf32>
      %10:2 = tf_executor.island wraps "tf.ExpandDims"(%9, %cst_0) : (tensor<?x?x560xf32>, tensor<i32>) -> tensor<?x1x?x560xf32>
      %11:2 = tf_executor.island wraps "tf.Conv2D"(%10, %cst) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<?x1x?x560xf32>, tensor<1x1x560x960xf32>) -> tensor<?x1x?x960xf32>
      %12:2 = tf_executor.island wraps "tf.Squeeze"(%11) {squeeze_dims = [1]} : (tensor<?x1x?x960xf32>) -> tensor<?x?x960xf32>
      tf_executor.fetch %12 : tensor<?x?x960xf32>
    }
    return %graph : tensor<?x?x960xf32>
  }
}
