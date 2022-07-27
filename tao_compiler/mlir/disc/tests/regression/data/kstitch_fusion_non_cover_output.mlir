// In this UT, `%1` is not covered by `%4`. Thus `%1` cannot be fused into the
// kStitch fusion.
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> (tensor<?x?xf32>, tensor<?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.Abs"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
      %2:2 = tf_executor.island wraps "tf.Slice"(%1, %arg1, %arg2) {device = ""} : (tensor<?x?xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
      %3:2 = tf_executor.island wraps "tf.Const"() {value = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
      %4:2 = tf_executor.island wraps "tf.Sum"(%2, %3) : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
      %5:2 = tf_executor.island wraps "tf.Abs"(%4) : (tensor<?xf32>) -> tensor<?xf32>
      tf_executor.fetch %1, %5 : tensor<?x?xf32>, tensor<?xf32>
    }
    return %graph#0, %graph#1 : tensor<?x?xf32>, tensor<?xf32>
  }
}