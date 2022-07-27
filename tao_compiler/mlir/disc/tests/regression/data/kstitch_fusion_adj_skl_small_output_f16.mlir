// This UT is to test:
// 1. Adjacent skeleton ops;
// 2. the last adjacent skeleton ops's shape is not the same with that of row-
//    reduction;
// 3. The datatype is FP16.
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<110x?xf16>) -> tensor<110xf16> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
      %2:2 = tf_executor.island wraps "tf.Sum"(%arg0, %1) : (tensor<110x?xf16>, tensor<1xi32>) -> tensor<110xf16>
      %3:2 = tf_executor.island wraps "tf.Abs"(%2) : (tensor<110xf16>) -> tensor<110xf16>
      tf_executor.fetch %3 : tensor<110xf16>
    }
    return %graph : tensor<110xf16>
  }
}