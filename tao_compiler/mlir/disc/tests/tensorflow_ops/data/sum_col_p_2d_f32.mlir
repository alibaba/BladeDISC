func.func @main(%arg0: tensor<110x?xf32>) -> tensor<?xf32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
  %graph = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[0]> : tensor<1xi32>} : () -> tensor<1xi32>
    %2:2 = tf_executor.island wraps "tf.Abs"(%arg0) : (tensor<110x?xf32>) -> tensor<110x?xf32>
    %3:2 = tf_executor.island wraps "tf.Sum"(%2, %1) : (tensor<110x?xf32>, tensor<1xi32>) -> tensor<?xf32>
    tf_executor.fetch %3 : tensor<?xf32>
  }
  return %graph : tensor<?xf32>
}