func @main(%arg0: tensor<?x?x?xf64>) -> tensor<?x?xf64> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
  %graph = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> tensor<1xi32>
    %2:2 = tf_executor.island wraps "tf.Abs"(%arg0) : (tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
    %3:2 = tf_executor.island wraps "tf.Sum"(%2, %1) : (tensor<?x?x?xf64>, tensor<1xi32>) -> tensor<?x?xf64>
    tf_executor.fetch %3 : tensor<?x?xf64>
  }
  return %graph : tensor<?x?xf64>
}