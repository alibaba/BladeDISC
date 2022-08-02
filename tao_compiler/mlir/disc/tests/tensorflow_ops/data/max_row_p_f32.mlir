func.func @main(%arg0: tensor<110x100x?xf32>) -> tensor<110x100xf32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
  %graph = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> tensor<1xi32>
    %2:2 = tf_executor.island wraps "tf.Max"(%arg0, %1) : (tensor<110x100x?xf32>, tensor<1xi32>) -> tensor<110x100xf32>
    tf_executor.fetch %2 : tensor<110x100xf32>
  }
  return %graph : tensor<110x100xf32>
}