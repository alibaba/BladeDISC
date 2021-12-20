func @main(%arg0: tensor<?x?x?xf32>) -> tensor<?xf32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
  %graph = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[1,2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %2:2 = tf_executor.island wraps "tf.Prod"(%arg0, %1) : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?xf32>
    tf_executor.fetch %2 : tensor<?xf32>
  }
  return %graph : tensor<?xf32>
}