// This case will cause illegal address error with PTXAS version of 11.0.
// PTXAS 11.1 works well. We have already check that it is not error codegen at
// MLIR level. Changing dimension values does not help to avoid the error. 
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<209x503xf32>) -> tensor<209xf32> attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %4:2 = tf_executor.island wraps "tf.Add"(%arg0, %arg0) : (tensor<209x503xf32>, tensor<209x503xf32>) -> tensor<209x503xf32>
      %c2:2 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %5:2 = tf_executor.island wraps "tf.Max"(%4, %c2) : (tensor<209x503xf32>, tensor<i32>) -> tensor<209xf32>
      tf_executor.fetch %5: tensor<209xf32>
    }
    return %graph : tensor<209xf32>
  }
}