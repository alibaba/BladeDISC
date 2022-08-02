module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main() -> (tensor<33x11xi32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<2> : tensor<33x11xi32>} : () -> (tensor<33x11xi32>)
      tf_executor.fetch %0 : tensor<33x11xi32>
    }
    return %graph : tensor<33x11xi32>
  }
}