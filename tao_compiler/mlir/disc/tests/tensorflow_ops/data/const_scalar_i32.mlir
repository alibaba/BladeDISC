module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main() -> (tensor<i32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<-1> : tensor<i32>} : () -> (tensor<i32>)
      tf_executor.fetch %0 : tensor<i32>
    }
    return %graph : tensor<i32>
  }
}