module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main() -> (tensor<101x99xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<-1.000000e+00> : tensor<101x99xf32>} : () -> (tensor<101x99xf32>)
      tf_executor.fetch %0 : tensor<101x99xf32>
    }
    return %graph : tensor<101x99xf32>
  }
}