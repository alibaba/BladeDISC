module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x100xf32>) -> (tensor<?x100xf16>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Cast"(%arg0) : (tensor<?x100xf32>) -> (tensor<?x100xf16>)
      tf_executor.fetch %0 : tensor<?x100xf16>
    }
    return %graph : tensor<?x100xf16>
  }
}