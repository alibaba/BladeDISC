module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func @main(%arg0: tensor<?x?xui8>, %arg1: tensor<?x?xui8>) -> (tensor<?x?xui8>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Div"(%arg0, %arg1) : (tensor<?x?xui8>, tensor<?x?xui8>) -> (tensor<?x?xui8>)
      tf_executor.fetch %0 : tensor<?x?xui8>
    }
    return %graph : tensor<?x?xui8>
  }
}