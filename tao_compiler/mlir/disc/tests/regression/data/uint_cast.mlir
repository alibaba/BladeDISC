module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?xui8>) -> (tensor<?x?xf32>) attributes {tf.entry_function = {inputs = "input0", outputs = "output0"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Cast"(%arg0) : (tensor<?x?xui8>) -> (tensor<?x?xf32>)
      tf_executor.fetch %0 : tensor<?x?xf32>
    }
    return %graph : tensor<?x?xf32>
  }
}