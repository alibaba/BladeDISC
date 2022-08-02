module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<2xi64> attributes {tf.entry_function = {inputs = "input0,input1,input2", 
      outputs = "output0"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Range"(%arg0, %arg1, %arg2) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2xi64>
      tf_executor.fetch %0 : tensor<2xi64>
    }
    return %graph : tensor<2xi64>
  }
}