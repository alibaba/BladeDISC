module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<20x?xf32>) -> (tensor<?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<-0.8> : tensor<768x340xf32>} : () -> tensor<768x340xf32>
      %1:2 = tf_executor.island wraps "tf.MatMul"(%arg0, %0) {transpose_a = false, transpose_b = false} : (tensor<20x?xf32>, tensor<768x340xf32>) -> (tensor<?x?xf32>)
      %2:2 = tf_executor.island wraps "tf.Const"() {value = dense<-0.1> : tensor<340xf32>} : () -> tensor<340xf32>
      %3:2 = tf_executor.island wraps "tf.BiasAdd"(%1, %2) : (tensor<?x?xf32>, tensor<340xf32>) -> tensor<?x?xf32>
      %4:2 = tf_executor.island wraps "tf.Square"(%3) : (tensor<?x?xf32>) -> tensor<?x?xf32>
      %5:2 = tf_executor.island wraps "tf.Const"() {value = dense<4.471500e-02> : tensor<f32>} : () -> tensor<f32>
      %6:2 = tf_executor.island wraps "tf.Mul"(%4, %5) : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
      %7:2 = tf_executor.island wraps "tf.Mul"(%6, %3) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
      %8:2 = tf_executor.island wraps "tf.AddV2"(%3, %7) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
      %9:2 = tf_executor.island wraps "tf.Const"() {value = dense<0.797884583> : tensor<f32>} : () -> tensor<f32>
      %10:2 = tf_executor.island wraps "tf.Mul"(%8, %9) : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
      %11:2 = tf_executor.island wraps "tf.Tanh"(%10) : (tensor<?x?xf32>) -> tensor<?x?xf32>
      %12:2 = tf_executor.island wraps "tf.Const"() {value = dense<1.797884583> : tensor<f32>} : () -> tensor<f32>
      %13:2 = tf_executor.island wraps "tf.AddV2"(%11, %12) : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
      %14:2 = tf_executor.island wraps "tf.Const"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
      %15:2 = tf_executor.island wraps "tf.Mul"(%3, %14) : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
      %16:2 = tf_executor.island wraps "tf.Mul"(%13, %15) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
      tf_executor.fetch %16 : tensor<?x?xf32>
    }
    return %graph : tensor<?x?xf32>
  }
}