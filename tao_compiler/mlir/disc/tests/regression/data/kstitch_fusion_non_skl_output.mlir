// This UT is a complex fusion pattern. Meanwhile, one of the output is not a
// skeleton op.
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x768xf32>, %arg1: tensor<?x?x768xf32>, %arg2: tensor<768xf32>) -> (tensor<?x?xf32>, tensor<?x?x768xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Shape"(%arg1) {device = ""} : (tensor<?x?x768xf32>) -> (tensor<3xi32>)
      %1:2 = tf_executor.island wraps "tf.Reshape"(%arg0, %0) {T = f32, Tshape = i32, device = ""} : (tensor<?x768xf32>, tensor<3xi32>) -> tensor<?x?x768xf32>
      %c0:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %tmp:2 = tf_executor.island wraps "tf.ExpandDims"(%arg2, %c0) : (tensor<768xf32>, tensor<i32>) -> (tensor<?x768xf32>)
      %3:2 = tf_executor.island wraps "tf.ExpandDims"(%tmp, %c0) : (tensor<?x768xf32>, tensor<i32>) -> (tensor<?x?x768xf32>)
      %4:2 = tf_executor.island wraps "tf.BroadcastTo"(%3, %0) : (tensor<?x?x768xf32>, tensor<3xi32>) -> tensor<?x?x768xf32>
      %5:2 = tf_executor.island wraps "tf.Add"(%1, %4) : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
      %6:2 = tf_executor.island wraps "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      %9:2 = tf_executor.island wraps "tf.BroadcastTo"(%6, %0) : (tensor<f32>, tensor<3xi32>) -> tensor<?x?x768xf32>
      %10:2 = tf_executor.island wraps "tf.Mul"(%arg1, %9) : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
      %11:2 = tf_executor.island wraps "tf.Add"(%5, %10) : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
      %12:2 = tf_executor.island wraps "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> tensor<1xi32>
      %13:2 = tf_executor.island wraps "tf.Sum"(%11, %12) : (tensor<?x?x768xf32>, tensor<1xi32>) -> tensor<?x?xf32>
      %14:2 = tf_executor.island wraps "tf.Const"() {value = dense<768.0> : tensor<f32>} : () -> tensor<f32>
      %16:2 = tf_executor.island wraps "tf.Shape"(%13) {device = ""} : (tensor<?x?xf32>) -> (tensor<2xi32>)
      %17:2 = tf_executor.island wraps "tf.BroadcastTo"(%14, %16) : (tensor<f32>, tensor<2xi32>) -> tensor<?x?xf32>
      %18:2 = tf_executor.island wraps "tf.Div"(%13, %17) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
      %c2:2 = tf_executor.island wraps "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
      %19:2 = tf_executor.island wraps "tf.ExpandDims"(%18, %c2) : (tensor<?x?xf32>, tensor<i32>) -> (tensor<?x?x?xf32>)
      %20:2 = tf_executor.island wraps "tf.BroadcastTo"(%19, %0) : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x768xf32>
      %21:2 = tf_executor.island wraps "tf.Sub"(%11, %20) : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
      %22:2 = tf_executor.island wraps "tf.Mul"(%21, %21) : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
      %25:2 = tf_executor.island wraps "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> tensor<1xi32>
      %26:2 = tf_executor.island wraps "tf.Sum"(%22, %25) : (tensor<?x?x768xf32>, tensor<1xi32>) -> tensor<?x?xf32>
      tf_executor.fetch %26, %21 : tensor<?x?xf32>, tensor<?x?x768xf32>
    }
    return %graph#0, %graph#1 : tensor<?x?xf32>, tensor<?x?x768xf32>
  }
}