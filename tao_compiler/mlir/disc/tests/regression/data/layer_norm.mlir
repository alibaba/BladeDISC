module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:1 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> tensor<1xi32>
      // E[x]
      %1:2 = tf_executor.island wraps "tf.Mean"(%arg0, %0) : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
      // Var[x] = E[x^2] - E[x]^2
      %2:2 = tf_executor.island wraps "tf.Mul"(%arg0, %arg0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      %3:2 = tf_executor.island wraps "tf.Mean"(%2, %0) : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
      %4:2 = tf_executor.island wraps "tf.Mul"(%1, %1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
      %5:2 = tf_executor.island wraps "tf.Sub"(%3, %4) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
      // rsqrt(Var[x] + epsilon)
      %6:2 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<1.000000e-05> : tensor<f32>} : () -> (tensor<f32>)
      %7:2 = tf_executor.island wraps "tf.Shape"(%5) {device = ""} : (tensor<?x?xf32>) -> (tensor<2xi32>)
      %8:2 = tf_executor.island wraps "tf.BroadcastTo"(%6, %7) : (tensor<f32>, tensor<2xi32>) -> tensor<?x?xf32>
      %9:2 = tf_executor.island wraps "tf.Add"(%5, %8) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
      %10:2 = tf_executor.island wraps "tf.Sqrt"(%9) : (tensor<?x?xf32>) -> tensor<?x?xf32>
      // (x - E[x]) / sqrt(Var[x] + epsilon) * gama + beta
      %c2:2 = tf_executor.island wraps "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
      %11:2 = tf_executor.island wraps "tf.Shape"(%arg0) {device = ""} : (tensor<?x?x?xf32>) -> (tensor<3xi32>)
      %e1:2 = tf_executor.island wraps "tf.ExpandDims"(%1, %c2) : (tensor<?x?xf32>, tensor<i32>) -> (tensor<?x?x?xf32>)
      %12:2 = tf_executor.island wraps "tf.BroadcastTo"(%e1, %11) : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
      %13:2 = tf_executor.island wraps "tf.Sub"(%arg0, %12) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      %e10:2 = tf_executor.island wraps "tf.ExpandDims"(%10, %c2) : (tensor<?x?xf32>, tensor<i32>) -> (tensor<?x?x?xf32>)
      %14:2 = tf_executor.island wraps "tf.BroadcastTo"(%e10, %11) : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
      %15:2 = tf_executor.island wraps "tf.Div"(%13, %14) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      %16:2 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<2.300000> : tensor<f32>} : () -> (tensor<f32>)
      %17:2 = tf_executor.island wraps "tf.BroadcastTo"(%16, %11) : (tensor<f32>, tensor<3xi32>) -> tensor<?x?x?xf32>
      %18:2 = tf_executor.island wraps "tf.Mul"(%15, %17) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      %19:2 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<3.300000> : tensor<f32>} : () -> (tensor<f32>)
      %20:2 = tf_executor.island wraps "tf.BroadcastTo"(%19, %11) : (tensor<f32>, tensor<3xi32>) -> tensor<?x?x?xf32>
      %21:2 = tf_executor.island wraps "tf.Add"(%18, %20) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      tf_executor.fetch %21 : tensor<?x?x?xf32>
    }
    return %graph : tensor<?x?x?xf32>
  }
}